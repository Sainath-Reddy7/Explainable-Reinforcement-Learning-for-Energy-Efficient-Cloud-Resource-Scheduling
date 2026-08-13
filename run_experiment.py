"""
run_experiment.py — End-to-end v2 pipeline.

  1. Load + clean the Borg dataset (cached).
  2. Train Double-DQN + tabular-Q, evaluate 8 schedulers across loads.
  3. Roll out the trained DQN and explain 60 decisions with 4 XAI methods.
  4. Score every method on fidelity, deletion/insertion AOPC, consistency,
     stability, infidelity and latency.
  5. Persist everything (JSON + a reproducibility manifest).
"""

from __future__ import annotations

import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from borg_loader import load_task_pool, pool_summary
from env import CloudSchedulingEnv
from train import run_full_comparison
from explainability import (
    KernelSHAPExplainer, gradient_x_input_explainer, occlusion_explainer,
    integrated_gradients_explainer, rollup_vm_contributions, time_explainer,
)
from fidelity import (
    top_k_fidelity, deletion_aopc, insertion_aopc, infidelity,
    consistency_score, stability_score,
)

HERE = Path(__file__).parent
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)


def load_config(path=None):
    path = path or (HERE / "config.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def collect_rollout(agent, env, episode_seed_offset, n_steps=None):
    states, actions, infos = [], [], []
    s = env.reset(episode_seed_offset=episode_seed_offset)
    done = False
    while not done:
        a = agent.act(s, greedy=True)
        states.append(s.copy())
        actions.append(a)
        s2, r, done, info = env.step(a)
        infos.append(info)
        s = s2
        if n_steps and len(states) >= n_steps:
            break
    return states, actions, infos


def main(cfg=None, seed=0):
    t_start = time.time()
    cfg = cfg or load_config()

    # ---- 1. dataset --------------------------------------------------------
    ds = cfg["dataset"]
    pool_path = HERE / ds["csv_path"] if not os.path.isabs(ds["csv_path"]) else Path(ds["csv_path"])
    cache_path = HERE / ds["cache_path"] if not os.path.isabs(ds["cache_path"]) else Path(ds["cache_path"])
    pool = load_task_pool(pool_path, cache_path, ds["keep_events"],
                          ds["min_duration_us"], ds["pool_size"], seed=seed)
    pool_stats = pool_summary(pool)
    print(f"[run] Borg pool ready: {pool_stats['n_tasks']} tasks")

    # ---- 2. train + compare ------------------------------------------------
    agent, ql, dqn_history, comparison = run_full_comparison(pool, cfg, seed=seed)
    with open(OUT_DIR / "comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    with open(OUT_DIR / "dqn_training_history.json", "w") as f:
        json.dump(dqn_history, f, indent=2)

    # ---- 3. rollout for XAI ------------------------------------------------
    print("=" * 78)
    print("Rolling out trained DQN for explainability analysis ...")
    env_cfg = cfg["env"]
    env = CloudSchedulingEnv(pool, num_vms=env_cfg["num_vms"],
                             num_tasks=env_cfg["tasks_per_episode"],
                             seed=seed, vm_seed=env_cfg["vm_seed"])

    bg_states, _, _ = collect_rollout(agent, env, episode_seed_offset=9000)
    background = np.array(bg_states[-cfg["xai"]["kernelshap"]["n_background"]:])
    background_mean = background.mean(axis=0)

    states, actions, infos = collect_rollout(agent, env, episode_seed_offset=9500)
    n_exp = cfg["xai"]["n_explained_decisions"]
    sample_idx = np.linspace(0, len(states) - 1, n_exp).astype(int)

    ksh = KernelSHAPExplainer(agent.q.predict, background,
                              n_coalitions=cfg["xai"]["kernelshap"]["n_coalitions"],
                              n_bg_draws=cfg["xai"]["kernelshap"]["n_bg_draws"],
                              seed=seed)
    occ_window = cfg["xai"]["occlusion"]["window"]
    ig_steps = cfg["xai"]["integrated_gradients"]["steps"]

    # per-method stores
    methods = ["kernelshap", "grad_x_input", "occlusion", "integrated_gradients"]
    phis = {m: [] for m in methods}
    fidelities = {m: [] for m in methods}
    del_aopcs, ins_aopcs = {m: [] for m in methods}, {m: [] for m in methods}
    infids = {m: [] for m in methods}
    latencies = {m: [] for m in methods}
    decision_log = []

    raw_infer = time_explainer(agent.q.predict, states[0][None, :], n_reps=20)

    for count, i in enumerate(sample_idx):
        s, a, info = states[i], actions[i], infos[i]

        # KernelSHAP
        t0 = time.time(); phi = ksh.explain(s, a); latencies["kernelshap"].append(time.time() - t0)
        phis["kernelshap"].append(phi)

        # Gradient × Input
        t0 = time.time(); phi_g = gradient_x_input_explainer(agent.q, s, a); latencies["grad_x_input"].append(time.time() - t0)
        phis["grad_x_input"].append(phi_g)

        # Occlusion
        t0 = time.time(); phi_o = occlusion_explainer(agent.q.predict, s, a, background_mean, occ_window); latencies["occlusion"].append(time.time() - t0)
        phis["occlusion"].append(phi_o)

        # Integrated Gradients
        t0 = time.time(); phi_ig = integrated_gradients_explainer(agent.q, s, a, steps=ig_steps); latencies["integrated_gradients"].append(time.time() - t0)
        phis["integrated_gradients"].append(phi_ig)

        # trust metrics for each method
        for m in methods:
            phi_m = phis[m][-1]
            fidelities[m].append(top_k_fidelity(agent.q.predict, s, phi_m, background_mean, a))
            del_aopcs[m].append(deletion_aopc(agent.q.predict, s, phi_m, background_mean, a))
            ins_aopcs[m].append(insertion_aopc(agent.q.predict, s, phi_m, background_mean, a))
            infids[m].append(infidelity(agent.q.predict, agent.q, s, phi_m, a, background_mean, seed=seed + count))

        # dashboard entry (use KernelSHAP as the headline explanation)
        vm_c = rollup_vm_contributions(phis["kernelshap"][-1], env.feature_names, env_cfg["num_vms"])
        top_feats = np.argsort(-np.abs(phis["kernelshap"][-1]))[:5]
        decision_log.append({
            "step": int(i),
            "chosen_vm": int(a),
            "exec_time_s": info["exec_time_s"],
            "energy_wh": info["energy_wh"],
            "cost": info["cost"],
            "deadline_met": bool(info["deadline_met"]),
            "priority": float(info["priority"]),
            "top_features": [{"name": env.feature_names[fi],
                              "phi": float(phis["kernelshap"][-1][fi])}
                             for fi in top_feats],
            "vm_contributions": vm_c,
        })

        if count % 10 == 0:
            print(f"  explained {count+1}/{len(sample_idx)}  "
                  f"(SHAP {latencies['kernelshap'][-1]*1000:.0f}ms, "
                  f"IG {latencies['integrated_gradients'][-1]*1000:.1f}ms, "
                  f"occ {latencies['occlusion'][-1]*1000:.1f}ms)")

    # ---- 4. aggregate trust metrics ---------------------------------------
    def agg_fid(fid_list):
        keys = fid_list[0].keys()
        return {k: float(np.mean([f[k] for f in fid_list])) for k in keys}

    states_arr = np.array(states)[sample_idx]
    trust = {}
    for m in methods:
        phis_arr = np.array(phis[m])
        cons, n_pairs = consistency_score(phis_arr, states_arr)
        stab, _ = stability_score(phis_arr, states_arr)
        trust[m] = {
            "fidelity": agg_fid(fidelities[m]),
            "deletion_aopc": float(np.mean(del_aopcs[m])),
            "insertion_aopc": float(np.mean(ins_aopcs[m])),
            "infidelity": float(np.mean(infids[m])),
            "consistency": cons,
            "stability": stab,
            "n_consistency_pairs": n_pairs,
            "mean_latency_sec": float(np.mean(latencies[m])),
        }
    trust["raw_inference_latency_sec"] = float(raw_infer)

    # ---- 5. persist --------------------------------------------------------
    with open(OUT_DIR / "decision_log.json", "w") as f:
        json.dump(decision_log, f, indent=2)
    with open(OUT_DIR / "trust_metrics.json", "w") as f:
        json.dump(trust, f, indent=2)

    # manifest
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": cfg,
        "dataset_stats": pool_stats,
        "total_runtime_sec": time.time() - t_start,
    }
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print("=" * 78)
    print("TRUST METRICS SUMMARY (4 methods)")
    print(json.dumps(trust, indent=2))
    print(f"\nTotal pipeline time: {time.time()-t_start:.1f}s")
    print(f"Outputs written to {OUT_DIR}/")
    return comparison, decision_log, trust


if __name__ == "__main__":
    main()
