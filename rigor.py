"""
rigor.py — Publication-rigor orchestrator.

Jobs (run in parallel shells, each well under 10 min):
  --job seed:N      multiseed run N (0-4): train DQN + evaluate ALL 8
                    schedulers across the 5 loads with a seed-varied env
                    -> results/publication/seedN.json
  --job ablation:X  X in {nomask, noper, noshaping, nodueling}: train the
                    variant (seed 0) + evaluate DQN only
                    -> results/publication/ablation_X.json
  --job aggregate   combines seeds + ablations, computes mean +/- std and
                    two-sided Wilcoxon signed-rank tests (normal
                    approximation) of DQN vs every baseline
                    -> results/publication/{multiseed.json, significance.json,
                       ablation.json, summary.txt}

Usage: python rigor.py --job seed:2
"""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).parent
OUT = HERE / "results" / "publication"
OUT.mkdir(parents=True, exist_ok=True)


def load_cfg():
    cfg = yaml.safe_load(open(HERE / "config.yaml"))
    cfg["dataset"]["csv_path"] = "../../borg_traces_data.csv"
    cfg["dataset"]["cache_path"] = "results/borg_task_pool.npz"
    return cfg


def load_pool(cfg):
    from borg_loader import load_task_pool
    ds = cfg["dataset"]
    return load_task_pool(HERE / ds["csv_path"], HERE / ds["cache_path"],
                          ds["keep_events"], ds["min_duration_us"],
                          ds["pool_size"], seed=0)


def evaluate_all(cfg, pool, seed, n_runs=3):
    """Train DQN at `seed` and evaluate every scheduler; returns per-load metrics."""
    from train import train_dqn, evaluate_scheduler
    from baselines import (FCFSScheduler, RoundRobinScheduler,
                           GreedyLeastLoadedScheduler, MinMinScheduler,
                           MaxMinScheduler, PSOScheduler, TabularQLearning)
    agent, _ = train_dqn(pool, cfg, seed=seed, verbose=False)

    class DQNWrapper:
        def act(self, env, state, greedy=True):
            return agent.act(state, greedy=True)
        def reset(self):
            pass

    class QLWrapper:
        def __init__(self):
            from train import train_tabular_q
            self._ql = None
        def act(self, env, state, greedy=True):
            return self._ql.act(env, state, greedy=True)
        def reset(self):
            pass

    # tabular Q trained once per seed (cheap)
    from train import train_tabular_q
    ql = train_tabular_q(pool, cfg, seed=seed, verbose=False)

    class QLW:
        def act(self, env, state, greedy=True):
            return ql.act(env, state, greedy=True)
        def reset(self):
            pass

    n_vm = cfg["env"]["num_vms"]
    p = cfg["baselines"]["pso"]
    scheds = {
        "FCFS": lambda: FCFSScheduler(),
        "RoundRobin": lambda: RoundRobinScheduler(n_vm),
        "GreedyLeastLoaded": lambda: GreedyLeastLoadedScheduler(),
        "Min-Min": lambda: MinMinScheduler(),
        "Max-Min": lambda: MaxMinScheduler(),
        "PSO": lambda: PSOScheduler(n_vm, **p),
        "Q-learning": lambda: QLW(),
        "DQN (ours)": lambda: DQNWrapper(),
    }
    out = {}
    for load in cfg["eval"]["loads"]:
        out[str(load)] = {}
        for name, factory in scheds.items():
            out[str(load)][name] = evaluate_scheduler(
                factory, pool, cfg, load, seed,
                n_runs=n_runs, vm_seed=cfg["env"]["vm_seed"])
    return out


def job_seed(k):
    cfg = load_cfg()
    pool = load_pool(cfg)
    res = evaluate_all(cfg, pool, seed=k)
    json.dump(res, open(OUT / f"seed{k}.json", "w"), indent=1)
    print(f"seed{k} done")


def job_ablation(name):
    cfg = load_cfg()
    pool = load_pool(cfg)
    if name == "nomask":
        cfg["agent"]["use_mask"] = False
    elif name == "noper":
        cfg["agent"]["per"]["enabled"] = False
    elif name == "noshaping":
        cfg["env"]["use_shaping"] = False
    elif name == "nodueling":
        cfg["agent"]["dueling"] = False
    else:
        raise ValueError(name)
    res = evaluate_all(cfg, pool, seed=0)
    json.dump(res, open(OUT / f"ablation_{name}.json", "w"), indent=1)
    print(f"ablation {name} done")


# --------------------------------------------------------------------- #
# Aggregation + statistics
# --------------------------------------------------------------------- #
def wilcoxon_signed_rank(x, y):
    """Two-sided Wilcoxon signed-rank on paired samples, normal approximation
    with continuity correction (valid for n > ~25). Returns (W, z, p)."""
    from math import erf, sqrt
    d = np.asarray(x, float) - np.asarray(y, float)
    d = d[d != 0]
    n = len(d)
    if n < 5:
        return None, None, None
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    # mid-ranks for ties (rare with continuous metrics)
    Wp = float(np.sum(ranks[d > 0]))
    W = min(Wp, n * (n + 1) / 2 - Wp)
    mu = n * (n + 1) / 4
    sigma = sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (W - mu) / sigma if sigma > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return W, z, p


def job_aggregate():
    METRICS = ["makespan_s", "cost", "energy_wh", "deadline_miss_rate", "di"]
    seeds = sorted(int(f.name[4:-5]) for f in OUT.glob("seed*.json"))
    assert seeds, "no seed runs found"
    data = {s: json.load(open(OUT / f"seed{s}.json")) for s in seeds}
    loads = sorted(data[seeds[0]].keys(), key=int)
    names = list(data[seeds[0]][loads[0]].keys())

    # ---- mean +/- std across seeds ----
    multiseed = {n: {} for n in names}
    for n in names:
        for m in METRICS:
            vals = [np.mean([data[s][l][n][m] for l in loads]) for s in seeds]
            multiseed[n][m] = {"mean": float(np.mean(vals)),
                               "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}

    # ---- significance: paired per (seed, load) DQN vs baseline ----
    sig = {}
    for opp in names:
        if opp == "DQN (ours)":
            continue
        sig[opp] = {}
        for m in METRICS:
            lower_is_better = True  # all five metrics
            dx, oy = [], []
            for s in seeds:
                for l in loads:
                    dx.append(data[s][l]["DQN (ours)"][m])
                    oy.append(data[s][l][opp][m])
            if lower_is_better:
                x, y = oy, dx          # positive difference = DQN better
            W, z, p = wilcoxon_signed_rank(x, y)
            wins = int(np.sum(np.asarray(x) > np.asarray(y)))
            sig[opp][m] = {"wilcoxon_p": p, "z": z, "dqn_better_count": wins,
                           "n_pairs": len(x),
                           "dqn_median_better": bool(np.median(x) > np.median(y))}

    # ---- ablation ----
    ab = {}
    full = data[seeds[0]]
    ab["full"] = {m: float(np.mean([full[l]["DQN (ours)"][m] for l in loads])) for m in METRICS}
    for f in OUT.glob("ablation_*.json"):
        variant = f.stem.replace("ablation_", "")
        r = json.load(open(f))
        ab[variant] = {m: float(np.mean([r[l]["DQN (ours)"][m] for l in loads])) for m in METRICS}

    json.dump(multiseed, open(OUT / "multiseed.json", "w"), indent=1)
    json.dump(sig, open(OUT / "significance.json", "w"), indent=1)
    json.dump(ab, open(OUT / "ablation.json", "w"), indent=1)

    # human summary
    lines = [f"PUBLICATION RIGOR SUMMARY  (seeds: {seeds})", "=" * 72]
    lines.append("\n--- Multiseed mean +/- std (across loads) ---")
    for n in names:
        mm = multiseed[n]
        lines.append(f"{n:<20} mk={mm['makespan_s']['mean']:8.0f}+-{mm['makespan_s']['std']:6.0f}  "
                     f"cost={mm['cost']['mean']:8.0f}+-{mm['cost']['std']:6.0f}  "
                     f"miss={mm['deadline_miss_rate']['mean']*100:5.1f}+-{mm['deadline_miss_rate']['std']*100:4.1f}%  "
                     f"DI={mm['di']['mean']:5.2f}+-{mm['di']['std']:4.2f}")
    lines.append("\n--- Wilcoxon signed-rank, DQN vs baseline (p<0.05 = significant) ---")
    for opp, mm in sig.items():
        for m in ["makespan_s", "cost", "deadline_miss_rate"]:
            st = mm[m]
            p = st["wilcoxon_p"]
            verdict = ("DQN BETTER" if st["dqn_median_better"] and p and p < 0.05 else
                       "DQN WORSE" if p and p < 0.05 else "n.s.")
            lines.append(f"  vs {opp:<20} {m:<20} p={p:.4f}  -> {verdict}" if p else f"  vs {opp:<20} {m:<20} p=n/a")
        lines.append("")
    lines.append("--- Ablation (seed 0, DQN only, avg across loads) ---")
    base = ab["full"]
    for variant, mm in ab.items():
        dm = mm["makespan_s"] - base["makespan_s"]
        dc = mm["cost"] - base["cost"]
        dmiss = (mm["deadline_miss_rate"] - base["deadline_miss_rate"]) * 100
        lines.append(f"{variant:<12} mk={mm['makespan_s']:8.0f} ({dm:+8.0f})  cost={mm['cost']:8.0f} ({dc:+8.0f})  miss={mm['deadline_miss_rate']*100:5.1f}% ({dmiss:+5.1f}pp)")
    open(OUT / "summary.txt", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    args = ap.parse_args()
    kind, _, arg = args.job.partition(":")
    if kind == "seed":
        job_seed(int(arg))
    elif kind == "ablation":
        job_ablation(arg)
    elif kind == "aggregate":
        job_aggregate()
    else:
        raise SystemExit(f"unknown job {args.job}")
