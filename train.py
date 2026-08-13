"""
train.py — DQN + tabular-Q training and the full scheduler comparison (v2).

Differences from v1
-------------------
* DQN agent uses Double DQN + PER + soft updates (see dqn_agent.py).
* Training resamples the Borg task pool each episode (curriculum) and tracks
  per-episode metrics so plots.py can draw learning curves.
* Evaluation harness is unchanged in spirit (fixed VM pool, varied task
  stream) but now also reports per-scheduler energy/cost in Watt-hours and $.
"""

from __future__ import annotations

import time

import numpy as np

from env import CloudSchedulingEnv
from dqn_agent import DQNAgent
from baselines import (FCFSScheduler, RoundRobinScheduler,
                        GreedyLeastLoadedScheduler, TabularQLearning,
                        MinMinScheduler, MaxMinScheduler, PSOScheduler)


# --------------------------------------------------------------------------- #
# Training                                                                    #
# --------------------------------------------------------------------------- #
def train_dqn(task_pool, cfg, seed=0, verbose=True):
    env_cfg, ag_cfg, q_cfg, tr_cfg = cfg["env"], cfg["agent"], cfg["qnetwork"], cfg["train"]
    env = CloudSchedulingEnv(task_pool,
                             num_vms=env_cfg["num_vms"],
                             num_tasks=env_cfg["tasks_per_episode"],
                             seed=seed,
                             adaptive_weights=env_cfg["adaptive_weights"],
                             vm_seed=env_cfg["vm_seed"])
    agent = DQNAgent(env.state_dim, env.action_dim,
                     hidden=tuple(q_cfg["hidden"]),
                     lr=q_cfg["lr"], dropout=q_cfg["dropout"],
                     buffer_size=ag_cfg["buffer_size"],
                     batch_size=ag_cfg["batch_size"],
                     soft_update_tau=ag_cfg["soft_update_tau"],
                     target_update_steps=ag_cfg["target_update_steps"],
                     double_dqn=ag_cfg["double_dqn"],
                     per_enabled=ag_cfg["per"]["enabled"],
                     per_alpha=ag_cfg["per"]["alpha"],
                     beta_start=ag_cfg["per"]["beta_start"],
                     beta_end=ag_cfg["per"]["beta_end"],
                     eps_start=ag_cfg["eps_start"],
                     eps_end=ag_cfg["eps_end"],
                     eps_decay=ag_cfg["eps_decay"],
                     seed=seed)
    lr_decay_step = max(1, ag_cfg.get("lr_decay_steps", 20000))

    history = []
    cr = tr_cfg.get("curriculum_range")
    for ep in range(tr_cfg["episodes"]):
        n_tasks = int(np.random.default_rng(seed + ep).integers(*cr)) if cr \
            else env_cfg["tasks_per_episode"]
        state = env.reset(episode_seed_offset=ep, num_tasks=n_tasks)
        ep_reward = 0.0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, float(done))
            loss = agent.train_step()
            # LR decay
            if agent.step_count % 1000 == 0 and agent.step_count > 0:
                agent.q.set_lr(q_cfg["lr"] * ag_cfg.get("lr_decay_rate", 0.95) ** (agent.step_count // 1000))
            state = next_state
            ep_reward += reward
        agent.decay_epsilon()
        m = env.episode_metrics()
        history.append({"episode": ep, "reward": ep_reward,
                        "loss": loss, **m, "eps": agent.eps, "n_tasks": n_tasks})
        if verbose and (ep % max(1, tr_cfg["episodes"] // 10) == 0 or ep == tr_cfg["episodes"] - 1):
            print(f"[DQN] ep {ep:3d}  reward={ep_reward:8.1f}  loss={loss or 0:.4f}  "
                  f"makespan={m['makespan_s']:7.0f}s  cost=${m['cost']:6.2f}  "
                  f"miss={m['deadline_miss_rate']:.3f}  eps={agent.eps:.3f}")
    return agent, history


def train_tabular_q(task_pool, cfg, seed=0, verbose=True):
    env_cfg, tr_cfg = cfg["env"], cfg["train"]
    env = CloudSchedulingEnv(task_pool,
                             num_vms=env_cfg["num_vms"],
                             num_tasks=env_cfg["tasks_per_episode"],
                             seed=seed, vm_seed=env_cfg["vm_seed"])
    ql = TabularQLearning(env_cfg["num_vms"], seed=seed)
    for ep in range(tr_cfg["episodes"]):
        state = env.reset(episode_seed_offset=ep)
        done = False
        while not done:
            action = ql.act(env, state)
            next_state, reward, done, _ = env.step(action)
            ql.update(state, action, reward, next_state, done)
            state = next_state
        ql.decay_epsilon()
        if verbose and (ep % max(1, tr_cfg["episodes"] // 10) == 0):
            print(f"[Q-tab] ep {ep:3d}  eps={ql.eps:.3f}  states={len(ql.q_table)}")
    return ql


# --------------------------------------------------------------------------- #
# Evaluation                                                                  #
# --------------------------------------------------------------------------- #
def evaluate_scheduler(scheduler_fn, task_pool, cfg, num_tasks, seed,
                       n_runs=5, vm_seed=42):
    env_cfg = cfg["env"]
    env = CloudSchedulingEnv(task_pool,
                             num_vms=env_cfg["num_vms"],
                             num_tasks=num_tasks,
                             seed=seed, vm_seed=vm_seed)
    results = []
    for run in range(n_runs):
        state = env.reset(episode_seed_offset=1000 + run, num_tasks=num_tasks)
        sched = scheduler_fn()
        if hasattr(sched, "reset"):
            sched.reset()
        done = False
        while not done:
            try:
                action = sched.act(env, state, greedy=True)
            except TypeError:
                action = sched.act(env, state)
            state, _, done, _ = env.step(action)
        results.append(env.episode_metrics())
    keys = results[0].keys()
    return {k: float(np.mean([r[k] for r in results])) for k in keys}


def run_full_comparison(task_pool, cfg, seed=0):
    env_cfg = cfg["env"]
    print("=" * 78)
    print("Training DQN (Double DQN + PER) ...")
    t0 = time.time()
    agent, dqn_history = train_dqn(task_pool, cfg, seed=seed)
    print(f"  trained in {time.time()-t0:.1f}s")

    print("=" * 78)
    print("Training tabular Q-learning baseline ...")
    ql = train_tabular_q(task_pool, cfg, seed=seed)

    print("=" * 78)
    print(f"Evaluating all schedulers across task loads {cfg['eval']['loads']} ...")

    class DQNWrapper:
        def act(self, env, state, greedy=True):
            return agent.act(state, greedy=True)
        def reset(self): pass

    class QLWrapper:
        def act(self, env, state, greedy=True):
            return ql.act(env, state, greedy=True)
        def reset(self): pass

    n_vm = env_cfg["num_vms"]
    pso_cfg = cfg["baselines"]["pso"]
    schedulers = {
        "FCFS": lambda: FCFSScheduler(),
        "RoundRobin": lambda: RoundRobinScheduler(n_vm),
        "GreedyLeastLoaded": lambda: GreedyLeastLoadedScheduler(),
        "Min-Min": lambda: MinMinScheduler(),
        "Max-Min": lambda: MaxMinScheduler(),
        "PSO": lambda: PSOScheduler(n_vm, **pso_cfg),
        "Q-learning": lambda: QLWrapper(),
        "DQN (ours)": lambda: DQNWrapper(),
    }

    all_results = {}
    for load in cfg["eval"]["loads"]:
        all_results[load] = {}
        for name, factory in schedulers.items():
            metrics = evaluate_scheduler(factory, task_pool, cfg, load, seed,
                                         n_runs=cfg["eval"]["runs_per_load"],
                                         vm_seed=env_cfg["vm_seed"])
            all_results[load][name] = metrics
            print(f"  load={load:4d}  {name:18s}  "
                  f"makespan={metrics['makespan_s']:7.0f}s  "
                  f"cost=${metrics['cost']:6.2f}  energy={metrics['energy_wh']:6.2f}Wh  "
                  f"miss={metrics['deadline_miss_rate']:.3f}  DI={metrics['di']:.3f}")
    return agent, ql, dqn_history, all_results
