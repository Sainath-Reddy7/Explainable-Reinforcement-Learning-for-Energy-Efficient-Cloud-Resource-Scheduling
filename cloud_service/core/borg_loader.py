"""
borg_loader.py — Parse the raw Google Borg trace CSV into a clean task pool.

The Borg dataset (borg_traces_data.csv, ~328 MB) is a real production cloud
workload trace published by Google.  Each row is an instance lifecycle event.
We collapse it into a pool of schedulable tasks that the CloudSchedulingEnv
samples from — replacing v1's synthetic random-task generator with genuine
workload distributions (CPU/memory requests, durations, priorities).

Cleaning steps
--------------
1. Read the CSV once (pandas, chunked if memory is tight).
2. Keep only rows whose ``event`` is a real scheduling/finish event
   (SCHEDULE / FINISH / ENABLE) — drops FAIL/EVICT/KILL noise.
3. Parse the stringified dict in ``resource_request`` -> {cpus, memory}.
4. Drop rows with null CPU/mem or non-positive duration.
5. Sample ``pool_size`` tasks, cache the result to .npz keyed by CSV mtime
   so subsequent runs reload in milliseconds.

The cached pool stores plain NumPy arrays consumed directly by env.py.
"""

from __future__ import annotations

import ast
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Fields we extract per task.  Env.py consumes these in this exact order.
TASK_FIELDS = (
    "cpu_req",       # requested cores (float, Borg-normalized 0-1)
    "mem_req",       # requested memory (float, Borg-normalized 0-1)
    "duration_us",   # wall-clock duration in microseconds
    "priority",      # Borg priority 0-450 (higher = more important)
    "sched_class",   # 0-3 latency-sensitive bucket (3 = most latency-sensitive)
    "coll_type",     # 0 = batch, 1 = service
)


def _parse_resource_request(s):
    """resource_request is stored as a Python-literal dict string."""
    if not isinstance(s, str):
        return {}
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return {}


def clean_borg_csv(csv_path: str | Path,
                   keep_events: list[str],
                   min_duration_us: int,
                   pool_size: int,
                   seed: int = 0) -> dict[str, np.ndarray]:
    """One-shot clean of the raw CSV -> dict of task arrays."""
    csv_path = Path(csv_path)
    print(f"[borg_loader] reading {csv_path.name} ({csv_path.stat().st_size/1e6:.0f} MB) ...")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"[borg_loader]   loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # ---- filter to real scheduled/finished tasks ---------------------------
    df = df[df["event"].isin(keep_events)].copy()
    print(f"[borg_loader]   {len(df):,} rows after event filter {keep_events}")

    # ---- parse resource_request --------------------------------------------
    req = df["resource_request"].apply(_parse_resource_request)
    cpu = req.apply(lambda d: d.get("cpus", np.nan)).astype(float)
    mem = req.apply(lambda d: d.get("memory", np.nan)).astype(float)

    # ---- duration (microseconds) -------------------------------------------
    duration = (df["end_time"] - df["start_time"]).astype(float)

    clean = pd.DataFrame({
        "cpu_req": cpu,
        "mem_req": mem,
        "duration_us": duration,
        "priority": df["priority"].astype(float),
        "sched_class": df["scheduling_class"].astype(float),
        "coll_type": df["collection_type"].astype(float),
    })

    # drop anything missing or physically meaningless
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna()
    clean = clean[clean["duration_us"] >= min_duration_us]
    clean = clean[clean["cpu_req"] > 0]
    print(f"[borg_loader]   {len(clean):,} rows after null/positivity filter")

    # ---- sample the pool ----------------------------------------------------
    n = min(pool_size, len(clean))
    if n < len(clean):
        clean = clean.sample(n=n, random_state=seed)
    print(f"[borg_loader]   sampled pool of {len(clean):,} tasks")

    return {col: clean[col].to_numpy(dtype=np.float32) for col in TASK_FIELDS}


def load_task_pool(csv_path: str | Path,
                   cache_path: str | Path,
                   keep_events: list[str],
                   min_duration_us: int,
                   pool_size: int,
                   seed: int = 0,
                   force: bool = False) -> dict[str, np.ndarray]:
    """Cached wrapper around :func:`clean_borg_csv`.

    Recomputes only when the CSV mtime changes, the cache is missing, or
    ``force`` is set — so the expensive 1-2 min parse happens once.
    """
    csv_path, cache_path = Path(csv_path), Path(cache_path)
    csv_mtime = csv_path.stat().st_mtime if csv_path.exists() else 0.0

    if not force and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if (float(cached["csv_mtime"]) == csv_mtime
                    and int(cached["pool_size"]) == pool_size
                    and int(cached["seed"]) == seed):
                print(f"[borg_loader] reusing cached pool {cache_path.name} "
                      f"({int(cached['pool_size']):,} tasks)")
                return {k: cached[k] for k in TASK_FIELDS}
        except Exception:
            pass  # corrupt/old cache -> fall through and rebuild

    pool = clean_borg_csv(csv_path, keep_events, min_duration_us, pool_size, seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save = {"csv_mtime": csv_mtime, "pool_size": pool_size, "seed": seed}
    save.update(pool)
    np.savez(cache_path, **save)
    print(f"[borg_loader] cached pool -> {cache_path}")
    return pool


def pool_summary(pool: dict[str, np.ndarray]) -> dict[str, float]:
    """Quick stats for the reproducibility manifest."""
    return {
        f"{k}_mean": float(np.mean(v)) for k, v in pool.items()
    } | {
        f"{k}_std": float(np.std(v)) for k, v in pool.items()
    } | {"n_tasks": int(len(next(iter(pool.values()))))}
