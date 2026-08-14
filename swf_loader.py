"""
swf_loader.py — Parse Standard Workload Format (SWF) HPC traces into the
same task-pool schema used by borg_loader, so the entire pipeline (env, DQN,
baselines, XAI) runs unmodified on a second REAL dataset.

Dataset: KTH SP2 (IBM SP2, Swedish Royal Institute of Technology, 1996).
Source: Parallel Workloads Archive — KTH-SP2-1996-2.1-cln.swf (cleaned log).
~28,476 production batch jobs with submit times, run times, processor and
memory requests.

SWF columns (1-indexed):
  1 job#  2 submit  3 wait  4 run  5 alloc_procs  6 avg_cpu  7 used_mem
  8 req_procs  9 req_mem  10 req_time  11 status  12 user  13 group
  14 exec#  15 queue  16 partition  17 preceding  18 think

Mapping to our task schema
--------------------------
  cpu_req     = req_procs / 100            (KTH SP2 has 100 nodes)
  mem_req     = req_memKB scaled to [0, ~0.05] via 99th percentile (Borg-like)
  duration_us = run_time * 1e6
  priority    = queue number * 75, clipped to [0, 450]   (queue = priority proxy)
  sched_class = latency bucket from run time (shorter job -> more sensitive)
  coll_type   = 0 (all HPC jobs are batch)

Only completed jobs (status == 1) with positive runtime are kept, mirroring
the Borg loader's SCHEDULE/FINISH/ENABLE filter.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

TASK_FIELDS = ("cpu_req", "mem_req", "duration_us",
               "priority", "sched_class", "coll_type")

TOTAL_NODES = 100          # KTH SP2 installation size
MAX_PRIORITY = 450.0


def _sched_class_from_runtime(run_s: np.ndarray) -> np.ndarray:
    """Short jobs are more latency-sensitive: <10min=3, <1h=2, <24h=1, else 0."""
    cls = np.zeros_like(run_s, dtype=np.float32)
    cls[run_s < 86400] = 1
    cls[run_s < 3600] = 2
    cls[run_s < 600] = 3
    return cls


def clean_swf(path: str | Path, pool_size: int = 5000,
              min_runtime_s: float = 1.0, seed: int = 0) -> dict[str, np.ndarray]:
    path = Path(path)
    opener = gzip.open if str(path).endswith(".gz") else open
    rows = []
    with opener(path, "rt") as f:
        for line in f:
            if line.startswith(";") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            submit = float(parts[1])
            run = float(parts[3])
            req_procs = float(parts[7]) if float(parts[7]) > 0 else float(parts[4])
            req_mem = float(parts[8]) if float(parts[8]) > 0 else 0.0
            status = int(parts[10])
            queue = int(parts[14])
            if status != 1 or run < min_runtime_s or req_procs <= 0:
                continue
            rows.append((submit, run, req_procs, req_mem, queue))
    import pandas as pd
    df = pd.DataFrame(rows, columns=["submit", "run", "req_procs", "req_mem", "queue"])
    print(f"[swf_loader] {path.name}: kept {len(df):,} completed jobs "
          f"(of {len(rows) + (0 if not rows else 0):,} parsed, "
          f"spanning {df['submit'].max()/86400:.0f} days)")

    # normalize CPU to Borg-like [0, ~1] scale (fraction of the machine)
    cpu = (df["req_procs"] / TOTAL_NODES).clip(0, 1).to_numpy(np.float32)

    # normalize memory to Borg-like [0, ~0.05] via the 99th percentile
    mem_raw = df["req_mem"].to_numpy(np.float64)
    p99 = np.percentile(mem_raw[mem_raw > 0], 99) if (mem_raw > 0).any() else 1.0
    mem = np.clip(mem_raw / max(p99, 1e-9) * 0.05, 0, 0.05).astype(np.float32)

    dur = (df["run"].to_numpy(np.float64) * 1e6).astype(np.float32)
    prio = np.clip(df["queue"].to_numpy(np.float64) * 75.0, 0, MAX_PRIORITY).astype(np.float32)
    sclass = _sched_class_from_runtime(df["run"].to_numpy(np.float64))
    ctype = np.zeros(len(df), dtype=np.float32)

    # sample pool
    n = min(pool_size, len(df))
    if n < len(df):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(df), size=n, replace=False)
        cpu, mem, dur = cpu[idx], mem[idx], dur[idx]
        prio, sclass, ctype = prio[idx], sclass[idx], ctype[idx]
    print(f"[swf_loader] sampled pool of {n:,} tasks")

    return dict(zip(TASK_FIELDS, (cpu, mem, dur, prio, sclass, ctype)))


def load_swf_pool(path: str | Path, cache_path: str | Path, pool_size: int = 5000,
                  seed: int = 0, force: bool = False) -> dict[str, np.ndarray]:
    """Cached wrapper (mtime-keyed), mirroring borg_loader.load_task_pool."""
    path, cache_path = Path(path), Path(cache_path)
    mtime = path.stat().st_mtime if path.exists() else 0.0
    if not force and cache_path.exists():
        try:
            cached = np.load(cache_path, allow_pickle=False)
            if (float(cached["src_mtime"]) == mtime
                    and int(cached["pool_size"]) == pool_size
                    and int(cached["seed"]) == seed):
                print(f"[swf_loader] reusing cached pool {cache_path.name} "
                      f"({int(cached['pool_size']):,} tasks)")
                return {k: cached[k] for k in TASK_FIELDS}
        except Exception:
            pass
    pool = clean_swf(path, pool_size=pool_size, seed=seed)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    save = {"src_mtime": mtime, "pool_size": pool_size, "seed": seed}
    save.update(pool)
    np.savez(cache_path, **save)
    print(f"[swf_loader] cached pool -> {cache_path}")
    return pool
