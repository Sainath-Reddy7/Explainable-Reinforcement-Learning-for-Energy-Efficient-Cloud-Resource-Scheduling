"""
energy_model.py — Realistic energy + carbon accounting (roadmap item S9).

v2 charged energy as `power x duration` with constant idle/busy power —
linear, idle-unaware, carbon-blind. Q1 reviewers in the energy-efficient
scheduling space reject that. This module supplies realistic METERS while
leaving the policies untouched (same action space, same reward during
training) — the study then measures how scheduler rankings change when the
physics is taken seriously:

  1. Fan et al. (2007) power model:  P(u) = P_idle + (P_peak - P_idle)(2u - u^2)
     the standard energy-proportionality curve (sub-linear, citable).
  2. DVFS P-states: K discrete frequency levels; dynamic power scales ~ f^3
     (voltage-scaled CMOS) and execution speed scales ~ f. Two governors:
       race-to-idle  run at f_max, finish early, sleep sooner
       pace          run at the lowest f whose stretched duration still
                     meets the task deadline estimate
  3. Sleep states: idle gaps longer than `sleep_threshold_s` cost P_sleep
     instead of P_idle, plus a one-shot wake energy penalty per wake.
  4. Carbon-aware: time-varying grid intensity g(t) (gCO2eq/kWh) with a
     realistic 24h double-peak profile (solar-heavy grid, morning + evening
     peaks, midday trough) — representative of published ElectricityMaps
     profiles; emissions = energy(t) * g(hour(t)) accumulated per decision.

Usage: attach to CloudSchedulingEnv(env=..., energy_model=EnergyModel(...)).
The env then reports energy_wh under the new physics and adds carbon_g
to episode metrics. Default (None) keeps the legacy v2 accounting.
"""

from __future__ import annotations

import numpy as np


# 24-hour carbon intensity profile, gCO2eq/kWh (hour -> intensity).
# Shape: overnight low, morning ramp, midday solar trough, evening peak.
# Representative of published day-ahead/solar-heavy grid profiles.
CARBON_PROFILE_H = np.array([
    320, 300, 290, 285, 290, 320,      # 00-05h
    380, 450, 470, 430, 360, 300,      # 06-11h  morning peak, solar ramps in
    260, 240, 235, 250, 290, 350,      # 12-17h  solar trough
    430, 480, 470, 430, 390, 350,      # 18-23h  evening peak
], dtype=np.float64)


def carbon_intensity(env_time_us: float) -> float:
    """Grid intensity at a virtual timestamp. Episodes run on microsecond
    clocks; map to hour-of-day so multi-day traces cycle the profile."""
    hour = (env_time_us / 3.6e9) % 24.0
    lo = int(hour) % 24
    hi = (lo + 1) % 24
    frac = hour - int(hour)
    return float(CARBON_PROFILE_H[lo] * (1 - frac) + CARBON_PROFILE_H[hi] * frac)


class EnergyModel:
    """Fan-curve power + DVFS governors + sleep states + carbon accounting."""

    P_STATES = (0.6, 0.8, 1.0)          # relative frequency levels
    F_EXP = 3.0                          # dynamic power ~ f^3 (V ~ f)

    def __init__(self, governor="race_to_idle", sleep_threshold_s=60.0,
                 p_sleep_frac=0.05, wake_penalty_wh=0.5,
                 carbon_aware=True, p_idle_frac=1.0):
        assert governor in ("race_to_idle", "pace", "static_max")
        self.governor = governor
        self.sleep_threshold_s = sleep_threshold_s
        self.p_sleep = p_sleep_frac        # fraction of idle power when asleep
        self.wake_penalty_wh = wake_penalty_wh
        self.carbon_aware = carbon_aware
        self.p_idle_frac = p_idle_frac

    # ---- instantaneous power (Fan et al. 2007 + DVFS) --------------------
    def power(self, vm, util_frac: float, freq: float) -> float:
        idle = vm.idle_power * self.p_idle_frac
        peak_dyn = vm.busy_power - vm.idle_power
        u = float(np.clip(util_frac, 0.0, 1.0))
        fan = 2 * u - u * u                        # energy-proportionality
        # static part scales mildly with f; dynamic part with f^3
        return idle + peak_dyn * fan * (freq ** self.F_EXP)

    def choose_freq(self, task_deadline_us: float, now_us: float,
                    dur_us: float) -> float:
        if self.governor == "race_to_idle" or self.governor == "static_max":
            return 1.0
        # pace: lowest P-state whose stretched duration still meets deadline
        slack = (task_deadline_us - now_us) / max(dur_us, 1e-6)
        for f in self.P_STATES:
            if f >= 1.0 / max(slack, 1e-6):
                return f
        return 1.0

    # ---- per-task energy ---------------------------------------------------
    def task_energy_wh(self, vm, dur_s: float, util_frac: float,
                       freq: float) -> float:
        p = self.power(vm, util_frac, freq)
        return p * (dur_s / freq) / 3600.0

    # ---- idle/sleep accounting ----------------------------------------------
    def idle_energy_wh(self, vm, idle_s: float) -> float:
        if idle_s <= 0:
            return 0.0
        if idle_s > self.sleep_threshold_s:
            # first seconds awake, then asleep (+ wake penalty charged by env)
            awake = self.sleep_threshold_s
            return (vm.idle_power * awake
                    + vm.idle_power * self.p_sleep * (idle_s - awake)) / 3600.0
        return vm.idle_power * idle_s / 3600.0

    # ---- carbon ------------------------------------------------------------
    def carbon_g(self, energy_wh: float, env_time_us: float) -> float:
        if not self.carbon_aware:
            return 0.0
        return energy_wh * carbon_intensity(env_time_us) / 1000.0


# --------------------------------------------------------------------------- #
# Study: scheduler rankings under different energy physics                    #
# --------------------------------------------------------------------------- #
def run_energy_study(pool, cfg, seed=0, num_tasks=300, n_runs=2,
                     schedulers=None, out=None):
    """Evaluate the same schedulers under increasingly realistic energy
    physics. Rankings that hold across models are robust; rankings that
    flip are artifacts of the v2 linear meter."""
    import copy as _copy
    import json
    from env import CloudSchedulingEnv
    from train import evaluate_scheduler
    from baselines import (FCFSScheduler, MinMinScheduler, MaxMinScheduler,
                           GreedyLeastLoadedScheduler, PSOScheduler)

    settings = {
        "v2_linear": None,                                  # legacy meter
        "fan": EnergyModel(governor="static_max"),
        "fan_dvfs_pace": EnergyModel(governor="pace"),
    }

    if schedulers is None:
        schedulers = {
            "FCFS": lambda: FCFSScheduler(),
            "Min-Min": lambda: MinMinScheduler(),
            "Max-Min": lambda: MaxMinScheduler(),
            "Greedy": lambda: GreedyLeastLoadedScheduler(),
            "PSO": lambda: PSOScheduler(cfg["env"]["num_vms"],
                                        **cfg["baselines"]["pso"]),
        }

    results = {}
    for name, em in settings.items():
        cfg2 = _copy.deepcopy(cfg)
        cfg2["env"]["energy_model"] = em
        results[name] = {}
        for sname, factory in schedulers.items():
            m = evaluate_scheduler(factory, pool, cfg2, num_tasks, seed,
                                   n_runs=n_runs, vm_seed=cfg["env"]["vm_seed"])
            results[name][sname] = {"energy_wh": m["energy_wh"],
                                    "carbon_g": m.get("carbon_g", 0.0),
                                    "cost": m["cost"],
                                    "miss": m["deadline_miss_rate"],
                                    "makespan_s": m["makespan_s"]}
    if out:
        json.dump(results, open(out, "w"), indent=1)
    return results
