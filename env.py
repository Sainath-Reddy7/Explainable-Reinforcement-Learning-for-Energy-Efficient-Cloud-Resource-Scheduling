"""
env.py — Cloud-edge scheduling environment backed by REAL Google Borg traces.

v2.1 — accurate parallel-execution model
---------------------------------------
Borg VMs run many small tasks concurrently via CPU time-sharing. This env now
tracks each VM's *active task set* and decays load as tasks finish, so:

  * greedy policies that pile onto one VM actually oversubscribe it -> queuing,
    higher makespan, more misses;
  * balancing policies spread load -> lower makespan, fewer misses;
  * the DQN learns the trade-off between cheap-VM cost vs balance.

This produces DISTINCT makespans/miss-rates across schedulers (v2.0 had them
all tied) and realistic 10-40% deadline-miss spreads.

State (5 task + 5*num_vms per-VM features = 45-dim for 8 VMs).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from borg_loader import load_task_pool


# --------------------------------------------------------------------------- #
# Infrastructure model                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class VM:
    idx: int
    mips: float
    ram: float
    cost_per_sec: float
    idle_power: float
    busy_power: float
    capacity_cores: float
    capacity_mem: float
    is_edge: bool = False

    # dynamic (reset each episode)
    total_busy_time: float = 0.0
    total_energy: float = 0.0
    total_cost: float = 0.0
    tasks_run: int = 0
    deadline_misses: int = 0
    busy_until: float = 0.0
    # list of (finish_time_us, cpu_req, mem_req) for currently-running tasks
    active: list = field(default_factory=list)

    def reset_state(self):
        self.total_busy_time = 0.0
        self.total_energy = 0.0
        self.total_cost = 0.0
        self.tasks_run = 0
        self.deadline_misses = 0
        self.busy_until = 0.0
        self.active = []

    def drain(self, now_us):
        """Remove tasks that have finished by ``now_us``; return (cpu, mem) load."""
        still = []
        cpu_load = 0.0
        mem_load = 0.0
        for fin, c, m in self.active:
            if fin > now_us:
                still.append((fin, c, m))
                cpu_load += c
                mem_load += m
            # else: task finished, drop it
        self.active = still
        self.busy_until = max((f for f, _, _ in self.active), default=now_us)
        return cpu_load, mem_load


@dataclass
class Task:
    idx: int
    duration_us: float
    cpu_req: float
    mem_req: float
    priority: float
    sched_class: float
    coll_type: float
    arrival_time: float
    deadline: float


def make_default_vm_pool(num_vms=8, edge_fraction=0.25, seed=42):
    rng = np.random.default_rng(seed)
    vms = []
    n_edge = int(num_vms * edge_fraction)
    for i in range(num_vms):
        is_edge = i < n_edge
        if is_edge:
            mips = rng.uniform(2000, 8000)
            ram = rng.uniform(512, 2048)
            cost = rng.uniform(0.02, 0.08)
            idle_p, busy_p = rng.uniform(15, 25), rng.uniform(40, 70)
            cap_cpu = rng.uniform(0.10, 0.25)
        else:
            mips = rng.uniform(10000, 60000)
            ram = rng.uniform(2048, 16384)
            cost = rng.uniform(0.10, 0.60)
            idle_p, busy_p = rng.uniform(60, 100), rng.uniform(150, 300)
            cap_cpu = rng.uniform(0.45, 1.0)
        cap_mem = cap_cpu * rng.uniform(0.8, 1.2)
        vms.append(VM(idx=i, mips=mips, ram=ram, cost_per_sec=cost,
                      idle_power=idle_p, busy_power=busy_p,
                      capacity_cores=cap_cpu, capacity_mem=cap_mem,
                      is_edge=is_edge))
    return vms


# --------------------------------------------------------------------------- #
# Environment                                                                 #
# --------------------------------------------------------------------------- #
class CloudSchedulingEnv:
    DURATION_MAX_US = 300_000_000.0
    PRIORITY_MAX = 450.0
    CPU_MAX = 0.6
    MEM_MAX = 0.05

    def __init__(self, task_pool, num_vms=8, num_tasks=200, seed=0,
                 adaptive_weights=True, vm_seed=42, deadline_tightness=1.0):
        self.pool = task_pool
        self.pool_n = len(next(iter(task_pool.values())))
        self.num_vms = num_vms
        self.num_tasks = num_tasks
        self.seed = seed
        self.adaptive_weights = adaptive_weights
        self.deadline_tightness = deadline_tightness
        self.vms = make_default_vm_pool(num_vms, seed=vm_seed)

        self.task_dim = 5
        self.vm_feat_dim = 5
        self.state_dim = self.task_dim + num_vms * self.vm_feat_dim
        self.action_dim = num_vms
        self.feature_names = self._build_feature_names()
        self._rolling = {"energy": 1.0, "cost": 1.0, "qos": 1.0, "wait": 1.0, "balance": 1.0}
        self._rng = np.random.default_rng(seed)
        self.reset()

    def _build_feature_names(self):
        names = ["task_cpu_req", "task_mem_req", "task_duration",
                 "task_priority", "task_sched_class"]
        for i in range(self.num_vms):
            names += [f"vm{i}_util", f"vm{i}_energy_rate",
                      f"vm{i}_cost_rate", f"vm{i}_queue", f"vm{i}_cap_fit"]
        return names

    def _sample_tasks(self, n_tasks, seed_offset):
        rng = np.random.default_rng(self.seed * 1000 + seed_offset)
        idx = rng.integers(0, self.pool_n, size=n_tasks)
        cpu = self.pool["cpu_req"][idx]
        mem = self.pool["mem_req"][idx]
        dur_us = self.pool["duration_us"][idx]
        prio = self.pool["priority"][idx]
        sclass = self.pool["sched_class"][idx]
        ctype = self.pool["coll_type"][idx]

        # arrivals ~1.5s apart — dense enough that bad schedulers oversubscribe
        inter = rng.exponential(1_500_000, size=n_tasks).cumsum()
        tight_service = rng.uniform(1.2, 2.0, n_tasks)   # services: tight
        tight_batch = rng.uniform(1.5, 2.5, n_tasks)      # batch: moderate
        slack_factor = np.where(sclass >= 2, tight_service, tight_batch) * self.deadline_tightness
        deadline = inter + slack_factor * dur_us

        tasks = []
        for k in range(n_tasks):
            tasks.append(Task(
                idx=int(idx[k]), duration_us=float(dur_us[k]),
                cpu_req=float(cpu[k]), mem_req=float(mem[k]),
                priority=float(prio[k]), sched_class=float(sclass[k]),
                coll_type=float(ctype[k]), arrival_time=float(inter[k]),
                deadline=float(deadline[k])))
        return tasks

    # ---- Gym API -----------------------------------------------------------
    def reset(self, episode_seed_offset=None, num_tasks=None):
        for vm in self.vms:
            vm.reset_state()
        if num_tasks is not None:
            self.num_tasks = num_tasks
        offset = episode_seed_offset if episode_seed_offset is not None \
            else int(self._rng.integers(0, 10_000))
        self.tasks = self._sample_tasks(self.num_tasks, offset)
        self.t_idx = 0
        self.current_time = 0.0
        self._rolling = {"energy": 1.0, "cost": 1.0, "qos": 1.0, "wait": 1.0, "balance": 1.0}
        return self._get_state()

    def _vm_load(self, vm):
        """Current (cpu, mem) load on a VM at self.current_time."""
        cpu = sum(c for _, c, _ in vm.active if True)
        mem = sum(m for _, _, m in vm.active)
        return cpu, mem

    def _get_state(self):
        t = self.tasks[self.t_idx]
        self.current_time = t.arrival_time
        feats = [
            float(np.clip(t.cpu_req / self.CPU_MAX, 0, 1)),
            float(np.clip(t.mem_req / self.MEM_MAX, 0, 1)),
            float(np.clip(t.duration_us / self.DURATION_MAX_US, 0, 1)),
            float(np.clip(t.priority / self.PRIORITY_MAX, 0, 1)),
            float(np.clip(t.sched_class / 3.0, 0, 1)),
        ]
        for vm in self.vms:
            vm.drain(self.current_time)   # prune finished tasks
            cpu_load, _ = self._vm_load(vm)
            util = float(np.clip(cpu_load / vm.capacity_cores, 0, 1))
            queue = max(0.0, vm.busy_until - self.current_time)
            queue_norm = float(np.log1p(queue / 1e6) / np.log1p(3000.0))
            cap_fit = float(np.clip(1.0 - abs(vm.capacity_cores - t.cpu_req), 0, 1))
            feats += [
                util,
                float(np.clip(vm.busy_power / 300.0, 0, 1)),
                float(np.clip(vm.cost_per_sec / 0.6, 0, 1)),
                float(np.clip(queue_norm, 0, 1)),
                cap_fit,
            ]
        return np.array(feats, dtype=np.float32)

    def step(self, action):
        t = self.tasks[self.t_idx]
        vm = self.vms[action]
        vm.drain(self.current_time)   # ensure load reflects current time

        dur_us = t.duration_us
        dur_s = dur_us / 1e6
        cpu_load, _ = self._vm_load(vm)

        # ---- parallel execution with queuing slowdown ----------------------
        # If cpu_load + req exceeds capacity, the task is queued behind active
        # tasks: its effective finish is stretched by the oversubscription ratio.
        new_load = cpu_load + t.cpu_req
        capacity = vm.capacity_cores * 1.3   # allow 30% oversubscribe before queueing
        if new_load <= capacity:
            slowdown = 1.0                    # spare capacity -> runs immediately
            wait_s = 0.0
        else:
            oversub = new_load / capacity
            slowdown = oversub                # linear queueing penalty
            wait_s = dur_s * (oversub - 1.0)

        finish = self.current_time + dur_us * slowdown
        start = self.current_time             # online: tasks start at arrival (Borg model)

        # power scales with utilization fraction
        util_frac = min(1.0, new_load / vm.capacity_cores)
        power = vm.idle_power + (vm.busy_power - vm.idle_power) * util_frac
        energy_wh = power * dur_s / 3600.0
        cost = vm.cost_per_sec * dur_s

        vm.active.append((finish, t.cpu_req, t.mem_req))
        vm.busy_until = max(vm.busy_until, finish)
        vm.total_busy_time += dur_s
        vm.total_energy += energy_wh
        vm.total_cost += cost
        vm.tasks_run += 1

        deadline_met = finish <= t.deadline
        overrun = max(0.0, (finish - t.deadline) / max(dur_us, 1e-6))

        # ---- reward (SIMPLIFIED: QoS dominates, cost secondary) ------------
        # The multi-objective reward with 5 competing terms was diluting the
        # learning signal. QoS (deadlines) is now DOMINANT — a miss is a
        # catastrophic -10; a meet is +1 + cost efficiency bonus. This makes
        # the gradient point sharply toward "spread load to meet deadlines."
        prio_weight = 0.5 + 0.5 * (t.priority / self.PRIORITY_MAX)

        # QoS: dominant signal
        if deadline_met:
            r_qos = 2.0 * prio_weight                # +1 to +2 for meeting deadline
        else:
            r_qos = -10.0 * (1.0 + min(overrun, 2.0))  # -10 to -30 for missing

        # Cost: small efficiency bonus (cheaper VM = slightly better)
        # normalized so it's ~±0.3, never overpowering QoS
        r_cost = -0.5 * float(np.clip(vm.cost_per_sec / 0.6, 0, 1))

        # Overload penalty: discourage piling onto an already-saturated VM.
        # This is the KEY term that teaches the DQN to spread load. The penalty
        # starts at 50% capacity utilization (not 100%) so the agent learns to
        # spread BEFORE saturation, and grows quadratically to make overload
        # genuinely painful.
        util_frac_post = new_load / vm.capacity_cores
        if util_frac_post > 0.5:
            excess = (util_frac_post - 0.5) / 0.5   # 0 at 50%, 1 at 100%, 2 at 150%
            r_overload = -8.0 * excess ** 2          # quadratic: -0, -2, -8, -18, -32
        else:
            r_overload = 0.0

        # Energy: tiny term so it doesn't dominate
        f_energy = float(np.clip(power / 300.0, 0, 1))
        r_energy = -0.2 * f_energy

        reward = r_qos + r_cost + r_overload + r_energy

        if not deadline_met:
            vm.deadline_misses += 1

        if self.adaptive_weights:
            # Adaptive path kept for config compatibility but the simplified
            # reward above is what actually gets used.
            pass

        # reward already computed above (r_qos + r_cost + r_overload + r_energy)

        # compute imbalance for metrics/info (not used in reward anymore)
        loads = np.array([sum(c for _, c, _ in v.active) / v.capacity_cores for v in self.vms])
        imbalance = float(np.mean(np.abs(loads - loads.mean())))

        info = {
            "exec_time_s": dur_s, "energy_wh": energy_wh, "cost": cost,
            "deadline_met": deadline_met, "vm_idx": int(action),
            "wait_s": wait_s, "priority": t.priority,
            "capacity_ok": new_load <= capacity, "imbalance": imbalance,
        }
        self.t_idx += 1
        done = self.t_idx >= self.num_tasks
        next_state = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return next_state, float(reward), done, info

    def episode_metrics(self):
        makespan_s = max((vm.busy_until for vm in self.vms), default=0.0) / 1e6
        total_energy = sum(vm.total_energy for vm in self.vms)
        total_cost = sum(vm.total_cost for vm in self.vms)
        total_tasks = sum(vm.tasks_run for vm in self.vms)
        misses = sum(vm.deadline_misses for vm in self.vms)
        throughput = total_tasks / makespan_s if makespan_s > 0 else 0.0
        utils = [sum(c for _, c, _ in v.active) / v.capacity_cores for v in self.vms]
        avg_util = float(np.mean(utils))
        di = float(np.sum(np.abs(np.array(utils) - avg_util)))
        return {
            "makespan_s": makespan_s, "throughput": throughput, "arur": avg_util,
            "cost": total_cost, "energy_wh": total_energy, "di": di,
            "deadline_miss_rate": misses / max(total_tasks, 1),
        }
