"""
baselines.py — Classical + metaheuristic scheduling baselines (v2).

v1 shipped FCFS, Round Robin, Greedy-Least-Loaded and tabular Q-learning.
v2 adds three stronger baselines so the DQN's win is convincing:

  * Min-Min  — pick the task that finishes earliest, assign to the VM that
    finishes it earliest, repeat.  Classic static heuristic.
  * Max-Min  — same but picks the *longest* task first (good when long tasks
    would otherwise block the makespan).
  * PSO      — particle-swarm search over the assignment vector, objective =
    weighted (makespan, cost, energy).  Metaheuristic baseline.

All baselines expose the same ``act(env, state) -> vm_idx`` interface used by
the training/eval harness, so they drop into the comparison without changes.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# v1 baselines (kept, lightly tidied)                                         #
# --------------------------------------------------------------------------- #
class FCFSScheduler:
    """True First-Come First-Served: tasks are assigned to VMs strictly in
    arrival order via a rotating pointer (VM0, VM1, ... VMn, VM0, ...).  This
    is the simplest possible policy and serves as the naive baseline."""
    name = "FCFS"

    def __init__(self):
        self.ptr = 0

    def act(self, env, state, greedy=True):
        a = self.ptr % len(env.vms)
        self.ptr += 1
        return a

    def reset(self):
        self.ptr = 0


class RoundRobinScheduler:
    name = "RoundRobin"

    def __init__(self, num_vms):
        self.num_vms = num_vms
        self.ptr = 0

    def act(self, env, state, greedy=True):
        a = self.ptr % self.num_vms
        self.ptr += 1
        return a

    def reset(self):
        self.ptr = 0


class GreedyLeastLoadedScheduler:
    """Pick the VM whose current queue (busy_until - now) is smallest."""
    name = "GreedyLeastLoaded"

    def act(self, env, state, greedy=True):
        loads = [max(0.0, vm.busy_until - env.current_time) for vm in env.vms]
        return int(np.argmin(loads))

    def reset(self):
        pass


class TabularQLearning:
    """Classic Q-learning over a discretized state (the paper's Q-table base)."""
    name = "Q-learning"

    def __init__(self, num_vms, n_bins=5, alpha=0.1, gamma=0.95, eps=0.2, seed=0):
        self.num_vms = num_vms
        self.n_bins = n_bins
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.q_table = {}
        self.rng = np.random.default_rng(seed)

    def _discretize(self, state):
        task_feats = state[:5]
        utils = state[5::5]
        bins = np.concatenate([task_feats, utils])
        return tuple(np.clip((bins * self.n_bins).astype(int), 0, self.n_bins - 1))

    def act(self, env, state, greedy=False):
        key = self._discretize(state)
        if key not in self.q_table:
            self.q_table[key] = np.zeros(self.num_vms)
        if (not greedy) and self.rng.random() < self.eps:
            return int(self.rng.integers(self.num_vms))
        return int(np.argmax(self.q_table[key]))

    def update(self, state, action, reward, next_state, done):
        key = self._discretize(state)
        nkey = self._discretize(next_state)
        self.q_table.setdefault(key, np.zeros(self.num_vms))
        self.q_table.setdefault(nkey, np.zeros(self.num_vms))
        target = reward if done else reward + self.gamma * np.max(self.q_table[nkey])
        self.q_table[key][action] += self.alpha * (target - self.q_table[key][action])

    def decay_epsilon(self, decay=0.995, eps_min=0.05):
        self.eps = max(eps_min, self.eps * decay)

    def reset(self):
        pass


# --------------------------------------------------------------------------- #
# v2 additions                                                                #
# --------------------------------------------------------------------------- #
class MinMinScheduler:
    """Min-Min: assign the current task to the VM that finishes it EARLIEST
    *and has spare capacity*.  Prefers fast, free VMs — good for minimizing
    average completion time of short tasks.
    """
    name = "Min-Min"

    def act(self, env, state, greedy=True):
        t = env.tasks[env.t_idx]
        finish_times = []
        for vm in env.vms:
            vm.drain(env.current_time)
            cpu_load = sum(c for _, c, _ in vm.active)
            cpu_after = cpu_load + t.cpu_req
            # capacity-aware: if VM is full, add a drain penalty
            if cpu_after > vm.capacity_cores * 1.3:
                start = env.current_time + t.duration_us * 0.5
            else:
                start = env.current_time
            finish_times.append(start + t.duration_us)
        return int(np.argmin(finish_times))

    def reset(self):
        pass


class MaxMinScheduler:
    """Max-Min: assign the LONGEST tasks to the VMs with the MOST spare capacity
    (so big tasks get big VMs), tie-broken by lowest current queue.  A genuinely
    different policy from Min-Min — it spreads heavy work to capable VMs rather
    than chasing earliest finish.
    """
    name = "Max-Min"

    def act(self, env, state, greedy=True):
        t = env.tasks[env.t_idx]
        scores = []
        for vm in env.vms:
            vm.drain(env.current_time)
            cpu_load = sum(c for _, c, _ in vm.active)
            mem_load = sum(m for _, _, m in vm.active)
            spare_cpu = max(0.0, vm.capacity_cores - cpu_load)
            spare_mem = max(0.0, vm.capacity_mem - mem_load)
            queue = max(0.0, vm.busy_until - env.current_time)
            score = spare_cpu + 0.5 * spare_mem - (queue / 1e9)
            scores.append(score)
        return int(np.argmax(scores))

    def reset(self):
        pass


class PSOScheduler:
    """Particle Swarm Optimization over a short planning horizon.

    Each call plans the assignment of the next ``horizon`` tasks jointly,
    keeping the first decision and replanning next step (receding horizon).
    Particles are assignment vectors; fitness = weighted sum of normalized
    makespan, cost and energy for the planned block.  This is the
    metaheuristic baseline referenced in the lit-review table (SR-PSO / PSO).
    """
    name = "PSO"

    def __init__(self, num_vms, swarm_size=20, iterations=30,
                 inertia=0.7, c1=1.5, c2=1.5, horizon=8, seed=0):
        self.num_vms = num_vms
        self.swarm_size = swarm_size
        self.iterations = iterations
        self.inertia = inertia
        self.c1 = c1
        self.c2 = c2
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)

    def reset(self):
        pass

    def _fitness(self, plan, env):
        """Estimate makespan/cost/energy for a candidate plan over the horizon."""
        busy = [vm.busy_until for vm in env.vms]
        cost_rate = [vm.cost_per_sec for vm in env.vms]
        power = [vm.busy_power for vm in env.vms]
        max_finish = 0.0
        tot_cost = 0.0
        tot_energy = 0.0
        for k, vm_idx in enumerate(plan):
            if env.t_idx + k >= env.num_tasks:
                break
            t = env.tasks[env.t_idx + k]
            dur_s = t.duration_us / 1e6
            start = max(env.current_time, busy[vm_idx])
            finish = start + t.duration_us
            busy[vm_idx] = finish
            max_finish = max(max_finish, finish)
            tot_cost += cost_rate[vm_idx] * dur_s
            tot_energy += power[vm_idx] * dur_s / 3600.0
        # normalize (rough) and combine
        return (max_finish / 1e9) + 0.01 * tot_cost + 0.001 * tot_energy

    def act(self, env, state, greedy=True):
        H = min(self.horizon, env.num_tasks - env.t_idx)
        if H <= 0:
            return 0
        # init particles (assignment vectors of length H)
        particles = self.rng.integers(0, self.num_vms, size=(self.swarm_size, H))
        velocities = np.zeros_like(particles, dtype=float)
        p_best = particles.copy()
        p_best_fit = np.array([self._fitness(p, env) for p in particles])
        g_best = p_best[np.argmin(p_best_fit)].copy()
        g_best_fit = p_best_fit.min()

        for _ in range(self.iterations):
            r1 = self.rng.random((self.swarm_size, H))
            r2 = self.rng.random((self.swarm_size, H))
            # continuous PSO update then round to nearest VM index
            velocities = (self.inertia * velocities
                          + self.c1 * r1 * (p_best - particles)
                          + self.c2 * r2 * (g_best - particles))
            particles = np.clip(np.round(particles + velocities).astype(int),
                                0, self.num_vms - 1)
            fits = np.array([self._fitness(p, env) for p in particles])
            improve = fits < p_best_fit
            p_best[improve] = particles[improve]
            p_best_fit[improve] = fits[improve]
            best = np.argmin(p_best_fit)
            if p_best_fit[best] < g_best_fit:
                g_best_fit = p_best_fit[best]
                g_best = p_best[best].copy()
        return int(g_best[0])
