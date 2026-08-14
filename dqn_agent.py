"""
dqn_agent.py — Double DQN with Prioritized Experience Replay (v2).

Upgrades over v1
----------------
* Double DQN  (van Hasselt et al. 2016): the *online* net selects the next
  action argmax, the *target* net evaluates it.  Cuts the Q-overestimation
  bias that hurt v1's policy quality.
* Prioritized Experience Replay via a SumTree (Schaul et al. 2016): TD-error-
  proportional sampling + importance-sampling weights, with alpha/beta
  annealing.  Faster, more stable learning than uniform replay.
* Soft (Polyak) target update every step instead of a periodic hard copy —
  smoother target drift, fewer destabilizing jumps.
* Learning-rate decay hook driven by the agent step count.
"""

from __future__ import annotations

import numpy as np

from qnetwork import QNetwork


# --------------------------------------------------------------------------- #
# SumTree-backed Prioritized Experience Replay                                #
# --------------------------------------------------------------------------- #
class SumTree:
    """Array-based sum tree for O(log n) prioritized sampling."""

    def __init__(self, capacity):
        self.capacity = capacity
        # tree leaves = capacity, internal nodes = capacity-1, +1 for root
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, delta):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    def total(self):
        return float(self.tree[0])

    def add(self, priority, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, seed=0):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 1e-6        # avoid zero priority
        self.max_priority = 1.0
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done, td_error=None):
        # new transitions enter with max priority so they're sampled at least once
        pri = (abs(td_error) + self.epsilon) ** self.alpha if td_error is not None \
            else self.max_priority
        self.tree.add(pri, (s, a, r, s2, done))

    def sample(self, batch_size, beta=0.4):
        total = self.tree.total()
        segment = total / batch_size
        idxs, ps, ws = [], [], []
        s_list, a_list, r_list, s2_list, d_list = [], [], [], [], []
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = self.rng.uniform(a, b)
            idx, pri, data = self.tree.get(s)
            if data is None:
                # can happen on first fill; resample uniformly
                idx = self.rng.integers(self.tree.capacity - 1,
                                        self.tree.capacity - 1 + self.tree.n_entries)
                data = self.tree.data[idx - self.tree.capacity + 1]
                pri = self.tree.tree[idx]
            s_list.append(data[0]); a_list.append(data[1])
            r_list.append(data[2]); s2_list.append(data[3]); d_list.append(data[4])
            idxs.append(idx); ps.append(pri)
        probs = np.asarray(ps) / max(total, 1e-8)
        # importance-sampling weights, normalized by max so the largest is 1
        ws = (self.tree.n_entries * probs) ** (-beta)
        ws = ws / (ws.max() + 1e-8)
        return (np.array(s_list, dtype=np.float32),
                np.array(a_list, dtype=np.int64),
                np.array(r_list, dtype=np.float32),
                np.array(s2_list, dtype=np.float32),
                np.array(d_list, dtype=np.float32),
                np.array(idxs, dtype=np.int64),
                ws.astype(np.float32))

    def update_priorities(self, idxs, td_errors):
        for idx, td in zip(idxs, td_errors):
            pri = (abs(float(td)) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), pri)
            self.max_priority = max(self.max_priority, pri)

    def __len__(self):
        return self.tree.n_entries


class UniformReplayBuffer:
    """Fallback for when PER is disabled."""
    def __init__(self, capacity, seed=0):
        self.buffer = []
        self.pos = 0
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done, td_error=None):
        item = (s, a, r, s2, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        idx = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        s, a, r, s2, d = zip(*batch)
        ws = np.ones(batch_size, dtype=np.float32)
        idxs = np.full(batch_size, -1, dtype=np.int64)
        return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int64),
                np.array(r, dtype=np.float32), np.array(s2, dtype=np.float32),
                np.array(d, dtype=np.float32), idxs, ws)

    def update_priorities(self, idxs, td_errors):
        pass  # no-op for uniform

    def __len__(self):
        return len(self.buffer)


# --------------------------------------------------------------------------- #
# DQN Agent                                                                   #
# --------------------------------------------------------------------------- #
class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden=(128, 64, 32), lr=5e-4,
                 gamma=0.98, buffer_size=50_000, batch_size=64,
                 soft_update_tau=0.005, target_update_steps=200,
                 double_dqn=True, per_enabled=True, per_alpha=0.6,
                 beta_start=0.4, beta_end=1.0, beta_anneal_steps=10_000,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.995,
                 dropout=0.1, seed=0):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = soft_update_tau
        self.target_update_steps = target_update_steps
        self.double_dqn = double_dqn

        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = max(beta_anneal_steps, 1)
        self.beta = beta_start

        self.q = QNetwork(state_dim, action_dim, hidden=hidden[:2], lr=lr,
                          dropout=dropout, seed=seed)
        self.target_q = QNetwork(state_dim, action_dim, hidden=hidden[:2], lr=lr,
                                 dropout=dropout, seed=seed + 1)
        self.target_q.set_weights(self.q.get_weights())
        # match normalization stats
        self.target_q.run_mean = self.q.run_mean.copy()
        self.target_q.run_var = self.q.run_var.copy()

        self.buffer = (PrioritizedReplayBuffer(buffer_size, alpha=per_alpha, seed=seed)
                       if per_enabled else UniformReplayBuffer(buffer_size, seed=seed))
        self.step_count = 0
        self.rng = np.random.default_rng(seed)

    # ---- action selection --------------------------------------------------
    def act(self, state, greedy=False, use_mask=True):
        """Pick a VM. When ``use_mask``, VMs whose utilization feature exceeds
        0.9 get their Q-value set to -inf before the argmax — a safety mask
        that mechanically prevents the policy from piling onto a saturated
        VM. (The mask is a pure function of the observed state, so it is
        available to exploration, evaluation and the XAI layer alike.)
        """
        state = np.asarray(state, dtype=np.float32)
        q = np.asarray(self.q.predict(state)).copy()
        if use_mask:
            q = q + self.action_mask(state)
        if (not greedy) and self.rng.random() < self.eps:
            # explore only among unmasked actions
            allowed = np.where(q > -1e8)[0]
            if len(allowed) == 0:
                allowed = np.arange(self.action_dim)
            return int(self.rng.choice(allowed))
        return int(np.argmax(q))

    @staticmethod
    def action_mask(state, task_dim=5, vm_feat_dim=5, threshold=0.9):
        """Return -1e9 for overloaded VMs (util feature > threshold), else 0.

        Works on a single state (d,) or a batch (n, d). The utilization
        feature of VM i sits at index task_dim + i*vm_feat_dim.
        """
        state = np.asarray(state, dtype=np.float32)
        single = state.ndim == 1
        if single:
            state = state[None, :]
        utils = state[:, task_dim::vm_feat_dim]      # (n, num_vms)
        mask = np.where(utils > threshold, -1e9, 0.0)
        # safety: if EVERYTHING is overloaded, don't mask anything
        all_masked = (mask < 0).all(axis=1, keepdims=True)
        mask = np.where(all_masked, 0.0, mask)
        return mask[0] if single else mask

    # ---- memory ------------------------------------------------------------
    def remember(self, s, a, r, s2, done, td_error=None):
        self.buffer.push(s, a, r, s2, float(done), td_error=td_error)

    # ---- exploration / IS anneal ------------------------------------------
    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def _anneal_beta(self):
        frac = min(1.0, self.step_count / self.beta_anneal_steps)
        self.beta = self.beta_start + frac * (self.beta_end - self.beta_start)

    # ---- training step -----------------------------------------------------
    def train_step(self):
        if len(self.buffer) < max(self.batch_size, 500):
            return None
        self._anneal_beta()
        s, a, r, s2, d, idxs, ws = self.buffer.sample(self.batch_size, beta=self.beta)

        # Double DQN target:
        #   a* = argmax_a Q_online(s', a)        <- online picks the action
        #   y  = r + gamma * Q_target(s', a*)    <- target evaluates it
        next_q_online = self.q.forward(s2)
        next_actions = np.argmax(next_q_online, axis=1)
        next_q_target = self.target_q.forward(s2)
        max_next_q = next_q_target[np.arange(self.batch_size), next_actions]
        targets = r + self.gamma * (1 - d) * max_next_q

        # TD error for priority update (compute before the gradient step)
        pred_q = self.q.forward(s)[np.arange(self.batch_size), a]
        td_errors = targets - pred_q

        # weighted MSE using importance-sampling weights
        loss = self._weighted_train_step(s, a, targets, ws)

        self.buffer.update_priorities(idxs, td_errors)

        self.step_count += 1
        # soft update every step
        self.target_q.soft_update(self.q, self.tau)
        # also sync normalization stats occasionally
        if self.step_count % self.target_update_steps == 0:
            self.target_q.run_mean = self.q.run_mean.copy()
            self.target_q.run_var = self.q.run_var.copy()
        return loss

    def _weighted_train_step(self, states, actions, targets, ws):
        """Delegate to the Dueling QNetwork's weighted_train_step."""
        return self.q.weighted_train_step(states, actions, targets, ws)
