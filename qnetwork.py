"""
qnetwork.py — Pure-NumPy MLP Q-function approximator (v2).

Upgrades over v1
----------------
* 3 hidden layers (default 128-64-32) instead of 2 (64-64) — more capacity
  for the richer Borg-backed state.
* Dropout on hidden activations (training only) — regularizes the small-data
  regime and improves explanation stability.
* Running mean/std input normalization (batch-norm-style) — keeps the input
  distribution stable across episodes with different Borg task mixes, which
  materially helps gradient quality and therefore Q-value argmax stability.
* Learning-rate schedule hook consumed by the DQN agent.
* Exact analytic gradients retained (needed for the gradient-based XAI
  methods and for the Infidelity trust metric).
"""

from __future__ import annotations

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


class QNetwork:
    """Configurable-depth MLP: input -> H1 -> H2 -> H3 -> linear Q-values."""

    def __init__(self, in_dim, out_dim, hidden=(128, 64, 32),
                 lr=5e-4, dropout=0.1, seed=0):
        rng = np.random.default_rng(seed)
        dims = [in_dim, *hidden, out_dim]
        self.params = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            setattr(self, f"W{i+1}",
                    rng.normal(0, np.sqrt(2 / fan_in), size=(dims[i], dims[i+1])).astype(np.float32))
            setattr(self, f"b{i+1}", np.zeros(dims[i+1], dtype=np.float32))
            self.params += [f"W{i+1}", f"b{i+1}"]
        self.n_layers = len(dims) - 1
        self.lr = lr
        self.base_lr = lr
        self.dropout = dropout
        self._init_adam()

        # running normalization stats (Welford-lite, scalar-free)
        self.run_mean = np.zeros(in_dim, dtype=np.float32)
        self.run_var = np.ones(in_dim, dtype=np.float32)
        self.run_count = 1e-4

    # ---- Adam ---------------------------------------------------------------
    def _init_adam(self):
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def set_lr(self, lr):
        self.lr = lr

    # ---- normalization ------------------------------------------------------
    def _normalize(self, x, update_stats=False):
        if update_stats:
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            n = x.shape[0]
            delta = batch_mean - self.run_mean
            tot = self.run_count + n
            self.run_mean += delta * (n / tot)
            self.run_var += (batch_var - self.run_var) * (n / tot)
            self.run_count = tot
        return (x - self.run_mean) / np.sqrt(self.run_var + 1e-8)

    # ---- forward ------------------------------------------------------------
    def forward(self, x, cache=False, train=False):
        x = np.asarray(x, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        x_norm = self._normalize(x, update_stats=train)
        acts = [x_norm]
        zs = []
        a = x_norm
        rng = np.random.default_rng(0)  # deterministic given seed; agent overrides if needed
        for i in range(1, self.n_layers + 1):
            W = getattr(self, f"W{i}")
            b = getattr(self, f"b{i}")
            z = a @ W + b
            zs.append(z)
            if i < self.n_layers:
                a = relu(z)
                if train and self.dropout > 0:
                    mask = (rng.random(a.shape) > self.dropout).astype(np.float32) / (1 - self.dropout)
                    a = a * mask
                acts.append(a)
            else:
                acts.append(z)  # linear output
        out = acts[-1]
        if cache:
            return out, (acts, zs, x_norm)
        return out[0] if single else out

    def predict(self, x):
        """Single or batch forward, inference mode (no dropout, no stat update)."""
        out = self.forward(x, cache=False, train=False)
        return out

    # ---- backward + update --------------------------------------------------
    def train_step(self, states, actions, targets, grad_clip=10.0):
        batch = states.shape[0]
        out, (acts, zs, x_norm) = self.forward(states, cache=True, train=True)

        pred_q = out[np.arange(batch), actions]
        td_error = pred_q - targets

        dout = np.zeros_like(out)
        dout[np.arange(batch), actions] = 2.0 * td_error / batch

        grads = {}
        # backprop through each layer
        da = dout
        for i in range(self.n_layers, 0, -1):
            W = getattr(self, f"W{i}")
            a_prev = acts[i - 1]
            grads[f"W{i}"] = a_prev.T @ da
            grads[f"b{i}"] = da.sum(axis=0)
            if i > 1:
                da = (da @ W.T) * relu_grad(zs[i - 2])
        # gradient w.r.t. normalized input (used by XAI)
        grads["dx_norm"] = da @ getattr(self, "W1").T if "W1" in grads else None
        self._adam_update(grads, grad_clip)
        return float(np.mean(td_error ** 2))

    def _adam_update(self, grads, grad_clip):
        self.t += 1
        for p in self.params:
            g = np.clip(grads[p], -grad_clip, grad_clip)
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * g
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            setattr(self, p, getattr(self, p) - update)

    # ---- weight sync --------------------------------------------------------
    def get_weights(self):
        return {p: getattr(self, p).copy() for p in self.params}

    def set_weights(self, weights):
        for p in self.params:
            setattr(self, p, weights[p].copy())

    def soft_update(self, other, tau):
        """Polyak averaging: theta <- tau*theta_other + (1-tau)*theta_self."""
        for p in self.params:
            cur = getattr(self, p)
            new = tau * getattr(other, p) + (1 - tau) * cur
            setattr(self, p, new.astype(np.float32))

    def input_gradient(self, instance, action_idx):
        """Exact dQ(s,a)/ds at an instance — used by Grad×Input and Infidelity."""
        x = np.asarray(instance, dtype=np.float32)[None, :]
        out, (acts, zs, x_norm) = self.forward(x, cache=True, train=False)
        d_out = np.zeros_like(out)
        d_out[0, action_idx] = 1.0
        da = d_out
        for i in range(self.n_layers, 0, -1):
            W = getattr(self, f"W{i}")
            if i > 1:
                da = (da @ W.T) * relu_grad(zs[i - 2])
        grad_norm = da @ getattr(self, "W1").T
        # chain through normalization: d/dx = (1/std) * d/dx_norm
        std = np.sqrt(self.run_var + 1e-8)
        return (grad_norm / std)[0]
