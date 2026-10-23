"""
qnetwork.py — Pure-NumPy Dueling DQN Q-network (v3).

Architecture
------------
Shared backbone: input → H1(128) → H2(64)
  ├── Value stream:     H2 → V(32) → V(1)     "how good is this VM pool state?"
  └── Advantage stream: H2 → A(32) → A(8)     "how much better is VM i vs average?"
Final Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]

Why Dueling?  When many VMs are similarly good/bad, the advantage stream
converges near zero and the VALUE stream still learns correctly.  A vanilla
DQN wastes capacity estimating redundant per-action values; Dueling separates
the two concerns so the network learns faster and generalizes better.

Also includes:
  * Running input normalization (batch-norm-style)
  * Dropout (training only)
  * N-step-return compatible (caller provides the target)
  * Exact analytic gradient for XAI methods
"""

from __future__ import annotations

import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


class QNetwork:
    """Dueling DQN: shared backbone + value stream + advantage stream."""

    def __init__(self, in_dim, out_dim, hidden=(128, 64),
                 stream_hidden=32, lr=5e-4, dropout=0.1, seed=0):
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.lr = lr
        self.base_lr = lr
        self.dropout = dropout

        # ---- shared backbone: in_dim -> h1 -> h2 ----
        h1, h2 = hidden
        self.W1 = rng.normal(0, np.sqrt(2 / in_dim), size=(in_dim, h1)).astype(np.float32)
        self.b1 = np.zeros(h1, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2 / h1), size=(h1, h2)).astype(np.float32)
        self.b2 = np.zeros(h2, dtype=np.float32)
        bb_dim = h2  # backbone output dimension

        # ---- value stream: bb_dim -> sv -> 1 ----
        sh = stream_hidden
        self.Wv1 = rng.normal(0, np.sqrt(2 / bb_dim), size=(bb_dim, sh)).astype(np.float32)
        self.bv1 = np.zeros(sh, dtype=np.float32)
        self.Wv2 = rng.normal(0, np.sqrt(2 / sh), size=(sh, 1)).astype(np.float32)
        self.bv2 = np.zeros(1, dtype=np.float32)

        # ---- advantage stream: bb_dim -> sa -> out_dim ----
        self.Wa1 = rng.normal(0, np.sqrt(2 / bb_dim), size=(bb_dim, sh)).astype(np.float32)
        self.ba1 = np.zeros(sh, dtype=np.float32)
        self.Wa2 = rng.normal(0, np.sqrt(2 / sh), size=(sh, out_dim)).astype(np.float32)
        self.ba2 = np.zeros(out_dim, dtype=np.float32)

        self.params = ["W1","b1","W2","b2",
                        "Wv1","bv1","Wv2","bv2",
                        "Wa1","ba1","Wa2","ba2"]
        self._init_adam()

        # running normalization stats
        self.run_mean = np.zeros(in_dim, dtype=np.float32)
        self.run_var = np.ones(in_dim, dtype=np.float32)
        self.run_count = 1e-4

    def _init_adam(self):
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def set_lr(self, lr):
        self.lr = lr

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

        # shared backbone
        z1 = x_norm @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)   # backbone output (batch, h2)

        if train and self.dropout > 0:
            rng = np.random.default_rng(self.t if self.t > 0 else 0)
            mask = (rng.random(a2.shape) > self.dropout).astype(np.float32) / (1 - self.dropout)
            a2 = a2 * mask

        # value stream
        zv1 = a2 @ self.Wv1 + self.bv1
        av1 = relu(zv1)
        v = av1 @ self.Wv2 + self.bv2       # (batch, 1)

        # advantage stream
        za1 = a2 @ self.Wa1 + self.ba1
        aa1 = relu(za1)
        adv = aa1 @ self.Wa2 + self.ba2      # (batch, out_dim)

        # dueling combine: Q = V + (A - mean(A))
        adv_mean = adv.mean(axis=1, keepdims=True)
        out = v + adv - adv_mean              # (batch, out_dim)

        if cache:
            cache_dict = {
                "x_norm": x_norm, "z1": z1, "a1": a1, "z2": z2, "a2": a2,
                "zv1": zv1, "av1": av1, "v": v,
                "za1": za1, "aa1": aa1, "adv": adv, "adv_mean": adv_mean,
                "out": out,
            }
            return out, cache_dict
        return out[0] if single else out

    def predict(self, x):
        return self.forward(x, cache=False, train=False)

    # ---- backward + update --------------------------------------------------
    def train_step(self, states, actions, targets, grad_clip=10.0):
        batch = states.shape[0]
        out, c = self.forward(states, cache=True, train=True)

        pred_q = out[np.arange(batch), actions]
        td_error = pred_q - targets

        # gradient of MSE loss w.r.t. output
        dout = np.zeros_like(out)
        dout[np.arange(batch), actions] = 2.0 * td_error / batch

        # ---- backprop through dueling combine ----
        # out = V + adv - adv_mean
        # dQ/dV = 1,  dQ/dadv_i = (1 - 1/n) for chosen action, -1/n for others
        # but dout already encodes which action, so:
        #   dV = sum_a dout[:,a]  (since V adds to all)
        #   dadv = dout - sum_a dout[:,a] / n   (because of the -mean term)
        dV = dout.sum(axis=1, keepdims=True)          # (batch, 1)
        dAdv = dout - dout.sum(axis=1, keepdims=True) / self.out_dim  # (batch, out_dim)

        # value stream backprop: V = av1 @ Wv2 + bv2
        dWv2 = c["av1"].T @ dV
        dbv2 = dV.sum(axis=0)
        dav1 = dV @ self.Wv2.T
        dzv1 = dav1 * relu_grad(c["zv1"])
        dWv1 = c["a2"].T @ dzv1
        dbv1 = dzv1.sum(axis=0)
        da2_from_v = dzv1 @ self.Wv1.T    # (batch, h2)

        # advantage stream backprop
        dWa2 = c["aa1"].T @ dAdv
        dba2 = dAdv.sum(axis=0)
        daa1 = dAdv @ self.Wa2.T
        dza1 = daa1 * relu_grad(c["za1"])
        dWa1 = c["a2"].T @ dza1
        dba1 = dza1.sum(axis=0)
        da2_from_a = dza1 @ self.Wa1.T    # (batch, h2)

        # combine gradients into shared backbone
        da2 = da2_from_v + da2_from_a
        dz2 = da2 * relu_grad(c["z2"])
        dW2 = c["a1"].T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(c["z1"])
        dW1 = c["x_norm"].T @ dz1
        db1 = dz1.sum(axis=0)

        grads = {
            "W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
            "Wv1": dWv1, "bv1": dbv1, "Wv2": dWv2, "bv2": dbv2,
            "Wa1": dWa1, "ba1": dba1, "Wa2": dWa2, "ba2": dba2,
        }
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

    # ---- weighted train step (for PER) --------------------------------------
    def weighted_train_step(self, states, actions, targets, ws, grad_clip=10.0):
        """Same as train_step but with per-sample importance weights (for PER)."""
        batch = states.shape[0]
        out, c = self.forward(states, cache=True, train=True)

        pred_q = out[np.arange(batch), actions]
        td_error = pred_q - targets

        dout = np.zeros_like(out)
        dout[np.arange(batch), actions] = 2.0 * (ws * td_error) / batch

        dV = dout.sum(axis=1, keepdims=True)
        dAdv = dout - dout.sum(axis=1, keepdims=True) / self.out_dim

        dWv2 = c["av1"].T @ dV; dbv2 = dV.sum(axis=0)
        dav1 = dV @ self.Wv2.T; dzv1 = dav1 * relu_grad(c["zv1"])
        dWv1 = c["a2"].T @ dzv1; dbv1 = dzv1.sum(axis=0)
        da2_from_v = dzv1 @ self.Wv1.T

        dWa2 = c["aa1"].T @ dAdv; dba2 = dAdv.sum(axis=0)
        daa1 = dAdv @ self.Wa2.T; dza1 = daa1 * relu_grad(c["za1"])
        dWa1 = c["a2"].T @ dza1; dba1 = dza1.sum(axis=0)
        da2_from_a = dza1 @ self.Wa1.T

        da2 = da2_from_v + da2_from_a
        dz2 = da2 * relu_grad(c["z2"])
        dW2 = c["a1"].T @ dz2; db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * relu_grad(c["z1"])
        dW1 = c["x_norm"].T @ dz1; db1 = dz1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                 "Wv1": dWv1, "bv1": dbv1, "Wv2": dWv2, "bv2": dbv2,
                 "Wa1": dWa1, "ba1": dba1, "Wa2": dWa2, "ba2": dba2}
        self._adam_update(grads, grad_clip)
        return float(np.mean(ws * td_error ** 2))

    # ---- weight sync --------------------------------------------------------
    def get_weights(self):
        return {p: getattr(self, p).copy() for p in self.params}

    def set_weights(self, weights):
        for p in self.params:
            setattr(self, p, weights[p].copy())

    def soft_update(self, other, tau):
        for p in self.params:
            cur = getattr(self, p)
            new = tau * getattr(other, p) + (1 - tau) * cur
            setattr(self, p, new.astype(np.float32))

    def input_gradient(self, instance, action_idx):
        """Exact dQ(s,a)/ds at an instance — for Grad×Input and Infidelity."""
        x = np.asarray(instance, dtype=np.float32)[None, :]
        out, c = self.forward(x, cache=True, train=False)
        d_out = np.zeros_like(out)
        d_out[0, action_idx] = 1.0

        # backprop through dueling combine
        dV = d_out.sum(axis=1, keepdims=True)
        dAdv = d_out - d_out.sum(axis=1, keepdims=True) / self.out_dim

        dav1 = dV @ self.Wv2.T; dzv1 = dav1 * relu_grad(c["zv1"])
        da2_v = dzv1 @ self.Wv1.T

        daa1 = dAdv @ self.Wa2.T; dza1 = daa1 * relu_grad(c["za1"])
        da2_a = dza1 @ self.Wa1.T

        da2 = da2_v + da2_a
        dz2 = da2 * relu_grad(c["z2"])
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(c["z1"])
        grad_norm = dz1 @ self.W1.T

        std = np.sqrt(self.run_var + 1e-8)
        return (grad_norm / std)[0]


class VanillaQNetwork:
    """Plain 3-layer MLP (ablation control for the Dueling architecture).
    Same public interface as QNetwork so agent / XAI / trust metrics run
    unmodified when the Dueling streams are disabled."""

    def __init__(self, in_dim, out_dim, hidden=(128, 64), lr=5e-4, dropout=0.1, seed=0):
        rng = np.random.default_rng(seed)
        h1, h2 = hidden
        dims = [in_dim, h1, h2, out_dim]
        self.params = []
        for i in range(len(dims) - 1):
            setattr(self, f"W{i+1}", rng.normal(0, np.sqrt(2/dims[i]), size=(dims[i], dims[i+1])).astype(np.float32))
            setattr(self, f"b{i+1}", np.zeros(dims[i+1], dtype=np.float32))
            self.params += [f"W{i+1}", f"b{i+1}"]
        self.n_layers = len(dims) - 1
        self.lr = lr
        self.dropout = dropout
        self._adam_init()
        self.run_mean = np.zeros(in_dim, dtype=np.float32)
        self.run_var = np.ones(in_dim, dtype=np.float32)
        self.run_count = 1e-4

    def _adam_init(self):
        self.m = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.v = {p: np.zeros_like(getattr(self, p)) for p in self.params}
        self.t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def set_lr(self, lr):
        self.lr = lr

    def _normalize(self, x, update_stats=False):
        if update_stats:
            n = x.shape[0]
            delta = x.mean(0) - self.run_mean
            tot = self.run_count + n
            self.run_mean += delta * (n / tot)
            self.run_var += (x.var(0) - self.run_var) * (n / tot)
            self.run_count = tot
        return (x - self.run_mean) / np.sqrt(self.run_var + 1e-8)

    def forward(self, x, cache=False, train=False):
        x = np.asarray(x, np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        xn = self._normalize(x, update_stats=train)
        acts, zs, a = [xn], [], xn
        for i in range(1, self.n_layers + 1):
            z = a @ getattr(self, f"W{i}") + getattr(self, f"b{i}")
            zs.append(z)
            a = relu(z) if i < self.n_layers else z
            if i < self.n_layers and train and self.dropout > 0:
                m = (np.random.default_rng(self.t or 0).random(a.shape) > self.dropout).astype(np.float32) / (1 - self.dropout)
                a = a * m
            acts.append(a)
        if cache:
            return acts[-1], {"acts": acts, "zs": zs}
        return acts[-1][0] if single else acts[-1]

    def predict(self, x):
        return self.forward(x)

    def _backward(self, states, actions, targets, ws, grad_clip=10.0):
        batch = states.shape[0]
        out, c = self.forward(states, cache=True, train=True)
        td = out[np.arange(batch), actions] - targets
        dout = np.zeros_like(out)
        dout[np.arange(batch), actions] = 2.0 * (ws * td) / batch
        grads, da = {}, dout
        for i in range(self.n_layers, 0, -1):
            W = getattr(self, f"W{i}")
            grads[f"W{i}"] = c["acts"][i-1].T @ da
            grads[f"b{i}"] = da.sum(0)
            if i > 1:
                da = (da @ W.T) * relu_grad(c["zs"][i-2])
        self._adam_update(grads, grad_clip)
        return float(np.mean(ws * td ** 2))

    def train_step(self, s, a, t, grad_clip=10.0):
        return self._backward(s, a, t, np.ones(len(a), np.float32), grad_clip)

    def weighted_train_step(self, s, a, t, ws, grad_clip=10.0):
        return self._backward(s, a, t, ws, grad_clip)

    def _adam_update(self, grads, grad_clip):
        self.t += 1
        for p in self.params:
            g = np.clip(grads[p], -grad_clip, grad_clip)
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * g
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * g ** 2
            setattr(self, p, getattr(self, p) - self.lr * (self.m[p] / (1 - self.beta1 ** self.t)) / (np.sqrt(self.v[p] / (1 - self.beta2 ** self.t)) + self.eps))

    def get_weights(self):
        return {p: getattr(self, p).copy() for p in self.params}

    def set_weights(self, w):
        for p in self.params:
            setattr(self, p, w[p].copy())

    def soft_update(self, other, tau):
        for p in self.params:
            setattr(self, p, (tau * getattr(other, p) + (1 - tau) * getattr(self, p)).astype(np.float32))

    def input_gradient(self, instance, action_idx):
        x = np.asarray(instance, np.float32)[None, :]
        out, c = self.forward(x, cache=True, train=False)
        d = np.zeros_like(out)
        d[0, action_idx] = 1.0
        for i in range(self.n_layers, 0, -1):
            W = getattr(self, f"W{i}")
            if i > 1:
                d = (d @ W.T) * relu_grad(c["zs"][i-2])
        return (d @ self.W1.T / np.sqrt(self.run_var + 1e-8))[0]
