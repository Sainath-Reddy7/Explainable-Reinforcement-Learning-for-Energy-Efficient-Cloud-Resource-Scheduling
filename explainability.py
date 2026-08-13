"""
explainability.py — Four post-hoc attribution methods for the DQN Q-network (v2).

v1 had KernelSHAP + Gradient×Input.  v2 adds two more so the trust study can
rank four methods on the same decisions:

  1. KernelSHAP         — model-agnostic, coalition sampling + WLS (Lundberg & Lee 2017)
  2. Gradient×Input     — exact analytic gradient × input (fast baseline)
  3. Occlusion          — slide a window, zero/mean each feature block, measure ΔQ
  4. Integrated Gradients — Riemann-sum the gradient along baseline->instance path
                          (Sundararajan et al. 2017).  Satisfies the axioms that
                          plain Grad×Input violates (sensitivity, implementation
                          invariance).

All methods return a per-feature attribution vector phi (same length as the
state vector).  rollup_vm_contributions() aggregates the raw phi into a
human-readable per-VM score for the dashboard.
"""

from __future__ import annotations

import time
from math import comb

import numpy as np


# --------------------------------------------------------------------------- #
# 1. KernelSHAP                                                               #
# --------------------------------------------------------------------------- #
def shapley_kernel_weight(d, k):
    if k == 0 or k == d:
        return 10_000.0
    return (d - 1) / (comb(d, k) * k * (d - k))


class KernelSHAPExplainer:
    def __init__(self, predict_fn, background, n_coalitions=250, n_bg_draws=8, seed=0):
        self.predict_fn = predict_fn
        self.background = np.asarray(background, dtype=np.float32)
        self.n_coalitions = n_coalitions
        self.n_bg_draws = n_bg_draws
        self.rng = np.random.default_rng(seed)

    def _coalition_value(self, instance, mask, action_idx):
        idx = self.rng.choice(len(self.background), size=self.n_bg_draws, replace=True)
        bg = self.background[idx]
        x_mix = np.where(mask[None, :].astype(bool), instance[None, :], bg)
        q = self.predict_fn(x_mix)
        return float(q[:, action_idx].mean())

    def explain(self, instance, action_idx):
        instance = np.asarray(instance, dtype=np.float32)
        d = len(instance)
        f_full = self._coalition_value(instance, np.ones(d), action_idx)
        f_empty = self._coalition_value(instance, np.zeros(d), action_idx)

        sizes = np.arange(1, d)
        size_weights = np.array([shapley_kernel_weight(d, k) for k in sizes])
        size_probs = size_weights / size_weights.sum()

        masks, weights, values = [], [], []
        for _ in range(self.n_coalitions):
            k = int(self.rng.choice(sizes, p=size_probs))
            chosen = self.rng.choice(d, size=k, replace=False)
            mask = np.zeros(d)
            mask[chosen] = 1
            masks.append(mask)
            weights.append(shapley_kernel_weight(d, k))
            values.append(self._coalition_value(instance, mask, action_idx))

        Z = np.array(masks)
        W = np.array(weights)
        y = np.array(values) - f_empty

        WZ = Z * W[:, None]
        A = Z.T @ WZ
        b = Z.T @ (W * y)
        A_inv = np.linalg.pinv(A + 1e-6 * np.eye(d))
        phi_unc = A_inv @ b
        total = f_full - f_empty
        ones = np.ones(d)
        correction = (ones @ A_inv @ b - total) / (ones @ A_inv @ ones + 1e-12)
        phi = phi_unc - A_inv @ ones * correction
        return phi


# --------------------------------------------------------------------------- #
# 2. Gradient × Input                                                         #
# --------------------------------------------------------------------------- #
def gradient_x_input_explainer(qnet, instance, action_idx):
    """dQ(s,a)/ds · s — exact analytic, ~0.1 ms."""
    grad = qnet.input_gradient(instance, action_idx)
    return grad * np.asarray(instance, dtype=np.float32)


# --------------------------------------------------------------------------- #
# 3. Occlusion                                                                #
# --------------------------------------------------------------------------- #
def occlusion_explainer(predict_fn, instance, action_idx, background_mean, window=3):
    """Slide a window over features; attribution = drop in Q when occluded."""
    instance = np.asarray(instance, dtype=np.float32)
    d = len(instance)
    base_q = predict_fn(instance[None, :])[0][action_idx]
    phi = np.zeros(d, dtype=np.float32)
    for i in range(d):
        x = instance.copy()
        lo = max(0, i - window // 2)
        hi = min(d, lo + window)
        x[lo:hi] = background_mean[lo:hi]
        q = predict_fn(x[None, :])[0][action_idx]
        # contribution = how much Q drops when this feature is removed
        phi[i] = (base_q - q) / max(hi - lo, 1)
    return phi


# --------------------------------------------------------------------------- #
# 4. Integrated Gradients                                                     #
# --------------------------------------------------------------------------- #
def integrated_gradients_explainer(qnet, instance, action_idx, baseline=None, steps=50):
    """IG = (x - x') ⊙ ∫_{α=0}^1 ∇Q(x' + α(x-x')) dα  ≈ Riemann sum."""
    instance = np.asarray(instance, dtype=np.float32)
    if baseline is None:
        baseline = np.zeros_like(instance)
    diff = instance - baseline
    total = np.zeros_like(instance)
    for step in range(steps):
        alpha = (step + 0.5) / steps
        interp = baseline + alpha * diff
        total += qnet.input_gradient(interp, action_idx)
    avg_grad = total / steps
    return diff * avg_grad


# --------------------------------------------------------------------------- #
# Rollup for the dashboard                                                    #
# --------------------------------------------------------------------------- #
def rollup_vm_contributions(phi, feature_names, num_vms, task_dim=5, vm_feat_dim=5):
    """Aggregate raw phi into per-VM + task-level contribution scores."""
    contribs = {"task_features": float(np.sum(phi[:task_dim]))}
    for i in range(num_vms):
        start = task_dim + i * vm_feat_dim
        contribs[f"vm_{i}"] = float(np.sum(phi[start:start + vm_feat_dim]))
    return contribs


def time_explainer(fn, *args, n_reps=5):
    t0 = time.time()
    for _ in range(n_reps):
        fn(*args)
    return (time.time() - t0) / n_reps
