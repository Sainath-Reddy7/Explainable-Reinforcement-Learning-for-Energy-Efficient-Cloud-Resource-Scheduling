"""
fidelity.py — Trust / faithfulness metrics for XAI attributions (v2).

v1 shipped Top-K Fidelity, Deletion-AOPC and Cosine Consistency.  v2:

  * fixes the AOPC normalization (the old code divided by |q0| which flips
    sign when q0 is negative, masking unfaithful explanations);
  * adds Insertion-AOPC (the complement of Deletion — add features
    most-important-first, area under the rising curve);
  * adds Infidelity (Yeh et al. 2019): expected squared change in prediction
    vs. the attribution-predicted change under random perturbations;
  * adds Stability / Local Lipschitz: attribution change rate vs. input
    change rate for nearby states — catches chaotic explainers.

Higher = better for: fidelity, deletion-AOPC, insertion-AOPC, consistency.
Lower = better for: infidelity, stability.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Top-K Fidelity                                                              #
# --------------------------------------------------------------------------- #
def top_k_fidelity(predict_fn, instance, phi, background_mean, action_idx, ks=(3, 5, 10)):
    """Keep only top-k |phi| features, replace rest with background mean.
    A faithful explanation preserves the argmax decision with fewer features."""
    out = {}
    order = np.argsort(-np.abs(phi))
    for k in ks:
        keep = set(order[:k].tolist())
        x = np.array([instance[i] if i in keep else background_mean[i]
                      for i in range(len(instance))], dtype=np.float32)
        q = predict_fn(x[None, :])[0]
        out[f"top{k}_action_match"] = int(np.argmax(q) == action_idx)
    return out


# --------------------------------------------------------------------------- #
# Deletion & Insertion AOPC                                                   #
# --------------------------------------------------------------------------- #
def deletion_curve(predict_fn, instance, phi, background_mean, action_idx, steps=10):
    """Q(action) as top features are removed most-important-first."""
    d = len(instance)
    order = np.argsort(-np.abs(phi))
    x = instance.copy().astype(np.float32)
    q0 = predict_fn(x[None, :])[0][action_idx]
    curve = [q0]
    chunk = max(1, d // steps)
    removed = set()
    for i in range(0, d, chunk):
        removed.update(order[i:i + chunk].tolist())
        x_pert = np.array([background_mean[j] if j in removed else instance[j]
                           for j in range(d)], dtype=np.float32)
        curve.append(predict_fn(x_pert[None, :])[0][action_idx])
    return np.array(curve)


def insertion_curve(predict_fn, instance, phi, background_mean, action_idx, steps=10):
    """Q(action) starting from background, adding top features most-important-first."""
    d = len(instance)
    order = np.argsort(-np.abs(phi))
    x = background_mean.astype(np.float32).copy()
    q0 = predict_fn(x[None, :])[0][action_idx]
    curve = [q0]
    chunk = max(1, d // steps)
    added = set()
    for i in range(0, d, chunk):
        added.update(order[i:i + chunk].tolist())
        x_pert = np.array([instance[j] if j in added else background_mean[j]
                           for j in range(d)], dtype=np.float32)
        curve.append(predict_fn(x_pert[None, :])[0][action_idx])
    return np.array(curve)


def deletion_aopc(predict_fn, instance, phi, background_mean, action_idx, steps=10):
    """Area Over the Perturbation Curve for deletion.

    Normalizes the Q-drop by the *decision margin* (Q-value spread across all
    actions), not the absolute Q-value. This is the right scale: a drop of 0.5
    matters a lot when the best-vs-worst VM gap is 1.0, but is noise when Q ~35.

    Higher = more faithful (removing top features hurts the chosen action faster).
    """
    curve = deletion_curve(predict_fn, instance, phi, background_mean, action_idx, steps)
    q0 = curve[0]
    drops = q0 - curve[1:]
    # normalize by Q-spread across all actions (the meaningful decision margin)
    all_q = predict_fn(instance[None, :])[0]
    q_spread = float(all_q.max() - all_q.min())
    denom = max(q_spread, 1e-3) * len(drops)
    return float(np.clip(np.sum(drops) / denom, -5.0, 5.0))


def insertion_aopc(predict_fn, instance, phi, background_mean, action_idx, steps=10):
    """Area under the insertion curve, normalized by the Q-spread.
    Higher = more faithful (adding top features raises Q fast relative to the
    decision margin)."""
    curve = insertion_curve(predict_fn, instance, phi, background_mean, action_idx, steps)
    gains = curve[1:] - curve[0]
    all_q = predict_fn(instance[None, :])[0]
    q_spread = float(all_q.max() - all_q.min())
    denom = max(q_spread, 1e-3) * len(gains)
    return float(np.clip(np.sum(gains) / denom, -5.0, 5.0))


# --------------------------------------------------------------------------- #
# Infidelity (Yeh et al. 2019)                                                #
# --------------------------------------------------------------------------- #
def infidelity(predict_fn, qnet, instance, phi, action_idx, background_mean,
               n_perturb=20, noise_scale=0.1, seed=0):
    """E[(phi · I) - (Q(x+I) - Q(x))]^2 over random binary perturbations I,
    normalized by the squared Q-spread so the score is scale-free.

    A faithful explanation's predicted change (phi·I) should match the true
    prediction change.  Lower = better.  Normalization by Q-spread makes the
    score comparable across networks with different absolute Q magnitudes."""
    rng = np.random.default_rng(seed)
    d = len(instance)
    instance = np.asarray(instance, dtype=np.float32)
    phi = np.asarray(phi, dtype=np.float32)
    all_q = predict_fn(instance[None, :])[0]
    q_spread = float(all_q.max() - all_q.min())
    errs = []
    for _ in range(n_perturb):
        mask = rng.integers(0, 2, size=d).astype(np.float32)
        I = (background_mean - instance) * mask
        x_pert = instance + I
        q0 = predict_fn(instance[None, :])[0][action_idx]
        q1 = predict_fn(x_pert[None, :])[0][action_idx]
        true_delta = q1 - q0
        explained_delta = float(np.dot(phi, I))
        # normalize the squared error by the squared Q-spread
        errs.append(((explained_delta - true_delta) / max(q_spread, 1e-3)) ** 2)
    return float(np.mean(errs))


# --------------------------------------------------------------------------- #
# Consistency & Stability                                                     #
# --------------------------------------------------------------------------- #
def consistency_score(phis, states, distance_threshold=0.35, max_pairs=60, seed=0):
    """Mean cosine similarity of attribution vectors for near-duplicate states."""
    n = len(states)
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if np.linalg.norm(states[i] - states[j]) < distance_threshold]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    if not pairs:
        return None, 0
    sims = []
    for i, j in pairs:
        a, b = phis[i], phis[j]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na > 1e-8 and nb > 1e-8:
            sims.append(float(np.dot(a, b) / (na * nb)))
    return (float(np.mean(sims)) if sims else None), len(pairs)


def stability_score(phis, states, distance_threshold=0.35, max_pairs=60, seed=0):
    """Local Lipschitz estimate: ||phi_i - phi_j|| / ||s_i - s_j|| for nearby
    states.  Lower = more stable (attributions don't jump on small input changes)."""
    n = len(states)
    rng = np.random.default_rng(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)
             if np.linalg.norm(states[i] - states[j]) < distance_threshold]
    if len(pairs) > max_pairs:
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]
    if not pairs:
        return None, 0
    ratios = []
    for i, j in pairs:
        ds = np.linalg.norm(states[i] - states[j])
        dp = np.linalg.norm(phis[i] - phis[j])
        if ds > 1e-6:
            ratios.append(float(dp / ds))
    return (float(np.mean(ratios)) if ratios else None), len(pairs)
