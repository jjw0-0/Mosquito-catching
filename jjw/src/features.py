from __future__ import annotations

import numpy as np


def _norm(a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a, axis=-1)


def _trajectory_basis(d1: np.ndarray) -> np.ndarray:
    """Return per-sample orthonormal axes aligned to recent motion."""
    n = d1.shape[0]
    forward = d1[:, -3:].mean(axis=1)
    norm = np.linalg.norm(forward, axis=1, keepdims=True)
    fallback = np.tile(np.array([[1.0, 0.0, 0.0]]), (n, 1))
    e1 = np.where(norm > 1e-9, forward / np.maximum(norm, 1e-12), fallback)

    global_up = np.tile(np.array([[0.0, 0.0, 1.0]]), (n, 1))
    e2 = np.cross(global_up, e1)
    e2_norm = np.linalg.norm(e2, axis=1, keepdims=True)
    alt = np.cross(np.tile(np.array([[0.0, 1.0, 0.0]]), (n, 1)), e1)
    e2 = np.where(e2_norm > 1e-9, e2 / np.maximum(e2_norm, 1e-12), alt)
    e2 = e2 / np.maximum(np.linalg.norm(e2, axis=1, keepdims=True), 1e-12)
    e3 = np.cross(e1, e2)
    return np.stack([e1, e2, e3], axis=1)


def _project_local(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    # vectors: (n, t, 3), basis: (n, 3 axes, 3 xyz) -> (n, t, 3 local axes)
    return np.einsum("ntc,nkc->ntk", vectors, basis)


def build_features(xyz: np.ndarray) -> np.ndarray:
    """Build trajectory features from an array shaped (n, 11, 3)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != (11, 3):
        raise ValueError(f"expected xyz shape (n, 11, 3), got {xyz.shape}")

    n = xyz.shape[0]
    d1 = np.diff(xyz, axis=1)      # 10 displacements over 40ms
    d2 = np.diff(d1, axis=1)       # 9 acceleration-like terms
    d3 = np.diff(d2, axis=1)       # 8 jerk-like terms
    last = xyz[:, -1]
    basis = _trajectory_basis(d1)

    speed = _norm(d1)
    accel_norm = _norm(d2)
    jerk_norm = _norm(d3)

    centered = xyz - last[:, None, :]
    local_centered = _project_local(centered, basis)
    local_d1 = _project_local(d1, basis)
    local_d2 = _project_local(d2, basis)
    local_d3 = _project_local(d3, basis)
    recent_windows = []
    for k in (1, 2, 3, 5, 10):
        recent_windows.append(d1[:, -k:].mean(axis=1))
        recent_windows.append(d1[:, -k:].std(axis=1))
    for k in (2, 3, 5, 9):
        recent_windows.append(d2[:, -k:].mean(axis=1))
        recent_windows.append(d2[:, -k:].std(axis=1))

    # Direction-change features: cos(angle) between consecutive displacements.
    v0 = d1[:, :-1]
    v1 = d1[:, 1:]
    denom = (_norm(v0) * _norm(v1) + 1e-12)
    cos_turn = np.sum(v0 * v1, axis=2) / denom

    # Least-squares slopes for short and long windows.
    trend_feats = []
    step_index = np.arange(11, dtype=np.float64)
    for m in (3, 4, 5, 7, 11):
        t = step_index[-m:]
        tc = t - t.mean()
        denom_t = float(np.sum(tc * tc))
        window = xyz[:, -m:, :]
        slope = np.sum((window - window.mean(axis=1, keepdims=True)) * tc[None, :, None], axis=1) / denom_t
        trend_feats.append(slope)

    feature_blocks = [
        xyz.reshape(n, -1),
        centered.reshape(n, -1),
        local_centered.reshape(n, -1),
        d1.reshape(n, -1),
        d2.reshape(n, -1),
        d3.reshape(n, -1),
        local_d1.reshape(n, -1),
        local_d2.reshape(n, -1),
        local_d3.reshape(n, -1),
        basis.reshape(n, -1),
        speed,
        accel_norm,
        jerk_norm,
        cos_turn,
        last,
        xyz[:, -1] - xyz[:, 0],
        xyz[:, -1] - xyz[:, -3],
        xyz[:, -1] - xyz[:, -6],
        np.concatenate(recent_windows, axis=1),
        np.concatenate(trend_feats, axis=1),
        _norm(last)[:, None],
        speed[:, -1:] - speed[:, -2:-1],
    ]
    features = np.concatenate(feature_blocks, axis=1)
    if not np.isfinite(features).all():
        raise ValueError("features contain non-finite values")
    return features


def physical_predictions(xyz: np.ndarray) -> dict[str, np.ndarray]:
    """Return deterministic physics-style +80ms candidate predictions."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != (11, 3):
        raise ValueError(f"expected xyz shape (n, 11, 3), got {xyz.shape}")

    d1 = np.diff(xyz, axis=1)
    last = xyz[:, -1]
    preds: dict[str, np.ndarray] = {"last": last.copy()}

    for k in range(1, 11):
        v = (xyz[:, -1] - xyz[:, -1 - k]) / k
        preds[f"cv_endpoint_k{k}"] = last + 2.0 * v
        preds[f"cv_avgdiff_k{k}"] = last + 2.0 * d1[:, -k:].mean(axis=1)

    # Discrete constant-acceleration variants. c=0.5 was strongest in initial train check.
    accel = d1[:, -1] - d1[:, -2]
    for c in (-1.0, 0.0, 0.25, 0.40, 0.46, 0.50, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64, 0.75, 1.0, 1.5, 2.0, 3.0):
        preds[f"accel_c{c:g}"] = last + 2.0 * d1[:, -1] + c * accel

    # Linear and quadratic extrapolations to step index 12 (0ms is index 10, +80ms is index 12).
    idx = np.arange(11, dtype=np.float64)
    for m in range(3, 12):
        t = idx[-m:]
        window = xyz[:, -m:, :]
        tc = t - t.mean()
        slope = np.sum((window - window.mean(axis=1, keepdims=True)) * tc[None, :, None], axis=1) / np.sum(tc * tc)
        intercept = window.mean(axis=1) - slope * t.mean()
        preds[f"linear_m{m}"] = slope * 12.0 + intercept

        vand = np.vstack([t * t, t, np.ones(m)]).T
        pinv = np.linalg.pinv(vand)
        coef = np.einsum("km,nmc->nkc", pinv, window)
        preds[f"quad_m{m}"] = coef[:, 0, :] * 144.0 + coef[:, 1, :] * 12.0 + coef[:, 2, :]

    return preds


def best_physical_name(preds: dict[str, np.ndarray], y: np.ndarray, metric) -> str:
    return max(preds, key=lambda name: metric(preds[name], y))
