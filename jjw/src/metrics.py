from __future__ import annotations

import numpy as np

R_HIT_RADIUS = 0.01


def distances(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Return per-row 3D Euclidean distances."""
    pred_arr = np.asarray(pred, dtype=float)
    true_arr = np.asarray(true, dtype=float)
    if pred_arr.shape != true_arr.shape:
        raise ValueError(f"shape mismatch: pred={pred_arr.shape}, true={true_arr.shape}")
    if pred_arr.ndim != 2 or pred_arr.shape[1] != 3:
        raise ValueError(f"expected (n, 3) arrays, got {pred_arr.shape}")
    return np.linalg.norm(pred_arr - true_arr, axis=1)


def r_hit(pred: np.ndarray, true: np.ndarray, radius: float = R_HIT_RADIUS) -> float:
    """DACON R-Hit metric: mean(distance <= radius)."""
    return float(np.mean(distances(pred, true) <= radius))


def score_summary(pred: np.ndarray, true: np.ndarray, radius: float = R_HIT_RADIUS) -> dict[str, float]:
    d = distances(pred, true)
    return {
        "hit": float(np.mean(d <= radius)),
        "mean_distance": float(np.mean(d)),
        "median_distance": float(np.median(d)),
        "p90_distance": float(np.quantile(d, 0.90)),
        "p95_distance": float(np.quantile(d, 0.95)),
        "p99_distance": float(np.quantile(d, 0.99)),
    }
