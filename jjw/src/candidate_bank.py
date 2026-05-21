from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .features import physical_predictions, _trajectory_basis


TARGET_STEP_INDEX = 12.0  # -400..0ms are indices 0..10; +80ms is index 12.


def _norm(a: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a, axis=-1)


def _poly_extrapolate(xyz: np.ndarray, window: int, degree: int, target_index: float = TARGET_STEP_INDEX) -> np.ndarray:
    idx = np.arange(xyz.shape[1], dtype=np.float64)
    t = idx[-window:]
    vand = np.vander(t, N=degree + 1, increasing=False)
    pinv = np.linalg.pinv(vand)
    coef = np.einsum("km,nmc->nkc", pinv, xyz[:, -window:, :])
    target_v = np.array([target_index ** p for p in range(degree, -1, -1)], dtype=np.float64)
    return np.einsum("d,ndc->nc", target_v, coef)


def _ridge_poly_extrapolate(
    xyz: np.ndarray,
    window: int,
    degree: int,
    alpha: float,
    target_index: float = TARGET_STEP_INDEX,
) -> np.ndarray:
    idx = np.arange(xyz.shape[1], dtype=np.float64)
    t = idx[-window:]
    t0 = t - t.mean()
    target = target_index - t.mean()
    vand = np.vander(t0, N=degree + 1, increasing=False)
    penalty = np.eye(degree + 1)
    penalty[-1, -1] = 0.0
    mat = np.linalg.solve(vand.T @ vand + alpha * penalty, vand.T)
    coef = np.einsum("dm,nmc->ndc", mat, xyz[:, -window:, :])
    target_v = np.array([target ** p for p in range(degree, -1, -1)], dtype=np.float64)
    return np.einsum("d,ndc->nc", target_v, coef)


def _local_poly_extrapolate(xyz: np.ndarray, window: int, degree: int) -> np.ndarray:
    d1 = np.diff(xyz, axis=1)
    basis = _trajectory_basis(d1)
    last = xyz[:, -1]
    local = np.einsum("ntc,nkc->ntk", xyz - last[:, None, :], basis)
    local_pred = _poly_extrapolate(local, window=window, degree=degree)
    return last + np.einsum("nk,nkc->nc", local_pred, basis)


def _rodrigues_rotate(v: np.ndarray, axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    axis_norm = np.linalg.norm(axis, axis=1, keepdims=True)
    fallback = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(v), 1))
    k = np.where(axis_norm > 1e-9, axis / np.maximum(axis_norm, 1e-12), fallback)
    ct = np.cos(theta)[:, None]
    st = np.sin(theta)[:, None]
    return v * ct + np.cross(k, v) * st + k * np.sum(k * v, axis=1, keepdims=True) * (1.0 - ct)


def _turn_candidates(xyz: np.ndarray) -> dict[str, np.ndarray]:
    d1 = np.diff(xyz, axis=1)
    last = xyz[:, -1]
    v_prev = d1[:, -2]
    v_last = d1[:, -1]
    cross = np.cross(v_prev, v_last)
    denom = np.maximum(_norm(v_prev) * _norm(v_last), 1e-12)
    sin = _norm(cross) / denom
    cos = np.sum(v_prev * v_last, axis=1) / denom
    theta = np.arctan2(sin, np.clip(cos, -1.0, 1.0))
    out: dict[str, np.ndarray] = {}
    for gain in (0.25, 0.5, 0.75, 1.0, 1.25):
        v1 = _rodrigues_rotate(v_last, cross, theta * gain)
        v2 = _rodrigues_rotate(v1, cross, theta * gain)
        out[f"turn_rodrigues_g{gain:g}"] = last + v1 + v2
    return out


def _average_rotation_vector(d1: np.ndarray, pairs: int) -> np.ndarray:
    """Recent average velocity rotation vector over `pairs` transitions.

    The vector direction is the instantaneous turn axis and the magnitude is
    the per-40ms turn angle.  Averaging the recent rotation vectors is a compact
    constant-turn-rate estimate that is more stable than using only the last two
    displacements on noisy trajectories.
    """
    v0 = d1[:, -(pairs + 1) : -1]
    v1 = d1[:, -pairs:]
    cross = np.cross(v0, v1)
    cross_norm = np.linalg.norm(cross, axis=2)
    dot = np.sum(v0 * v1, axis=2)
    theta = np.arctan2(cross_norm, dot)
    axis = cross / np.maximum(cross_norm[..., None], 1e-12)
    rotvec = np.where(cross_norm[..., None] > 1e-12, axis * theta[..., None], 0.0)
    weights = np.arange(1, pairs + 1, dtype=np.float64)
    weights = weights / weights.sum()
    return np.einsum("p,npd->nd", weights, rotvec)


def _rotate_by_rotation_vector(v: np.ndarray, rotvec: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rotvec, axis=1)
    axis = np.zeros_like(rotvec)
    mask = theta > 1e-12
    axis[mask] = rotvec[mask] / theta[mask, None]
    ct = np.cos(theta)[:, None]
    st = np.sin(theta)[:, None]
    return v * ct + np.cross(axis, v) * st + axis * np.sum(axis * v, axis=1, keepdims=True) * (1.0 - ct)


def _constant_turn_rate_candidates(xyz: np.ndarray) -> dict[str, np.ndarray]:
    """Coordinated-turn-style physical candidates.

    The existing acceleration grid moves along one last-acceleration vector.
    These variants add a different family: estimate a recent angular velocity
    from velocity direction changes, rotate the last velocity forward twice,
    and optionally apply scalar speed acceleration.  Single candidates need not
    beat accel_c*, but they increase family diversity for blends/selectors.
    """
    d1 = np.diff(xyz, axis=1)
    speed = np.linalg.norm(d1, axis=2)
    speed_delta = np.diff(speed, axis=1)
    last = xyz[:, -1]
    out: dict[str, np.ndarray] = {}
    for pairs in (1, 2, 3, 5):
        rotvec = _average_rotation_vector(d1, pairs)
        scalar_speed_accel = speed_delta[:, -pairs:].mean(axis=1)
        for turn_gain in (-0.25, 0.0, 0.25, 0.50, 0.75, 1.0):
            for speed_gain in (0.0, 0.25, 0.50, 0.75):
                for decay in (0.75, 1.0):
                    v = d1[:, -1].copy()
                    pred = last.copy()
                    step_rot = rotvec * turn_gain
                    for step in range(2):
                        v = _rotate_by_rotation_vector(v, step_rot * (decay**step))
                        v_norm = np.linalg.norm(v, axis=1)
                        next_speed = np.maximum(v_norm + speed_gain * scalar_speed_accel, 0.0)
                        v = v * (next_speed / np.maximum(v_norm, 1e-12))[:, None]
                        pred = pred + v
                    out[f"ctr_p{pairs}_tg{turn_gain:g}_sg{speed_gain:g}_d{decay:g}"] = pred
    return out


def _kalman_ca_predict(xyz: np.ndarray, q: float, r: float) -> np.ndarray:
    """Constant-acceleration Kalman filter/smoother, then 2-step prediction.

    State is [position, velocity_per_40ms, acceleration_per_step2] for each axis.
    q controls acceleration random-walk flexibility; r controls measurement noise.
    """
    n, t, _ = xyz.shape
    f = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    h = np.array([[1.0, 0.0, 0.0]])
    g = np.array([[1.0 / 6.0], [0.5], [1.0]])
    q_mat = q * (g @ g.T)
    r_mat = np.array([[r]])

    state = np.zeros((n, 3, 3), dtype=np.float64)  # sample, xyz-axis, state-dim
    state[:, :, 0] = xyz[:, 0, :]
    state[:, :, 1] = xyz[:, 1, :] - xyz[:, 0, :]
    state[:, :, 2] = xyz[:, 2, :] - 2.0 * xyz[:, 1, :] + xyz[:, 0, :]
    p = np.tile(np.eye(3)[None, None, :, :] * 1.0, (n, 3, 1, 1))

    filt_x = []
    filt_p = []
    pred_x = []
    pred_p = []
    for k in range(t):
        if k > 0:
            state = np.einsum("ij,naj->nai", f, state)
            p = np.einsum("ij,najk,lk->nail", f, p, f) + q_mat
        pred_x.append(state.copy())
        pred_p.append(p.copy())
        z = xyz[:, k, :][:, :, None]
        hx = state[:, :, :1]
        y = z - hx
        s = p[:, :, :1, :1] + r_mat
        k_gain = p[:, :, :, :1] / np.maximum(s, 1e-12)
        state = state + k_gain[:, :, :, 0] * y[:, :, 0, None]
        ikh = np.eye(3)[None, None, :, :] - k_gain @ h[None, None, :, :]
        p = ikh @ p
        filt_x.append(state.copy())
        filt_p.append(p.copy())

    # RTS backward smoothing.  The final smoothed state equals the final
    # filtered state, but keep the per-step buffer so future changes do not
    # accidentally use the k=0 state for forward prediction.
    smooth_states: list[np.ndarray | None] = [None] * t
    smooth_x = filt_x[-1]
    smooth_p = filt_p[-1]
    smooth_states[-1] = smooth_x
    for k in range(t - 2, -1, -1):
        pf = filt_p[k]
        pp = pred_p[k + 1]
        # small matrices, explicit inverse is fine here.
        c = pf @ f.T @ np.linalg.inv(pp)
        smooth_x = filt_x[k] + np.einsum("naij,naj->nai", c, smooth_x - pred_x[k + 1])
        smooth_p = pf + c @ (smooth_p - pp) @ np.swapaxes(c, -1, -2)
        smooth_states[k] = smooth_x

    f2 = f @ f
    final_state = smooth_states[-1]
    assert final_state is not None
    future = np.einsum("ij,naj->nai", f2, final_state)
    return future[:, :, 0]


def build_candidate_bank(xyz: np.ndarray, profile: str = "strong") -> dict[str, np.ndarray]:
    """Deterministic trajectory candidates for +80ms prediction.

    The bank is intentionally redundant.  R-Hit@1cm rewards having at least one
    candidate in the 1cm ball; a later selector/stacker decides which candidates
    are useful for each sample.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 3 or xyz.shape[1:] != (11, 3):
        raise ValueError(f"expected (n, 11, 3), got {xyz.shape}")

    out = physical_predictions(xyz)
    d1 = np.diff(xyz, axis=1)
    last = xyz[:, -1]
    accel = d1[:, -1] - d1[:, -2]

    # Dense local acceleration compensation grid around the known sweet spot.
    for c in np.round(np.arange(0.30, 0.801, 0.025), 3):
        out[f"accel_dense_c{c:g}"] = last + 2.0 * d1[:, -1] + c * accel

    # Cubic/quartic and ridge-regularized polynomial extrapolations.
    for degree in (1, 2, 3, 4):
        for window in range(max(degree + 1, 4), 12):
            out[f"poly_d{degree}_m{window}"] = _poly_extrapolate(xyz, window, degree)
            if degree >= 3 and window >= degree + 2:
                for alpha in (0.01, 0.1, 1.0):
                    out[f"ridgepoly_d{degree}_m{window}_a{alpha:g}"] = _ridge_poly_extrapolate(xyz, window, degree, alpha)

    # Local-frame polynomial extrapolations often reduce axis/global-coordinate bias.
    for degree in (1, 2, 3):
        for window in range(max(degree + 1, 4), 12):
            out[f"local_poly_d{degree}_m{window}"] = _local_poly_extrapolate(xyz, window, degree)

    # Savitzky-Golay smoothing/derivative candidates.
    for window in (5, 7, 9, 11):
        for poly in (2, 3):
            if poly >= window:
                continue
            smooth = savgol_filter(xyz, window_length=window, polyorder=poly, axis=1, mode="interp")
            sd1 = np.diff(smooth, axis=1)
            saccel = sd1[:, -1] - sd1[:, -2]
            for c in (0.0, 0.25, 0.5, 0.75, 1.0):
                out[f"sg_w{window}_p{poly}_accel_c{c:g}"] = smooth[:, -1] + 2.0 * sd1[:, -1] + c * saccel
            for degree in (1, 2, 3):
                if window >= degree + 1:
                    out[f"sg_w{window}_p{poly}_poly_d{degree}"] = _poly_extrapolate(smooth, min(window, 11), degree)

    out.update(_turn_candidates(xyz))
    out.update(_constant_turn_rate_candidates(xyz))

    # Kalman/RTS candidates.  A small grid is enough; the selector will decide.
    if profile in {"strong", "full"}:
        for q in (1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4):
            for r in (1e-7, 1e-6, 1e-5, 1e-4):
                out[f"kalman_ca_q{q:g}_r{r:g}"] = _kalman_ca_predict(xyz, q=q, r=r)

    clean: dict[str, np.ndarray] = {}
    for name, pred in out.items():
        arr = np.asarray(pred, dtype=np.float64)
        if arr.shape != (len(xyz), 3):
            continue
        if np.isfinite(arr).all():
            clean[_safe_name(name)] = arr
    return clean


def _safe_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z_.+-]+", "_", name)


@dataclass(frozen=True)
class CandidateGroup:
    is_accel: float
    is_poly: float
    is_sg: float
    is_kalman: float
    is_turn: float
    is_ml: float


def candidate_group_features(name: str) -> CandidateGroup:
    lower = name.lower()
    return CandidateGroup(
        is_accel=float("accel" in lower),
        is_poly=float("poly" in lower or "linear" in lower or "quad" in lower),
        is_sg=float(lower.startswith("sg_") or "_sg_" in lower),
        is_kalman=float("kalman" in lower),
        is_turn=float("turn" in lower or lower.startswith("ctr_")),
        is_ml=float("multibase" in lower or "gpu" in lower or "hgb" in lower),
    )
