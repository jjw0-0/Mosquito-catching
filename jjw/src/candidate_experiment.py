from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from .candidate_bank import build_candidate_bank, candidate_group_features
from .data import load_competition_data, write_submission
from .features import _trajectory_basis, build_features
from .metrics import score_summary


def _read_submission(path: str | Path) -> np.ndarray:
    return pd.read_csv(path)[["x", "y", "z"]].to_numpy(dtype=np.float64)


def _require_existing(paths: list[str | Path], purpose: str) -> None:
    missing = [str(Path(p)) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifact(s) for {purpose}: {missing}. "
            "Regenerate the referenced OOF/submission files before running this experiment."
        )


def _safe_float_from(pattern: str, text: str, default: float = 0.0) -> float:
    m = re.search(pattern, text)
    if not m:
        return default
    try:
        return float(m.group(1))
    except ValueError:
        return default


def _candidate_static_features(names: list[str]) -> np.ndarray:
    rows = []
    for name in names:
        group = candidate_group_features(name)
        lower = name.lower()
        c = _safe_float_from(r"(?:_c|accel_c|dense_c)([-+0-9.e]+)", lower)
        degree = _safe_float_from(r"(?:_d|poly_d)([0-9]+)", lower)
        window = _safe_float_from(r"(?:_m|_w)([0-9]+)", lower)
        q = _safe_float_from(r"_q([-+0-9.e]+)", lower)
        r = _safe_float_from(r"_r([-+0-9.e]+)", lower)
        rows.append(
            [
                group.is_accel,
                group.is_poly,
                group.is_sg,
                group.is_kalman,
                group.is_turn,
                group.is_ml,
                c,
                degree,
                window,
                np.log10(q) if q > 0 else 0.0,
                np.log10(r) if r > 0 else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def _sample_features(xyz: np.ndarray, stack: np.ndarray, ref: np.ndarray, gpu: np.ndarray | None) -> np.ndarray:
    d1 = np.diff(xyz, axis=1)
    d2 = np.diff(d1, axis=1)
    d3 = np.diff(d2, axis=1)
    speed = np.linalg.norm(d1, axis=2)
    accel = np.linalg.norm(d2, axis=2)
    jerk = np.linalg.norm(d3, axis=2)
    v0 = d1[:, -2]
    v1 = d1[:, -1]
    turn_cos = np.sum(v0 * v1, axis=1, keepdims=True) / np.maximum(
        np.linalg.norm(v0, axis=1, keepdims=True) * np.linalg.norm(v1, axis=1, keepdims=True), 1e-12
    )
    dist_ref = np.linalg.norm(stack - ref[:, None, :], axis=2)
    q = np.quantile(dist_ref, [0.05, 0.10, 0.25, 0.50, 0.75], axis=1).T
    gpu_delta = np.zeros((len(xyz), 1), dtype=np.float64)
    if gpu is not None:
        gpu_delta = np.linalg.norm(gpu - ref, axis=1, keepdims=True)
    compact = np.concatenate(
        [
            speed[:, -1:],
            speed[:, -3:].mean(axis=1, keepdims=True),
            speed[:, -3:].std(axis=1, keepdims=True),
            accel[:, -1:],
            accel[:, -3:].mean(axis=1, keepdims=True),
            jerk[:, -1:],
            turn_cos,
            q,
            gpu_delta,
        ],
        axis=1,
    )
    full = build_features(xyz).astype(np.float32)
    return np.concatenate([compact.astype(np.float32), np.clip(full, -10.0, 10.0)], axis=1).astype(np.float32)


def _local_basis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d1 = np.diff(xyz, axis=1)
    basis = _trajectory_basis(d1)
    scale = np.maximum(np.linalg.norm(d1[:, -3:].mean(axis=1), axis=1), 0.01)[:, None]
    return basis, scale


def _make_top_indices(
    stack: np.ndarray,
    ref: np.ndarray,
    top_m: int,
    anchors: list[np.ndarray] | None = None,
    always: list[int] | None = None,
) -> np.ndarray:
    """Build a per-sample shortlist from several anchors.

    Pure "nearest to current best" misses many oracle candidates.  Multiple
    anchors approximate IMM/MoE gating: current ensemble, multibase, GPU, and
    consensus-like references each retrieve different motion-mode candidates.
    """
    anchors = anchors or [ref]
    always = always or [0]
    n, k, _ = stack.shape
    quota = max(4, int(np.ceil(top_m / max(len(anchors), 1))))
    chunks = []
    primary_dist = np.linalg.norm(stack - ref[:, None, :], axis=2)
    for anchor in anchors:
        dist = np.linalg.norm(stack - anchor[:, None, :], axis=2)
        kk = min(quota, k)
        part = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
        dpart = np.take_along_axis(dist, part, axis=1)
        chunks.append(np.take_along_axis(part, np.argsort(dpart, axis=1), axis=1))
    raw = np.concatenate([np.tile(np.asarray(always, dtype=np.int64)[None, :], (n, 1)), *chunks], axis=1)
    out = np.empty((n, top_m), dtype=np.int64)
    global_fill = np.argsort(primary_dist, axis=1)[:, :top_m]
    for i in range(n):
        seen: list[int] = []
        for idx in raw[i]:
            ii = int(idx)
            if ii not in seen:
                seen.append(ii)
            if len(seen) == top_m:
                break
        if len(seen) < top_m:
            for idx in global_fill[i]:
                ii = int(idx)
                if ii not in seen:
                    seen.append(ii)
                if len(seen) == top_m:
                    break
        out[i] = np.asarray(seen[:top_m], dtype=np.int64)
    return out


def _make_rows(
    xyz: np.ndarray,
    stack: np.ndarray,
    names: list[str],
    top_idx: np.ndarray,
    ref: np.ndarray,
    multibase: np.ndarray,
    gpu: np.ndarray | None,
    sample_feat: np.ndarray,
    prior_hit: np.ndarray,
    prior_mean_dist: np.ndarray,
) -> np.ndarray:
    n, m = top_idx.shape
    rows = np.arange(n)[:, None]
    cand = stack[rows, top_idx]
    last = xyz[:, -1]
    basis, scale = _local_basis(xyz)
    static = _candidate_static_features(names)
    cand_static = static[top_idx]
    cand_prior = np.stack([prior_hit[top_idx], prior_mean_dist[top_idx]], axis=2)
    sf = np.repeat(sample_feat[:, None, :], m, axis=1)

    def project(v: np.ndarray) -> np.ndarray:
        return np.einsum("nmk,nck->nmc", v, basis) / scale[:, None, :]

    ref_delta = cand - ref[:, None, :]
    mb_delta = cand - multibase[:, None, :]
    last_delta = cand - last[:, None, :]
    blocks = [
        sf,
        cand_static,
        cand_prior.astype(np.float32),
        project(ref_delta).astype(np.float32),
        np.linalg.norm(ref_delta, axis=2, keepdims=True).astype(np.float32),
        project(mb_delta).astype(np.float32),
        np.linalg.norm(mb_delta, axis=2, keepdims=True).astype(np.float32),
        project(last_delta).astype(np.float32),
        np.linalg.norm(last_delta, axis=2, keepdims=True).astype(np.float32),
        (np.arange(m, dtype=np.float32)[None, :, None] / max(m - 1, 1)).repeat(n, axis=0),
    ]
    if gpu is not None:
        gpu_delta = cand - gpu[:, None, :]
        blocks.extend(
            [
                project(gpu_delta).astype(np.float32),
                np.linalg.norm(gpu_delta, axis=2, keepdims=True).astype(np.float32),
            ]
        )
    x = np.concatenate(blocks, axis=2).reshape(n * m, -1)
    if not np.isfinite(x).all():
        raise ValueError("selector features contain non-finite values")
    return x


def _score_oracle(stack: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    d = np.linalg.norm(stack - y[:, None, :], axis=2)
    best = d.min(axis=1)
    return {
        "oracle": {
            "hit": float(np.mean(best <= 0.01)),
            "mean_distance": float(best.mean()),
            "median_distance": float(np.median(best)),
        },
        "topk_oracle_by_distance_to_truth": {
            str(k): float(np.mean(np.partition(d, kth=min(k - 1, d.shape[1] - 1), axis=1)[:, k - 1] <= 0.01))
            for k in (1, 2, 3, 5, 10)
        },
    }


def _tune_snap(ref: np.ndarray, selected: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for alpha in np.linspace(0.2, 1.0, 17):
        pred = ref + alpha * (selected - ref)
        row = {"alpha": float(alpha), **score_summary(pred, y)}
        if best is None or row["hit"] > best["hit"] or (
            row["hit"] == best["hit"] and row["median_distance"] < best["median_distance"]
        ):
            best = row
    assert best is not None
    return best


def _selector_model(kind: str, seed: int):
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_features=0.6,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.045,
            l2_regularization=0.03,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            random_state=seed,
        )
    if kind == "hgb_gain":
        return HistGradientBoostingClassifier(
            max_iter=360,
            learning_rate=0.035,
            l2_regularization=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=30,
            random_state=seed,
        )
    if kind == "hgb_reg":
        return HistGradientBoostingRegressor(
            max_iter=320,
            learning_rate=0.04,
            l2_regularization=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            random_state=seed,
        )
    if kind == "extra_trees_reg":
        return ExtraTreesRegressor(
            n_estimators=450,
            max_features=0.65,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(kind)


def run_selector(
    xyz: np.ndarray,
    y: np.ndarray,
    stack: np.ndarray,
    names: list[str],
    ref: np.ndarray,
    multibase: np.ndarray,
    gpu: np.ndarray | None,
    top_m: int,
    seed: int,
    model_kind: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    anchors = [ref, multibase]
    if gpu is not None:
        anchors.append(gpu)
    top_idx = _make_top_indices(stack, ref, top_m, anchors=anchors, always=[0])
    sample_feat = _sample_features(xyz, stack, ref, gpu)
    hit = (np.linalg.norm(stack - y[:, None, :], axis=2) <= 0.01).astype(np.int8)
    dist = np.linalg.norm(stack - y[:, None, :], axis=2)
    oof_prob = np.zeros(top_idx.shape, dtype=np.float32)
    fold_reports = []
    rows = np.arange(len(y))[:, None]

    for fold, (tr, va) in enumerate(kfold.split(xyz), start=1):
        prior_hit = hit[tr].mean(axis=0).astype(np.float32)
        prior_dist = dist[tr].mean(axis=0).astype(np.float32)
        x_tr = _make_rows(xyz[tr], stack[tr], names, top_idx[tr], ref[tr], multibase[tr], None if gpu is None else gpu[tr], sample_feat[tr], prior_hit, prior_dist)
        y_tr = hit[tr[:, None], top_idx[tr]].reshape(-1)
        x_va = _make_rows(xyz[va], stack[va], names, top_idx[va], ref[va], multibase[va], None if gpu is None else gpu[va], sample_feat[va], prior_hit, prior_dist)
        model = _selector_model(model_kind, seed + fold)
        if model_kind == "hgb_gain":
            ref_hit = hit[tr, 0]
            cand_hit = hit[tr[:, None], top_idx[tr]]
            y_gain = (cand_hit & (~ref_hit[:, None])).reshape(-1).astype(np.int8)
            ref_hit_rows = ref_hit[:, None].repeat(top_m, axis=1).reshape(-1)
            cand_hit_rows = cand_hit.reshape(-1).astype(bool)
            # Penalize destructive replacements more than irrelevant both-miss rows.
            w = np.where((~cand_hit_rows) & ref_hit_rows, 3.0, np.where(y_gain.astype(bool), 5.0, 0.35))
            model.fit(x_tr, y_gain, sample_weight=w)
            prob = model.predict_proba(x_va)[:, 1].reshape(len(va), top_m)
        elif model_kind.endswith("_reg"):
            d_tr = dist[tr[:, None], top_idx[tr]].reshape(-1)
            target = np.log1p(np.minimum(d_tr, 0.05) / 0.01)
            w = 0.25 + 2.5 * np.exp(-((d_tr - 0.010) / 0.006) ** 2) + 1.0 * (d_tr <= 0.010)
            model.fit(x_tr, target, sample_weight=w)
            pred_dist = model.predict(x_va).reshape(len(va), top_m)
            prob = pred_dist
        else:
            model.fit(x_tr, y_tr)
            prob = model.predict_proba(x_va)[:, 1].reshape(len(va), top_m)
        oof_prob[va] = prob
        best_pos = prob.argmin(axis=1) if model_kind.endswith("_reg") else prob.argmax(axis=1)
        sel = top_idx[va, best_pos]
        pred = stack[va, sel]
        fold_reports.append({"fold": fold, **score_summary(pred, y[va])})
        print(f"[selector {model_kind}] fold={fold} hit={fold_reports[-1]['hit']:.5f}", flush=True)

    best_j = oof_prob.argmin(axis=1) if model_kind.endswith("_reg") else oof_prob.argmax(axis=1)
    selected_idx = top_idx[np.arange(len(y)), best_j]
    selected = stack[np.arange(len(y)), selected_idx]
    selected_score = score_summary(selected, y)

    # Because argmax can over-select, tune conservative probability margin and distance cap.
    ref_pos = np.argmin(np.where(top_idx == np.array(names).tolist().index("ref"), 0, 1), axis=1) if "ref" in names else np.zeros(len(y), dtype=int)
    # Safer: reference is constructed to be exactly distance 0 from ref, so use nearest slot.
    ref_pos = np.zeros(len(y), dtype=int)
    ref_prob = oof_prob[np.arange(len(y)), ref_pos]
    best_policy: dict[str, Any] | None = None
    best_selected = selected.copy()
    delta_ref = np.linalg.norm(selected - ref, axis=1)
    for margin in np.linspace(0.0, 0.25, 26):
        for dmax in (0.002, 0.004, 0.006, 0.010, 0.015, 0.025, 1.0):
            if model_kind.endswith("_reg"):
                use = (oof_prob[np.arange(len(y)), best_j] < ref_prob - margin) & (delta_ref <= dmax)
            else:
                use = (oof_prob[np.arange(len(y)), best_j] > ref_prob + margin) & (delta_ref <= dmax)
            pred = ref.copy()
            pred[use] = selected[use]
            snap = _tune_snap(ref, pred, y)
            final = ref + snap["alpha"] * (pred - ref)
            row = {
                "margin": float(margin),
                "dmax": float(dmax),
                "replace_rate": float(use.mean()),
                "snap_alpha": snap["alpha"],
                **score_summary(final, y),
            }
            if best_policy is None or row["hit"] > best_policy["hit"] or (
                row["hit"] == best_policy["hit"] and row["median_distance"] < best_policy["median_distance"]
            ):
                best_policy = row
                best_selected = final
    assert best_policy is not None
    report = {
        "top_m": top_m,
        "model_kind": model_kind,
        "selected_argmax": selected_score,
        "best_policy": best_policy,
        "folds": fold_reports,
    }
    artifacts = {
        "top_idx": top_idx,
        "oof_prob": oof_prob,
        "best_selected_oof": best_selected,
    }
    return best_selected, report, artifacts


def train_final_selector_and_predict(
    xyz: np.ndarray,
    y: np.ndarray,
    test_xyz: np.ndarray,
    train_stack: np.ndarray,
    test_stack: np.ndarray,
    names: list[str],
    ref: np.ndarray,
    ref_test: np.ndarray,
    multibase: np.ndarray,
    multibase_test: np.ndarray,
    gpu: np.ndarray | None,
    gpu_test: np.ndarray | None,
    report: dict[str, Any],
    seed: int,
) -> np.ndarray:
    top_m = int(report["top_m"])
    model_kind = str(report["model_kind"])
    policy = report["best_policy"]
    anchors = [ref, multibase]
    anchors_test = [ref_test, multibase_test]
    if gpu is not None and gpu_test is not None:
        anchors.append(gpu)
        anchors_test.append(gpu_test)
    top_idx = _make_top_indices(train_stack, ref, top_m, anchors=anchors, always=[0])
    top_idx_test = _make_top_indices(test_stack, ref_test, top_m, anchors=anchors_test, always=[0])
    sample_feat = _sample_features(xyz, train_stack, ref, gpu)
    sample_feat_test = _sample_features(test_xyz, test_stack, ref_test, gpu_test)
    hit = (np.linalg.norm(train_stack - y[:, None, :], axis=2) <= 0.01).astype(np.int8)
    dist = np.linalg.norm(train_stack - y[:, None, :], axis=2)
    prior_hit = hit.mean(axis=0).astype(np.float32)
    prior_dist = dist.mean(axis=0).astype(np.float32)
    x = _make_rows(xyz, train_stack, names, top_idx, ref, multibase, gpu, sample_feat, prior_hit, prior_dist)
    yy = hit[np.arange(len(y))[:, None], top_idx].reshape(-1)
    model = _selector_model(model_kind, seed + 999)
    if model_kind == "hgb_gain":
        ref_hit = hit[:, 0]
        cand_hit = hit[np.arange(len(y))[:, None], top_idx]
        y_gain = (cand_hit & (~ref_hit[:, None])).reshape(-1).astype(np.int8)
        ref_hit_rows = ref_hit[:, None].repeat(top_m, axis=1).reshape(-1)
        cand_hit_rows = cand_hit.reshape(-1).astype(bool)
        w = np.where((~cand_hit_rows) & ref_hit_rows, 3.0, np.where(y_gain.astype(bool), 5.0, 0.35))
        model.fit(x, y_gain, sample_weight=w)
    elif model_kind.endswith("_reg"):
        full_dist = np.linalg.norm(train_stack - y[:, None, :], axis=2)
        d = full_dist[np.arange(len(y))[:, None], top_idx].reshape(-1)
        target = np.log1p(np.minimum(d, 0.05) / 0.01)
        w = 0.25 + 2.5 * np.exp(-((d - 0.010) / 0.006) ** 2) + 1.0 * (d <= 0.010)
        model.fit(x, target, sample_weight=w)
    else:
        model.fit(x, yy)
    x_test = _make_rows(test_xyz, test_stack, names, top_idx_test, ref_test, multibase_test, gpu_test, sample_feat_test, prior_hit, prior_dist)
    if model_kind.endswith("_reg"):
        prob = model.predict(x_test).reshape(len(test_xyz), top_m)
        best_j = prob.argmin(axis=1)
    else:
        prob = model.predict_proba(x_test)[:, 1].reshape(len(test_xyz), top_m)
        best_j = prob.argmax(axis=1)
    selected_idx = top_idx_test[np.arange(len(test_xyz)), best_j]
    selected = test_stack[np.arange(len(test_xyz)), selected_idx]
    ref_prob = prob[:, 0]
    best_prob = prob[np.arange(len(test_xyz)), best_j]
    if model_kind.endswith("_reg"):
        use = (best_prob < ref_prob - float(policy["margin"])) & (
            np.linalg.norm(selected - ref_test, axis=1) <= float(policy["dmax"])
        )
    else:
        use = (best_prob > ref_prob + float(policy["margin"])) & (
            np.linalg.norm(selected - ref_test, axis=1) <= float(policy["dmax"])
        )
    pred = ref_test.copy()
    pred[use] = selected[use]
    pred = ref_test + float(policy["snap_alpha"]) * (pred - ref_test)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate-bank oracle and selector experiment.")
    parser.add_argument("--zip-path", default="/Users/jjw/Downloads/open.zip")
    parser.add_argument("--out-dir", default="outputs_candidate")
    parser.add_argument("--top-m", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260520)
    parser.add_argument("--profile", choices=["strong", "full"], default="strong")
    parser.add_argument("--models", default="hgb_gain,hgb", help="Comma-separated selector models to run")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_competition_data(args.zip_path)
    print("[candidates] building train", flush=True)
    train_bank = build_candidate_bank(data.train_xyz, profile=args.profile)
    print("[candidates] building test", flush=True)
    test_bank = build_candidate_bank(data.test_xyz, profile=args.profile)

    # Add OOF/test ML predictions as candidates and as the reference solution.
    multibase_paths = [
        "outputs/oof_multibase_cache.npz",
        "outputs/submission_multibase_local_trap_blend.csv",
    ]
    _require_existing(multibase_paths, "multibase reference candidates")
    mb = np.load(multibase_paths[0])
    multibase = mb["multibase_report_opt"]
    multibase_test = _read_submission(multibase_paths[1])
    train_bank["multibase_report_opt"] = multibase
    test_bank["multibase_report_opt"] = multibase_test

    gpu = None
    gpu_test = None
    gpu_paths = [
        Path("outputs_gpu_seq_c0p58/oof_cache.npz"),
        Path("outputs_gpu_seq_c0p58/submission_gpu_seq_raw.csv"),
    ]
    if all(path.exists() for path in gpu_paths):
        gpu = np.load(gpu_paths[0])["gpu_seq"]
        gpu_test = _read_submission(gpu_paths[1])
        train_bank["gpu_seq_raw"] = gpu
        test_bank["gpu_seq_raw"] = gpu_test
        for w in (0.45, 0.60):
            train_bank[f"blend_mb_gpu_w{w:g}"] = (1.0 - w) * multibase + w * gpu
            test_bank[f"blend_mb_gpu_w{w:g}"] = (1.0 - w) * multibase_test + w * gpu_test
    elif any(path.exists() for path in gpu_paths):
        _require_existing(gpu_paths, "optional GPU sequence candidates")

    ref_name = "blend_mb_gpu_w0.6" if "blend_mb_gpu_w0.6" in train_bank else "multibase_report_opt"
    ref = train_bank[ref_name]
    ref_test = test_bank[ref_name]
    # Ensure reference is first and named exactly "ref" so top slot is distance-zero.
    names = ["ref"] + [n for n in train_bank if n != ref_name]
    train_arrays = [ref] + [train_bank[n] for n in names[1:]]
    test_arrays = [ref_test] + [test_bank[n] for n in names[1:]]
    stack = np.stack(train_arrays, axis=1)
    test_stack = np.stack(test_arrays, axis=1)
    print(f"[stack] n={stack.shape[0]} k={stack.shape[1]} ref={ref_name}", flush=True)

    individual = {name: score_summary(stack[:, i, :], data.y) for i, name in enumerate(names)}
    top_individual = dict(sorted(individual.items(), key=lambda kv: kv[1]["hit"], reverse=True)[:40])
    oracle = _score_oracle(stack, data.y)
    report: dict[str, Any] = {
        "ref_name": ref_name,
        "num_candidates": len(names),
        "reference": score_summary(ref, data.y),
        "top_individual": top_individual,
        "oracle": oracle,
        "selectors": {},
    }
    print("[oracle]", json.dumps(oracle, ensure_ascii=False), flush=True)
    print("[top individual]", list(top_individual.items())[:8], flush=True)

    best_pred = ref
    best_selector: tuple[str, dict[str, Any]] | None = None
    selector_oofs: dict[str, np.ndarray] = {}
    for kind in [m.strip() for m in args.models.split(",") if m.strip()]:
        pred, sel_report, _ = run_selector(
            data.train_xyz,
            data.y,
            stack,
            names,
            ref,
            multibase,
            gpu,
            top_m=args.top_m,
            seed=args.seed,
            model_kind=kind,
        )
        selector_oofs[kind] = pred
        report["selectors"][kind] = sel_report
        if score_summary(pred, data.y)["hit"] > score_summary(best_pred, data.y)["hit"]:
            best_pred = pred
            best_selector = (kind, sel_report)
        print(f"[selector done] {kind} {sel_report['best_policy']}", flush=True)

    if best_selector is not None:
        kind, sel_report = best_selector
        pred_test = train_final_selector_and_predict(
            data.train_xyz,
            data.y,
            data.test_xyz,
            stack,
            test_stack,
            names,
            ref,
            ref_test,
            multibase,
            multibase_test,
            gpu,
            gpu_test,
            sel_report,
            args.seed,
        )
        filename = f"submission_candidate_selector_{kind}.csv"
        write_submission(out_dir / filename, data.test_ids, pred_test)
        report["best_submission"] = filename
        report["best_selector"] = kind
        report["best_oof"] = sel_report["best_policy"]
    else:
        write_submission(out_dir / "submission_reference.csv", data.test_ids, ref_test)
        report["best_submission"] = "submission_reference.csv"
        report["best_oof"] = report["reference"]

    (out_dir / "candidate_experiment_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez_compressed(out_dir / "candidate_selector_oof.npz", y=data.y, ref=ref, best=best_pred, **selector_oofs)
    print(f"[done] {out_dir / 'candidate_experiment_report.json'}", flush=True)


if __name__ == "__main__":
    main()
