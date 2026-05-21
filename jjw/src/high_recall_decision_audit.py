from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .candidate_bank import build_candidate_bank
from .candidate_experiment import _make_rows, _make_top_indices, _sample_features
from .ceiling_audit import _choose_ref, _load_known_oof, _oracle_summary, ref_perturbation_stack
from .data import load_competition_data
from .metrics import score_summary
from .posterior_experiment import _sample_context
from .train import trajectory_basis_and_scale


def _read_multibase(base_dir: Path) -> np.ndarray:
    path = base_dir / "outputs/oof_multibase_cache.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path)["multibase_report_opt"]


def _read_gpu(base_dir: Path) -> np.ndarray | None:
    path = base_dir / "outputs_gpu_seq_c0p58/oof_cache.npz"
    if not path.exists():
        return None
    return np.load(path)["gpu_seq"]


def _load_bank(data, base_dir: Path, profile: str) -> tuple[dict[str, np.ndarray], str, np.ndarray]:
    bank = build_candidate_bank(data.train_xyz, profile=profile)
    multibase = _read_multibase(base_dir)
    bank["multibase_report_opt"] = multibase
    gpu = _read_gpu(base_dir)
    if gpu is not None:
        bank["gpu_seq_raw"] = gpu
        for weight in (0.45, 0.60):
            bank[f"blend_mb_gpu_w{weight:g}"] = (1.0 - weight) * multibase + weight * gpu
    bank.update(_load_known_oof(base_dir, data.y))
    ref_name, ref = _choose_ref(bank, data.y)
    return bank, ref_name, ref


def _local_code_offsets(ref: np.ndarray, perturb_stack: np.ndarray, basis: np.ndarray) -> np.ndarray:
    local = np.zeros((perturb_stack.shape[1] + 1, 3), dtype=np.float64)
    local[1:] = np.einsum("kc,ac->ka", perturb_stack[0] - ref[0], basis[0])
    return local


def _cluster_decisions(
    proba: np.ndarray,
    classes: np.ndarray,
    candidates: np.ndarray,
    pair_distance: np.ndarray,
    thresholds: list[float],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    class_candidates = candidates[:, classes, :]
    out["argmax"] = class_candidates[np.arange(len(candidates)), proba.argmax(axis=1)]
    out["expected"] = np.einsum("nc,ncd->nd", proba, class_candidates)
    subdist = pair_distance[np.ix_(classes, classes)]
    for threshold in thresholds:
        adjacency = (subdist <= threshold).astype(np.float64)
        mass = proba @ adjacency.T
        center_idx = mass.argmax(axis=1)
        out[f"bayes_center_t{threshold:g}"] = class_candidates[np.arange(len(candidates)), center_idx]
        mask = adjacency[center_idx]
        weights = proba * mask
        weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
        out[f"bayes_mean_t{threshold:g}"] = np.einsum("nc,ncd->nd", weights, class_candidates)
    return out


def audit_codebook_decision(
    xyz: np.ndarray,
    y: np.ndarray,
    ref: np.ndarray,
    bank_stack: np.ndarray,
    seed: int,
    folds: int,
    radii: list[float],
) -> dict[str, Any]:
    basis, _ = trajectory_basis_and_scale(xyz)
    _, perturb_stack = ref_perturbation_stack(ref, basis, radii)
    code_stack = np.concatenate([ref[:, None, :], perturb_stack], axis=1)
    local_offsets = _local_code_offsets(ref, perturb_stack, basis)
    pair_distance = np.linalg.norm(local_offsets[:, None, :] - local_offsets[None, :, :], axis=2)
    dist = np.linalg.norm(code_stack - y[:, None, :], axis=2)
    labels = dist.argmin(axis=1)
    best_dist = dist[np.arange(len(y)), labels]
    features = _sample_context(xyz, ref, bank_stack)
    variants = {
        name: np.zeros_like(y)
        for name in [
            "argmax",
            "expected",
            *[f"bayes_center_t{t:g}" for t in (0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016)],
            *[f"bayes_mean_t{t:g}" for t in (0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016)],
        ]
    }
    topk_hit = {str(k): np.zeros(len(y), dtype=bool) for k in (1, 3, 5, 10, 20)}
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(features), start=1):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(features[tr_idx])
        x_va = scaler.transform(features[va_idx])
        model = ExtraTreesClassifier(
            n_estimators=700,
            max_features=0.55,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed + fold,
            n_jobs=-1,
        )
        weights = 0.25 + 2.0 * (best_dist[tr_idx] <= 0.01) + np.exp(-((best_dist[tr_idx] - 0.010) / 0.006) ** 2)
        model.fit(x_tr, labels[tr_idx], sample_weight=weights)
        proba = model.predict_proba(x_va)
        classes = model.classes_
        fold_variants = _cluster_decisions(
            proba,
            classes,
            code_stack[va_idx],
            pair_distance,
            thresholds=[0.004, 0.006, 0.008, 0.010, 0.012, 0.014, 0.016],
        )
        for name, pred in fold_variants.items():
            variants[name][va_idx] = pred
        order = np.argsort(-proba, axis=1)
        hit = dist[va_idx] <= 0.01
        for k in (1, 3, 5, 10, 20):
            cls = classes[order[:, : min(k, len(classes))]]
            topk_hit[str(k)][va_idx] = hit[np.arange(len(va_idx))[:, None], cls].any(axis=1)
        print(f"[codebook] fold={fold}", flush=True)
    return {
        "oracle": _oracle_summary(code_stack, y),
        "variants": {name: score_summary(pred, y) for name, pred in variants.items()},
        "topk_model_hit": {key: float(value.mean()) for key, value in topk_hit.items()},
    }


def _best_shrink_from_ref(ref: np.ndarray, pred: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    delta = pred - ref
    for alpha in np.linspace(0.0, 1.2, 25):
        row = {"alpha": float(alpha), **score_summary(ref + alpha * delta, y)}
        if best is None or row["hit"] > best["hit"] or (
            row["hit"] == best["hit"] and row["median_distance"] < best["median_distance"]
        ):
            best = row
    assert best is not None
    return best


def audit_bank_metric_decision(
    xyz: np.ndarray,
    y: np.ndarray,
    stack: np.ndarray,
    names: list[str],
    ref: np.ndarray,
    multibase: np.ndarray,
    gpu: np.ndarray | None,
    top_m: int,
    seed: int,
    folds: int,
) -> dict[str, Any]:
    anchors = [ref, multibase]
    if gpu is not None:
        anchors.append(gpu)
    top_idx = _make_top_indices(stack, ref, top_m, anchors=anchors, always=[0])
    all_dist = np.linalg.norm(stack - y[:, None, :], axis=2)
    short_dist = all_dist[np.arange(len(y))[:, None], top_idx]
    sample_feat = _sample_features(xyz, stack, ref, gpu)
    hit = (all_dist <= 0.01).astype(np.int8)
    prior_hit = hit.mean(axis=0).astype(np.float32)
    prior_dist = all_dist.mean(axis=0).astype(np.float32)
    variants = {
        name: np.zeros_like(y)
        for name in [
            "argmax_hit",
            "bayes_hit_t0.008",
            "bayes_hit_t0.01",
            "bayes_hit_t0.012",
            "bayes_hit_t0.015",
            "bayes_hit_t0.02",
        ]
    }
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(xyz), start=1):
        x_tr = _make_rows(
            xyz[tr_idx],
            stack[tr_idx],
            names,
            top_idx[tr_idx],
            ref[tr_idx],
            multibase[tr_idx],
            None if gpu is None else gpu[tr_idx],
            sample_feat[tr_idx],
            prior_hit,
            prior_dist,
        )
        y_tr = hit[tr_idx[:, None], top_idx[tr_idx]].reshape(-1)
        d_tr = all_dist[tr_idx[:, None], top_idx[tr_idx]].reshape(-1)
        model = HistGradientBoostingClassifier(
            max_iter=320,
            learning_rate=0.04,
            l2_regularization=0.05,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            random_state=seed + 100 + fold,
        )
        weights = 0.3 + 2.0 * y_tr + np.exp(-((d_tr - 0.010) / 0.006) ** 2)
        model.fit(x_tr, y_tr, sample_weight=weights)
        x_va = _make_rows(
            xyz[va_idx],
            stack[va_idx],
            names,
            top_idx[va_idx],
            ref[va_idx],
            multibase[va_idx],
            None if gpu is None else gpu[va_idx],
            sample_feat[va_idx],
            prior_hit,
            prior_dist,
        )
        proba = model.predict_proba(x_va)[:, 1].reshape(len(va_idx), top_m)
        candidates = stack[va_idx[:, None], top_idx[va_idx]]
        variants["argmax_hit"][va_idx] = candidates[np.arange(len(va_idx)), proba.argmax(axis=1)]
        pair_distance = np.linalg.norm(candidates[:, :, None, :] - candidates[:, None, :, :], axis=3)
        for threshold in (0.008, 0.010, 0.012, 0.015, 0.020):
            mass = ((pair_distance <= threshold) * proba[:, None, :]).sum(axis=2)
            variants[f"bayes_hit_t{threshold:g}"][va_idx] = candidates[np.arange(len(va_idx)), mass.argmax(axis=1)]
        print(f"[bank] fold={fold}", flush=True)
    return {
        "top_m": top_m,
        "shortlist_oracle": {
            "hit": float((short_dist.min(axis=1) <= 0.01).mean()),
            "mean_hit_count": float((short_dist <= 0.01).sum(axis=1).mean()),
        },
        "variants": {name: score_summary(pred, y) for name, pred in variants.items()},
        "best_shrink_from_ref": {name: _best_shrink_from_ref(ref, pred, y) for name, pred in variants.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit metric-aware high-recall posterior decisions for the 0.8 target.")
    parser.add_argument("--zip-path", default="/Users/jjw/Downloads/open.zip")
    parser.add_argument("--out-dir", default="outputs_high_recall_decision_audit")
    parser.add_argument("--profile", choices=["strong", "full"], default="strong")
    parser.add_argument("--top-m", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--skip-bank-model", action="store_true")
    args = parser.parse_args()

    base_dir = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_competition_data(args.zip_path)
    bank, ref_name, ref = _load_bank(data, base_dir, args.profile)
    names = ["ref"] + [name for name in bank if name != ref_name]
    stack = np.stack([ref] + [bank[name] for name in names[1:]], axis=1)
    multibase = bank["multibase_report_opt"]
    gpu = bank.get("gpu_seq_raw")
    report: dict[str, Any] = {
        "ref_name": ref_name,
        "reference": score_summary(ref, data.y),
        "bank_oracle": _oracle_summary(stack, data.y),
    }
    print(f"[ref] {ref_name} {report['reference']}", flush=True)
    report["codebook_metric_decision"] = audit_codebook_decision(
        data.train_xyz,
        data.y,
        ref,
        stack,
        seed=args.seed,
        folds=args.folds,
        radii=[0.003, 0.006, 0.010, 0.012],
    )
    if not args.skip_bank_model:
        report["bank_metric_decision"] = audit_bank_metric_decision(
            data.train_xyz,
            data.y,
            stack,
            names,
            ref,
            multibase,
            gpu,
            top_m=args.top_m,
            seed=args.seed,
            folds=args.folds,
        )
    (out_dir / "high_recall_decision_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[done] {out_dir / 'high_recall_decision_report.json'}", flush=True)


if __name__ == "__main__":
    main()
