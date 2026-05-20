from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

from .candidate_bank import build_candidate_bank
from .ceiling_audit import _choose_ref, _load_known_oof, _oracle_summary, ref_perturbation_stack
from .data import load_competition_data
from .features import build_features
from .metrics import score_summary
from .train import sample_weight_for, trajectory_basis_and_scale


def _sample_context(xyz: np.ndarray, ref: np.ndarray, bank_stack: np.ndarray | None = None) -> np.ndarray:
    basis, scale = trajectory_basis_and_scale(xyz)
    last = xyz[:, -1]
    d1 = np.diff(xyz, axis=1)
    d2 = np.diff(d1, axis=1)
    d3 = np.diff(d2, axis=1)
    ref_local = np.einsum("nc,nkc->nk", ref - last, basis) / scale
    blocks = [
        np.clip(build_features(xyz), -10.0, 10.0),
        ref_local,
        np.linalg.norm(ref - last, axis=1, keepdims=True),
        np.linalg.norm(d1[:, -1], axis=1, keepdims=True),
        np.linalg.norm(d1[:, -3:].mean(axis=1), axis=1, keepdims=True),
        np.linalg.norm(d2[:, -1], axis=1, keepdims=True),
        np.linalg.norm(d3[:, -1], axis=1, keepdims=True),
    ]
    if bank_stack is not None:
        dist_ref = np.linalg.norm(bank_stack - ref[:, None, :], axis=2)
        blocks.extend(
            [
                np.quantile(dist_ref, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75], axis=1).T,
                dist_ref.std(axis=1, keepdims=True),
            ]
        )
    feat = np.concatenate([b.astype(np.float32) for b in blocks], axis=1)
    if not np.isfinite(feat).all():
        raise ValueError("posterior features contain non-finite values")
    return feat


def _perturb_codebook(ref: np.ndarray, basis: np.ndarray, radii: list[float]) -> tuple[list[str], np.ndarray, np.ndarray]:
    names, stack = ref_perturbation_stack(ref, basis, radii)
    zero = ref[:, None, :]
    # Derive the codebook local offsets from sample 0; the same local codebook
    # is used for every sample and then projected through that sample's basis.
    local = np.einsum("kc,ac->ka", stack[0] - ref[0], basis[0])
    return ["ref", *names], np.concatenate([zero, stack], axis=1), np.vstack([np.zeros((1, 3)), local])


def _scan_policy(ref: np.ndarray, selected: np.ndarray, confidence: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for thr in np.linspace(0.0, 0.95, 96):
        use = confidence >= thr
        for alpha in np.linspace(0.20, 1.20, 21):
            pred = ref.copy()
            pred[use] = ref[use] + alpha * (selected[use] - ref[use])
            row = {
                "threshold": float(thr),
                "alpha": float(alpha),
                "replace_rate": float(use.mean()),
                **score_summary(pred, y),
            }
            if best is None or row["hit"] > best["hit"] or (
                row["hit"] == best["hit"] and row["median_distance"] < best["median_distance"]
            ):
                best = row
    assert best is not None
    return best


def run_codebook_classifier(
    features: np.ndarray,
    ref: np.ndarray,
    code_stack: np.ndarray,
    y: np.ndarray,
    seed: int,
    folds: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    dist = np.linalg.norm(code_stack - y[:, None, :], axis=2)
    labels = dist.argmin(axis=1)
    best_dist = dist[np.arange(len(y)), labels]
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    pred_idx = np.zeros(len(y), dtype=np.int64)
    pred_conf = np.zeros(len(y), dtype=np.float32)
    fold_reports = []
    for fold, (tr, va) in enumerate(kfold.split(features), start=1):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(features[tr])
        x_va = scaler.transform(features[va])
        model = ExtraTreesClassifier(
            n_estimators=500,
            max_features=0.55,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed + fold,
            n_jobs=-1,
        )
        # Focus on samples where the codebook can matter; far non-oracle rows
        # mostly teach arbitrary labels that cannot produce hits.
        weights = 0.25 + 2.5 * (best_dist[tr] <= 0.01) + 1.0 * np.exp(-((best_dist[tr] - 0.010) / 0.006) ** 2)
        model.fit(x_tr, labels[tr], sample_weight=weights)
        proba = model.predict_proba(x_va)
        classes = model.classes_
        j = proba.argmax(axis=1)
        pred_idx[va] = classes[j]
        pred_conf[va] = proba[np.arange(len(va)), j]
        pred = code_stack[va, pred_idx[va]]
        fold_reports.append({"fold": fold, **score_summary(pred, y[va])})
        print(f"[codebook clf] fold={fold} hit={fold_reports[-1]['hit']:.4f}", flush=True)
    selected = code_stack[np.arange(len(y)), pred_idx]
    report = {
        "argmax": score_summary(selected, y),
        "policy": _scan_policy(ref, selected, pred_conf, y),
        "folds": fold_reports,
        "label_hit_rate": float(np.mean(best_dist <= 0.01)),
        "num_classes": int(code_stack.shape[1]),
    }
    return selected, pred_conf, report


def run_residual_regressor(
    features: np.ndarray,
    xyz: np.ndarray,
    ref: np.ndarray,
    y: np.ndarray,
    seed: int,
    folds: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    basis, scale = trajectory_basis_and_scale(xyz)
    target = (np.einsum("nc,nkc->nk", y - ref, basis) / scale).astype(np.float32)
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros_like(y)
    fold_reports = []
    weights = sample_weight_for("trap", ref, y)
    for fold, (tr, va) in enumerate(kfold.split(features), start=1):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(features[tr])
        x_va = scaler.transform(features[va])
        model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=260,
                learning_rate=0.04,
                l2_regularization=0.05,
                max_leaf_nodes=31,
                min_samples_leaf=25,
                random_state=seed + 100 + fold,
            )
        )
        model.fit(x_tr, target[tr], sample_weight=weights[tr])
        pred_local = model.predict(x_va)
        corr = np.einsum("nk,nkc->nc", pred_local * scale[va], basis[va])
        oof[va] = ref[va] + corr
        fold_reports.append({"fold": fold, **score_summary(oof[va], y[va])})
        print(f"[residual reg] fold={fold} hit={fold_reports[-1]['hit']:.4f}", flush=True)
    correction = oof - ref
    best = None
    for shrink in np.linspace(0.0, 1.4, 71):
        pred = ref + shrink * correction
        row = {"shrink": float(shrink), **score_summary(pred, y)}
        if best is None or row["hit"] > best["hit"] or (
            row["hit"] == best["hit"] and row["median_distance"] < best["median_distance"]
        ):
            best = row
    assert best is not None
    return oof, {"raw": score_summary(oof, y), "best_shrink": best, "folds": fold_reports}


def main() -> None:
    parser = argparse.ArgumentParser(description="First-pass multi-modal posterior experiments for 0.8+ target.")
    parser.add_argument("--zip-path", default="/Users/jjw/Downloads/open.zip")
    parser.add_argument("--out-dir", default="outputs_posterior_experiment")
    parser.add_argument("--profile", choices=["quick", "strong"], default="strong")
    parser.add_argument("--radii", default="0.003,0.006,0.010,0.012")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    base_dir = Path.cwd()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_competition_data(args.zip_path)
    bank = build_candidate_bank(data.train_xyz, profile=args.profile)
    bank.update(_load_known_oof(base_dir, data.y))
    ref_name, ref = _choose_ref(bank, data.y)
    names = ["ref"] + [name for name in bank if name != ref_name]
    bank_stack = np.stack([ref] + [bank[name] for name in names[1:]], axis=1)
    basis, _ = trajectory_basis_and_scale(data.train_xyz)
    radii = [float(x) for x in args.radii.split(",") if x]
    code_names, code_stack, _ = _perturb_codebook(ref, basis, radii)

    report: dict[str, Any] = {
        "ref_name": ref_name,
        "reference": score_summary(ref, data.y),
        "bank_oracle": _oracle_summary(bank_stack, data.y),
        "codebook_radii": radii,
        "codebook_oracle": _oracle_summary(code_stack, data.y),
    }
    print(f"[ref] {ref_name} {report['reference']}", flush=True)
    print(f"[codebook oracle] {report['codebook_oracle']['oracle']}", flush=True)
    features = _sample_context(data.train_xyz, ref, bank_stack)

    selected, conf, clf_report = run_codebook_classifier(features, ref, code_stack, data.y, args.seed, args.folds)
    report["codebook_classifier"] = clf_report
    residual_oof, reg_report = run_residual_regressor(features, data.train_xyz, ref, data.y, args.seed, args.folds)
    report["residual_regressor"] = reg_report

    # Controlled blend between the two posterior views.
    best_blend = None
    for w in np.linspace(0.0, 1.0, 51):
        pred = (1.0 - w) * residual_oof + w * selected
        row = {"codebook_weight": float(w), **score_summary(pred, data.y)}
        if best_blend is None or row["hit"] > best_blend["hit"] or (
            row["hit"] == best_blend["hit"] and row["median_distance"] < best_blend["median_distance"]
        ):
            best_blend = row
    report["posterior_blend"] = best_blend

    (out_dir / "posterior_experiment_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(out_dir / "oof_posterior_experiment.npz", y=data.y, codebook=selected, confidence=conf, residual=residual_oof)
    print(f"[done] {out_dir / 'posterior_experiment_report.json'}", flush=True)


if __name__ == "__main__":
    main()
