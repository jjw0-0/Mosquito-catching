from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .candidate_bank import build_candidate_bank
from .ceiling_audit import _choose_ref, _load_known_oof, _oracle_summary, ref_perturbation_stack
from .data import load_competition_data
from .metrics import score_summary
from .posterior_experiment import _sample_context
from .train import trajectory_basis_and_scale


def _stable_seed_offset(text: str, modulo: int = 1000) -> int:
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") % modulo


def _binary_metrics(y_true: np.ndarray, score: np.ndarray, name: str) -> dict[str, Any]:
    y_true = y_true.astype(int)
    out: dict[str, Any] = {
        "name": name,
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) == 2 else float("nan"),
        "average_precision": float(average_precision_score(y_true, score)) if y_true.any() else float("nan"),
        "precision_at_fraction": {},
        "recall_at_fraction": {},
    }
    order = np.argsort(-score)
    for frac in (0.01, 0.02, 0.05, 0.10, 0.20, 0.33, 0.50):
        k = max(1, int(round(len(y_true) * frac)))
        sel = order[:k]
        out["precision_at_fraction"][f"{frac:g}"] = float(y_true[sel].mean())
        out["recall_at_fraction"][f"{frac:g}"] = float(y_true[sel].sum() / max(1, y_true.sum()))
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    points = []
    for target_precision in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90):
        ok = np.where(precision[:-1] >= target_precision)[0]
        if len(ok) == 0:
            points.append({"precision_floor": target_precision, "recall": 0.0, "threshold": None, "predicted_rate": 0.0})
            continue
        j = ok[np.argmax(recall[:-1][ok])]
        pred = score >= thresholds[j]
        points.append(
            {
                "precision_floor": target_precision,
                "precision": float(precision[j]),
                "recall": float(recall[j]),
                "threshold": float(thresholds[j]),
                "predicted_rate": float(pred.mean()),
            }
        )
    out["pr_operating_points"] = points
    return out


def _make_classifier(kind: str, seed: int):
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.04,
            l2_regularization=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=25,
            random_state=seed,
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=600,
            max_features=0.55,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(kind)


def _oof_binary_classifier(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    folds: int,
    kind: str,
    positive_weight: float = 1.0,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    score = np.zeros(len(y), dtype=np.float32)
    fold_reports = []
    for fold, (tr, va) in enumerate(kfold.split(x), start=1):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[tr])
        x_va = scaler.transform(x[va])
        model = _make_classifier(kind, seed + fold)
        weights = np.where(y[tr].astype(bool), positive_weight, 1.0)
        model.fit(x_tr, y[tr], sample_weight=weights)
        score[va] = model.predict_proba(x_va)[:, 1]
        pred = score[va] >= 0.5
        fold_reports.append(
            {
                "fold": fold,
                "positive_rate": float(y[va].mean()),
                "score_mean": float(score[va].mean()),
                "precision_at_0p5": float(np.mean(y[va][pred])) if pred.any() else 0.0,
                "predicted_rate_at_0p5": float(pred.mean()),
            }
        )
        print(f"[{kind}] fold={fold} positives={y[va].mean():.4f} score_mean={score[va].mean():.4f}", flush=True)
    return score, fold_reports


def _direction_codebook(ref: np.ndarray, basis: np.ndarray, radii: list[float]) -> tuple[list[str], np.ndarray]:
    names, stack = ref_perturbation_stack(ref, basis, radii)
    return ["ref", *names], np.concatenate([ref[:, None, :], stack], axis=1)


def _topk_codebook_hit_from_prob(proba: np.ndarray, classes: np.ndarray, code_hit: np.ndarray, k_values: list[int]) -> dict[str, float]:
    order = np.argsort(-proba, axis=1)
    out = {}
    for k in k_values:
        kk = min(k, order.shape[1])
        pred_classes = classes[order[:, :kk]]
        hit_any = code_hit[np.arange(len(code_hit))[:, None], pred_classes].any(axis=1)
        out[str(k)] = float(hit_any.mean())
    return out


def _weighted_codebook_predictions(
    proba: np.ndarray,
    classes: np.ndarray,
    code_stack_fold: np.ndarray,
    k_values: list[int],
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    class_stack = code_stack_fold[:, classes, :]
    out["expected_all"] = np.einsum("nc,ncd->nd", proba, class_stack)
    order = np.argsort(-proba, axis=1)
    for k in k_values:
        kk = min(k, proba.shape[1])
        idx = order[:, :kk]
        cls = classes[idx]
        p = np.take_along_axis(proba, idx, axis=1)
        p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
        cand = code_stack_fold[np.arange(len(code_stack_fold))[:, None], cls]
        out[f"top{k}_weighted"] = np.einsum("nk,nkd->nd", p, cand)
        out[f"top{k}_mean"] = cand.mean(axis=1)
    return out


def _oof_direction_classifier(
    x: np.ndarray,
    code_stack: np.ndarray,
    y: np.ndarray,
    seed: int,
    folds: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dist = np.linalg.norm(code_stack - y[:, None, :], axis=2)
    labels = dist.argmin(axis=1)
    code_hit = dist <= 0.01
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    pred_idx = np.zeros(len(y), dtype=np.int64)
    pred_conf = np.zeros(len(y), dtype=np.float32)
    topk_hit_accum = {str(k): np.zeros(len(y), dtype=bool) for k in (1, 3, 5, 10, 20)}
    pred_variants: dict[str, np.ndarray] = {"argmax": np.zeros_like(y)}
    for fold, (tr, va) in enumerate(kfold.split(x), start=1):
        scaler = StandardScaler()
        x_tr = scaler.fit_transform(x[tr])
        x_va = scaler.transform(x[va])
        model = ExtraTreesClassifier(
            n_estimators=700,
            max_features=0.55,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=seed + 100 + fold,
            n_jobs=-1,
        )
        best_dist = dist[np.arange(len(y)), labels]
        weights = 0.25 + 2.0 * (best_dist[tr] <= 0.01) + 1.0 * np.exp(-((best_dist[tr] - 0.010) / 0.006) ** 2)
        model.fit(x_tr, labels[tr], sample_weight=weights)
        proba = model.predict_proba(x_va)
        classes = model.classes_
        j = proba.argmax(axis=1)
        pred_idx[va] = classes[j]
        pred_conf[va] = proba[np.arange(len(va)), j]
        pred_variants["argmax"][va] = code_stack[va, pred_idx[va]]
        fold_variants = _weighted_codebook_predictions(proba, classes, code_stack[va], [3, 5, 10, 20])
        for name, pred in fold_variants.items():
            pred_variants.setdefault(name, np.zeros_like(y))[va] = pred
        topk = _topk_codebook_hit_from_prob(proba, classes, code_hit[va], [1, 3, 5, 10, 20])
        for key, value in topk.items():
            # Recompute per-row flags to preserve OOF masks.
            kk = min(int(key), proba.shape[1])
            cls = classes[np.argsort(-proba, axis=1)[:, :kk]]
            topk_hit_accum[key][va] = code_hit[va[:, None], cls].any(axis=1)
        print(f"[direction] fold={fold} top1_hit={topk['1']:.4f} top10_hit={topk['10']:.4f}", flush=True)
    variant_scores = {name: score_summary(pred, y) for name, pred in pred_variants.items()}
    return {
        "variants": variant_scores,
        "confidence": {
            "mean": float(pred_conf.mean()),
            "p90": float(np.quantile(pred_conf, 0.90)),
            "p99": float(np.quantile(pred_conf, 0.99)),
        },
        "oracle": _oracle_summary(code_stack, y),
        "topk_model_hit": {key: float(value.mean()) for key, value in topk_hit_accum.items()},
    }, pred_variants


def _scan_controlled_replacement(
    ref: np.ndarray,
    y: np.ndarray,
    candidates: dict[str, np.ndarray],
    gate_scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    miss_scores = {k: v for k, v in gate_scores.items() if k.startswith("ref_miss_")}
    hard_scores = {k: v for k, v in gate_scores.items() if k.startswith("hard_unrecoverable_")}
    fractions = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
    hard_keep_quantiles: tuple[float | None, ...] = (None, 0.99, 0.95, 0.90, 0.80, 0.70)
    for cand_name, cand in candidates.items():
        if cand_name == "argmax":
            # Argmax is already known to be unstable; keep it out of the
            # controlled policy scan to avoid overfitting obvious bad moves.
            continue
        delta = cand - ref
        for miss_name, miss_score in miss_scores.items():
            for frac in fractions:
                miss_thr = float(np.quantile(miss_score, 1.0 - frac))
                miss_mask = miss_score >= miss_thr
                hard_items = list(hard_scores.items()) or [("no_hard_filter", np.zeros(len(y)))]
                for hard_name, hard_score in hard_items:
                    for keep_q in hard_keep_quantiles:
                        if keep_q is None:
                            mask = miss_mask
                            hard_thr = None
                        else:
                            hard_thr = float(np.quantile(hard_score, keep_q))
                            mask = miss_mask & (hard_score <= hard_thr)
                        if mask.sum() == 0:
                            continue
                        for alpha in np.linspace(0.10, 1.20, 23):
                            pred = ref.copy()
                            pred[mask] = ref[mask] + alpha * delta[mask]
                            summary = score_summary(pred, y)
                            row = {
                                "candidate": cand_name,
                                "miss_score": miss_name,
                                "miss_fraction": float(frac),
                                "miss_threshold": miss_thr,
                                "hard_score": hard_name,
                                "hard_keep_quantile": None if keep_q is None else float(keep_q),
                                "hard_threshold": hard_thr,
                                "alpha": float(alpha),
                                "replace_rate": float(mask.mean()),
                                **summary,
                            }
                            if best is None or row["hit"] > best["hit"] or (
                                row["hit"] == best["hit"] and row["median_distance"] < best["median_distance"]
                            ):
                                best = row
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether ref miss and residual posterior are learnable enough for 0.8.")
    parser.add_argument("--zip-path", default="/Users/jjw/Downloads/open.zip")
    parser.add_argument("--out-dir", default="outputs_miss_detector")
    parser.add_argument("--profile", choices=["quick", "strong"], default="strong")
    parser.add_argument("--radii", default="0.003,0.006,0.010,0.012")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--detectors", default="hgb,extra_trees")
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
    _, code_stack = _direction_codebook(ref, basis, radii)

    ref_dist = np.linalg.norm(ref - data.y, axis=1)
    bank_oracle_dist = np.linalg.norm(bank_stack - data.y[:, None, :], axis=2).min(axis=1)
    code_oracle_dist = np.linalg.norm(code_stack - data.y[:, None, :], axis=2).min(axis=1)
    labels = {
        "ref_miss": (ref_dist > 0.01).astype(np.int8),
        "bank_recoverable": ((ref_dist > 0.01) & (bank_oracle_dist <= 0.01)).astype(np.int8),
        "code_recoverable": ((ref_dist > 0.01) & (code_oracle_dist <= 0.01)).astype(np.int8),
        "hard_unrecoverable": ((ref_dist > 0.01) & (bank_oracle_dist > 0.01) & (code_oracle_dist > 0.01)).astype(np.int8),
    }
    features = _sample_context(data.train_xyz, ref, bank_stack)
    report: dict[str, Any] = {
        "ref_name": ref_name,
        "reference": score_summary(ref, data.y),
        "bank_oracle": _oracle_summary(bank_stack, data.y),
        "code_oracle": _oracle_summary(code_stack, data.y),
        "labels": {name: {"positive_rate": float(value.mean()), "count": int(value.sum())} for name, value in labels.items()},
        "binary_detectors": {},
    }
    gate_scores: dict[str, np.ndarray] = {}
    print(f"[ref] {ref_name} {report['reference']}", flush=True)
    for label_name, y_label in labels.items():
        for kind in [item.strip() for item in args.detectors.split(",") if item.strip()]:
            print(f"[detector] {label_name}/{kind}", flush=True)
            score, folds = _oof_binary_classifier(
                features,
                y_label,
                seed=args.seed + _stable_seed_offset(label_name + kind),
                folds=args.folds,
                kind=kind,
                positive_weight=max(1.0, (len(y_label) - y_label.sum()) / max(1, y_label.sum())),
            )
            key = f"{label_name}_{kind}"
            gate_scores[key] = score
            report["binary_detectors"][key] = {
                "metrics": _binary_metrics(y_label, score, key),
                "folds": folds,
            }

    direction_report, direction_preds = _oof_direction_classifier(features, code_stack, data.y, args.seed, args.folds)
    report["direction_posterior"] = direction_report
    report["controlled_replacement"] = _scan_controlled_replacement(ref, data.y, direction_preds, gate_scores)
    (out_dir / "miss_detector_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "miss_detector_oof.npz",
        y=data.y,
        ref=ref,
        **{f"score_{name}": score for name, score in gate_scores.items()},
        **{f"direction_{name}": pred for name, pred in direction_preds.items()},
    )
    print(f"[done] {out_dir / 'miss_detector_report.json'}", flush=True)


if __name__ == "__main__":
    main()
