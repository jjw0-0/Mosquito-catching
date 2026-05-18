from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_competition_data, write_submission
from .features import build_features, physical_predictions, _trajectory_basis
from .metrics import r_hit, score_summary


def _spec(estimator: Any, weighting: str | None = None, target_space: str = "global") -> dict[str, Any]:
    return {"estimator": estimator, "weighting": weighting, "target_space": target_space}


def make_models(profile: str, seed: int) -> dict[str, dict[str, Any]]:
    if profile == "quick":
        return {
            "ridge": _spec(make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
            "extra_trees": _spec(
                ExtraTreesRegressor(
                    n_estimators=250, max_features=0.75, min_samples_leaf=2, random_state=seed + 11, n_jobs=-1
                )
            ),
            # R-Hit@1cm rewards getting borderline samples across the 1cm radius more
            # than reducing very large outliers.  This weighted tree gives less leverage
            # to far misses and consistently improved OOF hit-rate in re-validation.
            "extra_trees_near": _spec(
                ExtraTreesRegressor(
                    n_estimators=250, max_features=0.75, min_samples_leaf=2, random_state=seed + 13, n_jobs=-1
                ),
                weighting="near_gauss",
            ),
            "hgb": _spec(
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=250, learning_rate=0.045, l2_regularization=0.01,
                        max_leaf_nodes=31, random_state=seed + 21
                    )
                )
            ),
            "hgb_near": _spec(
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=220, learning_rate=0.05, l2_regularization=0.01,
                        max_leaf_nodes=15, random_state=seed + 23
                    )
                ),
                weighting="near_gauss",
            ),
            "hgb_local_trap": _spec(
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=220, learning_rate=0.05, l2_regularization=0.01,
                        max_leaf_nodes=15, random_state=seed + 24
                    )
                ),
                weighting="trap",
                target_space="local",
            ),
        }
    if profile == "full":
        return {
            "ridge": _spec(make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
            "extra_trees_a": _spec(
                ExtraTreesRegressor(
                    n_estimators=500, max_features=0.70, min_samples_leaf=2, random_state=seed + 11, n_jobs=-1
                )
            ),
            "extra_trees_near": _spec(
                ExtraTreesRegressor(
                    n_estimators=500, max_features=0.75, min_samples_leaf=2, random_state=seed + 13, n_jobs=-1
                ),
                weighting="near_gauss",
            ),
            "extra_trees_b": _spec(
                ExtraTreesRegressor(
                    n_estimators=700, max_features=0.55, min_samples_leaf=1, random_state=seed + 12, n_jobs=-1
                )
            ),
            "random_forest": _spec(
                RandomForestRegressor(
                    n_estimators=350, max_features=0.70, min_samples_leaf=2, random_state=seed + 31, n_jobs=-1
                )
            ),
            "hgb_a": _spec(
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=350, learning_rate=0.04, l2_regularization=0.01,
                        max_leaf_nodes=31, random_state=seed + 21
                    )
                )
            ),
            "hgb_near": _spec(
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=300, learning_rate=0.04, l2_regularization=0.01,
                        max_leaf_nodes=31, random_state=seed + 23
                    )
                ),
                weighting="near_gauss",
            ),
            "hgb_local_trap": _spec(
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=300, learning_rate=0.04, l2_regularization=0.01,
                        max_leaf_nodes=31, random_state=seed + 24
                    )
                ),
                weighting="trap",
                target_space="local",
            ),
            "hgb_b": _spec(
                MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        max_iter=450, learning_rate=0.03, l2_regularization=0.03,
                        max_leaf_nodes=47, random_state=seed + 22
                    )
                )
            ),
        }
    raise ValueError(f"unknown profile: {profile}")


def sample_weight_for(kind: str | None, base: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    if kind is None:
        return None
    base_distance = np.linalg.norm(base - y, axis=1)
    if kind == "near_gauss":
        weights = 0.2 + np.exp(-((base_distance - 0.010) / 0.007) ** 2)
    elif kind == "trap":
        weights = np.where(
            base_distance < 0.003,
            0.5,
            np.where(base_distance < 0.015, 2.0, np.where(base_distance < 0.03, 0.8, 0.05)),
        )
    elif kind == "clip_far":
        weights = np.where(base_distance < 0.04, 1.0, 0.2)
    else:
        raise ValueError(f"unknown weighting: {kind}")
    return weights / np.mean(weights)


def residual_target(residual: np.ndarray, basis: np.ndarray, scale: np.ndarray, target_space: str) -> np.ndarray:
    if target_space == "global":
        return residual
    local = np.einsum("nc,nkc->nk", residual, basis)
    if target_space == "local":
        return local
    if target_space == "local_scaled":
        return local / scale
    raise ValueError(f"unknown target_space: {target_space}")


def correction_from_prediction(prediction: np.ndarray, basis: np.ndarray, scale: np.ndarray, target_space: str) -> np.ndarray:
    if target_space == "global":
        return prediction
    local = prediction * scale if target_space == "local_scaled" else prediction
    if target_space in {"local", "local_scaled"}:
        return np.einsum("nk,nkc->nc", local, basis)
    raise ValueError(f"unknown target_space: {target_space}")


def trajectory_basis_and_scale(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d1 = np.diff(xyz, axis=1)
    basis = _trajectory_basis(d1)
    speed = np.linalg.norm(d1[:, -3:].mean(axis=1), axis=1)
    scale = np.maximum(speed, 0.01)[:, None]
    return basis, scale


def _prediction_matrix(preds: list[np.ndarray]) -> np.ndarray:
    return np.stack(preds, axis=0)


def find_best_shrink(base: np.ndarray, correction: np.ndarray, y: np.ndarray) -> dict[str, float]:
    best: dict[str, float] | None = None
    for shrink in np.linspace(-0.2, 1.4, 81):
        pred = base + shrink * correction
        row = {"shrink": float(shrink), **score_summary(pred, y)}
        if best is None or row["hit"] > best["hit"] or (
            row["hit"] == best["hit"] and row["median_distance"] < best["median_distance"]
        ):
            best = row
    assert best is not None
    return best


def find_best_blend(candidate_preds: dict[str, np.ndarray], y: np.ndarray, seed: int) -> dict[str, Any]:
    """Small random/grid blend search on OOF predictions."""
    names = list(candidate_preds)
    mat = _prediction_matrix([candidate_preds[n] for n in names])
    rng = np.random.default_rng(seed)
    best = {"hit": -1.0, "weights": None, "names": names, "summary": None}

    # Singletons and uniform blends are strong guardrails.
    candidates = []
    for i in range(len(names)):
        w = np.zeros(len(names)); w[i] = 1.0; candidates.append(w)
    candidates.append(np.ones(len(names)) / len(names))

    # Dirichlet search favors sparse-but-not-singleton ensembles.
    for alpha in (0.15, 0.30, 0.60, 1.0, 2.0):
        for _ in range(800):
            candidates.append(rng.dirichlet(np.full(len(names), alpha)))

    for w in candidates:
        pred = np.tensordot(w, mat, axes=(0, 0))
        summary = score_summary(pred, y)
        if summary["hit"] > best["hit"] or (
            summary["hit"] == best["hit"] and summary["median_distance"] < best["summary"]["median_distance"]
        ):
            best = {"hit": summary["hit"], "weights": w.tolist(), "names": names, "summary": summary}
    return best


def make_oof(
    features: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    basis: np.ndarray,
    scale: np.ndarray,
    models: dict[str, dict[str, Any]],
    folds: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, list[Any]], dict[str, dict[str, float]]]:
    residual = y - base
    kfold = KFold(n_splits=folds, shuffle=True, random_state=seed)
    oof: dict[str, np.ndarray] = {name: np.zeros_like(y) for name in models}
    fitted: dict[str, list[Any]] = {name: [] for name in models}
    fold_scores: dict[str, dict[str, float]] = {}
    model_weights = {name: sample_weight_for(spec.get("weighting"), base, y) for name, spec in models.items()}

    for fold, (tr_idx, va_idx) in enumerate(kfold.split(features), start=1):
        print(f"[fold {fold}/{folds}] train={len(tr_idx)} valid={len(va_idx)}", flush=True)
        for name, spec in models.items():
            model = clone(spec["estimator"])
            weights = model_weights[name]
            target_space = spec.get("target_space", "global")
            target = residual_target(residual, basis, scale, target_space)
            if weights is None:
                model.fit(features[tr_idx], target[tr_idx])
            else:
                model.fit(features[tr_idx], target[tr_idx], sample_weight=weights[tr_idx])
            pred_target = model.predict(features[va_idx])
            corr = correction_from_prediction(pred_target, basis[va_idx], scale[va_idx], target_space)
            pred = base[va_idx] + corr
            oof[name][va_idx] = pred
            fitted[name].append(model)
            row = score_summary(pred, y[va_idx])
            fold_scores[f"{name}/fold_{fold}"] = row
            print(f"  {name:16s} hit={row['hit']:.5f} med={row['median_distance']:.6f}", flush=True)
    return oof, fitted, fold_scores


def predict_with_fold_models(
    models: list[Any],
    features: np.ndarray,
    base: np.ndarray,
    basis: np.ndarray,
    scale: np.ndarray,
    target_space: str = "global",
    shrink: float = 1.0,
) -> np.ndarray:
    corrections = [
        correction_from_prediction(m.predict(features), basis, scale, target_space)
        for m in models
    ]
    correction = np.mean(corrections, axis=0)
    return base + shrink * correction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DACON mosquito trajectory solution and create submissions.")
    parser.add_argument("--zip-path", default="/Users/jjw/Downloads/open.zip", help="Path to DACON open.zip")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument(
        "--residual-base",
        default="accel_c0.5",
        help="Physical prediction used as residual-learning base. Defaults to the revalidated ML-friendly baseline.",
    )
    parser.add_argument("--no-submissions", action="store_true", help="Skip training final/full predictions and CSV generation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.zip_path}", flush=True)
    data = load_competition_data(args.zip_path)
    x_train, y, x_test = data.train_xyz, data.y, data.test_xyz

    print("[features] building train/test features", flush=True)
    f_train = build_features(x_train)
    f_test = build_features(x_test)
    basis_train, scale_train = trajectory_basis_and_scale(x_train)
    basis_test, scale_test = trajectory_basis_and_scale(x_test)
    phys_train = physical_predictions(x_train)
    phys_test = physical_predictions(x_test)

    physical_scores = {name: score_summary(pred, y) for name, pred in phys_train.items()}
    best_physical_name = max(physical_scores, key=lambda n: physical_scores[n]["hit"])
    best_base_name = args.residual_base
    if best_base_name not in phys_train:
        raise ValueError(f"unknown residual base {best_base_name!r}; available examples: {list(phys_train)[:8]}")
    base_train = phys_train[best_base_name]
    base_test = phys_test[best_base_name]
    print(f"[base] residual={best_base_name} {physical_scores[best_base_name]}", flush=True)
    print(f"[base] best_physical={best_physical_name} {physical_scores[best_physical_name]}", flush=True)

    models = make_models(args.profile, args.seed)
    print(f"[train] profile={args.profile} models={list(models)} folds={args.folds}", flush=True)
    oof_abs, fitted, fold_scores = make_oof(f_train, y, base_train, basis_train, scale_train, models, args.folds, args.seed)

    reports: dict[str, Any] = {
        "config": vars(args),
        "model_specs": {
            name: {"weighting": spec.get("weighting"), "target_space": spec.get("target_space", "global")}
            for name, spec in models.items()
        },
        "best_base_name": best_base_name,
        "best_physical_name": best_physical_name,
        "physical_scores": physical_scores,
        "fold_scores": fold_scores,
        "model_scores": {},
        "shrinks": {},
        "blend": {},
    }

    calibrated_oof: dict[str, np.ndarray] = {
        f"physical_{best_base_name}": base_train,
    }
    for name, pred in oof_abs.items():
        raw = score_summary(pred, y)
        correction = pred - base_train
        best_shrink = find_best_shrink(base_train, correction, y)
        reports["model_scores"][name] = raw
        reports["shrinks"][name] = best_shrink
        calibrated_oof[f"{name}_raw"] = pred
        calibrated_oof[f"{name}_shrink"] = base_train + best_shrink["shrink"] * correction
        print(
            f"[oof] {name:16s} raw_hit={raw['hit']:.5f} "
            f"best_hit={best_shrink['hit']:.5f} shrink={best_shrink['shrink']:.3f}",
            flush=True,
        )

    # Keep the blend search focused on calibrated candidates that are not clearly weak.
    blend_candidates = {
        name: pred for name, pred in calibrated_oof.items()
        if score_summary(pred, y)["hit"] >= score_summary(base_train, y)["hit"] - 0.005
    }
    blend = find_best_blend(blend_candidates, y, args.seed)
    reports["blend"] = blend
    print(f"[blend] hit={blend['summary']['hit']:.5f} names={blend['names']}", flush=True)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    model_path = out_dir / "models.pkl"
    with model_path.open("wb") as fp:
        pickle.dump(
            {
                "best_base_name": best_base_name,
                "best_physical_name": best_physical_name,
                "models": fitted,
                "model_specs": reports["model_specs"],
                "shrinks": reports["shrinks"],
                "blend": blend,
                "profile": args.profile,
                "seed": args.seed,
            },
            fp,
        )

    if args.no_submissions:
        print(f"[done] wrote {metrics_path} and {model_path}; skipped submissions", flush=True)
        return

    print("[predict] creating submissions", flush=True)
    submissions: dict[str, np.ndarray] = {
        "submission_physical.csv": base_test,
        "submission_best_physical.csv": phys_test[best_physical_name],
    }
    test_candidate_preds: dict[str, np.ndarray] = {
        f"physical_{best_base_name}": base_test,
        f"physical_{best_physical_name}": phys_test[best_physical_name],
    }
    for name, model_list in fitted.items():
        target_space = models[name].get("target_space", "global")
        raw_pred = predict_with_fold_models(
            model_list, f_test, base_test, basis_test, scale_test, target_space=target_space, shrink=1.0
        )
        shrink = reports["shrinks"][name]["shrink"]
        shrink_pred = predict_with_fold_models(
            model_list, f_test, base_test, basis_test, scale_test, target_space=target_space, shrink=shrink
        )
        submissions[f"submission_{name}_raw.csv"] = raw_pred
        submissions[f"submission_{name}_shrink.csv"] = shrink_pred
        test_candidate_preds[f"{name}_raw"] = raw_pred
        test_candidate_preds[f"{name}_shrink"] = shrink_pred

    # OOF-selected blend projected to test predictions.
    blend_names = blend["names"]
    blend_weights = np.asarray(blend["weights"], dtype=np.float64)
    blend_stack = _prediction_matrix([test_candidate_preds[n] for n in blend_names])
    submissions["submission_blend_hit_optimized.csv"] = np.tensordot(blend_weights, blend_stack, axes=(0, 0))

    # Robust manual variants for 5-per-day submission planning.
    tree_names = [n for n in test_candidate_preds if "extra_trees" in n and n.endswith("_shrink")]
    if tree_names:
        submissions["submission_tree_heavy.csv"] = np.mean([test_candidate_preds[n] for n in tree_names], axis=0)
    ml_shrink_names = [n for n in test_candidate_preds if n.endswith("_shrink") and n != f"physical_{best_base_name}"]
    if ml_shrink_names:
        submissions["submission_ml_shrink_mean.csv"] = np.mean([test_candidate_preds[n] for n in ml_shrink_names], axis=0)
        submissions["submission_physical_ml_50_50.csv"] = 0.5 * base_test + 0.5 * submissions["submission_ml_shrink_mean.csv"]

    for filename, pred in submissions.items():
        write_submission(out_dir / filename, data.test_ids, pred)
        print(f"  wrote {out_dir / filename}", flush=True)

    print(f"[done] metrics={metrics_path} models={model_path}", flush=True)


if __name__ == "__main__":
    main()
