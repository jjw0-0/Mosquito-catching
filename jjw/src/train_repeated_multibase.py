from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.multioutput import MultiOutputRegressor

from .data import load_competition_data, write_submission
from .features import build_features, physical_predictions
from .metrics import score_summary
from .train import (
    correction_from_prediction,
    find_best_blend,
    find_best_shrink,
    predict_with_fold_models,
    residual_target,
    sample_weight_for,
    trajectory_basis_and_scale,
)


BASES = [
    ("c0p46", "accel_c0.46"),
    ("c0p5", "accel_c0.5"),
    ("c0p58", "accel_c0.58"),
    ("c0p6", "accel_c0.6"),
]


def _read_submission(path: str | Path) -> np.ndarray:
    return pd.read_csv(path)[["x", "y", "z"]].to_numpy(dtype=np.float64)


def _require_existing(paths: list[Path], purpose: str) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifact(s) for {purpose}: {missing}. "
            "Regenerate the previous OOF/submission files or run with --no-include-previous."
        )


def _make_model(seed: int) -> MultiOutputRegressor:
    return MultiOutputRegressor(
        HistGradientBoostingRegressor(
            max_iter=240,
            learning_rate=0.045,
            l2_regularization=0.015,
            max_leaf_nodes=17,
            min_samples_leaf=20,
            random_state=seed + 24,
        )
    )


def _train_seed(
    features: np.ndarray,
    y: np.ndarray,
    base: np.ndarray,
    basis: np.ndarray,
    scale: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, list[Any], dict[str, Any]]:
    model_spec = _make_model(seed)
    target = residual_target(y - base, basis, scale, "local")
    weights = sample_weight_for("trap", base, y)
    kfold = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(y)
    models = []
    fold_hits = []
    for fold, (tr_idx, va_idx) in enumerate(kfold.split(features), start=1):
        model = clone(model_spec)
        model.fit(features[tr_idx], target[tr_idx], sample_weight=weights[tr_idx])
        corr = correction_from_prediction(model.predict(features[va_idx]), basis[va_idx], scale[va_idx], "local")
        oof[va_idx] = base[va_idx] + corr
        models.append(model)
        fold_hits.append(score_summary(oof[va_idx], y[va_idx])["hit"])
    return oof, models, {"seed": seed, "raw": score_summary(oof, y), "fold_hits": fold_hits}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repeated 5-fold multibase local-trap bagging.")
    parser.add_argument("--zip-path", default="/Users/jjw/Downloads/open.zip")
    parser.add_argument("--out-dir", default="outputs_repeated_multibase")
    parser.add_argument("--seeds", default="20260518,20260519,20260520,20260521,20260522")
    parser.add_argument("--include-previous", dest="include_previous", action="store_true", default=True)
    parser.add_argument("--no-include-previous", dest="include_previous", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    data = load_competition_data(args.zip_path)
    f_train = build_features(data.train_xyz)
    f_test = build_features(data.test_xyz)
    basis_train, scale_train = trajectory_basis_and_scale(data.train_xyz)
    basis_test, scale_test = trajectory_basis_and_scale(data.test_xyz)
    phys_train = physical_predictions(data.train_xyz)
    phys_test = physical_predictions(data.test_xyz)

    reports: dict[str, Any] = {"seeds": seeds, "bases": {}, "blend": None}
    oof_best: dict[str, np.ndarray] = {}
    test_best: dict[str, np.ndarray] = {}

    for label, base_name in BASES:
        print(f"[base] {label} {base_name}", flush=True)
        base = phys_train[base_name]
        base_test = phys_test[base_name]
        seed_oofs = []
        seed_tests = []
        seed_reports = []
        for seed in seeds:
            print(f"  [seed] {seed}", flush=True)
            oof, models, seed_report = _train_seed(f_train, data.y, base, basis_train, scale_train, seed)
            test_raw = predict_with_fold_models(models, f_test, base_test, basis_test, scale_test, target_space="local")
            seed_oofs.append(oof)
            seed_tests.append(test_raw)
            seed_reports.append(seed_report)
            print(f"    raw_hit={seed_report['raw']['hit']:.5f}", flush=True)

        avg_oof = np.mean(seed_oofs, axis=0)
        avg_test = np.mean(seed_tests, axis=0)
        best = find_best_shrink(base, avg_oof - base, data.y)
        pred_oof = base + best["shrink"] * (avg_oof - base)
        pred_test = base_test + best["shrink"] * (avg_test - base_test)
        oof_best[f"repeated_{label}"] = pred_oof
        test_best[f"repeated_{label}"] = pred_test
        write_submission(out_dir / f"submission_repeated_local_trap_{label}.csv", data.test_ids, pred_test)
        reports["bases"][label] = {
            "avg_raw": score_summary(avg_oof, data.y),
            "best": best,
            "seed_reports": seed_reports,
        }
        print(f"  [best] hit={best['hit']:.5f} shrink={best['shrink']:.3f}", flush=True)

    candidates = dict(oof_best)
    test_candidates = dict(test_best)
    if args.include_previous:
        previous_paths = [
            Path("outputs/oof_multibase_cache.npz"),
            Path("outputs/submission_multibase_local_trap_blend.csv"),
            Path("outputs_gpu_seq_c0p58/oof_cache.npz"),
            Path("outputs_gpu_seq_c0p58/submission_gpu_seq_raw.csv"),
        ]
        if all(path.exists() for path in previous_paths):
            mb = np.load(previous_paths[0])["multibase_report_opt"]
            mb_test = _read_submission(previous_paths[1])
            gpu = np.load(previous_paths[2])["gpu_seq"]
            gpu_test = _read_submission(previous_paths[3])
            for w in (0.0, 0.3, 0.45, 0.6):
                name = f"old_mb_gpu_w{w:g}"
                candidates[name] = (1.0 - w) * mb + w * gpu
                test_candidates[name] = (1.0 - w) * mb_test + w * gpu_test
        elif any(path.exists() for path in previous_paths):
            _require_existing(previous_paths, "previous multibase/GPU blend candidates")
        else:
            print("[include_previous] no previous artifacts found; blending repeated candidates only", flush=True)

    blend = find_best_blend(candidates, data.y, seed=20260520)
    reports["blend"] = blend
    blend_pred = np.tensordot(
        np.asarray(blend["weights"], dtype=np.float64),
        np.stack([test_candidates[n] for n in blend["names"]]),
        axes=(0, 0),
    )
    write_submission(out_dir / "submission_repeated_multibase_blend.csv", data.test_ids, blend_pred)
    np.savez_compressed(out_dir / "oof_repeated_multibase.npz", y=data.y, **candidates)
    (out_dir / "repeated_multibase_report.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[blend] hit={blend['summary']['hit']:.5f}", flush=True)


if __name__ == "__main__":
    main()
