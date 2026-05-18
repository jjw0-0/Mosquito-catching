from __future__ import annotations

import argparse
from pathlib import Path
import pickle

import numpy as np

from .data import load_competition_data, write_submission
from .features import build_features, physical_predictions
from .train import trajectory_basis_and_scale
from .train import predict_with_fold_models, _prediction_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recreate DACON submissions from trained artifacts.")
    parser.add_argument("--zip-path", default="/Users/jjw/Downloads/open.zip")
    parser.add_argument("--model-path", default="outputs/models.pkl")
    parser.add_argument("--out-dir", default="outputs/recreated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_competition_data(args.zip_path)
    f_test = build_features(data.test_xyz)
    basis_test, scale_test = trajectory_basis_and_scale(data.test_xyz)
    phys_test = physical_predictions(data.test_xyz)

    with Path(args.model_path).open("rb") as fp:
        artifact = pickle.load(fp)

    base_name = artifact["best_base_name"]
    best_physical_name = artifact.get("best_physical_name", base_name)
    base_test = phys_test[base_name]
    best_physical_test = phys_test[best_physical_name]
    candidate_preds = {
        f"physical_{base_name}": base_test,
        f"physical_{best_physical_name}": best_physical_test,
    }
    write_submission(out_dir / "submission_physical.csv", data.test_ids, base_test)
    write_submission(out_dir / "submission_best_physical.csv", data.test_ids, best_physical_test)

    model_specs = artifact.get("model_specs", {})
    for name, models in artifact["models"].items():
        target_space = model_specs.get(name, {}).get("target_space", "global")
        raw = predict_with_fold_models(
            models, f_test, base_test, basis_test, scale_test, target_space=target_space, shrink=1.0
        )
        shrink = artifact["shrinks"][name]["shrink"]
        shrunk = predict_with_fold_models(
            models, f_test, base_test, basis_test, scale_test, target_space=target_space, shrink=shrink
        )
        candidate_preds[f"{name}_raw"] = raw
        candidate_preds[f"{name}_shrink"] = shrunk
        write_submission(out_dir / f"submission_{name}_raw.csv", data.test_ids, raw)
        write_submission(out_dir / f"submission_{name}_shrink.csv", data.test_ids, shrunk)

    blend = artifact["blend"]
    weights = np.asarray(blend["weights"], dtype=np.float64)
    stack = _prediction_matrix([candidate_preds[n] for n in blend["names"]])
    blend_pred = np.tensordot(weights, stack, axes=(0, 0))
    write_submission(out_dir / "submission_blend_hit_optimized.csv", data.test_ids, blend_pred)
    print(f"wrote submissions to {out_dir}")


if __name__ == "__main__":
    main()
