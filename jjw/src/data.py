from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

EXPECTED_TIMESTEPS = np.arange(-400, 1, 40, dtype=int)
COORD_COLUMNS = ["x", "y", "z"]
SERIES_COLUMNS = ["timestep_ms", *COORD_COLUMNS]


@dataclass(frozen=True)
class CompetitionData:
    train_ids: np.ndarray
    train_xyz: np.ndarray  # (n_train, 11, 3)
    y: np.ndarray          # (n_train, 3)
    test_ids: np.ndarray
    test_xyz: np.ndarray   # (n_test, 11, 3)
    sample_submission: pd.DataFrame


def _read_series(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    with zf.open(name) as fp:
        df = pd.read_csv(fp)
    if list(df.columns) != SERIES_COLUMNS:
        raise ValueError(f"{name}: expected columns {SERIES_COLUMNS}, got {list(df.columns)}")
    if len(df) != len(EXPECTED_TIMESTEPS):
        raise ValueError(f"{name}: expected 11 rows, got {len(df)}")
    if not np.array_equal(df["timestep_ms"].to_numpy(dtype=int), EXPECTED_TIMESTEPS):
        raise ValueError(f"{name}: unexpected timestep_ms values")
    xyz = df[COORD_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError(f"{name}: contains non-finite coordinates")
    return xyz


def _read_many_series(zf: zipfile.ZipFile, prefix: str, ids: np.ndarray) -> np.ndarray:
    arr = np.empty((len(ids), len(EXPECTED_TIMESTEPS), 3), dtype=np.float64)
    for i, sample_id in enumerate(ids):
        arr[i] = _read_series(zf, f"{prefix}/{sample_id}.csv")
    return arr


def load_competition_data(zip_path: str | Path) -> CompetitionData:
    """Load and validate DACON open.zip without extracting it."""
    zip_path = Path(zip_path).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        required = {"train_labels.csv", "sample_submission.csv"}
        missing = required - names
        if missing:
            raise ValueError(f"missing files in zip: {sorted(missing)}")

        with zf.open("train_labels.csv") as fp:
            labels = pd.read_csv(fp)
        if list(labels.columns) != ["id", *COORD_COLUMNS]:
            raise ValueError(f"train_labels.csv columns mismatch: {list(labels.columns)}")
        train_ids = labels["id"].astype(str).to_numpy()
        if len(train_ids) != 10000:
            raise ValueError(f"expected 10000 train labels, got {len(train_ids)}")
        y = labels[COORD_COLUMNS].to_numpy(dtype=np.float64)
        if not np.isfinite(y).all():
            raise ValueError("train labels contain non-finite coordinates")

        with zf.open("sample_submission.csv") as fp:
            sample_submission = pd.read_csv(fp)
        if list(sample_submission.columns) != ["id", *COORD_COLUMNS]:
            raise ValueError(f"sample_submission.csv columns mismatch: {list(sample_submission.columns)}")
        test_ids = sample_submission["id"].astype(str).to_numpy()
        if len(test_ids) != 10000:
            raise ValueError(f"expected 10000 test rows, got {len(test_ids)}")

        expected_train = {f"train/{sid}.csv" for sid in train_ids}
        expected_test = {f"test/{sid}.csv" for sid in test_ids}
        missing_series = (expected_train | expected_test) - names
        if missing_series:
            preview = sorted(missing_series)[:5]
            raise ValueError(f"missing series files: {preview} ... total={len(missing_series)}")

        train_xyz = _read_many_series(zf, "train", train_ids)
        test_xyz = _read_many_series(zf, "test", test_ids)

    return CompetitionData(
        train_ids=train_ids,
        train_xyz=train_xyz,
        y=y,
        test_ids=test_ids,
        test_xyz=test_xyz,
        sample_submission=sample_submission,
    )


def write_submission(path: str | Path, ids: np.ndarray, pred: np.ndarray) -> None:
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != (len(ids), 3):
        raise ValueError(f"submission prediction shape mismatch: {pred.shape} vs ids={len(ids)}")
    if not np.isfinite(pred).all():
        raise ValueError("submission prediction contains non-finite values")
    out = pd.DataFrame({"id": ids, "x": pred[:, 0], "y": pred[:, 1], "z": pred[:, 2]})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
