from pathlib import Path

import numpy as np

from src.data import load_competition_data
from src.features import build_features, physical_predictions

ZIP_PATH = Path('/Users/jjw/Downloads/open.zip')


def test_load_open_zip_and_feature_shapes():
    data = load_competition_data(ZIP_PATH)
    assert data.train_xyz.shape == (10000, 11, 3)
    assert data.test_xyz.shape == (10000, 11, 3)
    assert data.y.shape == (10000, 3)
    feats = build_features(data.train_xyz[:10])
    assert feats.shape[0] == 10
    assert np.isfinite(feats).all()
    preds = physical_predictions(data.train_xyz[:10])
    assert 'accel_c0.5' in preds
    assert preds['accel_c0.5'].shape == (10, 3)
