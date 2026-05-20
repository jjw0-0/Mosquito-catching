from pathlib import Path

import numpy as np

from src.data import load_competition_data
from src.candidate_bank import _kalman_ca_predict, build_candidate_bank
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


def test_candidate_bank_shapes_and_finite_values():
    data = load_competition_data(ZIP_PATH)
    bank = build_candidate_bank(data.train_xyz[:8], profile="quick")
    assert bank
    for key in [
        "accel_dense_c0.6",
        "poly_d2_m4",
        "sg_w5_p2_accel_c0.5",
        "turn_rodrigues_g0.5",
    ]:
        assert key in bank
    for pred in bank.values():
        assert pred.shape == (8, 3)
        assert np.isfinite(pred).all()


def test_kalman_candidate_predicts_from_final_observation():
    t = np.arange(11, dtype=np.float64)
    xyz = np.zeros((1, 11, 3), dtype=np.float64)
    xyz[0, :, 0] = 0.01 * t
    pred = _kalman_ca_predict(xyz, q=1e-7, r=1e-7)
    assert pred.shape == (1, 3)
    assert np.isfinite(pred).all()
    assert pred[0, 0] > xyz[0, -1, 0]
