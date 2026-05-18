import numpy as np

from src.metrics import r_hit, distances


def test_r_hit_includes_one_centimeter_boundary():
    true = np.zeros((3, 3))
    pred = np.array([
        [0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0],
        [0.010001, 0.0, 0.0],
    ])
    assert np.allclose(distances(pred, true), [0.0, 0.01, 0.010001])
    assert r_hit(pred, true) == 2 / 3
