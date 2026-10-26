import numpy as np

from deepfake_detection.study.q4_extract import consistency_check


def test_consistency_pass():
    rng = np.random.default_rng(0)
    D = 16
    w = rng.normal(size=D)
    b = 0.2
    z = rng.normal(size=(10, D)) * 5
    norm = np.linalg.norm(z, axis=1)
    raw = z @ w + b
    direction = (z / norm[:, None]) @ w + b
    assert consistency_check(z, norm, raw, direction, w, b, tol=1e-3)


def test_consistency_fail_wrong_layer():
    rng = np.random.default_rng(0)
    D = 16
    w = rng.normal(size=D)
    b = 0.2
    z = rng.normal(size=(10, D)) * 5
    norm = np.linalg.norm(z, axis=1)
    raw = z @ w + b
    direction = (z / norm[:, None]) @ w + b
    other = rng.normal(size=(10, D)) * 5
    assert not consistency_check(other, norm, raw, direction, w, b, tol=1e-3)
