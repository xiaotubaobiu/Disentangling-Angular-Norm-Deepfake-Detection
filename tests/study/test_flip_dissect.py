# tests/study/test_flip_dissect.py
import numpy as np
from deepfake_detection.study.flip_dissect import recover_wdotu_b, score_gamma

def test_gamma_endpoints_recover_raw_and_dir():
    rng = np.random.default_rng(1)
    D = 8
    w = rng.normal(size=D); b = 0.3
    z = rng.normal(size=(50, D)) * 5.0
    norm = np.linalg.norm(z, axis=1)
    u = z / norm[:, None]
    s_raw = z @ w + b
    s_dir = u @ w + b
    wu, bb = recover_wdotu_b(s_raw, s_dir, norm)
    assert np.allclose(wu, u @ w, atol=1e-4)
    assert np.allclose(bb, b, atol=1e-4)
    # gamma=1 -> raw, gamma=0 -> dir
    assert np.allclose(score_gamma(wu, bb, norm, 1.0), s_raw, atol=1e-3)
    assert np.allclose(score_gamma(wu, bb, norm, 0.0), s_dir, atol=1e-3)
