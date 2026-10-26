# tests/study/test_auc_pairs.py
import numpy as np
from deepfake_detection.study.auc_pairs import (
    auc_mannwhitney, flip_counts, identity_check)

def test_auc_known():
    assert auc_mannwhitney(np.array([2.,3.]), np.array([0.,1.])) == 1.0

def test_auc_with_ties():
    # f=[1,2] r=[0,1]: pairs (1,0)win (1,1)tie (2,0)win (2,1)win
    # -> (3 wins + 0.5 tie)/4 = 0.875
    assert abs(auc_mannwhitney(np.array([1.,2.]), np.array([0.,1.])) - 0.875) < 1e-9

def test_identity_exact_no_ties():
    rng = np.random.default_rng(0)
    raw_f, raw_r = rng.normal(0.6,1,5000), rng.normal(0.4,1,5000)
    dir_f, dir_r = rng.normal(0.6,1,5000), rng.normal(0.4,1,5000)
    auc_raw = auc_mannwhitney(raw_f, raw_r)
    auc_dir = auc_mannwhitney(dir_f, dir_r)
    cc,ww,h,b = flip_counts(raw_f, raw_r, dir_f, dir_r)
    err = identity_check(auc_raw, auc_dir, h, b)
    assert err < 1e-6, err   # same-pair exact counting -> machine epsilon

def test_flip_counts_signs():
    cc,ww,h,b = flip_counts(np.array([1.]), np.array([0.]),
                            np.array([1.]), np.array([0.]))
    assert (cc,ww,h,b)==(1,0,0,0)
