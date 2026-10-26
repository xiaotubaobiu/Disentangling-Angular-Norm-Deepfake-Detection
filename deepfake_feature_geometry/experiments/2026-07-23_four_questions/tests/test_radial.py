# tests/study/test_radial.py
import numpy as np, pandas as pd
from study.radial import delta_d, rtg_stg, method_level_corr

def test_delta_and_rtg():
    d_ff = delta_d(q_fake=np.array([2.,3.]), q_real=np.array([0.,0.]))
    assert d_ff > 0
    d_cdf = delta_d(np.array([0.,0.]), np.array([2.,3.]))
    out = rtg_stg(d_ff, d_cdf)
    assert abs(out["RTG"]-abs(d_cdf-d_ff))<1e-9
    assert abs(out["STG"]-(d_cdf-d_ff))<1e-9
    assert out["sign_flip"]

def test_method_level_corr_runs():
    df = pd.DataFrame({"method":list("abcd")*2, "RTG":np.random.rand(8), "gain":np.random.rand(8)})
    r = method_level_corr(df, "RTG", "gain")
    assert "pearson" in r and "spearman" in r
