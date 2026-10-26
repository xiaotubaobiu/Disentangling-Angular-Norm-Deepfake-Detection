# tests/study_pooling/test_gamma.py
import numpy as np
from study_pooling.gamma import score_gamma, recover_wdotu

def test_gamma_endpoints():
    rng=np.random.default_rng(0); D=6
    w=rng.normal(size=D); b=0.4
    z=rng.normal(size=(40,D))*5; norm=np.linalg.norm(z,axis=1); u=z/norm[:,None]
    s_raw=z@w+b; s_dir=u@w+b
    wu,bb=recover_wdotu(s_raw,s_dir,norm)
    assert np.allclose(score_gamma(wu,bb,norm,1.0), s_raw, atol=1e-3)
    assert np.allclose(score_gamma(wu,bb,norm,0.0), s_dir, atol=1e-3)
