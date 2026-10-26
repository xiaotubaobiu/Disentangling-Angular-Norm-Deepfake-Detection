# tests/study_pooling/test_temperature.py
import numpy as np
from deepfake_detection.study_pooling.temperature import fit_temperature, fit_platt, nll, ece, brier, apply_T

def test_fit_temp_near_one_on_well_calibrated():
    rng=np.random.default_rng(0)
    N=4000
    # Properly calibrated: y~Bernoulli(p), s=logit(p) -> Bayes-optimal T~1.
    p=rng.uniform(0.02,0.98,N)
    y=(rng.uniform(size=N)<p).astype(int)
    s=np.log(p/(1-p))
    T=fit_temperature(s,y)
    assert 0.7 < T < 1.4, T

def test_platt_positive_A():
    rng=np.random.default_rng(1); y=rng.integers(0,2,1000)
    s=np.where(y==1, rng.normal(2,1,1000), rng.normal(-2,1,1000))
    A,B=fit_platt(s,y)
    assert A>0

def test_apply_T_and_metrics():
    s=np.array([-2.,2.]); y=np.array([0,1])
    p=apply_T(s,1.0)
    assert abs(p[1]-1/(1+np.exp(-2)))<1e-9
    assert 0.0<=ece(p,y,5)<=1.0
    assert 0.0<=brier(p,y)<=1.0
    assert nll(s,y)>0
