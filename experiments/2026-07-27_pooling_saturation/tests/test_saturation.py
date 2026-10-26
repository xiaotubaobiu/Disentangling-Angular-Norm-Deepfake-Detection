import numpy as np
from study_pooling.saturation import frame_stats, sat_ratios, sigmoid_grad_stats

def test_frame_stats():
    s = np.array([-5.,0.,5.])
    st = frame_stats(s, 1/(1+np.exp(-s)))
    assert abs(st["mean"]-0.0)<1e-9
    assert "q99" in st and "mean_abs" in st

def test_sat_ratios():
    p = np.array([0.001, 0.5, 0.999])
    r = sat_ratios(p, [0.01,0.05,0.10])
    assert r["sat_001"]==2/3 and r["sat_005"]==2/3

def test_sigmoid_grad_zero_in_saturation():
    p = np.array([0.999, 0.999])
    g = sigmoid_grad_stats(p)
    assert g["mean"] < 1e-2
