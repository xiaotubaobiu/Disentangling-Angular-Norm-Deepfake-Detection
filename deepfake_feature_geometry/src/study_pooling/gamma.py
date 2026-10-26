# src/study_pooling/gamma.py
from __future__ import annotations
import numpy as np
EPS=1e-6

def recover_wdotu(s_raw, s_dir, norm):
    s_raw=np.asarray(s_raw,float); s_dir=np.asarray(s_dir,float); norm=np.asarray(norm,float)
    denom=norm-1.0; safe=np.abs(denom)>EPS
    wu=np.where(safe,(s_raw-s_dir)/np.where(safe,denom,1.0),0.0)
    return wu, s_dir-wu

def score_gamma(w_dot_u, b, norm, gamma):
    norm=np.asarray(norm,float)
    return np.power(norm,gamma)*np.asarray(w_dot_u)+b
