# src/study_pooling/temperature.py
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar, minimize

def _sigmoid(x): return 1.0/(1.0+np.exp(-np.asarray(x,float)))

def nll(s, y, T=1.0, w=None):
    p = np.clip(_sigmoid(np.asarray(s,float)/max(T,1e-6)), 1e-12, 1-1e-12)
    y = np.asarray(y, float)
    l = -(y*np.log(p)+(1-y)*np.log(1-p))
    return float(np.average(l, weights=w)) if w is not None else float(np.mean(l))

def fit_temperature(s, y, sample_weight=None):
    s=np.asarray(s,float); y=np.asarray(y,float)
    res = minimize_scalar(lambda T: nll(s,y,max(T,1e-6),sample_weight), bounds=(1e-3,100.0), method="bounded")
    return float(res.x)

def fit_platt(s, y, sample_weight=None):
    s=np.asarray(s,float); y=np.asarray(y,float)
    def loss(ab):
        A,B=ab; p=np.clip(_sigmoid(A*s+B),1e-12,1-1e-12)
        l=-(y*np.log(p)+(1-y)*np.log(1-p))
        return np.average(l,weights=sample_weight) if sample_weight is not None else np.mean(l)
    res = minimize(loss, x0=np.array([1.0,0.0]), method="Nelder-Mead")
    A,B=float(res.x[0]),float(res.x[1])
    if A<=0:  # enforce monotonicity; fallback to A=1e-3
        A=1e-3
    return A,B

def apply_T(s,T): return _sigmoid(np.asarray(s,float)/max(T,1e-6))

def ece(p, y, n_bins=15):
    p=np.asarray(p,float); y=np.asarray(y,float)
    bins=np.linspace(0,1,n_bins+1); out=0.0; n=len(y)
    for i in range(n_bins):
        m=(p>=bins[i])&(p<bins[i+1])
        if m.sum()==0: continue
        out += (m.sum()/n)*abs(p[m].mean()-y[m].mean())
    return float(out)

def brier(p,y): p=np.asarray(p,float); y=np.asarray(y,float); return float(np.mean((p-y)**2))
