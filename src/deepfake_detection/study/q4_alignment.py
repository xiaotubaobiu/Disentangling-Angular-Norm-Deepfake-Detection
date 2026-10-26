# src/deepfake_detection/study/q4_alignment.py
"""Q4 geometry: chord (ambient) for global/PCA/LOFO, tangent/logmap for
within same-frame-group; cosine alignment; residual after projecting out w."""
from __future__ import annotations
import numpy as np
EPS=1e-12

def _unit(v):
    v=np.asarray(v,float); n=np.linalg.norm(v); return v/n if n>EPS else v

def chord(u_f, u_r):
    return np.asarray(u_f,float)-np.asarray(u_r,float)

def tangent(u_f, u_r):
    u_f=np.asarray(u_f,float); u_r=np.asarray(u_r,float)
    return u_f - np.dot(u_f,u_r)*u_r

def logmap(u_f, u_r):
    u_f=_unit(u_f); u_r=_unit(u_r)
    c=np.clip(np.dot(u_f,u_r),-1.0,1.0); th=np.arccos(c)
    if th<EPS: return np.zeros_like(u_f)
    # Sphere log map at base u_r pointing to u_f: |.| == th (geodesic distance).
    return (th/np.sin(th))*(u_f - c*u_r)

def cosine_between(a,b):
    return float(np.dot(_unit(a),_unit(b)))

def project_out_w(d, w):
    w=_unit(w); d=np.asarray(d,float)
    return d - np.dot(d,w)*w

def mean_chord(D):
    D=np.asarray(D,float)
    return _unit(D.mean(axis=0))
