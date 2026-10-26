# tests/study/test_q4_alignment.py
import numpy as np
from deepfake_detection.study.q4_alignment import (chord, tangent, logmap,
    cosine_between, project_out_w, mean_chord)

def test_displacements_and_cosine():
    u_r=np.array([1.,0.,0.]); u_f=np.array([0.6,0.8,0.])
    assert np.allclose(chord(u_f,u_r), np.array([-0.4,0.8,0.]))
    t=tangent(u_f,u_r); assert abs(np.dot(t,u_r))<1e-9   # tangent ⊥ u_r
    lg=logmap(u_f,u_r); theta=np.arccos(0.6)
    assert np.allclose(lg/np.linalg.norm(lg), t/np.linalg.norm(t))   # same direction as tangent
    assert abs(np.linalg.norm(lg)-theta)<1e-6                       # magnitude == theta
    assert abs(cosine_between(np.array([1.,0]),np.array([1.,0]))-1.0)<1e-9

def test_project_out_w():
    d=np.array([1.,1.]); w=np.array([1.,0.])
    r=project_out_w(d,w)
    assert abs(r[0])<1e-9 and abs(r[1]-1.0)<1e-9

def test_mean_chord_unit():
    D=np.array([[1.,0.,0.],[0.,1.,0.]])
    m=mean_chord(D)
    assert abs(np.linalg.norm(m)-1.0)<1e-9
