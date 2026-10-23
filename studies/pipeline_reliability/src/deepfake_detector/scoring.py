"""Pure FP64 scoring, inclusive operating points and ties-aware pair ledgers."""
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score


def sigmoid(x):
    return expit(np.asarray(x, dtype=np.float64))


def video_scores(z):
    z=np.asarray(z,dtype=np.float64)
    if z.ndim!=2 or not np.isfinite(z).all():
        raise ValueError('Expected finite video-by-frame logits')
    return z.mean(axis=1),sigmoid(z).mean(axis=1)


def fpr_threshold(real_scores, weights=None, budget=.05):
    x=np.asarray(real_scores,dtype=np.float64)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all() or not 0<=budget<1:
        raise ValueError('Nonempty finite independent source-real scores required')
    w=np.ones(len(x)) if weights is None else np.asarray(weights,dtype=np.float64)
    if w.shape!=x.shape or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('Invalid source-real weights')
    vals,inv=np.unique(x,return_inverse=True)
    mass=np.bincount(inv,weights=w,minlength=len(vals))
    tail=np.cumsum(mass[::-1])[::-1]/w.sum()
    # Each distinct observed value and its strict successor are candidates.
    ts=np.column_stack([vals,np.nextafter(vals,np.inf)]).ravel()
    rates=np.column_stack([tail,np.r_[tail[1:],0.]]).ravel()
    return float(ts[np.flatnonzero(rates<=budget+1e-14)[0]])


def auc(score,label,weights=None):
    return float(roc_auc_score(label,score,sample_weight=weights))


def pair_transition(before,after,label,weights=None,block=512):
    before=np.asarray(before,dtype=np.float64);after=np.asarray(after,dtype=np.float64)
    y=np.asarray(label);w=np.ones(len(y)) if weights is None else np.asarray(weights,dtype=np.float64)
    real=np.flatnonzero(y==0);fake=np.flatnonzero(y==1)
    if not len(real) or not len(fake):raise ValueError('Pair comparison needs both classes')
    table=np.zeros((3,3),dtype=np.float64)
    for start in range(0,len(fake),block):
        ids=fake[start:start+block]
        a=np.sign(before[ids,None]-before[real]).astype(int)+1
        b=np.sign(after[ids,None]-after[real]).astype(int)+1
        pw=w[ids,None]*w[real]
        table+=np.bincount((a*3+b).ravel(),weights=pw.ravel(),minlength=9).reshape(3,3)
    credit=np.array([0.,.5,1.])
    delta=float((table*(credit[None,:]-credit[:,None])).sum()/table.sum())
    return table,delta
