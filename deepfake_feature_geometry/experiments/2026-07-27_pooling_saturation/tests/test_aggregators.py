# tests/study_pooling/test_aggregators.py
import numpy as np
from study_pooling.aggregators import aggregate, jensen_gap, NAMES

def _sigmoid(x): return 1/(1+np.exp(-x))

def test_mean_logit_equals_sigmoid_of_mean_in_auc_rank():
    # AUC is rank-based and sigmoid is strictly increasing ->
    # mean_logit and sigmoid(mean_logit) give identical video ranking.
    s = np.array([[-2.,1.],[0.,0.],[3.,-1.]])
    ml = [aggregate("mean_logit", row) for row in s]
    sml = [aggregate("sigmoid_mean_logit", row) for row in s]
    order_ml = np.argsort(ml); order_sml = np.argsort(sml)
    assert np.array_equal(order_ml, order_sml)

def test_jensen_gap_sign():
    # non-constant logits -> Jensen gap != 0 (sigmoid concave-then-convex; for
    # spread logits mean(sig) != sig(mean)). Symmetric inputs like [10,-10]
    # cancel exactly (mean(s)=0, sigmoid(0)=0.5=(sigmoid(10)+sigmoid(-10))/2),
    # so use an asymmetric spread.
    s = np.array([5., 0.])
    j = jensen_gap(s)
    assert abs(j) > 1e-6

def test_clip_prob_monotone_in_c():
    # Asymmetric logits so clipping with small c actually changes the mean
    # sigmoid (symmetric inputs like [-5,5] would give 0.5 for any c by the
    # sigmoid's odd-symmetry, hiding the effect).
    s = np.array([10., 0.])
    full = aggregate("clip_prob", s, c=100.0)
    clipped = aggregate("clip_prob", s, c=2.0)
    assert abs(full - (0.5*_sigmoid(10.0) + 0.5*_sigmoid(0.0))) < 1e-9
    assert full != clipped  # clipping with small c changes the value vs large c

def test_names_complete():
    for n in ["mean_prob","mean_logit","sigmoid_mean_logit","median_logit","clip_prob","trimmed_logit","topk_logit"]:
        assert n in NAMES
