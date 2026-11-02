import numpy as np
from deepfake_detector import scoring as s


def test_probability_mean_is_not_sigmoid_of_mean_logit():
    assert hasattr(s,'video_scores'), 'video scoring not implemented'
    z=np.array([[-20.,5.],[-1.,-1.]])
    logit,prob=s.video_scores(z)
    assert np.allclose(logit,[-7.5,-1.])
    assert prob[0]>prob[1]
    assert not np.allclose(prob,s.sigmoid(logit))


def test_source_threshold_is_minimal_feasible_with_inclusive_ties():
    assert hasattr(s,'fpr_threshold'), 'FPR threshold not implemented'
    x=np.array([0.,1.,1.,2.])
    t=s.fpr_threshold(x,budget=.25)
    assert t==np.nextafter(1.,np.inf)
    assert np.mean(x>=t)==.25
    assert s.fpr_threshold(x,budget=0)==np.nextafter(2.,np.inf)


def test_affine_rescaling_preserves_frozen_predictions():
    assert hasattr(s,'fpr_threshold'), 'FPR threshold not implemented'
    x=np.arange(20,dtype=float);t=s.fpr_threshold(x)
    assert np.array_equal(x>=t, (4*x+3)>=(4*t+3))


def test_pair_ledger_reconstructs_auc_change_with_ties():
    assert hasattr(s,'pair_transition'), 'ties-aware pair ledger not implemented'
    before=np.array([0.,2.,1.,2.]);after=np.array([0.,1.,1.,3.]);y=np.array([0,0,1,1])
    table,delta=s.pair_transition(before,after,y)
    assert table.shape==(3,3) and table.sum()==4
    assert np.isclose(delta,s.auc(after,y)-s.auc(before,y))
