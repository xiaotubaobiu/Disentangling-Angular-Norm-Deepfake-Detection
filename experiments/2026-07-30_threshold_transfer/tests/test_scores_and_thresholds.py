"""Score definition and threshold-rule tests.

1. the study's `signed_logit` really is the two-class fake probability in logit
   space: `softmax([real, fake])[fake] == sigmoid(fake - real)`;
2. `t_BA` and `t_equal` are two different definitions, not synonyms.
"""
from __future__ import annotations

import numpy as np
import pytest

from study_threshold.thresholds import (
    best_ba_threshold,
    confusion,
    equal_error_threshold,
    fpr_threshold,
    rate_curves,
)


def _softmax(a):
    a = np.asarray(a, float)
    e = np.exp(a - a.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, float)))


def test_two_class_softmax_fake_prob_equals_sigmoid_signed_logit():
    rng = np.random.default_rng(0)
    logits = rng.normal(0, 5, size=(2000, 2))          # [real_logit, fake_logit]
    p_fake_softmax = _softmax(logits)[:, 1]
    p_fake_sigmoid = _sigmoid(logits[:, 1] - logits[:, 0])
    assert np.allclose(p_fake_softmax, p_fake_sigmoid, atol=1e-12)


def test_signed_logit_is_monotone_in_fake_probability():
    s = np.linspace(-20, 20, 101)
    p = _sigmoid(s)
    assert np.all(np.diff(p) > 0)      # ranking by s == ranking by p
    assert abs(_sigmoid(0.0) - 0.5) < 1e-12


def test_t_ba_and_t_equal_are_different_definitions():
    """Asymmetric variances: the BA optimum is not the equal-error point."""
    rng = np.random.default_rng(1)
    fake = rng.normal(3.0, 3.0, 400)     # wide fake class
    real = rng.normal(0.0, 0.4, 400)     # narrow real class
    scores = np.concatenate([fake, real])
    labels = np.concatenate([np.ones(400, int), np.zeros(400, int)])

    t_ba = best_ba_threshold(scores, labels)
    t_eq = equal_error_threshold(scores, labels)
    assert t_ba != t_eq, "these two rules must not collapse into one number"

    ba_at_ba = confusion(scores, labels, t_ba)["balanced_accuracy"]
    ba_at_eq = confusion(scores, labels, t_eq)["balanced_accuracy"]
    assert ba_at_ba >= ba_at_eq                      # t_BA maximises BA
    c_eq = confusion(scores, labels, t_eq)
    c_ba = confusion(scores, labels, t_ba)
    assert abs(c_eq["tpr"] - c_eq["tnr"]) <= abs(c_ba["tpr"] - c_ba["tnr"])  # t_equal balances


def test_fpr_threshold_respects_the_budget():
    rng = np.random.default_rng(2)
    scores = np.concatenate([rng.normal(2, 1, 500), rng.normal(-1, 1, 500)])
    labels = np.concatenate([np.ones(500, int), np.zeros(500, int)])
    for target in (0.01, 0.05):
        t = fpr_threshold(scores, labels, target)
        assert confusion(scores, labels, t)["fpr"] <= target + 1e-12
    assert fpr_threshold(scores, labels, 0.05) <= fpr_threshold(scores, labels, 0.01)


def test_rate_curves_match_the_explicit_confusion_scan():
    rng = np.random.default_rng(3)
    scores = np.concatenate([rng.normal(1, 1, 60), rng.normal(-1, 1, 40)])
    labels = np.concatenate([np.ones(60, int), np.zeros(40, int)])
    cands, tpr, tnr = rate_curves(scores, labels)
    for i in rng.choice(len(cands), size=25, replace=False):
        c = confusion(scores, labels, cands[i])
        assert c["tpr"] == pytest.approx(tpr[i])
        assert c["tnr"] == pytest.approx(tnr[i])
