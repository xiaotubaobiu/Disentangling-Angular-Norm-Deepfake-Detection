"""Regression tests for length-mode scoring (infer_modes.score_rawtrain_modes).

Bug 1: length mode fed the raw feature norm (||h|| ~ 20-50) into
softmax([-||h||, +||h||]), which saturates prob_fake to 1.0 for every frame
and collapses the length-only AUC to 0.5. These tests lock down the fix:
prob_fake must be a NON-saturating, strictly monotonic transform of ||h|| so
that the length-only ranking (hence AUC) is preserved.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from diagnosis.infer_modes import score_rawtrain_modes


def _make_features(norms: np.ndarray, dim: int = 8) -> np.ndarray:
    """Build (N, dim) features whose row L2 norms equal `norms`."""
    base = np.zeros((len(norms), dim), dtype=np.float32)
    base[:, 0] = 1.0  # unit direction
    return base * norms.reshape(-1, 1).astype(np.float32)


def test_length_prob_fake_not_saturated():
    """Realistic norms (20-50) must NOT saturate prob_fake to 1.0."""
    norms = np.linspace(20.0, 50.0, 200).astype(np.float32)
    feats = _make_features(norms)
    weight = np.zeros((2, 8), dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)

    out = score_rawtrain_modes(feats, norms, weight, bias)
    prob = out["length"].prob_fake

    # Not collapsed to a single saturated value.
    assert prob.max() - prob.min() > 0.1, "length prob_fake is saturated"
    assert np.all(prob > 0.0) and np.all(prob < 1.0)


def test_length_prob_fake_monotonic_in_norm():
    """prob_fake must be strictly increasing in ||h|| (rank-preserving)."""
    norms = np.array([21.0, 25.0, 30.0, 44.0, 49.0], dtype=np.float32)
    feats = _make_features(norms)
    weight = np.zeros((2, 8), dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)

    prob = score_rawtrain_modes(feats, norms, weight, bias)["length"].prob_fake
    # Since input norms are ascending, prob_fake must be ascending too.
    assert np.all(np.diff(prob) > 0), "length prob_fake not monotonic in norm"


def test_length_only_auc_recovers_norm_ranking():
    """Length-only AUC must equal the AUC of the raw norm (rank identity)."""
    rng = np.random.default_rng(0)
    # Fakes have larger norms than reals, with overlap.
    real_norms = rng.normal(30.0, 4.0, size=150).astype(np.float32)
    fake_norms = rng.normal(36.0, 4.0, size=150).astype(np.float32)
    norms = np.concatenate([real_norms, fake_norms])
    labels = np.concatenate([np.zeros(150), np.ones(150)])
    feats = _make_features(norms)
    weight = np.zeros((2, 8), dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)

    prob = score_rawtrain_modes(feats, norms, weight, bias)["length"].prob_fake

    auc_prob = roc_auc_score(labels, prob)
    auc_raw_norm = roc_auc_score(labels, norms)
    # Monotonic transform => identical AUC (up to float noise).
    assert abs(auc_prob - auc_raw_norm) < 1e-9
    # And it is a real signal, not 0.5.
    assert auc_prob > 0.7


def test_length_degenerate_constant_norm():
    """Constant norms (sigma=0) must not divide-by-zero; prob_fake = 0.5."""
    norms = np.full(10, 33.0, dtype=np.float32)
    feats = _make_features(norms)
    weight = np.zeros((2, 8), dtype=np.float32)
    bias = np.zeros(2, dtype=np.float32)

    prob = score_rawtrain_modes(feats, norms, weight, bias)["length"].prob_fake
    assert np.allclose(prob, 0.5)
