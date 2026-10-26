"""Four inference modes derived from raw features + classifier weight/bias.

Per spec section III:

rawtrain checkpoints evaluate under:
  - raw:        features -> classifier (uses both direction and length).
  - direction:  features L2-normalized -> same classifier (uses direction only).
  - length:     feature L2 norm used directly as the fake score. prob_fake is
                a non-saturating logistic of the standardized norm (NOT
                softmax([-||h||, +||h||]), which saturates to 1.0 and would
                collapse length-only AUC to 0.5).

normtrain checkpoints evaluate under:
  - normtrain_norm: features L2-normalized -> classifier (the trained mode).

For rawtrain checkpoints we reuse the SAME classifier weight/bias for raw and
direction — the only difference is whether we pre-normalize features. This is
exactly the test-time L2 normalization studied in the existing 30-model
analysis. We deliberately do NOT swap in a separately-trained direction head
(that is MIDT, out of scope here).

All math is in float32 on CPU to keep the per-frame Parquet deterministic and
GPU-free for downstream pandas work. We use scipy/numpy softmax (numerically
stable).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.norm_correction import apply_partial_norm


# Canonical mode names used throughout the diagnostic pipeline.
RAWTRAIN_MODES = ("raw", "direction", "length")
NORMTRAIN_MODES = ("normtrain_norm",)
EPS = 1e-12


@dataclass
class ModeScores:
    """Per-frame scores for one inference mode, aligned with FrameFeatures."""

    mode: str
    real_logit: np.ndarray       # (N,) float32
    fake_logit: np.ndarray       # (N,) float32
    signed_logit: np.ndarray     # (N,) = fake_logit - real_logit
    prob_fake: np.ndarray        # (N,) softmax(logit)[:, 1]


def _softmax_last_axis(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax along the last axis for a (N, 2) array."""
    m = np.max(logits, axis=-1, keepdims=True)
    e = np.exp(logits - m)
    return e / np.sum(e, axis=-1, keepdims=True)


def _scores_from_two_class_logits(logits: np.ndarray) -> ModeScores:
    """Convert (N, 2) logits to ModeScores (mode is set by caller afterwards).

    real_logit = logits[:, 0]; fake_logit = logits[:, 1].
    """
    real = logits[:, 0].astype(np.float32)
    fake = logits[:, 1].astype(np.float32)
    prob_fake = _softmax_last_axis(logits)[:, 1].astype(np.float32)
    return ModeScores(
        mode="",  # filled in by caller
        real_logit=real,
        fake_logit=fake,
        signed_logit=(fake - real).astype(np.float32),
        prob_fake=prob_fake,
    )


def _linear_logits(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """features @ W.T + b for a torch.nn.Linear-style weight (out, in) + bias (out,)."""
    return features.astype(np.float32) @ weight.astype(np.float32).T + bias.astype(np.float32)


def score_rawtrain_modes(
    features: np.ndarray,
    feature_norms: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> dict[str, ModeScores]:
    """Score a rawtrain checkpoint under raw / direction / length.

    Args:
        features: (N, D) raw backbone features.
        feature_norms: (N,) ||features|| (precomputed; used by length mode).
        weight: (2, D) classifier.weight.
        bias: (2,) classifier.bias.

    Returns:
        {"raw": ModeScores, "direction": ModeScores, "length": ModeScores}.
    """
    out: dict[str, ModeScores] = {}

    # raw: features @ W.T + b
    raw_logits = _linear_logits(features, weight, bias)
    raw_scores = _scores_from_two_class_logits(raw_logits)
    raw_scores.mode = "raw"
    out["raw"] = raw_scores

    # direction: L2-normalized features (alpha=1.0) -> same classifier
    feat_norm = apply_partial_norm(
        _to_tensor(features), alpha=1.0
    ).cpu().numpy()
    dir_logits = _linear_logits(feat_norm, weight, bias)
    dir_scores = _scores_from_two_class_logits(dir_logits)
    dir_scores.mode = "direction"
    out["direction"] = dir_scores

    # length: the feature L2 norm ||h|| is the intrinsic "fake score".
    #
    # We keep real_logit/fake_logit = (-||h||, +||h||) for diagnostics
    # (signed_logit = 2||h|| stays monotonic in the norm), but we must NOT
    # derive prob_fake from softmax([-||h||, +||h||]).  With ||h|| typically
    # in [20, 50], softmax([-n, +n])[1] = 1/(1+e^{-2n}) saturates to 1.0 for
    # every frame, so all prob_fake become 1.0, every video score becomes 1.0,
    # and the length-only AUC collapses to 0.5 (rank information destroyed).
    #
    # AUC and the mean-pool video aggregation only need a *strictly monotonic*,
    # non-saturating transform of ||h||.  We standardize the norm (z-score over
    # the current checkpoint/domain frame set) and pass it through a logistic
    # sigmoid: z in ~[-3, 3] keeps prob_fake in ~[0.05, 0.95] with no
    # saturation, and monotonicity guarantees frame-level length AUC equals the
    # AUC of the raw norm.  This restores the spec's "length-only" evaluation.
    length_score = feature_norms.astype(np.float32)
    length_logits = np.stack([-length_score, length_score], axis=1)
    length_scores = _scores_from_two_class_logits(length_logits)
    sigma = float(length_score.std())
    if sigma < EPS:
        z = np.zeros_like(length_score)
    else:
        z = (length_score - float(length_score.mean())) / sigma
    length_scores.prob_fake = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    length_scores.mode = "length"
    out["length"] = length_scores

    return out


def score_normtrain_mode(
    features: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
) -> dict[str, ModeScores]:
    """Score a normtrain checkpoint under its trained (normalized) mode.

    The normtrain classifier was trained on L2-normalized features, so the
    faithful inference is: features -> L2-normalize -> classifier.
    """
    feat_norm = apply_partial_norm(_to_tensor(features), alpha=1.0).cpu().numpy()
    logits = _linear_logits(feat_norm, weight, bias)
    scores = _scores_from_two_class_logits(logits)
    scores.mode = "normtrain_norm"
    return {"normtrain_norm": scores}


def modes_for_checkpoint(train_mode: str) -> tuple[str, ...]:
    """Which modes apply to a checkpoint of the given train_mode."""
    if train_mode == "raw":
        return RAWTRAIN_MODES
    if train_mode == "normtrain":
        return NORMTRAIN_MODES
    raise ValueError(f"Unknown train_mode: {train_mode!r}")


def _to_tensor(np_array: np.ndarray):
    """Wrap an ndarray as a torch tensor for the existing norm helpers."""
    import torch

    return torch.from_numpy(np_array)
