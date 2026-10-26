"""Threshold selection (source domain) and threshold-dependent metrics.

Four selection rules, all applied to *video-level* scores of the source domain:

* ``t_BA``    — maximises balanced accuracy ``(TPR + TNR) / 2``
* ``t_equal`` — minimises ``|TPR - TNR|`` (the equal-error operating point)
* ``t_FPR1``  — smallest threshold whose source FPR is ``<= 1%``
* ``t_FPR5``  — smallest threshold whose source FPR is ``<= 5%``

``t_BA`` and ``t_equal`` are different objectives and generally different numbers:
BA peaks where the *sum* TPR+TNR is largest, which need not be where the two are
equal (they coincide only for symmetric, equal-variance class distributions). Both
are reported so the gap is visible instead of assumed away.

Two further thresholds come from calibration rather than a search, and answer
"does ``p = 0.5`` after calibration work as a deployment rule?":

* ``t_p50_temperature`` — ``s = 0``. Temperature scaling is a pure rescaling, so
  ``p = 0.5`` maps to ``s = 0`` for every ``T > 0``: the threshold is
  temperature-invariant, which is precisely why temperature scaling cannot repair
  a threshold shift.
* ``t_p50_platt``       — ``s = -B / A`` (an affine map, so it *can* move the
  operating point).

A threshold near ``p = 0.5`` says nothing about calibration quality: ECE is a
property of the whole probability distribution, not of one crossing point.

Everything is computed from two sorted class-conditional arrays and
`np.searchsorted`, i.e. O(n log n) per call rather than a Python scan, because the
oracle threshold has to be recomputed inside every bootstrap resample.
Decision rule everywhere: **predict fake iff score >= threshold**.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12
RULES = ("t_BA", "t_equal", "t_FPR1", "t_FPR5")
CALIBRATION_RULES = ("t_p50_temperature", "t_p50_platt")


def _split(scores, labels):
    scores = np.asarray(scores, float)
    labels = np.asarray(labels, int)
    return scores[labels == 1], scores[labels == 0]


def candidate_thresholds(scores) -> np.ndarray:
    """Midpoints between consecutive unique scores, plus outer margins."""
    uniq = np.unique(np.asarray(scores, float))
    if uniq.size == 1:
        return np.array([uniq[0] - 1.0, uniq[0], uniq[0] + 1.0])
    mids = (uniq[:-1] + uniq[1:]) / 2.0
    span = float(uniq[-1] - uniq[0])
    pad = max(span * 0.01, 1e-6)
    return np.concatenate([[uniq[0] - pad], mids, [uniq[-1] + pad]])


def rate_curves(scores, labels):
    """(candidates, tpr, tnr) evaluated at every candidate threshold, vectorised."""
    fake, real = _split(scores, labels)
    cands = candidate_thresholds(scores)
    fake_sorted = np.sort(fake)
    real_sorted = np.sort(real)
    # count(x >= t) = n - searchsorted(sorted_x, t, "left")
    tp = fake_sorted.size - np.searchsorted(fake_sorted, cands, side="left")
    fp = real_sorted.size - np.searchsorted(real_sorted, cands, side="left")
    tpr = tp / max(fake_sorted.size, 1)
    fpr = fp / max(real_sorted.size, 1)
    return cands, tpr, 1.0 - fpr


def confusion(scores, labels, threshold: float) -> dict:
    """Hard-decision metrics at `threshold` (predict fake iff score >= threshold)."""
    fake, real = _split(scores, labels)
    tp = int((fake >= threshold).sum())
    fn = int(fake.size - tp)
    fp = int((real >= threshold).sum())
    tn = int(real.size - fp)
    tpr = tp / max(fake.size, 1)
    tnr = tn / max(real.size, 1)
    precision = tp / max(tp + fp, 1)
    f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) > 0 else 0.0
    n = tp + fn + fp + tn
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": float(tpr), "tnr": float(tnr),
        "fpr": float(1.0 - tnr), "fnr": float(1.0 - tpr),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "accuracy": float((tp + tn) / max(n, 1)),
        "precision": float(precision),
        "f1": float(f1),
        "predicted_fake_rate": float((tp + fp) / max(n, 1)),
        "n_fake": int(fake.size), "n_real": int(real.size),
    }


def best_ba_threshold(scores, labels) -> float:
    """Threshold maximising balanced accuracy (ties -> smallest threshold)."""
    cands, tpr, tnr = rate_curves(scores, labels)
    return float(cands[int(np.argmax((tpr + tnr) / 2.0))])


def best_ba(scores, labels) -> float:
    cands, tpr, tnr = rate_curves(scores, labels)
    return float(np.max((tpr + tnr) / 2.0))


def equal_error_threshold(scores, labels) -> float:
    """Threshold minimising |TPR - TNR| (ties -> smallest threshold)."""
    cands, tpr, tnr = rate_curves(scores, labels)
    return float(cands[int(np.argmin(np.abs(tpr - tnr)))])


def fpr_threshold(scores, labels, target_fpr: float) -> float:
    """Smallest threshold whose FPR on these (source) scores is <= target_fpr."""
    cands, _, tnr = rate_curves(scores, labels)
    fpr = 1.0 - tnr
    order = np.argsort(cands)
    cands, fpr = cands[order], fpr[order]
    ok = np.where(fpr <= target_fpr + 1e-12)[0]
    return float(cands[ok[0]]) if ok.size else float(cands[-1])


def select_source_thresholds(scores, labels, temperature: float | None = None,
                             platt_a: float | None = None,
                             platt_b: float | None = None) -> dict:
    """All source-domain thresholds in one dict (score space = video signed logit)."""
    out = {
        "t_BA": best_ba_threshold(scores, labels),
        "t_equal": equal_error_threshold(scores, labels),
        "t_FPR1": fpr_threshold(scores, labels, 0.01),
        "t_FPR5": fpr_threshold(scores, labels, 0.05),
    }
    if temperature is not None:
        out["t_p50_temperature"] = 0.0
    if platt_a is not None and platt_b is not None:
        out["t_p50_platt"] = float(-platt_b / platt_a) if platt_a != 0 else float("nan")
    return out


def oracle_ba(scores, labels) -> tuple[float, float]:
    """(threshold, BA) of the best operating point achievable on THESE scores.

    Diagnostic only: on the target domain this consumes target labels, so it is an
    upper bound that no deployable rule can claim.
    """
    cands, tpr, tnr = rate_curves(scores, labels)
    ba = (tpr + tnr) / 2.0
    i = int(np.argmax(ba))
    return float(cands[i]), float(ba[i])


def auc_mann_whitney(scores, labels) -> float:
    """Threshold-free ranking metric (ties counted as 0.5)."""
    from study.auc_pairs import auc_mannwhitney

    fake, real = _split(scores, labels)
    if fake.size == 0 or real.size == 0:
        return float("nan")
    return float(auc_mannwhitney(fake, real))
