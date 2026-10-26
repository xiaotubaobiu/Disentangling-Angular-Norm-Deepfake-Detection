"""Cross-domain threshold transfer library (phase 2026-07-30).

Question: after calibrating and picking a decision threshold on FF++ validation,
does that frozen threshold still behave on CDF?

Design rules baked into this package:

* the **primary** video score is ``mean_t(signed_logit)`` (`mean_logit`), never
  ``mean_t(sigmoid(signed_logit))``; the latter is exposed only as a deliberately
  mis-specified baseline (`score_col="video_prob"`),
* calibrators and thresholds are fitted on the **source** domain only; every
  function that touches the target domain takes already-frozen parameters,
* raw and direction are always evaluated on the *same* video set, with the same
  aggregation, the same bootstrap clusters and the same metric code.

Reuses, rather than reimplements:
``config_io`` (YAML `_base_`), ``study.io.dedup_reals`` (real-frame dedup),
``study_pooling.aggregators.aggregate`` (video aggregation),
``study_pooling.temperature`` (temperature / Platt / ECE / Brier / NLL),
``study.cluster_bootstrap.cluster_bootstrap_metric`` (video-cluster bootstrap).
"""

from study_threshold import (  # noqa: F401
    bootstrap,
    calibration,
    io,
    reporting,
    score_shift,
    thresholds,
    transfer,
)

__all__ = [
    "io",
    "calibration",
    "thresholds",
    "transfer",
    "score_shift",
    "bootstrap",
    "reporting",
]
