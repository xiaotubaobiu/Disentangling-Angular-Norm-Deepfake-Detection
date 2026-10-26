"""Method-label normalization for the diagnostic pipeline.

Per-frame method labels come directly from loader.dataset.records (see
feature_extractor.py), not from a video_id lookup. This avoids a subtle but
critical bug: in FF++/CDF the same pair_id (= the 'video_id' the loader
emits) is reused across forgery methods — e.g. pair_id '000_003' is generated
by simswap, blendface, faceswap, ..., so a {video_id -> method} dict would
silently lose all but one method.

This module only holds the canonical taxonomy + normalization rules:

- DF40 method names (e.g. 'simswap') → kept verbatim.
- 'real' (ffpp/cdf real records, already relabeled by load_json_index) → 'real'.
- 'DFD_real' → 'real'.
- 'DFD_fake' → 'dfd_native' (spec section VIII: do not guess the DFD method).
"""
from __future__ import annotations

from data.constants import ALL_METHODS


# Canonical method id assignment: 16 DF40 methods get ids 0..15 (kept as names
# for readability in the diagnostic CSVs / Parquet).
METHOD_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(ALL_METHODS)}
REAL_METHOD_ID = "real"
DFD_NATIVE_METHOD_ID = "dfd_native"


def method_to_id(method_str: str) -> str:
    """Normalize a method string from a FrameRecord into our diagnostic taxonomy.

    FrameRecord.method can be:
      - a DF40 method name (e.g. 'simswap') for fake ffpp/cdf records,
      - 'real' for real ffpp/cdf records (load_json_index sets this),
      - 'DFD_real' or 'DFD_fake' for DFD records (load_dfd_json_index sets these).

    DFD_real collapses to 'real'; DFD_fake collapses to 'dfd_native'. Real
    videos keep the 'real' label regardless of origin so they form a single
    shared negative class within each domain.
    """
    if method_str == "real":
        return REAL_METHOD_ID
    if method_str == "DFD_real":
        return REAL_METHOD_ID
    if method_str == "DFD_fake":
        return DFD_NATIVE_METHOD_ID
    if method_str in METHOD_TO_INDEX:
        return method_str
    # Unknown method string — keep it verbatim so the problem is visible in
    # downstream output rather than silently mislabeled.
    return method_str
