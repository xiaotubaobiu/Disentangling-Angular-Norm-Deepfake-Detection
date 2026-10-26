"""Frame-MIDT diagnostic pipeline.

Read-only analysis over the 30 existing checkpoints (3 backbones × 5 seeds × 2
training modes). Produces per-frame / per-video / per-method / aggregate
Parquet + CSV artifacts, hypothesis summary JSON, and diagnostic figures.

This package does NOT modify checkpoints, add training losses, or implement
MIDT. It only inspects what the existing checkpoints already produce under
different inference-mode post-processings of the same raw features.
"""

from diagnosis.checkpoint_index import (
    CheckpointInfo,
    discover_checkpoints,
)
from diagnosis.method_map import (
    DFD_NATIVE_METHOD_ID,
    METHOD_TO_INDEX,
    REAL_METHOD_ID,
    method_to_id,
)

__all__ = [
    "CheckpointInfo",
    "discover_checkpoints",
    "METHOD_TO_INDEX",
    "REAL_METHOD_ID",
    "DFD_NATIVE_METHOD_ID",
    "method_to_id",
]
