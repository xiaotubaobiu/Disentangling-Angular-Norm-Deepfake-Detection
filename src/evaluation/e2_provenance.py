"""Stable source fingerprint for reportable E2 evaluation runs.

The bundle covers the evaluator and local code that can change its samples,
visual features, prompts, scores, aggregation, or metrics.  It intentionally
excludes reporting/gate code, which validates this fingerprint independently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from collections.abc import Sequence


EVALUATION_SOURCE_FILES = (
    "experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py",
    "src/config_io.py",
    "src/data/builders.py",
    "src/data/constants.py",
    "src/data/crops.py",
    "src/data/datasets.py",
    "src/data/frequency.py",
    "src/data/index_external.py",
    "src/data/index_ffpp.py",
    "src/data/sampling.py",
    "src/data/semantic_dataset.py",
    "src/data/semantic_language.py",
    "src/data/transforms.py",
    "src/engine/ddp.py",
    "src/engine/metrics.py",
    "src/evaluation/e2_feature_cache.py",
    "src/evaluation/e2_provenance.py",
    "src/evaluation/sample_eval.py",
    "src/evaluation/semantic_controls.py",
    "src/evaluation/semantic_eval.py",
    "src/models/clip_semantic_jev.py",
    "src/models/factory.py",
)


def evaluation_source_bundle_sha256(
    repo_root: str | Path,
    relative_paths: Sequence[str] = EVALUATION_SOURCE_FILES,
) -> str:
    """Hash path names and bytes, independent of caller-supplied path order."""
    root = Path(repo_root)
    entries = []
    for relative in sorted(set(relative_paths)):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"source bundle path must be repository-relative: {relative}")
        content_sha = hashlib.sha256((root / path).read_bytes()).hexdigest()
        entries.append((path.as_posix(), content_sha))
    payload = json.dumps(entries, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def feature_cache_provenance(
    cache_root: str | Path | None,
    repo_root: str | Path,
) -> dict:
    """Bind an evaluation run to one mode and all three frozen domain caches.

    Including all three domains, even for a subset invocation, lets separate
    condition/domain calls share one output directory without mixed caches.
    """
    if cache_root is None:
        return {"feature_mode": "direct", "feature_cache_root": None,
                "feature_cache_manifest_sha256": {}}
    root = Path(cache_root)
    if not root.is_absolute():
        root = Path(repo_root) / root
    root = root.resolve()
    manifests = {
        domain: hashlib.sha256((root / domain / "manifest.json").read_bytes()).hexdigest()
        for domain in ("ffpp", "cdf", "dfd")
    }
    return {"feature_mode": "cache", "feature_cache_root": str(root),
            "feature_cache_manifest_sha256": manifests}
