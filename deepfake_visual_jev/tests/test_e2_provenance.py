"""Evaluation provenance must bind every result-defining local source file."""

from pathlib import Path

import pytest

from evaluation.e2_provenance import (
    EVALUATION_SOURCE_FILES,
    evaluation_source_bundle_sha256,
    feature_cache_provenance,
)


def test_source_bundle_changes_when_dependency_or_path_changes(tmp_path):
    (tmp_path / "a.py").write_text("a = 1\n")
    (tmp_path / "b.py").write_text("b = 2\n")
    paths = ("a.py", "b.py")
    original = evaluation_source_bundle_sha256(tmp_path, paths)
    assert original == evaluation_source_bundle_sha256(tmp_path, tuple(reversed(paths)))
    (tmp_path / "b.py").write_text("b = 3\n")
    assert original != evaluation_source_bundle_sha256(tmp_path, paths)
    (tmp_path / "b.py").write_text("b = 2\n")
    assert original != evaluation_source_bundle_sha256(tmp_path, ("a.py",))
    with pytest.raises(FileNotFoundError):
        evaluation_source_bundle_sha256(tmp_path, ("missing.py",))


def test_required_bundle_covers_result_defining_dependencies():
    required = {
        "experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py",
        "src/evaluation/semantic_eval.py",
        "src/evaluation/semantic_controls.py",
        "src/evaluation/e2_feature_cache.py",
        "src/evaluation/sample_eval.py",
        "src/data/semantic_dataset.py",
        "src/data/semantic_language.py",
        "src/data/builders.py",
        "src/data/datasets.py",
        "src/data/index_ffpp.py",
        "src/data/transforms.py",
        "src/models/clip_semantic_jev.py",
        "src/models/factory.py",
        "src/config_io.py",
        "src/evaluation/e2_provenance.py",
    }
    assert required <= set(EVALUATION_SOURCE_FILES)
    repo_root = Path(__file__).resolve().parents[1]
    assert all((repo_root / relative).is_file() for relative in EVALUATION_SOURCE_FILES)
    assert len(evaluation_source_bundle_sha256(repo_root)) == 64


def test_cache_provenance_locks_all_domains_even_for_split_invocations(tmp_path):
    root = tmp_path / "cache"
    for domain in ("ffpp", "cdf", "dfd"):
        path = root / domain / "manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text('{"domain":"' + domain + '"}')
    direct = feature_cache_provenance(None, tmp_path)
    assert direct == {"feature_mode": "direct", "feature_cache_root": None,
                      "feature_cache_manifest_sha256": {}}
    cached = feature_cache_provenance("cache", tmp_path)
    assert cached["feature_mode"] == "cache"
    assert cached["feature_cache_root"] == str(root.resolve())
    assert set(cached["feature_cache_manifest_sha256"]) == {"ffpp", "cdf", "dfd"}
    (root / "cdf" / "manifest.json").write_text('{"domain":"cdf","changed":true}')
    assert cached != feature_cache_provenance(root, tmp_path)
    (root / "dfd" / "manifest.json").unlink()
    with pytest.raises(FileNotFoundError):
        feature_cache_provenance(root, tmp_path)
