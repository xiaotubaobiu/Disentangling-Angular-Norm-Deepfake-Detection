"""Independent project relocation must be repeatable and preserve evidence."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


@pytest.fixture
def project(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/relocate_artifacts.py"
    spec = importlib.util.spec_from_file_location("project_relocation", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = tmp_path / "workspace"
    root = original / "projects/visual_jev"
    phase = Path("experiments/2026-09-22_semantic_visual_jev_e2")
    e2 = root / phase
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "E2", e2)

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def artifact(relative):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(relative).encode())
        return hashlib.sha256(path.read_bytes()).hexdigest()

    write(root / "provenance/import_manifest.json", {"original_root": str(original)})
    h1 = Path("experiments/h1/best_model.pth")
    h1_hash = artifact(h1)
    for name in ("preflight.json", "execution_manifest.json"):
        write(e2 / "results" / name, {"checkpoints": {"h1": {"path": str(original / h1), "sha256": h1_hash}}})
    cache = phase / "results/feature_cache"
    hashes = {domain: artifact(cache / domain / "manifest.json") for domain in ("ffpp", "cdf", "dfd")}
    for seed in (42, 7, 2048):
        checkpoint = phase / f"results/training/seed_{seed}/best_model.pth"
        digest = artifact(checkpoint)
        write(e2 / f"results/evaluation_primary_v2/seed_{seed}/run_provenance.json", {
            "checkpoint_path": str(original / checkpoint), "checkpoint_sha256": digest,
            "feature_cache_root": str(original / cache), "feature_cache_manifest_sha256": hashes,
            "seed": seed, "scientific_marker": "unchanged"})
        write(e2 / f"results/baseline_h1/seed_{seed}/evaluation.json", {"checkpoint": str(original / h1)})
    return module, root, phase


def test_relocation_is_idempotent_inside_original_root(project):
    module, root, _ = project
    module.relocate()
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")}
    module.relocate()
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")} == before
    assert len(json.loads((root / "provenance/path_relocations.json").read_text())) == 1


def test_project_can_move_again_without_original_workspace(project, tmp_path, monkeypatch):
    module, root, phase = project
    module.relocate()
    moved = tmp_path / "detached"
    shutil.move(root, moved)
    monkeypatch.setattr(module, "ROOT", moved)
    monkeypatch.setattr(module, "E2", moved / phase)
    module.relocate()
    path = moved / phase / "results/evaluation_primary_v2/seed_42/run_provenance.json"
    data = json.loads(path.read_text())
    assert Path(data["checkpoint_path"]).is_relative_to(moved)
    assert data["scientific_marker"] == "unchanged"
    assert len(json.loads((moved / "provenance/path_relocations.json").read_text())) == 2
    assert (moved / "provenance/path_metadata_original" / phase / "results/evaluation_primary_v2/seed_42/run_provenance.json").is_file()


def test_bad_checkpoint_rejects_before_metadata_rewrite(project):
    module, root, phase = project
    path = root / phase / "results/evaluation_primary_v2/seed_42/run_provenance.json"
    before = path.read_bytes()
    (root / phase / "results/training/seed_42/best_model.pth").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        module.relocate()
    assert path.read_bytes() == before
    assert not (root / "provenance/path_relocations.json").exists()
