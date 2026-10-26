"""Supported training backbones must not depend on the archived Xception configs.

Guards the 2026-07-29 config cleanup: `configs/xception_base.yaml` (and the
phase-local xception configs) were deleted into
`.deleted_configs_backup_20260729.tar.gz`, so nothing in the maintained training
path may reference them, while `src/models/xception.py` stays importable for old
checkpoints.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[3]
PHASE = Path(__file__).resolve().parents[1]
SUPPORTED = ["clip", "dinov2", "efficientnet"]


def _launch_training():
    path = PHASE / "scripts" / "launch_training.py"
    spec = importlib.util.spec_from_file_location("launch_training", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("backbone", SUPPORTED)
def test_supported_backbone_config_exists_and_loads(backbone):
    from config_io import load_config

    cfg_path = REPO / "configs" / f"{backbone}_base.yaml"
    assert cfg_path.is_file(), f"{cfg_path} missing"
    cfg = load_config(cfg_path)
    assert cfg["model"]["name"], "resolved config must name a model"


def test_generated_configs_never_reference_archived_xception():
    mod = _launch_training()
    for name, spec in mod.BACKBONES.items():
        assert "xception" not in name
        assert (REPO / spec["base_config"]).is_file(), (
            f"{name} points at missing {spec['base_config']}"
        )


def test_requesting_xception_fails_with_archive_message():
    mod = _launch_training()
    with pytest.raises(SystemExit) as exc:
        mod.resolve_backbones(["xception"])
    assert str(exc.value) == (
        "Xception training configs are archived in "
        ".deleted_configs_backup_20260729.tar.gz and are not maintained in the "
        "current experiment."
    )


def test_generation_runs_for_supported_backbones(tmp_path, monkeypatch):
    mod = _launch_training()
    monkeypatch.setattr(
        "sys.argv",
        ["launch_training.py", "--dinov2-lr", "5e-5", "--backbones", "dinov2",
         "--config-dir", str(tmp_path)],
    )
    mod.main()
    written = sorted(p.name for p in tmp_path.glob("*.yaml"))
    assert len(written) == 10  # 2 variants x 5 seeds
    body = yaml.safe_load((tmp_path / written[0]).read_text())
    assert body["_base_"] == "configs/dinov2_base.yaml"
    assert (REPO / body["_base_"]).is_file()


def test_legacy_xception_model_still_importable():
    """Model code is legacy-but-kept: old checkpoints must stay loadable."""
    from models.factory import build_model

    build_model({"name": "xception", "num_classes": 2})


def test_no_repo_file_references_deleted_xception_config():
    hits = []
    for pattern in ("*.py", "*.yaml", "*.md"):
        for path in list(REPO.glob(f"configs/{pattern}")) \
                + list((REPO / "src").rglob(pattern)) \
                + list((REPO / "experiments").glob(f"*/scripts/{pattern}")) \
                + list((REPO / "experiments").glob(f"*/configs/**/{pattern}")):
            text = path.read_text(errors="ignore")
            if "xception_base.yaml" in text and "archived" not in text:
                hits.append(str(path.relative_to(REPO)))
    assert not hits, f"stale references to the deleted xception config: {hits}"
