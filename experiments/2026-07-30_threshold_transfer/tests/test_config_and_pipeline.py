"""Config-level and end-to-end checks.

8.  every input path declared in `threshold_study.yaml` exists;
9.  stage 3 / stage 4 configs still resolve after the stage-2 layout migration
    (this phase depends on the same run directory, so it re-checks it);
10. `run_all.py --dry-run` completes on an isolated output tree.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
CONFIG = PHASE / "configs/threshold_study.yaml"
CFG = yaml.safe_load(CONFIG.read_text())

LOCAL_ARTIFACT = (ROOT / CFG["inputs"]["frame_scores"]).is_file()
needs_artifacts = pytest.mark.skipif(
    not LOCAL_ARTIFACT, reason="phase-2 frame parquet is a git-ignored local artifact")


def test_config_declares_the_expected_protocol():
    assert CFG["domains"]["source"] == "ffpp"
    assert CFG["domains"]["target"] == "cdf"
    assert CFG["protocol"]["video_aggregator"] == "mean_logit"
    assert CFG["protocol"]["baseline_aggregator"] == "mean_prob"
    assert CFG["protocol"]["bootstrap_iterations"] == 2000
    assert isinstance(CFG["global_seed"], int)
    assert CFG["dims"]["modes"] == ["raw", "direction"]
    assert len(CFG["dims"]["backbones"]) == 3 and len(CFG["dims"]["seeds"]) == 5
    assert len(CFG["dims"]["methods"]) == 16


@needs_artifacts
def test_every_declared_input_path_exists():
    for key, rel in CFG["inputs"].items():
        assert (ROOT / rel).exists(), f"inputs.{key} -> {rel} missing"


@needs_artifacts
def test_stage3_and_stage4_configs_resolve_after_stage2_migration():
    stage3 = yaml.safe_load(
        (ROOT / "experiments/2026-07-23_four_questions/configs/study_config.yaml").read_text())
    stage4 = yaml.safe_load(
        (ROOT / "experiments/2026-07-27_pooling_saturation/configs/"
                "pooling_saturation_config.yaml").read_text())
    for rel in (stage3["parquet"]["frame"], stage3["parquet"]["video"], stage3["ckpt_root"],
                stage4["parquet_frame"], stage4["ckpt_root"]):
        assert (ROOT / rel).exists(), f"{rel} missing after migration"
    run_dir = ROOT / CFG["inputs"]["frame_scores"]
    assert run_dir.parent.name == "parallel_20260721_213914"
    assert run_dir.parent.parent.name == "results"
    assert (run_dir.parent / "figures").is_dir()
    assert (run_dir.parent / "logs").is_dir()
    assert (run_dir.parent / "reports" / "provenance.md").is_file()


@needs_artifacts
def test_run_all_dry_run_completes(tmp_path):
    """Full pipeline on a subset, writing into a throwaway tree."""
    cfg = yaml.safe_load(CONFIG.read_text())
    for key in cfg["paths"]:
        d = tmp_path / key
        d.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = str(d)
    cfg["outputs"]["report"] = str(tmp_path / "report.md")
    cfg["outputs"]["root_digest"] = str(tmp_path / "digest.md")
    cfg_path = tmp_path / "threshold_study.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    proc = subprocess.run(
        [sys.executable, str(PHASE / "scripts" / "run_all.py"), "--dry-run",
         "--config", str(cfg_path)],
        cwd=str(ROOT), capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
    for name in ("00_input_audit.csv", "01_source_videos.csv", "02_calibrators.csv",
                 "03_source_thresholds.csv", "04_transfer_pooled.csv",
                 "04_transfer_per_method.csv", "04_bootstrap.csv", "05_score_shift.csv"):
        assert (tmp_path / "tables" / name).is_file(), f"{name} not produced"
    assert Path(cfg["outputs"]["report"]).is_file()
    assert Path(cfg["outputs"]["root_digest"]).is_file()
    report = Path(cfg["outputs"]["report"]).read_text()
    for heading in ("## 0. Answers", "R_transfer", "Leakage statement"):
        assert heading in report
