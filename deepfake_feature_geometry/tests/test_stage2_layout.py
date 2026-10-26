"""Repo-level guards for the canonical stage-2 artifact layout.

The 2026-07-21 diagnosis run directory is the single source of truth for the
frame-level scores that stages 3, 4 and 5 consume. It drifted twice (artifacts
moved out of `results/`), so these tests pin the layout down:

- `results/parallel_20260721_213914/` is the only run directory,
- tables + parquet at its root, figures in `figures/`, logs in `logs/`,
  provenance in `reports/`,
- stage 3 / 4 / threshold-transfer configs resolve to existing files,
- the `cdf_*_dir_raw.py` helpers write into `<run_dir>/figures`.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
STAGE2 = REPO / "experiments/2026-07-21_frame_midterm_diagnosis"
RUN = STAGE2 / "results/parallel_20260721_213914"

pytestmark = pytest.mark.skipif(
    not RUN.is_dir(),
    reason="stage-2 run directory is a local artifact (git-ignored); skipping layout checks",
)


def _load(script_name: str):
    path = STAGE2 / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_single_run_directory_under_results():
    runs = [p for p in (STAGE2 / "results").iterdir() if p.is_dir()]
    assert runs == [RUN], f"expected exactly one run dir ({RUN.name}), found {[p.name for p in runs]}"
    # no stray copy at the phase root
    assert not (STAGE2 / RUN.name).exists()
    assert not (STAGE2 / "figures").exists()
    assert not (STAGE2 / "reports").exists()


def test_run_directory_subdirs():
    assert (RUN / "figures").is_dir()
    assert (RUN / "logs").is_dir()
    assert (RUN / "reports" / "provenance.md").is_file()
    assert list((RUN / "logs").glob("*.log")), "worker/launcher logs must live in logs/"
    assert list((RUN / "figures").glob("*.png"))


def test_frame_parquet_readable():
    import duckdb

    pq = RUN / "per_frame_scores.parquet"
    assert pq.is_file()
    n, n_ck = duckdb.query(
        f"select count(*), count(distinct checkpoint_id) from read_parquet('{pq}')"
    ).fetchone()
    assert n == 39_830_640
    assert n_ck == 30


@pytest.mark.parametrize(
    "config_rel, keys",
    [
        ("experiments/2026-07-23_four_questions/configs/study_config.yaml",
         ["parquet.frame", "parquet.video", "ckpt_root"]),
        ("experiments/2026-07-27_pooling_saturation/configs/pooling_saturation_config.yaml",
         ["parquet_frame", "ckpt_root"]),
        ("experiments/2026-07-30_threshold_transfer/configs/threshold_study.yaml",
         ["inputs.frame_scores", "inputs.checkpoint_root"]),
    ],
)
def test_downstream_configs_resolve(config_rel, keys):
    import yaml

    cfg_path = REPO / config_rel
    if not cfg_path.exists():
        pytest.skip(f"{config_rel} not present")
    raw = yaml.safe_load(cfg_path.read_text())
    for key in keys:
        node = raw
        for part in key.split("."):
            node = node[part]
        assert (REPO / node).exists(), f"{config_rel}:{key} -> {node} does not exist"


def _synthetic_run_dir(tmp_path: Path) -> Path:
    """Minimal per_video_scores.parquet covering all backbones/seeds the helpers loop over."""
    mod = _load("cdf_balanced_dir_raw.py")
    rows = []
    rng = __import__("numpy").random.default_rng(0)
    for backbone in mod.BACKBONES:
        for seed in mod.SEED_ORDER:
            for mode, shift in (("raw", 0.0), ("direction", 0.1)):
                for method in ["fsgan", "sadtalker"]:
                    for i in range(20):
                        rows.append(dict(backbone=backbone, seed=seed, train_mode="raw",
                                         infer_mode=mode, domain="cdf", method_id=method,
                                         video_id=f"{method}_{i}",
                                         video_prob_fake=float(0.6 + shift + rng.normal(0, .05))))
                for i in range(20):
                    rows.append(dict(backbone=backbone, seed=seed, train_mode="raw",
                                     infer_mode=mode, domain="cdf", method_id="real",
                                     video_id=f"real_{i}",
                                     video_prob_fake=float(0.4 + shift + rng.normal(0, .05))))
    run = tmp_path / "parallel_test"
    run.mkdir()
    pd.DataFrame(rows).to_parquet(run / "per_video_scores.parquet")
    return run


def test_balanced_helper_writes_into_run_dir_figures(tmp_path, monkeypatch):
    run = _synthetic_run_dir(tmp_path)
    mod = _load("cdf_balanced_dir_raw.py")
    monkeypatch.setattr("sys.argv", ["cdf_balanced_dir_raw.py", str(run)])
    mod.main()
    assert (run / "cdf_balanced_delta_auc.csv").is_file()
    figs = sorted(p.name for p in (run / "figures").glob("*.png"))
    assert figs == ["balanced_heatmap_cdf_clip.png", "balanced_heatmap_cdf_dinov2.png",
                    "balanced_heatmap_cdf_efficientnet.png", "balanced_heatmap_cdf_pooled.png"]


def test_native_helper_writes_into_run_dir(tmp_path, monkeypatch):
    run = _synthetic_run_dir(tmp_path)
    mod = _load("cdf_native_dir_raw.py")
    monkeypatch.setattr("sys.argv", ["cdf_native_dir_raw.py", str(run)])
    mod.main()
    assert (run / "cdf_delta_auc_pooled.csv").is_file()
    assert (run / "cdf_delta_auc_per_method.csv").is_file()
