"""Leakage guards: the target domain may not enter fitting or threshold selection.

Two independent levels of evidence:

* **static** — stages 02 and 03 must not mention the target parquet input or the
  target loader anywhere in their source;
* **dynamic** — stages 01→02→03 are executed with
  `study_threshold.io.load_target_frames` replaced by a function that raises, and
  with the config's `inputs.frame_scores` pointing at a non-existent file. If any
  of them tried to read CDF they would fail.
"""
from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pytest
import yaml

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(PHASE / "scripts"))

CONFIG = PHASE / "configs/threshold_study.yaml"


def _tmp_config(tmp_path: Path) -> Path:
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg["inputs"]["frame_scores"] = "does/not/exist/target.parquet"   # target unreachable
    for key in cfg["paths"]:
        d = tmp_path / key
        d.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = str(d)
    out = tmp_path / "threshold_study.yaml"
    out.write_text(yaml.safe_dump(cfg))
    return out


def _code_without_docstrings(path: Path) -> str:
    """Executable source only: docstrings and comments dropped.

    The stage docstrings *do* mention the target parquet (to state that they never
    read it), so a raw text scan would be meaningless.
    """
    import ast

    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


@pytest.mark.parametrize("script", ["02_fit_calibrators.py", "03_select_source_thresholds.py"])
def test_fit_and_selection_scripts_never_name_the_target(script):
    code = _code_without_docstrings(PHASE / "scripts" / script)
    assert "load_target_frames" not in code
    assert "frame_scores" not in code
    assert "cdf" not in code.lower()
    assert "04_target" not in code


def test_calibration_and_threshold_selection_run_without_the_target(tmp_path, monkeypatch):
    cfg_path = _tmp_config(tmp_path)

    def _explode(*a, **k):
        raise AssertionError("target domain must not be read during fitting/selection")

    import study_threshold.io as st_io

    monkeypatch.setattr(st_io, "load_target_frames", _explode)

    stage01 = import_module("01_prepare_ffpp_validation")
    stage02 = import_module("02_fit_calibrators")
    stage03 = import_module("03_select_source_thresholds")

    for mod in (stage01, stage02, stage03):
        monkeypatch.setattr(sys, "argv", [mod.__name__, "--config", str(cfg_path), "--dry-run"])
        assert mod.main() == 0

    tables = tmp_path / "tables"
    assert (tables / "02_calibrators.csv").is_file()
    assert (tables / "03_source_thresholds.csv").is_file()


def test_source_thresholds_table_declares_its_selection_domain(tmp_path, monkeypatch):
    import pandas as pd

    cfg_path = _tmp_config(tmp_path)
    stage01 = import_module("01_prepare_ffpp_validation")
    stage02 = import_module("02_fit_calibrators")
    stage03 = import_module("03_select_source_thresholds")
    for mod in (stage01, stage02, stage03):
        monkeypatch.setattr(sys, "argv", [mod.__name__, "--config", str(cfg_path), "--dry-run"])
        mod.main()
    df = pd.read_csv(tmp_path / "tables" / "03_source_thresholds.csv")
    assert set(df["selected_on"]) == {"ffpp_validation"}
    assert set(df["rule"]) >= {"t_BA", "t_equal", "t_FPR1", "t_FPR5"}


def test_dfd_is_excluded_by_config_and_loader():
    cfg = yaml.safe_load(CONFIG.read_text())
    assert cfg["domains"]["exclude"] == ["dfd"]
    from study_threshold.io import load_target_frames

    with pytest.raises(ValueError, match="excluded"):
        load_target_frames("unused.parquet", "clip", 42, domain="dfd",
                           exclude_domains=("dfd",))
