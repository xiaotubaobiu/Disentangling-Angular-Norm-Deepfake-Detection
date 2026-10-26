"""Replay the threshold pipeline's CPU dry-run into a new output directory."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
PHASE = ROOT / "experiments/2026-07-30_threshold_transfer"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new output directory (must not exist)")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = (args.output or ROOT / "runs" / f"threshold_{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((PHASE / "configs/threshold_study.yaml").read_text())
    cfg["paths"] = {name: str(output / name) for name in cfg["paths"]}
    cfg["outputs"] = {"report": str(output / "reports/threshold_transfer_final_report.md"),
                      "root_digest": str(output / "digest.md")}
    config = output / "configs/threshold_study.yaml"
    config.parent.mkdir()
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(ROOT / "src")}
    subprocess.run([sys.executable, str(PHASE / "scripts/run_all.py"),
                    "--config", str(config), "--dry-run"], cwd=ROOT, env=env, check=True)
    print(f"Dry-run outputs: {output}")


if __name__ == "__main__":
    main()
