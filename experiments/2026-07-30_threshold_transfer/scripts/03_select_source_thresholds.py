#!/usr/bin/env python3
"""Stage 03 — pick the deployment thresholds on FF++ validation and freeze them.

Inputs : results/val_cache/source_video_scores.parquet (stage 01)
         results/tables/02_calibrators.csv             (stage 02)
Outputs: results/tables/03_source_thresholds.csv
GPU    : no.
Target-domain labels: **no — by construction**. Only source video scores are
         read; the CDF parquet is never opened (asserted in tests).

Rules: t_BA (max balanced accuracy), t_equal (min |TPR-TNR|), t_FPR1 / t_FPR5
(source FPR budgets), plus the calibration-implied p=0.5 thresholds
(t_p50_temperature = 0 by construction, t_p50_platt = -B/A).

Run standalone:
    python experiments/2026-07-30_threshold_transfer/scripts/03_select_source_thresholds.py
    python .../03_select_source_thresholds.py --dry-run
"""
from __future__ import annotations

import argparse
import sys
from importlib import import_module
from pathlib import Path

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PHASE / "scripts"))

import pandas as pd  # noqa: E402

from study_threshold.io import MODES, score_column  # noqa: E402
from study_threshold.reporting import cell_id, phase_paths, write_table  # noqa: E402
from study_threshold.thresholds import confusion, select_source_thresholds  # noqa: E402

_stage01 = import_module("01_prepare_ffpp_validation")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = _stage01.load_cfg(Path(args.config) if args.config else None)
    paths = phase_paths(PHASE, cfg)
    backbones, seeds, _ = _stage01.resolve_dims(cfg, args.dry_run)

    source = pd.read_parquet(paths["val_cache"] / _stage01.OUT_NAME)
    calibrators = pd.read_csv(paths["tables"] / "02_calibrators.csv")

    rows = []
    for backbone in backbones:
        for seed in seeds:
            ckpt = cell_id(backbone, seed)
            videos = source[source.checkpoint_id == ckpt]
            for mode in MODES:
                cal = calibrators[(calibrators.checkpoint_id == ckpt)
                                  & (calibrators["mode"] == mode)]
                if cal.empty:
                    raise SystemExit(f"no calibrator for {ckpt}/{mode}; run stage 02 first")
                cal = cal.iloc[0]
                col = score_column(mode)
                scores = videos[col].to_numpy(float)
                labels = videos["label"].to_numpy(int)
                thresholds = select_source_thresholds(
                    scores, labels,
                    temperature=float(cal.temperature),
                    platt_a=float(cal.platt_a), platt_b=float(cal.platt_b),
                )
                for rule, t in thresholds.items():
                    src = confusion(scores, labels, t)
                    rows.append({
                        "checkpoint_id": ckpt, "backbone": backbone, "seed": seed,
                        "mode": mode, "rule": rule, "threshold": float(t),
                        "source_ba": src["balanced_accuracy"],
                        "source_tpr": src["tpr"], "source_tnr": src["tnr"],
                        "source_fpr": src["fpr"], "source_fnr": src["fnr"],
                        "source_precision": src["precision"], "source_f1": src["f1"],
                        "source_predicted_fake_rate": src["predicted_fake_rate"],
                        "n_source_fake": src["n_fake"], "n_source_real": src["n_real"],
                        "temperature": float(cal.temperature),
                        "platt_a": float(cal.platt_a), "platt_b": float(cal.platt_b),
                        "selected_on": "ffpp_validation",
                    })
                summary = {r: thresholds[r] for r in ("t_BA", "t_equal", "t_FPR1", "t_FPR5")}
                print(f"[thr] {ckpt} {mode:9s} " +
                      " ".join(f"{k}={v:+.3f}" for k, v in summary.items()) +
                      f" | t_BA vs t_equal gap {abs(summary['t_BA'] - summary['t_equal']):.3f}")

    write_table(pd.DataFrame(rows), paths["tables"] / "03_source_thresholds.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
