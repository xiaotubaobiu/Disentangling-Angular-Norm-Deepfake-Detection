#!/usr/bin/env python3
"""Stage 04 — apply the frozen FF++ thresholds to CDF and measure what happens.

Inputs : results/val_cache/source_video_scores.parquet (stage 01)
         results/tables/02_calibrators.csv             (stage 02)
         results/tables/03_source_thresholds.csv       (stage 03)
         inputs.frame_scores (phase-2 CDF frame parquet, read-only)
Outputs: results/tables/04_target_video_scores.parquet  (video-level CDF cache)
         results/tables/04_transfer_pooled.csv
         results/tables/04_transfer_per_method.csv
         results/tables/04_bootstrap.csv
GPU    : no (parquet only).
Target-domain labels: **yes** — this is the evaluation stage. Target labels are
         used for metrics and for the *diagnostic* oracle threshold only; nothing
         computed here feeds back into calibration or threshold selection.

Run standalone:
    python experiments/2026-07-30_threshold_transfer/scripts/04_evaluate_transfer.py
    python .../04_evaluate_transfer.py --dry-run
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

from study_threshold.bootstrap import paired_bootstrap  # noqa: E402
from study_threshold.calibration import Calibrator  # noqa: E402
from study_threshold.io import (  # noqa: E402
    MODES,
    aggregate_videos,
    assert_modes_share_videos,
    load_target_frames,
    method_slice,
    score_column,
)
from study_threshold.reporting import cell_id, phase_paths, write_table  # noqa: E402
from study_threshold.thresholds import confusion, oracle_ba  # noqa: E402
from study_threshold.transfer import baseline_mean_prob_rows, transfer_rows  # noqa: E402

_stage01 = import_module("01_prepare_ffpp_validation")
TARGET_CACHE = "04_target_video_scores.parquet"


def _calibrator(row: pd.Series, mode: str) -> Calibrator:
    """Rebuild the frozen calibrator from the stage-02 table (no refitting)."""
    return Calibrator(
        mode=mode,
        temperature=float(row.temperature), platt_a=float(row.platt_a),
        platt_b=float(row.platt_b),
        frame_nll_before=float(row.frame_nll_before), frame_nll_after=float(row.frame_nll_after),
        frame_ece_before=float(row.frame_ece_before), frame_ece_after=float(row.frame_ece_after),
        frame_brier_before=float(row.frame_brier_before),
        frame_brier_after=float(row.frame_brier_after),
        video_nll_before=float(row.video_nll_before),
        video_nll_after_temperature=float(row.video_nll_after_temperature),
        video_nll_after_platt=float(row.video_nll_after_platt),
        video_ece_before=float(row.video_ece_before),
        video_ece_after_temperature=float(row.video_ece_after_temperature),
        video_ece_after_platt=float(row.video_ece_after_platt),
        n_val_frames=int(row.n_val_frames), n_val_videos=int(row.n_val_videos),
    )


def _bootstrap_cell(target: pd.DataFrame, thresholds: dict, cfg: dict,
                    n_iter: int) -> list[dict]:
    """Paired (raw vs direction) CIs on the shared video resamples."""
    seed = int(cfg["protocol"]["bootstrap_seed"])
    out = []

    def ba_at(mode: str, t: float):
        col = score_column(mode)
        return lambda df: confusion(df[col].to_numpy(float), df["label"].to_numpy(int),
                                    t)["balanced_accuracy"]

    def regret_at(mode: str, t: float):
        col = score_column(mode)

        def _fn(df):
            s = df[col].to_numpy(float)
            y = df["label"].to_numpy(int)
            _, ba_star = oracle_ba(s, y)
            return ba_star - confusion(s, y, t)["balanced_accuracy"]

        return _fn

    def fpr_at(mode: str, t: float):
        col = score_column(mode)
        return lambda df: confusion(df[col].to_numpy(float), df["label"].to_numpy(int), t)["fpr"]

    quantities = [
        ("target_ba@t_BA", ba_at, "t_BA"),
        ("r_transfer@t_BA", regret_at, "t_BA"),
        ("target_fpr@t_FPR5", fpr_at, "t_FPR5"),
    ]
    for name, factory, rule in quantities:
        res = paired_bootstrap(
            target,
            factory("raw", thresholds["raw"][rule]),
            factory("direction", thresholds["direction"][rule]),
            n_iter=n_iter, rng_seed=seed,
        )
        res.update({"quantity": name, "rule": rule,
                    "threshold_raw": thresholds["raw"][rule],
                    "threshold_direction": thresholds["direction"][rule]})
        out.append(res)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-bootstrap", action="store_true")
    args = ap.parse_args()

    cfg = _stage01.load_cfg(Path(args.config) if args.config else None)
    paths = phase_paths(PHASE, cfg)
    backbones, seeds, methods = _stage01.resolve_dims(cfg, args.dry_run)
    n_iter = int(cfg["dry_run"]["bootstrap_iterations"] if args.dry_run
                 else cfg["protocol"]["bootstrap_iterations"])
    aggregator = cfg["protocol"]["video_aggregator"]
    frame_pq = ROOT / cfg["inputs"]["frame_scores"]
    target_domain = cfg["domains"]["target"]

    source = pd.read_parquet(paths["val_cache"] / _stage01.OUT_NAME)
    calibrators = pd.read_csv(paths["tables"] / "02_calibrators.csv")
    thresholds_tbl = pd.read_csv(paths["tables"] / "03_source_thresholds.csv")

    pooled_rows, method_rows, boot_rows, target_tables = [], [], [], []

    for backbone in backbones:
        for seed in seeds:
            ckpt = cell_id(backbone, seed)
            src_videos = source[source.checkpoint_id == ckpt]
            frames = load_target_frames(frame_pq, backbone, seed, domain=target_domain,
                                        methods=methods,
                                        exclude_domains=tuple(cfg["domains"]["exclude"]))
            target = aggregate_videos(frames, aggregator=aggregator)
            assert_modes_share_videos(target)
            target.insert(0, "backbone", backbone)
            target.insert(1, "seed", seed)
            target.insert(2, "checkpoint_id", ckpt)
            target_tables.append(target)

            cell_thresholds: dict[str, dict] = {}
            for mode in MODES:
                cal_row = calibrators[(calibrators.checkpoint_id == ckpt)
                                     & (calibrators["mode"] == mode)].iloc[0]
                calib = _calibrator(cal_row, mode)
                th = thresholds_tbl[(thresholds_tbl.checkpoint_id == ckpt)
                                    & (thresholds_tbl["mode"] == mode)]
                th_map = dict(zip(th["rule"], th["threshold"]))
                cell_thresholds[mode] = th_map

                rows = transfer_rows(src_videos, target, mode, th_map, calib)
                rows += baseline_mean_prob_rows(src_videos, target, mode)
                for r in rows:
                    r.update({"checkpoint_id": ckpt, "backbone": backbone, "seed": seed})
                pooled_rows += rows

                for method in methods:
                    src_m = method_slice(src_videos, method)
                    tgt_m = method_slice(target, method)
                    if src_m.empty or tgt_m.empty:
                        continue
                    m_rows = transfer_rows(src_m, tgt_m, mode, th_map, calib,
                                           rule_scope="per_method")
                    for r in m_rows:
                        r.update({"checkpoint_id": ckpt, "backbone": backbone,
                                  "seed": seed, "method_id": method})
                    method_rows += m_rows

                ba = [r for r in rows if r["rule"] == "t_BA"][0]
                print(f"[transfer] {ckpt} {mode:9s} t_BA={ba['source_threshold']:+.3f} "
                      f"| src BA {ba['source_ba']:.3f} -> tgt BA {ba['target_ba']:.3f} "
                      f"(oracle {ba['target_oracle_ba']:.3f}, R={ba['r_transfer']:.3f}) "
                      f"| tgt AUC {ba['target_auc']:.3f} "
                      f"| tgt FPR {ba['target_fpr']:.3f}")

            if not args.skip_bootstrap:
                for res in _bootstrap_cell(target, cell_thresholds, cfg, n_iter):
                    res.update({"checkpoint_id": ckpt, "backbone": backbone, "seed": seed})
                    boot_rows.append(res)

    all_target = pd.concat(target_tables, ignore_index=True)
    all_target.to_parquet(paths["tables"] / TARGET_CACHE, index=False)
    print(f"[write] {paths['tables'] / TARGET_CACHE}  ({len(all_target)} rows)")

    write_table(pd.DataFrame(pooled_rows), paths["tables"] / "04_transfer_pooled.csv")
    write_table(pd.DataFrame(method_rows), paths["tables"] / "04_transfer_per_method.csv")
    if boot_rows:
        write_table(pd.DataFrame(boot_rows), paths["tables"] / "04_bootstrap.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
