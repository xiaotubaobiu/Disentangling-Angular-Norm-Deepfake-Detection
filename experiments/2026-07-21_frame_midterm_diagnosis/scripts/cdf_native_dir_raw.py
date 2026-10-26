"""Native-count CDF direction − raw AUC, pooled and per method.

Recovered generator for two artifacts of this phase that had no surviving
script: `cdf_delta_auc_pooled.csv` and `cdf_delta_auc_per_method.csv`
(both written 2026-07-22 16:48 by an ad-hoc `cdf_dir_raw_heatmap.py` that was
never committed and is lost). The recipe below was reverse-engineered from the
CSVs and reproduces them to floating-point identity (max |diff| = 1.1e-16 on
`auc_raw`, `auc_direction`, `delta`; `n_fake` / `n_real` exact).

"Native" means every video is used as it appears in `per_video_scores.parquet`:
all fakes of a method against all 178 CDF real videos, no downsampling. The
companion `cdf_balanced_dir_raw.py` does the balanced (178v178, paired
subsample) version, which is the one to trust when comparing methods with very
different fake counts.

Scope: CDF only, `train_mode=raw` checkpoints, `infer_mode ∈ {raw, direction}`,
score = `video_prob_fake` (= mean over frames of sigmoid(signed_logit)).

Usage:
    python experiments/2026-07-21_frame_midterm_diagnosis/scripts/cdf_native_dir_raw.py \\
        experiments/2026-07-21_frame_midterm_diagnosis/results/parallel_20260721_213914

    # write elsewhere (e.g. to diff against the historical CSVs)
    ... <run_dir> --out-dir /tmp/check
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

METHOD_ORDER = [
    "simswap", "inswap", "blendface", "faceswap",
    "fsgan", "mobileswap", "e4s", "facedancer",
    "fomm", "facevid2vid", "wav2lip", "sadtalker",
    "MRAA", "pirender", "tpsm", "lia",
]
SEED_ORDER = [7, 42, 123, 999, 2048]
BACKBONES = ["clip", "dinov2", "efficientnet"]


def _paired_modes(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (raw, direction) frames aligned row-by-row on (method_id, video_id)."""
    raw = base[base["infer_mode"] == "raw"].sort_values(
        ["method_id", "video_id"]).reset_index(drop=True)
    direction = base[base["infer_mode"] == "direction"].sort_values(
        ["method_id", "video_id"]).reset_index(drop=True)
    if len(raw) != len(direction):
        raise ValueError(
            f"raw/direction row counts differ ({len(raw)} vs {len(direction)}); "
            "the two modes must cover the same (method_id, video_id) set"
        )
    return raw, direction


def compute(pv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (pooled_df, per_method_df) of native-count delta AUC on CDF."""
    sub = pv[(pv["domain"] == "cdf")
             & (pv["train_mode"] == "raw")
             & (pv["infer_mode"].isin(["raw", "direction"]))]

    pooled_rows, method_rows = [], []
    for backbone in BACKBONES:
        for seed in SEED_ORDER:
            base = sub[(sub["backbone"] == backbone) & (sub["seed"] == seed)]
            if base.empty:
                continue
            raw, direction = _paired_modes(base)
            is_real = (raw["method_id"] == "real").to_numpy()

            # Pooled: all fakes of this checkpoint vs all reals.
            labels = (~is_real).astype(int)
            auc_raw = roc_auc_score(labels, raw["video_prob_fake"])
            auc_dir = roc_auc_score(labels, direction["video_prob_fake"])
            pooled_rows.append({
                "backbone": backbone, "seed": seed,
                "delta": auc_dir - auc_raw,
                "auc_raw": auc_raw, "auc_direction": auc_dir,
            })

            # Per method: that method's fakes vs all reals.
            for method in METHOD_ORDER:
                is_method = (raw["method_id"] == method).to_numpy()
                if not is_method.any():
                    continue
                sel = is_real | is_method
                labels_m = is_method[sel].astype(int)
                auc_raw_m = roc_auc_score(labels_m, raw["video_prob_fake"][sel])
                auc_dir_m = roc_auc_score(labels_m, direction["video_prob_fake"][sel])
                method_rows.append({
                    "backbone": backbone, "seed": seed, "method_id": method,
                    "n_fake": int(is_method.sum()), "n_real": int(is_real.sum()),
                    "auc_raw": auc_raw_m, "auc_direction": auc_dir_m,
                    "delta": auc_dir_m - auc_raw_m,
                })
    return pd.DataFrame(pooled_rows), pd.DataFrame(method_rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir", help="diagnosis run directory containing per_video_scores.parquet")
    ap.add_argument("--out-dir", default=None, help="where to write the CSVs (default: run_dir)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pv = pd.read_parquet(
        run_dir / "per_video_scores.parquet",
        columns=["backbone", "seed", "train_mode", "infer_mode", "domain",
                 "method_id", "video_id", "video_prob_fake"],
    )
    pooled, per_method = compute(pv)
    pooled.to_csv(out_dir / "cdf_delta_auc_pooled.csv", index=False)
    per_method.to_csv(out_dir / "cdf_delta_auc_per_method.csv", index=False)

    print(f"wrote {out_dir / 'cdf_delta_auc_pooled.csv'}  ({len(pooled)} rows)")
    print(f"wrote {out_dir / 'cdf_delta_auc_per_method.csv'}  ({len(per_method)} rows)")
    print(f"pooled: direction>raw in {int((pooled['delta'] > 0).sum())}/{len(pooled)} "
          f"checkpoints, mean delta = {pooled['delta'].mean():+.5f}")
    print(f"per method: direction>raw in {int((per_method['delta'] > 0).sum())}/{len(per_method)} "
          f"cells, mean delta = {per_method['delta'].mean():+.5f}")


if __name__ == "__main__":
    main()
