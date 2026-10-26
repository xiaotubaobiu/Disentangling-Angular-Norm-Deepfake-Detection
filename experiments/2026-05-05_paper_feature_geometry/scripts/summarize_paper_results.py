#!/usr/bin/env python3
"""Summarize video-level AUC results from a retained results CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csv",
        default=str(Path(__file__).resolve().parents[1] / "results" / "auc_video_level_60rows.csv"),
        help="CSV with columns: backbone, train_mode, seed, inference_mode, ffpp_auc, cdf_auc, dfd_auc",
    )
    args = parser.parse_args()

    path = Path(args.results_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing results CSV: {path}. Provide --results-csv pointing to a local retained results table."
        )

    df = pd.read_csv(path)
    auc_cols = ["ffpp_auc", "cdf_auc", "dfd_auc"]
    summary = (
        df.groupby(["backbone", "train_mode", "inference_mode"], as_index=False)[auc_cols]
        .agg(["mean", "std"])
        .round(6)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
