from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def probability_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fnr - fpr)))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "acc_at_0_5": float(accuracy_score(labels, scores >= 0.5)),
        "eer": float((fpr[index] + fnr[index]) / 2.0),
        "count": int(len(labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    source = Path(args.parquet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filters = [
        ("backbone", "=", "clip"),
        ("seed", "=", 42),
        ("train_mode", "=", "raw"),
        ("infer_mode", "=", "raw"),
    ]
    columns = [
        "checkpoint_id", "domain", "method_id", "label", "video_id",
        "image_path", "prob_fake",
    ]
    frames = pq.read_table(source, columns=columns, filters=filters).to_pandas()
    checkpoint_ids = frames["checkpoint_id"].unique().tolist()
    if checkpoint_ids != ["clip_raw_s42"]:
        raise ValueError(f"Unexpected cached checkpoint IDs: {checkpoint_ids}")

    result = {
        "model_id": "baseline_a_clip_raw_s42",
        "cache": str(source.resolve()),
        "checkpoint_id": checkpoint_ids[0],
        "deduplication_key": ["image_path", "label", "video_id"],
        "video_group_key": ["method_id", "video_id", "label"],
        "domains": {},
    }
    for domain in ("ffpp", "cdf"):
        domain_frames = frames[frames["domain"] == domain].drop_duplicates(
            subset=["image_path", "label", "video_id"],
            keep="first",
        )
        video_rows = (
            domain_frames
            .groupby(["method_id", "video_id", "label"], as_index=False, sort=False)
            .agg(prob_fake=("prob_fake", "mean"), num_frames=("prob_fake", "size"))
        )
        video_rows.insert(
            0,
            "video_key",
            video_rows["method_id"].astype(str) + "::" + video_rows["video_id"].astype(str),
        )
        result["domains"][domain] = {
            "frame": probability_metrics(
                domain_frames["label"].to_numpy(dtype=np.int64),
                domain_frames["prob_fake"].to_numpy(dtype=np.float64),
            ),
            "video": probability_metrics(
                video_rows["label"].to_numpy(dtype=np.int64),
                video_rows["prob_fake"].to_numpy(dtype=np.float64),
            ),
            "rows_before_dedup": int((frames["domain"] == domain).sum()),
            "rows_after_dedup": int(len(domain_frames)),
        }
        video_rows.to_csv(output_dir / f"{domain}_video_predictions.csv", index=False)

    (output_dir / "evaluation.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
