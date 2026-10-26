from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    training_root = Path(args.training_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed_dir in sorted(training_root.glob("seed_*")):
        candidates = sorted(seed_dir.glob("*/output/meta.json"))
        if not candidates:
            continue
        meta_path = candidates[-1]
        meta = json.loads(meta_path.read_text())
        rows.append({
            "seed": int(seed_dir.name.split("_", 1)[1]),
            "meta_path": str(meta_path.resolve()),
            "best_epoch": int(meta["best_epoch"]),
            "val_auc": float(meta["final_val_auc"]),
            "ffpp_video_auc": float(meta["ffpp_test_auc"]),
            "cdf_video_auc": float(meta["cdf_test_auc"]),
            "dfd_video_auc": float(meta["dfd_test_auc"]),
        })
    if not rows:
        raise FileNotFoundError(f"No completed seed metadata under {training_root}")

    fields = list(rows[0])
    with (output_dir / "seed_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metric_names = ["val_auc", "ffpp_video_auc", "cdf_video_auc", "dfd_video_auc"]
    summary = {
        "num_seeds": len(rows),
        "seeds": [row["seed"] for row in rows],
        "metrics": {},
    }
    for name in metric_names:
        values = [row[name] for row in rows]
        summary["metrics"][name] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "values": values,
        }
    (output_dir / "seed_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
