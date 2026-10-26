from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation.model_comparison import compare_paired_video_predictions


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["prob_fake"] = float(row["prob_fake"])
        row["num_frames"] = int(row["num_frames"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir)
    candidate_dir = Path(args.candidate_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comparisons = {}
    for domain in ("ffpp", "cdf"):
        comparisons[domain] = compare_paired_video_predictions(
            read_rows(baseline_dir / f"{domain}_video_predictions.csv"),
            read_rows(candidate_dir / f"{domain}_video_predictions.csv"),
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
            noninferiority_margin=0.01,
        )

    swap_check = json.loads((candidate_dir / "evaluation.json").read_text())["domains"]["ffpp"]["candidate_swap_check"]
    if not swap_check or not swap_check["passed"]:
        verdict = "evidence insufficient"
        reason = "candidate-order correctness check failed"
    elif comparisons["ffpp"]["passes_point_estimate_gate"]:
        verdict = "H1 supported"
        reason = "FF++ video-AUROC point estimate is within the pre-registered 0.01 margin"
    else:
        verdict = "H1 not supported"
        reason = "FF++ video-AUROC point estimate is below the pre-registered -0.01 margin"

    result = {
        "primary_domain": "ffpp",
        "decision_rule": "delta_auc_b_minus_a >= -0.01",
        "verdict": verdict,
        "reason": reason,
        "candidate_swap_check": swap_check,
        "comparisons": comparisons,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
