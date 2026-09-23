"""Exploratory per-method AUROC against the same domain's real videos."""
from collections import defaultdict
import argparse
import csv
import hashlib
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2] / "results"
OUT = ROOT / "progress/fixed_method_diagnostic.csv"
META = ROOT / "progress/fixed_method_diagnostic_manifest.json"


def read(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({r["video_key"] for r in rows}) != len(rows):
        raise ValueError(f"empty or duplicate video predictions: {path}")
    identity = {r["video_key"]: (r["label"], r["num_frames"]) for r in rows}
    if any(r["video_key"] != r["method_id"] + "::" + r["video_id"] for r in rows):
        raise ValueError(f"malformed video identity: {path}")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return rows, identity, {"path": str(path.resolve()), "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=ROOT / "evaluation")
    args = parser.parse_args()
    evaluation_root = args.evaluation_root.resolve()
    output, sources = [], []
    for seed in (42, 7, 2048):
        for domain in ("ffpp", "cdf"):
            paths = {"H1": ROOT / f"baseline_h1/seed_{seed}/{domain}_video_predictions.csv",
                     "E2": evaluation_root / f"seed_{seed}/fixed_h1/{domain}_videos.csv"}
            datasets = {}
            for model, path in paths.items():
                rows, identity, source = read(path)
                datasets[model] = (rows, identity)
                sources.append(source)
            if datasets["H1"][1] != datasets["E2"][1]:
                raise ValueError(f"H1/E2 physical video coverage differs: seed={seed}, domain={domain}")
            methods = defaultdict(list)
            for row in datasets["H1"][0]:
                if row["label"] == "1":
                    methods[row["method_id"]].append(row["video_key"])
            real = [r["video_key"] for r in datasets["H1"][0] if r["label"] == "0"]
            if not real or not methods:
                raise ValueError(f"missing real/fake videos: seed={seed}, domain={domain}")
            by_model = {model: {r["video_key"]: r for r in rows} for model, (rows, _) in datasets.items()}
            for method in sorted(methods):
                keys = real + methods[method]
                scores = {}
                for model in ("H1", "E2"):
                    selected = [by_model[model][key] for key in keys]
                    scores[model] = roc_auc_score([int(r["label"]) for r in selected],
                                                  [float(r["prob_fake"]) for r in selected])
                output.append(dict(seed=seed, domain=domain, method_id=method,
                                   n_real=len(real), n_fake=len(methods[method]),
                                   h1_auroc=scores["H1"], e2_auroc=scores["E2"],
                                   delta_e2_minus_h1=scores["E2"]-scores["H1"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    META.write_text(json.dumps({"status": "descriptive_only", "evaluation_root": str(evaluation_root), "definition": "each fake method versus all physical real videos in the same domain; seed-wise AUROC, no multiplicity correction or confidence interval", "rows": len(output), "sources": sources}, indent=2) + "\n")
    print(f"{len(output)} method×seed×domain rows; CDF negative={sum(r['delta_e2_minus_h1'] < 0 for r in output if r['domain']=='cdf')}/48")


if __name__ == "__main__":
    main()
