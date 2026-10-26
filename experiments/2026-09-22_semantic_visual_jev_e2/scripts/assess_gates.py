"""Load final E2 artifacts and write the locked three-seed gate report."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from evaluation.e2_gates import CONDITIONS, DOMAINS, SEEDS, THRESHOLDS, evaluate_gates
from config_io import load_config


def _read(path, seed, domain, condition=None):
    if not path.is_file():
        raise ValueError(f"missing required artifact: {path}")
    if path.suffix == ".json":
        records = json.loads(path.read_text())
        if not isinstance(records, list):
            raise ValueError(f"metrics must be a JSON record list: {path}")
        frame = pd.DataFrame(records)
    else:
        # Video identifiers (including leading zeroes) are strings, never numbers.
        frame = pd.read_csv(path, keep_default_na=False, dtype={
            column: str for column in ("video_id", "video_key", "method_id", "variant_id",
                                       "question_id", "question_pair_id", "pairing_key")})
    if frame.empty:
        raise ValueError(f"missing evidence in empty artifact: {path}")
    for column, value in (("seed", seed), ("domain", domain), ("condition", condition)):
        if value is None:
            continue
        if column in frame and not frame[column].eq(value).all():
            raise ValueError(f"{column} metadata disagrees with path: {path}")
        frame[column] = value
    frame["source_file"] = str(path.resolve())
    return frame


def load_artifacts(evaluation_root, baseline_root):
    roots = Path(evaluation_root), Path(baseline_root)
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"missing artifact root: {root}")
        found = {p.name for p in root.glob("seed_*") if p.is_dir()}
        expected = {f"seed_{seed}" for seed in SEEDS}
        if found != expected:
            raise ValueError(f"{root}: exactly fixed seed directories {sorted(expected)} required; found {sorted(found)}")
    output = {name: [] for name in ("metric_rows", "control_rows", "permutation_rows", "baseline_rows", "candidate_rows", "semantic_rows")}
    dfd_audited = any((roots[0] / f"seed_{seed}" / "permutation/dfd_metrics.json").exists() for seed in SEEDS)
    for seed in SEEDS:
        if dfd_audited:
            output["permutation_rows"].append(_read(roots[0] / f"seed_{seed}" / "permutation/dfd_metrics.json", seed, "dfd", "permutation"))
        for domain in DOMAINS:
            output["baseline_rows"].append(_read(roots[1] / f"seed_{seed}" / f"{domain}_video_predictions.csv", seed, domain))
            directory = roots[0] / f"seed_{seed}"
            for condition in CONDITIONS:
                metrics = _read(directory / condition / f"{domain}_metrics.json", seed, domain, condition)
                output["permutation_rows" if condition == "permutation" else "metric_rows"].append(metrics)
                if condition == "fixed_h1":
                    output["candidate_rows"].append(_read(directory / condition / f"{domain}_videos.csv", seed, domain, condition))
                elif condition in ("seen", "heldout"):
                    output["semantic_rows"].append(_read(directory / condition / f"{domain}_videos.csv", seed, domain, condition))
                elif condition == "counterfactual" or condition.startswith("meaningless_"):
                    suffix = "controls" if condition == "counterfactual" else "paired_videos"
                    output["control_rows"].append(_read(directory / condition / f"{domain}_{suffix}.csv", seed, domain, condition))
    return {name: pd.concat(frames, ignore_index=True) for name, frames in output.items()}


def _diagnostic_table(frame):
    if frame.empty:
        return "No applicable diagnostic rows were supplied.\n"
    columns = [name for name in ("seed", "domain", "condition", "task_id", "method_id", "metric", "value", "count", "source_file") if name in frame]
    return "```csv\n" + frame[columns].to_csv(index=False, na_rep="NA") + "```\n"


def write_report(report, artifacts, output_root, report_path):
    output_root, report_path = Path(output_root), Path(report_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    (output_root / "gate_status.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    pd.DataFrame([dict(name=gate.name, status=gate.status, point_passed=gate.point_passed,
                       ci_passed=gate.ci_passed, evidence=json.dumps(gate.evidence, allow_nan=False))
                  for gate in report.gates.values()]).to_csv(output_root / "gate_status.csv", index=False)
    # Preserve seed-wise primary and secondary measurements plus adjudicated deltas.
    raw = pd.concat([artifacts["metric_rows"], artifacts["permutation_rows"]], ignore_index=True)
    seed_rows = pd.concat([raw, pd.DataFrame(report.seed_metrics)], ignore_index=True)
    grouping = [c for c in ("domain", "condition", "task_id", "method_id", "metric") if c in raw]
    raw_sd = raw.groupby(grouping, dropna=False).value.transform("std")
    seed_rows.loc[:len(raw)-1, "seed_sd"] = raw_sd.to_numpy()
    pd.DataFrame(seed_rows).to_csv(output_root / "seed_metrics.csv", index=False)
    pd.DataFrame(report.bootstrap_intervals).to_csv(output_root / "bootstrap_intervals.csv", index=False)
    lines = ["# E2 validation result", "", f"Verdict: **{report.verdict}**", "",
             "Final preregistered adjudication." if report.final else "NONFINAL diagnostic run; this verdict cannot establish final support.", "",
             f"Fixed seeds: {', '.join(map(str, SEEDS))}. Primary domains: ffpp, cdf. Bootstrap resamples: {report.n_resamples}.", "",
             "The same label-and-method-stratified physical-video multiplicities are used for both paired conditions and all three fixed seeds. All language rows stay inside each video cluster. Seeds are not resampled. Seed SD uses ddof=1.", "",
             "G1 lower bounds use the 5th percentile (one-sided 95%); G6 uses the 2.5th percentile (two-sided 95%). The displayed upper bound is the 97.5th percentile. G1 requires lower >= -0.01; G6 requires lower > 0. Any point failure overrides confidence-only uncertainty.", "",
             "## Exact thresholds", "", "```json", json.dumps(report.thresholds, indent=2), "```", "",
             "G3 additionally requires prediction consistency exactly 1.0 for every seed/domain audit.", "",
             "DFD remains exploratory for semantic performance. If any DFD permutation artifact exists, all three seeds are required and that audit also contributes to G3.", "",
             "## Gate evidence", ""]
    for name, gate in report.gates.items():
        lines.extend([f"### {name}: {gate.status}", "", "```json", json.dumps(asdict(gate), indent=2, allow_nan=False), "```", ""])
    metrics = artifacts["metric_rows"]
    diagnostics = metrics.loc[metrics.method_id.notna() & metrics.method_id.ne("__all__")]
    calibration = metrics.loc[metrics.metric.isin(["brier", "ece"])]
    lines.extend(["## Per-method diagnostics", "", "Undefined single-class AUROC is printed as NA and is not primary gate evidence.", "", _diagnostic_table(diagnostics),
                  "## Calibration diagnostics", "", "Brier and ECE are secondary diagnostics; they do not override gate failures.", "", _diagnostic_table(calibration),
                  "## Source files", "", *[f"- {path}" for path in report.sources], ""])
    report_path.write_text("\n".join(lines))


def assess_run(evaluation_root, baseline_root, output_root, report_path, *, cfg=None):
    artifacts = load_artifacts(evaluation_root, baseline_root)
    report = evaluate_gates(**artifacts, cfg=cfg)
    write_report(report, artifacts, output_root, report_path)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if not isinstance(cfg, dict) or cfg.get("final", True) is not True:
            raise ValueError("CLI config must describe final preregistered adjudication")
        report = assess_run(args.evaluation_root, args.baseline_root, args.output_root, args.report, cfg=cfg)
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        # Fail closed, and replace any stale supported status at this output path.
        payload = dict(verdict="insufficient_evidence", adjudicated=False,
                       error=str(exc), gates={}, thresholds=THRESHOLDS,
                       final=True, n_resamples=10000)
        args.output_root.mkdir(parents=True, exist_ok=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        (args.output_root / "gate_status.json").write_text(json.dumps(payload, indent=2) + "\n")
        pd.DataFrame([dict(status="invalid_evidence", error=str(exc))]).to_csv(args.output_root / "gate_status.csv", index=False)
        pd.DataFrame(columns=["gate", "evidence", "point", "lower", "upper"]).to_csv(args.output_root / "bootstrap_intervals.csv", index=False)
        pd.DataFrame(columns=["seed", "metric", "value"]).to_csv(args.output_root / "seed_metrics.csv", index=False)
        args.report.write_text(f"# E2 validation result\n\nVerdict: **insufficient_evidence**\n\nAdjudication was rejected because required evidence is missing or invalid:\n\n{exc}\n")
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(dict(verdict=report.verdict, final=report.final, n_resamples=report.n_resamples)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
