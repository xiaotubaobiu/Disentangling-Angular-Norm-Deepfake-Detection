#!/usr/bin/env python3
"""Export supplementary E2 report tables on CPU, without changing gate results.

Defaults are relative to this script, so the command works from any directory.
All required inputs must exist and validate before any report table is written.
Frame NPZ files are opened individually; only the columns needed for a metric
are materialized. Frame rows are rendered frame x language-variant observations,
not independent physical frames, and never participate in a scientific gate.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)


SEEDS = (42, 7, 2048)
DOMAINS = ("ffpp", "cdf", "dfd")
PRIMARY = {"T1": "auroc", "T2": "balanced_accuracy", "T3": "auroc",
           "T4": "balanced_accuracy", "T5": "macro_balanced_accuracy"}
FIXED_METRICS = (
    "auroc", "accuracy_at_0_5", "balanced_accuracy_at_0_5",
    "real_recall_at_0_5", "fake_recall_at_0_5", "brier", "ap",
    "majority_accuracy",
)


def tasks(domain, condition):
    if condition == "fixed_h1":
        return ("T1",)
    return ("T1", "T2", "T3") if domain == "dfd" else tuple(PRIMARY)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite_probability(values, context):
    values = np.asarray(values, dtype=float)
    require(np.isfinite(values).all(), f"{context}: nonfinite probability")
    require(((values >= 0) & (values <= 1)).all(), f"{context}: probability outside [0,1]")
    return values


def read_fixed(path, seed, domain, model):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_key", "method_id", "video_id", "label", "prob_fake", "num_frames"}
        require(required.issubset(reader.fieldnames or []), f"{path}: missing columns")
        records = list(reader)
    require(bool(records), f"{path}: empty predictions")
    identities = {}
    labels, scores, frame_counts = [], [], []
    for row in records:
        key = row["video_key"]
        require(key == row["method_id"] + "::" + row["video_id"], f"{path}: malformed video key")
        require(key not in identities, f"{path}: duplicate video {key}")
        require(row["label"] in {"0", "1"}, f"{path}: invalid binary label")
        count = int(row["num_frames"])
        require(count > 0, f"{path}: nonpositive frame count")
        if model == "E2":
            expected = {"seed": str(seed), "domain": domain, "condition": "fixed_h1", "task_id": "T1"}
            require(all(row.get(k) == v for k, v in expected.items()), f"{path}: wrong E2 context")
        label = int(row["label"])
        identities[key] = (label, count)
        labels.append(label)
        scores.append(float(row["prob_fake"]))
        frame_counts.append(count)
    y = np.asarray(labels)
    p = finite_probability(scores, path)
    require(set(y) == {0, 1}, f"{path}: fixed-video primary population lacks a class")
    predicted = p >= 0.5
    tn, fp, fn, tp = [int(v) for v in confusion_matrix(y, predicted, labels=[0, 1]).ravel()]
    row = dict(model=model, seed=seed, domain=domain, condition="fixed_h1", task_id="T1",
               count=len(y), n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()),
               frame_count=sum(frame_counts), tn=tn, fp=fp, fn=fn, tp=tp,
               auroc=float(roc_auc_score(y, p)), accuracy_at_0_5=float(accuracy_score(y, predicted)),
               balanced_accuracy_at_0_5=float(balanced_accuracy_score(y, predicted)),
               real_recall_at_0_5=float(recall_score(y, predicted, pos_label=0)),
               fake_recall_at_0_5=float(recall_score(y, predicted, pos_label=1)),
               brier=float(brier_score_loss(y, p)), ap=float(average_precision_score(y, p)),
               majority_accuracy=float(max((y == 0).sum(), (y == 1).sum()) / len(y)),
               unit="physical_video", source_file=str(path))
    return row, identities


def read_metric_json(path, seed, domain, condition):
    records = json.loads(path.read_text())
    require(isinstance(records, list) and records, f"{path}: expected nonempty metric list")
    expected = {"seed": seed, "domain": domain, "condition": condition}
    required = {"seed", "domain", "condition", "task_id", "method_id", "metric", "count", "value"}
    for row in records:
        require(isinstance(row, dict) and required.issubset(row), f"{path}: missing metric fields")
        require(all(row[k] == v for k, v in expected.items()), f"{path}: wrong metric context")
    require({r["task_id"] for r in records} == set(tasks(domain, condition)), f"{path}: task set differs")
    return records


def one_overall(records, task, metric, path):
    rows = [r for r in records if r["task_id"] == task and r["metric"] == metric
            and r["method_id"] == "__all__"]
    require(len(rows) == 1, f"{path}: expected exactly one overall {task}/{metric}")
    row = rows[0]
    value = row["value"]
    require(isinstance(value, (int, float)) and math.isfinite(value) and 0 <= value <= 1,
            f"{path}: undefined or invalid primary metric {task}/{metric}")
    require(isinstance(row["count"], int) and row["count"] > 0, f"{path}: invalid metric count")
    return row


def semantic_video_rows(path, seed, domain, condition):
    records = read_metric_json(path, seed, domain, condition)
    output = []
    for task in tasks(domain, condition):
        row = one_overall(records, task, PRIMARY[task], path)
        output.append(dict(model="E2", seed=seed, domain=domain, condition=condition,
                           task_id=task, metric=PRIMARY[task], value=row["value"], count=row["count"],
                           unit="video_x_language_variant", status="ok", na_reason="",
                           source_file=str(path)))
    return output


def frame_rows(path, seed, domain, condition):
    output = []
    with np.load(path, allow_pickle=False) as archive:
        require("task_id" in archive.files, f"{path}: missing task_id")
        task_ids = archive["task_id"]
        require(task_ids.ndim == 1 and len(task_ids) > 0, f"{path}: invalid task_id column")
        n = len(task_ids)

        def column(name, selection=None):
            require(name in archive.files, f"{path}: missing column {name}")
            values = archive[name]
            require(values.ndim == 1 and len(values) == n, f"{path}: wrong shape for {name}")
            return values if selection is None else values[selection]

        for name, expected in (("seed", seed), ("domain", domain), ("condition", condition)):
            require(np.all(column(name) == expected), f"{path}: wrong {name} context")
        require(set(np.unique(task_ids)) == set(tasks(domain, condition)), f"{path}: task set differs")

        for task in tasks(domain, condition):
            selected = task_ids == task
            row_count = int(selected.sum())
            expected_classes = (0, 1)
            if task in {"T1", "T2", "T3"}:
                truth = column("label", selected)
                require(np.isin(truth, (0, 1)).all(), f"{path}: invalid {task} label")
                probability = finite_probability(column("prob_fake", selected), f"{path}/{task}")
                predicted = probability >= 0.5
            elif task == "T4":
                target = column("target_semantic", selected)
                require(np.isin(target, ("yes", "no")).all(), f"{path}: invalid T4 target")
                truth = (target == "yes").astype(int)
                probability = finite_probability(column("p_yes", selected), f"{path}/T4")
                predicted = probability >= 0.5
            else:
                expected_classes = ("camera", "swap", "reenact")
                truth = column("method_family", selected)
                require(np.isin(truth, expected_classes).all(), f"{path}: invalid T5 family")
                probability = np.column_stack([
                    finite_probability(column(f"p_{name}", selected), f"{path}/T5/{name}")
                    for name in expected_classes
                ])
                require(np.allclose(probability.sum(axis=1), 1.0, rtol=0, atol=1e-6),
                        f"{path}: T5 candidate probabilities do not sum to one")
                predicted = np.asarray(expected_classes)[probability.argmax(axis=1)]
            observed, counts = np.unique(truth, return_counts=True)
            missing = [str(c) for c in expected_classes if c not in set(observed)]
            reason = ""
            value = None
            if missing:
                reason = "缺少真实标签类别 " + ",".join(missing) + "；次要帧指标不定义，未按现有类别缩减平均。"
            elif task in {"T1", "T3"}:
                value = float(roc_auc_score(truth, probability))
            else:
                value = float(balanced_accuracy_score(truth, predicted))
            output.append(dict(model="E2", seed=seed, domain=domain, condition=condition,
                               task_id=task, metric=PRIMARY[task], value=value, count=row_count,
                               unit="frame_x_language_variant", status="NA" if missing else "ok",
                               na_reason=reason,
                               class_counts=json.dumps({str(k): int(v) for k, v in zip(observed, counts)},
                                                       sort_keys=True),
                               used_for_gate=False, source_file=str(path)))
            del truth, probability, predicted, selected
    return output


def summarize(rows):
    keys = ("model", "domain", "condition", "task_id", "metric", "unit")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, group in grouped.items():
        require(len(group) == len(SEEDS) and {r["seed"] for r in group} == set(SEEDS),
                f"summary {identity}: exactly three distinct fixed seeds required")
        values = [r["value"] for r in group if r["value"] is not None]
        complete = len(values) == len(SEEDS)
        reasons = sorted({r.get("na_reason", "") for r in group if r.get("na_reason")})
        output.append(dict(zip(keys, identity), mean=statistics.mean(values) if complete else None,
                           sample_sd=statistics.stdev(values) if complete else None,
                           n_seeds=len(group), n_valid_seeds=len(values),
                           count_min=min(r["count"] for r in group), count_max=max(r["count"] for r in group),
                           status="ok" if complete else "NA",
                           na_reason="；".join(reasons) if not complete else ""))
    return output


def write_csv(path, rows):
    require(bool(rows), f"{path}: refusing to write an empty table")
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: "NA" if value is None else value for key, value in row.items()} for row in rows)
    os.replace(temporary, path)


def main(argv=None):
    result_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, default=result_root / "evaluation")
    parser.add_argument("--baseline-root", type=Path, default=result_root / "baseline_h1")
    parser.add_argument("--output-dir", type=Path, default=result_root / "summary")
    parser.add_argument("--primary-only", action="store_true", help="report only FF++ and CDF")
    args = parser.parse_args(argv)
    domains = DOMAINS[:2] if args.primary_only else DOMAINS
    evaluation, baseline, destination = (path.resolve() for path in
                                         (args.evaluation_root, args.baseline_root, args.output_dir))

    required = []
    for seed in SEEDS:
        for domain in ("ffpp", "cdf"):
            required.extend((baseline / f"seed_{seed}/{domain}_video_predictions.csv",
                             evaluation / f"seed_{seed}/fixed_h1/{domain}_videos.csv",
                             evaluation / f"seed_{seed}/fixed_h1/{domain}_metrics.json"))
        for domain in domains:
            for condition in ("fixed_h1", "seen", "heldout"):
                required.append(evaluation / f"seed_{seed}/{condition}/{domain}_frames.npz")
                if condition != "fixed_h1":
                    required.append(evaluation / f"seed_{seed}/{condition}/{domain}_metrics.json")
    missing = [str(path) for path in required if not path.is_file()]
    require(not missing, f"缺少 {len(missing)}/{len(required)} 个必要输入；未生成报告表。\n" + "\n".join(missing))
    input_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in required}

    fixed, semantic, frames = [], [], []
    for seed in SEEDS:
        for domain in ("ffpp", "cdf"):
            h1, h1_ids = read_fixed(baseline / f"seed_{seed}/{domain}_video_predictions.csv", seed, domain, "H1")
            e2, e2_ids = read_fixed(evaluation / f"seed_{seed}/fixed_h1/{domain}_videos.csv", seed, domain, "E2")
            require(h1_ids == e2_ids, f"seed {seed}/{domain}: H1/E2 video keys, labels or frame counts differ")
            metric_path = evaluation / f"seed_{seed}/fixed_h1/{domain}_metrics.json"
            records = read_metric_json(metric_path, seed, domain, "fixed_h1")
            for metric, column_name in (("auroc", "auroc"), ("accuracy", "accuracy_at_0_5"),
                                        ("brier", "brier"), ("ap", "ap")):
                row = one_overall(records, "T1", metric, metric_path)
                require(row["count"] == e2["count"] and abs(row["value"] - e2[column_name]) <= 1e-10,
                        f"{metric_path}: saved {metric} disagrees with recomputed videos")
            fixed.extend((h1, e2))
        for domain in domains:
            for condition in ("seen", "heldout"):
                semantic.extend(semantic_video_rows(
                    evaluation / f"seed_{seed}/{condition}/{domain}_metrics.json", seed, domain, condition))
            for condition in ("fixed_h1", "seen", "heldout"):
                path = evaluation / f"seed_{seed}/{condition}/{domain}_frames.npz"
                print(f"读取次要帧指标：seed={seed} domain={domain} condition={condition}", flush=True)
                frames.extend(frame_rows(path, seed, domain, condition))

    fixed_long = [dict(model=row["model"], seed=row["seed"], domain=row["domain"],
                       condition=row["condition"], task_id=row["task_id"], metric=metric,
                       value=row[metric], count=row["count"], unit=row["unit"])
                  for row in fixed for metric in FIXED_METRICS]
    tables = {
        "report_fixed_video_by_seed.csv": fixed,
        "report_fixed_video_summary.csv": summarize(fixed_long),
        "report_semantic_video_by_seed.csv": semantic,
        "report_semantic_video_summary.csv": summarize(semantic),
        "report_semantic_frame_by_seed.csv": frames,
        "report_semantic_frame_summary.csv": summarize(frames),
    }
    for path in required:
        require(input_stats[str(path)] == (path.stat().st_size, path.stat().st_mtime_ns),
                f"{path}: input changed during export; no tables written")
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()), "seeds": list(SEEDS),
        "domains": list(domains),
        "evaluation_root": str(evaluation), "baseline_root": str(baseline),
        "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
        "binary_threshold": "probability >= 0.5 predicts positive",
        "seed_summary": "三个固定种子的指标算术均值和样本标准差（ddof=1）；不是集成指标或置信区间。",
        "fixed_video_unit": "物理视频；与H1按method_id::video_id、标签、帧数严格配对。",
        "semantic_video_unit": "沿用评估JSON的video×语言变体聚合行；count不是独立物理视频数。",
        "frame_unit": "帧×语言变体计分单位；同一物理帧可出现多行；所有帧指标仅供次要描述，不用于gate。",
        "frame_metric_rules": {"T1/T3": "label和prob_fake的AUROC",
                               "T2": "label与映射后的prob_fake>=0.5的balanced accuracy",
                               "T4": "target_semantic==yes与p_yes>=0.5的balanced accuracy",
                               "T5": "method_family与argmax(p_camera,p_swap,p_reenact)的macro balanced accuracy"},
        "limitations": ["H1无逐帧NPZ，因此帧表仅包含E2。"] + ([] if args.primary_only else ["DFD仅支持T1-T3，作为探索域。"]) + [
                        "次要帧指标缺真实标签类别则NA；任一种子NA时均值和SD也NA，不忽略缺失种子。",
                        "不修改、不替代assess_gates.py的正式gate判决；本脚本不生成gate结论。"],
        "undefined_secondary_rows": sum(row["status"] == "NA" for row in frames),
        "inputs": [{"path": str(path), "bytes": input_stats[str(path)][0],
                    "mtime_ns": input_stats[str(path)][1]} for path in required],
        "outputs": {name: {"rows": len(rows), "columns": list(rows[0])} for name, rows in tables.items()},
    }
    destination.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(destination / name, rows)
    manifest_path = destination / "report_tables_manifest.json"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, manifest_path)
    print(json.dumps({"output_dir": str(destination), "rows": {k: len(v) for k, v in tables.items()},
                      "undefined_secondary_rows": manifest["undefined_secondary_rows"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"汇总失败：{error}", file=sys.stderr)
        raise SystemExit(2)
