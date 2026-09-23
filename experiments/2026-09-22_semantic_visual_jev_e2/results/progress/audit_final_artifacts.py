#!/usr/bin/env python3
"""CPU-only, read-only cross-check of the E2 result matrix.

Use --primary-only after all FF++/CDF groups finish; the default checks all 63
groups. This does not adjudicate or change any
scientific gate. Exit 0 means the requested artifact checks passed; exit 2 means
missing/invalid evidence. Only the JSON audit output is written. No scientific
module is imported and no smoke directory is accepted as an input.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd


E2 = Path(__file__).resolve().parents[2]
REPO = E2.parents[1]
SEEDS = (42, 7, 2048)
DOMAINS = ("ffpp", "cdf", "dfd")
CONDITIONS = ("fixed_h1", "seen", "heldout", "permutation", "counterfactual",
              "meaningless_candidates", "meaningless_questions")
KEYS = ["seed", "domain", "task_id", "variant_id", "question_id",
        "question_yes_means", "question_pair_id", "method_id", "video_id",
        "video_key", "label", "method_family", "target_semantic"]
PROBS = ["p_real", "p_fake", "p_no", "p_yes", "p_camera", "p_swap", "p_reenact", "prob_fake"]
STRINGS = [key for key in KEYS if key not in ("seed", "label")] + ["pairing_key", "condition"]
PRIMARY = {"T1": "auroc", "T2": "balanced_accuracy", "T3": "auroc",
           "T4": "balanced_accuracy", "T5": "macro_balanced_accuracy"}
MEANINGFUL_ATOL = 1e-6  # Separate inference batch shapes can change float32 rounding.


def require(test, message):
    if not test:
        raise ValueError(message)


def columns(frame, names, context):
    missing = set(names) - set(frame.columns)
    require(not missing, f"{context}: missing columns {sorted(missing)}")
    require(not frame.empty, f"{context}: empty evidence")


def expected_tasks(domain, condition):
    tasks = {"T1", "T2", "T3"} if domain == "dfd" else set(PRIMARY)
    if condition == "fixed_h1":
        return {"T1"}
    if condition in ("counterfactual", "meaningless_questions"):
        return tasks & {"T2", "T4"}
    return tasks


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False, dtype={name: str for name in STRINGS})


def align(left, right, keys, context):
    columns(left, keys, context)
    columns(right, keys, context)
    require(not left.duplicated(keys).any(), f"{context}: duplicate left identities")
    require(not right.duplicated(keys).any(), f"{context}: duplicate right identities")
    left = left.sort_values(keys).reset_index(drop=True)
    right = right.sort_values(keys).reset_index(drop=True)
    require(left[keys].equals(right[keys]), f"{context}: row identities/coverage differ")
    return left, right


def numeric_equal(left, right, context, *, atol=0):
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            f"{context}: nonfinite or differently shaped values")
    difference = float(np.max(np.abs(a - b))) if a.size else 0.0
    require(np.allclose(a, b, rtol=0, atol=atol), f"{context}: max absolute error {difference} > {atol}")
    return difference


def validate_videos(frame, seed, domain, condition):
    columns(frame, KEYS + ["condition", "num_frames"] + PROBS, "video predictions")
    for name, value in (("seed", seed), ("domain", domain), ("condition", condition)):
        require(frame[name].eq(value).all(), f"video {name} disagrees with path")
    require(set(frame.task_id) == expected_tasks(domain, condition), "unexpected/missing video tasks")
    require(not frame.duplicated(KEYS).any(), "duplicate video/variant identities")
    require(frame.video_key.eq(frame.method_id + "::" + frame.video_id).all(), "invalid method-qualified video keys")
    require(frame.label.isin([0, 1]).all(), "nonbinary physical labels")
    counts = frame.num_frames.to_numpy(dtype=float)
    require(np.isfinite(counts).all() and (counts > 0).all() and (counts == np.floor(counts)).all(),
            "num_frames must be positive integers")
    probabilities = frame[PROBS].to_numpy(dtype=float)
    require(np.isfinite(probabilities).all() and ((probabilities >= 0) & (probabilities <= 1)).all(),
            "video probabilities must be finite and lie in [0, 1]")
    return {"video_variant_rows": int(len(frame)), "physical_videos": int(frame.video_key.nunique()),
            "tasks": sorted(set(frame.task_id)),
            "rows_by_task": {str(k): int(v) for k, v in frame.groupby("task_id").size().items()},
            "frames_by_task": {str(k): int(v) for k, v in frame.groupby("task_id").num_frames.sum().items()}}


def frame_inventory(path, videos, seed, domain, condition):
    """Inspect all array headers; load only compact metadata, not image paths."""
    headers = {}
    with zipfile.ZipFile(path) as archive:
        require(len(archive.namelist()) == len(set(archive.namelist())), "duplicate NPZ members")
        for name in archive.namelist():
            require(name.endswith(".npy"), f"unexpected NPZ member {name}")
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError(f"unsupported NPY header version {version}")
            require(len(shape) == 1 and shape[0] > 0 and not dtype.hasobject,
                    f"invalid/object-valued NPZ array: {name}")
            headers[name[:-4]] = {"rows": int(shape[0]), "dtype": str(dtype)}
    required = set(KEYS + ["condition", "sample_id", "image_path", "candidate_order"] + PROBS)
    require(required <= set(headers), f"frame archive missing columns {sorted(required - set(headers))}")
    lengths = {item["rows"] for item in headers.values()}
    require(len(lengths) == 1, "frame archive array lengths disagree")
    with np.load(path, allow_pickle=False) as arrays:
        for name, value in (("seed", seed), ("domain", domain), ("condition", condition)):
            require(np.all(arrays[name] == value), f"frame archive {name} disagrees with path")
        frame_meta = pd.DataFrame({name: arrays[name] for name in ("task_id", "video_key")})
    require(set(frame_meta.task_id) == expected_tasks(domain, condition), "unexpected/missing frame tasks")
    actual = frame_meta.groupby(["task_id", "video_key"]).size().rename("num_frames").reset_index()
    expected = videos.groupby(["task_id", "video_key"]).num_frames.sum().reset_index()
    actual, expected = align(actual, expected, ["task_id", "video_key"], "frame/video coverage")
    numeric_equal(actual.num_frames, expected.num_frames, "frame/video counts")
    return {"rendered_frame_rows": lengths.pop(), "array_columns": headers,
            "frame_rows_by_task": {str(k): int(v) for k, v in frame_meta.groupby("task_id").size().items()}}


def physical_coverage(videos, fixed, condition):
    base = fixed[["video_key", "method_id", "video_id", "label", "num_frames"]]
    require(not base.video_key.duplicated().any(), "Fixed-H1 has duplicate physical videos")
    identity = ["video_key", "method_id", "video_id", "label"]
    for task, rows in videos.groupby("task_id"):
        actual = rows.groupby(identity, as_index=False).num_frames.sum()
        actual, reference = align(actual, base, identity, f"{condition}/{task} physical coverage")
        multiplier = 2 if task == "T4" or (condition == "counterfactual" and task == "T2") else 1
        numeric_equal(actual.num_frames, reference.num_frames * multiplier,
                      f"{condition}/{task} total physical frame counts")


def audit_g6(paired, heldout, control, condition, domain):
    """Compare canonical meaningful probabilities; hidden candidate order is excluded.

    Evaluator outputs retain the original meaningful question/variant identities
    for both controls. Counterbalanced pseudoword candidate positions cannot be
    used as join keys; their probabilities have already been canonically mapped.
    """
    columns(paired, KEYS + ["condition", "num_frames", "meaningful_num_frames", "pairing_key"]
            + PROBS + ["meaningful_" + name for name in PROBS], "G6 paired videos")
    require(paired.condition.eq(condition).all(), "G6 paired condition differs")
    require(paired.pairing_key.ne("").all() and not paired.pairing_key.duplicated().any(), "invalid G6 pairing keys")
    tasks = expected_tasks(domain, condition)
    require(set(paired.task_id) == tasks, "G6 paired task coverage differs")
    wanted = heldout.loc[heldout.task_id.isin(tasks)]
    left, right = align(paired, wanted, KEYS, f"{condition} meaningful/heldout identities")
    numeric_equal(left.num_frames, right.num_frames, "G6 control/heldout frame counts")
    numeric_equal(left.meaningful_num_frames, right.num_frames, "G6 meaningful/heldout frame counts")
    errors = {}
    for name in PROBS:
        errors[name] = numeric_equal(left["meaningful_" + name], right[name],
                                     f"G6 meaningful/heldout {name}", atol=MEANINGFUL_ATOL)
    left, right = align(paired, control, KEYS, f"{condition} paired/control identities")
    numeric_equal(left.num_frames, right.num_frames, "G6 paired/control frame counts")
    numeric_equal(left[PROBS], right[PROBS], "G6 paired/control probabilities", atol=1e-10)
    return {"paired_video_variant_rows": int(len(paired)), "tasks": sorted(tasks),
            "meaningful_vs_heldout_max_error": errors, "probability_atol": MEANINGFUL_ATOL,
            "join_excludes_candidate_position": True}


def audit_counterfactual(videos, controls, fixed, domain):
    columns(controls, ["seed", "domain", "condition", "task_id", "video_key", "metric", "value", "count"],
            "counterfactual controls")
    require(set(controls.task_id) == expected_tasks(domain, "counterfactual"), "counterfactual control tasks differ")
    require(controls.condition.eq("counterfactual").all(), "counterfactual control condition differs")
    for name in ("seed", "domain"):
        require(controls[name].eq(videos[name].iloc[0]).all(), f"counterfactual control {name} differs")
    require(not controls.duplicated(["task_id", "video_key"]).any(), "duplicate counterfactual control videos")
    require(controls["count"].eq(1).all(), "counterfactual control counts must be one")
    # Each variant covers a subset of video frames: weight by num_frames before
    # answering the literal yes/no question. Unweighted variant means are wrong.
    weighted = videos.assign(weighted_yes=videos.p_yes * videos.num_frames)
    grouped = weighted.groupby(["task_id", "video_key", "question_yes_means"], as_index=False).agg(
        weighted_yes=("weighted_yes", "sum"), num_frames=("num_frames", "sum"))
    grouped["p_yes"] = grouped.weighted_yes / grouped.num_frames
    output = {}
    for task, polarities, metric in (("T2", ["real", "fake"], "polarity_consistency"),
                                      ("T4", ["swap", "reenact"], "true_family_ordering")):
        if task not in expected_tasks(domain, "counterfactual"):
            continue
        data = grouped.loc[grouped.task_id == task]
        require(set(data.question_yes_means) == set(polarities), f"{task} counterfactual polarities differ")
        pivot = data.pivot(index="video_key", columns="question_yes_means", values="p_yes")
        require(pivot.notna().all().all(), f"{task} incomplete video polarity pairs")
        if task == "T2":
            real_yes, fake_yes = pivot["real"].to_numpy(), pivot["fake"].to_numpy()
            expected = ((real_yes >= .5) != (fake_yes >= .5)) & (((1 - real_yes) >= .5) == (fake_yes >= .5))
            checked = pd.DataFrame({"video_key": pivot.index, "expected": expected.astype(float)})
            near = (np.minimum(np.abs(real_yes - .5), np.abs(fake_yes - .5)) <= MEANINGFUL_ATOL)
            output["t2_near_half_probability_pairs"] = int(near.sum())
        else:
            fake = fixed.loc[fixed.label == 1, ["video_key", "method_family"]].set_index("video_key")
            pivot = pivot.loc[fake.index]
            true_score = np.where(fake.method_family.eq("swap"), pivot["swap"], pivot["reenact"])
            other_score = np.where(fake.method_family.eq("swap"), pivot["reenact"], pivot["swap"])
            checked = pd.DataFrame({"video_key": pivot.index, "expected": (true_score > other_score).astype(float),
                                    "meaningful_score": true_score, "control_score": other_score,
                                    "drop": true_score - other_score})
        selected = controls.loc[controls.task_id == task]
        require(selected.metric.eq(metric).all(), f"{task} wrong counterfactual metric")
        checked, selected = align(checked, selected, ["video_key"], f"{task} counterfactual output coverage")
        numeric_equal(checked.expected, selected.value, f"{task} recomputed counterfactual outcomes")
        if task == "T4":
            for name in ("meaningful_score", "control_score", "drop"):
                numeric_equal(checked[name], selected[name], f"T4 counterfactual {name}", atol=1e-10)
        output[task] = {"physical_video_pairs": int(len(checked)), "recomputed_mean": float(checked.expected.mean())}
    return output


class Audit:
    def __init__(self, domains=DOMAINS, evaluation_root=E2 / "results/evaluation"):
        self.domains = tuple(domains)
        self.evaluation_root = Path(evaluation_root).resolve()
        self.files = {}
        self.groups = []
        self.errors = []
        self.checks = []

    def fingerprint(self, path):
        path = Path(path).resolve()
        key = str(path)
        if key not in self.files:
            before = path.stat()
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            after = path.stat()
            require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
                    f"file changed while hashing: {path}")
            self.files[key] = {"bytes": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": digest}
        return self.files[key]

    def check(self, name, callback):
        try:
            evidence = callback()
            self.checks.append({"name": name, "status": "pass", "evidence": evidence})
        except Exception as exc:
            error = {"check": name, "error": f"{type(exc).__name__}: {exc}"}
            self.errors.append(error)
            self.checks.append({"name": name, "status": "fail", **error})

    def inventory(self):
        missing = []
        for seed in SEEDS:
            for domain in self.domains:
                for condition in CONDITIONS:
                    directory = self.evaluation_root / f"seed_{seed}" / condition
                    suffixes = ["metrics.json"]
                    if condition != "permutation":
                        suffixes += ["videos.csv", "frames.npz"]
                    if condition == "counterfactual":
                        suffixes += ["controls.csv"]
                    if condition.startswith("meaningless_"):
                        suffixes += ["paired_videos.csv"]
                    files = []
                    for suffix in suffixes:
                        path = directory / f"{domain}_{suffix}"
                        exists = path.is_file() and path.stat().st_size > 0
                        files.append({"path": str(path), "exists_nonempty": exists,
                                      "bytes": path.stat().st_size if path.is_file() else None})
                        if not exists:
                            missing.append(str(path))
                    self.groups.append({"seed": seed, "domain": domain, "condition": condition,
                                        "inventory_complete": all(item["exists_nonempty"] for item in files),
                                        "files": files})
        for seed in SEEDS:
            extra = [E2 / f"results/evaluation/seed_{seed}/run_provenance.json",
                     E2 / f"results/baseline_h1/seed_{seed}/evaluation.json"]
            extra += [E2 / f"results/baseline_h1/seed_{seed}/{domain}_video_predictions.csv" for domain in ("ffpp", "cdf")]
            missing.extend(str(path) for path in extra if not path.is_file() or path.stat().st_size == 0)
        for path in [E2 / "results/execution_manifest.json"] + [E2 / f"results/feature_cache/{d}/manifest.json" for d in self.domains]:
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
        if missing:
            self.errors.append({"check": "required_inventory", "error": "missing or empty final artifacts",
                                "missing_paths": missing})
        return not missing

    def h1_provenance(self, seed):
        lock_path = E2 / "results/execution_manifest.json"
        self.fingerprint(lock_path)
        lock = json.loads(lock_path.read_text())
        expected = lock["checkpoints"][f"h1_jev_seed_{seed}"]
        path = E2 / f"results/baseline_h1/seed_{seed}/evaluation.json"
        self.fingerprint(path)
        evaluation = json.loads(path.read_text())
        checkpoint = Path(evaluation["checkpoint"]).resolve()
        require(checkpoint == Path(expected["path"]).resolve(), "H1 evaluation.json checkpoint differs from execution manifest")
        require(f"seed_{seed}" in checkpoint.parts, "H1 checkpoint path has wrong seed")
        require(evaluation.get("model_name") == "clip_visual_jev", "H1 comparator model differs")
        require(evaluation.get("model_id") == f"h1_jev_seed{seed}", "H1 comparator model_id has wrong seed")
        config = Path(evaluation["config"])
        require(config.name == f"visual_jev_s{seed}.yaml", "H1 evaluation config filename has wrong seed")
        require(self.fingerprint(checkpoint)["sha256"] == expected["sha256"], "H1 checkpoint hash differs from execution manifest")
        require({"ffpp", "cdf"} <= set(evaluation["domains"]), "H1 evaluation.json lacks primary domains")
        return {"checkpoint": str(checkpoint), "checkpoint_sha256": expected["sha256"], "model_id": evaluation["model_id"]}

    def e2_provenance(self, seed):
        path = self.evaluation_root / f"seed_{seed}/run_provenance.json"
        self.fingerprint(path)
        data = json.loads(path.read_text())
        require(data.get("seed") == seed and data.get("reportable") is True
                and data.get("smoke") is False and data.get("diagnostic_override") is False,
                "formal provenance is not a matching reportable final seed")
        checkpoint = E2 / f"results/training/seed_{seed}/best_model.pth"
        require(Path(data["checkpoint_path"]).resolve() == checkpoint.resolve(), "unexpected E2 checkpoint path")
        require(self.fingerprint(checkpoint)["sha256"] == data["checkpoint_sha256"], "E2 checkpoint hash differs")
        return {"checkpoint": str(checkpoint), "reportable": True}

    def group(self, entry):
        seed, domain, condition = (entry[key] for key in ("seed", "domain", "condition"))
        paths = {}
        for file in entry["files"]:
            path = Path(file["path"])
            file.update(self.fingerprint(path))
            paths[path.name.removeprefix(domain + "_")] = path
        records = json.loads(paths["metrics.json"].read_text())
        require(isinstance(records, list) and records, "metrics must be a nonempty record list")
        metrics = pd.DataFrame(records)
        columns(metrics, ["seed", "domain", "condition", "task_id", "metric", "value", "count"], "metrics")
        for name, value in (("seed", seed), ("domain", domain), ("condition", condition)):
            require(metrics[name].eq(value).all(), f"metrics {name} disagrees with path")
        entry["metric_rows"] = int(len(metrics))
        if condition == "permutation":
            require(set(metrics.task_id) == {"__all__"}, "permutation task ID differs")
            require(set(metrics.metric) == {"max_probability_error", "prediction_consistency"}
                    and len(metrics) == 2, "permutation metric schema differs")
            manifest_path = E2 / f"results/feature_cache/{domain}/manifest.json"
            self.fingerprint(manifest_path)
            manifest = json.loads(manifest_path.read_text())
            physical = min(4096, int(manifest["shape"][0]))
            expected_count = physical * (6 if domain == "dfd" else 14)
            require(metrics["count"].eq(expected_count).all(), "incomplete exhaustive permutation comparison count")
            require(np.isfinite(metrics.value.astype(float)).all(), "nonfinite permutation metric")
            entry.update(audited_physical_samples=physical, comparison_count=expected_count)
            return None
        require(set(metrics.task_id) == expected_tasks(domain, condition), "unexpected/missing metric tasks")
        videos = read_csv(paths["videos.csv"])
        entry.update(validate_videos(videos, seed, domain, condition))
        for task, task_videos in videos.groupby("task_id"):
            selected = metrics.loc[(metrics.task_id == task) & (metrics.get("method_id", "") == "__all__")
                                   & (metrics.metric == PRIMARY[task])]
            require(len(selected) == 1, f"{task}: primary metric missing/duplicated")
            require(int(selected["count"].iloc[0]) == len(task_videos), f"{task}: primary count differs from video rows")
            value = float(selected.value.iloc[0])
            require(np.isfinite(value) and 0 <= value <= 1, f"{task}: invalid primary metric")
        entry["frames"] = frame_inventory(paths["frames.npz"], videos, seed, domain, condition)
        return videos

    def run(self):
        if not self.inventory():
            return
        for seed in SEEDS:
            self.check(f"seed_{seed}/h1_provenance", lambda s=seed: self.h1_provenance(s))
            self.check(f"seed_{seed}/e2_provenance", lambda s=seed: self.e2_provenance(s))
            for domain in self.domains:
                data = {}
                for entry in self.groups:
                    if entry["seed"] != seed or entry["domain"] != domain:
                        continue
                    condition = entry["condition"]
                    def read_group(item=entry, name=condition):
                        result = self.group(item)
                        if result is not None:
                            data[name] = result
                        return {"metric_rows": item["metric_rows"], "files": len(item["files"])}
                    self.check(f"seed_{seed}/{domain}/{condition}/files_and_counts", read_group)
                if "fixed_h1" not in data:
                    continue
                fixed = data["fixed_h1"]
                for condition, videos in data.items():
                    self.check(f"seed_{seed}/{domain}/{condition}/physical_coverage",
                               lambda v=videos, c=condition: physical_coverage(v, fixed, c))
                if domain != "dfd":
                    def compare_h1():
                        path = E2 / f"results/baseline_h1/seed_{seed}/{domain}_video_predictions.csv"
                        self.fingerprint(path)
                        baseline = read_csv(path)
                        identity = ["video_key", "method_id", "video_id", "label"]
                        left, right = align(fixed, baseline, identity, "Fixed-H1/H1 physical videos")
                        numeric_equal(left.num_frames, right.num_frames, "Fixed-H1/H1 frame counts")
                        return {"paired_physical_videos": len(left), "physical_frames": int(left.num_frames.sum())}
                    self.check(f"seed_{seed}/{domain}/fixed_h1_vs_h1", compare_h1)
                for condition in ("meaningless_candidates", "meaningless_questions"):
                    if condition in data and "heldout" in data:
                        path = self.evaluation_root / f"seed_{seed}/{condition}/{domain}_paired_videos.csv"
                        self.check(f"seed_{seed}/{domain}/{condition}/heldout_crosscheck",
                                   lambda c=condition, p=path: audit_g6(read_csv(p), data["heldout"], data[c], c, domain))
                if "counterfactual" in data:
                    path = self.evaluation_root / f"seed_{seed}/counterfactual/{domain}_controls.csv"
                    self.check(f"seed_{seed}/{domain}/counterfactual_recomputed",
                               lambda: audit_counterfactual(data["counterfactual"], read_csv(path), fixed, domain))
                del data
        for name, recorded in self.files.items():
            path = Path(name)
            if not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != (recorded["bytes"], recorded["mtime_ns"]):
                self.errors.append({"check": "stable_inputs", "error": f"input changed during audit: {name}"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-only", action="store_true", help="audit the 42 required FF++/CDF groups")
    parser.add_argument("--evaluation-root", type=Path, default=E2 / "results/evaluation")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    domains = DOMAINS[:2] if args.primary_only else DOMAINS
    output = (args.output or E2 / "results/progress" /
              ("audit_primary_artifacts.json" if args.primary_only else "audit_final_artifacts.json")).resolve()
    require(output.parent == (E2 / "results/progress").resolve() and output.suffix == ".json",
            "audit output must be a JSON file directly under E2 results/progress")
    audit = Audit(domains, args.evaluation_root)
    try:
        audit.run()
    except Exception as exc:
        audit.errors.append({"check": "audit_execution", "error": f"{type(exc).__name__}: {exc}"})
    payload = {
        "audit_script": str(Path(__file__).resolve()), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "input_root": str(audit.evaluation_root),
        "status": "pass" if not audit.errors else "fail", "domains": list(domains),
        "expected_groups": len(SEEDS) * len(domains) * len(CONDITIONS),
        "inventory_complete_groups": sum(item["inventory_complete"] for item in audit.groups),
        "checks_by_status": dict(Counter(item["status"] for item in audit.checks)),
        "groups": audit.groups, "checks": audit.checks, "errors": audit.errors, "files_sha256": audit.files,
        "limitations": [
            "Complements assess_gates.py; does not replace its locked provenance validation or G1-G6 adjudication.",
            "Hashes identify audited files; H1 prediction files lack a preregistered cryptographic binding to their evaluator/checkpoint. The available evaluation.json and execution-manifest checkpoint binding are checked.",
            "Frame NPZ headers and task/video/context arrays are inspected; full frame probability replay, image-content rehashing, and per-sample text reconstruction are not performed.",
            "Canonical meaningful probabilities may differ by up to 1e-6 across inference batch shapes. Candidate positions are intentionally excluded from meaningless-control joins.",
            "The frozen meaningless-candidate bank is not strictly matched to meaningful-candidate length; this audit cannot remedy that protocol limitation.",
        ],
    }
    output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: payload[key] for key in ("status", "expected_groups", "inventory_complete_groups", "checks_by_status")}
                     | {"error_count": len(audit.errors), "output": str(output)}))
    return 0 if not audit.errors else 2


if __name__ == "__main__":
    sys.exit(main())
