"""Locked E2 gates and paired physical-video bootstrap, conditional on three seeds.

Language rows are retained inside physical clusters. One stratified cluster draw
is shared by both conditions AND all three seeds, preserving their covariance.
Numeric sufficient statistics and sorted score ties are prepared before draws.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Mapping

import numpy as np
import pandas as pd

from evaluation.semantic_eval import GROUP_KEYS, _task_values
from data.semantic_dataset import method_family

SEEDS = (42, 7, 2048)
DOMAINS = ("ffpp", "cdf")
PRIMARY = {"T1": "auroc", "T2": "balanced_accuracy", "T3": "auroc",
           "T4": "balanced_accuracy", "T5": "macro_balanced_accuracy"}
CONDITIONS = ("fixed_h1", "seen", "heldout", "permutation", "counterfactual",
              "meaningless_candidates", "meaningless_questions")
THRESHOLDS = dict(noninferiority_margin=.01, heldout_macro_drop_max=.02,
                  heldout_task_drop_max=.05, permutation_error_max=1e-6,
                  t2_balanced_accuracy_min=.80, t2_polarity_consistency_min=.95,
                  t4_balanced_accuracy_min=.65, t4_counterfactual_order_min=.80,
                  t5_macro_balanced_accuracy_min=.60, meaningless_drop_min=.10)


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lower: float
    upper: float
    n_resamples: int
    n_videos: int
    per_seed: tuple[float, ...]
    seed_sd: float
    lower_quantile: float
    rows_per_seed: tuple[int, ...]
    baseline_per_seed: tuple[float, ...]
    candidate_per_seed: tuple[float, ...]


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    point_passed: bool
    ci_passed: bool | None
    evidence: dict


@dataclass(frozen=True)
class GateReport:
    verdict: str
    gates: dict[str, GateResult]
    thresholds: dict
    n_resamples: int
    final: bool
    seed_metrics: tuple[dict, ...]
    bootstrap_intervals: tuple[dict, ...]
    sources: tuple[str, ...]


def _require(frame, columns, context):
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{context}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{context}: missing evidence")


def _finite(values, context):
    try:
        result = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: must be finite numeric evidence") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{context}: must be finite numeric evidence")
    return result


def _probabilities(frame, columns):
    _require(frame, columns, "probabilities")
    values = _finite(frame[list(columns)], "probabilities")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("probabilities must lie in [0, 1]")
    return values


def _population(frame, context):
    _require(frame, ["seed", "domain", "video_key", "method_id", "video_id", "label"], context)
    if set(frame.seed) != set(SEEDS):
        raise ValueError(f"{context}: exactly fixed seeds {SEEDS} required")
    if frame.domain.nunique() != 1:
        raise ValueError(f"{context}: one domain required per bootstrap")
    labels = _finite(frame.label, context)
    if not np.isin(labels, [0, 1]).all():
        raise ValueError(f"{context}: binary labels required")
    expected = frame.method_id.astype(str) + "::" + frame.video_id.astype(str)
    if not frame.video_key.eq(expected).all():
        raise ValueError(f"{context}: video_key must be method_id::video_id")
    if frame.groupby("video_key").label.nunique().gt(1).any():
        raise ValueError(f"{context}: mismatched labels within video cluster")
    reference = None
    for seed in SEEDS:
        keys = set(frame.loc[frame.seed == seed, "video_key"])
        if reference is not None and keys != reference:
            raise ValueError(f"{context}: video-key sets differ across seeds")
        reference = keys
    return sorted(reference)


class _ClusterMetric:
    """Exact weighted AUROC or class recall, with all language rows retained."""
    def __init__(self, rows, task, cluster_ids, n_clusters):
        self.task = task
        self.ids = np.asarray(cluster_ids, dtype=int)
        self.n_clusters = n_clusters
        if task in ("T1", "T3", "auroc"):
            y = rows.label.to_numpy(dtype=int)
            score = _probabilities(rows, ["prob_fake"])[:, 0]
            if set(y) != {0, 1}:
                raise ValueError("AUROC requires both labels")
            order = np.argsort(score, kind="stable")
            self.ids = self.ids[order]
            self.positive = y[order] == 1
            self.starts = np.r_[0, np.flatnonzero(np.diff(score[order])) + 1]
            return
        if task == "T2":
            truth = rows.label.to_numpy(dtype=int)
            predicted = _probabilities(rows, ["prob_fake"])[:, 0] >= .5
        elif task == "T4":
            _require(rows, ["target_semantic"], task)
            if not rows.target_semantic.isin(["yes", "no"]).all():
                raise ValueError("T4 requires yes/no target semantics")
            truth = (rows.target_semantic == "yes").to_numpy(dtype=int)
            predicted = _probabilities(rows, ["p_yes"])[:, 0] >= .5
        elif task == "T5":
            _require(rows, ["method_family"], task)
            classes = np.array(["camera", "swap", "reenact"])
            if not rows.method_family.isin(classes).all():
                raise ValueError("T5 requires known method families")
            truth = rows.method_family.to_numpy()
            probabilities = _probabilities(rows, [f"p_{c}" for c in classes])
            predicted = classes[probabilities.argmax(axis=1)]
        else:
            raise ValueError(f"unknown primary metric/task {task}")
        classes = np.unique(truth)
        if len(classes) < 2:
            raise ValueError(f"{task}: primary metric requires at least two classes")
        self.totals = np.array([np.bincount(self.ids, weights=truth == c, minlength=n_clusters) for c in classes])
        self.correct = np.array([np.bincount(self.ids, weights=(truth == c) & (predicted == truth), minlength=n_clusters) for c in classes])

    def __call__(self, weights):
        if self.task in ("T1", "T3", "auroc"):
            row_weights = weights[:, self.ids]
            positives = np.add.reduceat(row_weights * self.positive, self.starts, axis=1)
            negatives = np.add.reduceat(row_weights * ~self.positive, self.starts, axis=1)
            preceding = np.cumsum(negatives, axis=1) - negatives
            return (positives * (preceding + .5 * negatives)).sum(axis=1) / (positives.sum(axis=1) * negatives.sum(axis=1))
        total = np.einsum("bn,cn->bc", weights, self.totals)
        correct = np.einsum("bn,cn->bc", weights, self.correct)
        if (total == 0).any():
            raise ValueError("bootstrap lost a required class; invalid strata")
        return (correct / total).mean(axis=1)


def paired_seed_video_bootstrap(baseline, candidate, metric, n_resamples=10000,
                                seed=20260922, *, lower_quantile=.05):
    """Candidate minus baseline; shared physical draws across the fixed seeds.

    metric is auroc or a task ID (T2/T4 disambiguate balanced accuracy).
    Each input contains exactly one domain and all three fixed seeds.
    """
    baseline, candidate = pd.DataFrame(baseline).copy(), pd.DataFrame(candidate).copy()
    keys = _population(baseline, "baseline")
    if keys != _population(candidate, "candidate"):
        raise ValueError("paired video-key sets differ")
    if set(baseline.domain) != set(candidate.domain):
        raise ValueError("paired domains differ")
    if int(n_resamples) != n_resamples or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if lower_quantile not in (.025, .05):
        raise ValueError("lower_quantile must be .025 or .05")
    full_keys = [c for c in GROUP_KEYS if c not in ("seed", "domain", "condition")]
    semantic = metric != "auroc"
    pair_keys = full_keys if semantic else ["video_key", "label", "method_id", "video_id"]
    _require(baseline, pair_keys, "baseline pairs")
    _require(candidate, pair_keys, "candidate pairs")
    prepared, counts, seed_points, baseline_points, candidate_points = [], [], [], [], []
    key_index = {key: i for i, key in enumerate(keys)}
    for experiment_seed in SEEDS:
        left = baseline.loc[baseline.seed == experiment_seed].sort_values(pair_keys).reset_index(drop=True)
        right = candidate.loc[candidate.seed == experiment_seed].sort_values(pair_keys).reset_index(drop=True)
        if left.duplicated(pair_keys).any() or right.duplicated(pair_keys).any():
            raise ValueError("duplicate paired video/variant rows")
        # CSV type inference must not change identities; callers read IDs as strings.
        left_identity = left[pair_keys].astype(str)
        right_identity = right[pair_keys].astype(str)
        if not left_identity.equals(right_identity):
            raise ValueError("paired video-key sets differ or labels/variant identities mismatch")
        if "num_frames" in left and "num_frames" in right:
            if not np.array_equal(left.num_frames, right.num_frames):
                raise ValueError("paired frame counts differ")
            if (_finite(left.num_frames, "frame counts") <= 0).any():
                raise ValueError("frame counts must be positive")
        ids = left.video_key.map(key_index).to_numpy(dtype=int)
        pair = (_ClusterMetric(left, metric, ids, len(keys)), _ClusterMetric(right, metric, ids, len(keys)))
        prepared.append(pair)
        one = np.ones((1, len(keys)))
        baseline_points.append(float(pair[0](one)[0]))
        candidate_points.append(float(pair[1](one)[0]))
        seed_points.append(candidate_points[-1] - baseline_points[-1])
        counts.append(len(left))
    physical = baseline.drop_duplicates("video_key").set_index("video_key").loc[keys]
    strata = [np.asarray(indices, dtype=int) for indices in physical.reset_index().groupby(["label", "method_id"]).indices.values()]
    rng = np.random.default_rng(seed)
    deltas = np.empty(int(n_resamples))
    # Limit temporary row-weight matrices even for suites with many variants.
    batch_size = max(1, min(64, 1_000_000 // max(counts)))
    for start in range(0, int(n_resamples), batch_size):
        size = min(batch_size, int(n_resamples) - start)
        weights = np.zeros((size, len(keys)), dtype=float)
        for indices in strata:
            weights[:, indices] = rng.multinomial(len(indices), np.full(len(indices), 1 / len(indices)), size=size)
        deltas[start:start + size] = np.mean([right(weights) - left(weights) for left, right in prepared], axis=0)
    _finite(deltas, "bootstrap replicates")
    return BootstrapResult(float(np.mean(seed_points)), float(np.quantile(deltas, lower_quantile)),
                           float(np.quantile(deltas, .975)), int(n_resamples), len(keys),
                           tuple(seed_points), float(np.std(seed_points, ddof=1)),
                           lower_quantile, tuple(counts), tuple(baseline_points), tuple(candidate_points))


def _configuration(cfg):
    cfg = dict(cfg or {})
    for name, value in cfg.get("gates", {}).items():
        if name not in THRESHOLDS or value != THRESHOLDS[name]:
            raise ValueError(f"immutable gate threshold: {name}")
    evaluation = cfg.get("evaluation", {})
    count = cfg.get("n_resamples", evaluation.get("bootstrap_samples", 10000))
    final = cfg.get("final", True)
    if final and count != 10000:
        raise ValueError("final adjudication requires exactly 10,000 resamples")
    if int(count) != count or count < 1:
        raise ValueError("n_resamples must be a positive integer")
    if "primary_domains" in evaluation:
        domains = evaluation["primary_domains"]
        if len(domains) != len(DOMAINS) or set(domains) != set(DOMAINS):
            raise ValueError("primary domains are immutable")
    bootstrap_seed = cfg.get("seed", evaluation.get("bootstrap_seed", 20260922))
    if final:
        if evaluation.get("bootstrap_samples", 10000) != 10000:
            raise ValueError("final adjudication requires exactly 10,000 resamples")
        if bootstrap_seed != 20260922 or evaluation.get("bootstrap_seed", 20260922) != 20260922:
            raise ValueError("final bootstrap seed is locked to 20260922")
    return int(count), int(bootstrap_seed), bool(final)


def _semantic_scores(rows, baseline, task):
    """Verify complete canonical video renders and reuse Task 5 primary metrics.

    The suite renders each physical frame once per task, except T4 which renders
    both family questions. Variant groups partition those frames; their summed
    counts must recover the fixed-H1 physical-video frame count. Variant names
    and the number of populated groups need not agree across seeds/videos.
    """
    _require(rows, list(GROUP_KEYS) + ["num_frames"], f"{task} canonical videos")
    if rows[list(GROUP_KEYS)].isna().any().any():
        raise ValueError("canonical semantic videos have missing variant identities")
    if rows.duplicated(list(GROUP_KEYS)).any():
        raise ValueError("duplicate canonical semantic video/variant rows")
    keys = _population(baseline, "baseline")
    if _population(rows, f"{task} canonical videos") != keys:
        raise ValueError("canonical semantic video-key sets differ from baseline")
    truth = baseline.drop_duplicates("video_key").set_index("video_key").loc[keys, "label"]
    if set(truth) != {0, 1}:
        raise ValueError("primary semantic metrics require both physical labels")
    actual = rows.drop_duplicates("video_key").set_index("video_key").loc[keys, "label"]
    if not np.array_equal(truth, actual):
        raise ValueError("canonical semantic video labels differ from baseline")
    families = {key: method_family(key.split("::", 1)[0], label) for key, label in truth.items()}
    expected_families = rows.video_key.map(families)
    if expected_families.eq("unknown").any() or not rows.method_family.eq(expected_families).all():
        raise ValueError("canonical semantic method families disagree with physical classes")
    if rows.variant_id.eq("").any() or rows.question_id.eq("").any():
        raise ValueError("canonical semantic videos require nonempty variant and question identities")
    counts = _finite(rows.num_frames, "semantic frame counts")
    if (counts <= 0).any() or (counts != np.floor(counts)).any():
        raise ValueError("semantic frame counts must be positive integers")
    physical_truth = np.where(rows.label.to_numpy() == 1, "fake", "real")
    if task in ("T1", "T3"):
        target = physical_truth
        probability_columns = ["p_real", "p_fake"]
        canonical = rows.p_fake if "p_fake" in rows else []
    elif task in ("T2", "T4"):
        polarities = ["real", "fake"] if task == "T2" else ["swap", "reenact"]
        if not rows.question_yes_means.isin(polarities).all():
            raise ValueError(f"{task} has invalid question classes")
        target = np.where(rows.question_yes_means == (physical_truth if task == "T2" else rows.method_family), "yes", "no")
        probability_columns = ["p_no", "p_yes"]
        _require(rows, probability_columns, task)
        canonical = np.where(rows.question_yes_means.eq("fake"), rows.p_yes, rows.p_no) if task == "T2" else None
        if task == "T4":
            if rows.question_pair_id.eq("").any():
                raise ValueError("T4 requires complete family variant pairs")
            pairs = rows.groupby(["seed", "video_key", "question_pair_id"]).agg(
                size=("question_id", "size"), questions=("question_id", "nunique"),
                polarities=("question_yes_means", "nunique"),
                min_frames=("num_frames", "min"), max_frames=("num_frames", "max"))
            if not (pairs["size"].eq(2) & pairs.questions.eq(2) & pairs.polarities.eq(2)
                    & pairs.min_frames.eq(pairs.max_frames)).all():
                raise ValueError("incomplete or mismatched T4 family variant pairs")
    else:
        target = rows.method_family
        probability_columns = ["p_camera", "p_swap", "p_reenact"]
        _require(rows, probability_columns, task)
        canonical = 1 - rows.p_camera.to_numpy(dtype=float)
    if not np.array_equal(rows.target_semantic, target):
        raise ValueError(f"{task} target semantics disagree with physical/question classes")
    probabilities = _probabilities(rows, probability_columns)
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("canonical semantic probabilities must sum to one")
    fake_probabilities = _probabilities(rows, ["prob_fake"])[:, 0]
    if canonical is not None and not np.allclose(fake_probabilities, _finite(canonical, "canonical probability"), rtol=0, atol=1e-6):
        raise ValueError("canonical fake probability differs from semantic probabilities")
    values = []
    for seed in SEEDS:
        selected = rows.loc[rows.seed == seed]
        expected_counts = baseline.loc[baseline.seed == seed].set_index("video_key").loc[keys, "num_frames"]
        actual_counts = selected.groupby("video_key").num_frames.sum().reindex(keys)
        if not np.array_equal(actual_counts, expected_counts * (2 if task == "T4" else 1)):
            raise ValueError(f"{task} variant frame counts differ from physical video coverage")
        values.append(_task_values(task, selected)[PRIMARY[task]])
    return _finite(values, f"{task} recomputed primary scores")


def evaluate_gates(metric_rows, control_rows, permutation_rows, baseline_rows, candidate_rows, cfg=None, *, semantic_rows=None):
    """Adjudicate normalized artifacts; malformed/missing evidence raises ValueError.

    control_rows combines video-level counterfactual records and full paired
    meaningless-control video records (meaningful_ probability columns).
    semantic_rows is required canonical seen/heldout video evidence for G2/G4/G5.
    Diagnostic mode can reduce resamples, and is explicitly marked nonfinal.
    """
    n_resamples, rng_seed, final = _configuration(cfg)
    metrics, controls, permutation, baseline, candidate = [pd.DataFrame(f).copy() for f in
        (metric_rows, control_rows, permutation_rows, baseline_rows, candidate_rows)]
    semantic = pd.DataFrame(semantic_rows).copy()
    for frame, name in ((metrics, "metrics"), (controls, "controls"), (permutation, "permutation"), (baseline, "baseline"), (candidate, "candidate"), (semantic, "canonical semantic videos")):
        _require(frame, ["seed", "domain"], name)
        if set(frame.seed) != set(SEEDS):
            raise ValueError(f"{name}: exactly fixed seeds {SEEDS} required")
        if not set(DOMAINS).issubset(set(frame.domain)):
            raise ValueError(f"{name}: missing primary domains")
    _require(metrics, ["condition", "task_id", "metric", "value", "count", "method_id"], "metrics")
    _require(controls, ["condition", "task_id", "video_key", "pairing_key"], "controls")
    _require(permutation, ["condition", "task_id", "metric", "value", "count"], "permutation")
    _require(semantic, ["condition", "task_id"], "canonical semantic videos")
    if not semantic.condition.isin(["seen", "heldout"]).all() or not semantic.task_id.isin(PRIMARY).all():
        raise ValueError("canonical semantic videos contain unexpected conditions/tasks")
    evidence = {f"G{i}": {} for i in range(1, 7)}
    points = {name: [] for name in evidence}
    cis = {"G1": [], "G6": []}
    seed_metrics, intervals, pending = [], [], []

    def check(gate, key, values, threshold, direction="min"):
        values = _finite(values, key)
        point = float(values.mean())
        evidence[gate][key] = dict(point=point, threshold=threshold, direction=direction,
                                    per_seed=dict(zip(map(str, SEEDS), map(float, values))),
                                    seed_sd=float(values.std(ddof=1)))
        points[gate].append(point >= threshold if direction == "min" else point <= threshold)
        for experiment_seed, value in zip(SEEDS, values):
            seed_metrics.append(dict(gate=gate, evidence=key, seed=experiment_seed, value=float(value)))

    def metric_values(domain, condition, task, metric):
        selected = metrics.loc[(metrics.domain == domain) & (metrics.condition == condition)
                               & (metrics.task_id == task) & (metrics.metric == metric)
                               & (metrics.method_id == "__all__")]
        return scalar_values(selected, f"{domain}/{condition}/{task}/{metric}")

    def scalar_values(selected, context):
        if selected.duplicated("seed").any():
            raise ValueError(f"{context}: duplicate metric evidence")
        if set(selected.seed) != set(SEEDS):
            raise ValueError(f"{context}: missing required metric/seeds")
        selected = selected.set_index("seed").loc[list(SEEDS)]
        counts = _finite(selected["count"], context)
        if (counts <= 0).any() or (counts != np.floor(counts)).any():
            raise ValueError(f"{context}: counts must be positive integers")
        return _finite(selected.value, context)

    def semantic_values(domain, condition, task, base):
        stored_values = metric_values(domain, condition, task, PRIMARY[task])
        stored = metrics.loc[(metrics.domain == domain) & (metrics.condition == condition)
                             & (metrics.task_id == task) & (metrics.metric == PRIMARY[task])
                             & (metrics.method_id == "__all__")].set_index("seed").loc[list(SEEDS)]
        rows = semantic.loc[(semantic.domain == domain) & (semantic.condition == condition) & (semantic.task_id == task)]
        values = _semantic_scores(rows, base, task)
        if not np.array_equal(rows.groupby("seed").size().reindex(SEEDS), stored["count"]):
            raise ValueError(f"{domain}/{condition}/{task}: stored counts differ from canonical video rows")
        if not np.allclose(values, stored_values, rtol=0, atol=1e-10):
            raise ValueError(f"{domain}/{condition}/{task}: stored primary score differs from canonical video probabilities")
        return values

    def permutation_gate(domain):
        perm = permutation.loc[permutation.domain == domain]
        if not perm.condition.eq("permutation").all() or not perm.task_id.eq("__all__").all():
            raise ValueError("permutation evidence must be global audit metrics")
        error = scalar_values(perm.loc[perm.metric == "max_probability_error"], f"{domain} permutation error")
        consistency = scalar_values(perm.loc[perm.metric == "prediction_consistency"], f"{domain} permutation consistency")
        audit_counts = perm.pivot(index="seed", columns="metric", values="count")
        if not audit_counts.max(axis=1).eq(audit_counts.min(axis=1)).all():
            raise ValueError("permutation metric counts differ")
        if (error < 0).any() or ((consistency < 0) | (consistency > 1)).any():
            raise ValueError("invalid permutation evidence")
        # Global extrema, never the mean across seeds or domains.
        evidence["G3"][domain] = dict(max_probability_error=float(error.max()),
                                      prediction_consistency=float(consistency.min()),
                                      comparison_counts=perm.loc[perm.metric == "max_probability_error", ["seed", "count"]].to_dict("records"),
                                      error_threshold=1e-6, consistency_threshold=1.)
        points["G3"].append(bool((error <= 1e-6).all() and (consistency == 1).all()))

    for domain in DOMAINS:
        base = baseline.loc[baseline.domain == domain]
        cand = candidate.loc[candidate.domain == domain]
        _require(base, ["prob_fake", "num_frames"], "baseline")
        _require(cand, ["prob_fake", "num_frames", "task_id", "condition"], "candidate")
        keys = _population(base, "baseline")
        if keys != _population(cand, "candidate"):
            raise ValueError("paired video-key sets differ")
        if "condition" in cand and not cand.condition.eq("fixed_h1").all():
            raise ValueError("G1 candidate must contain only fixed_h1 predictions")
        if not cand.task_id.eq("T1").all():
            raise ValueError("Fixed-H1 predictions must be T1")
        fixed_scores = metric_values(domain, "fixed_h1", "T1", "auroc")
        fixed_counts = metrics.loc[(metrics.domain == domain) & (metrics.condition == "fixed_h1")
                                   & (metrics.task_id == "T1") & (metrics.metric == "auroc")
                                   & (metrics.method_id == "__all__")].set_index("seed").loc[list(SEEDS), "count"]
        if not np.array_equal(fixed_counts, cand.groupby("seed").size().reindex(SEEDS)):
            raise ValueError("Fixed-H1 metric counts differ from video rows")
        pending.append(("G1", domain, base, cand, "auroc", .05, -.01, fixed_scores))
        task_drops = []
        for task, metric in PRIMARY.items():
            seen, heldout = semantic_values(domain, "seen", task, base), semantic_values(domain, "heldout", task, base)
            if ((seen < 0) | (seen > 1) | (heldout < 0) | (heldout > 1)).any():
                raise ValueError("primary scores must lie in [0, 1]")
            drop = seen - heldout
            task_drops.append(drop)
            check("G2", f"{domain}/{task}", drop, .05, "max")
            if task == "T2":
                check("G4", f"{domain}/T2/heldout", heldout, .80)
            if task in ("T4", "T5"):
                check("G5", f"{domain}/{task}/heldout", heldout, .65 if task == "T4" else .60)
        check("G2", f"{domain}/macro", np.mean(task_drops, axis=0), .02, "max")
        permutation_gate(domain)
        for task, metric, gate, threshold in (("T2", "polarity_consistency", "G4", .95), ("T4", "true_family_ordering", "G5", .80)):
            rows = controls.loc[(controls.domain == domain) & (controls.condition == "counterfactual") & (controls.task_id == task)]
            _require(rows, ["metric", "value", "count"], f"{domain}/{task} counterfactual")
            if not rows.metric.eq(metric).all() or rows.duplicated(["seed", "video_key"]).any() or rows.pairing_key.duplicated().any():
                raise ValueError("duplicate or invalid counterfactual pairs")
            values = _finite(rows.value, "counterfactual values")
            if not np.isin(values, [0, 1]).all() or not rows["count"].eq(1).all():
                raise ValueError("counterfactual evidence must be video-level binary outcomes")
            if task == "T4":
                _require(rows, ["meaningful_score", "control_score", "drop"], "T4 counterfactual scores")
                scores = _probabilities(rows, ["meaningful_score", "control_score"])
                expected_drop = scores[:, 0] - scores[:, 1]
                if not np.allclose(_finite(rows["drop"], "T4 counterfactual drop"),
                                   expected_drop, rtol=0, atol=1e-10):
                    raise ValueError("T4 counterfactual drop differs from paired scores")
                ordering = (expected_drop > 0).astype(float)
                if not np.array_equal(values, ordering):
                    raise ValueError("T4 true-family ordering differs from paired scores")
                rows = rows.assign(value=ordering)
            per_seed = []
            for experiment_seed in SEEDS:
                expected = base.loc[(base.seed == experiment_seed) & ((base.label == 1) if task == "T4" else True)]
                selected = rows.loc[rows.seed == experiment_seed]
                if set(selected.video_key) != set(expected.video_key):
                    raise ValueError("counterfactual video-key sets differ or missing seeds")
                per_seed.append(float(selected.value.mean()))
            check(gate, f"{domain}/{metric}", per_seed, threshold)
            evidence[gate][f"{domain}/{metric}"]["counts_per_seed"] = {
                str(s): int((rows.seed == s).sum()) for s in SEEDS}
        for condition in ("meaningless_candidates", "meaningless_questions"):
            for task, metric in PRIMARY.items():
                if condition == "meaningless_questions" and task not in ("T2", "T4"):
                    continue
                rows = controls.loc[(controls.domain == domain) & (controls.condition == condition) & (controls.task_id == task)].copy()
                _require(rows, list(GROUP_KEYS) + ["num_frames", "meaningful_num_frames"], f"{domain}/{condition}/{task}")
                if rows[list(GROUP_KEYS)].isna().any().any():
                    raise ValueError("missing paired video/variant identity")
                if rows.pairing_key.isna().any() or rows.pairing_key.eq("").any() or rows.pairing_key.duplicated().any() or rows.duplicated(list(GROUP_KEYS)).any():
                    raise ValueError("duplicate or missing paired video keys")
                if not rows.num_frames.eq(rows.meaningful_num_frames).all():
                    raise ValueError("paired frame counts differ")
                if keys != _population(rows, "control"):
                    raise ValueError("control video-key sets differ")
                base_labels = base.drop_duplicates("video_key").set_index("video_key").loc[keys, "label"]
                control_labels = rows.drop_duplicates("video_key").set_index("video_key").loc[keys, "label"]
                if not np.array_equal(base_labels, control_labels):
                    raise ValueError("control physical video labels differ from baseline")
                columns = ("p_camera", "p_swap", "p_reenact") if task == "T5" else (("p_yes",) if task == "T4" else ("prob_fake",))
                _probabilities(rows, columns)
                _probabilities(rows, ["meaningful_" + c for c in columns])
                meaningful = rows.copy()
                for column in columns:
                    meaningful[column] = rows["meaningful_" + column]
                normalized = metric_values(domain, condition, task, metric + "_drop")
                stored = metrics.loc[(metrics.domain == domain) & (metrics.condition == condition)
                                     & (metrics.task_id == task) & (metrics.metric == metric + "_drop")
                                     & (metrics.method_id == "__all__")].set_index("seed").loc[list(SEEDS)]
                actual_counts = rows.groupby("seed").size().reindex(SEEDS).to_numpy()
                if not np.array_equal(stored["count"].to_numpy(), actual_counts):
                    raise ValueError("paired control row counts differ from normalized metric counts")
                pending.append(("G6", f"{domain}/{condition}/{task}", rows, meaningful, task, .025, .10, normalized))
    for domain in sorted(set(permutation.domain) - set(DOMAINS)):
        if domain != "dfd":
            raise ValueError(f"unknown audited domain: {domain}")
        permutation_gate(domain)
    # Validate every artifact's schema before any potentially costly resampling.
    for gate, key, left, right, task, quantile, threshold, normalized in pending:
        # A one-draw preparation validates exact identities, label alignment and
        # metric classes before the final 10,000-draw jobs start.
        prepared = paired_seed_video_bootstrap(left, right, task, 1, rng_seed, lower_quantile=quantile)
        recomputed = prepared.candidate_per_seed if gate == "G1" else prepared.per_seed
        if not np.allclose(recomputed, normalized, rtol=0, atol=1e-10):
            raise ValueError(f"{key}: normalized primary metric differs from paired probabilities")
        if gate == "G6":
            domain, condition, task_id = key.split("/")
            stored = metrics.loc[(metrics.domain == domain) & (metrics.condition == condition)
                                 & (metrics.task_id == task_id) & (metrics.metric == PRIMARY[task_id] + "_drop")
                                 & (metrics.method_id == "__all__")].set_index("seed").loc[list(SEEDS)]
            _require(stored, ["drop", "meaningful_score", "control_score"], key)
            for column, expected in (("drop", prepared.per_seed), ("meaningful_score", prepared.candidate_per_seed), ("control_score", prepared.baseline_per_seed)):
                if not np.allclose(_finite(stored[column], key), expected, rtol=0, atol=1e-10):
                    raise ValueError(f"{key}: stored {column} differs from paired probabilities")
    for gate, key, left, right, task, quantile, threshold, normalized in pending:
        result = paired_seed_video_bootstrap(left, right, task, n_resamples, rng_seed, lower_quantile=quantile)
        record = asdict(result)
        record.update(baseline_mean=float(np.mean(result.baseline_per_seed)),
                      candidate_mean=float(np.mean(result.candidate_per_seed)),
                      baseline_seed_sd=float(np.std(result.baseline_per_seed, ddof=1)),
                      candidate_seed_sd=float(np.std(result.candidate_per_seed, ddof=1)))
        record.update(threshold=threshold, ci_threshold=-.01 if gate == "G1" else 0., ci_comparison=">=" if gate == "G1" else ">")
        evidence[gate][key] = record
        points[gate].append(result.point >= threshold)
        cis[gate].append(result.lower >= -.01 if gate == "G1" else result.lower > 0)
        intervals.append(dict(gate=gate, evidence=key, **record))
        for experiment_seed, value in zip(SEEDS, result.per_seed):
            seed_metrics.append(dict(gate=gate, evidence=key, seed=experiment_seed, value=value,
                                     seed_sd=result.seed_sd))
    gates = {}
    for name in evidence:
        point = bool(points[name]) and all(points[name])
        ci = all(cis[name]) if name in cis else None
        status = "fail" if not point else ("inconclusive" if ci is False else "pass")
        gates[name] = GateResult(name, status, point, ci, evidence[name])
    verdict = "not_supported" if any(not g.point_passed for g in gates.values()) else (
        "insufficient_evidence" if any(g.ci_passed is False for g in gates.values()) else "supported")
    sources = sorted({str(value) for frame in (metrics, controls, permutation, baseline, candidate, semantic)
                      if "source_file" in frame for value in frame.source_file.dropna().unique()})
    return GateReport(verdict, gates, dict(THRESHOLDS), n_resamples, final,
                      tuple(seed_metrics), tuple(intervals), tuple(sources))
