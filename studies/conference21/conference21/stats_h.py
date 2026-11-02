"""Pure CPU, same-cohort H Raw statistics with shared source-group bootstrap.

No file/model loading, provenance promotion, target selection or GPU work.
Caller verifies frozen role/domain/geometry metadata. SOURCE_CAL must contain
the entire all-B-success real pool for each model; common/AB never calibrates.
Operating decisions use frozen FP64 score >= tau. AUC keeps rank_score objects
at their original precision, and unresolved ordering is explicitly unavailable.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import time

import mpmath as mp
import numpy as np
import pandas as pd

from . import scoring_d as scoring
from .bootstrap_fast import _DrawCounts, _GroupIndex, _RankCell, _draw_multiplicities, _options

FORMAL_SEEDS = (7, 42, 123, 2026, 3407)
IDENTITY = ("video_uid", "label", "method_id", "group_id")
METRICS = ("pooled_auc", "macro_auc", "fpr", "tpr", "pooled_ba", "macro_tpr", "macro_ba")
COUNTS = ("n_real", "n_fake", "false_positives", "true_positives")
THRESHOLD_KINDS = ("default", "source_real")
CELL_HISTORY = {"H00": "B", "H01": "B", "H10": "AB", "H11": "AB"}


def normalize_video_table(videos, *, real_only=False):
    """Validate unique natural videos, preserve rank objects and sort by UID.

    score must be machine FP64 (integer numeric inputs are normalized to FP64);
    arbitrary precision belongs exclusively in rank_score. No prior weights or
    draw IDs are accepted, so caller cannot silently double-bootstrap a table.
    """
    required = set(IDENTITY) | {"score"}
    if not isinstance(videos, pd.DataFrame) or not required <= set(videos):
        raise ValueError(f"required video columns: {sorted(required)}")
    q = videos.copy().reset_index(drop=True)
    if q.empty or q[list(required)].isna().any().any() or q.video_uid.duplicated().any():
        raise ValueError("nonempty unique videos with complete identity required")
    for column in ("video_uid", "method_id", "group_id"):
        if not q[column].map(lambda value: isinstance(value, str) and bool(value)).all():
            raise ValueError(f"nonempty string {column} required")
    if not set(q.label) <= {0, 1}:
        raise ValueError("labels must be real=0, fake=1")
    q["label"] = q.label.astype(np.int8)
    if real_only and not q.label.eq(0).all():
        raise ValueError("SOURCE_CAL accepts real videos only")
    if not real_only and set(q.label) != {0, 1}:
        raise ValueError("fixed TEST requires both real=0 and fake=1")
    if not q.loc[q.label.eq(0), "method_id"].eq("real").all() or q.loc[q.label.eq(1), "method_id"].eq("real").any():
        raise ValueError("method identity must agree with real/fake label")
    for field, expected in (("representation", "raw"), ("grouping", "correct")):
        if field in q and not q[field].eq(expected).all():
            raise ValueError("H accepts only Raw, correctly grouped natural video scores")
    if {"weight", "draw_id"} & set(q):
        raise ValueError("original unweighted natural videos required")
    if not pd.api.types.is_numeric_dtype(q.score.dtype) or pd.api.types.is_complex_dtype(q.score.dtype) or pd.api.types.is_bool_dtype(q.score.dtype) or any(isinstance(value, mp.mpf) for value in q.score) or (pd.api.types.is_float_dtype(q.score.dtype) and q.score.dtype != np.dtype("float64")):
        raise ValueError("operating score must be FP64; preserve high precision separately in rank_score")
    try:
        q["score"] = q.score.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("finite FP64 operating scores required") from error
    if not np.isfinite(q.score).all():
        raise ValueError("finite FP64 operating scores required")
    if "rank_score" not in q:
        q["rank_score"] = pd.Series(q.score.tolist(), dtype=object)
    if any(not mp.isfinite(value) for value in q.rank_score):
        raise ValueError("finite original-precision rank_score required")
    if "ranking_usable" in q:
        if not q.ranking_usable.map(lambda value: isinstance(value, (bool, np.bool_))).all():
            raise ValueError("ranking_usable must contain boolean flags")
    else:
        q["ranking_usable"] = True
    return q.sort_values("video_uid", kind="stable").reset_index(drop=True)


def validate_cohort(cells, *, fixed_methods):
    """Reject all-B/common cross-subtraction and identity/count-only pairing."""
    if not isinstance(cells, Mapping) or set(cells) not in (
        {"H00", "H10"}, {"H00", "H01", "H10", "H11"}
    ):
        raise ValueError("cells must be H00/H10 all-B, or all four H00/H01/H10/H11 common cells")
    methods = tuple(fixed_methods)
    if not methods or len(set(methods)) != len(methods) or any(not isinstance(method, str) or not method or method == "real" for method in methods):
        raise ValueError("unique nonempty registered fixed fake methods required")
    normalized = {name: normalize_video_table(cells[name]) for name in sorted(cells)}
    reference = normalized["H00"][list(IDENTITY)]
    for name, q in normalized.items():
        if not q[list(IDENTITY)].equals(reference):
            raise ValueError(f"{name} cohort video/group/label/method identity differs from H00")
        if not set(q.loc[q.label.eq(1), "method_id"]) <= set(methods):
            raise ValueError("fake methods outside the registered fixed methods")
    return normalized


def _source_thresholds(source_cal, test, aggregation):
    if not isinstance(source_cal, Mapping) or set(source_cal) != {"B", "AB"}:
        raise ValueError("source_cal requires B and AB models on all-B real SOURCE_CAL")
    cal = {history: normalize_video_table(source_cal[history], real_only=True) for history in ("B", "AB")}
    if not cal["B"][list(IDENTITY)].equals(cal["AB"][list(IDENTITY)]):
        raise ValueError("both model source calibration pools must share exact identities")
    for q in cal.values():
        if "aggregation" in q and not q.aggregation.eq(aggregation).all():
            raise ValueError("SOURCE_CAL aggregation does not match requested aggregation")
        if set(q.video_uid) & set(test.video_uid):
            raise ValueError("SOURCE_CAL and TEST video UID overlap")
    thresholds = {history: scoring.empirical_fpr_threshold(q.score, alpha=.05) for history, q in cal.items()}
    metadata = {}
    for history, q in cal.items():
        predicted = q.score.ge(thresholds[history])
        metadata[history] = dict(threshold=thresholds[history], n_real=len(q),
            false_positives=int(predicted.sum()), empirical_fpr=float(predicted.mean()),
            allowed_false_positives=int(math.floor(.05 * len(q) + 1e-12)),
            calibration_version="B", score_space="FP64", decision="score>=threshold",
            calibration_uid_sha256=hashlib.sha256("\n".join(q.video_uid).encode()).hexdigest())
    return thresholds, metadata


def independent_auc(labels, values, weights=None):
    """Direct integer twice-concordance, separate from rank bins/scoring code."""
    y, x = list(labels), list(values)
    if not y or len(y) != len(x) or not set(y) <= {0, 1} or any(not mp.isfinite(value) for value in x):
        raise ValueError("aligned finite binary ranks required")
    w = [1] * len(y) if weights is None else list(weights)
    if len(w) != len(y) or any(isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value or value < 0 for value in w):
        raise ValueError("nonnegative integer multiplicities required")
    w = [int(value) for value in w]
    real = [(value, weight) for label, value, weight in zip(y, x, w) if label == 0 and weight]
    fake = [(value, weight) for label, value, weight in zip(y, x, w) if label == 1 and weight]
    nr, nf = sum(weight for _, weight in real), sum(weight for _, weight in fake)
    if not nr or not nf:
        raise ValueError("independent AUC requires two classes")
    twice = sum(wf * wr * (2 if f > r else 1 if f == r else 0)
                for f, wf in fake for r, wr in real)
    return twice / (2 * nr * nf)


def _independent_metrics(q, threshold, methods):
    """Integer threshold counts and direct pair AUC; never invokes scorer."""
    labels, values = q.label.tolist(), q.score.tolist()
    nr, nf = labels.count(0), labels.count(1)
    fp = sum(label == 0 and value >= threshold for label, value in zip(labels, values))
    tp = sum(label == 1 and value >= threshold for label, value in zip(labels, values))
    missing = tuple(method for method in methods if not ((q.label == 1) & (q.method_id == method)).any())
    macro_operating = bool(methods and not missing)
    usable = bool(q.ranking_usable.all())
    mt = float(np.mean([sum(value >= threshold for value in q.loc[q.label.eq(1) & q.method_id.eq(method), "score"])
        / int((q.label.eq(1) & q.method_id.eq(method)).sum()) for method in methods])) if macro_operating else float("nan")
    macro_auc = float("nan")
    if usable and macro_operating:
        macro_auc = float(np.mean([independent_auc(q.loc[q.label.eq(0) | q.method_id.eq(method), "label"],
            q.loc[q.label.eq(0) | q.method_id.eq(method), "rank_score"]) for method in methods]))
    return dict(n_real=nr, n_fake=nf, false_positives=fp, true_positives=tp,
        fpr=fp/nr, tpr=tp/nf, pooled_ba=(1-fp/nr+tp/nf)/2,
        macro_tpr=mt, macro_ba=(1-fp/nr+mt)/2, macro_auc=macro_auc,
        pooled_auc=independent_auc(labels, q.rank_score) if usable else float("nan"))


class _PreparedCell:
    """One rank preparation, integer group counts and two frozen operating rules."""
    def __init__(self, videos, group_indices, methods, thresholds):
        self.rank = _RankCell(videos, group_indices, methods)
        self.methods = methods
        self.thresholds = thresholds
        self.predicted = {kind: videos.score.to_numpy(dtype=np.float64) >= threshold for kind, threshold in thresholds.items()}

    def evaluate(self, dense):
        rank = self.rank
        base = rank.evaluate(dense)
        w = dense[rank.group_indices]
        nr, nf = int(w[rank.real].sum()), int(w[rank.fake].sum())
        method_mass = np.bincount(rank.fake_method, weights=w[rank.fake], minlength=len(self.methods))
        missing = tuple(method for method, mass in zip(self.methods, method_mass) if mass <= 0)
        operating_valid = bool(self.methods and not missing)
        out = {}
        for kind, threshold in self.thresholds.items():
            predicted = self.predicted[kind]
            fp, tp = int(w[rank.real & predicted].sum()), int(w[rank.fake & predicted].sum())
            fpr, tpr = fp/nr, tp/nf
            mt = float("nan")
            if operating_valid:
                method_tp = np.bincount(rank.fake_method, weights=(w[rank.fake]*predicted[rank.fake]), minlength=len(self.methods))
                mt = float(np.mean(method_tp/method_mass))
            out[kind] = dict(base, threshold=threshold, threshold_kind=kind,
                n_real=nr, n_fake=nf, false_positives=fp, true_positives=tp,
                false_positive_mass=float(fp), fpr=fpr, tpr=tpr, pooled_ba=(1-fpr+tpr)/2,
                macro_tpr=mt, macro_ba=(1-fpr+mt)/2,
                macro_operating_valid=operating_valid, missing_methods=missing,
                operating_score_field="score")
        return out


def _comparison_terms(four):
    terms = {"H10-H00": (("H10", 1), ("H00", -1))}
    if four:
        terms.update({
            "H01-H00": (("H01", 1), ("H00", -1)),
            "H11-H10": (("H11", 1), ("H10", -1)),
            "(H11-H10)-(H01-H00)": (("H11", 1), ("H10", -1), ("H01", -1), ("H00", 1)),
            "H11-H01": (("H11", 1), ("H01", -1))})
    return terms


def _comparison_rows(values, terms):
    return [dict(comparison=name, threshold_kind=kind, metric=metric,
        estimate=float(sum(coefficient*values[cell][kind][metric] for cell, coefficient in expression)))
        for name, expression in terms.items() for kind in THRESHOLD_KINDS for metric in METRICS]


def _flatten(values, comparisons):
    result = {f"{kind}:cell:{cell}:{metric}": metrics[metric]
        for cell, kinds in values.items() for kind, metrics in kinds.items() for metric in (*METRICS, *COUNTS)}
    result.update({f"{row['threshold_kind']}:comparison:{row['comparison']}:{row['metric']}": row["estimate"]
        for row in comparisons})
    return result


def _intervals(draws, point):
    rows = []
    for key, estimate in point.items():
        kind, target_kind, target, metric = key.split(":", 3)
        observed = draws[key].dropna() if key in draws else pd.Series(dtype=float)
        rows.append(dict(threshold_kind=kind, target_kind=target_kind, target=target,
            metric=metric, estimate=estimate, lower=float(observed.quantile(.025)) if len(observed) else float("nan"),
            upper=float(observed.quantile(.975)) if len(observed) else float("nan"),
            valid_repeats=len(observed), interval_type="pointwise_conditional_percentile"))
    return pd.DataFrame(rows)


def analyze_cohort(cells, source_cal, *, fixed_methods, aggregation="mean_logit",
                   seed=20261023, repeats=2000, min_groups=20, max_group_fraction=.5):
    """Analyze one exact H TEST cohort; never combine all-B and common tables.

    Returns point_metrics and comparisons DataFrames, frozen source thresholds,
    independent_checks, and bootstrap (draws/intervals/invalid counts). Draw
    columns are '<threshold_kind>:<cell|comparison>:<target>:<metric>'.
    group_draw_counts is an exact sparse Mapping, exportable via dict(mapping).
    Both operating rules share every TEST-group draw; SOURCE_CAL stays fixed.
    Metrics/effects are in native units; no direction/significance filtering.
    """
    _options(repeats, min_groups, max_group_fraction)
    if aggregation not in ("mean_logit", "mean_probability"):
        raise ValueError("aggregation must be mean_logit or mean_probability")
    methods = tuple(fixed_methods)
    normalized = validate_cohort(cells, fixed_methods=methods)
    reference = normalized["H00"]
    if any("aggregation" in q and not q.aggregation.eq(aggregation).all() for q in normalized.values()):
        raise ValueError("TEST aggregation does not match requested aggregation")
    thresholds, threshold_metadata = _source_thresholds(source_cal, reference, aggregation)
    groups = _GroupIndex.make(reference.group_id.unique())
    group_indices = np.asarray([groups.lookup[group] for group in reference.group_id], dtype=np.int64)
    sizes = np.bincount(group_indices, minlength=len(groups.names))
    default = 0. if aggregation == "mean_logit" else .5
    started = time.perf_counter()
    prepared = {name: _PreparedCell(q, group_indices, methods,
        {"default": default, "source_real": thresholds[CELL_HISTORY[name]]}) for name, q in normalized.items()}
    ones = np.ones(len(groups.names), dtype=np.int64)
    point_values = {name: cell.evaluate(ones) for name, cell in prepared.items()}
    metrics = pd.DataFrame([dict(cell=name, model_history=CELL_HISTORY[name], **metric)
        for name, kinds in point_values.items() for metric in kinds.values()])
    terms = _comparison_terms(len(normalized) == 4)
    comparisons = _comparison_rows(point_values, terms)
    checks = []
    for name, q in normalized.items():
        for kind in THRESHOLD_KINDS:
            independent = _independent_metrics(q, point_values[name][kind]["threshold"], methods)
            errors = {}
            for metric, actual in independent.items():
                expected = point_values[name][kind][metric]
                if math.isnan(float(actual)) and math.isnan(float(expected)):
                    continue
                if not np.isfinite(actual) or not np.isfinite(expected) or abs(actual-expected) > 1e-10:
                    raise ArithmeticError(f"independent H metric mismatch: {name}/{kind}/{metric}: {actual} vs {expected}")
                errors[metric] = abs(actual-expected)
            checks.append(dict(cell=name, threshold_kind=kind,
                status="PASSED" if bool(q.ranking_usable.all()) else "OPERATING_PASSED_RANKING_UNRESOLVED",
                max_abs_error=max(errors.values(), default=0.), metric_abs_errors=errors))
    point_flat = _flatten(point_values, comparisons)
    bootstrap = dict(n_source_groups=len(groups.names), source_groups=groups.names,
        largest_group_fraction=float(sizes.max()/sizes.sum()), repeats=int(repeats), seed=seed,
        min_groups=int(min_groups), max_group_fraction=max_group_fraction,
        macro_invalid_repeats=0, pooled_invalid_repeats=0, ranking_invalid_repeats=0,
        inference_scope="conditional_on_fixed_models_B_source_cal_and_caller_known_or_proxy_TEST_groups",
        ancestry_certified=False, source_cal_resampled=False, training_seeds_resampled=False,
        point_estimates=point_flat, preparation_seconds=time.perf_counter()-started, bootstrap_seconds=0.)
    if len(groups.names) < min_groups or bootstrap["largest_group_fraction"] > max_group_fraction:
        bootstrap.update(status="insufficient_source_groups", draws=pd.DataFrame(), intervals=None,
            limitation="Too few or concentrated source groups; per-seed descriptions only.")
    else:
        rows = []
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        for repeat in range(int(repeats)):
            dense = _draw_multiplicities(groups, rng)
            record = dict(repeat=repeat, group_draw_counts=_DrawCounts(groups, dense))
            try:
                values = {name: cell.evaluate(dense) for name, cell in prepared.items()}
            except ValueError as error:
                if "two-class AUC" not in str(error):
                    raise
                bootstrap["pooled_invalid_repeats"] += 1
                bootstrap["macro_invalid_repeats"] += 1
                rows.append(dict(record, pooled_valid=False, macro_valid=False, macro_operating_valid=False, ranking_valid=False))
                continue
            macro_operating_valid = all(value["macro_operating_valid"] for kinds in values.values() for value in kinds.values())
            macro_valid = all(value["macro_valid"] for kinds in values.values() for value in kinds.values())
            ranking_valid = all(value["ranking_usable"] for kinds in values.values() for value in kinds.values())
            bootstrap["macro_invalid_repeats"] += int(not macro_valid)
            bootstrap["ranking_invalid_repeats"] += int(not ranking_valid)
            rows.append(dict(record, pooled_valid=True, macro_valid=macro_valid, macro_operating_valid=macro_operating_valid, ranking_valid=ranking_valid,
                **_flatten(values, _comparison_rows(values, terms))))
        draws = pd.DataFrame(rows)
        bootstrap.update(status="ok" if bool(draws.pooled_valid.any()) else "no_valid_draws",
            draws=draws, intervals=_intervals(draws, point_flat), bootstrap_seconds=time.perf_counter()-started,
            saved_multiplicity_array_bytes=sum(value.storage_bytes for value in draws.group_draw_counts))
    return dict(point_metrics=metrics, comparisons=pd.DataFrame(comparisons),
        thresholds=thresholds, threshold_metadata=threshold_metadata, bootstrap=bootstrap,
        independent_checks=checks, aggregation=aggregation,
        cohort_kind="common_2x2" if len(normalized) == 4 else "all_B_deployment",
        cohort_video_uid_sha256=hashlib.sha256("\n".join(reference.video_uid).encode()).hexdigest(),
        cohort_identity_sha256=hashlib.sha256(reference[list(IDENTITY)].to_json(orient="values").encode()).hexdigest(),
        n_videos=len(reference), fixed_methods=methods, representation="raw",
        limitation="Kernel verifies numeric identities only; driver must verify frozen role/domain/cohort/cache provenance.")


def summarize_seed_comparisons(rows, *, group_columns=None):
    """Exactly five registered training seeds per stratum, sample SD ddof=1.

    Missing numerical estimates remain unavailable for that five-model summary;
    no silent drop of a seed, no bootstrap-draw/model pooling, no 15-model IID.
    Known backbone/domain/cohort/aggregation strata are retained automatically.
    """
    required = {"seed", "comparison", "threshold_kind", "metric", "estimate"}
    if not isinstance(rows, pd.DataFrame) or rows.empty or not required <= set(rows):
        raise ValueError("nonempty per-seed comparison table required")
    candidates = ("backbone", "domain", "cohort", "cohort_kind", "aggregation", "comparison", "threshold_kind", "metric")
    columns = [column for column in candidates if column in rows] if group_columns is None else list(group_columns)
    if not {"comparison", "threshold_kind", "metric"} <= set(columns) or any(column not in rows for column in columns) or "seed" in columns:
        raise ValueError("comparison, threshold_kind and metric grouping columns required")
    # Never permit explicit grouping to collapse a known model/cohort stratum.
    if any(column in rows and column not in columns for column in candidates):
        raise ValueError("grouping must preserve backbone/domain/cohort/aggregation strata")
    out = []
    for key, part in rows.groupby(columns, dropna=False, sort=False):
        if part.seed.duplicated().any():
            raise ValueError("duplicate training seed within comparison stratum")
        if set(part.seed) != set(FORMAL_SEEDS) or len(part) != 5:
            raise ValueError("exact five registered training seeds (7,42,123,2026,3407) required")
        values = part.estimate.to_numpy(dtype=np.float64)
        finite = np.isfinite(values)
        complete = bool(finite.all())
        identities = key if isinstance(key, tuple) else (key,)
        out.append(dict(zip(columns, identities), n_seeds=5, expected_seeds=FORMAL_SEEDS,
            n_valid_seeds=int(finite.sum()), complete_five_seeds=True, all_estimates_available=complete,
            estimate_mean=float(np.mean(values)) if complete else float("nan"),
            estimate_sample_sd=float(np.std(values, ddof=1)) if complete else float("nan"),
            n_positive=int((values[finite] > 1e-10).sum()),
            n_negative=int((values[finite] < -1e-10).sum()),
            n_zero_within_tolerance=int((np.abs(values[finite]) <= 1e-10).sum()),
            status="ok" if complete else "unavailable_seed_estimate", model_count=5,
            inference_scope="descriptive_across_five_training_seeds"))
    return pd.DataFrame(out)
