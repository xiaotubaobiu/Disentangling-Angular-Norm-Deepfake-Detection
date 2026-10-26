"""Canonical semantic probabilities, video aggregation, and E2 metrics."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from data.semantic_language import SEMANTIC_TO_ID


SEMANTICS = tuple(
    name for name, _ in sorted(SEMANTIC_TO_ID.items(), key=lambda item: item[1])
)
if tuple(SEMANTIC_TO_ID.values()) != tuple(range(len(SEMANTIC_TO_ID))):
    raise RuntimeError("SEMANTIC_TO_ID must be contiguous in declaration order")

PROBABILITY_COLUMNS = tuple(f"p_{name}" for name in SEMANTICS)

# These identify a rendered semantic condition.  In particular,
# candidate_order keeps exhaustive permutations distinct while sample_id is
# deliberately absent so it can be used as the within-condition unique key.
SEMANTIC_CONDITION_KEYS = (
    "seed",
    "domain",
    "condition",
    "task_id",
    "variant_id",
    "question_id",
    "question_yes_means",
    "candidate_order",
)

GROUP_KEYS = (
    "seed",
    "domain",
    "condition",
    "task_id",
    "variant_id",
    "question_id",
    "question_yes_means",
    "question_pair_id",
    "method_id",
    "video_id",
    "video_key",
    "label",
    "method_family",
    "target_semantic",
)

METRIC_COLUMNS = (
    "seed",
    "domain",
    "condition",
    "task_id",
    "metric",
    "value",
    "count",
    "method_id",
    "variant_id",
    "question_yes_means",
)


def map_candidate_probabilities(
    probabilities: torch.Tensor,
    semantic_ids: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter candidate-position probabilities into canonical semantic order."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch, candidates]")
    if semantic_ids.shape != probabilities.shape or mask.shape != probabilities.shape:
        raise ValueError("probabilities, semantic_ids, and mask must have equal shapes")
    if mask.dtype is not torch.bool:
        raise ValueError("mask must be boolean")

    valid_ids = semantic_ids[mask]
    if ((valid_ids < 0) | (valid_ids >= len(SEMANTICS))).any():
        raise ValueError("invalid semantic ID in a valid candidate position")
    for row in range(probabilities.shape[0]):
        row_ids = semantic_ids[row][mask[row]].detach().cpu().tolist()
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("duplicate semantic IDs in one candidate set")

    mapped = probabilities.new_zeros((probabilities.shape[0], len(SEMANTICS)))
    for column in range(probabilities.shape[1]):
        valid = mask[:, column]
        if valid.any():
            mapped[valid, semantic_ids[valid, column]] = probabilities[valid, column]
    return mapped


def canonical_fake_probability(row: Mapping[str, object]) -> float:
    """Return a task-independent fake probability for one semantic result."""
    task_id = str(row["task_id"])
    if task_id in {"T1", "T3", "fixed_h1", "Fixed-H1"}:
        return float(row["p_fake"])
    if task_id == "T2":
        polarity = row.get("question_yes_means")
        if polarity == "real":
            return float(row["p_no"])
        if polarity == "fake":
            return float(row["p_yes"])
        raise ValueError("T2 question_yes_means must be 'real' or 'fake'")
    if task_id == "T4":
        if "p_yes_swap" in row and "p_yes_reenact" in row:
            return max(float(row["p_yes_swap"]), float(row["p_yes_reenact"]))
        raise ValueError("T4 fake probability requires paired swap and reenact scores")
    if task_id == "T5":
        return 1.0 - float(row["p_camera"])
    raise ValueError(f"unknown semantic task: {task_id}")


def validate_domain_task(domain: str, task_id: str) -> None:
    """Reject semantic family tasks that DFD cannot label reliably."""
    if domain.lower() == "dfd" and task_id not in {"T1", "T2", "T3"}:
        raise ValueError("DFD supports only T1-T3")


def _as_batch_list(value, size: int) -> list:
    if isinstance(value, torch.Tensor):
        values = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        values = [value] * size
    else:
        values = list(value)
    if len(values) != size:
        raise ValueError(f"batch field has {len(values)} items; expected {size}")
    return values


def _context(condition: str | Mapping[str, object]) -> dict[str, object]:
    if isinstance(condition, str):
        return {"condition": condition}
    result = dict(condition)
    if not result.get("condition"):
        raise ValueError("condition context must include condition")
    return result


def frame_semantic_rows(
    batch: Mapping[str, object],
    probabilities: torch.Tensor,
    condition: str | Mapping[str, object],
) -> list[dict]:
    """Convert one inference batch into canonical, serialization-safe rows."""
    semantic_ids = batch["candidate_semantic_ids"]
    candidate_mask = batch["candidate_mask"]
    if not isinstance(semantic_ids, torch.Tensor) or not isinstance(
        candidate_mask, torch.Tensor
    ):
        raise TypeError("candidate semantic IDs and mask must be tensors")
    mapped = map_candidate_probabilities(
        probabilities.detach(), semantic_ids.to(probabilities.device),
        candidate_mask.to(probabilities.device),
    ).cpu()
    size = probabilities.shape[0]
    context = _context(condition)

    def values(name: str, default):
        return _as_batch_list(batch.get(name, default), size)

    sample_ids = values("sample_id", "")
    image_paths = values("image_path", "")
    method_ids = values("method_id", "unknown")
    video_ids = values("video_id", "")
    video_keys = values(
        "video_key",
        [f"{method_id}::{video_id}" for method_id, video_id in zip(method_ids, video_ids)],
    )
    labels = values("label", 0)
    task_ids = values("task_id", "")
    variants = values("variant_id", "")
    question_ids = values("question_id", "")
    question_pair_ids = values("question_pair_id", "")
    polarities = values("question_yes_means_name", "")
    families = values("method_family", "")
    targets = values("target_semantic", "")
    render_ids = values("render_id", "")
    seeds = (
        _as_batch_list(context["seed"], size)
        if "seed" in context
        else values("seed", 0)
    )
    domains = (
        _as_batch_list(context["domain"], size)
        if "domain" in context
        else values("domain", "")
    )

    rows: list[dict] = []
    ids_cpu = semantic_ids.detach().cpu()
    mask_cpu = candidate_mask.detach().cpu()
    for index in range(size):
        task_id = str(task_ids[index])
        validate_domain_task(str(domains[index]), task_id)
        ordered_semantics = [
            SEMANTICS[int(semantic_id)]
            for semantic_id, valid in zip(ids_cpu[index], mask_cpu[index])
            if bool(valid)
        ]
        row = {
            "seed": int(seeds[index]),
            "domain": str(domains[index]),
            "condition": str(context["condition"]),
            "task_id": task_id,
            "variant_id": str(variants[index]),
            "question_id": str(question_ids[index]),
            "question_yes_means": str(polarities[index]),
            "question_pair_id": str(question_pair_ids[index]),
            "candidate_order": "|".join(ordered_semantics),
            "sample_id": str(sample_ids[index]),
            "image_path": str(image_paths[index]),
            "method_id": str(method_ids[index]),
            "video_id": str(video_ids[index]),
            "video_key": str(video_keys[index]),
            "label": int(labels[index]),
            "method_family": str(families[index]),
            "target_semantic": str(targets[index]),
            "render_id": str(render_ids[index]),
        }
        for semantic, column in SEMANTIC_TO_ID.items():
            row[f"p_{semantic}"] = float(mapped[index, column])
        if task_id == "T4":
            # An individual family question is not itself a fake probability;
            # Task 6 can pair swap/reenact scores using pairing_key.
            row["prob_fake"] = float("nan")
        else:
            row["prob_fake"] = canonical_fake_probability(row)
        row["pairing_key"] = "::".join(
            (
                str(row["seed"]),
                row["domain"],
                row["condition"],
                row["sample_id"],
                row["video_key"],
            )
        )
        rows.append(row)
    return rows


def semantic_condition_key(row: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(row.get(key, "") for key in SEMANTIC_CONDITION_KEYS)


def _deduplication_key(row: Mapping[str, object]) -> tuple[object, ...]:
    sample_id = row.get("sample_id")
    if not sample_id:
        raise ValueError("semantic row is missing sample_id")
    return (*semantic_condition_key(row), sample_id)


def _equal_values(left: object, right: object) -> bool:
    try:
        if math.isnan(float(left)) and math.isnan(float(right)):
            return True
    except (TypeError, ValueError):
        pass
    return left == right


def _equal_rows(left: Mapping[str, object], right: Mapping[str, object]) -> bool:
    return left.keys() == right.keys() and all(
        _equal_values(left[key], right[key]) for key in left
    )


def deduplicate_distributed_rows(rows: Iterable[dict]) -> list[dict]:
    """Remove identical DistributedSampler padding without hiding conflicts."""
    unique: OrderedDict[tuple[object, ...], dict] = OrderedDict()
    for row in rows:
        key = _deduplication_key(row)
        previous = unique.get(key)
        if previous is None:
            unique[key] = row
        elif not _equal_rows(previous, row):
            raise ValueError(f"conflicting distributed rows for semantic sample {key}")
    return list(unique.values())


def _with_aggregation_defaults(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    defaults = {
        "seed": 0,
        "domain": "",
        "condition": "",
        "variant_id": "",
        "question_id": "",
        "question_yes_means": "",
        "question_pair_id": "",
        "candidate_order": "",
        "method_family": "",
        "target_semantic": "",
        "image_path": "",
        "pairing_key": "",
        "render_id": "",
    }
    for column, default in defaults.items():
        if column not in frame:
            frame[column] = default
    for column in PROBABILITY_COLUMNS + ("prob_fake",):
        if column not in frame:
            frame[column] = np.nan
    if "video_key" not in frame:
        frame["video_key"] = (
            frame["method_id"].astype(str) + "::" + frame["video_id"].astype(str)
        )
    return frame


def _attach_paired_t4_fake_probabilities(frame: pd.DataFrame) -> None:
    """Validate and attach one exact swap/reenact pair per T4 frame."""
    selected = frame["task_id"].astype(str) == "T4"
    if not selected.any():
        return
    pair_keys = [
        "seed",
        "domain",
        "condition",
        "task_id",
        "sample_id",
        "method_id",
        "video_id",
        "video_key",
        "question_pair_id",
    ]
    for _, pair in frame.loc[selected].groupby(pair_keys, sort=False, dropna=False):
        if len(pair) == 1:
            raise ValueError("incomplete T4 question pair")
        polarities = pair["question_yes_means"].astype(str).tolist()
        if (
            len(pair) != 2
            or sorted(polarities) != ["reenact", "swap"]
            or pair["question_id"].astype(str).nunique() != 2
        ):
            raise ValueError("duplicate or ambiguous T4 question pair")
        swap_yes = float(
            pair.loc[pair["question_yes_means"] == "swap", "p_yes"].iloc[0]
        )
        reenact_yes = float(
            pair.loc[pair["question_yes_means"] == "reenact", "p_yes"].iloc[0]
        )
        probability = max(swap_yes, reenact_yes)
        frame.loc[pair.index, "prob_fake"] = probability


def aggregate_semantic_videos(rows: Iterable[dict] | pd.DataFrame) -> pd.DataFrame:
    """Average canonical frame probabilities by semantic condition and video."""
    records = rows.to_dict("records") if isinstance(rows, pd.DataFrame) else list(rows)
    output_columns = list(GROUP_KEYS) + list(PROBABILITY_COLUMNS) + [
        "prob_fake",
        "num_frames",
        "pairing_key",
    ]
    if not records:
        return pd.DataFrame(columns=output_columns)

    seen: set[tuple[object, ...]] = set()
    for row in records:
        key = _deduplication_key(row)
        if key in seen:
            raise ValueError(f"duplicate sample_id inside semantic condition: {key}")
        seen.add(key)

    frame = _with_aggregation_defaults(records)
    _attach_paired_t4_fake_probabilities(frame)
    for domain, task_id in frame[["domain", "task_id"]].drop_duplicates().itertuples(
        index=False, name=None
    ):
        validate_domain_task(str(domain), str(task_id))

    expected_keys = frame["method_id"].astype(str) + "::" + frame["video_id"].astype(str)
    if not (frame["video_key"].astype(str) == expected_keys).all():
        raise ValueError("video_key must equal method_id::video_id")
    video_identity = [
        "seed",
        "domain",
        "condition",
        "method_id",
        "video_id",
        "video_key",
    ]
    label_counts = frame.groupby(
        video_identity, sort=False, dropna=False
    )["label"].nunique(dropna=False)
    if (label_counts > 1).any():
        raise ValueError("conflicting labels for one semantic video")

    value_columns = list(PROBABILITY_COLUMNS) + ["prob_fake"]
    aggregations = {column: (column, "mean") for column in value_columns}
    aggregations.update(
        num_frames=("sample_id", "count"),
        pairing_key=("pairing_key", "first"),
    )
    return (
        frame.groupby(list(GROUP_KEYS), sort=False, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 15) -> float:
    """H1-compatible equal-width ECE over the fake-class probability."""
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            selected = (scores >= edges[index]) & (scores <= edges[index + 1])
        else:
            selected = (scores >= edges[index]) & (scores < edges[index + 1])
        if selected.any():
            error += selected.mean() * abs(
                float(labels[selected].mean()) - float(scores[selected].mean())
            )
    return float(error)


def _binary_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    predictions = (scores >= 0.5).astype(int)
    return {
        "auroc": (
            float(roc_auc_score(labels, scores))
            if len(np.unique(labels)) == 2
            else float("nan")
        ),
        "ap": (
            float(average_precision_score(labels, scores))
            if np.any(labels == 1)
            else float("nan")
        ),
        "brier": float(brier_score_loss(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "ece": _ece(labels, scores),
    }


def _classification_metrics(
    labels: np.ndarray, predictions: np.ndarray, *, macro_name: bool = False
) -> dict[str, float]:
    prefix = "macro_" if macro_name else ""
    f1_average = "macro" if macro_name else "binary"
    return {
        f"{prefix}balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        f"{prefix}f1": float(
            f1_score(labels, predictions, average=f1_average, zero_division=0)
        ),
    }


def _task_values(task_id: str, rows: pd.DataFrame) -> dict[str, float]:
    if task_id in {"T1", "T3", "fixed_h1", "Fixed-H1"}:
        labels = rows["label"].to_numpy(dtype=int)
        scores = rows["prob_fake"].to_numpy(dtype=float)
        return _binary_metrics(labels, scores)
    if task_id == "T2":
        labels = rows["label"].to_numpy(dtype=int)
        scores = rows["prob_fake"].to_numpy(dtype=float)
        return _classification_metrics(labels, (scores >= 0.5).astype(int))
    if task_id == "T4":
        labels = (rows["target_semantic"].astype(str) == "yes").to_numpy(dtype=int)
        predictions = (rows["p_yes"].to_numpy(dtype=float) >= 0.5).astype(int)
        return _classification_metrics(labels, predictions)
    if task_id == "T5":
        classes = ("camera", "swap", "reenact")
        truth = rows["method_family"].astype(str).to_numpy()
        probabilities = rows[[f"p_{name}" for name in classes]].to_numpy(dtype=float)
        predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
        return _classification_metrics(truth, predicted, macro_name=True)
    raise ValueError(f"unknown semantic task: {task_id}")


def compute_task_metrics(video_rows: pd.DataFrame | Iterable[dict]) -> pd.DataFrame:
    """Compute normalized overall and per-method video metric rows."""
    frame = (
        video_rows.copy()
        if isinstance(video_rows, pd.DataFrame)
        else pd.DataFrame(list(video_rows))
    )
    if frame.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    for column, default in (("seed", 0), ("domain", ""), ("condition", "")):
        if column not in frame:
            frame[column] = default

    results: list[dict] = []
    context_keys = ["seed", "domain", "condition", "task_id"]
    for context, task_rows in frame.groupby(context_keys, sort=False, dropna=False):
        seed, domain, condition, task_id = context
        validate_domain_task(str(domain), str(task_id))
        slices = [("__all__", task_rows)]
        slices.extend(
            (str(method_id), method_rows)
            for method_id, method_rows in task_rows.groupby("method_id", sort=False)
        )
        for method_id, selected in slices:
            count = int(len(selected))
            base = {
                "seed": int(seed),
                "domain": str(domain),
                "condition": str(condition),
                "task_id": str(task_id),
                "count": count,
                "method_id": method_id,
                "variant_id": "__all__",
                "question_yes_means": "__all__",
            }
            results.append({**base, "metric": "count", "value": float(count)})
            for metric, value in _task_values(str(task_id), selected).items():
                results.append({**base, "metric": metric, "value": value})
    return pd.DataFrame(results, columns=METRIC_COLUMNS)
