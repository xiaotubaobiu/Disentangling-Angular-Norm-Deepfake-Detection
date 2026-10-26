"""Exhaustive and paired controls for E2 semantic evaluation."""

from __future__ import annotations

import hashlib
import itertools
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch
import torch.distributed as dist
import pandas as pd

from data.semantic_dataset import SemanticTemplateSampler, method_family
from data.semantic_language import LanguageBank, SEMANTIC_TO_ID, load_language_bank
from evaluation.semantic_eval import compute_task_metrics


class ControlRequest(dict):
    """A collatable mapping with convenient attribute access in audits/tests."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def _stable_u64(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def _stable_permutation(size: int, *parts: object) -> tuple[int, ...]:
    order = list(range(size))
    random.Random(_stable_u64(*parts)).shuffle(order)
    return tuple(order)


def _default_language_bank() -> LanguageBank:
    manifest = (
        Path(__file__).resolve().parents[2]
        / "experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml"
    )
    return load_language_bank(manifest)


def _resolve_bank(sample: Mapping[str, object], bank: LanguageBank | None) -> LanguageBank:
    if bank is not None:
        return bank
    embedded = sample.get("language_bank")
    if isinstance(embedded, LanguageBank):
        return embedded
    return _default_language_bank()


def enumerate_candidate_permutations(candidate_count: int) -> tuple[tuple[int, ...], ...]:
    """Return every valid order for the locked binary/ternary candidate sets."""
    if candidate_count not in (2, 3):
        raise ValueError("candidate_count must be two or three")
    return tuple(itertools.permutations(range(candidate_count)))


def _distributed_audit_totals(
    maximum_error: torch.Tensor,
    consistent_count: torch.Tensor,
    comparison_count: torch.Tensor,
) -> tuple[float, int, int]:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(maximum_error, op=dist.ReduceOp.MAX)
        dist.all_reduce(consistent_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(comparison_count, op=dist.ReduceOp.SUM)
    return (
        float(maximum_error.item()),
        int(consistent_count.item()),
        int(comparison_count.item()),
    )


@torch.no_grad()
def permutation_audit(
    model,
    image_features: torch.Tensor,
    question_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    reduce_across_ranks: bool = True,
) -> dict[str, float | int]:
    """Audit all 2!/3! orders after remapping to the original candidate order.

    ``image_features`` is intentionally supplied rather than raw images so the
    caller extracts the frozen visual representation exactly once per batch.
    Global max/count reductions happen here when torch.distributed is active.
    """
    if candidate_embeddings.ndim != 3 or candidate_mask.ndim != 2:
        raise ValueError("candidate embeddings/mask must be rank three/two")
    if candidate_embeddings.shape[:2] != candidate_mask.shape:
        raise ValueError("candidate embeddings and mask shapes do not align")
    batch_size = candidate_embeddings.shape[0]
    if image_features.shape[0] != batch_size or question_embeddings.shape[0] != batch_size:
        raise ValueError("all audit tensors must have the same batch size")
    candidate_counts = candidate_mask.sum(dim=1)
    if not torch.all((candidate_counts == 2) | (candidate_counts == 3)):
        raise ValueError("every audit sample must contain two or three candidates")
    if not torch.all(candidate_mask == (
        torch.arange(candidate_mask.shape[1], device=candidate_mask.device)[None, :]
        < candidate_counts[:, None]
    )):
        raise ValueError("valid candidates must precede padding")

    was_training = bool(getattr(model, "training", False))
    model.eval()
    try:
        reference_logits = model.classify_features(
            image_features,
            question_embeddings,
            candidate_embeddings,
            candidate_mask,
        )
        reference_probabilities = torch.softmax(reference_logits.float(), dim=-1)
        maximum_error = torch.zeros((), dtype=torch.float64, device=image_features.device)
        consistent_count = torch.zeros((), dtype=torch.long, device=image_features.device)
        comparison_count = torch.zeros((), dtype=torch.long, device=image_features.device)

        for candidate_count in (2, 3):
            selected = torch.where(candidate_counts == candidate_count)[0]
            if selected.numel() == 0:
                continue
            subset_candidates = candidate_embeddings[selected]
            subset_mask = candidate_mask[selected]
            subset_reference = reference_probabilities[selected]
            for permutation in enumerate_candidate_permutations(candidate_count):
                full_order = tuple(permutation) + tuple(
                    range(candidate_count, candidate_embeddings.shape[1])
                )
                order = torch.tensor(
                    full_order, dtype=torch.long, device=candidate_embeddings.device
                )
                permuted_logits = model.classify_features(
                    image_features[selected],
                    question_embeddings[selected],
                    subset_candidates.index_select(1, order),
                    subset_mask.index_select(1, order),
                )
                permuted_probabilities = torch.softmax(
                    permuted_logits.float(), dim=-1
                )
                remapped = torch.zeros_like(permuted_probabilities)
                remapped[:, order] = permuted_probabilities
                error = (remapped - subset_reference).abs().max().to(torch.float64)
                maximum_error = torch.maximum(maximum_error, error)
                consistent = (
                    remapped.argmax(dim=-1) == subset_reference.argmax(dim=-1)
                )
                consistent_count += consistent.sum()
                comparison_count += consistent.numel()
    finally:
        model.train(was_training)

    if reduce_across_ranks:
        error, consistent, comparisons = _distributed_audit_totals(
            maximum_error, consistent_count, comparison_count
        )
    else:
        error, consistent, comparisons = maximum_error.item(), consistent_count.item(), comparison_count.item()
    return {
        "max_probability_error": error,
        "prediction_consistency": consistent / comparisons if comparisons else float("nan"),
        "consistent_count": consistent,
        "comparison_count": comparisons,
    }


def select_global_audit_indices(
    sample_ids: Sequence[str], limit: int = 4096
) -> tuple[int, ...]:
    """Select one global first-N subset using canonical sample-id ordering."""
    if limit <= 0:
        raise ValueError("audit limit must be positive")
    normalized = [str(sample_id) for sample_id in sample_ids]
    if len(normalized) != len(set(normalized)):
        raise ValueError("permutation audit sample IDs must be unique")
    return tuple(sorted(range(len(normalized)), key=normalized.__getitem__)[:limit])


def _pair_id(task_id: str, split: str, sample_id: str, candidate_set: str) -> str:
    digest = format(
        _stable_u64("counterfactual", task_id, split, sample_id, candidate_set),
        "016x",
    )
    return f"{task_id}::{split}::{sample_id}::{digest}"


def build_counterfactual_requests(
    task_id: str,
    split: str,
    sample: Mapping[str, object],
    bank: LanguageBank | None = None,
    *,
    seed: int = 42,
) -> tuple[ControlRequest, ControlRequest]:
    """Render one exact opposite-polarity/family pair for a physical sample."""
    if task_id not in {"T2", "T4"}:
        raise ValueError("counterfactual controls support only T2 or T4")
    bank = _resolve_bank(sample, bank)
    language = bank.task(task_id, split)
    expected_polarities = ("real", "fake") if task_id == "T2" else ("swap", "reenact")
    questions = {}
    for polarity in expected_polarities:
        candidates = tuple(
            question for question in language.questions if question.yes_means == polarity
        )
        if not candidates:
            raise ValueError(f"{task_id} {split} has no {polarity} question")
        questions[polarity] = candidates[
            _stable_u64("counterfactual_question", sample["sample_id"], polarity)
            % len(candidates)
        ]
    candidate_set = language.candidate_sets[
        _stable_u64("counterfactual_candidates", sample["sample_id"], task_id)
        % len(language.candidate_sets)
    ]
    permutation = _stable_permutation(
        len(candidate_set.options), "counterfactual_order", sample["sample_id"], task_id
    )
    pair_id = _pair_id(task_id, split, str(sample["sample_id"]), candidate_set.name)
    label = int(sample["label"])
    family = method_family(str(sample["method_id"]), label)
    sampler = SemanticTemplateSampler(bank, seed=seed)
    requests = []
    for polarity in expected_polarities:
        rendered = sampler.render(
            dict(sample),
            task_id,
            split,
            0,
            permutation=permutation,
            force_question=questions[polarity].name,
            force_candidate_set=candidate_set.name,
        )
        rendered.update(
            question_pair_id=pair_id,
            render_id=pair_id,
            control_pair_id=pair_id,
            hidden_visual_truth="real" if label == 0 else "fake",
            hidden_method_family=family,
            hidden_question_polarity=polarity,
        )
        requests.append(ControlRequest(rendered))
    return tuple(requests)  # type: ignore[return-value]


def _control_candidate_entries(bank: LanguageBank, candidate_count: int):
    key = "binary" if candidate_count == 2 else "ternary"
    records = bank.raw["controls"]["candidate_sets"][key]
    by_id = {entry.name: entry for entry in bank.entries("control", "candidate")}
    return tuple(by_id[record["id"]] for record in records)


def _valid_candidate_semantics(request: Mapping[str, object]) -> list[str]:
    mask = request["candidate_mask"]
    semantics = request["candidate_semantics"]
    mask_values = mask.detach().cpu().tolist() if isinstance(mask, torch.Tensor) else list(mask)
    return [
        str(semantic)
        for semantic, valid in zip(semantics, mask_values)
        if bool(valid)
    ]


def _control_pair_id(control: str, request: Mapping[str, object], sample_id: str) -> str:
    identity = (
        control,
        sample_id,
        request.get("task_id", ""),
        request.get("variant_id", ""),
        request.get("question_id", ""),
        request.get("question_yes_means_name", ""),
    )
    return f"{control}::{format(_stable_u64(*identity), '016x')}"


def build_meaningless_candidate_request(
    meaningful_request: Mapping[str, object],
    sample_id: str,
    bank: LanguageBank | None = None,
) -> ControlRequest:
    """Replace candidates with manifest controls and hidden balanced meanings."""
    bank = _resolve_bank(meaningful_request, bank)
    request = ControlRequest(dict(meaningful_request))
    original_semantics = _valid_candidate_semantics(meaningful_request)
    candidate_count = len(original_semantics)
    if candidate_count not in (2, 3):
        raise ValueError("meaningless controls require two or three candidates")
    canonical_semantics = tuple(
        sorted(set(original_semantics), key=SEMANTIC_TO_ID.__getitem__)
    )
    if len(canonical_semantics) != candidate_count:
        raise ValueError("meaningful candidate semantics must be unique")
    hidden_order = _stable_permutation(
        candidate_count, "meaningless_candidates:hidden", sample_id
    )
    hidden_semantics = tuple(canonical_semantics[index] for index in hidden_order)
    controls = _control_candidate_entries(bank, candidate_count)
    word_order = _stable_permutation(
        candidate_count, "meaningless_candidates:surface", sample_id
    )
    ordered_controls = tuple(controls[index] for index in word_order)
    width = len(meaningful_request["candidate_mask"])
    text_ids = torch.full((width,), -1, dtype=torch.long)
    semantic_ids = torch.full((width,), -1, dtype=torch.long)
    mask = torch.zeros(width, dtype=torch.bool)
    for index, (entry, semantic) in enumerate(zip(ordered_controls, hidden_semantics)):
        text_ids[index] = entry.text_id
        semantic_ids[index] = SEMANTIC_TO_ID[semantic]
        mask[index] = True
    target_semantic = str(meaningful_request["target_semantic"])
    target_index = hidden_semantics.index(target_semantic)
    padding = ("",) * (width - candidate_count)
    pair_id = _control_pair_id("meaningless_candidates", meaningful_request, sample_id)
    request.update(
        candidate_text_ids=text_ids,
        candidate_semantic_ids=semantic_ids,
        candidate_mask=mask,
        candidate_ids=tuple(entry.name for entry in ordered_controls) + padding,
        candidate_texts=tuple(entry.text for entry in ordered_controls) + padding,
        candidate_semantics=hidden_semantics + padding,
        target_index=torch.tensor(target_index, dtype=torch.long),
        variant_id=f"{meaningful_request.get('variant_id', '')}::meaningless_candidates",
        render_id=f"{pair_id}::control",
        control_pair_id=pair_id,
        hidden_target_semantic=target_semantic,
        hidden_candidate_semantics=hidden_semantics + padding,
    )
    return request


def build_meaningless_question_request(
    meaningful_request: Mapping[str, object],
    sample_id: str,
    bank: LanguageBank | None = None,
) -> ControlRequest:
    """Replace a T2/T4 question while retaining evaluation-only polarity."""
    if str(meaningful_request.get("task_id")) not in {"T2", "T4"}:
        raise ValueError("meaningless question controls support only T2 or T4")
    bank = _resolve_bank(meaningful_request, bank)
    questions = bank.entries("control", "question")
    if not questions:
        raise ValueError("language manifest contains no control questions")
    question = questions[
        _stable_u64("meaningless_questions", sample_id) % len(questions)
    ]
    pair_id = _control_pair_id("meaningless_questions", meaningful_request, sample_id)
    request = ControlRequest(dict(meaningful_request))
    request.update(
        question_text_id=torch.tensor(question.text_id, dtype=torch.long),
        question_id=question.name,
        question_text=question.text,
        variant_id=f"{meaningful_request.get('variant_id', '')}::meaningless_questions",
        render_id=f"{pair_id}::control",
        control_pair_id=pair_id,
        hidden_question_polarity=meaningful_request.get("question_yes_means_name", ""),
    )
    return request


def paired_control_rows(
    metadata: Sequence[Mapping[str, object]],
    meaningful_probabilities: torch.Tensor,
    control_probabilities: torch.Tensor,
) -> list[dict]:
    """Create exact per-sample meaningful/control score-drop records."""
    if meaningful_probabilities.shape != control_probabilities.shape:
        raise ValueError("meaningful and control probability shapes differ")
    if meaningful_probabilities.ndim != 2 or meaningful_probabilities.shape[1] != len(SEMANTIC_TO_ID):
        raise ValueError("canonical probabilities must have shape [batch, semantics]")
    if len(metadata) != meaningful_probabilities.shape[0]:
        raise ValueError("control metadata and probability batch sizes differ")
    pair_ids = [str(row.get("control_pair_id", "")) for row in metadata]
    if any(not pair_id for pair_id in pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("each row must have a unique control pair ID")

    meaningful_cpu = meaningful_probabilities.detach().cpu()
    control_cpu = control_probabilities.detach().cpu()
    rows = []
    for index, source in enumerate(metadata):
        target_semantic = str(source["target_semantic"])
        try:
            target_column = SEMANTIC_TO_ID[target_semantic]
        except KeyError as error:
            raise ValueError(f"invalid hidden target semantic: {target_semantic}") from error
        meaningful_score = float(meaningful_cpu[index, target_column])
        control_score = float(control_cpu[index, target_column])
        drop = meaningful_score - control_score
        row = dict(source)
        row.update(
            pairing_key=pair_ids[index],
            metric="target_probability_drop_diagnostic",
            value=drop,
            count=1,
            meaningful_score=meaningful_score,
            control_score=control_score,
            drop=drop,
        )
        rows.append(row)
    return rows


def control_metric_rows(rows: Iterable[Mapping[str, object]]) -> list[dict]:
    """G6 primary metric drops over exact paired *video* predictions.

    Meaningful probabilities are prefixed in the same records, so both scores
    use precisely the same videos/questions. Bootstrap inference is Task 7.
    """
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return []
    if frame.pairing_key.eq("").any() or frame.pairing_key.duplicated().any():
        raise ValueError("paired videos require unique nonempty pairing keys")
    meaningful = frame.copy()
    required = {"T1": ("prob_fake",), "T2": ("prob_fake",),
                "T3": ("prob_fake",), "T4": ("p_yes",),
                "T5": ("p_camera", "p_swap", "p_reenact")}
    for task in frame.task_id.unique():
        for column in required[task]:
            if f"meaningful_{column}" not in frame:
                raise ValueError(f"missing paired meaningful probability: {column}")
    for column in frame:
        if column.startswith("meaningful_"):
            meaningful[column.removeprefix("meaningful_")] = frame[column]
    primary = {"T1": "auroc", "T2": "balanced_accuracy", "T3": "auroc",
               "T4": "balanced_accuracy", "T5": "macro_balanced_accuracy"}
    keys = ["seed", "domain", "condition", "task_id", "metric", "method_id"]
    left = compute_task_metrics(meaningful)
    right = compute_task_metrics(frame)
    joined = left.merge(right, on=keys, validate="one_to_one", suffixes=("_meaningful", "_control"))
    output = []
    for row in joined.to_dict("records"):
        if row["method_id"] != "__all__" or row["metric"] != primary[row["task_id"]]:
            continue
        score, control = float(row["value_meaningful"]), float(row["value_control"])
        output.append({
            **{key: row[key] for key in keys},
            "metric": f"{row['metric']}_drop", "primary_metric": row["metric"],
            "value": score - control, "drop": score - control,
            "meaningful_score": score, "control_score": control,
            "count": int(row["count_control"]),
            "pairing_key": f"{row['seed']}::{row['domain']}::{row['condition']}::{row['task_id']}",
        })
    return output


def permutation_metric_rows(
    result: Mapping[str, float | int], *, seed: int, domain: str
) -> list[dict]:
    """Normalize a global permutation result for Task 7 consumption."""
    count = int(result["comparison_count"])
    base = {
        "seed": int(seed),
        "domain": str(domain),
        "condition": "permutation",
        "task_id": "__all__",
        "count": count,
        "pairing_key": f"{seed}::{domain}::permutation",
    }
    return [
        {
            **base,
            "metric": "max_probability_error",
            "value": float(result["max_probability_error"]),
        },
        {
            **base,
            "metric": "prediction_consistency",
            "value": float(result["prediction_consistency"]),
        },
    ]
