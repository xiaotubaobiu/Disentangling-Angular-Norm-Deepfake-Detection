from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import default_collate

from data.semantic_dataset import SemanticDataset, SemanticTemplateSampler
from data.semantic_language import SEMANTIC_TO_ID
from data.semantic_language import load_language_bank
from evaluation.semantic_eval import (
    aggregate_semantic_videos,
    canonical_fake_probability,
    compute_task_metrics,
    deduplicate_distributed_rows,
    frame_semantic_rows,
    map_candidate_probabilities,
    validate_domain_task,
)


SCRIPT = (
    Path(__file__).parents[1]
    / "experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py"
)
LANGUAGE_BANK = (
    Path(__file__).parents[1]
    / "experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml"
)


def _load_evaluator_script():
    spec = importlib.util.spec_from_file_location("evaluate_semantic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _frame_row(
    method_id: str,
    video_id: str,
    frame: int,
    label: int,
    probability: float,
    *,
    task_id: str = "T1",
    variant_id: str = "v1",
) -> dict:
    return {
        "seed": 42,
        "domain": "ffpp",
        "condition": "seen",
        "task_id": task_id,
        "variant_id": variant_id,
        "question_id": "q1",
        "question_yes_means": "",
        "candidate_order": "real|fake",
        "sample_id": f"{method_id}-{video_id}-{frame}",
        "image_path": f"/{method_id}/{video_id}/{frame}.png",
        "method_id": method_id,
        "video_id": video_id,
        "video_key": f"{method_id}::{video_id}",
        "label": label,
        "method_family": "camera" if label == 0 else "swap",
        "target_semantic": "real" if label == 0 else "fake",
        "p_real": 1.0 - probability,
        "p_fake": probability,
        "p_no": 0.0,
        "p_yes": 0.0,
        "p_camera": 0.0,
        "p_swap": 0.0,
        "p_reenact": 0.0,
        "prob_fake": probability,
    }


def test_candidate_probabilities_use_canonical_semantic_order():
    probabilities = torch.tensor([[0.7, 0.3, 0.0], [0.2, 0.3, 0.5]])
    semantic_ids = torch.tensor(
        [
            [SEMANTIC_TO_ID["fake"], SEMANTIC_TO_ID["real"], -1],
            [SEMANTIC_TO_ID["reenact"], SEMANTIC_TO_ID["camera"], SEMANTIC_TO_ID["swap"]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])

    mapped = map_candidate_probabilities(probabilities, semantic_ids, mask)

    torch.testing.assert_close(
        mapped,
        torch.tensor(
            [
                [0.3, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.3, 0.5, 0.2],
            ]
        ),
    )


@pytest.mark.parametrize(
    ("semantic_ids", "message"),
    [
        ([[0, 0]], "duplicate semantic IDs"),
        ([[0, 7]], "invalid semantic ID"),
        ([[0, -1]], "invalid semantic ID"),
    ],
)
def test_candidate_mapping_rejects_duplicate_or_invalid_valid_ids(
    semantic_ids, message
):
    with pytest.raises(ValueError, match=message):
        map_candidate_probabilities(
            torch.tensor([[0.5, 0.5]]),
            torch.tensor(semantic_ids),
            torch.tensor([[True, True]]),
        )


def test_t2_fake_probability_depends_on_question_polarity():
    authentic = {
        "task_id": "T2",
        "question_yes_means": "real",
        "p_yes": 0.2,
        "p_no": 0.8,
    }
    manipulated = {
        "task_id": "T2",
        "question_yes_means": "fake",
        "p_yes": 0.8,
        "p_no": 0.2,
    }
    assert canonical_fake_probability(authentic) == pytest.approx(0.8)
    assert canonical_fake_probability(manipulated) == pytest.approx(0.8)


def test_t5_fake_probability_is_one_minus_camera():
    row = {
        "task_id": "T5",
        "p_camera": 0.15,
        "p_swap": 0.60,
        "p_reenact": 0.25,
    }
    assert canonical_fake_probability(row) == pytest.approx(0.85)


def test_frame_rows_retain_t4_pairing_and_canonical_probabilities():
    batch = {
        "sample_id": ["sample-1"],
        "image_path": ["/swap/001/0.png"],
        "method_id": ["simswap"],
        "video_id": ["001"],
        "video_key": ["simswap::001"],
        "label": torch.tensor([1]),
        "task_id": ["T4"],
        "variant_id": ["swap-question::answers"],
        "question_id": ["swap-question"],
        "question_yes_means_name": ["swap"],
        "candidate_semantic_ids": torch.tensor(
            [[SEMANTIC_TO_ID["no"], SEMANTIC_TO_ID["yes"], -1]]
        ),
        "candidate_mask": torch.tensor([[True, True, False]]),
        "candidate_ids": [("no-answer",), ("yes-answer",), ("",)],
        "candidate_semantics": [("no",), ("yes",), ("",)],
        "target_semantic": ["yes"],
        "method_family": ["swap"],
        "render_id": ["render-1"],
    }

    rows = frame_semantic_rows(
        batch,
        torch.tensor([[0.2, 0.8, 0.0]]),
        {"condition": "heldout", "seed": 7, "domain": "cdf"},
    )

    assert rows[0]["p_yes"] == pytest.approx(0.8)
    assert rows[0]["question_yes_means"] == "swap"
    assert rows[0]["question_id"] == "swap-question"
    assert rows[0]["candidate_order"] == "no|yes"
    assert rows[0]["pairing_key"] == "7::cdf::heldout::sample-1::simswap::001"


def test_frame_rows_accept_per_sample_seed_and_domain_context():
    batch = {
        "seed": torch.tensor([42, 42]),
        "domain": ["ffpp", "ffpp"],
        "sample_id": ["a", "b"],
        "method_id": ["real", "simswap"],
        "video_id": ["1", "2"],
        "label": torch.tensor([0, 1]),
        "task_id": ["T1", "T1"],
        "candidate_semantic_ids": torch.tensor(
            [
                [SEMANTIC_TO_ID["real"], SEMANTIC_TO_ID["fake"]],
                [SEMANTIC_TO_ID["real"], SEMANTIC_TO_ID["fake"]],
            ]
        ),
        "candidate_mask": torch.ones((2, 2), dtype=torch.bool),
    }

    rows = frame_semantic_rows(
        batch, torch.tensor([[0.9, 0.1], [0.2, 0.8]]), "seen"
    )

    assert [row["seed"] for row in rows] == [42, 42]
    assert [row["domain"] for row in rows] == ["ffpp", "ffpp"]


def test_video_aggregation_keeps_same_id_from_different_methods_separate():
    rows = [
        _frame_row("simswap", "001", 1, 1, 0.9),
        _frame_row("blendface", "001", 1, 1, 0.2),
        _frame_row("real", "001", 1, 0, 0.1),
    ]
    videos = aggregate_semantic_videos(rows)
    assert set(videos.video_key) == {
        "simswap::001",
        "blendface::001",
        "real::001",
    }


def test_video_aggregation_rejects_duplicate_sample_inside_condition():
    row = _frame_row("simswap", "001", 1, 1, 0.9)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        aggregate_semantic_videos([row, dict(row)])


def test_video_aggregation_rejects_conflicting_labels_for_one_video():
    first = _frame_row("simswap", "001", 1, 1, 0.9)
    second = _frame_row("simswap", "001", 2, 0, 0.1)

    with pytest.raises(ValueError, match="conflicting labels"):
        aggregate_semantic_videos([first, second])


def test_video_aggregation_rejects_cross_variant_physical_label_conflict():
    first = _frame_row("simswap", "001", 1, 1, 0.9, variant_id="v1")
    second = _frame_row("simswap", "001", 2, 0, 0.1, variant_id="v2")
    second.update(task_id="T3", question_id="q2", candidate_order="fake|real")

    with pytest.raises(ValueError, match="conflicting labels"):
        aggregate_semantic_videos([first, second])


def test_video_aggregation_keeps_same_sample_across_variants_distinct():
    first = _frame_row("simswap", "001", 1, 1, 0.9, variant_id="v1")
    second = dict(first, variant_id="v2", p_fake=0.3, prob_fake=0.3)

    videos = aggregate_semantic_videos([first, second])

    assert len(videos) == 2
    assert set(videos.variant_id) == {"v1", "v2"}


def test_frame_permutations_aggregate_into_one_video_variant():
    first = _frame_row("simswap", "001", 1, 1, 0.8)
    second = _frame_row("simswap", "001", 2, 1, 0.2)
    second["candidate_order"] = "fake|real"

    videos = aggregate_semantic_videos([first, second])

    assert len(videos) == 1
    assert videos.iloc[0].prob_fake == pytest.approx(0.5)
    assert videos.iloc[0].num_frames == 2
    assert "candidate_order" not in videos.columns


def test_t4_aggregation_pairs_family_questions_before_fake_probability():
    base = _frame_row("simswap", "001", 0, 1, 0.0, task_id="T4")
    swap = dict(
        base,
        variant_id="swap-question::answers",
        question_id="swap-question",
        question_yes_means="swap",
        candidate_order="no|yes",
        target_semantic="yes",
        p_yes=0.8,
        p_no=0.2,
        prob_fake=float("nan"),
    )
    reenact = dict(
        base,
        variant_id="reenact-question::answers",
        question_id="reenact-question",
        question_yes_means="reenact",
        candidate_order="yes|no",
        target_semantic="no",
        p_yes=0.1,
        p_no=0.9,
        prob_fake=float("nan"),
    )

    videos = aggregate_semantic_videos([swap, reenact])

    assert len(videos) == 2
    assert videos.prob_fake.tolist() == pytest.approx([0.8, 0.8])


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        (
            [
                dict(
                    _frame_row("simswap", "001", 0, 1, 0.0, task_id="T4"),
                    question_yes_means="swap",
                    question_pair_id="pair-1",
                    p_yes=0.8,
                    prob_fake=float("nan"),
                )
            ],
            "incomplete T4 question pair",
        ),
        (
            [
                dict(
                    _frame_row("simswap", "001", 0, 1, 0.0, task_id="T4"),
                    variant_id=f"swap-{index}",
                    question_id=f"swap-{index}",
                    question_yes_means="swap",
                    question_pair_id="pair-1",
                    p_yes=0.8,
                    prob_fake=float("nan"),
                )
                for index in range(2)
            ],
            "duplicate or ambiguous T4 question pair",
        ),
    ],
)
def test_t4_aggregation_rejects_incomplete_or_duplicate_pairs(rows, message):
    with pytest.raises(ValueError, match=message):
        aggregate_semantic_videos(rows)


def test_real_evaluator_expands_and_pairs_both_t4_questions():
    script = _load_evaluator_script()
    bank = load_language_bank(LANGUAGE_BANK)

    class OneSampleDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            assert index == 0
            return {
                "image": torch.zeros(3, 2, 2),
                "label": torch.tensor(1),
                "method_id": "simswap",
                "video_id": "001",
                "video_key": "simswap::001",
                "image_path": "/simswap/001/0.png",
                "sample_id": "simswap-001-0",
            }

    ordinary = SemanticDataset(
        OneSampleDataset(),
        SemanticTemplateSampler(bank, seed=42),
        mode="suite",
        tasks=("T4",),
        split="heldout",
    )
    expanded = script.PairedT4EvaluationDataset(ordinary)
    rendered = [expanded[index] for index in range(len(expanded))]

    assert len(rendered) == 2
    assert {row["question_yes_means_name"] for row in rendered} == {
        "swap",
        "reenact",
    }
    assert len({row["question_pair_id"] for row in rendered}) == 1
    assert len({row["question_id"] for row in rendered}) == 2

    batch = default_collate(rendered)
    probabilities = torch.zeros((2, 3))
    for index, yes_probability in enumerate((0.8, 0.1)):
        yes_column = int(
            torch.where(
                batch["candidate_semantic_ids"][index] == SEMANTIC_TO_ID["yes"]
            )[0].item()
        )
        no_column = int(
            torch.where(
                batch["candidate_semantic_ids"][index] == SEMANTIC_TO_ID["no"]
            )[0].item()
        )
        probabilities[index, yes_column] = yes_probability
        probabilities[index, no_column] = 1.0 - yes_probability
    frame_rows = frame_semantic_rows(
        batch,
        probabilities,
        {"condition": "heldout", "seed": 42, "domain": "ffpp"},
    )
    videos = aggregate_semantic_videos(frame_rows)

    assert len(videos) == 2
    assert videos.prob_fake.tolist() == pytest.approx([0.8, 0.8])
    assert videos.prob_fake.notna().all()


def test_distributed_padding_is_deduplicated_only_when_rows_match():
    row = _frame_row("simswap", "001", 1, 1, 0.9)
    assert deduplicate_distributed_rows([row, dict(row)]) == [row]
    # Different batch shapes/ranks may differ by one float32 probability ULP.
    assert deduplicate_distributed_rows([row, dict(row, p_real=row["p_real"] + 6e-8)]) == [row]

    with pytest.raises(ValueError, match="conflicting distributed rows"):
        deduplicate_distributed_rows([row, dict(row, p_fake=0.1)])
    with pytest.raises(ValueError, match="conflicting distributed rows"):
        deduplicate_distributed_rows([row, dict(row, label=0)])


def test_binary_and_multiclass_metrics_are_normalized_and_sliced():
    binary_rows = [
        _frame_row("real", "r1", 0, 0, 0.1),
        _frame_row("real", "r2", 0, 0, 0.2),
        _frame_row("simswap", "f1", 0, 1, 0.8),
        _frame_row("simswap", "f2", 0, 1, 0.9),
    ]
    videos = aggregate_semantic_videos(binary_rows)
    metrics = compute_task_metrics(videos)

    assert {
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
    } <= set(metrics.columns)
    overall = metrics[metrics.method_id == "__all__"].set_index("metric")
    assert overall.loc["auroc", "value"] == pytest.approx(1.0)
    assert overall.loc["ap", "value"] == pytest.approx(1.0)
    assert overall.loc["ece", "count"] == 4
    assert set(metrics.method_id) == {"__all__", "real", "simswap"}

    t5 = pd.DataFrame(
        [
            dict(binary_rows[0], task_id="T5", method_family="camera", p_camera=0.8, p_swap=0.1, p_reenact=0.1),
            dict(binary_rows[2], task_id="T5", method_family="swap", p_camera=0.1, p_swap=0.8, p_reenact=0.1),
            dict(binary_rows[3], task_id="T5", method_family="reenact", p_camera=0.1, p_swap=0.2, p_reenact=0.7),
        ]
    )
    t5_metrics = compute_task_metrics(t5)
    t5_overall = t5_metrics[t5_metrics.method_id == "__all__"].set_index("metric")
    assert t5_overall.loc["macro_balanced_accuracy", "value"] == pytest.approx(1.0)
    assert t5_overall.loc["macro_f1", "value"] == pytest.approx(1.0)


def test_t2_f1_is_binary_positive_class_not_macro_average():
    rows = [
        dict(
            _frame_row("mixed", str(index), 0, label, 0.9, task_id="T2"),
            question_yes_means="fake",
        )
        for index, label in enumerate((1, 1, 1, 1, 0))
    ]

    metrics = compute_task_metrics(pd.DataFrame(rows))
    overall = metrics[metrics.method_id == "__all__"].set_index("metric")

    assert overall.loc["f1", "value"] == pytest.approx(8.0 / 9.0)


def test_dfd_metrics_exclude_t4_and_t5():
    for task_id in ("T1", "T2", "T3"):
        validate_domain_task("dfd", task_id)
    for task_id in ("T4", "T5", "fixed_h1"):
        with pytest.raises(ValueError, match="DFD supports only T1-T3"):
            validate_domain_task("dfd", task_id)


def test_evaluator_manifest_gate_and_split_scope():
    module = _load_evaluator_script()

    assert module.required_text_splits("fixed_h1") == ("fixed",)
    assert module.required_text_splits("seen") == ("train",)
    assert module.required_text_splits("heldout") == ("heldout",)
    module.validate_checkpoint_manifest(
        {"language_manifest_sha256": "abc"}, "abc", ("heldout",)
    )
    with pytest.raises(ValueError, match="language manifest SHA256"):
        module.validate_checkpoint_manifest({}, "abc", ("heldout",))
    with pytest.raises(ValueError, match="language manifest SHA256"):
        module.validate_checkpoint_manifest(
            {"language_manifest_sha256": "wrong"}, "abc", ("heldout",)
        )
    module.validate_checkpoint_manifest({}, "abc", ("seen", "fixed_h1"))
