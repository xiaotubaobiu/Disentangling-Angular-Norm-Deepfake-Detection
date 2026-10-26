from __future__ import annotations

import importlib.util
from collections import Counter
from pathlib import Path

import pytest
import torch
import pandas as pd
from torch.utils.data import DataLoader

from data.semantic_dataset import SemanticTemplateSampler
from data.semantic_language import SEMANTIC_TO_ID, load_language_bank
from evaluation.semantic_controls import (
    build_counterfactual_requests,
    build_meaningless_candidate_request,
    build_meaningless_question_request,
    control_metric_rows,
    enumerate_candidate_permutations,
    paired_control_rows,
    permutation_audit,
    select_global_audit_indices,
)


ROOT = Path(__file__).parents[1]
MANIFEST = (
    ROOT
    / "experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml"
)
SCRIPT = (
    ROOT
    / "experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py"
)


@pytest.fixture
def bank():
    return load_language_bank(MANIFEST)


def _sample(label=1, method_id="simswap", sample_id="sample-1"):
    return {
        "image": torch.zeros(3, 8, 8),
        "label": torch.tensor(label),
        "method_id": method_id,
        "video_id": "001",
        "video_key": f"{method_id}::001",
        "image_path": f"/{sample_id}.png",
        "sample_id": sample_id,
    }


def _render(bank, *, task_id="T2", label=1, method_id="simswap", sample_id="sample-1"):
    return SemanticTemplateSampler(bank, seed=42).render(
        _sample(label, method_id, sample_id), task_id, "heldout", 0
    )


def _load_evaluator_script():
    spec = importlib.util.spec_from_file_location("evaluate_semantic_task6", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_permutation_enumeration_is_complete_and_rejects_other_arities():
    assert enumerate_candidate_permutations(2) == ((0, 1), (1, 0))
    assert enumerate_candidate_permutations(3) == (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    )
    with pytest.raises(ValueError, match="two or three"):
        enumerate_candidate_permutations(1)


class _EquivariantModel:
    def __init__(self):
        self.training = False
        self.calls = 0

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
        return self

    def classify_features(self, image, question, candidates, mask):
        self.calls += 1
        return candidates[..., 0].masked_fill(~mask, float("-inf"))


def test_permutation_audit_remaps_every_binary_and_ternary_order():
    model = _EquivariantModel()
    image = torch.zeros(2, 2)
    questions = torch.zeros(2, 2)
    candidates = torch.tensor(
        [[[0.2], [0.8], [99.0]], [[0.1], [0.3], [0.9]]]
    )
    mask = torch.tensor([[True, True, False], [True, True, True]])

    result = permutation_audit(model, image, questions, candidates, mask)

    assert result["max_probability_error"] == pytest.approx(0.0)
    assert result["prediction_consistency"] == pytest.approx(1.0)
    assert result["consistent_count"] == 8
    assert result["comparison_count"] == 8
    # One reference call plus two binary and six ternary permutations.
    assert model.calls == 9


def test_permutation_audit_exposes_canonical_prediction_failures():
    class PositionBiased(_EquivariantModel):
        def classify_features(self, image, question, candidates, mask):
            self.calls += 1
            bias = torch.tensor([2.0, 0.0, -2.0])[: candidates.shape[1]]
            return (candidates[..., 0] + bias).masked_fill(~mask, float("-inf"))

    result = permutation_audit(
        PositionBiased(),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.tensor([[[0.0], [0.1], [0.2]]]),
        torch.ones(1, 3, dtype=torch.bool),
    )

    assert result["max_probability_error"] > 0.0
    assert result["prediction_consistency"] < 1.0


def test_global_audit_subset_uses_one_sorted_sample_order():
    sample_ids = ["z", "a", "m", "b"]
    assert select_global_audit_indices(sample_ids, limit=3) == (1, 3, 2)
    assert select_global_audit_indices(sample_ids, limit=99) == (1, 3, 2, 0)
    with pytest.raises(ValueError, match="unique"):
        select_global_audit_indices(["same", "same"], limit=1)


@pytest.mark.parametrize(
    ("task_id", "label", "method_id", "polarities", "targets"),
    [
        ("T2", 1, "simswap", {"real", "fake"}, {"no", "yes"}),
        ("T4", 1, "simswap", {"swap", "reenact"}, {"yes", "no"}),
    ],
)
def test_counterfactuals_share_explicit_pair_id_and_keep_truth(
    bank, task_id, label, method_id, polarities, targets
):
    requests = build_counterfactual_requests(
        task_id,
        "heldout",
        _sample(label, method_id, "pair-sample"),
        bank,
        seed=42,
    )

    assert len(requests) == 2
    assert {request.question_yes_means_name for request in requests} == polarities
    assert {request.target_semantic for request in requests} == targets
    assert len({request.question_pair_id for request in requests}) == 1
    assert len({request.render_id for request in requests}) == 1
    assert len({tuple(request.candidate_text_ids.tolist()) for request in requests}) == 1
    assert all(request.hidden_visual_truth == ("real" if label == 0 else "fake") for request in requests)
    assert all(request.hidden_method_family == "swap" for request in requests)


def test_counterfactual_rejects_nonpaired_tasks(bank):
    with pytest.raises(ValueError, match="T2 or T4"):
        build_counterfactual_requests("T1", "heldout", _sample(), bank)


def test_meaningless_candidates_are_manifest_only_repeatable_and_balanced(bank):
    allowed_ids = {entry.text_id for entry in bank.entries("control", "candidate")}
    assignments = []
    # Fixed unseen IDs, 4,000 trials: each label/position cell has expectation
    # 1,000. A 100-count deviation exceeds four binomial standard deviations.
    for index in range(4000):
        meaningful = _render(
            bank,
            task_id="T2",
            label=index % 2,
            method_id="simswap" if index % 2 else "real",
            sample_id=f"balance-holdout-{index}",
        )
        request = build_meaningless_candidate_request(
            meaningful, meaningful["sample_id"], bank
        )
        repeated = build_meaningless_candidate_request(
            meaningful, meaningful["sample_id"], bank
        )
        assert request.candidate_text_ids.tolist() == repeated.candidate_text_ids.tolist()
        assert request.candidate_semantic_ids.tolist() == repeated.candidate_semantic_ids.tolist()
        assert set(request.candidate_text_ids[request.candidate_mask].tolist()) <= allowed_ids
        assert request.target_semantic == meaningful["target_semantic"]
        assignments.append((request.target_index.item(), int(request.label)))

    counts = Counter(assignments)
    assert set(counts) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert all(abs(count - 1000) <= 100 for count in counts.values())


def test_meaningless_ternary_candidates_preserve_hidden_truth_not_surface_semantics(bank):
    meaningful = _render(bank, task_id="T5", sample_id="ternary")
    request = build_meaningless_candidate_request(meaningful, "ternary", bank)

    assert request.target_semantic == "swap"
    assert request.candidate_mask.tolist() == [True, True, True]
    assert {
        value for value in request.candidate_semantics if value
    } == {"camera", "swap", "reenact"}
    assert request.target_index.item() == request.candidate_semantics.index("swap")
    assert all("nonsense" in candidate_id for candidate_id in request.candidate_ids)


def test_meaningless_question_is_manifest_only_and_retains_hidden_polarity(bank):
    meaningful = _render(bank, task_id="T4", sample_id="question-control")
    request = build_meaningless_question_request(
        meaningful, "question-control", bank
    )
    question_entries = {entry.text_id: entry for entry in bank.entries("control", "question")}

    assert request.question_text_id.item() in question_entries
    assert request.question_text == question_entries[request.question_text_id.item()].text
    assert request.question_yes_means_name == meaningful["question_yes_means_name"]
    assert request.question_yes_means.item() == meaningful["question_yes_means"].item()
    assert request.target_semantic == meaningful["target_semantic"]
    assert request.control_pair_id


def test_paired_control_rows_retain_sample_scores_drop_and_normalized_schema():
    metadata = [
        {
            "seed": 42,
            "domain": "ffpp",
            "condition": "meaningless_questions",
            "task_id": "T2",
            "sample_id": "sample-1",
            "video_key": "simswap::001",
            "target_semantic": "yes",
            "control_pair_id": "pair-1",
        }
    ]
    meaningful = torch.tensor([[0.0, 0.0, 0.2, 0.8, 0.0, 0.0, 0.0]])
    control = torch.tensor([[0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.0]])

    rows = paired_control_rows(metadata, meaningful, control)

    assert rows == [
        {
            **metadata[0],
            "pairing_key": "pair-1",
            "metric": "target_probability_drop_diagnostic",
            "value": pytest.approx(0.4),
            "count": 1,
            "meaningful_score": pytest.approx(0.8),
            "control_score": pytest.approx(0.4),
            "drop": pytest.approx(0.4),
        }
    ]


@pytest.mark.parametrize("task_id,metric", [("T1", "auroc"), ("T2", "balanced_accuracy"), ("T3", "auroc"), ("T4", "balanced_accuracy"), ("T5", "macro_balanced_accuracy")])
def test_g6_uses_paired_primary_performance_metrics(task_id, metric):
    rows = []
    for index, family in enumerate(("camera", "swap", "reenact")):
        label = int(index != 0)
        row = dict(seed=42, domain="ffpp", condition="meaningless_candidates", task_id=task_id,
                   video_key=f"{family}::{index}", method_id=family, label=label,
                   method_family=family, target_semantic="yes" if label else "no",
                   pairing_key=f"pair-{index}", prob_fake=0.5, meaningful_prob_fake=0.1 if not label else 0.9,
                   p_yes=0.5, meaningful_p_yes=0.1 if not label else 0.9)
        for semantic in ("camera", "swap", "reenact"):
            row[f"p_{semantic}"] = 1 / 3
            row[f"meaningful_p_{semantic}"] = 0.8 if semantic == family else 0.1
        rows.append(row)
    result = control_metric_rows(rows)
    assert len(result) == 1
    assert result[0]["metric"] == f"{metric}_drop"
    assert result[0]["meaningful_score"] == 1.0
    assert result[0]["value"] == pytest.approx(2 / 3 if task_id == "T5" else 0.5)
    assert result[0]["count"] == 3
    assert result[0]["pairing_key"]


def test_hidden_candidate_assignment_is_label_and_position_independent(bank):
    for task in ("T2", "T5"):
        original = _render(bank, task_id=task, sample_id="independent")
        changed = dict(original, label=torch.tensor(0), target_semantic="no" if task == "T2" else "camera")
        changed["candidate_semantics"] = tuple(reversed(original["candidate_semantics"][:2])) + ("",) if task == "T2" else tuple(reversed(original["candidate_semantics"]))
        left = build_meaningless_candidate_request(original, "independent", bank)
        right = build_meaningless_candidate_request(changed, "independent", bank)
        assert left.candidate_semantic_ids.tolist() == right.candidate_semantic_ids.tolist()
        assert left.candidate_text_ids.tolist() == right.candidate_text_ids.tolist()


@pytest.mark.parametrize("task_id,size", [("T2", 2), ("T5", 3)])
def test_hidden_word_meaning_position_assignments_are_counterbalanced(bank, task_id, size):
    meaningful = _render(bank, task_id=task_id)
    counts = Counter()
    # Independent fixed population, with an a priori six-sigma cell bound.
    population = 6000
    for index in range(population):
        request = build_meaningless_candidate_request(meaningful, f"surface-holdout-{index}", bank)
        for position in range(size):
            counts[(position, request.candidate_ids[position], request.candidate_semantics[position])] += 1
    expectation = population / (size * size)
    sigma = (population * (1 / size**2) * (1 - 1 / size**2)) ** 0.5
    assert len(counts) == size**3
    assert all(abs(value - expectation) <= 6 * sigma for value in counts.values())


@pytest.mark.parametrize("condition,control_kind", [("meaningless_candidates", "candidate"), ("meaningless_questions", "question")])
def test_condition_lookup_encodes_only_its_reachable_control_kind(bank, condition, control_kind):
    module = _load_evaluator_script()

    class Encoder:
        def encode_texts(self, texts):
            self.encoded = set(texts)
            return torch.zeros(len(texts), 2)

    model = Encoder()
    module.build_condition_text_lookup(model, bank, condition, torch.device("cpu"))
    allowed = {entry.text for entry in bank.text_entries if entry.split == "heldout" or (entry.split == "control" and entry.kind == control_kind)}
    assert model.encoded == allowed


class _PhysicalSamples:
    def __len__(self):
        return 6

    def __getitem__(self, index):
        family = ("real", "simswap", "fomm")[index // 2]
        return _sample(int(index >= 2), family, f"physical-{index}")


class _FeatureModel(_EquivariantModel):
    def __init__(self):
        super().__init__()
        self.extractions = 0

    def extract_features(self, images):
        self.extractions += 1
        return images.flatten(1)[:, :2]

    def __call__(self, images, questions, candidates, mask):
        return self.classify_features(self.extract_features(images), questions, candidates, mask)


class _Lookup:
    def lookup(self, question_ids, candidate_ids, mask):
        return question_ids.float()[:, None], candidate_ids.float()[:, :, None] / 100


@pytest.mark.parametrize("condition", ["counterfactual", "meaningless_candidates", "meaningless_questions", "permutation"])
def test_evaluator_control_integration_writes_paired_video_data(bank, tmp_path, monkeypatch, condition):
    module = _load_evaluator_script()
    monkeypatch.setattr(module, "build_eval_dataset", lambda cfg, domain: _PhysicalSamples())
    cfg = {"train": {"seed": 42, "num_workers": 0, "per_gpu_batch": 5}}
    loader = module.build_condition_loader(cfg, bank, domain="ffpp", condition=condition, distributed=False)
    model = _FeatureModel()
    rows = module.evaluate_condition(model, loader, _Lookup(), torch.device("cpu"), seed=42, domain="ffpp", condition=condition)
    module.write_outputs(rows, output_dir=tmp_path, condition=condition, domain="ffpp")
    import json
    metrics = json.loads((tmp_path / condition / "ffpp_metrics.json").read_text())
    assert metrics
    assert all({"seed", "domain", "condition", "task_id", "metric", "value", "count", "pairing_key"} <= set(row) for row in metrics)
    if condition.startswith("meaningless"):
        paired = pd.read_csv(tmp_path / condition / "ffpp_paired_videos.csv")
        assert {"meaningful_p_yes", "meaningful_prob_fake", "p_yes", "prob_fake", "pairing_key"} <= set(paired)
        assert not paired.pairing_key.duplicated().any()
        assert model.extractions == len(loader)
        assert any(row["metric"].endswith("_drop") for row in metrics)
        # Existing Task5 video groups retain different answer variants.
        t4 = paired[paired.task_id == "T4"]
        assert len(t4) == 12
        assert t4.groupby("video_key").num_frames.sum().tolist() == [4, 4, 4]
    elif condition == "counterfactual":
        controls = pd.read_csv(tmp_path / condition / "ffpp_controls.csv")
        ordering = controls[controls.metric == "true_family_ordering"]
        assert len(ordering) == 2  # fake video pairs, not four frame pairs
    else:
        assert {row["metric"] for row in metrics} == {"max_probability_error", "prediction_consistency"}
        assert metrics[0]["count"] == 6 * (4 * 2 + 6)


def test_global_subset_limits_physical_samples_before_task_expansion(bank, monkeypatch):
    module = _load_evaluator_script()

    class ManySamples:
        def __len__(self):
            return 4100

        def __getitem__(self, index):
            return _sample(sample_id=f"physical-{4099 - index:05d}")

    monkeypatch.setattr(module, "build_eval_dataset", lambda cfg, domain: ManySamples())
    loader = module.build_condition_loader({"train": {"num_workers": 0}}, bank, domain="ffpp", condition="permutation", distributed=False)
    assert len(loader.dataset) == 4096 * 5
    assert loader.dataset.base_dataset.indices[0] == 4099
    assert loader.dataset.base_dataset.indices[-1] == 4


def test_counterfactual_rejects_a_missing_physical_partner(bank):
    module = _load_evaluator_script()
    dataset = module.CounterfactualEvaluationDataset(_PhysicalSamples(), bank, tasks=("T2",), split="heldout", seed=42)
    rows = module.infer_condition_rows(_FeatureModel(), DataLoader(dataset, batch_size=4), _Lookup(), torch.device("cpu"), seed=42, domain="ffpp", condition="counterfactual")
    with pytest.raises(ValueError, match="physical.*pair"):
        module.counterfactual_control_rows(rows[1:])


def _distributed_permutation_worker(rank, init_path, output_dir):
    import torch.distributed as distributed
    torch.set_num_threads(1)
    distributed.init_process_group("gloo", init_method=f"file://{init_path}", rank=rank, world_size=2)
    try:
        module = _load_evaluator_script()
        batch = {"image": torch.zeros(1, 3, 2, 2), "question_text_id": torch.tensor([0]),
                 "candidate_text_ids": torch.tensor([[0, 1, -1]]),
                 "candidate_mask": torch.tensor([[True, True, False]])}
        # Deliberately uneven loops: one rank has zero batches.
        result = module.infer_permutation_metrics(_FeatureModel(), [batch, batch] if rank == 0 else [], _Lookup(), torch.device("cpu"), seed=42, domain="ffpp")
        torch.save(result, Path(output_dir) / f"rank-{rank}.pt")
    finally:
        distributed.destroy_process_group()


def test_permutation_ddp_reduces_once_after_uneven_rank_batches(tmp_path):
    import os
    import subprocess
    import sys
    code = "import runpy, sys; runpy.run_path(sys.argv[1])['_distributed_permutation_worker'](int(sys.argv[2]), sys.argv[3], sys.argv[4])"
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1")
    processes = [subprocess.Popen([sys.executable, "-c", code, __file__, str(rank), str(tmp_path / "init"), str(tmp_path)], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for rank in range(2)]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=45)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    for rank in range(2):
        result = torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True)
        assert result[0]["count"] == 4
        assert result[0]["value"] == pytest.approx(0.0)
        assert result[1]["value"] == 1.0


def test_evaluator_registers_all_conditions_with_minimal_text_scope():
    module = _load_evaluator_script()

    assert tuple(module.CONDITION_SPECS) == (
        "fixed_h1",
        "seen",
        "heldout",
        "permutation",
        "counterfactual",
        "meaningless_candidates",
        "meaningless_questions",
    )
    assert module.required_text_splits("fixed_h1") == ("fixed",)
    assert module.required_text_splits("seen") == ("train",)
    assert module.required_text_splits("heldout") == ("heldout",)
    assert module.required_text_splits("permutation") == ("heldout",)
    assert module.required_text_splits("counterfactual") == ("heldout",)
    assert module.required_text_splits("meaningless_candidates") == (
        "heldout",
        "control",
    )
    assert module.required_text_splits("meaningless_questions") == (
        "heldout",
        "control",
    )


def test_every_control_is_manifest_gated():
    module = _load_evaluator_script()
    controls = (
        "permutation",
        "counterfactual",
        "meaningless_candidates",
        "meaningless_questions",
    )
    for condition in controls:
        with pytest.raises(ValueError, match="language manifest SHA256"):
            module.validate_checkpoint_manifest({}, "expected", (condition,))


def test_control_pairing_rejects_mismatched_or_duplicate_pair_ids():
    meaningful = torch.zeros((2, 7))
    control = torch.zeros((2, 7))
    base = {
        "seed": 42,
        "domain": "ffpp",
        "condition": "meaningless_candidates",
        "task_id": "T1",
        "sample_id": "sample-1",
        "video_key": "real::001",
        "target_semantic": "real",
        "control_pair_id": "same",
    }
    with pytest.raises(ValueError, match="unique control pair"):
        paired_control_rows([base, dict(base)], meaningful, control)
