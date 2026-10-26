from pathlib import Path
import sys

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(PHASE / 'scripts'), str(ROOT / 'src')]

from bridge_scorer import compose
from evaluate_stage_b import aggregate, balanced_accuracy, polarity_consistency
from train import Examples
from data.semantic_language import load_language_bank
from evaluate_stage_c import check_stage_b, output_dir, score as stage_c_score


def test_question_candidate_is_one_semantic_input():
    assert compose('T2', 'Is this image authentic?', 'Yes.') == (
        'Question: Is this image authentic? Answer: Yes.')
    assert compose('T5', 'Which family?', 'an identity-swapped face.') == (
        'Question: Which family? Answer: This is an identity-swapped face.')


def test_post_failure_diagnostic_requires_explicit_isolated_mode():
    import pytest

    failed = {'stage_b_status': 'FAIL'}
    with pytest.raises(ValueError):
        check_stage_b(failed, exploratory=False)
    check_stage_b(failed, exploratory=True)
    assert output_dir(True) != output_dir(False)


def test_pinned_tokenizer_uses_roberta_bpe_not_characters():
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(
        PHASE / 'results/pretrained/roberta-base-tokenizer')
    tokens = tokenizer.convert_ids_to_tokens(tokenizer(
        'Question: Is this facial image authentic? Answer: Yes.')['input_ids'])
    assert 'Question' in tokens
    assert 'Ġauthentic' in tokens
    assert len(tokens) < 20


def test_balanced_draws_use_all_tasks_and_families():
    rows = [
        {'method': 'real', 'label': 0},
        {'method': 'faceswap', 'label': 1},
        {'method': 'MRAA', 'label': 1},
    ]
    from data.semantic_dataset import method_family
    assert {method_family(row['method'], row['label']) for row in rows} == {
        'camera', 'swap', 'reenact'}
    bank = load_language_bank(PHASE.parent / '2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml')
    data = Examples(rows, bank, split='train', seed=42, epoch=1, count=300)
    counts = {task: 0 for task in ('T1', 'T2', 'T3', 'T4', 'T5')}
    t4_answer_counts = {'yes': 0, 'no': 0}
    for index, task, question in data.draws:
        counts[task] += 1
        if task == 'T4':
            family = method_family(rows[index]['method'], rows[index]['label'])
            yes_means = next(entry.yes_means for entry in bank.task('T4', 'train').questions
                             if entry.name == question)
            t4_answer_counts['yes' if family == yes_means else 'no'] += 1
    assert counts == {task: 60 for task in counts}
    assert t4_answer_counts == {'yes': 30, 'no': 30}


def test_stage_b_metrics_require_correct_polarity():
    rows = []
    for key, label in [('real', 0), ('fake', 1)]:
        for polarity in ('real', 'fake'):
            yes = (label == 0) == (polarity == 'real')
            rows.append({'video_key': key, 'variant_id': 'q::answers',
                         'label': label, 'yes_means': polarity,
                         'p_yes': 0.9 if yes else 0.1})
    assert balanced_accuracy(rows)[0] == 1.0
    assert polarity_consistency(rows) == (1.0, 2)


def test_video_aggregation_uses_mean_frame_probability():
    frames = [
        {'video_key': 'real::1', 'variant_id': 'q::a', 'question_id': 'q',
         'label': 0, 'yes_means': 'real', 'p_yes': 0.2},
        {'video_key': 'real::1', 'variant_id': 'q::a', 'question_id': 'q',
         'label': 0, 'yes_means': 'real', 'p_yes': 0.8},
    ]
    videos = aggregate(frames)
    assert len(videos) == 1
    assert videos[0]['frames'] == 2
    assert videos[0]['p_yes'] == 0.5


def test_stage_c_maps_mixed_candidate_counts_to_semantics():
    import torch

    class StubScorer:
        def __call__(self, images, tasks, questions, candidates):
            rows = []
            width = max(map(len, candidates))
            for options in candidates:
                logits = [2.0 if 'camera' in text else 1.0 for text in options]
                rows.append(logits + [-1e4] * (width - len(options)))
            return torch.tensor(rows)

    batch = [
        {'image': None, 'task_id': 'T1', 'question_text': 'Which?',
         'candidate_texts': ('real.', 'fake.', ''),
         'candidate_semantics': ('real', 'fake', ''),
         'candidate_mask': torch.tensor([True, True, False])},
        {'image': None, 'task_id': 'T5', 'question_text': 'Which?',
         'candidate_texts': ('camera.', 'swap.', 'reenact.'),
         'candidate_semantics': ('camera', 'swap', 'reenact'),
         'candidate_mask': torch.tensor([True, True, True])},
    ]
    normal = stage_c_score(StubScorer(), batch)
    reversed_order = stage_c_score(StubScorer(), batch, reverse=True)
    assert set(normal[0]) == {'real', 'fake'}
    assert set(normal[1]) == {'camera', 'swap', 'reenact'}
    for original, permuted in zip(normal, reversed_order):
        assert original.keys() == permuted.keys()
        for semantic in original:
            assert abs(original[semantic] - permuted[semantic]) < 1e-6
