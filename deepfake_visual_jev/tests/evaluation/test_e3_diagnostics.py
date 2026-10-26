from pathlib import Path

import torch
from transformers import AutoProcessor

from data.semantic_dataset import SemanticTemplateSampler
from data.semantic_language import load_language_bank
from evaluation.smolvlm_diagnostics import extended_language_bank, NegationDataset, MatchedTextPool
from models.smolvlm_candidate import CandidatePrompts


ROOT = Path(__file__).resolve().parents[2]
PHASE = ROOT / 'experiments/2026-09-23_e3_smolvlm_candidate'


def banks():
    original = load_language_bank(ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml')
    extended = extended_language_bank(original, PHASE / 'configs/diagnostics.yaml')
    return original, extended


def sample(label):
    return {'sample_id': 'same-physical-frame', 'method_id': 'simswap' if label else 'real',
            'label': torch.tensor(label), 'image': torch.empty(3, 1, 1),
            'video_key': 'source::v', 'video_id': 'v', 'image_path': '/v.png'}


def test_diagnostics_append_texts_without_rebinding_training_or_heldout_ids():
    original, extended = banks()
    for entry in original.text_entries:
        assert extended.text(entry.text_id) == entry.text
        assert extended.text_id(entry.name) == entry.text_id
    assert {e.text for e in extended.entries('diagnostic')}.isdisjoint(
        {e.text for e in original.entries('train')})


def test_explicit_negation_reverses_yes_truth_for_both_authenticity_classes():
    _, bank = banks()
    real = NegationDataset([sample(0)], bank, seed=42)
    fake = NegationDataset([sample(1)], bank, seed=42)
    assert [real[i]['target_semantic'] for i in range(4)] == ['yes', 'no', 'no', 'yes']
    assert [fake[i]['target_semantic'] for i in range(4)] == ['no', 'yes', 'yes', 'no']
    assert real[0]['question_pair_id'] == real[1]['question_pair_id']
    assert real[2]['question_pair_id'] == real[3]['question_pair_id']


def test_neutral_replacements_match_full_prompt_lengths_and_ignore_targets():
    _, bank = banks()
    processor = AutoProcessor.from_pretrained(PHASE / 'results/pretrained/smolvlm_256m', local_files_only=True)
    prompts = CandidatePrompts(processor)
    pool = MatchedTextPool(bank, prompts, PHASE / 'configs/diagnostics.yaml')
    sampler = SemanticTemplateSampler(bank, 42)
    for task in ('T1', 'T2', 'T4', 'T5'):
        request = sampler.render(sample(1), task, 'heldout', 0)
        for kind in (('candidates', 'questions') if task in ('T2', 'T4') else ('candidates',)):
            changed = pool.replace(request, kind)
            altered_truth = dict(request, target_semantic='different hidden truth', target_index=torch.tensor(2))
            again = pool.replace(altered_truth, kind)
            assert changed['question_text_id'] == again['question_text_id']
            torch.testing.assert_close(changed['candidate_text_ids'], again['candidate_text_ids'])
            for i in range(int(request['candidate_mask'].sum())):
                q = bank.text(int(request['question_text_id']))
                c = bank.text(int(request['candidate_text_ids'][i]))
                cq = pool.text(int(changed['question_text_id']))
                cc = pool.text(int(changed['candidate_text_ids'][i]))
                assert len(prompts.encode_pair(q, c)) == len(prompts.encode_pair(cq, cc))
                assert (c != cc) if kind == 'candidates' else (q != cq)
