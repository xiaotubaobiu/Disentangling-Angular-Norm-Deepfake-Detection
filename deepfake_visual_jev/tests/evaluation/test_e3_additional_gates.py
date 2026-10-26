from pathlib import Path
import importlib.util
import json

import pandas as pd
import pytest
import yaml

from evaluation.smolvlm_gates import negation_results, composition_results, matched_control_results


CONFIG = yaml.safe_load((Path(__file__).resolve().parents[2] / 'experiments/2026-09-23_e3_smolvlm_candidate/configs/diagnostics.yaml').read_text())


def negation_fixture():
    rows = []
    for seed in (42, 7, 2048):
        for domain in ('ffpp', 'cdf'):
            for label in (0, 1):
                for question in CONFIG['negation']['questions']:
                    yes = (question['yes_means'] == 'fake') == bool(label)
                    rows.append({'seed': seed, 'domain': domain, 'video_key': f'v{label}',
                                 'question_id': question['id'], 'label': label,
                                 'p_yes': .9 if yes else .1, 'prob_fake': .9 if label else .1})
    return pd.DataFrame(rows)


def test_negation_gate_catches_semantic_failure_even_with_good_overall_authenticity():
    rows = negation_fixture()
    assert all(r['passed'] for r in negation_results(rows, CONFIG))
    changed = rows.copy()
    mask = changed.question_id.eq('zzz_neg_auth_negative') & changed.domain.eq('cdf')
    changed.loc[mask, 'prob_fake'] = 1 - changed.loc[mask, 'prob_fake']
    changed.loc[mask, 'p_yes'] = 1 - changed.loc[mask, 'p_yes']
    results = negation_results(changed, CONFIG)
    failed = [r for r in results if not r['passed']]
    assert {r['metric'] for r in failed} == {'negative_balanced_accuracy', 'complement_consistency'}
    assert all(r['domain'] == 'cdf' for r in failed)


def test_composition_gate_requires_both_accuracy_and_preservation_of_heldout_scores():
    rows = []
    for seed in (42, 7, 2048):
        for domain in ('ffpp', 'cdf'):
            for family, label in [('camera', 0), ('swap', 1), ('reenact', 1)]:
                for task in ('T1', 'T5'):
                    rows.append({'seed': seed, 'domain': domain, 'task_id': task, 'label': label,
                                 'method_family': family, 'prob_fake': .9 if label else .1,
                                 **{f'p_{c}': .9 if c == family else .05 for c in ('camera', 'swap', 'reenact')}})
    heldout = pd.DataFrame(rows)
    assert all(r['passed'] for r in composition_results(heldout, heldout, CONFIG))
    bad = heldout.copy()
    bad.loc[bad.task_id.eq('T5'), ['p_camera', 'p_swap', 'p_reenact']] = [.9, .05, .05]
    results = composition_results(bad, heldout, CONFIG)
    assert all(not r['passed'] for r in results if r['task_id'] == 'T5')


def test_matched_control_bootstrap_measures_balanced_accuracy_for_all_binary_tasks():
    rows = []
    for seed in (42, 7, 2048):
        for domain in ('ffpp', 'cdf'):
            for kind, tasks in [('candidates', ('T1', 'T2', 'T3', 'T4', 'T5')), ('questions', ('T2', 'T4'))]:
                for task in tasks:
                    for family, method, label in [('camera', 'real', 0), ('swap', 'simswap', 1), ('reenact', 'fomm', 1)]:
                        for polarity in (('swap', 'reenact') if task == 'T4' else ('fake',)):
                            target_yes = family == polarity if task == 'T4' else bool(label)
                            row = {'seed': seed, 'domain': domain, 'condition': f'meaningless_matched_{kind}',
                                   'task_id': task, 'method_id': method, 'video_id': family,
                                   'video_key': f'{method}::{family}', 'label': label, 'method_family': family,
                                   'question_id': polarity, 'variant_id': polarity, 'question_yes_means': polarity,
                                   'question_pair_id': 'pair', 'num_frames': 1,
                                   'target_semantic': ('yes' if target_yes else 'no') if task in ('T2', 'T4') else (family if task == 'T5' else ('fake' if label else 'real')),
                                   'prob_fake': .5, 'meaningful_prob_fake': .9 if label else .1,
                                   'p_yes': .5, 'meaningful_p_yes': .9 if target_yes else .1}
                            for c in ('camera', 'swap', 'reenact'):
                                row[f'p_{c}'] = 1 / 3
                                row[f'meaningful_p_{c}'] = .9 if family == c else .05
                            rows.append(row)
    results = matched_control_results(pd.DataFrame(rows), CONFIG)
    assert len(results) == 14
    for result in results:
        expected = 2 / 3 if result['task_id'] == 'T5' else .5
        assert result['point'] == pytest.approx(expected)
        assert result['lower'] == pytest.approx(expected)
        assert result['passed']


def test_missing_scientific_evidence_replaces_stale_success_with_unadjudicated_status(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[2] / 'experiments/2026-09-23_e3_smolvlm_candidate/scripts/assess_candidate.py'
    spec = importlib.util.spec_from_file_location('e3_assessment_missing_evidence', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'PHASE', tmp_path)
    output = tmp_path / 'results/summary/gate_status.json'
    output.parent.mkdir(parents=True)
    output.write_text(json.dumps({'adjudicated': True, 'stage_1_succeeded': True}))
    assert module.main() == 2
    status = json.loads(output.read_text())
    assert status['adjudicated'] is False and status['stage_1_succeeded'] is None
