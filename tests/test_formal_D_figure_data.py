"""Figure inputs must retain complete matched seeds and their own AUC baselines."""
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

import pandas as pd
import pytest


SEEDS = [7, 42, 123, 2026, 3407]
BONES = ['clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0']


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/figures/formal_D_figure_data.py'
    assert path.is_file(), 'Verified formal figure data loader is not implemented'
    spec = importlib.util.spec_from_file_location('formal_D_figure_data', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_root(root, backbones=BONES):
    """Synthetic in-memory AUC cells are written only inside pytest tmp_path."""
    for bone in backbones:
        folder = root / 'analysis/D' / bone / 'five_seed_summary'
        folder.mkdir(parents=True)
        rows = []
        effects = []
        analyses = {}
        for seed in SEEDS:
            analysis = root / 'analysis/D' / f'D_formal_{bone}_seed{seed}' / 'complete.json'
            analysis.parent.mkdir()
            analysis.write_text(json.dumps(dict(run_id=analysis.parent.name)))
            analyses[str(seed)] = sha(analysis)
            for domain, aggregation in itertools.product(['source_test', 'target_full'], ['mean_logit', 'mean_probability']):
                # One setting reverses; the other has a larger gain at a lower
                # absolute collision AUC. Both preserve their own Raw baseline.
                values = (.8, .7, .5, .55) if aggregation == 'mean_logit' else (.78, .79, .45, .49)
                for (grouping, representation), auc in zip(itertools.product(['correct', 'collision'], ['raw', 'l2']), values):
                    nr, nf = (140, 2884 if grouping == 'correct' else 280) if domain == 'source_test' else (157, 12037 if grouping == 'correct' else 1830)
                    rows.append(dict(backbone=bone, seed=seed, domain=domain, grouping=grouping,
                        representation=representation, aggregation=aggregation, pooled_auc=auc,
                        ranking_usable=True, n_real=nr, n_fake=nf))
                a, b, c, d = values
                effects.append(dict(backbone=bone, seed=seed, domain=domain, aggregation=aggregation,
                    delta_correct=b-a, delta_collision=d-c, interaction=(d-c)-(b-a)))
        pd.DataFrame(rows).to_csv(folder/'per_seed_metrics.csv', index=False, float_format='%.17g')
        pd.DataFrame(effects).to_csv(folder/'per_seed_effects.csv', index=False, float_format='%.17g')
        complete = dict(schema='D_FIVE_SEED_SUMMARY_COMPLETE_V1', status='SUMMARY_COMPLETE',
            backbone=bone, training_seed_count=5, identity=dict(seeds=SEEDS, analysis_complete_sha256=analyses),
            artifact_sha256={name: sha(folder/name) for name in ['per_seed_metrics.csv', 'per_seed_effects.csv']})
        (folder/'complete.json').write_text(json.dumps(complete))
    return root


def rewrite_hash(folder, name):
    p = folder/'complete.json'
    d = json.loads(p.read_text())
    d['artifact_sha256'][name] = sha(folder/name)
    p.write_text(json.dumps(d))


def test_all_five_seeds_and_each_setting_keep_their_own_baseline(tmp_path):
    m = module()
    data = m.load_verified(fixture_root(tmp_path))
    assert len(data['metrics']) == 240 and len(data['effects']) == 60
    absolute, paired = m.figure_tables(data['metrics'], data['effects'], 'target_full')
    assert len(absolute) == 120 and len(paired) == 60
    q = absolute[(absolute.backbone == 'clip_vit_b16') & (absolute.seed == 7) & (absolute.aggregation == 'mean_probability')]
    s = q.set_index(['grouping', 'representation']).pooled_auc
    assert s['correct', 'l2'] == .79 and s['collision', 'l2'] == .49
    p = paired[(paired.backbone == 'clip_vit_b16') & (paired.seed == 7) & (paired.aggregation == 'mean_probability')]
    assert p.set_index('effect').estimate_pp['delta_correct'] == pytest.approx(1.)
    assert p.set_index('effect').estimate_pp['interaction'] == pytest.approx(3.)


def test_missing_strict_summary_is_rejected_even_when_models_exist(tmp_path):
    m = module()
    fixture_root(tmp_path)
    (tmp_path/'analysis/D/dinov2_vit_b14/five_seed_summary/complete.json').unlink()
    with pytest.raises(ValueError, match='strict five-seed summary'):
        m.load_verified(tmp_path)


def test_changed_summary_file_is_rejected(tmp_path):
    m = module()
    fixture_root(tmp_path)
    p = tmp_path/'analysis/D/clip_vit_b16/five_seed_summary/per_seed_metrics.csv'
    p.write_text(p.read_text()+'\n')
    with pytest.raises(ValueError, match='hash'):
        m.load_verified(tmp_path)


def test_summary_cannot_reference_a_changed_model_analysis(tmp_path):
    m = module()
    fixture_root(tmp_path)
    p = tmp_path/'analysis/D/D_formal_clip_vit_b16_seed7/complete.json'
    p.write_text('{}')
    with pytest.raises(ValueError, match='analysis'):
        m.load_verified(tmp_path)


def test_missing_or_duplicate_setting_cannot_be_silently_averaged(tmp_path):
    m = module()
    fixture_root(tmp_path)
    folder = tmp_path/'analysis/D/clip_vit_b16/five_seed_summary'
    p = folder/'per_seed_metrics.csv'
    d = pd.read_csv(p)
    d.iloc[:-1].to_csv(p, index=False)
    rewrite_hash(folder, p.name)
    with pytest.raises(ValueError, match='complete matched'):
        m.load_verified(tmp_path)


def test_effects_must_reconstruct_from_all_eight_auc_cells(tmp_path):
    m = module()
    fixture_root(tmp_path)
    folder = tmp_path/'analysis/D/clip_vit_b16/five_seed_summary'
    p = folder/'per_seed_effects.csv'
    d = pd.read_csv(p)
    d.loc[0, 'delta_correct'] = .8
    d.to_csv(p, index=False)
    rewrite_hash(folder, p.name)
    with pytest.raises(ValueError, match='reconstruct'):
        m.load_verified(tmp_path)


def test_explicit_single_backbone_still_requires_its_full_five_seeds(tmp_path):
    m = module()
    fixture_root(tmp_path, ['dinov2_vit_b14'])
    d = m.load_verified(tmp_path, backbones=['dinov2_vit_b14'])
    assert set(d['metrics'].backbone) == {'dinov2_vit_b14'}
    folder = tmp_path/'analysis/D/dinov2_vit_b14/five_seed_summary'
    p = folder/'complete.json'
    record = json.loads(p.read_text())
    record['identity']['seeds'] = [7, 42, 123, 2026]
    p.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='five registered'):
        m.load_verified(tmp_path, backbones=['dinov2_vit_b14'])


def test_seed_sd_is_computed_from_paired_differences(tmp_path):
    m = module()
    frame = pd.DataFrame(dict(backbone=['clip_vit_b16']*5, aggregation=['mean_logit']*5,
        effect=['delta_correct']*5, seed=SEEDS, estimate_pp=[-2., -1., 0., 1., 2.]))
    result = m.paired_seed_summary(frame).iloc[0]
    assert result.estimate_mean_pp == 0
    assert result.estimate_sample_sd_pp == pytest.approx(2.5**.5)
    assert result.n_training_seeds == 5
