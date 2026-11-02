"""Formal cache provenance, directional effects and sparse class-group guards."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from journal_detector import analysis_d, bootstrap_fast, cache_d, protocol, scoring_d, summary_d, trainer


@pytest.fixture
def formal_context(tmp_path, monkeypatch):
    config = json.loads((Path(__file__).resolve().parents[1] / 'configs/journal.json').read_text())
    config.update(protocol=protocol.PROTOCOL, artifact_root=str(tmp_path),
                  h_formal_seeds=[7, 42, 123], development_seeds=[1007, 1042])
    trainer.save_json(tmp_path/'data/D/freeze.json',{'status':'FROZEN','fixture':'SYNTHETIC_ONLY'})
    for backbone in config['backbones']:
        recipe = tmp_path / 'recipes' / (backbone + '.json')
        trainer.save_json(recipe, dict(schema='EPOCH_RECIPE_V2',status='FROZEN',route='D',
            backbone=backbone, chosen_epochs=9,
            data_freeze_sha256=trainer.digest(tmp_path/'data/D/freeze.json'),
            target_development_used=True, development_seeds=[1007, 1042]))
        for task in config['runs'].values():
            if task['backbone'] == backbone:
                task.update(epochs=9, epoch_recipe_sha256=trainer.digest(recipe),
                            epoch_recipe_path=str(recipe.relative_to(tmp_path)))
    run_id = next(iter(config['runs']))
    task = config['runs'][run_id]
    identity = {'verified_trainer_identity': 'synthetic-boundary'}
    monkeypatch.setattr(trainer, 'training_identity', lambda *args: identity.copy())
    out = tmp_path / 'runs' / run_id
    out.mkdir(parents=True)
    checkpoint = out / 'best_model.pth'
    checkpoint.write_bytes(b'CONTEXT_ONLY_NO_MODEL_INFERENCE')
    receipt = dict(status='TRAINING_COMPLETE', identity=identity,
        best_sha256=trainer.digest(checkpoint), epochs=6, max_epochs=9, best_epoch=2,
        selection='source_dev_pooled_mean_probability_auc_ema', target_used=False,
        target_development_used=True, epoch_recipe_sha256=task['epoch_recipe_sha256'])
    trainer.save_json(out / 'complete.json', receipt)
    config_path = tmp_path / 'configs/execution.json'
    trainer.save_json(config_path, config)
    return config_path, run_id, receipt


def test_formal_context_keeps_target_assisted_budget_provenance(formal_context):
    path, run, _ = formal_context
    base, config, task = cache_d._context(path, run)
    assert base['target_development_used'] is True
    assert base['formal_trajectory_target_used'] is False
    assert base['protocol'] == protocol.PROTOCOL
    assert base['epoch_recipe_sha256'] == task['epoch_recipe_sha256']
    assert base['epoch_recipe_path'] == str(Path(config['artifact_root']) / task['epoch_recipe_path'])
    assert 'cdf_dev' not in cache_d.ROLES
    with pytest.raises(ValueError, match='role'):
        cache_d._role_identity(base, 'cdf_dev', Path(config['artifact_root']) / 'data/D/cdf_dev.parquet')


@pytest.mark.parametrize('field,value', [
    ('target_development_used', False), ('target_used', True),
    ('epoch_recipe_sha256', 'f' * 64),
])
def test_formal_context_rejects_false_or_mismatched_budget_provenance(formal_context, field, value):
    path, run, receipt = formal_context
    receipt[field] = value
    trainer.save_json(path.parent.parent / 'runs' / run / 'complete.json', receipt)
    with pytest.raises(ValueError):
        cache_d._context(path, run)


def test_formal_context_rejects_changed_frozen_epoch_recipe(formal_context):
    path, run, _ = formal_context
    config = json.loads(path.read_text())
    recipe = path.parent.parent / config['runs'][run]['epoch_recipe_path']
    recipe.write_text(recipe.read_text() + '\n')
    with pytest.raises(ValueError, match='epoch recipe'):
        cache_d._context(path, run)


def test_development_run_cannot_enter_formal_cache_context(formal_context):
    path, run, _ = formal_context
    config = json.loads(path.read_text())
    config['runs'][run].update(phase='development', recipe_frozen=False,
        seed=1007, cdf_development_manifest='data/D/cdf_dev.parquet')
    trainer.save_json(path, config)
    with pytest.raises(ValueError, match='formal recipe'):
        cache_d._context(path, run)


def test_cache_context_requires_all_fifteen_frozen_formal_D_runs(formal_context):
    path, run, _ = formal_context
    config=json.loads(path.read_text())
    config['runs'].pop(next(key for key in config['runs'] if key!=run))
    trainer.save_json(path,config)
    with pytest.raises(ValueError,match='fifteen|15'):
        cache_d._context(path,run)


def effect_cells(correct, collision):
    return pd.DataFrame([dict(grouping=g, representation=r, aggregation=a,
        pooled_auc=.6 + (delta if r == 'l2' else 0.), macro_auc=.6)
        for a in ('mean_logit', 'mean_probability')
        for g, delta in [('correct', correct), ('collision', collision)]
        for r in ('raw', 'l2')])


def test_directional_reversal_is_distinct_from_opposite_sign_reversal():
    apparent = scoring_d.d_effects(effect_cells(-.1, .1), seed=7)
    opposite = scoring_d.d_effects(effect_cells(.1, -.1), seed=42)
    assert apparent.reversal.all() and opposite.reversal.all()
    assert apparent.reversal_correct_negative_collision_positive.all()
    assert not opposite.reversal_correct_negative_collision_positive.any()
    rows = pd.concat([scoring_d.d_effects(effect_cells(-.1, .1) if seed == 7 else effect_cells(.1, -.1), seed=seed)
        for seed in (7, 42, 123, 2026, 3407)], ignore_index=True)
    summary = scoring_d.summarize_seed_effects(rows)
    assert summary.n_seed_reversals.eq(5).all()
    assert summary.n_seed_reversals_correct_negative_collision_positive.eq(1).all()
    assert summary.n_seed_reversals_correct_positive_collision_negative.eq(4).all()
    report = summary_d.effect_summary(rows.assign(backbone='clip_vit_b16', domain='target_full'))
    assert report.reversal_count_denominator.eq(5).all()


def sparse_fake_videos():
    rows = [dict(video_uid=f'r{i}', group_id=f'g{i}', label=0,
                 method_id='real', score=i / 100) for i in range(30)]
    rows += [dict(video_uid=f'f{i}', group_id=f'g{i % 3}', label=1,
                  method_id='m', score=.4 + i / 100) for i in range(18)]
    return pd.DataFrame(rows)


def test_many_real_groups_cannot_hide_only_three_independent_fake_groups():
    videos = sparse_fake_videos()
    raw = bootstrap_fast.bootstrap_video_metrics_fast(videos, repeats=5)
    assert raw['intervals'] is not None  # Demonstrates the old total-group blind spot.
    guarded = analysis_d.guard_bootstrap_group_support(raw, videos)
    assert guarded['n_real_groups'] == 30
    assert guarded['n_fake_groups'] == 3
    assert guarded['method_fake_group_counts'] == {'m': 3}
    assert guarded['intervals'] is None
    assert guarded['conditional_interval_stability'] == 'insufficient_independent_class_groups'
    assert guarded['point_estimate']['pooled_auc'] == 1.


def test_joint_closure_bootstrap_reports_sparse_fake_group_counts():
    rows = []
    for row in sparse_fake_videos().to_dict('records'):
        for i in range(8):
            rows.append(dict(**row, image_path=f"/{row['video_uid']}/{i}.png",
                video_id=row['video_uid'], domain='cdf', role='target_full',
                rgb_sha256=f"{row['video_uid']}/{i}", raw_logit=row['score'], l2_logit=row['score']))
    result = scoring_d.bootstrap_d_comparisons(pd.DataFrame(rows), repeats=5)
    assert result['n_closure_groups'] == 30
    assert result['n_fake_groups'] == 3 and result['n_real_groups'] == 30
    assert result['intervals'] is None
    assert result['conditional_interval_stability'] == 'insufficient_independent_class_groups'


def test_high_precision_ranking_survives_sparse_group_receipt_roundtrip(tmp_path):
    import mpmath as mp
    with mp.workdps(320):
        values = [mp.mpf(1) - mp.mpf('1e-200'), mp.mpf(1) - mp.mpf('1e-210')]
    q = pd.DataFrame(dict(video_uid=['real', 'fake'], group_id=['r', 'f'],
        label=[0, 1], method_id=['real', 'm'], score=[1., 1.], rank_score=values, ranking_usable=True))
    analysis_d.write_scores(tmp_path / 'scores.csv', q)
    restored = analysis_d.read_scores(tmp_path / 'scores.csv')
    assert analysis_d.independent_auc(restored.label, restored.rank_score) == 1.
    assert all(isinstance(value, mp.mpf) for value in restored.rank_score)
