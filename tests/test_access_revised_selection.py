"""A publication revision must replace all requested H models and retain D."""
import importlib.util
import itertools
from pathlib import Path

import pandas as pd
import pytest


def module():
    path = Path(__file__).resolve().parents[1] / 'scripts/reproducibility/access_20261103/assemble_revision.py'
    spec = importlib.util.spec_from_file_location('access_assembly', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def inputs():
    bones = ['clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0']
    old, files = [], {}
    for route, bone in itertools.product(['D', 'H'], bones):
        for seed in ([7,42,123,2026,3407] if route == 'D' else [7,42,123]):
            for history in (['B'] if route == 'D' else ['B','AB']):
                run = f'{route}_formal_{bone}_{history}_seed{seed}'
                old.append(dict(run_id=run, route=route, backbone=bone, history=history,
                    seed=seed, max_epochs=60, actual_epochs=60, best_epoch=45,
                    best_model_sha256='old-' + run))
                if route == 'H' and bone != 'clip_vit_b16':
                    prefix = Path('/revision/runs') / run
                    files[prefix / 'complete.json'] = dict(status='TRAINING_COMPLETE', epochs=10,
                        max_epochs=10, best_epoch=3, early_stopped=False, success=1850,
                        skip=0, target_used=False, epoch_recipe_sha256='recipe', best_sha256='new-' + run)
                    files[prefix / 'resolved_config.json'] = dict(epochs=10, fixed_epoch_budget=True, lr=1e-4)
                    files[prefix / 'history.json'] = [dict(dev={'pooled_auc': .7}, selection={'best_score': .75}) for _ in range(10)]
    return pd.DataFrame(old), files


def test_both_revised_backbones_replace_twelve_models_and_keep_other_rows():
    old, files = inputs()
    result = module().revised_model_selection(old, Path('/revision'), files.__getitem__)
    assert len(result) == 33 and result.run_id.is_unique
    retained = old[(old.route == 'D') | (old.backbone == 'clip_vit_b16')]
    pd.testing.assert_frame_equal(result.set_index('run_id').loc[retained.run_id, retained.columns[1:]],
        retained.set_index('run_id'), check_dtype=False)
    fresh = result[(result.route == 'H') & (result.backbone != 'clip_vit_b16')]
    assert len(fresh) == 12 and fresh.actual_epochs.eq(10).all() and fresh.best_epoch.eq(3).all()
    assert fresh.best_model_sha256.str.startswith('new-').all()


def test_an_old_fifteen_epoch_efficientnet_cannot_enter_current_selection():
    old, files = inputs()
    files[Path('/revision/runs/H_formal_efficientnet_b0_B_seed7/complete.json')]['epochs'] = 15
    with pytest.raises(ValueError, match='fixed10'):
        module().revised_model_selection(old, Path('/revision'), files.__getitem__)


def test_the_two_histories_must_select_the_same_epoch():
    old, files = inputs()
    files[Path('/revision/runs/H_formal_efficientnet_b0_AB_seed7/complete.json')]['best_epoch'] = 4
    with pytest.raises(ValueError, match='paired epoch'):
        module().revised_model_selection(old, Path('/revision'), files.__getitem__)


def test_missing_old_registered_model_is_not_silently_filled():
    old, files = inputs()
    with pytest.raises(ValueError, match='registered'):
        module().revised_model_selection(old.iloc[1:], Path('/revision'), files.__getitem__)
