"""Verify a completed formal H pair using the actual registered worker config."""
from pathlib import Path
import argparse
import json
import math
import datetime
import numpy as np
import pandas as pd
import torch
from journal_detector import trainer as t, cache_h
from journal_detector.trainer_h import PairedSelection, validate_pair_frames

ROOT = Path('/home/z/workspace/deepfake_journal_20261102')
BACKBONES = ('clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0')


def accept(seed, backbone):
    spec = json.loads((ROOT / 'operations/jobs' / f'H_formal_{backbone}_pair_seed{seed}/spec.json').read_text())
    config_path = Path(spec['argv'][spec['argv'].index('--config') + 1]).resolve()
    assert config_path in {ROOT / 'configs/h_execution.json', ROOT / f'configs/h_execution_{backbone}.json'}
    expected_config_sha = t.digest(config_path)
    config = json.loads(config_path.read_text())
    runs = {h: f'H_formal_{backbone}_{h}_seed{seed}' for h in ['B', 'AB']}
    base, _, _ = cache_h._context(config_path, runs['B'])
    assert base['formal_trajectory_target_used'] is False and base['target_development_used'] is False
    folders = {h: ROOT / 'runs' / run for h, run in runs.items()}
    tasks = {h: config['runs'][run] for h, run in runs.items()}
    maximum, learning_rate = tasks['B']['epochs'], tasks['B']['lr']
    completed = {h: json.loads((p / 'complete.json').read_text()) for h, p in folders.items()}
    history = {h: json.loads((p / 'history.json').read_text()) for h, p in folders.items()}
    terminal = ROOT / 'operations/jobs' / f'H_formal_{backbone}_pair_seed{seed}/exit.json'
    assert json.loads(terminal.read_text())['status'] == 'SUCCEEDED'
    state_path = Path(completed['B']['pair_state_path'])
    assert state_path.is_relative_to(ROOT / 'pairs')
    state_sha = t.digest(state_path)
    state = torch.load(state_path, map_location='cpu', weights_only=False)
    identities = {h: t.training_identity(config, task) for h, task in tasks.items()}
    assert state['identity'] == identities
    frame = {h: pd.read_parquet(ROOT / tasks[h]['train_manifest']) for h in tasks}
    validate_pair_frames(frame['B'], frame['AB'])
    steps = math.ceil(len(frame['B']) / 128)
    epochs = len(history['B'])
    assert len(history['AB']) == epochs and state['epoch'] == epochs
    selection = PairedSelection()
    hashes = {}
    successes = {h: 0 for h in tasks}
    skips = {h: 0 for h in tasks}
    consecutive = {h: 0 for h in tasks}
    for idx in range(epochs):
        epoch = idx + 1
        assert not selection.stopped
        b, ab = history['B'][idx], history['AB'][idx]
        selection.observe(b['dev']['pooled_auc'], ab['dev']['pooled_auc'], epoch)
        expected_draws, expected_methods = t.epoch_draws(frame['B'], seed, epoch, steps * 128)
        for h in tasks:
            row = history[h][idx]
            assert row['epoch'] == epoch and row['selection'] == selection.state_dict()
            assert row['paired_dev'] == b['paired_dev'] == ab['paired_dev']
            assert row['dev'] == row['paired_dev'][h]
            assert row['lr'] == t.epoch_lr(idx, maximum, learning_rate)
            assert row['attempts'] == steps and row['draw_count'] == steps * 128
            for suffix, expected in [('samples', expected_draws), ('methods', expected_methods)]:
                path = folders[h] / f'epoch_{epoch:02d}_{suffix}.npy'
                assert np.array_equal(np.load(path, allow_pickle=False), expected)
                hashes[str(path.relative_to(ROOT))] = t.digest(path)
            path = folders[h] / f'epoch_{epoch:02d}_updates.jsonl'
            events = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(events) == steps
            for offset, event in enumerate(events):
                assert event['attempt'] == idx * steps + offset and event['epoch'] == epoch
                assert event['lr'] == row['lr'] and math.isfinite(event['loss'])
                assert event['skip'] is (event['scale_after'] < event['scale_before'])
                assert event['success'] is (not event['skip'])
                if event['success']:
                    assert event['gradient_finite'] and math.isfinite(event['gradient_norm'])
                consecutive[h] = consecutive[h] + 1 if event['skip'] else 0
                assert consecutive[h] < 8
            assert sum(e['success'] for e in events) == row['success']
            assert sum(e['skip'] for e in events) == row['skip']
            assert math.isclose(sum(e['loss'] for e in events) / steps, row['train_loss'], abs_tol=1e-14)
            successes[h] += row['success']
            skips[h] += row['skip']
            hashes[str(path.relative_to(ROOT))] = t.digest(path)
    assert state['selection'] == selection.state_dict()
    assert state['attempts'] == steps * epochs
    assert state['consecutive_skips'] == consecutive
    assert selection.stopped or epochs == maximum
    tensors = {}
    for h, folder in folders.items():
        task, complete = tasks[h], completed[h]
        assert task['phase'] == 'formal' and task['lr'] == learning_rate == {'clip_vit_b16': 1e-5, 'dinov2_vit_b14': 5e-5, 'efficientnet_b0': 1e-4}[backbone] and task['epochs'] == maximum
        assert task['microbatch'] == 128 and task['history'] == h
        assert json.loads((folder / 'resolved_config.json').read_text()) == task
        assert complete['status'] == 'TRAINING_COMPLETE' and complete['identity'] == identities[h]
        assert complete['target_used'] is False and complete['target_development_used'] is False
        assert complete['best_epoch'] == selection.best_epoch and complete['early_stopped'] is selection.stopped
        assert complete['attempts'] == epochs * steps and complete['success'] == successes[h] and complete['skip'] == skips[h]
        assert complete['pair_state_sha256'] == state_sha and Path(complete['pair_state_path']) == state_path
        assert state['histories'][h] == history[h]
        path = folder / 'best_model.pth'
        assert t.digest(path) == complete['best_sha256']
        best = torch.load(path, map_location='cpu', weights_only=False)
        assert best['weights'] == 'ema' and best['ema_decay'] == .999
        assert best['identity'] == identities[h] and best['epoch'] == selection.best_epoch
        assert best['auc'] == selection.best_score
        assert set(best['model']) == set(state['best'][h])
        for key, tensor in best['model'].items():
            assert torch.isfinite(tensor).all() and torch.equal(tensor, state['best'][h][key]), (h, key)
        tensors[h] = len(best['model'])
        for name in ['complete.json', 'history.json', 'resolved_config.json', 'best_model.pth']:
            p = folder / name
            hashes[str(p.relative_to(ROOT))] = t.digest(p)
    assert t.digest(state_path) == state_sha and t.digest(config_path) == expected_config_sha
    output = dict(status='PASS_COMPLETED_SOURCE_PAIR_WEIGHTS_DRAWS_UPDATES',
        created=datetime.datetime.now().astimezone().isoformat(), seed=seed, backbone=backbone, maximum_epochs=maximum,
        actual_stop_epoch=epochs, selected_common_epoch=selection.best_epoch,
        selected_paired_DEV_B_auc=selection.best_score, early_stopped=selection.stopped,
        attempted_updates_per_history=epochs * steps, successful_updates=successes, amp_skips=skips,
        all_epochs_draws_and_method_arrays_replayed=True, complete_step_event_counts_replayed=True,
        all_best_EMA_tensors_match_atomic_pair_state=tensors, state_sha256=state_sha,
        source_only=True, target_evaluated=False, formal_budget_frozen=True,
        source_learning_adequacy_not_certified_by_integrity=True,
        producer_script_sha256=t.digest(__file__), config_sha256=expected_config_sha, artifacts_sha256=hashes)
    path = ROOT / f'operations/H_formal_{backbone}_seed{seed}_full_pair_acceptance.json'
    t.save_json(path, output)
    return {k: v for k, v in output.items() if k != 'artifacts_sha256'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True, choices=[7, 42, 123])
    p.add_argument('--backbone', required=True, choices=BACKBONES)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print(json.dumps(accept(args.seed, args.backbone), indent=2))
