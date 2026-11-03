"""Read-only post-training state/EMA acceptance for one registered D model."""
from pathlib import Path
import argparse
import json
import torch
from journal_detector import cache_d, trainer

ROOT = Path('/home/z/workspace/deepfake_journal_20261102')


def accept(run):
    base, config, task = cache_d._context(ROOT / 'configs/execution.json', run)
    folder = ROOT / 'runs' / run
    paths = {k: folder / v for k, v in dict(complete='complete.json', history='history.json',
        state='state.pt', best='best_model.pth', resolved='resolved_config.json').items()}
    hashes = {k: trainer.digest(v) for k, v in paths.items()}
    receipt = json.loads(paths['complete'].read_text())
    history = json.loads(paths['history'].read_text())
    assert json.loads(paths['resolved'].read_text()) == task
    assert task['route'] == 'D' and task['phase'] == 'formal'
    assert receipt['target_used'] is False
    assert len(history) == receipt['epochs']
    assert [r['epoch'] for r in history] == list(range(1, len(history) + 1))
    selection = trainer.Selection()
    for index, row in enumerate(history):
        assert not selection.stopped
        assert 'cdf_dev' not in row and 'development_stop' not in row
        selection.observe(float(row['dev']['pooled_auc']), row['epoch'])
        assert row['selection'] == selection.state_dict()
        assert row['attempts'] == row['success'] + row['skip']
        assert row['draw_count'] == row['attempts'] * 128
        assert row['lr'] == trainer.epoch_lr(index, task['epochs'], task['lr'])
    assert receipt['best_epoch'] == selection.best_epoch
    assert receipt['early_stopped'] is selection.stopped
    assert selection.stopped or len(history) == task['epochs']
    for k in ['attempts', 'success', 'skip']:
        assert receipt[k] == sum(r[k] for r in history)
    state = torch.load(paths['state'], map_location='cpu', weights_only=False)
    best = torch.load(paths['best'], map_location='cpu', weights_only=False)
    assert state['epoch'] == len(history)
    assert state['history'] == history and state['selection'] == selection.state_dict()
    assert state['best_epoch'] == selection.best_epoch and state['best_score'] == selection.best_score
    assert state['identity'] == receipt['identity'] == best['identity'] == base['training_identity']
    assert best['weights'] == 'ema' and best['epoch'] == selection.best_epoch
    assert best['auc'] == selection.best_score and best['ema_decay'] == .999
    assert set(state['best_model']) == set(best['model'])
    for name, value in best['model'].items():
        assert torch.equal(value, state['best_model'][name]), name
        assert torch.isfinite(value).all(), name
    cache_d._validate_head(best['model']['classifier.weight'].numpy(),
        best['model']['classifier.bias'].numpy(), base['dimension'])
    assert all(trainer.digest(v) == hashes[k] for k, v in paths.items())
    result = dict(status='PASS', run_id=run, scope='This completed formal D model only',
        complete_sha256=hashes['complete'], state_sha256=hashes['state'], best_sha256=hashes['best'],
        source_context=base, actual_epochs=len(history), max_epochs=task['epochs'],
        best_epoch=selection.best_epoch, best_source_auc=selection.best_score,
        consecutive_source_nonimprovements=selection.wait, success=receipt['success'], skip=receipt['skip'],
        early_stopped=selection.stopped, formal_trajectory_target_used=False,
        source_selected_EMA_matches_atomic_state_best=True)
    output = ROOT / 'operations' / (run + '_training_acceptance.json')
    trainer.save_json(output, result)
    return {k: v for k, v in result.items() if k != 'source_context'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(accept(args.run), indent=2))
