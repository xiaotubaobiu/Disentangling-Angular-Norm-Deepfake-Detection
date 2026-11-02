"""CPU provenance, atomic resume, and FP32 boundary tests for new D caches."""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from reliability_v2 import trainer

if importlib.util.find_spec('reliability_v2.cache_d') is not None:
    from reliability_v2 import cache_d as cache
else:
    cache = None


class MeanEncoder(torch.nn.Module):
    def forward(self, x):
        return x.mean(dim=(2, 3)).mean(dim=1, keepdim=True).expand(-1, 512).contiguous()


class SmallDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MeanEncoder()
        self.classifier = torch.nn.Linear(512, 2)


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    assert cache is not None, 'protocol D cache module has not been implemented'
    root = tmp_path / 'artifacts'
    data = root / 'data/D'
    data.mkdir(parents=True)
    rows = []
    for i in range(4):
        image = tmp_path / f'{i}.png'
        Image.new('RGB', (7, 9), (20 + i * 40,) * 3).save(image)
        rows.append(dict(sample_id=f's{i}', image_path=str(image), label=i % 2,
                         video_uid=f'ff/method/{i}', method_id='real' if i % 2 == 0 else 'm',
                         rgb_sha256=trainer.digest(image), content_sha256=trainer.digest(image),
                         bytes=image.stat().st_size, mtime_ns=image.stat().st_mtime_ns, role='train'))
    frame = pd.DataFrame(rows)
    for role in ['train', 'dev', 'source_cal', 'target_full']:
        frame.assign(role=role).to_parquet(data / f'{role}.parquet', index=False)
    init = root / 'init.bin'
    init.write_bytes(b'explicit-local-pretraining-fixture')
    lock = root / 'configs/requirements.execution.lock.txt'
    lock.parent.mkdir()
    lock.write_text('cpu-test-environment\n')
    gate = root / 'qualification/gate.json'
    trainer.save_json(gate, dict(status='PASS', training_code_sha256=trainer.code_hash()))
    task = dict(backbone='clip_vit_b16', seed=7, epochs=2, phase='formal', recipe_frozen=True,
                train_manifest='data/D/train.parquet', dev_manifest='data/D/dev.parquet',
                microbatch=8, lr=1e-5)
    config = dict(artifact_root=str(root), workers=0, cache_workers=0, cache_batch_size=2,
                  cache_chunk_rows=4096, effective_batch=128, runs={'D_test': task},
                  backbones={'clip_vit_b16': dict(weight_path=str(init), weight_sha256=trainer.digest(init))})
    path = root / 'configs/test.json'
    trainer.save_json(path, config)
    identity = dict(config=hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest(),
                    train_manifest=trainer.digest(data / 'train.parquet'),
                    dev_manifest=trainer.digest(data / 'dev.parquet'), initialization=trainer.digest(init),
                    environment=trainer.digest(lock), code=trainer.code_hash(), qualification=trainer.digest(gate))
    run = root / 'runs/D_test'
    run.mkdir(parents=True)
    model = SmallDetector()
    with torch.no_grad():
        model.classifier.weight[0].fill_(1.)
        model.classifier.weight[1].fill_(1e10)
        model.classifier.bias.copy_(torch.tensor([0.25, -0.25]))
    torch.save(dict(model=model.state_dict(), identity=identity, epoch=2), run / 'final.pt')
    trainer.save_json(run / 'complete.json', dict(status='TRAINING_COMPLETE', identity=identity,
                      final_sha256=trainer.digest(run / 'final.pt'), epochs=2, selection='frozen_last', target_used=False))
    from reliability_v2 import models
    monkeypatch.setattr(models, 'build_model', lambda backbone, weight_path: SmallDetector())
    return dict(root=root, config=config, config_path=path, run=run, frame=frame, model=model)


def test_extract_preserves_original_fp32_head_and_features(experiment):
    e = experiment
    result = cache.extract(e['config_path'], 'D_test', device='cpu')
    assert result['status'] == 'CACHE_COMPLETE'
    assert set(result['roles']) == {'train', 'dev', 'source_cal', 'target_full'}
    role = e['root'] / 'features/D/D_test/source_cal'
    meta, frame = cache.verify_cache(role)
    df, h, w, b = cache.load_features(role)
    assert h.dtype == w.dtype == b.dtype == np.float32
    assert h.shape == (4, 512) and w.shape == (2, 512) and b.shape == (2,)
    assert np.array_equal(w, e['model'].classifier.weight.detach().numpy())
    assert w.astype(np.float64)[1, 0] - w.astype(np.float64)[0, 0] == 9999999999.
    assert frame.equals(df) and frame.role.eq('source_cal').all()
    assert meta['precision'] == 'FP32' and meta['epsilon'] == 1e-12
    assert trainer.digest(role / 'manifest.parquet') == trainer.digest(e['root'] / 'data/D/source_cal.parquet')
    assert cache.extract(e['config_path'], 'D_test', device='cpu') == result


@pytest.mark.parametrize('change', ['checkpoint', 'manifest', 'config', 'code', 'complete'])
def test_verify_rejects_changed_live_provenance(experiment, monkeypatch, change):
    e = experiment
    cache.extract(e['config_path'], 'D_test', device='cpu')
    role = e['root'] / 'features/D/D_test/train'
    if change == 'checkpoint':
        with (e['run'] / 'final.pt').open('ab') as stream:
            stream.write(b'changed')
    elif change == 'manifest':
        p = e['root'] / 'data/D/train.parquet'
        q = pd.read_parquet(p)
        q.loc[0, 'sample_id'] = 'changed'
        q.to_parquet(p, index=False)
    elif change == 'config':
        config = e['config']
        config['cache_batch_size'] = 3
        trainer.save_json(e['config_path'], config)
    elif change == 'code':
        monkeypatch.setattr(cache, '_cache_code_hash', lambda: 'changed')
    else:
        receipt = json.loads((e['run'] / 'complete.json').read_text())
        receipt['status'] = 'INCOMPLETE'
        trainer.save_json(e['run'] / 'complete.json', receipt)
    with pytest.raises(ValueError, match='identity|hash|complete|provenance'):
        cache.verify_cache(role)


@pytest.mark.parametrize('change', ['phase', 'frozen', 'epoch', 'identity'])
def test_extract_refuses_dev_unfrozen_or_stale_final(experiment, change):
    e = experiment
    if change in ['phase', 'frozen']:
        task = e['config']['runs']['D_test']
        task['phase' if change == 'phase' else 'recipe_frozen'] = 'development' if change == 'phase' else False
        trainer.save_json(e['config_path'], e['config'])
    else:
        state = torch.load(e['run'] / 'final.pt', weights_only=False)
        state['epoch' if change == 'epoch' else 'identity'] = 1 if change == 'epoch' else {'bad': 'identity'}
        torch.save(state, e['run'] / 'final.pt')
        receipt = json.loads((e['run'] / 'complete.json').read_text())
        receipt['final_sha256'] = trainer.digest(e['run'] / 'final.pt')
        trainer.save_json(e['run'] / 'complete.json', receipt)
    with pytest.raises(ValueError, match='formal|frozen|epoch|identity'):
        cache.extract(e['config_path'], 'D_test', device='cpu')
    # A lifetime lock may create its directory before checkpoint validation;
    # no image-derived output or completed role may be exported on rejection.
    assert not list((e['root'] / 'features').rglob('*.npy'))
    assert not list((e['root'] / 'features').rglob('complete.json'))


def test_corrupt_committed_features_are_rejected(experiment):
    e = experiment
    cache.extract(e['config_path'], 'D_test', device='cpu')
    role = e['root'] / 'features/D/D_test/train'
    chunk = role / 'chunk_000000.npy'
    x = np.load(chunk)
    x[0, 0] += 1.
    np.save(chunk, x)
    with pytest.raises(ValueError, match='hash'):
        cache.verify_cache(role)
    with pytest.raises(ValueError, match='hash'):
        cache.extract(e['config_path'], 'D_test', device='cpu')


@pytest.mark.parametrize('case', ['dtype', 'finite', 'dimension', 'rows'])
def test_feature_boundary_rejects_bad_arrays(case):
    assert cache is not None, 'protocol D cache module has not been implemented'
    x = np.ones((3, 512), dtype=np.float32)
    if case == 'dtype': x = x.astype(np.float64)
    if case == 'finite': x[0, 0] = np.nan
    if case == 'dimension': x = x[:, :-1]
    rows = 4 if case == 'rows' else 3
    with pytest.raises(ValueError, match='feature'):
        cache._validate_features(x, rows, 512)


def test_partial_chunk_resume_reuses_only_valid_commits(experiment):
    e = experiment
    manifest = e['root'] / 'data/D/source_cal.parquet'
    frame = pd.concat([e['frame']] * 1025, ignore_index=True).iloc[:4098].copy()
    frame['sample_id'] = [f's{i}' for i in range(len(frame))]
    frame['role'] = 'source_cal'
    frame.to_parquet(manifest, index=False)
    base, _, _ = cache._context(e['config_path'], 'D_test')
    identity = cache._role_identity(base, 'source_cal', manifest)
    h = np.arange(len(frame) * 512, dtype=np.float32).reshape(len(frame), 512)
    h[0] = 0.
    w = e['model'].classifier.weight.detach().numpy()
    b = e['model'].classifier.bias.detach().numpy()
    role = e['root'] / 'features/D/D_test/source_cal'
    starts = []
    def broken(start):
        starts.append(start)
        yield h[start:4096]
        raise RuntimeError('simulated interruption before final chunk commit')
    with pytest.raises(RuntimeError, match='simulated'):
        cache._cache_role(role, manifest, identity, w, b, broken, 4096)
    assert starts == [0]
    assert (role / 'chunk_000000.json').exists()
    assert not (role / 'complete.json').exists()
    with pytest.raises(ValueError, match='incomplete'):
        cache.verify_cache(role)
    (role / 'chunk_000001.npy').write_bytes(b'incomplete uncommitted file')
    def resumed(start):
        starts.append(start)
        yield h[start:]
    meta = cache._cache_role(role, manifest, identity, w, b, resumed, 4096)
    assert starts == [0, 4096]
    assert meta['rows'] == 4098 and meta['zero_norm_count'] == 1
    _, actual, _, _ = cache.load_features(role)
    assert np.array_equal(actual, h)
    assert len(meta['chunks']) == 2 and meta['chunks'][1]['rows'] == 2


def test_tampered_chunk_row_alignment_is_rejected(experiment):
    e = experiment
    cache.extract(e['config_path'], 'D_test', device='cpu')
    role = e['root'] / 'features/D/D_test/train'
    marker = role / 'chunk_000000.json'
    q = json.loads(marker.read_text())
    q['row_sha256'] = 'changed'
    trainer.save_json(marker, q)
    with pytest.raises(ValueError, match='chunk|row'):
        cache.verify_cache(role)


def test_zero_features_are_allowed_and_counted():
    assert cache is not None, 'protocol D cache module has not been implemented'
    assert cache._validate_features(np.zeros((2, 512), np.float32), 2, 512) == 2


@pytest.mark.parametrize('operation', ['extract', 'verify'])
def test_changed_source_image_is_rejected(experiment, operation):
    e = experiment
    if operation == 'verify':
        cache.extract(e['config_path'], 'D_test', device='cpu')
    Image.new('RGB', (11, 13), (240, 10, 20)).save(e['frame'].iloc[0].image_path)
    with pytest.raises(ValueError, match='input.*hash|input.*identity'):
        if operation == 'verify':
            cache.verify_cache(e['root'] / 'features/D/D_test/train')
        else:
            cache.extract(e['config_path'], 'D_test', device='cpu')


def test_run_specific_qualification_gate_is_immutable_across_routes(experiment):
    e = experiment
    root = e['root']
    d_gate = root / 'qualification/D_gate.json'
    d_gate.write_bytes((root / 'qualification/gate.json').read_bytes())
    task = e['config']['runs']['D_test']
    task['qualification_gate'] = 'qualification/D_gate.json'
    trainer.save_json(e['config_path'], e['config'])
    state = torch.load(e['run'] / 'final.pt', weights_only=False)
    state['identity']['config'] = hashlib.sha256(json.dumps(task, sort_keys=True).encode()).hexdigest()
    state['identity']['qualification'] = trainer.digest(d_gate)
    torch.save(state, e['run'] / 'final.pt')
    receipt = json.loads((e['run'] / 'complete.json').read_text())
    receipt['identity'] = state['identity']
    receipt['final_sha256'] = trainer.digest(e['run'] / 'final.pt')
    trainer.save_json(e['run'] / 'complete.json', receipt)
    trainer.save_json(root / 'qualification/gate.json', dict(status='H_PENDING', training_code_sha256='changed'))
    cache.extract(e['config_path'], 'D_test', device='cpu')
    cache.verify_cache(root / 'features/D/D_test/train')


def test_concurrent_extract_cannot_bypass_run_lifetime_lock(experiment):
    e = experiment
    lock = e['root'] / 'features/D/D_test/extract.lock'
    with trainer.exclusive_file(lock):
        with pytest.raises(BlockingIOError):
            cache.extract(e['config_path'], 'D_test', device='cpu')
    result = cache.extract(e['config_path'], 'D_test', device='cpu')
    assert result['status'] == 'CACHE_COMPLETE'
    with trainer.exclusive_file(lock):
        pass  # The entire completed extraction has released its ownership.


def test_extract_lock_remains_owned_during_encoder_forward(experiment, monkeypatch):
    e = experiment
    lock = e['root'] / 'features/D/D_test/extract.lock'
    class OwnedEncoder(MeanEncoder):
        def forward(self, x):
            with pytest.raises(BlockingIOError):
                with trainer.exclusive_file(lock):
                    pass
            return super().forward(x)
    def build(backbone, weight_path):
        model = SmallDetector()
        model.encoder = OwnedEncoder()
        return model
    from reliability_v2 import models
    monkeypatch.setattr(models, 'build_model', build)
    cache.extract(e['config_path'], 'D_test', device='cpu')
    with trainer.exclusive_file(lock):
        pass


def test_cuda_ownership_guard_rejects_before_any_model_initialization(experiment, monkeypatch):
    e = experiment
    e['config']['allowed_gpu_uuids'] = ['GPU-approved-unit-fixture']
    e['config']['allowed_gpus'] = [0, 1, 2, 3]
    trainer.save_json(e['config_path'], e['config'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-protected-unit-fixture')
    from reliability_v2 import models
    def forbidden_model(*args, **kwargs):
        raise AssertionError('model initialization reached before GPU ownership guard')
    monkeypatch.setattr(models, 'build_model', forbidden_model)
    with pytest.raises(ValueError, match='single approved GPU0-3 UUID required'):
        cache.extract(e['config_path'], 'D_test', device='cuda')
    with trainer.exclusive_file(e['root'] / 'features/D/D_test/extract.lock'):
        pass  # A failed GPU guard must release the per-run extraction lock.


def test_gpu_and_run_reservations_span_body_and_release_on_failure(experiment, monkeypatch):
    e = experiment
    uuid = 'GPU-approved-unit-fixture'
    e['config']['allowed_gpu_uuids'] = [uuid]
    e['config']['allowed_gpus'] = [0, 1, 2, 3]
    trainer.save_json(e['config_path'], e['config'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', uuid)
    queries = []
    def fake_query(args, **kwargs):
        queries.append(args)
        return '' if '--query-compute-apps=gpu_uuid,pid' in args else f'{uuid}, 0\n'
    monkeypatch.setattr(trainer.subprocess, 'check_output', fake_query)
    locks = [e['root'] / 'features/D/D_test/extract.lock', e['root'] / f'gpu_locks/{uuid}.lock']
    def body(config_path, run_id, device):
        for lock in locks:
            with pytest.raises(BlockingIOError):
                with trainer.exclusive_file(lock):
                    pass
        raise RuntimeError('simulated inference failure inside both lifetime reservations')
    monkeypatch.setattr(cache, '_extract', body, raising=False)
    from reliability_v2 import models
    def forbidden_model(*args, **kwargs):
        raise AssertionError('GPU model initialization is forbidden in CPU lifecycle tests')
    monkeypatch.setattr(models, 'build_model', forbidden_model)
    with pytest.raises(RuntimeError, match='both lifetime reservations'):
        cache.extract(e['config_path'], 'D_test', device='cuda')
    assert len(queries) == 2
    for lock in locks:
        with trainer.exclusive_file(lock):
            pass

