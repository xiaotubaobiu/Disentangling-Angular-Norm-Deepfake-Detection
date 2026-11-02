"""H cache tests use synthetic PNGs and sealed CPU-only formal evidence."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from reliability_v2 import data_h, qualification_h, queue_h as q, trainer, worker_h as worker

cache = None
if importlib.util.find_spec('reliability_v2.cache_h'):
    from reliability_v2 import cache_h as cache

ROLES = ('B/SOURCE_CAL', 'B/SOURCE_TEST', 'B/TARGET_TEST',
         'common/B/SOURCE_TEST', 'common/B/TARGET_TEST',
         'common/AB/SOURCE_TEST', 'common/AB/TARGET_TEST')


class MeanEncoder(torch.nn.Module):
    def forward(self, x):
        assert not self.training and x.dtype == torch.float32
        assert not torch.is_autocast_enabled(x.device.type)
        return x.mean((1, 2, 3))[:, None].expand(-1, 512).contiguous()


class ToyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MeanEncoder()
        self.classifier = torch.nn.Linear(512, 2)


def save(path, obj):
    trainer.save_json(path, obj)


def rgb_hash(path):
    with Image.open(path) as image:
        x = np.asarray(image.convert('RGB'), dtype=np.uint8)
    return hashlib.sha256(str(x.shape).encode() + b'\0' + x.tobytes()).hexdigest()


def test_cache_module_is_available():
    assert cache is not None, 'H FP32 cache module is not implemented'


@pytest.fixture
def experiment(tmp_path, monkeypatch, request):
    assert cache is not None, 'H FP32 cache module is not implemented'
    root = tmp_path / 'synthetic-H'
    (root / 'configs').mkdir(parents=True)
    (root / 'qualification').mkdir()
    init = root / 'init.pt'
    init.write_bytes(b'synthetic-pretraining')
    (root / 'configs/requirements.execution.lock.txt').write_text('synthetic CPU environment\n')
    tables = {}
    keys = [f'{h}/{r}' for h in q.HISTORIES for r in ('train', 'dev')]
    keys += [f'{p}/{r}' for p in ('B', 'common/B', 'common/AB') for r in ('SOURCE_CAL', 'SOURCE_TEST', 'TARGET_TEST')]
    for key in keys:
        role = key.split('/')[-1].upper()
        version = 'AB' if '/AB/' in '/' + key + '/' else 'B'
        methods = ['real'] if role == 'SOURCE_CAL' else ['real', 'DeepFakeDetection'] if role == 'TARGET_TEST' else list(q.METHODS)
        rows = []
        for method in methods:
            uid = f'{role}/{method}/video'
            for slot in range(8):
                image = root / 'images' / role / version / method / f'{slot}.png'
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (7, 9), (20 + slot * 20, 30 + slot, 40)).save(image)
                mother = f'{uid}/{slot}'
                rows.append(dict(image_path=str(image), label=int(method != 'real'), method_id=method,
                    video_uid=uid, group_id=uid, domain='target' if role == 'TARGET_TEST' else 'source', role=role,
                    sample_id=mother, mother_id=mother, mother_rgb_sha256=hashlib.sha256(mother.encode()).hexdigest(),
                    mother_frame_index=slot, mother_timestamp_seconds=float(slot), processing_version=version,
                    content_sha256=trainer.digest(image), rgb_sha256=rgb_hash(image)))
        p = root / f'data/H/{key}.parquet'
        p.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        if role == 'SOURCE_CAL' and request.node.name == 'test_atomic_resume_preserves_committed_chunks':
            parts = []
            for i in range(513):
                part = frame.copy()
                part['video_uid'] = f'video-{i}'
                part['group_id'] = f'g-{i}'
                part['mother_id'] = [f'{i}/{j}' for j in range(8)]
                part['sample_id'] = part.mother_id
                parts.append(part)
            frame = pd.concat(parts, ignore_index=True)
        frame.to_parquet(p, index=False)
        tables[key] = dict(path=str(p), sha256=trainer.digest(p), rows=len(frame), videos=frame.video_uid.nunique())
    freeze = dict(status='FROZEN', tables=tables, training_draw_count=128)
    save(root / 'data/H/freeze.json', freeze)
    gate = dict(status='PASS', h_code_bundle_sha256=worker.h_code_hash(),
                backbones={b: dict(microbatch=32, fp32_microbatch=16) for b in q.BACKBONES})
    save(root / 'qualification/H_gate.json', gate)
    # Substitute only expensive production data/qualification acceptance.
    # Task validation, completion seals and the five-pair ledger remain real.
    monkeypatch.setattr(data_h, 'validate_h_freeze', lambda root: freeze)
    monkeypatch.setattr(qualification_h, 'validate_qualification', lambda root: gate)
    base = dict(artifact_root=str(root), allowed_gpus=[0, 1, 2, 3], allowed_gpu_uuids=['u0', 'u1', 'u2', 'u3'],
        workers=4, effective_batch=128, backbones={b: dict(weight_path=str(init), weight_sha256=trainer.digest(init)) for b in q.BACKBONES})
    save(root / 'configs/base.json', base)
    c = dict(base, schema='H_EXECUTION_V2', development_seeds=[1007, 1042], formal_seeds=list(q.FORMAL_SEEDS),
        H_training_draw_count=128, H_training_rows=40, H_data_freeze_sha256=trainer.digest(root / 'data/H/freeze.json'),
        H_qualification_sha256=trainer.digest(root / 'qualification/H_gate.json'), h_code_bundle_sha256=worker.h_code_hash(),
        H_microbatches={b: {'AMP': 32, 'FP32': 16} for b in q.BACKBONES}, cache_batch_size=16, cache_workers=0, runs={})
    recipes = {b: dict(lr=1e-5, epochs=30) for b in q.BACKBONES}
    recipe = root / 'configs/H_frozen_recipes.json'
    save(recipe, dict(target_scores_used=False, recipes=recipes))
    tasks = q.formal_tasks(c, recipes, recipe_sha256=trainer.digest(recipe))
    formal_ids = list(tasks)
    precision = 'FP32' if request.node.name == 'test_uniform_FP32_fallback_exports_only_effective_models' else 'AMP'
    if precision == 'FP32':
        tasks.update(q.formal_tasks(c, recipes, recipe_sha256=trainer.digest(recipe), precision='FP32', backbone='clip_vit_b16'))
    c['runs'].update(tasks)
    p = root / 'configs/H_execution.json'
    save(p, c)
    model = ToyDetector()
    with torch.no_grad():
        model.classifier.weight[0].fill_(1.)
        model.classifier.weight[1].fill_(1e10)
        model.classifier.bias.copy_(torch.tensor([.25, -.25]))
    for run, task in tasks.items():
        if task['backbone'] != 'clip_vit_b16':
            continue
        _, _, identity = worker.validate_task(p, run)
        out = root / 'runs' / run
        out.mkdir(parents=True)
        history = []
        for epoch in range(1, 31):
            sample = np.arange(128, dtype=np.int64) % 40
            aug = np.c_[np.zeros(128), np.ones((128, 2))]
            method = np.array(['real'] * 64 + ['Deepfakes'] * 16 + ['Face2Face'] * 16 + ['FaceSwap'] * 16 + ['NeuralTextures'] * 16)
            for suffix, x in [('samples', sample), ('augmentations', aug), ('methods', method)]:
                np.save(out / f'epoch_{epoch:02d}_{suffix}.npy', x, allow_pickle=False)
            (out / f'epoch_{epoch:02d}_updates.jsonl').write_text(json.dumps(
                dict(attempt=epoch - 1, success=True, skip=False, loss=.1, gradient_finite=True)) + '\n')
            history.append(dict(epoch=epoch, attempts=1, draw_count=128, success=1, skip=0,
                sample_sha256=trainer.digest(out / f'epoch_{epoch:02d}_samples.npy'),
                augmentation_sha256=trainer.digest(out / f'epoch_{epoch:02d}_augmentations.npy'),
                method_counts={str(k): int(v) for k, v in zip(*np.unique(method, return_counts=True))}))
        torch.save(dict(model=model.state_dict(), identity=identity, epoch=30), out / 'final.pt')
        (out / 'source_best.pt').write_bytes(b'synthetic-best')
        save(out / 'history.json', history)
        save(out / 'complete.json', dict(status='TRAINING_COMPLETE', identity=identity, epochs=30, attempts=30,
            success=30, skip=0, selection='frozen_last', target_used=False,
            final_sha256=trainer.digest(out / 'final.pt'), source_best_sha256=trainer.digest(out / 'source_best.pt')))
        save(out / 'identity.json', identity)
        save(out / 'resolved_config.json', c)
        save(out / 'H_training_start.json', worker._start_record(p, run, c, identity))
        torch.save(dict(identity=identity, epoch=30, attempts=30, history=history), out / 'state.pt')
        for name in worker._completion_artifacts(30):
            if not (out / name).exists():
                (out / name).write_bytes(b'synthetic-source-dev')
        worker.finalize_completion_seal(p, run)
    ledger = q.audit_paired_ledgers(p, 'clip_vit_b16', precision=precision)
    lp = root / f'reports/H_clip_vit_b16_{precision}_paired_ledger.json'
    save(lp, ledger)
    effective = {}
    for b in q.BACKBONES:
        effective_precision = precision if b == 'clip_vit_b16' else 'AMP'
        effective[b] = dict(precision=effective_precision, run_ids=[
            q.formal_id(b, h, seed, effective_precision) for seed in q.FORMAL_SEEDS for h in q.HISTORIES])
    export = dict(status='PASS', backbone='clip_vit_b16', precision=precision, all_five_pairs_checked=True,
        effective_runs=effective['clip_vit_b16']['run_ids'], paired_ledger_path=str(lp), paired_ledger_sha256=trainer.digest(lp),
        H_data_freeze_sha256=c['H_data_freeze_sha256'], h_code_bundle_sha256=c['h_code_bundle_sha256'], target_performance_used=False)
    save(root / 'qualification/H_target_export_clip_vit_b16.json', export)
    registry = dict(schema='H_RUN_REGISTRY_V2', formal_model_count=30, formal_run_ids=formal_ids, effective_runs=effective,
        target_export_allowed=True, status='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING')
    save(root / 'H_run_registry.json', registry)
    from reliability_v2 import models
    monkeypatch.setattr(models, 'build_model', lambda *args: ToyDetector())
    return dict(root=root, config=c, config_path=p, run=q.formal_id('clip_vit_b16', 'B', 7, precision),
                model=model, freeze=freeze, gate=gate, export=export, registry=registry, ledger=ledger, lp=lp)


def role_root(e, role='B/SOURCE_CAL'):
    return e['root'] / 'features/H' / e['run'] / role


def test_full_fp32_export_and_exact_reuse_without_encoder(experiment, monkeypatch):
    e = experiment
    result = cache.extract(e['config_path'], e['run'], device='cpu')
    assert cache.ROLES == ROLES and tuple(result['roles']) == ROLES
    frame, h, w, b = cache.load_features(role_root(e), expected_run_id=e['run'], expected_role='B/SOURCE_CAL', config_path=e['config_path'])
    assert h.shape == (8, 512) and h.dtype == w.dtype == b.dtype == np.float32
    np.testing.assert_array_equal(w, e['model'].classifier.weight.detach().numpy())
    assert w.astype(np.float64)[1, 0] - w.astype(np.float64)[0, 0] == 9999999999.
    expected = np.array([trainer.image_tensor(row.image_path, 'clip_vit_b16').mean().item() for row in frame.itertuples()], dtype=np.float32)
    np.testing.assert_allclose(h[:, 0], expected, atol=1e-6, rtol=1e-6)
    from reliability_v2 import models
    monkeypatch.setattr(models, 'build_model', lambda *args: pytest.fail('completed cache initialized encoder'))
    assert cache.extract(e['config_path'], e['run'], device='cpu') == result


@pytest.mark.parametrize('mutation', ['missing_gate', 'unchecked_pairs', 'registry_pending', 'registry_disallowed', 'registry_runs', 'superseded_amp', 'stale_ledger', 'four_pairs', 'one_pair', 'paired_stream'])
def test_no_export_before_five_pairs_and_effective_registry(experiment, mutation):
    e = experiment
    gp = e['root'] / 'qualification/H_target_export_clip_vit_b16.json'
    if mutation == 'missing_gate': gp.unlink()
    elif mutation == 'unchecked_pairs': e['export']['all_five_pairs_checked'] = False; save(gp, e['export'])
    elif mutation.startswith('registry_'):
        if mutation == 'registry_pending': e['registry']['status'] = 'TRAINING'
        elif mutation == 'registry_disallowed': e['registry']['target_export_allowed'] = False
        else: e['registry']['effective_runs']['clip_vit_b16']['run_ids'].pop()
        save(e['root'] / 'H_run_registry.json', e['registry'])
    elif mutation == 'superseded_amp':
        e['export'].update(precision='FP32', effective_runs=[q.formal_id('clip_vit_b16', h, s, 'FP32') for s in q.FORMAL_SEEDS for h in q.HISTORIES]); save(gp, e['export'])
    elif mutation in ('four_pairs', 'one_pair'):
        e['ledger']['pairs'] = e['ledger']['pairs'][:4 if mutation == 'four_pairs' else 1]
        save(e['lp'], e['ledger']); e['export']['paired_ledger_sha256'] = trainer.digest(e['lp']); save(gp, e['export'])
    elif mutation == 'paired_stream':
        out = e['root'] / 'runs' / q.formal_id('clip_vit_b16', 'AB', 3407)
        np.save(out / 'epoch_30_samples.npy', np.ones(128, np.int64))
    else: e['lp'].write_text('{}')
    with pytest.raises((ValueError, KeyError), match='gate|pairs|registry|effective|ledger|hash|identity|precision|artifact'):
        cache.extract(e['config_path'], e['run'], device='cpu')
    assert not list((e['root'] / 'features').rglob('*.npy'))


@pytest.mark.parametrize('mutation', ['config', 'freeze', 'gate', 'table', 'checkpoint', 'head', 'cache_code', 'helper_code', 'row_order'])
def test_live_identity_is_revalidated(experiment, monkeypatch, mutation):
    e = experiment
    cache.extract(e['config_path'], e['run'], device='cpu')
    if mutation == 'config': e['config']['cache_batch_size'] = 8; save(e['config_path'], e['config'])
    elif mutation == 'freeze': (e['root'] / 'data/H/freeze.json').write_text('{}')
    elif mutation == 'gate': (e['root'] / 'qualification/H_gate.json').write_text('{}')
    elif mutation in ('table', 'row_order'):
        p = e['root'] / 'data/H/B/SOURCE_CAL.parquet'; f = pd.read_parquet(p)
        if mutation == 'table': f.loc[0, 'mother_id'] = 'changed'
        else: f = f.iloc[::-1]
        f.to_parquet(p, index=False)
    elif mutation == 'checkpoint': (e['root'] / 'runs' / e['run'] / 'final.pt').write_bytes(b'changed')
    elif mutation == 'head':
        p = role_root(e) / 'head_weight.npy'; x = np.load(p); x[0, 0] = 7; np.save(p, x)
    elif mutation == 'cache_code': monkeypatch.setattr(cache, '_cache_code_hash', lambda: 'changed')
    else: monkeypatch.setattr(cache, '_helper_code_hash', lambda: 'changed')
    with pytest.raises(ValueError, match='hash|identity|freeze|gate|changed'):
        cache.verify_cache(role_root(e))


@pytest.mark.parametrize('mutation', ['wrong_run', 'wrong_role', 'wrong_config', 'noncanonical'])
def test_requested_identity_and_canonical_location(experiment, mutation):
    e = experiment; cache.extract(e['config_path'], e['run'], device='cpu')
    kw = dict(expected_run_id=e['run'], expected_role='B/SOURCE_CAL', config_path=e['config_path'])
    path = role_root(e)
    if mutation == 'wrong_run': kw['expected_run_id'] = q.formal_id('clip_vit_b16', 'AB', 7)
    elif mutation == 'wrong_role': kw['expected_role'] = 'B/SOURCE_TEST'
    elif mutation == 'wrong_config': kw['config_path'] = e['root'] / 'configs/base.json'
    else:
        import shutil
        moved = e['root'] / 'copied-cache'; shutil.copytree(path, moved); path = moved
    with pytest.raises(ValueError, match='requested|canonical|identity'):
        cache.load_features(path, **kw)


@pytest.mark.parametrize('mutation', ['bytes', 'rgb', 'missing_hash', 'missing_file', 'seven_frames', 'duplicate_rgb', 'target_role', 'cal_fake'])
def test_manifest_and_image_integrity_fail_closed(experiment, mutation):
    e = experiment; p = e['root'] / 'data/H/B/SOURCE_CAL.parquet'; f = pd.read_parquet(p)
    if mutation == 'bytes': Image.new('RGB', (7, 9), (250, 0, 0)).save(f.iloc[0].image_path)
    elif mutation == 'missing_file': Path(f.iloc[0].image_path).unlink()
    else:
        if mutation == 'rgb': f.loc[0, 'rgb_sha256'] = '0' * 64
        elif mutation == 'missing_hash': f = f.drop(columns='content_sha256')
        elif mutation == 'seven_frames': f = f.iloc[:-1]
        elif mutation == 'duplicate_rgb': f.loc[1, 'rgb_sha256'] = f.loc[0, 'rgb_sha256']
        elif mutation == 'target_role': f['domain'] = 'target'
        else: f['label'] = 1; f['method_id'] = 'Deepfakes'
        f.to_parquet(p, index=False)
        e['freeze']['tables']['B/SOURCE_CAL'].update(sha256=trainer.digest(p), rows=len(f))
    with pytest.raises(ValueError, match='image|RGB|hash|eight|source|CAL|columns|label|role|coverage'):
        cache._read_manifest(p, 'B/SOURCE_CAL')


def test_unchanged_stat_cannot_hide_modified_bytes(experiment):
    import os
    e = experiment; cache.extract(e['config_path'], e['run'], device='cpu')
    f = pd.read_parquet(e['root'] / 'data/H/B/SOURCE_CAL.parquet'); p = Path(f.iloc[0].image_path); before = p.stat()
    raw = bytearray(p.read_bytes()); raw[-1] ^= 1; p.write_bytes(raw); os.utime(p, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match='image.*hash|byte.*hash'):
        cache.verify_cache(role_root(e))


def test_atomic_resume_preserves_committed_chunks(experiment):
    e = experiment
    p = e['root'] / 'data/H/B/SOURCE_CAL.parquet'
    f = pd.read_parquet(p)
    base, _, _ = cache._context(e['config_path'], e['run'])
    identity = cache._role_identity(base, 'B/SOURCE_CAL', p)
    h = np.arange(len(f) * 512, dtype=np.float32).reshape(len(f), 512); h[0] = 0
    w = e['model'].classifier.weight.detach().numpy(); b = e['model'].classifier.bias.detach().numpy()
    starts = []
    def broken(start):
        starts.append(start); yield h[start:4096]; raise RuntimeError('interrupted fixture')
    with pytest.raises(RuntimeError, match='interrupted'):
        cache._cache_role(role_root(e), p, identity, w, b, broken)
    assert (role_root(e) / 'chunk_000000.json').exists()
    (role_root(e) / 'chunk_000001.npy').write_bytes(b'uncommitted')
    def resumed(start): starts.append(start); yield h[start:]
    cache._cache_role(role_root(e), p, identity, w, b, resumed)
    assert starts == [0, 4096]
    _, actual, _, _ = cache.load_features(role_root(e)); np.testing.assert_array_equal(actual, h)
    meta, _ = cache.verify_cache(role_root(e)); assert meta['zero_norm_count'] == 1


@pytest.mark.parametrize('mutation', ['chunk_bytes', 'chunk_rows', 'head_metadata', 'missing_complete'])
def test_committed_corruption_is_never_repaired(experiment, mutation):
    e = experiment; cache.extract(e['config_path'], e['run'], device='cpu'); path = role_root(e)
    if mutation == 'chunk_bytes': (path / 'chunk_000000.npy').write_bytes(b'corrupt')
    elif mutation == 'missing_complete': (path / 'complete.json').unlink(); (path / 'chunk_000000.npy').write_bytes(b'corrupt')
    elif mutation == 'chunk_rows':
        marker = json.loads((path / 'chunk_000000.json').read_text()); marker['row_sha256'] = 'bad'; save(path / 'chunk_000000.json', marker)
    else:
        meta = json.loads((path / 'complete.json').read_text()); meta['head']['weight_sha256'] = 'bad'; save(path / 'complete.json', meta)
    with pytest.raises(ValueError, match='hash|chunk|head'):
        cache.extract(e['config_path'], e['run'], device='cpu')


def test_duplicate_lock_spans_encoder(experiment, monkeypatch):
    e = experiment; lock = e['root'] / 'features/H' / e['run'] / 'extract.lock'
    with trainer.exclusive_file(lock):
        with pytest.raises(BlockingIOError): cache.extract(e['config_path'], e['run'], device='cpu')
    class Guard(MeanEncoder):
        def forward(self, x):
            with pytest.raises(BlockingIOError):
                with trainer.exclusive_file(lock): pass
            return super().forward(x)
    def build(*args):
        m = ToyDetector(); m.encoder = Guard(); return m
    from reliability_v2 import models
    monkeypatch.setattr(models, 'build_model', build)
    cache.extract(e['config_path'], e['run'], device='cpu')


def test_D_priority_guard_precedes_GPU_ownership(experiment, monkeypatch):
    e = experiment
    def boundary(config): raise ValueError('D retains priority')
    monkeypatch.setattr(worker, 'verify_resource_boundary', boundary)
    monkeypatch.setattr(trainer, 'owned_gpu', lambda *args: pytest.fail('ownership reached after failed boundary'))
    with pytest.raises(ValueError, match='D retains priority'):
        cache.extract(e['config_path'], e['run'], device='cuda')


def test_GPU_and_run_ownership_span_entire_body(experiment, monkeypatch):
    e = experiment; events = []; held = []
    monkeypatch.setattr(worker, 'verify_resource_boundary', lambda config: events.append('boundary'))
    @contextlib.contextmanager
    def owned(*args):
        events.append('owned'); held.append(True)
        try: yield
        finally: held.pop(); events.append('released')
    monkeypatch.setattr(trainer, 'owned_gpu', owned)
    def body(*args):
        assert held
        with pytest.raises(BlockingIOError):
            with trainer.exclusive_file(e['root'] / 'features/H' / e['run'] / 'extract.lock'): pass
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(cache, '_extract', body)
    with pytest.raises(RuntimeError, match='synthetic failure'): cache.extract(e['config_path'], e['run'])
    assert events == ['boundary', 'owned', 'boundary', 'released'] and not held


def test_initial_cache_estimate_keeps_50GiB_reserve(experiment, monkeypatch):
    import shutil
    e = experiment
    monkeypatch.setattr(shutil, 'disk_usage', lambda root: type('Usage', (), {'free': 50 * 1024**3})())
    with pytest.raises(ValueError, match='capacity|reserve'):
        cache.extract(e['config_path'], e['run'], device='cpu')
    assert not list((e['root'] / 'features').rglob('*.npy'))



def test_fresh_boundary_rechecked_immediately_before_encoder(experiment, monkeypatch):
    e = experiment
    calls = []
    def boundary(config):
        calls.append(True)
        if len(calls) == 3:
            raise ValueError('D priority changed before model allocation')
    monkeypatch.setattr(worker, 'verify_resource_boundary', boundary)
    monkeypatch.setattr(trainer, 'owned_gpu', lambda *args: contextlib.nullcontext())
    from reliability_v2 import models
    monkeypatch.setattr(models, 'build_model', lambda *args: pytest.fail('model allocated before fresh boundary'))
    with pytest.raises(ValueError, match='D priority changed'):
        cache.extract(e['config_path'], e['run'], device='cuda')
    assert len(calls) == 3


def test_uniform_FP32_fallback_exports_only_effective_models(experiment):
    e = experiment
    result = cache.extract(e['config_path'], e['run'], device='cpu')
    assert e['run'].endswith('_FP32') and result['identity']['effective_precision'] == 'FP32'
    cache.verify_cache(role_root(e), expected_run_id=e['run'], expected_role='B/SOURCE_CAL', config_path=e['config_path'])
    amp_run = q.formal_id('clip_vit_b16', 'B', 7)
    with pytest.raises(ValueError, match='effective|precision'):
        cache.extract(e['config_path'], amp_run, device='cpu')
    assert not list((e['root'] / 'features/H' / amp_run).rglob('*.npy'))


def test_resealed_cached_head_still_must_match_original_checkpoint(experiment):
    e = experiment
    cache.extract(e['config_path'], e['run'], device='cpu')
    path = role_root(e)
    head = path / 'head_weight.npy'
    x = np.load(head); x[0, 0] = 7.; np.save(head, x)
    meta = json.loads((path / 'complete.json').read_text())
    meta['head']['weight_sha256'] = trainer.digest(head)
    save(path / 'complete.json', meta)
    with pytest.raises(ValueError, match='original checkpoint head'):
        cache.verify_cache(path)


def test_RGB_PNG_mode_is_checked_even_when_bytes_and_RGB_hashes_match(experiment):
    e = experiment
    path = e['root'] / 'data/H/B/SOURCE_CAL.parquet'
    f = pd.read_parquet(path); image = Path(f.iloc[0].image_path)
    Image.new('L', (7, 9), 55).save(image)
    f.loc[0, 'content_sha256'] = trainer.digest(image)
    f.loc[0, 'rgb_sha256'] = rgb_hash(image)
    f.to_parquet(path, index=False)
    with pytest.raises(ValueError, match='RGB PNG'):
        cache._read_manifest(path, 'B/SOURCE_CAL')
