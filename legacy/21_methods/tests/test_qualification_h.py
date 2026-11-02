"""CPU-only H qualification contracts; never launch a real pretrained/GPU job."""
import copy
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from reliability_v2 import qualification_h as q
from reliability_v2 import trainer


METHODS = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']


def source_rows(tmp_path, role, version, real=8, fake=2):
    rows = []
    for method in ['real', *METHODS]:
        n = real if method == 'real' else fake
        for video in range(n):
            uid = f'FFPP/source/{method}/{role}_{video:02d}'
            for frame in range(8):
                key = f'{uid}/mother/{frame:03d}'
                path = tmp_path / f'{role}_{version}_{method}_{video}_{frame}.png'
                color = (frame * 17 + video, (31 if method == 'real' else 97), 9 if version == 'B' else 19)
                Image.new('RGB', (8, 8), color).save(path)
                mass = (.5 if method == 'real' else .125) / n / 8
                rgb = np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
                rgb_digest = hashlib.sha256(str(rgb.shape).encode() + b'\0' + rgb.tobytes()).hexdigest()
                rows.append(dict(image_path=str(path), label=int(method != 'real'),
                    method_id=method, video_id=f'{role}_{video:02d}', video_uid=uid,
                    group_id=f'{role}/{method}/{video}', domain='source', role=role,
                    rgb_sha256=rgb_digest, mother_id=key, sample_id=key,
                    mother_rgb_sha256=f'{key}_rgb', mother_frame_index=frame,
                    mother_timestamp_seconds=frame / 10, processing_version=version,
                    source_graph_certified=True, content_sha256=trainer.digest(path),
                    sampling_mass=mass, method_contributions=json.dumps({method: mass})))
    return pd.DataFrame(rows)


@pytest.fixture
def accepted_root(tmp_path, monkeypatch):
    data = tmp_path / 'data/H'
    data.mkdir(parents=True)
    tables = {}
    for role in ['TRAIN', 'DEV']:
        for version in ['B', 'AB']:
            frame = source_rows(tmp_path, role, version)
            path = data / f'{version}_{role}.parquet'
            frame.to_parquet(path, index=False)
            tables[f'{version}/{role.lower()}'] = dict(path=str(path), sha256=trainer.digest(path), rows=len(frame), videos=frame.video_uid.nunique())
    cal = source_rows(tmp_path, 'SOURCE_CAL', 'B', real=2, fake=1)
    cal = cal[cal.label.eq(0)].drop(columns=['sampling_mass', 'method_contributions'])
    path = data / 'B_SOURCE_CAL.parquet'
    cal.to_parquet(path, index=False)
    tables['B/SOURCE_CAL'] = dict(path=str(path), sha256=trainer.digest(path), rows=len(cal), videos=2)
    freeze = dict(status='FROZEN', tables=tables, training_draw_count=128)
    trainer.save_json(data / 'freeze.json', freeze)
    calls = []
    def validate(root, *, deep=False):
        calls.append(Path(root))
        value = json.loads((Path(root) / 'data/H/freeze.json').read_text())
        if value['status'] != 'FROZEN':
            raise ValueError('H freeze not accepted')
        for table in value['tables'].values():
            if trainer.digest(table['path']) != table['sha256']:
                raise ValueError('H table changed')
        return value
    monkeypatch.setitem(sys.modules, 'reliability_v2.data_h', types.SimpleNamespace(validate_h_freeze=validate))
    monkeypatch.setattr(q, '_h_code_hash', lambda: 'h_bundle_fixture')
    weight = tmp_path / 'initial.pt'
    weight.write_bytes(b'offline initialization fixture')
    base = dict(artifact_root=str(tmp_path), effective_batch=128, workers=4,
        allowed_gpus=[0, 1, 2, 3], allowed_gpu_uuids=[f'GPU-{i}' for i in range(4)],
        backbones={b: dict(weight_path=str(weight), weight_sha256=trainer.digest(weight), learning_rate=.001) for b in q.BACKBONES})
    trainer.save_json(tmp_path / 'configs/base.json', base)
    (tmp_path / 'configs/requirements.execution.lock.txt').write_text('fixture environment\n')
    trainer.save_json(tmp_path / 'qualification/gpu_gate.json', dict(status='PASS', training_code_sha256=trainer.code_hash(), backbones={b: dict(status='PASS') for b in q.BACKBONES}, target_used=False))
    return tmp_path, calls


def test_prepare_uses_accepted_tables_and_writes_only_probe_directory(accepted_root):
    root, calls = accepted_root
    before = {str(p.relative_to(root)): trainer.digest(p) for p in root.rglob('*') if p.is_file()}
    receipt = q.prepare_probe(root)
    assert calls and receipt['status'] == 'PROBES_READY'
    assert receipt['target_used'] is False
    for version in ['B', 'AB']:
        frame = pd.read_parquet(root / receipt['tables'][f'{version}/train']['path'])
        assert len(frame) == 128 and frame.label.value_counts().to_dict() == {0: 64, 1: 64}
        assert frame.groupby('video_uid').size().eq(8).all()
        assert frame.groupby('method_id').sampling_mass.sum().to_dict() == pytest.approx({'real': .5, **{m: .125 for m in METHODS}})
    b = pd.read_parquet(root / receipt['tables']['B/train']['path'])
    ab = pd.read_parquet(root / receipt['tables']['AB/train']['path'])
    assert b.mother_id.tolist() == ab.mother_id.tolist()
    assert trainer.epoch_draws(b, 1007, 1)[0].tobytes() == trainer.epoch_draws(ab, 1007, 1)[0].tobytes()
    bootstrap = json.loads((root / 'qualification/H/probes/bootstrap_gate.json').read_text())
    assert bootstrap['approved_routes'] == ['H_QUALIFICATION']
    assert not (root / 'qualification/H_gate.json').exists()
    after = {str(p.relative_to(root)): trainer.digest(p) for p in root.rglob('*') if p.is_file()}
    assert all(after[k] == v for k, v in before.items())
    assert all(k.startswith('qualification/H/probes/') for k in after.keys() - before.keys())
    assert q.prepare_probe(root) == receipt


def test_reject_unfrozen_before_any_probe_write(accepted_root):
    root, _ = accepted_root
    path = root / 'data/H/freeze.json'
    freeze = json.loads(path.read_text()); freeze['status'] = 'PENDING'
    trainer.save_json(path, freeze)
    with pytest.raises(ValueError, match='freeze'):
        q.prepare_probe(root)
    assert not (root / 'qualification/H/probes').exists()


@pytest.mark.parametrize('mutation', ['target', 'mother', 'method', 'short_video'])
def test_paired_probe_fails_closed(tmp_path, mutation):
    b = source_rows(tmp_path, 'TRAIN', 'B')
    ab = source_rows(tmp_path, 'TRAIN', 'AB')
    if mutation == 'target': b.loc[0, 'domain'] = 'target'
    elif mutation == 'mother': ab.loc[0, 'mother_id'] = 'wrong'
    elif mutation == 'method': b.loc[0, 'method_id'] = 'unregistered'
    else: b = b.iloc[:-1]
    with pytest.raises(ValueError):
        q._paired_source_probes(b, ab, 'TRAIN')


def test_probe_detects_changed_source_bytes(accepted_root):
    root, _ = accepted_root
    receipt = q.prepare_probe(root)
    frame = pd.read_parquet(root / receipt['tables']['B/train']['path'])
    Path(frame.iloc[0].image_path).write_bytes(b'changed')
    with pytest.raises(ValueError, match='byte|content'):
        q.prepare_probe(root)


def test_byte_comparator_rejects_signed_zero_and_dtype():
    with pytest.raises(AssertionError):
        q.assert_byte_equal(torch.tensor([0.]), torch.tensor([-0.]))
    with pytest.raises(AssertionError):
        q.assert_byte_equal(np.array([1], dtype='int64'), np.array([1], dtype='int32'))
    assert q.tree_hash({'x': np.array([1.])}) == q.tree_hash({'x': np.array([1.])})


def test_independent_dev_macro_and_real_only_threshold():
    dev = pd.DataFrame([dict(video_uid='r', method_id='real', label=0, score=0.), *[
        dict(video_uid=m, method_id=m, label=1, score=float(i - 1)) for i, m in enumerate(METHODS)]])
    cal = pd.DataFrame(dict(video_uid=['r1', 'r2'], method_id=['real'] * 2, label=[0, 0], score=[1., 1.]))
    result = q.recompute_source_metrics(dev, cal)
    assert result['dev']['method_auc'] == {'Deepfakes': 0., 'Face2Face': .5, 'FaceSwap': 1., 'NeuralTextures': 1.}
    assert result['dev']['macro_auc'] == .625
    assert result['source_cal']['threshold'] == np.nextafter(1., np.inf)
    assert result['source_cal']['false_positives'] == 0
    assert 'auc' not in result['source_cal']
    with pytest.raises(ValueError, match='real'):
        q.recompute_source_metrics(dev, cal.assign(label=[0, 1]))


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.project = torch.nn.Linear(3, 512); self.dropout = torch.nn.Dropout(.2)
    def forward(self, x):
        return self.dropout(self.project(x.mean(dim=(2, 3))))


def test_real_sampler_loader_fp32_resume_and_identity_rejection(tmp_path):
    from reliability_v2.models import Detector
    torch.set_num_threads(2)
    frame = source_rows(tmp_path, 'TRAIN', 'B')
    dev = source_rows(tmp_path, 'DEV', 'B', real=2, fake=1)
    cal = source_rows(tmp_path, 'SOURCE_CAL', 'B', real=2, fake=1)
    cal = cal[cal.label.eq(0)]
    identity = {'config': 'c', 'data': 'd', 'code': 's', 'initialization': 'w'}
    result = q._run_history(frame, dev, cal, 'clip_vit_b16', {'weight_path': 'unused', 'learning_rate': .001},
        identity, tmp_path / 'runtime', 32, 'FP32', device='cpu',
        model_factory=lambda b, p: Detector(ToyEncoder(), 512))
    assert result['status'] == 'PASS' and result['loader_workers'] == 4
    assert result['all_trainable_parameters_received_gradients']
    assert result['resume_parameter_optimizer_scaler_rng_byte_exact']
    assert result['resume_FP32_features_byte_exact']
    assert result['changed_identity_fields_rejected'] == ['config', 'data', 'code', 'initialization']
    assert result['features_dimension'] == 512
    assert result['continuous_attempts'] == result['resumed_attempts'] == 2


def test_accept_cannot_create_gate_without_real_qualification(accepted_root):
    root, _ = accepted_root
    q.prepare_probe(root)
    with pytest.raises(ValueError, match='qualification|receipt'):
        q.accept_qualification(root)
    assert not (root / 'qualification/H_gate.json').exists()


def test_worker_rejects_invalid_precision_before_gpu(accepted_root, monkeypatch):
    root, _ = accepted_root
    monkeypatch.setattr(trainer, 'owned_gpu', lambda *a: pytest.fail('GPU touched'))
    with pytest.raises(ValueError, match='precision'):
        q.worker(root, 'clip_vit_b16', precision='BF16')


def test_validator_requires_gate_and_current_bundle(accepted_root):
    root, _ = accepted_root
    with pytest.raises(ValueError, match='gate'):
        q.validate_qualification(root)


def test_gate_schema_binds_six_receipts_and_rejects_changed_bundle(accepted_root, monkeypatch):
    root, _ = accepted_root
    probe = q.prepare_probe(root)
    binding = probe['binding']
    config = json.loads((root / 'configs/base.json').read_text())
    verified = []
    # Test aggregation/binding independently from the real-loader resume test.
    def verify(r, out, identity, p):
        verified.append((identity['backbone'], identity['precision']))
        return json.loads((out / 'receipt.json').read_text())
    monkeypatch.setattr(q, '_verify_attempt', verify)
    monkeypatch.setattr(q, '_receipt_header', lambda r, out, identity: json.loads((out / 'receipt.json').read_text()))
    for backbone in q.BACKBONES:
        for precision in q.PRECISIONS:
            path = root / 'qualification/H' / backbone / precision / 'micro32/receipt.json'
            path.parent.mkdir(parents=True)
            (path.parent / 'checkpoint.pt').write_bytes(b'isolated aggregation fixture')
            (path.parent / 'metrics.json').write_text('{"value":1}')
            trainer.save_json(path, {'test_receipt': backbone + precision,
                'evidence_sha256': {name: trainer.digest(path.parent / name) for name in ['checkpoint.pt', 'metrics.json']}})
    gate = q.accept_qualification(root)
    assert gate['approved_routes'] == ['H']
    assert gate['H_data_freeze_sha256'] == trainer.digest(root / 'data/H/freeze.json')
    assert gate['qualification_code_sha256'] == q.qualification_code_hash()
    assert len(verified) == 6
    assert all(v['microbatch'] == v['fp32_microbatch'] == 32 for v in gate['backbones'].values())
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('cached validation deserialized a checkpoint'))
    validated = q.validate_qualification(root)
    assert {k: v for k, v in validated.items() if k != '_validation'} == gate
    assert validated['_validation']['mode'] == 'accepted_evidence_stat_freshness'
    assert len(gate['accepted_evidence_stats']) == 6
    metrics = root / 'qualification/H/clip_vit_b16/AMP/micro32/metrics.json'
    metrics_stat = metrics.stat()
    metrics.write_text('{"value":2}')
    import os
    os.utime(metrics, ns=(metrics_stat.st_atime_ns, metrics_stat.st_mtime_ns))
    with pytest.raises(ValueError, match='small qualification evidence byte hash'):
        q.validate_qualification(root)
    metrics.write_text('{"value":1}')
    os.utime(metrics, ns=(metrics_stat.st_atime_ns, metrics_stat.st_mtime_ns))
    checkpoint = root / 'qualification/H/clip_vit_b16/AMP/micro32/checkpoint.pt'
    original = checkpoint.stat()
    os.utime(checkpoint, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000))
    with pytest.raises(ValueError, match='stat fingerprint'):
        q.validate_qualification(root)
    os.utime(checkpoint, ns=(original.st_atime_ns, original.st_mtime_ns))
    monkeypatch.setattr(q, '_h_code_hash', lambda: 'changed_bundle')
    with pytest.raises(ValueError, match='stale'):
        q.validate_qualification(root)


def test_smaller_microbatch_requires_higher_oom_evidence(accepted_root, monkeypatch):
    root, _ = accepted_root
    probe = q.prepare_probe(root)
    _, config, binding = q._context(root)
    monkeypatch.setattr(q, '_verify_attempt', lambda *args: {})
    out = root / 'qualification/H/clip_vit_b16/AMP'
    trainer.save_json(out / 'micro16/receipt.json', {'test': 'smaller_pass'})
    with pytest.raises(ValueError, match='only verified OOM'):
        q._qualified_backbones(root, binding, probe, config)
    identity = q._attempt_identity(root, binding, probe, config, 'clip_vit_b16', 32, 'AMP')
    trainer.save_json(out / 'micro32/failure.json', dict(status='REVIEW_REQUIRED', identity=identity))
    with pytest.raises(ValueError, match='only verified OOM'):
        q._qualified_backbones(root, binding, probe, config)
    trainer.save_json(out / 'micro32/failure.json', dict(status='OOM_RETRY_ALLOWED', identity=identity))
    # It proceeds past AMP16 and stops at the independently missing FP32 proof.
    with pytest.raises(ValueError, match='FP32'):
        q._qualified_backbones(root, binding, probe, config)


def test_selected_pixel_digest_checked_even_with_matching_file_digest(accepted_root):
    root, _ = accepted_root
    freeze = json.loads((root / 'data/H/freeze.json').read_text())
    path = Path(freeze['tables']['B/train']['path'])
    frame = pd.read_parquet(path); frame.loc[0, 'rgb_sha256'] = 'wrong_pixel_digest'
    frame.to_parquet(path, index=False)
    freeze['tables']['B/train']['sha256'] = trainer.digest(path)
    trainer.save_json(root / 'data/H/freeze.json', freeze)
    with pytest.raises(ValueError, match='RGB'):
        q.prepare_probe(root)


def test_full_two_history_receipt_verification_on_cpu_and_tamper_detection(accepted_root):
    from reliability_v2.models import Detector
    root, _ = accepted_root
    torch.set_num_threads(2)
    probe = q.prepare_probe(root)
    _, config, binding = q._context(root)
    identity = q._attempt_identity(root, binding, probe, config, 'clip_vit_b16', 32, 'FP32')
    out = root / 'qualification/H/clip_vit_b16/FP32/micro32'
    histories = {}
    cal = pd.read_parquet(root / probe['tables']['B/SOURCE_CAL']['path'])
    for version in ['B', 'AB']:
        frame = pd.read_parquet(root / probe['tables'][f'{version}/train']['path'])
        dev = pd.read_parquet(root / probe['tables'][f'{version}/dev']['path'])
        histories[version] = q._run_history(frame, dev, cal, 'clip_vit_b16', config['backbones']['clip_vit_b16'],
            {**identity, 'history': version}, out / version, 32, 'FP32', device='cpu',
            model_factory=lambda b, p: Detector(ToyEncoder(), 512))
    # Synthetic device metadata is confined to pytest's temporary root; this
    # exercises validation and does not create/claim a production qualification.
    receipt = dict(status='PASS', identity=identity, backbone='clip_vit_b16', precision='FP32',
        microbatch=32, effective_batch=128, loader_workers=4, source_only=True, target_used=False,
        device_uuid='GPU-0', histories=histories,
        evidence_sha256={str(p.relative_to(out)): trainer.digest(p) for p in out.rglob('*') if p.is_file()})
    trainer.save_json(out / 'receipt.json', receipt)
    assert q._verify_attempt(root, out, identity, probe) == receipt
    assert not (root / 'qualification/H_gate.json').exists()
    (out / 'AB/metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='byte hash'):
        q._verify_attempt(root, out, identity, probe)


def test_reuse_rejects_rehashed_probe_not_in_accepted_source(accepted_root):
    root, _ = accepted_root
    receipt = q.prepare_probe(root)
    path = root / receipt['tables']['B/train']['path']
    frame = pd.read_parquet(path)
    frame.loc[0, 'group_id'] = 'invented_source_group'
    frame.to_parquet(path, index=False)
    receipt['tables']['B/train']['sha256'] = trainer.digest(path)
    trainer.save_json(root / 'qualification/H/probes/receipt.json', receipt)
    with pytest.raises(AssertionError):
        q.prepare_probe(root)


def test_geometry_frame_row_interface_uses_source_domain(tmp_path):
    from reliability_v2.geometry_h import h_frame_rows
    fixture = source_rows(tmp_path, 'TRAIN', 'B')
    fixture = fixture[fixture.video_uid.eq(fixture.iloc[0].video_uid)]
    first = fixture.iloc[0]
    result = dict(video=dict(video_uid=first.video_uid, video_id=first.video_id,
        label=int(first.label), method=first.method_id, domain='source', role='TRAIN',
        group_id=first.group_id, source_graph_certified=True), frames=[])
    for row in fixture.itertuples():
        result['frames'].append(dict(mother_rgb_sha256=row.mother_rgb_sha256,
            mother_index=row.mother_frame_index, mother_timestamp=row.mother_timestamp_seconds,
            B=dict(ok=True, path=row.image_path, rgb_sha256=row.rgb_sha256, byte_sha256=row.content_sha256)))
    frame = pd.DataFrame(h_frame_rows(result, 'B'))
    assert frame.domain.eq('source').all() and frame.video_uid.str.startswith('FFPP/source/').all()
    q._verify_bytes(q._source(frame, 'TRAIN', 'B'))
    with pytest.raises(ValueError, match='source-only'):
        q._source(frame.assign(domain='target'), 'TRAIN', 'B')


def test_direct_worker_resource_boundary_precedes_owned_gpu(accepted_root, monkeypatch):
    root, _ = accepted_root
    q.prepare_probe(root)
    order = []
    def resource(config, *, require_D_priority):
        assert require_D_priority is True
        order.append('resource_boundary')
        raise RuntimeError('D priority is pending')
    monkeypatch.setitem(sys.modules, 'reliability_v2.worker_h', types.SimpleNamespace(verify_resource_boundary=resource))
    monkeypatch.setattr(trainer, 'owned_gpu', lambda *args: pytest.fail('GPU ownership was attempted before D priority'))
    with pytest.raises(RuntimeError, match='D priority'):
        q.worker(root, 'clip_vit_b16')
    assert order == ['resource_boundary']
    failure = json.loads((root / 'qualification/H/clip_vit_b16/AMP/micro32/failure.json').read_text())
    assert failure['status'] == 'REVIEW_REQUIRED'


def test_next_attempt_validates_oom_chain_and_stops_non_oom(accepted_root):
    root, _ = accepted_root
    probe = q.prepare_probe(root)
    assert q.next_attempt(root, 'clip_vit_b16', 'AMP')['microbatch'] == 32
    _, config, binding = q._context(root)
    out = root / 'qualification/H/clip_vit_b16/AMP'
    identity = q._attempt_identity(root, binding, probe, config, 'clip_vit_b16', 32, 'AMP')
    trainer.save_json(out / 'micro32/failure.json', dict(status='OOM_RETRY_ALLOWED', identity=identity))
    plan = q.next_attempt(root, 'clip_vit_b16', 'AMP')
    assert plan['status'] == 'READY' and plan['microbatch'] == 16 and len(plan['oom_failures']) == 1
    identity = q._attempt_identity(root, binding, probe, config, 'clip_vit_b16', 16, 'AMP')
    trainer.save_json(out / 'micro16/failure.json', dict(status='REVIEW_REQUIRED', identity=identity))
    with pytest.raises(ValueError, match='numeric/protocol review'):
        q.next_attempt(root, 'clip_vit_b16', 'AMP')


def test_next_attempt_deep_verifies_completed_receipt_without_launch(accepted_root, monkeypatch):
    root, _ = accepted_root
    q.prepare_probe(root)
    out = root / 'qualification/H/clip_vit_b16/FP32/micro32'
    trainer.save_json(out / 'receipt.json', {'temporary': 'resume-plan contract fixture'})
    calls = []
    monkeypatch.setattr(q, '_verify_attempt', lambda r, p, identity, probe: calls.append(identity))
    monkeypatch.setattr(trainer, 'owned_gpu', lambda *a: pytest.fail('resume plan touched GPU'))
    plan = q.next_attempt(root, 'clip_vit_b16', 'FP32')
    assert plan['status'] == 'COMPLETE' and plan['microbatch'] == 32
    assert plan['receipt_sha256'] == trainer.digest(out / 'receipt.json')
    assert len(calls) == 1 and calls[0]['precision'] == 'FP32'


def test_acceptance_uses_frozen_data_bindings_but_full_GPU_proofs(tmp_path,monkeypatch):
    calls=[]
    def context(root,*,deep=False):
        calls.append(('data',deep));assert deep is False,'accepted H data was fully scanned again'
        return {},{},{}
    monkeypatch.setattr(q,'_context',context)
    monkeypatch.setattr(q,'_probe_receipt',lambda *a:{})
    def proofs(*args,deep=True,**kwargs):
        calls.append(('proofs',deep));assert deep is True
        raise ValueError('missing genuine six GPU proofs')
    monkeypatch.setattr(q,'_qualified_backbones',proofs)
    with pytest.raises(ValueError,match='six GPU'):
        q.accept_qualification(tmp_path)
    assert calls==[('data',False),('proofs',True)]
    assert not (tmp_path/'qualification/H_gate.json').exists()
