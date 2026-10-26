"""Synthetic CPU-only H gate integration tests; no production evidence is written."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from reliability_v2 import geometry_h as g


def test_module_contract_is_available():
    assert importlib.util.find_spec('reliability_v2.data_h') is not None, 'H data gate is missing'
    gate = importlib.import_module('reliability_v2.data_h')
    assert all(callable(getattr(gate, name, None)) for name in
               ('inspect_h_data', 'freeze_h_data', 'validate_h_freeze'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def png(path, seed, size):
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    pixels[0, :8, 0] = np.frombuffer(seed.to_bytes(8, 'little'), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR))
    return pixels


def build(root, conflict=False):
    out = root / 'data/H'
    videos = []
    for role in ('TRAIN', 'DEV', 'SOURCE_TEST'):
        for method in ('real', *g.METHODS):
            videos.append((role, method))
    videos += [('SOURCE_CAL', 'real'), ('TARGET_TEST', 'real'),
               ('TARGET_TEST', 'DeepFakeDetection'), ('SOURCE_UNUSED', 'Deepfakes'),
               ('SOURCE_UNUSED', 'Face2Face')]
    if conflict:
        videos += [('TRAIN', 'Deepfakes'), ('DEV', 'Deepfakes')]
    results, candidates, paths = [], [], []
    for number, (role, method) in enumerate(videos):
        uid = f'fixture/{role}/{method}/{number}'
        directory = out / 'pixels' / str(number)
        directory.mkdir(parents=True)
        source = directory / 'source.mp4'
        source.write_bytes(f'synthetic-no-video-decode-{number}'.encode())
        video = dict(dataset='DFD' if role == 'TARGET_TEST' else 'FFPP',
                     domain='target' if role == 'TARGET_TEST' else 'source',
                     method=method, label=int(method != 'real'), video_id=str(number),
                     video_uid=uid, role=role, source_ids=[str(number)],
                     source_group=f'group/{number}', group_id=f'group/{number}',
                     source_graph_certified=role != 'TARGET_TEST', path=str(source))
        frames = []
        for frame_number in range(8):
            seed = 1 + number * 1000 + frame_number * 10
            frame_dir = directory / str(frame_number)
            mother_path = frame_dir / 'mother.png'
            mother = png(mother_path, seed, 32)
            frame = dict(mother_index=frame_number, mother_timestamp=frame_number / 10,
                         mother_rgb_sha256=g.rgb_hash(mother), mother_path=str(mother_path),
                         mother_byte_sha256=digest(mother_path))
            for offset, version in enumerate(g.VERSIONS, 1):
                image = frame_dir / f'{version}.png'
                pixels = png(image, seed + offset, 256 if version == 'A' else 224)
                mask = frame_dir / f'{version}_mask.png'
                assert cv2.imwrite(str(mask), np.full(pixels.shape[:2], 255, np.uint8))
                frame[version] = dict(ok=True, path=str(image), byte_sha256=digest(image),
                                      rgb_sha256=g.rgb_hash(pixels), mask_path=str(mask),
                                      mask_byte_sha256=digest(mask), padding_ratio=0.0,
                                      input_rgb_sha256=frame['mother_rgb_sha256'])
            frame['AB']['input_rgb_sha256'] = frame['A']['rgb_sha256']
            frame['AB']['parent_byte_sha256'] = frame['A']['byte_sha256']
            frames.append(frame)
        result = dict(video=video, frames=frames, common_B_AB=True,
                      eligible={version: True for version in g.VERSIONS},
                      protocol_sha256=g.protocol_hash(), video_byte_sha256=digest(source))
        results.append(result)
        candidates.append(dict(video))
        paths.append(str(directory / 'manifest.json'))
    if conflict:
        # A SOURCE_UNUSED original pixel bridges two extra active roles. Both
        # extras are quarantined while every registered method remains active.
        bridge = results[-3]['frames'][0]
        for result in results[-2:]:
            result['frames'][0].update({key: bridge[key] for key in
                                       ('mother_path', 'mother_byte_sha256', 'mother_rgb_sha256')})
            for version in ('A', 'B'):
                result['frames'][0][version]['input_rgb_sha256'] = bridge['mother_rgb_sha256']
    for path, result in zip(paths, results):
        write(path, result)
    write(out / 'H_candidates_all.json', candidates)
    write(out / 'H_processed_manifest_paths.json', paths)
    qualification = root / 'qualification'
    source_report = qualification / 'source_report.json'
    write(source_report, {'status': 'PASS', 'source_only': True})
    write(qualification / 'H_geometry_frozen.json', {
        'status': 'FROZEN_BEFORE_TARGET_PREPROCESSING', 'protocol': g.protocol(),
        'protocol_sha256': g.protocol_hash(), 'code_sha256': digest(g.__file__),
        'source_report': str(source_report), 'source_report_sha256': digest(source_report)})
    g.audit_existing(out / 'H_processed_manifest_paths.json', out)
    return len(candidates)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    gate = importlib.import_module('reliability_v2.data_h')
    count = build(tmp_path)
    monkeypatch.setattr(gate, '_EXPECTED_CANDIDATES', count)
    monkeypatch.setattr(gate, '_EXPECTED_SOURCE_UNUSED', 2, raising=False)
    return tmp_path, gate


def review(root, gate, **changes):
    out = root / 'data/H'
    candidate = json.loads((out / 'H_data_freeze_candidate.json').read_text())
    value = dict(schema='H_DATA_INDEPENDENT_REVIEW_V1', status='PASS',
                 reviewer={'identity': 'synthetic-independent-reviewer', 'independent': True},
                 blockers=[], candidate_sha256=digest(out / 'H_data_freeze_candidate.json'),
                 full_audit_sha256=digest(out / 'H_full_candidate_audit_receipt.json'),
                 tables_sha256={key: item['sha256'] for key, item in candidate['tables'].items()},
                 geometry_code_sha256=digest(g.__file__), data_gate_code_sha256=digest(gate.__file__),
                 geometry_protocol_sha256=g.protocol_hash(),
                 quarantine_disposition='VERIFIED_EXCLUDED_FROM_ALL_ACTIVE_TABLES')
    value.update(changes)
    path = root / 'reports/synthetic_review.json'
    write(path, value)
    return path


def rebind_table(out, key, mutate):
    candidate_path = out / 'H_data_freeze_candidate.json'
    full_path = out / 'H_full_candidate_audit_receipt.json'
    candidate, full = [json.loads(path.read_text()) for path in (candidate_path, full_path)]
    table = Path(candidate['tables'][key]['path'])
    frame = mutate(pd.read_parquet(table))
    frame.to_parquet(table, index=False)
    item = {'path': str(table), 'sha256': digest(table), 'rows': len(frame), 'videos': frame.video_uid.nunique()}
    candidate['tables'][key] = item
    full['tables'][key] = item
    write(candidate_path, candidate)
    full['freeze_candidate_sha256'] = digest(candidate_path)
    write(full_path, full)


def test_inspect_is_read_only_and_freeze_preserves_all_logical_mothers(evidence):
    root, gate = evidence
    before = {str(p): digest(p) for p in root.rglob('*') if p.is_file()}
    report = gate.inspect_h_data(root)
    assert report['blockers'] == []
    assert report['training_draw_count'] == 128  # 5 methods * 8 logical mothers -> ceil128
    assert len(report['tables']) == 13
    assert before == {str(p): digest(p) for p in root.rglob('*') if p.is_file()}
    receipt = gate.freeze_h_data(root, review(root, gate))
    assert receipt['status'] == 'FROZEN'
    assert receipt['training_draw_count'] == 128
    assert receipt['tables']['B/train']['rows'] == 40
    assert receipt == gate.validate_h_freeze(root)
    with pytest.raises(ValueError, match='already|exists'):
        gate.freeze_h_data(root, root / 'reports/synthetic_review.json')


def test_reviewed_historical_conflict_does_not_deadlock_freeze(tmp_path, monkeypatch):
    gate = importlib.import_module('reliability_v2.data_h')
    monkeypatch.setattr(gate, '_EXPECTED_CANDIDATES', build(tmp_path, conflict=True))
    monkeypatch.setattr(gate, '_EXPECTED_SOURCE_UNUSED', 2, raising=False)
    assert json.loads((tmp_path / 'data/H/H_data_freeze_candidate.json').read_text())['status'] == 'AUDIT_CONFLICTS_REVIEW_REQUIRED'
    report = gate.inspect_h_data(tmp_path)
    assert report['blockers'] == []
    assert report['quarantined_videos'] == 3
    with pytest.raises(ValueError, match='quarantine'):
        gate.freeze_h_data(tmp_path, review(tmp_path, gate, quarantine_disposition=''))
    assert gate.freeze_h_data(tmp_path, review(tmp_path, gate))['status'] == 'FROZEN'


@pytest.mark.parametrize('field,value', [
    ('status', 'FAIL'), ('blockers', ['unresolved']), ('candidate_sha256', '0' * 64),
    ('full_audit_sha256', '0' * 64), ('tables_sha256', {}),
    ('data_gate_code_sha256', '0' * 64),
    ('reviewer', {'identity': 'author', 'independent': False}),
])
def test_freeze_requires_independent_hash_bound_pass(evidence, field, value):
    root, gate = evidence
    with pytest.raises(ValueError):
        gate.freeze_h_data(root, review(root, gate, **{field: value}))
    assert not (root / 'data/H/freeze.json').exists()


@pytest.mark.parametrize('filename', [
    'H_full_candidate_audit_receipt.json', 'H_artifact_integrity_receipt.json',
    'H_content_source_audit.json', 'H_processed_manifest_paths.json',
])
def test_inspect_blocks_missing_evidence(evidence, filename):
    root, gate = evidence
    (root / 'data/H' / filename).unlink()
    assert gate.inspect_h_data(root)['blockers']


@pytest.mark.parametrize('mutation', ['geometry_code', 'candidate_count', 'artifact_status', 'schema'])
def test_inspect_rejects_stale_or_incomplete_full_audit(evidence, mutation):
    root, gate = evidence
    path = root / 'data/H/H_full_candidate_audit_receipt.json'
    receipt = json.loads(path.read_text())
    key, value = {'geometry_code': ('fresh_code_sha256', '0' * 64),
                  'candidate_count': ('candidate_videos', 1),
                  'artifact_status': ('artifact_integrity_status', 'VALIDATING'),
                  'schema': ('audit_schema', 'OLD_SCHEMA')}[mutation]
    receipt[key] = value
    write(path, receipt)
    assert gate.inspect_h_data(root)['blockers']


def test_corrupt_png_and_changed_manifest_block_inspection(evidence):
    root, gate = evidence
    paths = json.loads((root / 'data/H/H_processed_manifest_paths.json').read_text())
    manifest = json.loads(Path(paths[0]).read_text())
    image = Path(manifest['frames'][0]['B']['path'])
    image.write_bytes(b'corrupt PNG after full audit')
    assert any('integrity' in item.lower() for item in gate.inspect_h_data(root)['blockers'])
    manifest['video']['label'] = 1
    write(paths[0], manifest)
    assert gate.inspect_h_data(root)['blockers']


@pytest.mark.parametrize('kind', ['asymmetric_dedup', 'wrong_mass', 'cross_role_group', 'source_unused'])
def test_rebound_bad_active_table_is_not_trusted(evidence, kind):
    root, gate = evidence
    def mutate(frame):
        if kind == 'asymmetric_dedup':
            return frame.iloc[1:].copy()
        if kind == 'wrong_mass':
            frame.loc[0, 'sampling_mass'] *= 2
        if kind == 'cross_role_group':
            dev = pd.read_parquet(root / 'data/H/B/dev.parquet')
            frame.loc[:7, 'group_id'] = dev.iloc[0].group_id
        if kind == 'source_unused':
            candidates = json.loads((root / 'data/H/H_candidates_all.json').read_text())
            uid = next(row['video_uid'] for row in candidates if row['role'] == 'SOURCE_UNUSED')
            frame.loc[:7, 'video_uid'] = uid
        return frame
    rebind_table(root / 'data/H', 'B/train', mutate)
    assert gate.inspect_h_data(root)['blockers']


def test_validation_rejects_table_and_review_changes_after_freeze(evidence):
    root, gate = evidence
    review_path = review(root, gate)
    gate.freeze_h_data(root, review_path)
    review_path.write_text('{}')
    with pytest.raises(ValueError):
        gate.validate_h_freeze(root)
    review_path = review(root, gate)
    path = root / 'data/H/B/train.parquet'
    path.write_bytes(b'changed')
    with pytest.raises(ValueError):
        gate.validate_h_freeze(root)


def test_missing_review_fails_with_gate_error_without_creating_freeze(evidence):
    root, gate = evidence
    with pytest.raises(ValueError, match='review|evidence'):
        gate.freeze_h_data(root, root / 'missing_review.json')
    assert not (root / 'data/H/freeze.json').exists()


def test_shallow_validation_binds_metadata_while_deep_checks_actual_pixels(evidence):
    root, gate = evidence
    receipt = gate.freeze_h_data(root, review(root, gate))
    paths = json.loads((root / 'data/H/H_processed_manifest_paths.json').read_text())
    manifest = json.loads(Path(paths[0]).read_text())
    Path(manifest['frames'][0]['B']['path']).write_bytes(b'corruption after freeze')
    assert gate.validate_h_freeze(root) == receipt
    with pytest.raises(ValueError, match='integrity'):
        gate.validate_h_freeze(root, deep=True)


def test_shallow_validation_rejects_changed_processed_metadata(evidence):
    root, gate = evidence
    gate.freeze_h_data(root, review(root, gate))
    paths = json.loads((root / 'data/H/H_processed_manifest_paths.json').read_text())
    with Path(paths[0]).open('a') as stream:
        stream.write(' ')
    with pytest.raises(ValueError, match='manifest|metadata'):
        gate.validate_h_freeze(root)


@pytest.mark.parametrize('field', ['training_draw_count', 'table_rows', 'geometry_identity'])
def test_shallow_validation_rejects_corrupt_freeze_summary(evidence, field):
    root, gate = evidence
    gate.freeze_h_data(root, review(root, gate))
    path = root / 'data/H/freeze.json'
    receipt = json.loads(path.read_text())
    if field == 'training_draw_count':
        receipt['training_draw_count'] = 1
    elif field == 'table_rows':
        receipt['tables']['B/train']['rows'] = 1
    else:
        del receipt['identities']['geometry_freeze']
    write(path, receipt)
    with pytest.raises(ValueError):
        gate.validate_h_freeze(root)


def test_unused_candidate_cannot_be_promoted_with_total_and_role_names_preserved(evidence):
    root, gate = evidence
    out = root / 'data/H'
    candidates = json.loads((out / 'H_candidates_all.json').read_text())
    promoted = next(row for row in candidates if row['role'] == 'SOURCE_UNUSED')
    promoted['role'] = 'TRAIN'
    paths = json.loads((out / 'H_processed_manifest_paths.json').read_text())
    for path in paths:
        result = json.loads(Path(path).read_text())
        if result['video']['video_uid'] == promoted['video_uid']:
            result['video']['role'] = 'TRAIN'
            write(path, result)
    write(out / 'H_candidates_all.json', candidates)
    g.audit_existing(out / 'H_processed_manifest_paths.json', out)
    report = gate.inspect_h_data(root)
    assert any('SOURCE_UNUSED' in blocker for blocker in report['blockers'])


def test_audit_disposition_must_agree_with_reconstructed_conflict_evidence(evidence):
    root, gate = evidence
    out = root / 'data/H'
    candidate_path = out / 'H_data_freeze_candidate.json'
    full_path = out / 'H_full_candidate_audit_receipt.json'
    candidate, full = [json.loads(path.read_text()) for path in (candidate_path, full_path)]
    candidate['status'] = full['status'] = 'AUDIT_CONFLICTS_REVIEW_REQUIRED'
    write(candidate_path, candidate)
    full['freeze_candidate_sha256'] = digest(candidate_path)
    write(full_path, full)
    assert gate.inspect_h_data(root)['blockers']


def test_production_candidate_count_cannot_be_bypassed_by_smoke_fixture(tmp_path):
    gate = importlib.import_module('reliability_v2.data_h')
    build(tmp_path)
    report = gate.inspect_h_data(tmp_path)
    assert any('8431' in blocker for blocker in report['blockers'])
    with pytest.raises(ValueError):
        gate.freeze_h_data(tmp_path, review(tmp_path, gate))
