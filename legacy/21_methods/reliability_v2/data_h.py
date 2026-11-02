"""Read-only H cohort inspection and an explicitly reviewed, immutable freeze.

The external review JSON uses schema ``H_DATA_INDEPENDENT_REVIEW_V1`` with
``status: PASS``, ``reviewer: {identity: <nonempty string>, independent: true}``,
``blockers: []``, and exact SHA256 bindings named ``candidate_sha256``,
``full_audit_sha256``, ``tables_sha256`` (all thirteen table keys),
``geometry_code_sha256``, ``data_gate_code_sha256``, and
``geometry_protocol_sha256``. Historical full-graph conflicts additionally need
``quarantine_disposition: VERIFIED_EXCLUDED_FROM_ALL_ACTIVE_TABLES``.

No API authors a review, regenerates evidence, decodes a source video, calls a
detector, uses a GPU, or reads model scores. Inspection rechecks existing source
bytes and PNGs through the geometry verifier, including masks and decoded RGB.
Inspection, freeze creation and ``validate_h_freeze(..., deep=True)`` are full
integrity scans. Default validation is SHALLOW_METADATA_BINDING: it rehashes
code, review, audit evidence, all tables and every processed manifest, but does
not open source videos or PNGs. Consumers must verify the actual bytes used by
training/inference separately; default validation does not claim fresh pixel
integrity. The receipt records both validation scopes explicitly.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry_h as geometry

_EXPECTED_CANDIDATES = 8431
_EXPECTED_SOURCE_UNUSED = 560
_AUDIT_SCHEMA = 'FULL_CANDIDATE_GRAPH_INCLUDING_SOURCE_UNUSED_V2'
_REVIEW_SCHEMA = 'H_DATA_INDEPENDENT_REVIEW_V1'
_FREEZE_SCHEMA = 'H_DATA_FREEZE_V1'
_ROLES = ('SOURCE_CAL', 'SOURCE_TEST', 'TARGET_TEST')
_TABLE_KEYS = tuple([f'{version}/{role}' for role in ('train', 'dev') for version in ('B', 'AB')]
                    + [f'{prefix}/{role}' for prefix in ('B', 'common/B', 'common/AB') for role in _ROLES])
_AUDIT_STATUSES = {'AUDIT_COMPLETE_READY_FOR_H_DATA_FREEZE_REVIEW', 'AUDIT_CONFLICTS_REVIEW_REQUIRED'}
_VALIDATION_POLICY = {'freeze': 'FULL_ARTIFACT_REHASH_AND_COHORT_RECONSTRUCTION',
                      'default': 'SHALLOW_METADATA_BINDING',
                      'deep': 'FULL_ARTIFACT_REHASH_AND_COHORT_RECONSTRUCTION'}


class HDataGateError(ValueError):
    """An H freeze or its current evidence failed closed."""


def _require(condition, reason):
    if not condition:
        raise HDataGateError(reason)


def _sha(path):
    path = Path(path)
    before = path.stat()
    value = geometry.sha256_file(path)
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns, before.st_ino)
             == (after.st_size, after.st_mtime_ns, after.st_ino), f'File changed during inspection: {path}')
    return value


def _identity(path):
    path = Path(path).resolve()
    _require(path.is_file(), f'Missing evidence file: {path}')
    return {'path': str(path), 'sha256': _sha(path)}


def _json(path, identities, name):
    identity = _identity(path)
    value = json.loads(Path(identity['path']).read_text())
    _require(_sha(identity['path']) == identity['sha256'], f'Evidence changed while read: {path}')
    identities[name] = identity
    return value


def _equal(actual, expected, reason):
    _require(actual == expected, reason)


def _table_check(key, frame, quarantined_uids):
    required = {'image_path', 'label', 'method_id', 'video_uid', 'group_id', 'domain', 'role',
                'rgb_sha256', 'content_sha256', 'mother_id', 'sample_id', 'mother_rgb_sha256',
                'mother_frame_index', 'mother_timestamp_seconds', 'processing_version'}
    _require(required <= set(frame.columns), f'{key}: missing table columns {sorted(required-set(frame.columns))}')
    _require(not frame.empty and not frame[list(required)].isna().any().any(), f'{key}: empty/null active table')
    role = key.split('/')[-1].upper()
    _require(set(frame.role) == {role}, f'{key}: incorrect active role')
    _require(not set(frame.video_uid) & quarantined_uids, f'{key}: quarantined or SOURCE_UNUSED video is active')
    _require(not frame.mother_id.duplicated().any(), f'{key}: duplicated mother identity or asymmetric pixel dedup')
    _require(frame.sample_id.equals(frame.mother_id), f'{key}: logical sample identity differs from mother')
    _require(frame.groupby('video_uid').size().eq(8).all()
             and frame.groupby('video_uid').rgb_sha256.nunique().eq(8).all()
             and frame.groupby('video_uid').mother_rgb_sha256.nunique().eq(8).all(),
             f'{key}: each video must retain eight unique final RGB and mother slots')
    for column in ('label', 'method_id', 'group_id', 'role', 'domain'):
        _require(frame.groupby('video_uid')[column].nunique().eq(1).all(), f'{key}: inconsistent video {column}')
    _require(set(frame.label) <= {0, 1} and frame.label.eq(frame.method_id.ne('real').astype(int)).all(),
             f'{key}: method/label inconsistency')
    expected = {'real'} if role == 'SOURCE_CAL' else ({'real', 'DeepFakeDetection'} if role == 'TARGET_TEST'
                                                     else {'real', *geometry.METHODS})
    _equal(set(frame.method_id), expected, f'{key}: registered method coverage changed')
    _require(set(frame.processing_version) == ({'AB'} if '/AB/' in '/'+key+'/' else {'B'}),
             f'{key}: incorrect geometry version')


def _pair_check(b, ab, key, weighted=False):
    columns = ['mother_id', 'sample_id', 'mother_rgb_sha256', 'mother_frame_index',
               'mother_timestamp_seconds', 'video_uid', 'label', 'method_id', 'group_id', 'role', 'domain']
    if weighted:
        columns += ['sampling_mass', 'method_contributions']
    _require(all(column in b and column in ab for column in columns), f'{key}: missing paired metadata')
    _require(b[columns].reset_index(drop=True).equals(ab[columns].reset_index(drop=True)),
             f'{key}: ordered B/AB mothers, labels, methods, source groups or weights differ')


def _weight_check(frame, key):
    _require({'sampling_mass', 'method_contributions'} <= set(frame.columns), f'{key}: missing sampling weights')
    weights = frame.sampling_mass.to_numpy(dtype=float)
    _require(np.isfinite(weights).all() and (weights > 0).all(), f'{key}: invalid sampling weights')
    videos = frame[['method_id', 'video_uid']].drop_duplicates().groupby('method_id').size().to_dict()
    for method in ('real', *geometry.METHODS):
        subset = frame[frame.method_id.eq(method)]
        probability = .5 if method == 'real' else .125
        expected = probability / videos[method] / 8
        _require(np.allclose(subset.sampling_mass, expected, rtol=0, atol=1e-14),
                 f'{key}: method/video/frame sampling mass mismatch for {method}')
        _require(np.isclose(subset.sampling_mass.sum(), probability, rtol=0, atol=1e-12),
                 f'{key}: method mass mismatch for {method}')
        for row in subset.itertuples(index=False):
            contribution = json.loads(row.method_contributions)
            _require(set(contribution) == {method} and np.isclose(contribution[method], expected, rtol=0, atol=1e-14),
                     f'{key}: logical mother method contribution mismatch')


def _inspect(root, report):
    root = Path(root).resolve(strict=True)
    out = root / 'data/H'
    identities = report['identities']
    identities['geometry_code'] = _identity(geometry.__file__)
    identities['data_gate_code'] = _identity(__file__)
    frozen_geometry = _json(root / 'qualification/H_geometry_frozen.json', identities, 'geometry_freeze')
    _equal(frozen_geometry.get('status'), 'FROZEN_BEFORE_TARGET_PREPROCESSING', 'Source geometry was not frozen before target preprocessing')
    _equal(frozen_geometry.get('protocol_sha256'), geometry.protocol_hash(), 'Frozen geometry protocol is stale')
    _equal(frozen_geometry.get('protocol'), geometry.protocol(), 'Frozen geometry protocol payload differs from current code')
    identities['source_qualification'] = _identity(frozen_geometry['source_report'])
    _equal(identities['source_qualification']['sha256'], frozen_geometry.get('source_report_sha256'), 'Geometry source qualification hash mismatch')
    candidate = _json(out / 'H_data_freeze_candidate.json', identities, 'freeze_candidate')
    full = _json(out / 'H_full_candidate_audit_receipt.json', identities, 'full_audit')
    integrity = _json(out / 'H_artifact_integrity_receipt.json', identities, 'artifact_integrity')
    content = _json(out / 'H_content_source_audit.json', identities, 'content_audit')
    candidates = _json(out / 'H_candidates_all.json', identities, 'candidates')
    paths = _json(out / 'H_processed_manifest_paths.json', identities, 'processed_paths')
    for obj, name in ((full, 'full audit'), (candidate, 'candidate'), (integrity, 'integrity'), (content, 'content audit')):
        _equal(obj.get('audit_schema'), _AUDIT_SCHEMA, f'{name}: full candidate SOURCE_UNUSED graph schema required')
    for obj, name in ((full, 'full audit'), (candidate, 'candidate')):
        _require(obj.get('status') in _AUDIT_STATUSES, f'{name}: incomplete or failed audit')
        _equal(obj.get('candidate_videos'), _EXPECTED_CANDIDATES, f'{name}: exactly {_EXPECTED_CANDIDATES} candidate videos required')
        _require(obj.get('data_frozen') is False, f'{name}: invalid pre-freeze status')
        _equal(obj.get('geometry_protocol_sha256'), geometry.protocol_hash(), f'{name}: stale geometry protocol')
    _equal(full['status'], candidate['status'], 'Candidate/full audit disposition mismatch')
    _require(candidate.get('target_performance_used') is False, 'Candidate used target performance')
    _equal(full.get('processed_manifest_count'), _EXPECTED_CANDIDATES, 'Full audit is incomplete')
    _equal(full.get('artifact_integrity_status'), 'PASSED', 'Full audit artifact integrity not PASSED')
    for obj, key, name in ((full, 'fresh_code_sha256', 'full audit'),
                           (candidate, 'code_sha256', 'candidate'), (integrity, 'fresh_code_sha256', 'integrity')):
        _equal(obj.get(key), identities['geometry_code']['sha256'], f'{name}: fresh current geometry code hash required')
    for key, identity in (('candidate_manifest_sha256', 'candidates'), ('processed_paths_sha256', 'processed_paths'),
                          ('content_source_audit_sha256', 'content_audit'), ('freeze_candidate_sha256', 'freeze_candidate'),
                          ('artifact_integrity_receipt_sha256', 'artifact_integrity')):
        _equal(full.get(key), identities[identity]['sha256'], f'Full audit binding mismatch: {identity}')
    _equal(integrity.get('status'), 'PASSED', 'Artifact integrity must be PASSED')
    _equal(integrity.get('candidate_videos'), _EXPECTED_CANDIDATES, 'Artifact integrity candidate count mismatch')
    _equal(integrity.get('source_video_byte_hashes_verified'), _EXPECTED_CANDIDATES, 'Source byte audit incomplete')
    _equal(full.get('source_videos_decoded'), 0, 'Full audit unexpectedly decoded source videos')
    _equal(full.get('detector_calls'), 0, 'Full audit unexpectedly called detector')
    _require(isinstance(candidates, list) and isinstance(paths, list)
             and len(candidates) == len(paths) == _EXPECTED_CANDIDATES,
             f'Exactly {_EXPECTED_CANDIDATES} complete candidate manifests required')
    unused_count = sum(row['role'] == 'SOURCE_UNUSED' for row in candidates)
    _equal(unused_count, _EXPECTED_SOURCE_UNUSED,
           f'Full graph requires exactly {_EXPECTED_SOURCE_UNUSED} SOURCE_UNUSED candidates')
    report['source_unused_candidate_videos'] = unused_count
    by_uid = {row['video_uid']: row for row in candidates}
    _require(len(by_uid) == len(candidates) and len(set(paths)) == len(paths), 'Duplicate candidate UID or manifest path')
    _require(set(row['role'] for row in candidates) == {'TRAIN', 'DEV', 'SOURCE_UNUSED', *_ROLES},
             'Candidate role inventory must include SOURCE_UNUSED and every active role')
    results, actual_uids, metadata, inventory = [], [], hashlib.sha256(), hashlib.sha256()
    for path in paths:
        identity = _identity(path)
        result = json.loads(Path(path).read_text())
        uid = result['video']['video_uid']
        _require(uid in by_uid, f'Unregistered processed candidate UID: {uid}')
        _equal(result['video'], by_uid[uid], f'Candidate/processed video identity mismatch: {uid}')
        _equal(result.get('protocol_sha256'), geometry.protocol_hash(), f'Candidate manifest stale protocol: {uid}')
        metadata.update(json.dumps([str(path), identity['sha256']]).encode() + b'\n')
        for version in geometry.VERSIONS:
            outputs = [frame[version] for frame in result['frames'] if frame[version]['ok']]
            expected_eligible = len(result['frames']) == len(outputs) == 8 and len({row['rgb_sha256'] for row in outputs}) == 8
            _equal(result['eligible'][version], expected_eligible, f'Incorrect eligibility for {uid}/{version}')
        _equal(result['common_B_AB'], result['eligible']['B'] and result['eligible']['AB'], f'Incorrect common eligibility: {uid}')
        for frame in result['frames']:
            for version in ('A', 'B'):
                if frame[version]['ok']:
                    _equal(frame[version].get('input_rgb_sha256'), frame['mother_rgb_sha256'], f'Mother input binding mismatch: {uid}/{version}')
            if frame['AB']['ok']:
                _require(frame['A']['ok'], f'AB succeeded without A: {uid}')
                _equal(frame['AB'].get('input_rgb_sha256'), frame['A']['rgb_sha256'], f'AB input is not saved A RGB: {uid}')
                _equal(frame['AB'].get('parent_byte_sha256'), frame['A']['byte_sha256'], f'AB parent byte hash mismatch: {uid}')
        try:
            verified = geometry.verify_record_artifacts(result)
        except (OSError, ValueError, KeyError) as exc:
            raise HDataGateError(f'Artifact integrity failed for {uid}: {exc}') from exc
        inventory.update(json.dumps([str(path), verified['artifact_fingerprint_sha256']]).encode() + b'\n')
        _equal(_sha(path), identity['sha256'], f'Candidate manifest changed during integrity check: {uid}')
        actual_uids.append(uid)
        results.append(result)
    _require(len(set(actual_uids)) == len(paths) and set(actual_uids) == set(by_uid), 'Processed candidate UID set mismatch')
    _equal(metadata.hexdigest(), integrity.get('metadata_manifest_inventory_sha256'), 'Candidate manifest inventory changed after full audit')
    _equal(inventory.hexdigest(), integrity.get('artifact_inventory_sha256'), 'Artifact integrity inventory changed after full audit')
    recomputed, quarantine = geometry.h_content_audit(results)
    _equal(content, recomputed, 'Full candidate graph/content audit differs from manifests')
    _equal(candidate.get('quarantined_group_count'), len(quarantine), 'Candidate quarantine count mismatch')
    quarantined_uids = {row['video']['video_uid'] for row in results
                        if row['video']['group_id'] in quarantine or row['video']['role'] == 'SOURCE_UNUSED'}
    report['quarantined_videos'] = sum(row['video']['group_id'] in quarantine for row in results)
    report['historical_conflicts'] = bool(content['cross_role_rgb_conflicts'] or content['label_rgb_conflicts']
                                          or content['source_group_role_conflicts'])
    _equal(candidate['status'], 'AUDIT_CONFLICTS_REVIEW_REQUIRED' if report['historical_conflicts']
           else 'AUDIT_COMPLETE_READY_FOR_H_DATA_FREEZE_REVIEW',
           'Candidate audit disposition contradicts reconstructed full-graph conflict evidence')
    _equal(set(candidate.get('tables', {})), set(_TABLE_KEYS), 'Candidate must bind all thirteen H tables')
    _equal(full.get('tables'), candidate['tables'], 'Full audit table bindings differ from candidate')
    expected_tables = geometry.h_evaluation_rows(results, quarantine)
    for role in ('TRAIN', 'DEV'):
        b, ab = geometry.h_training_rows(results, role, quarantine)
        expected_tables[f'B/{role.lower()}'], expected_tables[f'AB/{role.lower()}'] = b, ab
    frames = {}
    for key in _TABLE_KEYS:
        bound = candidate['tables'][key]
        path = (out / f'{key}.parquet').resolve(strict=True)
        _equal(Path(bound['path']).resolve(strict=True), path, f'{key}: candidate table path mismatch')
        _equal(_sha(path), bound['sha256'], f'{key}: table SHA256 mismatch')
        frame = pd.read_parquet(path)
        _equal(_sha(path), bound['sha256'], f'{key}: table changed while read')
        _equal(bound['rows'], len(frame), f'{key}: table row count mismatch')
        _equal(bound['videos'], frame.video_uid.nunique(), f'{key}: table video count mismatch')
        _table_check(key, frame, quarantined_uids)
        expected = pd.DataFrame(expected_tables[key])
        try:
            pd.testing.assert_frame_equal(frame.reset_index(drop=True), expected.reset_index(drop=True), check_exact=True)
        except AssertionError as exc:
            raise HDataGateError(f'{key}: active table does not exactly match full audited candidate cohort: {exc}') from exc
        report['tables'][key] = {'path': str(path), 'sha256': bound['sha256'], 'rows': len(frame),
                                 'videos': int(frame.video_uid.nunique())}
        frames[key] = frame
    for role in ('train', 'dev'):
        b, ab = frames[f'B/{role}'], frames[f'AB/{role}']
        _pair_check(b, ab, role, weighted=True)
        _weight_check(b, f'B/{role}')
        _weight_check(ab, f'AB/{role}')
        audit = candidate['training'][role.upper()]
        _equal(audit['logical_mother_rows'], len(b), f'{role}: candidate logical mother count mismatch')
        _equal(audit['draw_count'], math.ceil(len(b) / 128) * 128, f'{role}: candidate draw count mismatch')
        _require(audit['paired_mother_ids_identical'] is True, f'{role}: candidate pair audit failed')
    for role in _ROLES:
        common_b, common_ab = frames[f'common/B/{role}'], frames[f'common/AB/{role}']
        _pair_check(common_b, common_ab, role)
        all_b = frames[f'B/{role}']
        _require(set(common_b.mother_id) <= set(all_b.mother_id), f'{role}: common cohort is not an all-B subset')
        subset = all_b[all_b.mother_id.isin(common_b.mother_id)].reset_index(drop=True)
        _require(common_b.reset_index(drop=True).equals(subset), f'{role}: common B differs from all-B cohort rows')
    active = pd.concat([frames[key] for key in ('B/train', 'AB/train', 'B/dev', 'AB/dev',
                        *(f'{prefix}/{role}' for prefix in ('B', 'common/AB') for role in _ROLES))], ignore_index=True)
    for column in ('group_id', 'video_uid', 'mother_rgb_sha256', 'rgb_sha256'):
        _require(active.groupby(column).role.nunique().le(1).all(), f'Active cross-role source/RGB leakage: {column}')
    # Original and derived hashes share one namespace, as in the full graph.
    rgb_rows = pd.concat([active[['role', 'label', column]].rename(columns={column: 'rgb'})
                          for column in ('rgb_sha256', 'mother_rgb_sha256')], ignore_index=True)
    _require(rgb_rows.groupby('rgb').role.nunique().le(1).all(), 'Active original/derived RGB cross-role leakage')
    _require(rgb_rows.groupby('rgb').label.nunique().le(1).all(), 'Active RGB label conflict')
    report['training_draw_count'] = math.ceil(len(frames['B/train']) / 128) * 128
    report['logical_train_mother_rows'] = len(frames['B/train'])
    report['geometry_protocol_sha256'] = geometry.protocol_hash()
    report['candidate_videos'] = len(candidates)
    report['candidate_manifest_inventory_sha256'] = metadata.hexdigest()
    report['artifact_inventory_sha256'] = inventory.hexdigest()
    # Detect evidence/table mutation during the full integrity scan.
    _unchanged(report)


def _unchanged(report):
    for identity in [*report['identities'].values(), *report['tables'].values()]:
        _equal(_sha(identity['path']), identity['sha256'], f'Evidence changed during inspection: {identity["path"]}')
    if 'candidate_manifest_inventory_sha256' in report:
        paths = json.loads(Path(report['identities']['processed_paths']['path']).read_text())
        _require(len(paths) == len(set(paths)) == report['candidate_videos'] == _EXPECTED_CANDIDATES,
                 'Processed metadata manifest count mismatch')
        metadata = hashlib.sha256()
        for path in paths:
            metadata.update(json.dumps([str(path), _sha(path)]).encode() + b'\n')
        _equal(metadata.hexdigest(), report['candidate_manifest_inventory_sha256'],
               'Processed manifest metadata inventory changed after freeze/audit')


def inspect_h_data(root):
    """Return blockers and verified identities, without writing any file.

    Production accepts exactly 8431 candidates including the 560 SOURCE_UNUSED
    videos. A failed or partial audit is a blocker, never an implicit waiver.
    Historical conflicts are acceptable only if recomputation proves all of
    their connected components absent from every active cohort.
    """
    report = {'schema': 'H_DATA_INSPECTION_V1', 'status': 'BLOCKED', 'blockers': [],
              'tables': {}, 'identities': {}, 'target_performance_used': False}
    try:
        _inspect(root, report)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        report['blockers'].append(str(exc))
    if not report['blockers']:
        report['status'] = 'READY_FOR_INDEPENDENT_REVIEW'
    return report


def _review(path, report):
    identity = _identity(path)
    review = json.loads(Path(identity['path']).read_text())
    _equal(review.get('schema'), _REVIEW_SCHEMA, 'Independent review schema mismatch')
    _equal(review.get('status'), 'PASS', 'Independent review must explicitly PASS')
    _equal(review.get('blockers'), [], 'Independent review has unresolved blockers')
    reviewer = review.get('reviewer', {})
    _require(isinstance(reviewer.get('identity'), str) and reviewer['identity'].strip()
             and reviewer.get('independent') is True, 'Independent reviewer identity/attestation required')
    bindings = {'candidate_sha256': report['identities']['freeze_candidate']['sha256'],
                'full_audit_sha256': report['identities']['full_audit']['sha256'],
                'tables_sha256': {key: item['sha256'] for key, item in report['tables'].items()},
                'geometry_code_sha256': report['identities']['geometry_code']['sha256'],
                'data_gate_code_sha256': report['identities']['data_gate_code']['sha256'],
                'geometry_protocol_sha256': report['geometry_protocol_sha256']}
    for key, expected in bindings.items():
        _equal(review.get(key), expected, f'Independent review hash binding mismatch: {key}')
    if report['historical_conflicts']:
        _equal(review.get('quarantine_disposition'), 'VERIFIED_EXCLUDED_FROM_ALL_ACTIVE_TABLES',
               'Independent review must explicitly verify quarantine disposition')
    _equal(_sha(identity['path']), identity['sha256'], 'Independent review changed while read')
    return {**identity, 'reviewer': reviewer}


def freeze_h_data(root, review_path):
    """Create data/H/freeze.json exclusively after a hash-bound external PASS.

    Refuses every existing freeze, including a malformed one. This function has
    no candidate-count override and never changes tables or audit artifacts.
    """
    path = Path(root) / 'data/H/freeze.json'
    _require(not path.exists() and not path.is_symlink(), 'H freeze already exists; refusing overwrite')
    report = inspect_h_data(root)
    _require(not report['blockers'], 'H data inspection blocked: ' + '; '.join(report['blockers']))
    independent_review = _review(review_path, report)
    _unchanged(report)
    _equal(_sha(independent_review['path']), independent_review['sha256'], 'Independent review changed before freeze')
    receipt = {**report, 'schema': _FREEZE_SCHEMA, 'status': 'FROZEN',
               'utc': datetime.now(timezone.utc).isoformat(), 'independent_review': independent_review,
               'training_draw_count': report['training_draw_count'],
               'code_sha256': report['identities']['data_gate_code']['sha256'],
               'geometry_code_sha256': report['identities']['geometry_code']['sha256'],
               'validation_policy': dict(_VALIDATION_POLICY),
               'independence_scope': 'Known FF source groups and complete exact RGB graph including SOURCE_UNUSED; DFD filename parent proxies do not certify complete ancestry'}
    serialized = json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    with path.open('x') as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


def _frozen_metadata(receipt, root):
    """Rebind receipt summaries to the already reviewed immutable evidence."""
    root = Path(root).resolve()
    out = root / 'data/H'
    named_paths = {'geometry_code': Path(geometry.__file__), 'data_gate_code': Path(__file__),
                   'geometry_freeze': root / 'qualification/H_geometry_frozen.json',
                   'freeze_candidate': out / 'H_data_freeze_candidate.json',
                   'full_audit': out / 'H_full_candidate_audit_receipt.json',
                   'artifact_integrity': out / 'H_artifact_integrity_receipt.json',
                   'content_audit': out / 'H_content_source_audit.json',
                   'candidates': out / 'H_candidates_all.json',
                   'processed_paths': out / 'H_processed_manifest_paths.json'}
    _equal(set(receipt.get('identities', {})), {*named_paths, 'source_qualification'},
           'Frozen H evidence identity inventory is incomplete')
    for name, path in named_paths.items():
        _equal(Path(receipt['identities'][name]['path']).resolve(), path.resolve(),
               f'Frozen H evidence path changed: {name}')
    _equal(receipt.get('validation_policy'), _VALIDATION_POLICY, 'Frozen H validation scope policy changed')
    def load(name):
        return json.loads(Path(receipt['identities'][name]['path']).read_text())
    candidate, full, integrity, content, frozen_geometry = [load(name) for name in
        ('freeze_candidate', 'full_audit', 'artifact_integrity', 'content_audit', 'geometry_freeze')]
    candidate_records = load('candidates')
    _equal(len(candidate_records), _EXPECTED_CANDIDATES, 'Frozen H complete candidate inventory count mismatch')
    unused_count = sum(row['role'] == 'SOURCE_UNUSED' for row in candidate_records)
    _equal(unused_count, _EXPECTED_SOURCE_UNUSED, f'Frozen H requires {_EXPECTED_SOURCE_UNUSED} SOURCE_UNUSED candidates')
    _equal(receipt.get('source_unused_candidate_videos'), unused_count, 'Frozen H SOURCE_UNUSED count binding mismatch')
    for obj in (candidate, full, integrity, content):
        _equal(obj.get('audit_schema'), _AUDIT_SCHEMA, 'Frozen H full audit schema mismatch')
    _require(candidate.get('status') in _AUDIT_STATUSES and full.get('status') == candidate['status'],
             'Frozen H audit disposition mismatch')
    _equal(full.get('artifact_integrity_status'), 'PASSED', 'Frozen H full artifact audit was not PASSED')
    _equal(integrity.get('status'), 'PASSED', 'Frozen H artifact integrity was not PASSED')
    for obj, field in ((candidate, 'code_sha256'), (full, 'fresh_code_sha256'), (integrity, 'fresh_code_sha256')):
        _equal(obj.get(field), receipt['geometry_code_sha256'], 'Frozen H audit code binding mismatch')
        _equal(obj.get('candidate_videos'), _EXPECTED_CANDIDATES, 'Frozen H audit candidate count mismatch')
    for field, identity_name in (('candidate_manifest_sha256', 'candidates'), ('processed_paths_sha256', 'processed_paths'),
                                  ('content_source_audit_sha256', 'content_audit'), ('freeze_candidate_sha256', 'freeze_candidate'),
                                  ('artifact_integrity_receipt_sha256', 'artifact_integrity')):
        _equal(full.get(field), receipt['identities'][identity_name]['sha256'], f'Frozen H full audit binding mismatch: {field}')
    _equal(full.get('tables'), candidate.get('tables'), 'Frozen H candidate/full audit table binding mismatch')
    _equal(set(candidate.get('tables', {})), set(_TABLE_KEYS), 'Frozen H candidate table inventory mismatch')
    for key in _TABLE_KEYS:
        item = dict(candidate['tables'][key])
        item['path'] = str(Path(item['path']).resolve())
        _equal(item['path'], str((out / f'{key}.parquet').resolve()), f'Frozen H candidate table path mismatch: {key}')
        _equal(receipt['tables'][key], item, f'Frozen H table summary mismatch: {key}')
    mothers = receipt['tables']['B/train']['rows']
    _require(isinstance(mothers, int) and mothers > 0, 'Frozen H training mother count is invalid')
    _equal(receipt.get('logical_train_mother_rows'), mothers, 'Frozen H logical mother count mismatch')
    _equal(receipt.get('training_draw_count'), ((mothers + 127) // 128) * 128, 'Frozen H training draw count mismatch')
    _equal(candidate['training']['TRAIN']['logical_mother_rows'], mothers, 'Frozen H candidate mother count mismatch')
    _equal(candidate['training']['TRAIN']['draw_count'], receipt['training_draw_count'], 'Frozen H candidate draw count mismatch')
    _equal(receipt.get('candidate_manifest_inventory_sha256'), integrity.get('metadata_manifest_inventory_sha256'),
           'Frozen H processed metadata inventory binding mismatch')
    _equal(receipt.get('artifact_inventory_sha256'), integrity.get('artifact_inventory_sha256'),
           'Frozen H artifact inventory binding mismatch')
    _equal(receipt.get('historical_conflicts'), bool(content['cross_role_rgb_conflicts'] or content['label_rgb_conflicts']
                                                    or content['source_group_role_conflicts']), 'Frozen H conflict disposition mismatch')
    _equal(candidate['status'], 'AUDIT_CONFLICTS_REVIEW_REQUIRED' if receipt['historical_conflicts']
           else 'AUDIT_COMPLETE_READY_FOR_H_DATA_FREEZE_REVIEW', 'Frozen H audit disposition contradicts full-graph conflicts')
    _equal(receipt.get('quarantined_videos'), content.get('quarantined_videos'), 'Frozen H quarantine count mismatch')
    _equal(frozen_geometry.get('status'), 'FROZEN_BEFORE_TARGET_PREPROCESSING', 'Frozen H source geometry qualification changed')
    _equal(frozen_geometry.get('protocol_sha256'), receipt['geometry_protocol_sha256'], 'Frozen H source geometry protocol mismatch')
    _equal(Path(frozen_geometry['source_report']).resolve(), Path(receipt['identities']['source_qualification']['path']).resolve(),
           'Frozen H source qualification path mismatch')
    _equal(frozen_geometry.get('source_report_sha256'), receipt['identities']['source_qualification']['sha256'],
           'Frozen H source qualification binding mismatch')


def validate_h_freeze(root, *, deep=False):
    """Verify a frozen receipt and return it unchanged, without writing.

    Default SHALLOW_METADATA_BINDING rehashes frozen code/review/audit evidence,
    all thirteen table files and every processed manifest's metadata. It does
    not rehash or decode source-video/PNG bytes. ``deep=True`` additionally runs
    the complete artifact verification and candidate cohort reconstruction.
    The returned receipt's UTC and full-artifact evidence describe the freeze
    event, not a new full pixel audit by shallow validation.
    """
    path = Path(root) / 'data/H/freeze.json'
    identity = _identity(path)
    receipt = json.loads(path.read_text())
    _equal(receipt.get('schema'), _FREEZE_SCHEMA, 'H freeze schema mismatch')
    _equal(receipt.get('status'), 'FROZEN', 'H data is not FROZEN')
    _equal(receipt.get('candidate_videos'), _EXPECTED_CANDIDATES, f'Frozen H requires {_EXPECTED_CANDIDATES} candidates')
    _equal(set(receipt.get('tables', {})), set(_TABLE_KEYS), 'Frozen H must bind all thirteen tables')
    _equal(receipt.get('blockers'), [], 'Frozen H receipt contains blockers')
    _equal(receipt.get('code_sha256'), _sha(__file__), 'Frozen data gate code changed')
    _equal(receipt.get('geometry_code_sha256'), _sha(geometry.__file__), 'Frozen geometry code changed')
    _equal(receipt.get('geometry_protocol_sha256'), geometry.protocol_hash(), 'Frozen geometry protocol changed')
    _equal(receipt['code_sha256'], receipt['identities']['data_gate_code']['sha256'], 'Frozen data gate identity differs')
    _equal(receipt['geometry_code_sha256'], receipt['identities']['geometry_code']['sha256'], 'Frozen geometry code identity differs')
    _frozen_metadata(receipt, root)
    _unchanged(receipt)
    if deep:
        report = inspect_h_data(root)
        _require(not report['blockers'], 'H frozen data verification blocked: ' + '; '.join(report['blockers']))
        for key in report:
            if key not in ('schema', 'status'):
                _equal(receipt.get(key), report[key], f'Frozen H identity mismatch: {key}')
    else:
        report = receipt
    review = _review(receipt['independent_review']['path'], report)
    _equal(receipt['independent_review'], review, 'Frozen independent review changed')
    _equal(_sha(path), identity['sha256'], 'Freeze receipt changed during validation')
    return receipt
