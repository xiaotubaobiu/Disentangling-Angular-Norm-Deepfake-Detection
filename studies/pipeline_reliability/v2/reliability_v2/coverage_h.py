"""Read-only H coverage, selection and geometry summaries without model scores.

The pure kernel accepts arbitrary complete synthetic cohorts. Production has
one entry point, ``report_coverage(root)``, which first calls the unmodified H
freeze validator. It neither writes nor offers an unfrozen/count override.
Missing detection, padding or mask metadata is explicit, never a zero value.
Padding is descriptive and never changes a candidate's frozen eligibility.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path
import statistics

VERSIONS = ('A', 'B', 'AB')
ROLES = ('TRAIN', 'DEV', 'SOURCE_CAL', 'SOURCE_TEST', 'TARGET_TEST', 'SOURCE_UNUSED')
TABLE_KEYS = tuple([f'{v}/{r}' for r in ('train', 'dev') for v in ('B', 'AB')]
                   + [f'{p}/{r}' for p in ('B', 'common/B', 'common/AB')
                      for r in ('SOURCE_CAL', 'SOURCE_TEST', 'TARGET_TEST')])
INDEPENDENCE_SCOPE = ('Known FF source groups and complete exact RGB graph including SOURCE_UNUSED; '
                      'DFD filename parent proxies do not certify complete ancestry or manipulated-face identity')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _rate(numerator, denominator):
    return {'numerator': numerator, 'denominator': denominator,
            'rate': numerator / denominator if denominator else None,
            'status': 'OBSERVED' if denominator else 'NO_DENOMINATOR'}


def _metadata_status(observed, expected):
    return ('NOT_APPLICABLE' if not expected else 'MISSING' if not observed
            else 'COMPLETE' if observed == expected else 'PARTIAL')


def _validate(candidates, manifests):
    candidates = list(candidates)
    if isinstance(manifests, Mapping):
        _require(all(key == value['video']['video_uid'] for key, value in manifests.items()),
                 'Manifest mapping key differs from its video UID')
        manifests = list(manifests.values())
    else:
        manifests = list(manifests)
    by_uid = {row['video_uid']: row for row in candidates}
    results = {row['video']['video_uid']: row for row in manifests}
    _require(len(by_uid) == len(candidates), 'Duplicate candidate video UID')
    _require(len(results) == len(manifests), 'Duplicate processed video UID')
    _require(set(by_uid) == set(results), 'Candidate and processed manifest UID sets differ')
    for uid, candidate in by_uid.items():
        row = results[uid]
        _require(row['video'] == candidate, f'Candidate/manifest identity mismatch: {uid}')
        _require(candidate['role'] in ROLES and candidate['label'] in (0, 1), f'Invalid role/label: {uid}')
        _require(bool(uid) and all(k in candidate for k in ('dataset', 'domain', 'method')), f'Incomplete identity: {uid}')
        frames = row['frames']
        _require(isinstance(frames, list), f'Invalid frame inventory: {uid}')
        if not frames:
            _require(isinstance(row.get('video_failure'), str) and bool(row['video_failure']),
                     f'No mothers without a recorded video failure: {uid}')
        else:
            _require(len(frames) == 8 and len({f['mother_rgb_sha256'] for f in frames}) == 8,
                     f'Exactly eight unique selected mothers required: {uid}')
            _require(not row.get('video_failure'), f'Video failure contradicts selected mothers: {uid}')
            _require(len({f['mother_index'] for f in frames}) == 8, f'Duplicate mother frame index: {uid}')
        for version in VERSIONS:
            errors, hashes = [], []
            for i, frame in enumerate(frames):
                output = frame[version]
                _require(type(output['ok']) is bool, f'Invalid frame success flag: {uid}/{version}')
                if output['ok']:
                    _require(isinstance(output.get('rgb_sha256'), str) and bool(output['rgb_sha256']),
                             f'Missing successful output RGB identity: {uid}/{version}')
                    hashes.append(output['rgb_sha256'])
                    if version == 'AB':
                        _require(frame['A']['ok'], f'AB succeeded without A: {uid}')
                else:
                    _require(isinstance(output.get('reason'), str) and bool(output['reason']),
                             f'Missing failure reason: {uid}/{version}')
                    errors.append(f'{i}:{output["reason"]}')
            if len(hashes) == 8 and len(set(hashes)) < 8:
                errors.append('insufficient_unique_final_rgb')
            eligible = len(frames) == len(hashes) == 8 and len(set(hashes)) == 8
            _require(type(row['eligible'][version]) is bool and row['eligible'][version] == eligible,
                     f'Incorrect fixed-eight eligibility: {uid}/{version}')
            if 'version_failures' in row:
                _require(row['version_failures'].get(version) == errors,
                         f'Stale version failure reasons: {uid}/{version}')
        _require(type(row['common_B_AB']) is bool
                 and row['common_B_AB'] == (row['eligible']['B'] and row['eligible']['AB']),
                 f'Incorrect common B/AB eligibility: {uid}')
    return candidates, [results[uid] for uid in sorted(results)]


def _excluded(row, quarantine):
    return row['video']['role'] == 'SOURCE_UNUSED' or row['video']['video_uid'] in quarantine


def _coverage(rows, quarantine):
    n = len(rows)
    eligible = {v: sum(r['eligible'][v] for r in rows) for v in VERSIONS}
    eligible['common_B_AB'] = sum(r['common_B_AB'] for r in rows)
    unused = sum(r['video']['role'] == 'SOURCE_UNUSED' for r in rows)
    quarantined = sum(r['video']['video_uid'] in quarantine for r in rows)
    excluded = sum(_excluded(r, quarantine) for r in rows)
    failures = {}
    for version in VERSIONS:
        frame_reasons, video_reasons, exact_entries = Counter(), Counter(), Counter()
        for row in rows:
            reasons = set()
            if row.get('video_failure'):
                reasons.add(row['video_failure'])
            for i, frame in enumerate(row['frames']):
                if not frame[version]['ok']:
                    reason = frame[version]['reason']
                    frame_reasons[reason] += 1
                    reasons.add(reason)
                    exact_entries[f'{i}:{reason}'] += 1
            outputs = [f[version] for f in row['frames'] if f[version]['ok']]
            if len(outputs) == 8 and len({o['rgb_sha256'] for o in outputs}) < 8:
                reasons.add('insufficient_unique_final_rgb')
                exact_entries['insufficient_unique_final_rgb'] += 1
            video_reasons.update(reasons)
        failures[version] = {'failed_videos': n - eligible[version],
                             'failure_rate': _rate(n - eligible[version], n),
                             'frame_reasons': dict(sorted(frame_reasons.items())),
                             'video_reasons': dict(sorted(video_reasons.items())),
                             'exact_version_failure_entries': dict(sorted(exact_entries.items()))}
    available = {}
    for version in ('A', 'B', 'AB', 'common_B_AB'):
        available[version] = _rate(sum(not _excluded(r, quarantine)
                                      and (r['common_B_AB'] if version == 'common_B_AB' else r['eligible'][version])
                                      for r in rows), n)
    selected = sum(len(r['frames']) for r in rows)
    return {'candidate_videos': n, 'mother_slots': {'per_video': 8, 'prescribed': 8*n,
            'selected': selected, 'unselected': 8*n-selected},
            'video_level_failures': {'videos': sum(bool(r.get('video_failure')) for r in rows),
                'reasons': dict(sorted(Counter(r['video_failure'] for r in rows if r.get('video_failure')).items()))},
            'processing_eligibility': {v: _rate(value, n) for v, value in eligible.items()},
            'rejection_reasons': failures,
            'administrative_exclusions': {'source_unused_videos': unused, 'quarantined_videos': quarantined,
                 'excluded_union_videos': excluded, 'remaining_candidate_videos': n-excluded},
            'available_cohorts': available,
            'processing_eligibility_after_administrative_exclusions': {
                v: _rate(available[v]['numerator'], n-excluded) for v in available}}


def _quantile(values, probability):
    if not values:
        return None
    position = (len(values)-1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (values[upper]-values[lower]) * (position-lower)


def _padding(outputs):
    observed, invalid, masks = [], 0, 0
    for output in outputs:
        value = output.get('padding_ratio')
        if value is not None:
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1:
                observed.append(float(value))
            else:
                invalid += 1
        masks += bool(output.get('mask_path') and output.get('mask_byte_sha256'))
    observed.sort()
    n, produced = len(observed), len(outputs)
    return {'produced_frames': produced, 'observed_frames': n,
            'missing_padding_frames': produced-n-invalid, 'invalid_padding_frames': invalid,
            'metadata_status': _metadata_status(n, produced),
            'mean': statistics.fmean(observed) if n else None,
            'std_population': statistics.pstdev(observed) if n else None,
            'min': observed[0] if n else None, 'max': observed[-1] if n else None,
            'p05': _quantile(observed, .05), 'p50': _quantile(observed, .5), 'p95': _quantile(observed, .95),
            'mask_metadata': {'recorded_masks': masks, 'missing_masks': produced-masks,
                              'metadata_status': _metadata_status(masks, produced)}}


def _detections(rows, stage):
    expected, recorded, multiface, missing_failure, missing_unknown, invalid = 0, 0, 0, 0, 0, 0
    known_videos, multiface_videos = set(), set()
    for row in rows:
        for frame in row['frames']:
            if stage == 'a_redetection' and not frame['A']['ok']:
                continue
            expected += 1
            metadata = frame.get(stage)
            if metadata is None:
                failure = (not frame['A']['ok'] and not frame['B']['ok']) if stage == 'original_detection' else not frame['AB']['ok']
                if failure:
                    missing_failure += 1
                else:
                    missing_unknown += 1
                continue
            count = metadata.get('qualified_face_count')
            multi = metadata.get('multiface')
            if type(count) is not int or count < 1 or type(multi) is not bool or multi != (count > 1):
                invalid += 1
                continue
            uid = row['video']['video_uid']
            recorded += 1
            multiface += multi
            known_videos.add(uid)
            if multi:
                multiface_videos.add(uid)
    return {'expected_detection_slots': expected, 'recorded_detections': recorded,
            'missing_after_recorded_failure': missing_failure, 'missing_without_failure_evidence': missing_unknown,
            'invalid_detection_records': invalid, 'metadata_status': _metadata_status(recorded, expected),
            'multiface_frames': _rate(multiface, recorded),
            'any_multiface_videos': _rate(len(multiface_videos), len(known_videos)),
            'videos_with_detection_metadata': len(known_videos),
            'candidate_video_detection_coverage': _rate(len(known_videos), len(rows))}


def _geometry(rows, quarantine):
    padding = {}
    for version in VERSIONS:
        scopes = {'all_successful_frames': [], 'eligible_video_frames': [],
                  'ineligible_video_frames': [], 'available_video_frames': []}
        for row in rows:
            outputs = [f[version] for f in row['frames'] if f[version]['ok']]
            scopes['all_successful_frames'].extend(outputs)
            scopes['eligible_video_frames' if row['eligible'][version] else 'ineligible_video_frames'].extend(outputs)
            if not _excluded(row, quarantine) and row['eligible'][version]:
                scopes['available_video_frames'].extend(outputs)
        padding[version] = {scope: _padding(outputs) for scope, outputs in scopes.items()}
    return {'candidate_videos': len(rows),
            'selected_mother_frames': sum(len(r['frames']) for r in rows),
            'detections': {stage: _detections(rows, stage) for stage in ('original_detection', 'a_redetection')},
            'padding': padding}


def _active_tables(tables, rows, quarantine):
    if tables is None:
        return {key: {'status': 'NOT_PROVIDED', 'natural_videos': None, 'mother_rows': None} for key in TABLE_KEYS}
    _require(isinstance(tables, Mapping) and set(tables) == set(TABLE_KEYS), 'All thirteen active tables must be provided')
    report = {}
    fields = ('video_uid', 'role', 'method_id', 'label', 'domain', 'processing_version', 'mother_id',
              'mother_rgb_sha256', 'rgb_sha256', 'mother_frame_index', 'mother_timestamp_seconds',
              'image_path', 'sample_id', 'group_id')
    for key in TABLE_KEYS:
        parts = key.split('/')
        role, version = parts[-1].upper(), parts[-2]
        common = len(parts) == 3 or role in ('TRAIN', 'DEV')
        selected = [r for r in rows if r['video']['role'] == role and not _excluded(r, quarantine)
                    and (r['common_B_AB'] if common else r['eligible']['B'])]
        expected = []
        for row in selected:
            video = row['video']
            for i, frame in enumerate(row['frames']):
                mother_id = f'{video["video_uid"]}/mother/{i:03d}/{frame["mother_rgb_sha256"]}'
                expected.append((video['video_uid'], role, video['method'], video['label'], video['domain'], version,
                    mother_id, frame['mother_rgb_sha256'], frame[version]['rgb_sha256'], frame['mother_index'],
                    frame['mother_timestamp'], frame[version]['path'], mother_id,
                    video.get('group_id') or video.get('source_group')))
        table = tables[key]
        records = table.to_dict('records') if hasattr(table, 'to_dict') else list(table)
        _require(all(set(fields) <= set(r) for r in records), f'Missing identity columns: {key}')
        actual = [tuple(r[field] for field in fields) for r in records]
        _require(Counter(actual) == Counter(expected), f'Active table does not match complete frozen mother cohort: {key}')
        report[key] = {'status': 'VERIFIED', 'natural_videos': len(selected), 'mother_rows': len(records),
                       'role': role, 'processing_version': version,
                       'cohort': 'common_B_AB' if common else 'all_B',
                       'by_method_label': [{'method': method, 'label': label, 'natural_videos': count,
                                            'mother_rows': 8*count}
                                          for (method, label), count in sorted(Counter(
                                              (r['video']['method'], r['video']['label']) for r in selected).items())]}
    return report


def summarize_coverage(candidates, manifests, *, active_tables=None, quarantined_video_uids=()):
    """Summarize a complete UID-matched cohort without altering its selection.

    ``manifests`` is an iterable of manifest objects or a UID-to-object mapping;
    optional ``active_tables`` maps all thirteen names to records/DataFrames.
    Every supplied manifest must match one unique candidate in full. All
    eight-mother eligibility/final-RGB rules apply regardless of cohort size.
    """
    candidates, rows = _validate(candidates, manifests)
    quarantine = set(quarantined_video_uids)
    _require(quarantine <= {r['video_uid'] for r in candidates}, 'Unknown quarantined video UID')
    groups, role_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        v = row['video']
        groups[(v['dataset'], v['domain'], v['role'], v['method'], v['label'])].append(row)
        role_groups[(v['dataset'], v['domain'], v['role'])].append(row)
    coverage, geometry = [], []
    for (dataset, domain, role, method, label), subset in sorted(groups.items()):
        identity = dict(dataset=dataset, domain=domain, role=role, method=method, label=label)
        coverage.append({**identity, **_coverage(subset, quarantine)})
        geometry.append({**identity, **_geometry(subset, quarantine)})
    test_cohorts = {}
    for dataset, role in (('FFPP', 'SOURCE_TEST'), ('DFD', 'TARGET_TEST')):
        subset = [r for r in rows if r['video']['dataset'] == dataset and r['video']['role'] == role]
        available = _coverage(subset, quarantine)['available_cohorts']
        test_cohorts[f'{dataset}/{role}'] = {'candidate_videos': len(subset), 'all_B': available['B'],
            'common_B_AB': available['common_B_AB'],
            'common_retention_of_all_B': _rate(available['common_B_AB']['numerator'], available['B']['numerator'])}
    training_dev = {}
    for role in ('TRAIN', 'DEV'):
        subset = [r for r in rows if r['video']['role'] == role]
        coverage_row = _coverage(subset, quarantine)
        common = coverage_row['available_cohorts']['common_B_AB']
        training_dev[role] = {'candidate_videos': len(subset), 'common_B_AB': common,
            'common_B_AB_after_administrative_exclusions':
                coverage_row['processing_eligibility_after_administrative_exclusions']['common_B_AB'],
            'mother_rows_per_history': 8*common['numerator']}
    return {'schema': 'H_COVERAGE_V1', 'status': 'COMPLETE', 'target_performance_used': False,
            'summary': _coverage(rows, quarantine),
            'coverage_by_role': [dict(dataset=k[0], domain=k[1], role=k[2], **_coverage(v, quarantine))
                                 for k, v in sorted(role_groups.items())],
            'coverage_by_role_method_label': coverage,
            'geometry_summary': _geometry(rows, quarantine), 'geometry_by_role_method_label': geometry,
            'test_cohorts': test_cohorts, 'training_dev_cohorts': training_dev,
            'active_tables': _active_tables(active_tables, rows, quarantine),
            'denominators': {
                'natural_videos': 'Distinct complete video_uid candidates; real UID counted once, no frame-weighted video rates',
                'processing_eligibility': 'All candidates in the stated dataset/domain/role/method/label cell, including SOURCE_UNUSED/quarantine',
                'available_cohorts': 'Eligible videos excluding SOURCE_UNUSED and quarantine; rate denominator remains all cell candidates',
                'processing_eligibility_after_administrative_exclusions': 'Eligible available videos / candidates after union of administrative exclusions',
                'training_dev': 'Both B and AB tables require common_B_AB; evaluation all_B requires only B',
                'mother_slots': 'Eight prescribed per candidate; selected mothers counted separately; decode failures remain unselected',
                'failure_reasons': 'frame_reasons counts failed slots; video_reasons counts distinct affected videos per reason and may overlap',
                'multiface_frames': 'Each stage separately: multiface successful detections / recorded valid detection metadata',
                'any_multiface_videos': 'Videos with any recorded multiface detection / videos with recorded valid detection metadata at that stage',
                'a_redetection': 'Expected only for successful A outputs; missing after failed AB is explicit, never treated as a single face',
                'padding': 'Observed finite padding ratios among successful output frames in each named scope; no pixel or mask re-audit',
                'padding_mask': 'Reported frozen mask-derived padding; AB inherits A support by frozen geometry, with A and AB reported separately',
                'missing_metadata': 'Missing/invalid geometry is counted and yields null statistics when unobserved; never zero-filled',
                'no_selection': 'No scores, thresholds, method-specific dropping, replacement frames or high-padding exclusion'},
            'independence_scope': INDEPENDENCE_SCOPE}


def _sha(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    after = path.stat()
    _require((before.st_size, before.st_mtime_ns, before.st_ino) ==
             (after.st_size, after.st_mtime_ns, after.st_ino), f'File changed while hashing: {path}')
    return digest.hexdigest()


def _check(identity):
    _require(_sha(identity['path']) == identity['sha256'], f'Frozen input hash mismatch: {identity["path"]}')


def _read(identity):
    _check(identity)
    value = json.loads(Path(identity['path']).read_text())
    _check(identity)
    return value


def report_coverage(root):
    """Return a freeze-bound production report without writing any files.

    Uses the gate's default shallow metadata validation, then independently
    binds every input read and exact table membership. Frozen full-artifact
    evidence is preserved as provenance; this call does not freshly audit
    source-video/PNG/mask bytes and does not invoke decoding or detection.
    """
    from . import data_h, geometry_h
    import pandas as pd

    receipt = data_h.validate_h_freeze(root)
    freeze_path = (Path(root) / 'data/H/freeze.json').resolve()
    freeze_identity = {'path': str(freeze_path), 'sha256': _sha(freeze_path)}
    _require(_read(freeze_identity) == receipt, 'Freeze changed after acceptance')
    identities = receipt['identities']
    for identity in identities.values():
        _check(identity)
    candidates = _read(identities['candidates'])
    paths = _read(identities['processed_paths'])
    content = _read(identities['content_audit'])
    _require(len(paths) == len(set(paths)) == len(candidates) == receipt['candidate_videos'],
             'Complete frozen processed manifest paths required')
    manifests, manifest_identities, inventory = [], [], hashlib.sha256()
    for path in paths:
        identity = {'path': path, 'sha256': _sha(path)}
        manifests.append(_read(identity))
        manifest_identities.append(identity)
        inventory.update(json.dumps([str(path), identity['sha256']]).encode() + b'\n')
    _require(inventory.hexdigest() == receipt['candidate_manifest_inventory_sha256'],
             'Frozen processed manifest metadata inventory mismatch')
    _require(set(receipt['tables']) == set(TABLE_KEYS), 'Frozen report requires all thirteen tables')
    tables = {}
    for key, identity in receipt['tables'].items():
        _check(identity)
        table = pd.read_parquet(identity['path'])
        _check(identity)
        videos = int(table.video_uid.nunique()) if len(table) else 0
        _require(len(table) == identity['rows'] and videos == identity['videos'], f'Frozen table summary mismatch: {key}')
        tables[key] = table
    groups = set(content['quarantined_original_groups']) | set(content['source_group_role_conflicts'])
    quarantine = {r['video']['video_uid'] for r in manifests if r['video']['group_id'] in groups}
    _require(len(quarantine) == content['quarantined_videos'], 'Frozen quarantine count differs from manifests')
    code = {name: {'path': str(Path(path).resolve()), 'sha256': _sha(path)}
            for name, path in {'coverage_h': __file__, 'data_h': data_h.__file__, 'geometry_h': geometry_h.__file__}.items()}
    report = summarize_coverage(candidates, manifests, active_tables=tables, quarantined_video_uids=quarantine)
    report['independence_scope'] = receipt['independence_scope']
    report['provenance'] = {'freeze': freeze_identity, 'input_identities': identities,
        'independent_review': receipt['independent_review'], 'tables': receipt['tables'],
        'processed_manifests': manifest_identities,
        'processed_manifest_inventory_sha256': inventory.hexdigest(),
        'artifact_inventory_sha256': receipt['artifact_inventory_sha256'],
        'code_dependencies': code, 'validation_policy': receipt['validation_policy'],
        'fresh_validation_scope': 'SHALLOW_METADATA_BINDING_AND_EXACT_COHORT_RECONSTRUCTION',
        'fresh_pixel_mask_or_source_video_audit': False}
    for identity in [freeze_identity, *identities.values(), *receipt['tables'].values(),
                     receipt['independent_review'], *code.values()]:
        _check(identity)
    final_inventory = hashlib.sha256()
    for path in paths:
        final_inventory.update(json.dumps([str(path), _sha(path)]).encode() + b'\n')
    _require(final_inventory.hexdigest() == inventory.hexdigest(), 'Processed manifests changed during coverage reporting')
    return report
