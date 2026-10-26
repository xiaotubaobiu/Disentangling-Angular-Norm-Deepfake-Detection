"""Synthetic metadata-only H coverage tests; no production freeze is weakened."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def module():
    assert importlib.util.find_spec('reliability_v2.coverage_h') is not None, 'H coverage module is missing'
    return importlib.import_module('reliability_v2.coverage_h')


def video(number, role='SOURCE_TEST', method='real', dataset='FFPP'):
    return dict(dataset=dataset, domain='target' if dataset == 'DFD' else 'source',
                video_uid=f'{dataset}/{role}/{method}/{number}', video_id=str(number),
                role=role, method=method, label=int(method != 'real'),
                group_id=f'group/{number}', source_graph_certified=dataset == 'FFPP')


def result(candidate, failure=None, decode_failure=False):
    uid = candidate['video_uid']
    frames = []
    if not decode_failure:
        for i in range(8):
            frame = dict(mother_index=i, mother_timestamp=i / 10,
                         mother_rgb_sha256=f'{uid}/mother/{i}',
                         original_detection={'qualified_face_count': 2 if i == 0 else 1, 'multiface': i == 0},
                         a_redetection={'qualified_face_count': 1, 'multiface': False})
            for version in ('A', 'B', 'AB'):
                frame[version] = dict(ok=True, rgb_sha256=f'{uid}/{version}/{i}',
                                      path=f'/{uid}/{version}/{i}.png', padding_ratio={'A': .1, 'B': .2, 'AB': .3}[version],
                                      mask_path=f'/{uid}/{version}/{i}_mask.png', mask_byte_sha256='mask')
            frames.append(frame)
    if failure:
        version, i, reason = failure
        frames[i][version] = dict(ok=False, reason=reason)
        if version == 'AB':
            frames[i].pop('a_redetection')
    eligible = {v: bool(frames) and all(f[v]['ok'] for f in frames) for v in ('A', 'B', 'AB')}
    out = dict(video=copy.deepcopy(candidate), frames=frames, eligible=eligible,
               common_B_AB=eligible['B'] and eligible['AB'])
    if decode_failure:
        out['video_failure'] = 'insufficient_unique_mothers'
    else:
        out['version_failures'] = {v: [f'{i}:{f[v]["reason"]}' for i, f in enumerate(frames) if not f[v]['ok']]
                                   for v in ('A', 'B', 'AB')}
    return out


def test_coverage_preserves_all_b_denominator_and_failure_reasons():
    c = [video(1), video(2, method='Deepfakes'), video(3, method='FaceSwap')]
    m = [result(c[0]), result(c[1], ('AB', 3, 'no_qualified_face')), result(c[2], decode_failure=True)]
    before = copy.deepcopy(m)
    report = module().summarize_coverage(c, m)
    row = report['summary']
    assert row['candidate_videos'] == 3
    assert row['processing_eligibility']['B'] == {'numerator': 2, 'denominator': 3, 'rate': 2/3, 'status': 'OBSERVED'}
    assert row['processing_eligibility']['AB']['numerator'] == 1
    assert row['processing_eligibility']['common_B_AB']['numerator'] == 1
    assert row['rejection_reasons']['AB']['frame_reasons'] == {'no_qualified_face': 1}
    assert row['rejection_reasons']['AB']['video_reasons'] == {'no_qualified_face': 1, 'insufficient_unique_mothers': 1}
    assert row['mother_slots'] == {'prescribed': 24, 'selected': 16, 'unselected': 8, 'per_video': 8}
    assert report['test_cohorts']['FFPP/SOURCE_TEST']['all_B']['numerator'] == 2
    assert report['test_cohorts']['FFPP/SOURCE_TEST']['common_B_AB']['numerator'] == 1
    assert before == m


def test_quarantine_and_unused_are_administrative_not_processing_failures():
    c = [video(1, 'TRAIN'), video(2, 'TRAIN', 'Deepfakes'), video(3, 'SOURCE_UNUSED', 'Face2Face'),
         video(4, 'TARGET_TEST', dataset='DFD')]
    report = module().summarize_coverage(c, [result(v) for v in c], quarantined_video_uids=[c[1]['video_uid']])
    row = report['summary']
    assert row['candidate_videos'] == 4
    assert row['processing_eligibility']['B']['numerator'] == 4
    assert row['administrative_exclusions']['quarantined_videos'] == 1
    assert row['administrative_exclusions']['source_unused_videos'] == 1
    assert row['administrative_exclusions']['excluded_union_videos'] == 2
    assert row['available_cohorts']['B']['numerator'] == 2
    assert row['available_cohorts']['B']['denominator'] == 4
    assert all(v['failed_videos'] == 0 for v in row['rejection_reasons'].values())
    groups = report['coverage_by_role_method_label']
    assert {(x['dataset'], x['role'], x['method'], x['label']) for x in groups} == {
        ('FFPP', 'TRAIN', 'real', 0), ('FFPP', 'TRAIN', 'Deepfakes', 1),
        ('FFPP', 'SOURCE_UNUSED', 'Face2Face', 1), ('DFD', 'TARGET_TEST', 'real', 0)}
    assert all(v['status'] == 'NOT_PROVIDED' and v['natural_videos'] is None for v in report['active_tables'].values())
    assert len(report['active_tables']) == 13


def test_multiface_denominators_and_padding_include_ineligible_video_outputs():
    c = [video(1), video(2, method='Deepfakes')]
    m = [result(c[0]), result(c[1], ('AB', 1, 'no_qualified_face'))]
    m[1]['frames'][0]['AB']['padding_ratio'] = .9
    report = module().summarize_coverage(c, m)
    geometry = report['geometry_summary']
    src, redetect = [geometry['detections'][k] for k in ('original_detection', 'a_redetection')]
    assert src['multiface_frames'] == {'numerator': 2, 'denominator': 16, 'rate': .125, 'status': 'OBSERVED'}
    assert src['any_multiface_videos']['numerator'] == 2
    assert redetect['multiface_frames']['denominator'] == 15
    assert redetect['missing_after_recorded_failure'] == 1
    padding = geometry['padding']['AB']
    assert padding['all_successful_frames']['produced_frames'] == 15
    assert padding['eligible_video_frames']['produced_frames'] == 8
    assert padding['ineligible_video_frames']['produced_frames'] == 7
    assert padding['ineligible_video_frames']['max'] == .9
    assert padding['all_successful_frames']['mask_metadata']['recorded_masks'] == 15
    assert geometry['padding']['A']['all_successful_frames']['mean'] == pytest.approx(.1)
    assert geometry['padding']['AB']['all_successful_frames']['mean'] == pytest.approx(.34)


def test_missing_geometry_metadata_is_explicit_never_a_zero_measurement():
    c = [video(1)]
    m = [result(c[0])]
    for f in m[0]['frames']:
        f.pop('original_detection')
        for v in ('A', 'B', 'AB'):
            f[v].pop('padding_ratio')
            f[v].pop('mask_byte_sha256')
    geometry = module().summarize_coverage(c, m)['geometry_summary']
    detection = geometry['detections']['original_detection']
    assert detection['multiface_frames']['rate'] is None
    assert detection['multiface_frames']['denominator'] == 0
    assert detection['missing_without_failure_evidence'] == 8
    assert detection['metadata_status'] == 'MISSING'
    padding = geometry['padding']['B']['all_successful_frames']
    assert padding['mean'] is None and padding['observed_frames'] == 0
    assert padding['missing_padding_frames'] == 8 and padding['metadata_status'] == 'MISSING'
    assert padding['mask_metadata']['missing_masks'] == 8


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'duplicate_candidate', 'duplicate_manifest', 'identity',
                                     'seven_mothers', 'duplicate_mother', 'imputed_AB', 'wrong_common', 'stale_reasons'])
def test_invalid_or_incomplete_manifest_mapping_is_rejected(mutation):
    c = [video(1)]
    m = [result(c[0])]
    if mutation == 'missing': m = []
    elif mutation == 'extra': m.append(result(video(2)))
    elif mutation == 'duplicate_candidate': c.append(copy.deepcopy(c[0]))
    elif mutation == 'duplicate_manifest': m.append(copy.deepcopy(m[0]))
    elif mutation == 'identity': m[0]['video']['method'] = 'FaceSwap'
    elif mutation == 'seven_mothers': m[0]['frames'].pop()
    elif mutation == 'duplicate_mother': m[0]['frames'][1]['mother_rgb_sha256'] = m[0]['frames'][0]['mother_rgb_sha256']
    elif mutation == 'imputed_AB': m[0]['frames'][1]['AB'] = {'ok': False, 'reason': 'no_qualified_face'}
    elif mutation == 'wrong_common': m[0]['common_B_AB'] = False
    elif mutation == 'stale_reasons': m[0]['version_failures']['AB'] = ['0:no_qualified_face']
    with pytest.raises(ValueError): module().summarize_coverage(c, m)


def tables_for(candidates, manifests):
    from reliability_v2 import geometry_h as g
    tables = g.h_evaluation_rows(copy.deepcopy(manifests))
    for role in ('TRAIN', 'DEV'):
        b, ab = g.h_training_rows(copy.deepcopy(manifests), role)
        tables[f'B/{role.lower()}'], tables[f'AB/{role.lower()}'] = b, ab
    return tables


def test_active_tables_count_natural_videos_and_validate_exact_mother_membership():
    c = [video(1, 'TRAIN'), video(2, method='Deepfakes')]
    m = [result(v) for v in c]
    tables = tables_for(c, m)
    report = module().summarize_coverage(c, m, active_tables=tables)
    assert report['active_tables']['B/train']['natural_videos'] == 1
    assert report['active_tables']['B/train']['mother_rows'] == 8
    assert report['active_tables']['common/AB/SOURCE_TEST']['natural_videos'] == 1
    bad = copy.deepcopy(tables)
    bad['B/SOURCE_TEST'][0]['mother_id'] = 'stale/mother'
    with pytest.raises(ValueError): module().summarize_coverage(c, m, active_tables=bad)
    bad = copy.deepcopy(tables)
    bad['B/SOURCE_TEST'] = bad['B/SOURCE_TEST'][1:]
    with pytest.raises(ValueError): module().summarize_coverage(c, m, active_tables=bad)


def test_duplicate_final_rgb_failure_is_reported_without_frame_failure_imputation():
    c = [video(1)]
    m = [result(c[0])]
    m[0]['frames'][1]['AB']['rgb_sha256'] = m[0]['frames'][0]['AB']['rgb_sha256']
    m[0]['eligible']['AB'] = m[0]['common_B_AB'] = False
    m[0]['version_failures']['AB'] = ['insufficient_unique_final_rgb']
    row = module().summarize_coverage(c, m)['summary']['rejection_reasons']['AB']
    assert row['failed_videos'] == 1
    assert row['frame_reasons'] == {}
    assert row['video_reasons'] == {'insufficient_unique_final_rgb': 1}


def test_train_dev_require_common_success_even_when_b_processing_succeeds():
    c = [video(1, 'TRAIN'), video(2, 'TRAIN', 'Deepfakes'), video(3, 'DEV')]
    m = [result(c[0]), result(c[1], ('AB', 0, 'no_qualified_face')), result(c[2], decode_failure=True)]
    report = module().summarize_coverage(c, m)
    assert report['summary']['processing_eligibility']['B']['numerator'] == 2
    assert report['training_dev_cohorts']['TRAIN']['common_B_AB'] == {
        'numerator': 1, 'denominator': 2, 'rate': .5, 'status': 'OBSERVED'}
    assert report['training_dev_cohorts']['TRAIN']['mother_rows_per_history'] == 8
    assert report['training_dev_cohorts']['DEV']['mother_rows_per_history'] == 0
    assert report['summary']['video_level_failures'] == {
        'videos': 1, 'reasons': {'insufficient_unique_mothers': 1}}


def test_source_and_target_all_b_and_common_are_separate_with_class_counts():
    c = [video(1), video(2, method='FaceSwap'), video(3, 'TARGET_TEST', dataset='DFD'),
         video(4, 'TARGET_TEST', 'DeepFakeDetection', 'DFD')]
    m = [result(c[0]), result(c[1]), result(c[2]), result(c[3], ('AB', 2, 'no_qualified_face'))]
    report = module().summarize_coverage(c, m, active_tables=tables_for(c, m))
    assert report['test_cohorts']['FFPP/SOURCE_TEST']['all_B']['numerator'] == 2
    assert report['test_cohorts']['FFPP/SOURCE_TEST']['common_B_AB']['numerator'] == 2
    assert report['test_cohorts']['DFD/TARGET_TEST']['all_B']['numerator'] == 2
    assert report['test_cohorts']['DFD/TARGET_TEST']['common_B_AB']['numerator'] == 1
    assert report['active_tables']['B/TARGET_TEST']['by_method_label'] == [
        {'method': 'DeepFakeDetection', 'label': 1, 'natural_videos': 1, 'mother_rows': 8},
        {'method': 'real', 'label': 0, 'natural_videos': 1, 'mother_rows': 8}]


@pytest.mark.parametrize('field,value', [('image_path', '/wrong.png'), ('sample_id', 'wrong'), ('group_id', 'other')])
def test_active_table_paths_sample_ids_and_source_groups_cannot_drift(field, value):
    c = [video(1)]
    m = [result(c[0])]
    tables = tables_for(c, m)
    tables['B/SOURCE_TEST'][0][field] = value
    with pytest.raises(ValueError): module().summarize_coverage(c, m, active_tables=tables)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return {'path': str(path), 'sha256': digest(path)}


def wrapper_fixture(tmp_path, monkeypatch):
    from reliability_v2 import data_h, geometry_h
    c = [video(1)]
    m = [result(c[0])]
    out = tmp_path / 'data/H'
    paths = [str(out / 'processed/1.json')]
    item = write(paths[0], m[0])
    identities = {
        'candidates': write(out / 'H_candidates_all.json', c),
        'processed_paths': write(out / 'H_processed_manifest_paths.json', paths),
        'content_audit': write(out / 'H_content_source_audit.json', geometry_h.h_content_audit(copy.deepcopy(m))[0])}
    tables = {}
    for key, rows in tables_for(c, m).items():
        path = out / f'{key}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        tables[key] = {'path': str(path), 'sha256': digest(path), 'rows': len(rows),
                       'videos': len({r['video_uid'] for r in rows})}
    inventory = hashlib.sha256((json.dumps([paths[0], item['sha256']]) + '\n').encode()).hexdigest()
    receipt = {'identities': identities, 'tables': tables, 'candidate_videos': 1,
               'candidate_manifest_inventory_sha256': inventory, 'artifact_inventory_sha256': 'frozen-pixel-inventory',
               'independent_review': write(tmp_path / 'review.json', {'status': 'PASS'}),
               'independence_scope': 'synthetic fixture, no ancestry certification', 'validation_policy': {'default': 'SHALLOW_METADATA_BINDING'}}
    write(out / 'freeze.json', receipt)
    # Only substitute external acceptance for wrapper I/O tests. Production
    # acceptance is separately exercised below with the real unchanged gate.
    monkeypatch.setattr(data_h, 'validate_h_freeze', lambda root: copy.deepcopy(receipt))
    return receipt, paths


def test_wrapper_binds_inputs_all_tables_code_and_does_not_write(tmp_path, monkeypatch):
    receipt, paths = wrapper_fixture(tmp_path, monkeypatch)
    before = {str(p): digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    report = module().report_coverage(tmp_path)
    assert report['summary']['candidate_videos'] == 1
    assert report['provenance']['freeze']['sha256'] == digest(tmp_path / 'data/H/freeze.json')
    assert report['provenance']['input_identities'] == receipt['identities']
    assert report['provenance']['processed_manifest_inventory_sha256'] == receipt['candidate_manifest_inventory_sha256']
    assert report['provenance']['processed_manifests'] == [{'path': paths[0], 'sha256': digest(paths[0])}]
    assert set(report['provenance']['code_dependencies']) == {'coverage_h', 'data_h', 'geometry_h'}
    assert len(report['provenance']['tables']) == 13
    assert report['independence_scope'] == receipt['independence_scope']
    assert report['target_performance_used'] is False
    assert before == {str(p): digest(p) for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mutation', ['table', 'manifest', 'candidates', 'freeze'])
def test_wrapper_rejects_stale_or_corrupt_accepted_inputs(tmp_path, monkeypatch, mutation):
    receipt, paths = wrapper_fixture(tmp_path, monkeypatch)
    path = {'table': receipt['tables']['B/SOURCE_TEST']['path'], 'manifest': paths[0],
            'candidates': receipt['identities']['candidates']['path'], 'freeze': tmp_path / 'data/H/freeze.json'}[mutation]
    Path(path).write_bytes(b'changed after acceptance')
    with pytest.raises(ValueError): module().report_coverage(tmp_path)


def test_production_wrapper_has_no_unfrozen_or_small_cohort_bypass(tmp_path):
    coverage = module()
    with pytest.raises((OSError, ValueError)): coverage.report_coverage(tmp_path)
    write(tmp_path / 'data/H/freeze.json', {'schema': 'H_DATA_FREEZE_V1', 'status': 'FROZEN', 'candidate_videos': 1})
    with pytest.raises(ValueError, match='8431'): coverage.report_coverage(tmp_path)
    with pytest.raises(TypeError): coverage.report_coverage(tmp_path, allow_unfrozen=True)
