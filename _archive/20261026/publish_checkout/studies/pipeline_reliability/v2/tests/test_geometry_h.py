"""H geometry acceptance tests; CPU only and no external model downloads."""
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from reliability_v2 import geometry_h as g


def faces(box=(40, 35, 180, 190), score=.9):
    return g.Face(np.asarray(box, dtype=float), score,
                  np.asarray([[80, 90], [140, 90], [110, 120],
                              [85, 150], [135, 150]], dtype=float))


def mothers(n=8):
    yy, xx = np.mgrid[:240, :240]
    result = []
    for i in range(n):
        rgb = np.stack([(xx + i * 13) % 256, (yy + i * 23) % 256,
                        (xx + yy + i * 31) % 256], axis=-1).astype(np.uint8)
        result.append(g.MotherFrame(i, i / 25, rgb, g.rgb_hash(rgb)))
    return result


def test_alignment_maps_translated_scaled_template_with_expanded_fov():
    # A wrong margin direction, origin, or non-similarity transform breaks this.
    base = np.asarray([[38.2946, 51.6963], [73.5318, 51.5014],
                       [56.0252, 71.7366], [41.5493, 92.3655],
                       [70.7299, 92.2041]])
    expected = 128 + (base * (256 / 112) - 128) / 1.2
    src = base * 2 + [23, 31]
    mat = g.similarity_matrix(src)
    mapped = np.c_[src, np.ones(5)] @ mat.T
    np.testing.assert_allclose(mapped, expected, atol=1e-7)
    np.testing.assert_allclose(mat[:, :2].T @ mat[:, :2],
                               np.eye(2) * ((256 / 112) / 2 / 1.2) ** 2,
                               atol=1e-10)


def test_b_crop_preserves_orientation_and_reports_outside_padding():
    rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
    face = faces((0, 0, 80, 40))
    crop = g.crop_b(rgb, face)
    assert crop.rgb.shape == (224, 224, 3)
    np.testing.assert_allclose(crop.matrix[:, :2], np.eye(2) * (224 / 120))
    assert .25 < crop.padding_ratio < .6
    assert set(np.unique(crop.valid_mask)) <= {0, 255}


def test_ab_redetects_saved_a_png_and_inherits_a_valid_mask(tmp_path):
    # Reusing original bbox instead of invoking detection on the A PNG fails.
    class Detector:
        def detect(self, rgb):
            if rgb.shape[:2] == (256, 256):
                assert np.array_equal(rgb, g.read_rgb(tmp_path / '000' / 'A.png'))
                return [faces((0, 0, 256, 256))]
            return [faces((-30, -20, 210, 210))]
    single = mothers(1)
    out = g.process_frame(single[0], Detector(), tmp_path / '000')
    assert out['AB']['ok']
    a_mask = cv2.imread(str(tmp_path / '000' / 'A_mask.png'), 0)
    ab_mask = cv2.imread(str(tmp_path / '000' / 'AB_mask.png'), 0)
    matrix = np.asarray(out['AB']['matrix'])
    expected = cv2.warpAffine(a_mask, matrix, (224, 224),
                              flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    np.testing.assert_array_equal(ab_mask, expected)
    assert out['AB']['input_rgb_sha256'] == out['A']['rgb_sha256']
    assert out['AB']['padding_ratio'] > 0


def test_same_mothers_all_versions_and_any_one_failure_excludes_video(tmp_path):
    class Detector:
        def detect(self, rgb):
            if rgb.shape[:2] == (256, 256) and int(rgb[128, 128, 0]) > 255:
                raise AssertionError('unreachable')
            return [faces()]
    selected = mothers()
    good = g.process_mothers(selected, Detector(), tmp_path / 'good')
    ids = [row['mother_rgb_sha256'] for row in good['frames']]
    assert ids == [m.rgb_sha256 for m in selected]
    assert all(good['eligible'][v] for v in ('A', 'B', 'AB'))
    class MissingOne:
        def detect(self, rgb):
            if np.array_equal(rgb, selected[3].rgb):
                return []
            return [faces()]
    bad = g.process_mothers(selected, MissingOne(), tmp_path / 'bad')
    assert bad['eligible'] == {'A': False, 'B': False, 'AB': False}
    assert len(bad['frames']) == 8  # no replacement selection after detection
    assert bad['frames'][3]['B']['reason'] == 'no_qualified_face'


def test_unique_rgb_selection_happens_before_face_detection():
    source = mothers(12)
    sequence = [source[0], source[0], *source[1:]]
    selected = g.select_mothers(sequence)
    assert [m.index for m in selected] == [0, 2, 3, 5, 6, 8, 9, 11]
    assert len({m.rgb_sha256 for m in selected}) == 8
    with pytest.raises(g.GeometryFailure, match='insufficient_unique_mothers'):
        g.select_mothers([source[0]] * 100)


def test_crop_collapse_excludes_version_even_when_mothers_are_unique(tmp_path):
    selected = mothers()
    for m in selected:
        m.rgb[30:200, 30:200] = 1
        m.rgb_sha256 = g.rgb_hash(m.rgb)
    class Detector:
        def detect(self, rgb):
            return [faces((70, 70, 140, 140))]
    result = g.process_mothers(selected, Detector(), tmp_path)
    assert not result['eligible']['B']
    assert 'insufficient_unique_final_rgb' in result['version_failures']['B']


def test_selector_uses_area_then_score_then_coordinate_and_ignores_invalid():
    first = faces((0, 0, 10, 10), .8)
    tied = faces((1, 1, 11, 11), .8)
    small = faces((2, 2, 11, 11), .99)
    invalid = faces((0, 0, 0, 200), .99)
    selected, count = g.select_face([tied, small, invalid, first])
    assert count == 3
    np.testing.assert_array_equal(selected.bbox, [0, 0, 10, 10])
    selected, _ = g.select_face([first, faces((1, 1, 11, 11), .9)])
    np.testing.assert_array_equal(selected.bbox, [1, 1, 11, 11])
    with pytest.raises(g.GeometryFailure, match='degenerate_landmarks'):
        g.similarity_matrix(np.zeros((5, 2)))


def test_partition_is_group_disjoint_shared_with_d_and_roles_are_restricted(tmp_path):
    pairs = [[f'{i:03d}', f'{i+360:03d}'] for i in range(360)]
    mapping = g.source_pair_partition(pairs)
    assert sum(v == 'TRAIN' for v in mapping.values()) == 306
    assert sum(v == 'DEV' for v in mapping.values()) == 54
    assert mapping == g.source_pair_partition(list(reversed(pairs)))
    assert mapping['000_360'] == ('TRAIN' if 0 in np.random.default_rng(20261023).permutation(360)[:306] else 'DEV')
    records = [{'video_uid': 'x', 'role': 'TARGET_TEST', 'path': '/none', 'method': 'real'}]
    with pytest.raises(ValueError, match='source TRAIN/DEV'):
        g.source_qualification_sample(records)
    with pytest.raises(ValueError, match='source TRAIN/DEV'):
        g.require_source_qualification_role('SOURCE_TEST')


def test_letterbox_rgb_normalization_and_nms_tie_are_deterministic():
    rgb = np.zeros((100, 200, 3), np.uint8)
    rgb[..., 0] = 255
    blob, transform = g.detector_blob(rgb)
    assert blob.shape == (1, 3, 640, 640)
    np.testing.assert_allclose(blob[0, :, 0, 0], [127.5 / 128, -127.5 / 128, -127.5 / 128])
    # Small A PNGs must remain at native size: upscaling to640 made all ten
    # source qualification videos fail AB redetection (model scale limit).
    assert transform['scale_x'] == 1 and transform['scale_y'] == 1
    assert np.all(blob[0, :, 400, 0] == -127.5 / 128)
    dets = np.asarray([[0, 0, 10, 10, .9], [1, 1, 11, 11, .9], [30, 30, 40, 40, .8]])
    assert g.nms_indices(dets) == [0, 2]


def test_no_target_decode_before_source_geometry_freeze(tmp_path):
    record = {'role': 'TARGET_TEST', 'path': '/not_allowed_to_read'}
    with pytest.raises(ValueError, match='frozen'):
        g.process_record(record, object(), tmp_path)
    assert not list(tmp_path.iterdir())


def test_nonfinite_keypoints_record_version_failure_in_valid_json(tmp_path):
    face = faces()
    face.landmarks[0, 0] = np.nan
    class Detector:
        def detect(self, rgb):
            return [face]
    result = g.process_mothers(mothers(), Detector(), tmp_path)
    assert not result['eligible']['A'] and not result['eligible']['AB']
    assert result['eligible']['B']
    # Invalid detector values must remain reportable, never abort JSON output.
    g.write_json(tmp_path / 'failure.json', result)
    assert json.loads((tmp_path / 'failure.json').read_text())['frames'][0]['A']['reason'] == 'nonfinite_or_missing_landmarks'


def test_full_decode_fallback_finds_unique_frames_before_detection(tmp_path):
    video = tmp_path / 'short_unique_burst.avi'
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'FFV1'), 25., (64, 64))
    assert writer.isOpened()
    # Uniform32samples miss many of the eight unique early frames. A full
    # decode must discover them before selecting any detector inputs.
    for i in range(100):
        frame = np.full((64, 64, 3), i * 20 if 1 <= i <= 8 else 0, np.uint8)
        writer.write(frame)
    writer.release()
    selected, report = g.decode_mothers(video)
    assert report['strategy'] == 'full_chronological_decode_fallback'
    assert len(selected) == len({m.rgb_sha256 for m in selected}) == 8
    assert [m.index for m in selected] == sorted(m.index for m in selected)


def h_result(role, method, video_id, mother_prefix=None):
    uid = f'FFPP/source/{method}/{video_id}'
    return {'video': {'role': role, 'method': method, 'label': int(method != 'real'),
                      'video_uid': uid, 'video_id': video_id, 'group_id': video_id,
                      'domain': 'source', 'source_ids': [video_id]},
            'eligible': {'A': True, 'B': True, 'AB': True},
            'common_B_AB': True,
            'frames': [{'mother_rgb_sha256': f'{mother_prefix or uid}-{i}',
                        'mother_index': i, 'mother_timestamp': i / 25,
                        'B': {'ok': True, 'path': f'/B/{uid}/{i}.png', 'rgb_sha256': f'B-{uid}-{i}'},
                        'AB': {'ok': True, 'path': f'/AB/{uid}/{i}.png', 'rgb_sha256': f'AB-{uid}-{i}'}}
                       for i in range(8)]}


def test_h_paired_rows_preserve_logical_slots_and_exact_method_sampling_mass():
    results = [h_result('TRAIN', method, str(i)) for i, method in
               enumerate(('real', *g.METHODS))]
    results.append(h_result('TRAIN', 'real', 'second_real'))
    b, ab = g.h_training_rows(results, 'TRAIN')
    assert len(b) == len(ab) == 48
    assert [row['mother_id'] for row in b] == [row['mother_id'] for row in ab]
    assert [row['sample_id'] for row in b] == [row['sample_id'] for row in ab]
    assert [row['sampling_mass'] for row in b] == [row['sampling_mass'] for row in ab]
    assert np.isclose(sum(row['sampling_mass'] for row in b), 1)
    assert np.isclose(sum(row['sampling_mass'] for row in b if row['label'] == 0), .5)
    for method in g.METHODS:
        assert np.isclose(sum(row['sampling_mass'] for row in b if row['method_id'] == method), .125)
    for row in b:
        assert json.loads(row['method_contributions']) == {row['method_id']: row['sampling_mass']}


def test_h_content_audit_blocks_cross_role_and_true_fake_rgb_conflicts():
    results = [h_result('TRAIN', 'real', 'a', 'same'),
               h_result('DEV', 'Deepfakes', 'b', 'same')]
    report, blocked = g.h_content_audit(results)
    assert set(blocked) == {'a', 'b'}
    assert report['cross_role_rgb_conflicts']
    assert report['label_rgb_conflicts']
    b, ab = g.h_training_rows(results, 'TRAIN', blocked)
    assert b == ab == []


def test_h_b_deployment_coverage_is_independent_of_ab_success():
    result = h_result('TARGET_TEST', 'real', 'target')
    result['eligible']['AB'] = False
    result['common_B_AB'] = False
    tables = g.h_evaluation_rows([result])
    assert len(tables['B/TARGET_TEST']) == 8
    assert tables['common/B/TARGET_TEST'] == tables['common/AB/TARGET_TEST'] == []


def test_unused_val_fake_rgb_bridge_quarantines_cal_train_closure_without_exposure():
    train = h_result('TRAIN', 'Deepfakes', 'train', 'bridge')
    unused = h_result('SOURCE_UNUSED', 'Face2Face', 'unused', 'bridge')
    calibration = h_result('SOURCE_CAL', 'real', 'cal', 'calibration')
    unused['video']['group_id'] = 'cal'  # official known val source pair
    results = [train, unused, calibration]
    report, blocked = g.h_content_audit(results)
    assert blocked == {'cal', 'train'}
    assert report['audit_schema'] == 'FULL_CANDIDATE_GRAPH_INCLUDING_SOURCE_UNUSED_V2'
    assert list(report['source_group_role_conflicts'].values()) == [['SOURCE_CAL', 'TRAIN']]
    assert all('SOURCE_UNUSED' not in roles for roles in report['source_group_role_conflicts'].values())
    assert g.h_training_rows(results, 'TRAIN', blocked) == ([], [])
    assert all(not rows for rows in g.h_evaluation_rows(results, blocked).values())
    unquarantined_tables = g.h_evaluation_rows(results)
    assert all(row['role'] != 'SOURCE_UNUSED' for rows in unquarantined_tables.values() for row in rows)


def test_unused_role_only_adds_graph_edges_without_an_extra_active_role():
    train = h_result('TRAIN', 'Deepfakes', 'train', 'bridge')
    unused = h_result('SOURCE_UNUSED', 'Face2Face', 'unused', 'bridge')
    report, blocked = g.h_content_audit([train, unused])
    assert not blocked
    assert not report['cross_role_rgb_conflicts']
    assert not report['source_group_role_conflicts']
    assert train['video']['group_id'] == unused['video']['group_id']


def test_fresh_audit_rejects_incomplete_candidate_manifest_set(tmp_path):
    artifact = tmp_path / 'data'
    artifact.mkdir()
    record = h_result('TRAIN', 'real', 'one')
    record['protocol_sha256'] = g.protocol_hash()
    g.write_json(artifact / 'H_candidates_all.json',
                 [record['video'], {'video_uid': 'missing/candidate'}])
    g.write_json(tmp_path / 'one.json', record)
    g.write_json(artifact / 'H_processed_manifest_paths.json', [str(tmp_path / 'one.json')])
    with pytest.raises(ValueError, match='complete candidate'):
        g.audit_existing(artifact / 'H_processed_manifest_paths.json', artifact)
    assert not (artifact / 'H_full_candidate_audit_receipt.json').exists()


def cached_pixel_record(tmp_path):
    video_path = tmp_path / 'source.mp4'
    video_path.write_bytes(b'controlled-source-video-byte-identity')
    image_root = tmp_path / 'images'
    directory = image_root / 'FFPP/TRAIN/real/cached'
    class Detector:
        def detect(self, rgb):
            return [faces()]
    result = g.process_mothers(mothers(), Detector(), directory)
    result['video'] = {'dataset': 'FFPP', 'role': 'TRAIN', 'method': 'real',
                       'video_id': 'cached', 'video_uid': 'FFPP/source/real/cached',
                       'path': str(video_path), 'group_id': 'source-pair',
                       'label': 0, 'domain': 'source'}
    result.update(protocol_sha256=g.protocol_hash(),
                  video_byte_sha256=g.sha256_file(video_path), elapsed_seconds=.1)
    g.write_json(directory / 'manifest.json', result)
    return result, image_root, directory


def test_resume_refuses_missing_successful_derived_png(tmp_path):
    result, image_root, directory = cached_pixel_record(tmp_path)
    Path(result['frames'][0]['B']['path']).unlink()
    with pytest.raises(g.GeometryFailure, match='artifact_integrity'):
        g._worker_record((result['video'], image_root, tmp_path / 'freeze.json'))


@pytest.mark.parametrize('field', ['mother', 'crop', 'mask'])
def test_artifact_integrity_rejects_changed_png_bytes(tmp_path, field):
    result, _, _ = cached_pixel_record(tmp_path)
    frame = result['frames'][0]
    path = {'mother': frame['mother_path'], 'crop': frame['AB']['path'],
            'mask': frame['AB']['mask_path']}[field]
    with Path(path).open('ab') as file:
        file.write(b'changed-byte-content')
    with pytest.raises(g.GeometryFailure, match='artifact_integrity'):
        g.verify_record_artifacts(result)


def test_artifact_integrity_checks_rgb_hash_and_binds_table_byte_hash(tmp_path):
    result, _, _ = cached_pixel_record(tmp_path)
    summary = g.verify_record_artifacts(result)
    assert summary['verified_files'] == 56  #8mothers+24crops+24masks
    row = g.h_frame_rows(result, 'B')[0]
    assert row['content_sha256'] == result['frames'][0]['B']['byte_sha256']
    result['frames'][0]['B']['rgb_sha256'] = '0' * 64
    with pytest.raises(g.GeometryFailure, match='artifact_integrity.*RGB'):
        g.verify_record_artifacts(result)

