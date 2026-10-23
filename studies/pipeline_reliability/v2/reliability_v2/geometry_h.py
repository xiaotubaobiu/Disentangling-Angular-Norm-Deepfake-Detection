"""H c23 mother selection, SCRFD-only geometry, and source qualification.

No recognition models, network requests, or implicit weight downloads. Inputs
and detector calls are RGB. Output matrices map input pixel centres to output
pixel centres; all image warps are OpenCV bicubic with constant-zero padding.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import time
from typing import Iterable

import cv2
import numpy as np

WEIGHT = '/home/z/.insightface/models/buffalo_l/det_10g.onnx'
WEIGHT_SHA256 = '5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91'
DATA_ROOT = Path('/Dataset/deepfake_detection/FaceForensics++')
SPLIT_ROOT = Path('/home/z/project/deepfake_detector/artifacts/data/source_splits')
ARTIFACT_ROOT = Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
METHODS = ('Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures')
TEMPLATE_112 = np.asarray([[38.2946, 51.6963], [73.5318, 51.5014],
                           [56.0252, 71.7366], [41.5493, 92.3655],
                           [70.7299, 92.2041]], dtype=np.float64)
VERSIONS = ('A', 'B', 'AB')


class GeometryFailure(ValueError):
    """A prespecified, reportable video/frame eligibility failure."""


@dataclass
class Face:
    bbox: np.ndarray
    score: float
    landmarks: np.ndarray


@dataclass
class MotherFrame:
    index: int
    timestamp: float
    rgb: np.ndarray
    rgb_sha256: str


@dataclass
class Crop:
    rgb: np.ndarray
    valid_mask: np.ndarray
    matrix: np.ndarray

    @property
    def padding_ratio(self):
        return float(1 - np.count_nonzero(self.valid_mask) / self.valid_mask.size)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def rgb_hash(rgb):
    array = np.ascontiguousarray(rgb, dtype=np.uint8)
    return hashlib.sha256(str(array.shape).encode() + b'\0' + array.tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise GeometryFailure('png_read_failed')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_crop(crop, directory, version):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path, mask_path = directory / f'{version}.png', directory / f'{version}_mask.png'
    params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(str(path), cv2.cvtColor(crop.rgb, cv2.COLOR_RGB2BGR), params):
        raise GeometryFailure('png_write_failed')
    if not cv2.imwrite(str(mask_path), crop.valid_mask, params):
        raise GeometryFailure('mask_write_failed')
    return {'ok': True, 'path': str(path), 'mask_path': str(mask_path),
            'rgb_sha256': rgb_hash(crop.rgb), 'byte_sha256': sha256_file(path),
            'mask_byte_sha256': sha256_file(mask_path),
            'png_bytes': path.stat().st_size, 'mask_bytes': mask_path.stat().st_size,
            'padding_ratio': crop.padding_ratio, 'matrix': crop.matrix.tolist()}


def select_face(faces: Iterable[Face], threshold=.5):
    candidates = []
    for face in faces:
        box = np.asarray(face.bbox, dtype=np.float64)
        if (box.shape == (4,) and np.all(np.isfinite(box))
                and np.isfinite(face.score) and face.score >= threshold
                and box[2] > box[0] and box[3] > box[1]):
            candidates.append(face)
    if not candidates:
        raise GeometryFailure('no_qualified_face')
    def key(face):
        x1, y1, x2, y2 = face.bbox
        # Landmarks resolve otherwise exact ties without detector output order.
        kps = np.nan_to_num(face.landmarks, nan=np.inf).ravel().tolist()
        return (-(x2 - x1) * (y2 - y1), -face.score, x1, y1, x2, y2, *kps)
    return sorted(candidates, key=key)[0], len(candidates)


def similarity_matrix(landmarks):
    source = np.asarray(landmarks, dtype=np.float64)
    if source.shape != (5, 2) or not np.all(np.isfinite(source)):
        raise GeometryFailure('nonfinite_or_missing_landmarks')
    if np.linalg.matrix_rank(source - source.mean(axis=0), tol=1e-8) < 2:
        raise GeometryFailure('degenerate_landmarks')
    target = 128 + (TEMPLATE_112 * (256 / 112) - 128) / 1.2
    design = np.zeros((10, 4), dtype=np.float64)
    design[0::2, 0] = source[:, 0]
    design[0::2, 1] = -source[:, 1]
    design[0::2, 2] = 1
    design[1::2, 0] = source[:, 1]
    design[1::2, 1] = source[:, 0]
    design[1::2, 3] = 1
    a, b, tx, ty = np.linalg.lstsq(design, target.ravel(), rcond=None)[0]
    if not np.all(np.isfinite([a, b, tx, ty])) or a * a + b * b < 1e-12:
        raise GeometryFailure('degenerate_affine')
    return np.asarray([[a, -b, tx], [b, a, ty]])


def warp(rgb, matrix, size, valid_mask=None):
    if valid_mask is None:
        valid_mask = np.full(rgb.shape[:2], 255, np.uint8)
    output = cv2.warpAffine(rgb, matrix, (size, size), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.warpAffine(valid_mask, matrix, (size, size), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return Crop(output, mask, matrix)


def crop_b(rgb, face, valid_mask=None):
    x1, y1, x2, y2 = np.asarray(face.bbox, dtype=np.float64)
    side = 1.5 * max(x2 - x1, y2 - y1)
    if not np.isfinite(side) or side <= 0:
        raise GeometryFailure('zero_or_nonfinite_bbox')
    scale = 224 / side
    # Output centre111.5: pixel-centre convention, unrotated similarity.
    matrix = np.asarray([[scale, 0, 111.5 - scale * (x1 + x2) / 2],
                         [0, scale, 111.5 - scale * (y1 + y2) / 2]])
    return warp(rgb, matrix, 224, valid_mask)


def select_mothers(frames, count=8):
    unique, seen = [], set()
    for frame in sorted(frames, key=lambda row: (row.timestamp, row.index)):
        if frame.rgb_sha256 not in seen:
            unique.append(frame)
            seen.add(frame.rgb_sha256)
    if len(unique) < count:
        raise GeometryFailure('insufficient_unique_mothers')
    positions = np.rint(np.linspace(0, len(unique) - 1, count)).astype(int)
    return [unique[i] for i in positions]


def full_decode_mothers(path):
    # First pass keeps only metadata and hashes, bounding RGB memory even for
    # long high-resolution clips. Second chronological pass retains selected8.
    capture = cv2.VideoCapture(str(path))
    unique, seen, decoded_count = [], set(), 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        digest = rgb_hash(rgb)
        if digest not in seen:
            seen.add(digest)
            unique.append((decoded_count, float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000, digest))
        decoded_count += 1
    capture.release()
    if len(unique) < 8:
        raise GeometryFailure('insufficient_unique_mothers')
    positions = np.rint(np.linspace(0, len(unique) - 1, 8)).astype(int)
    wanted = {unique[i][0]: unique[i] for i in positions}
    capture = cv2.VideoCapture(str(path))
    selected, index = [], 0
    while index <= max(wanted):
        ok, bgr = capture.read()
        if not ok:
            capture.release()
            raise GeometryFailure('selected_mother_decode_failed')
        if index in wanted:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _, timestamp, digest = wanted[index]
            if rgb_hash(rgb) != digest:
                capture.release()
                raise GeometryFailure('decode_rgb_not_reproducible')
            selected.append(MotherFrame(index, timestamp, rgb, digest))
        index += 1
    capture.release()
    return selected, decoded_count, len(unique)


def decode_mothers(path):
    """Uniform-time <=32 candidates, RGB dedup, then chronological full fallback.

    Selection is finished before any detector operation. OpenCV's decoded PTS
    and frame ordinal are retained; ffprobe supplies duration for equal-time
    seeks. The same algorithm applies to every domain and label.
    """
    started = time.monotonic()
    probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                            '-show_entries', 'stream=duration,avg_frame_rate,nb_frames:format=duration',
                            '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    info = json.loads(probe.stdout)
    stream = info['streams'][0]
    duration = float(stream.get('duration', info.get('format', {}).get('duration', 0)))
    if not np.isfinite(duration) or duration <= 0:
        raise GeometryFailure('invalid_video_duration')
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise GeometryFailure('decode_open_failed')
    fps = capture.get(cv2.CAP_PROP_FPS)
    end = max(0., duration - (1 / fps if fps > 0 else .001))
    candidates, failures = [], []
    for timestamp in np.linspace(0, end, 32):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
        ok, bgr = capture.read()
        if not ok:
            failures.append(float(timestamp))
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        index = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        pts = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        candidates.append(MotherFrame(index, pts, rgb, rgb_hash(rgb)))
    capture.release()
    strategy = 'uniform_time_candidates_32'
    candidate_count = len(candidates)
    unique_count = len({row.rgb_sha256 for row in candidates})
    try:
        selected = select_mothers(candidates)
    except GeometryFailure:
        strategy = 'full_chronological_decode_fallback'
        candidates.clear()
        selected, candidate_count, unique_count = full_decode_mothers(path)
    return selected, {'strategy': strategy, 'candidate_count': candidate_count,
                      'unique_candidate_rgb': unique_count,
                      'candidate_decode_failure_timestamps': failures,
                      'duration_seconds': duration, 'decode_seconds': time.monotonic() - started,
                      'selected_frames': [{'frame_index': m.index, 'timestamp_seconds': m.timestamp,
                                           'rgb_sha256': m.rgb_sha256} for m in selected]}


def face_json(face, count):
    points = np.asarray(face.landmarks)
    valid = points.shape == (5, 2) and bool(np.all(np.isfinite(points)))
    return {'bbox': np.asarray(face.bbox).tolist(), 'score': float(face.score),
            'landmarks': points.tolist() if valid else None, 'landmarks_finite': valid,
            'qualified_face_count': count, 'multiface': count > 1}


def process_frame(mother, detector, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    mother_path = directory / 'mother.png'
    if not cv2.imwrite(str(mother_path), cv2.cvtColor(mother.rgb, cv2.COLOR_RGB2BGR),
                       [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise GeometryFailure('mother_png_write_failed')
    result = {'mother_index': mother.index, 'mother_timestamp': mother.timestamp,
              'mother_rgb_sha256': mother.rgb_sha256, 'mother_path': str(mother_path),
              'mother_png_bytes': mother_path.stat().st_size,
              'mother_byte_sha256': sha256_file(mother_path)}
    try:
        face, count = select_face(detector.detect(mother.rgb))
        result['original_detection'] = face_json(face, count)
    except (GeometryFailure, ValueError) as exc:
        for version in VERSIONS:
            result[version] = {'ok': False, 'reason': str(exc)}
        return result
    try:
        result['B'] = save_crop(crop_b(mother.rgb, face), directory, 'B')
        result['B']['input_rgb_sha256'] = mother.rgb_sha256
    except (GeometryFailure, ValueError, cv2.error) as exc:
        result['B'] = {'ok': False, 'reason': str(exc)}
    try:
        aligned = warp(mother.rgb, similarity_matrix(face.landmarks), 256)
        result['A'] = save_crop(aligned, directory, 'A')
        result['A']['input_rgb_sha256'] = mother.rgb_sha256
    except (GeometryFailure, ValueError, cv2.error) as exc:
        result['A'] = {'ok': False, 'reason': str(exc)}
        result['AB'] = {'ok': False, 'reason': 'A_failed'}
        return result
    try:
        # Deliberate disk read of the lossless A PNG: original coords are never reused.
        a_rgb = read_rgb(result['A']['path'])
        a_face, a_count = select_face(detector.detect(a_rgb))
        result['a_redetection'] = face_json(a_face, a_count)
        result['AB'] = save_crop(crop_b(a_rgb, a_face, aligned.valid_mask), directory, 'AB')
        result['AB']['input_rgb_sha256'] = rgb_hash(a_rgb)
        result['AB']['parent_byte_sha256'] = result['A']['byte_sha256']
    except (GeometryFailure, ValueError, cv2.error) as exc:
        result['AB'] = {'ok': False, 'reason': str(exc)}
    return result


def process_mothers(mothers, detector, output):
    if len(mothers) != 8 or len({m.rgb_sha256 for m in mothers}) != 8:
        raise GeometryFailure('insufficient_unique_mothers')
    started = time.monotonic()
    frames = [process_frame(mother, detector, Path(output) / f'{i:03d}')
              for i, mother in enumerate(mothers)]
    eligible, failures = {}, {}
    for version in VERSIONS:
        errors = [f'{i}:{frame[version]["reason"]}' for i, frame in enumerate(frames)
                  if not frame[version]['ok']]
        hashes = [frame[version]['rgb_sha256'] for frame in frames if frame[version]['ok']]
        if len(hashes) == 8 and len(set(hashes)) < 8:
            errors.append('insufficient_unique_final_rgb')
        failures[version] = errors
        eligible[version] = not errors
    return {'frames': frames, 'eligible': eligible, 'version_failures': failures,
            'common_B_AB': eligible['B'] and eligible['AB'],
            'processing_seconds': time.monotonic() - started}


def detector_blob(rgb):
    height, width = rgb.shape[:2]
    scale = min(1., 640 / width, 640 / height)
    new_width, new_height = max(1, int(width * scale)), max(1, int(height * scale))
    resized = cv2.resize(rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((640, 640, 3), np.uint8)
    canvas[:new_height, :new_width] = resized
    # RGB enters directly; no second swapRB (cv2 reference starts from BGR).
    blob = ((canvas.astype(np.float32) - 127.5) / 128).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(blob), {'scale_x': new_width / width,
                                      'scale_y': new_height / height,
                                      'new_width': new_width, 'new_height': new_height,
                                      'letterbox_origin': [0, 0]}


def nms_indices(dets, threshold=.4):
    if len(dets) == 0:
        return []
    order = sorted(range(len(dets)), key=lambda i: (-dets[i, 4], *dets[i, :4], i))
    keep = []
    areas = (dets[:, 2] - dets[:, 0] + 1) * (dets[:, 3] - dets[:, 1] + 1)
    while order:
        current, rest = order[0], order[1:]
        keep.append(current)
        if not rest:
            break
        left = np.maximum(dets[current, :2], dets[rest, :2])
        right = np.minimum(dets[current, 2:4], dets[rest, 2:4])
        intersection = np.prod(np.maximum(0, right - left + 1), axis=1)
        overlap = intersection / (areas[current] + areas[rest] - intersection)
        order = [index for index, iou in zip(rest, overlap) if iou <= threshold]
    return keep


class SCRFD5:
    """Minimal pinned local SCRFD 9-output CPU adapter, no model-zoo imports."""
    def __init__(self, weight=WEIGHT, threads=2):
        if sha256_file(weight) != WEIGHT_SHA256:
            raise ValueError('SCRFD weight SHA256 mismatch')
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(weight), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        self.outputs = [row.name for row in self.session.get_outputs()]
        if len(self.outputs) != 9:
            raise ValueError('Expected pinned 3-stride SCRFD five-point 9-output model')
        self.calls, self.seconds = 0, 0.

    def detect(self, rgb):
        started = time.monotonic()
        blob, transform = detector_blob(rgb)
        outputs = self.session.run(self.outputs, {self.input: blob})
        all_boxes, all_points, all_scores = [], [], []
        for i, stride in enumerate((8, 16, 32)):
            scores = np.asarray(outputs[i]).reshape(-1)
            distances = np.asarray(outputs[i + 3]).reshape(-1, 4) * stride
            offsets = np.asarray(outputs[i + 6]).reshape(-1, 5, 2) * stride
            centres = np.stack(np.mgrid[:640 // stride, :640 // stride][::-1], axis=-1)
            centres = np.repeat(centres.reshape(-1, 2) * stride, 2, axis=0)
            selected = np.flatnonzero(scores >= .5)
            centres, distances, offsets = centres[selected], distances[selected], offsets[selected]
            boxes = np.c_[centres - distances[:, :2], centres + distances[:, 2:]]
            points = centres[:, None, :] + offsets
            boxes[:, [0, 2]] /= transform['scale_x']
            boxes[:, [1, 3]] /= transform['scale_y']
            points[:, :, 0] /= transform['scale_x']
            points[:, :, 1] /= transform['scale_y']
            all_boxes.append(boxes)
            all_points.append(points)
            all_scores.append(scores[selected])
        boxes, points, scores = np.vstack(all_boxes), np.vstack(all_points), np.concatenate(all_scores)
        detections = np.c_[boxes, scores]
        keep = nms_indices(detections)
        self.calls += 1
        self.seconds += time.monotonic() - started
        return [Face(boxes[i], float(scores[i]), points[i]) for i in keep]


def source_pair_partition(pairs):
    normalized = sorted({tuple(sorted(map(str, pair))) for pair in pairs})
    if len(normalized) != 360 or any(len(pair) != 2 for pair in normalized):
        raise ValueError('Expected360 unique official TRAIN pairs')
    permutation = np.random.default_rng(20261023).permutation(360)
    train = set(permutation[:306].tolist())
    return {'_'.join(pair): ('TRAIN' if i in train else 'DEV')
            for i, pair in enumerate(normalized)}


def build_manifest(dataset_root=DATA_ROOT, split_root=SPLIT_ROOT):
    dataset_root, split_root = Path(dataset_root), Path(split_root)
    pairs = {name: json.loads((split_root / f'{name}.json').read_text())
             for name in ('train', 'val', 'test')}
    mapping = source_pair_partition(pairs['train'])
    id_info, conflicts = {}, []
    for split, items in pairs.items():
        for pair in items:
            group = '_'.join(sorted(pair))
            role = mapping[group] if split == 'train' else {'val': 'SOURCE_CAL', 'test': 'SOURCE_TEST'}[split]
            for sid in pair:
                if sid in id_info:
                    conflicts.append({'source_id': sid, 'previous': id_info[sid], 'new': [role, group]})
                id_info[sid] = (role, group)
    records, exclusions = [], []
    real_root = dataset_root / 'original_sequences/youtube/c23/videos'
    for sid, (role, group) in sorted(id_info.items()):
        path = real_root / f'{sid}.mp4'
        row = {'dataset': 'FFPP', 'domain': 'source', 'method': 'real', 'label': 0,
               'video_id': sid, 'video_uid': f'FFPP/source/real/{sid}', 'role': role,
               'source_group': f'FFPP/pair/{group}', 'source_ids': [sid],
               'source_evidence': 'official_ffpp_pair', 'path': str(path)}
        (records if path.is_file() else exclusions).append(row if path.is_file() else {**row, 'reason': 'missing_video'})
    for method in METHODS:
        directory = dataset_root / f'manipulated_sequences/{method}/c23/videos'
        for path in sorted(directory.glob('*.mp4')):
            ids = path.stem.split('_')
            if len(ids) != 2 or any(sid not in id_info for sid in ids):
                exclusions.append({'path': str(path), 'reason': 'unknown_source_ids'})
                continue
            roles = {id_info[sid][0] for sid in ids}
            groups = {id_info[sid][1] for sid in ids}
            if len(roles) != 1 or len(groups) != 1:
                exclusions.append({'path': str(path), 'reason': 'source_pair_or_role_conflict', 'source_ids': ids})
                continue
            role = next(iter(roles))
            if role == 'SOURCE_CAL':
                exclusions.append({'path': str(path), 'reason': 'official_val_fake_not_used'})
                continue
            records.append({'dataset': 'FFPP', 'domain': 'source', 'method': method, 'label': 1,
                            'video_id': path.stem, 'video_uid': f'FFPP/source/{method}/{path.stem}',
                            'role': role, 'source_group': f'FFPP/pair/{next(iter(groups))}',
                            'source_ids': ids, 'source_evidence': 'official_ffpp_pair_and_filename',
                            'path': str(path)})
    dfd_real = dataset_root / 'original_sequences/actors/c23/videos'
    real_ids = {path.stem for path in dfd_real.glob('*.mp4')}
    for label, method, directory in [(0, 'real', dfd_real),
             (1, 'DeepFakeDetection', dataset_root / 'manipulated_sequences/DeepFakeDetection/c23/videos')]:
        for path in sorted(directory.glob('*.mp4')):
            candidates = []
            if label == 0:
                candidates = [path.stem]
            elif '__' in path.stem:
                actors, scene = path.stem.split('__', 1)
                scene = scene.rsplit('__', 1)[0]
                candidates = [f'{actor}__{scene}' for actor in actors.split('_')
                              if f'{actor}__{scene}' in real_ids]
            records.append({'dataset': 'DFD', 'domain': 'target', 'method': method, 'label': label,
                            'video_id': path.stem, 'video_uid': f'DFD/target/{method}/{path.stem}',
                            'role': 'TARGET_TEST', 'source_group': None, 'source_ids': candidates,
                            'source_evidence': 'filename_candidates_unverified', 'unknown_ancestors': True,
                            'path': str(path)})
    counts = Counter((row['role'], row['method']) for row in records)
    audit = {'counts': {f'{role}/{method}': count for (role, method), count in sorted(counts.items())},
             'total_manifest_videos': len(records), 'exclusion_reasons': dict(Counter(row['reason'] for row in exclusions)),
             'source_id_conflicts': conflicts, 'split_hashes': {name: sha256_file(split_root / f'{name}.json') for name in pairs},
             'pair_partition_algorithm': 'sorted normalized official pairs; default_rng(20261023).permutation; first306TRAIN',
             'dfd_fake_candidate_parent_counts': dict(Counter(str(len(row['source_ids'])) for row in records
                       if row['dataset'] == 'DFD' and row['label'] == 1)),
             'dfd_ancestor_certification': False,
             'target_videos_decoded_or_detected': 0}
    return records, mapping, exclusions, audit


def require_source_qualification_role(role):
    if role not in ('TRAIN', 'DEV'):
        raise ValueError('Geometry qualification permits only source TRAIN/DEV')


def source_qualification_sample(records, per_cell=1):
    groups = defaultdict(list)
    for row in records:
        require_source_qualification_role(row['role'])
        groups[(row['role'], row['method'])].append(row)
    selected = []
    for key in sorted(groups):
        selected.extend(sorted(groups[key], key=lambda row: hashlib.sha256(
            ('20261023/' + row['video_uid']).encode()).hexdigest())[:per_cell])
    return selected


def protocol():
    return {'schema': 'H_geometry_v2.1', 'detector': 'SCRFD5 local detection only',
            'weight_path': WEIGHT, 'weight_sha256': WEIGHT_SHA256,
            'detector_input': [640, 640], 'detector_score_threshold': .5,
            'detector_nms_iou': .4, 'detector_nms_boxes': 'inclusive pixel areas +1; stable score/coordinate order',
            'detector_letterbox': 'no upscale min(1,640/width,640/height); top-left640 zero; int floor new dimensions; exact x/y inverse scales',
            'detector_normalization': 'RGB (pixel-127.5)/128',
            'detector_resize': 'OpenCV INTER_LINEAR', 'mother_count': 8,
            'mother_selection': '32 uniform-time decoded candidates; dedup RGB then rounded linspace8; full chronological fallback if <8',
            'rgb_hash_definition': 'sha256(str(shape).encode()+NUL+contiguous_uint8_RGB)',
            'A_size': 256, 'A_margin': 1.2, 'A_template112': TEMPLATE_112.tolist(),
            'A_template256_formula': '128+(template112*(256/112)-128)/1.2',
            'A_transform': 'least-squares orientation-preserving2D similarity; no RANSAC',
            'B_size': 224, 'B_side': '1.5*max(bbox_width,bbox_height)',
            'B_transform': 'unrotated pixel-centre affine; bbox centre maps to111.5',
            'AB_input': 'disk-decoded A PNG; independent detector call and same B algorithm',
            'interpolation': 'OpenCV INTER_CUBIC', 'opencv_version': cv2.__version__,
            'padding': 'BORDER_CONSTANT zero', 'mask': 'binary valid pixel-centre mapping INTER_NEAREST; AB inherits A mask',
            'face_selector': 'max bbox area, max confidence, lexicographic bbox then landmarks',
            'failures': 'any selected frame failure excludes version; <8unique final RGB excludes version; no replacement',
            'padding_exclusion_threshold': None, 'source_qualification_roles': ['TRAIN', 'DEV'],
            'target_tuning_allowed': False, 'DFD_selector_scope': 'largest detected face; manipulated face not certified'}


def protocol_hash():
    return hashlib.sha256(json.dumps(protocol(), sort_keys=True).encode()).hexdigest()


def process_record(record, detector, output, freeze=None):
    if record['role'] not in ('TRAIN', 'DEV'):
        if freeze is None or json.loads(Path(freeze).read_text()).get('protocol_sha256') != protocol_hash():
            raise ValueError('Source geometry must be frozen before non-TRAIN/DEV preprocessing')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        mothers, decode = decode_mothers(record['path'])
        result = process_mothers(mothers, detector, output)
        result['decode'] = decode
    except (GeometryFailure, ValueError, subprocess.CalledProcessError, cv2.error) as exc:
        result = {'eligible': {v: False for v in VERSIONS}, 'common_B_AB': False,
                  'frames': [], 'video_failure': str(exc)}
    result.update({'video': record, 'video_byte_sha256': sha256_file(record['path']),
                   'protocol_sha256': protocol_hash(), 'elapsed_seconds': time.monotonic() - started})
    write_json(output / 'manifest.json', result)
    return result


def qualify_source(records, output, detector, per_cell=1):
    output = Path(output)
    sample = source_qualification_sample(records, per_cell)
    if not sample:
        raise ValueError('No source qualification records')
    write_json(output / 'H_geometry_protocol.json', protocol())
    write_json(output / 'H_source_qualification_sample.json', sample)
    results, started = [], time.monotonic()
    for row in sample:
        directory = output / 'images' / row['role'] / row['method'] / row['video_id']
        result = process_record(row, detector, directory)
        results.append(result)
        print(json.dumps({'video_uid': row['video_uid'], 'eligible': result['eligible'],
                          'seconds': result['elapsed_seconds']}), flush=True)
        write_json(output / 'H_source_qualification_progress.json', {'completed': len(results), 'total': len(sample)})
    metrics = {}
    for version in VERSIONS:
        frames = [frame[version] for row in results for frame in row['frames'] if frame[version]['ok']]
        ratios = [frame['padding_ratio'] for frame in frames]
        metrics[version] = {'eligible_videos': sum(row['eligible'][version] for row in results),
                            'successful_frames': len(frames),
                            'padding_mean': float(np.mean(ratios)) if ratios else None,
                            'padding_max': float(max(ratios)) if ratios else None,
                            'png_bytes': sum(frame['png_bytes'] for frame in frames),
                            'mask_bytes': sum(frame['mask_bytes'] for frame in frames)}
    detections = [frame[k] for row in results for frame in row['frames'] for k in ('original_detection', 'a_redetection') if k in frame]
    source_counts = Counter(row['video']['role'] for row in results)
    report = {'status': 'SOURCE_QUALIFIED' if all(all(row['eligible'].values()) for row in results) else 'SOURCE_FAILURES_REVIEW_REQUIRED',
              'protocol_sha256': protocol_hash(), 'code_sha256': sha256_file(__file__),
              'videos': len(results), 'roles': dict(source_counts), 'per_cell': per_cell,
              'versions': metrics, 'common_B_AB_videos': sum(row['common_B_AB'] for row in results),
              'multiface_detections': sum(row['multiface'] for row in detections), 'detections': len(detections),
              'elapsed_seconds': time.monotonic() - started,
              'detector_calls': getattr(detector, 'calls', None), 'detector_seconds': getattr(detector, 'seconds', None),
              'decode_seconds': sum(row.get('decode', {}).get('decode_seconds', 0) for row in results),
              'ffpp_dfd_candidate_videos': 8431, 'expected_mothers_before_exclusions': 67448,
              'png_storage_extrapolation_bytes': sum(m['png_bytes'] + m['mask_bytes'] for m in metrics.values()) / len(results) * 8431,
              'extrapolation_note': 'source10-cell sample only; no target images inspected; includes A/B/AB PNG and masks, excludes manifests',
              'target_videos_decoded_or_detected': 0, 'target_tuning': False,
              'results': [{'video_uid': row['video']['video_uid'], 'eligible': row['eligible'],
                           'manifest': str(output / 'images' / row['video']['role'] / row['video']['method'] / row['video']['video_id'] / 'manifest.json')}
                          for row in results]}
    write_json(output / 'H_source_qualification_report.json', report)
    return report


class SourceGroups:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            following = self.parent[key]
            self.parent[key] = root
            key = following
        return root

    def union(self, left, right):
        left, right = self.find(left), self.find(right)
        self.parent[max(left, right)] = min(left, right)


def all_h_candidates(manifest, metadata_exclusions):
    records = [dict(row) for row in manifest]
    for exclusion in metadata_exclusions:
        if exclusion['reason'] != 'official_val_fake_not_used':
            continue
        path = Path(exclusion['path'])
        method = path.parents[2].name
        ids = path.stem.split('_')
        records.append({'dataset': 'FFPP', 'domain': 'source', 'method': method,
                        'label': 1, 'video_id': path.stem,
                        'video_uid': f'FFPP/source/{method}/{path.stem}',
                        'role': 'SOURCE_UNUSED', 'source_ids': ids,
                        'source_group': f'FFPP/pair/{"_".join(sorted(ids))}',
                        'source_evidence': 'official_val_fake_excluded_from_training_and_tuning',
                        'path': str(path)})
    proxy = SourceGroups()
    for row in records:
        if row['dataset'] != 'DFD':
            continue
        proxy.find(row['video_uid'])
        for parent in row['source_ids']:
            proxy.union(row['video_uid'], f'DFD/target/real/{parent}')
    for row in records:
        if row['dataset'] == 'FFPP':
            row['group_id'] = row['source_group']
            row['source_graph_certified'] = True
        else:
            row['group_id'] = 'DFD/proxy/' + hashlib.sha256(proxy.find(row['video_uid']).encode()).hexdigest()[:20]
            row['source_graph_certified'] = False
    return sorted(records, key=lambda row: (row['role'], row['method'], row['video_uid']))


def h_content_audit(results):
    """Close known groups over original and derived RGB before cohort freezing.

    SOURCE_UNUSED is inventoried but cannot cause training/calibration exposure.
    A conflicting connected group is quarantined in its entirety. DFD grouping
    remains conditional on explicitly unverified filename parent proxies.
    """
    groups, rgb, group_rows = SourceGroups(), defaultdict(list), defaultdict(list)
    for result in results:
        video = result['video']
        group = video.get('group_id') or video.get('source_group') or video['video_uid']
        video['group_id'] = group
        groups.find(group)
        group_rows[group].append(video)
        if video['role'] == 'SOURCE_UNUSED':
            continue
        for frame in result['frames']:
            hashes = [('mother', frame['mother_rgb_sha256'])]
            hashes.extend((version, frame[version]['rgb_sha256']) for version in VERSIONS
                          if version in frame and frame[version]['ok'])
            for version, digest in hashes:
                rgb[digest].append({'group_id': group, 'video_uid': video['video_uid'],
                                    'role': video['role'], 'label': video['label'], 'version': version})
    cross_role, label_conflicts, rgb_edges = [], [], 0
    for digest, members in rgb.items():
        components = sorted({row['group_id'] for row in members})
        for other in components[1:]:
            groups.union(components[0], other)
            rgb_edges += 1
        if len({row['role'] for row in members}) > 1:
            cross_role.append({'rgb_sha256': digest, 'members': members})
        if len({row['label'] for row in members}) > 1:
            label_conflicts.append({'rgb_sha256': digest, 'members': members})
    roles_by_root = defaultdict(set)
    for group, videos in group_rows.items():
        roles_by_root[groups.find(group)].update(row['role'] for row in videos if row['role'] != 'SOURCE_UNUSED')
    source_conflict_roots = {root for root, roles in roles_by_root.items() if len(roles) > 1}
    for collision in label_conflicts:
        source_conflict_roots.update(groups.find(row['group_id']) for row in collision['members'])
    blocked = {group for group in group_rows if groups.find(group) in source_conflict_roots}
    for result in results:
        result['video']['group_id'] = groups.find(result['video']['group_id'])
    report = {'unique_rgb_hashes': len(rgb), 'rgb_group_edges': rgb_edges,
              'source_group_components': len({groups.find(group) for group in group_rows}),
              'cross_role_rgb_conflicts': cross_role, 'label_rgb_conflicts': label_conflicts,
              'source_group_role_conflicts': {root: sorted(roles_by_root[root]) for root in sorted(source_conflict_roots)},
              'quarantined_original_groups': sorted(blocked),
              'quarantined_videos': sum(result['video']['group_id'] in source_conflict_roots for result in results),
              'DFD_group_scope': 'conservative filename candidate closure plus exact RGB; complete ancestry not certified',
              'SOURCE_UNUSED_scope': 'inventoried, never included in training/tuning/calibration/test tables'}
    return report, blocked | source_conflict_roots


def h_frame_rows(result, version):
    video = result['video']
    rows = []
    for i, frame in enumerate(result['frames']):
        if not frame[version]['ok']:
            continue
        # Same logical slot in B/AB, irrespective of duplicate RGB across videos.
        mother_id = f'{video["video_uid"]}/mother/{i:03d}/{frame["mother_rgb_sha256"]}'
        rows.append({'image_path': frame[version]['path'], 'label': video['label'],
                     'method_id': video['method'], 'video_id': video['video_id'],
                     'video_uid': video['video_uid'], 'group_id': video.get('group_id') or video.get('source_group'),
                     'domain': video['domain'], 'role': video['role'],
                     'rgb_sha256': frame[version]['rgb_sha256'],
                     'mother_id': mother_id, 'sample_id': mother_id,
                     'mother_rgb_sha256': frame['mother_rgb_sha256'],
                     'mother_frame_index': frame['mother_index'],
                     'mother_timestamp_seconds': frame['mother_timestamp'],
                     'processing_version': version,
                     'source_graph_certified': video.get('source_graph_certified', False)})
    return rows


def h_training_rows(results, role, quarantined_groups=()):
    selected = sorted((row for row in results if row['video']['role'] == role and row['common_B_AB']
                       and row['video'].get('group_id') not in quarantined_groups),
                      key=lambda row: row['video']['video_uid'])
    counts = Counter(row['video']['method'] for row in selected)
    b_rows, ab_rows = [], []
    for result in selected:
        video = result['video']
        method = video['method']
        mass = (.5 if method == 'real' else .5 / 4) / counts[method] / 8
        for version, destination in [('B', b_rows), ('AB', ab_rows)]:
            rows = h_frame_rows(result, version)
            if len(rows) != 8:
                raise ValueError('Common-success video must contain8rows in each history')
            for row in rows:
                row['sampling_mass'] = mass
                row['method_contributions'] = json.dumps({method: mass}, sort_keys=True)
            destination.extend(rows)
    return b_rows, ab_rows


def h_evaluation_rows(results, quarantined_groups=()):
    tables = {f'{prefix}/{role}': [] for prefix in ('B', 'common/B', 'common/AB')
              for role in ('SOURCE_CAL', 'SOURCE_TEST', 'TARGET_TEST')}
    for result in sorted(results, key=lambda row: row['video']['video_uid']):
        role = result['video']['role']
        if role not in ('SOURCE_CAL', 'SOURCE_TEST', 'TARGET_TEST'):
            continue
        if result['video'].get('group_id') in quarantined_groups:
            continue
        if result['eligible']['B']:
            tables[f'B/{role}'].extend(h_frame_rows(result, 'B'))
        if result['common_B_AB']:
            for version in ('B', 'AB'):
                tables[f'common/{version}/{role}'].extend(h_frame_rows(result, version))
    return tables


def finalize_h(manifest_paths, artifacts):
    import pandas as pd
    artifacts = Path(artifacts)
    results = [json.loads(Path(path).read_text()) for path in manifest_paths]
    audit, quarantined = h_content_audit(results)
    write_json(artifacts / 'H_content_source_audit.json', audit)
    write_json(artifacts / 'H_final_source_groups.json',
               [{'video_uid': row['video']['video_uid'], 'group_id': row['video']['group_id'],
                 'source_graph_certified': row['video'].get('source_graph_certified', False)} for row in results])
    coverage = []
    for (role, method), subset in _group_results(results).items():
        item = {'role': role, 'method': method, 'candidate_videos': len(subset),
                'common_B_AB': sum(row['common_B_AB'] for row in subset),
                'video_decode_failures': sum('video_failure' in row for row in subset)}
        for version in VERSIONS:
            item[f'{version}_eligible_videos'] = sum(row['eligible'][version] for row in subset)
            outputs = [frame[version] for row in subset for frame in row['frames'] if frame[version]['ok']]
            item[f'{version}_successful_frames'] = len(outputs)
            item[f'{version}_padding_mean'] = float(np.mean([frame['padding_ratio'] for frame in outputs])) if outputs else None
            item[f'{version}_padding_max'] = max((frame['padding_ratio'] for frame in outputs), default=None)
            reasons = Counter(reason for row in subset for reason in row.get('version_failures', {}).get(version, []))
            item[f'{version}_failure_reasons'] = dict(reasons)
        detections = [frame[key] for row in subset for frame in row['frames']
                      for key in ('original_detection', 'a_redetection') if key in frame]
        item['detections'] = len(detections)
        item['multiface_detections'] = sum(row['multiface'] for row in detections)
        coverage.append(item)
    write_json(artifacts / 'H_coverage_by_role_method.json', coverage)
    tables, training_audit = {}, {}
    for role in ('TRAIN', 'DEV'):
        b_rows, ab_rows = h_training_rows(results, role, quarantined)
        tables[f'B/{role.lower()}'], tables[f'AB/{role.lower()}'] = b_rows, ab_rows
        training_audit[role] = {'logical_mother_rows': len(b_rows),
                                'draw_count': ((len(b_rows) + 127) // 128) * 128,
                                'video_counts': dict(Counter(row['method_id'] for row in b_rows[::8])),
                                'sampling_mass': sum(row['sampling_mass'] for row in b_rows),
                                'paired_mother_ids_identical': [row['mother_id'] for row in b_rows] == [row['mother_id'] for row in ab_rows],
                                'method_probability': {method: sum(row['sampling_mass'] for row in b_rows if row['method_id'] == method)
                                                       for method in ('real', *METHODS)}}
    tables.update(h_evaluation_rows(results, quarantined))
    columns = ['image_path', 'label', 'method_id', 'video_id', 'video_uid', 'group_id', 'domain', 'role', 'rgb_sha256']
    table_info = {}
    for key, rows in tables.items():
        path = artifacts / f'{key}.parquet'
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=None if rows else columns).to_parquet(path, index=False)
        table_info[key] = {'path': str(path), 'rows': len(rows),
                           'videos': len({row['video_uid'] for row in rows}), 'sha256': sha256_file(path)}
    audit_ok = (not audit['cross_role_rgb_conflicts'] and not audit['label_rgb_conflicts']
                and not audit['source_group_role_conflicts']
                and all(np.isclose(row['sampling_mass'], 1) for row in training_audit.values()))
    report = {'status': 'AUDIT_COMPLETE_READY_FOR_H_DATA_FREEZE_REVIEW' if audit_ok else 'AUDIT_CONFLICTS_REVIEW_REQUIRED',
              'geometry_protocol_sha256': protocol_hash(), 'code_sha256': sha256_file(__file__),
              'candidate_videos': len(results), 'training': training_audit, 'tables': table_info,
              'data_frozen': False, 'target_performance_used': False,
              'quarantined_group_count': len(quarantined), 'coverage_report': str(artifacts / 'H_coverage_by_role_method.json')}
    write_json(artifacts / 'H_data_freeze_candidate.json', report)
    return report


def _group_results(results):
    grouped = defaultdict(list)
    for row in results:
        grouped[(row['video']['role'], row['video']['method'])].append(row)
    return dict(sorted(grouped.items()))


_H_DETECTOR = None


def _worker_init(threads):
    global _H_DETECTOR
    cv2.setNumThreads(1)
    _H_DETECTOR = SCRFD5(threads=threads)


def _worker_record(task):
    record, image_root, freeze = task
    directory = Path(image_root) / record['dataset'] / record['role'] / record['method'] / record['video_id']
    manifest = directory / 'manifest.json'
    if manifest.is_file():
        previous = json.loads(manifest.read_text())
        if (previous.get('protocol_sha256') == protocol_hash()
                and previous.get('video_byte_sha256') == sha256_file(record['path'])
                and all('mother_path' in frame and Path(frame['mother_path']).is_file() for frame in previous['frames'])):
            return str(manifest), previous['eligible'], previous['elapsed_seconds'], True
    result = process_record(record, _H_DETECTOR, directory, freeze=freeze)
    return str(manifest), result['eligible'], result['elapsed_seconds'], False


def preprocess_all(args):
    if args.workers > 4 or args.threads > 2 or args.workers < 1 or args.threads < 1:
        raise ValueError('H CPU preprocessing is bounded to4workers and2ORTthreads each')
    frozen = json.loads(args.freeze.read_text())
    if frozen.get('protocol_sha256') != protocol_hash():
        raise ValueError('Geometry freeze hash mismatch')
    records = all_h_candidates(json.loads(args.manifest.read_text()), json.loads(args.exclusions.read_text()))
    if args.limit:
        records = records[:args.limit]
    args.artifacts.mkdir(parents=True, exist_ok=True)
    args.image_root.mkdir(parents=True, exist_ok=True)
    write_json(args.artifacts / 'H_candidates_all.json', records)
    source_report = json.loads(Path(frozen['source_report']).read_text())
    projected = source_report['png_storage_extrapolation_bytes']
    free = os.statvfs(args.image_root).f_bavail * os.statvfs(args.image_root).f_frsize
    # Source mother PNGs can exceed three cropped PNGs substantially. Reserve
    # a conservative20x cropped estimate +50GB, without deleting any data.
    if free < projected * 20 + 50e9:
        raise ValueError('Dataset free space below conservative mother/crop reserve')
    started, completed, manifests, failure_counts = time.monotonic(), 0, [], Counter()
    progress_path = args.artifacts / 'H_preprocessing_progress.json'
    def progress(status):
        elapsed = time.monotonic() - started
        value = {'status': status, 'pid': os.getpid(), 'workers': args.workers,
                 'ORT_threads_each': args.threads, 'GPU_used': False,
                 'total_videos': len(records), 'completed_videos': completed,
                 'elapsed_seconds': elapsed,
                 'eta_seconds': elapsed / completed * (len(records) - completed) if completed else None,
                 'version_ineligible_video_counts': dict(failure_counts),
                 'geometry_protocol_sha256': protocol_hash(), 'geometry_freeze': str(args.freeze),
                 'image_root': str(args.image_root), 'data_frozen': False}
        write_json(progress_path, value)
        return value
    print(json.dumps(progress('RUNNING')), flush=True)
    context = multiprocessing.get_context('spawn')
    tasks = [(row, str(args.image_root), str(args.freeze)) for row in records]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
                             initializer=_worker_init, initargs=(args.threads,)) as executor:
        futures = {executor.submit(_worker_record, task): task[0]['video_uid'] for task in tasks}
        for future in as_completed(futures):
            path, eligible, seconds, reused = future.result()
            completed += 1
            manifests.append(path)
            failure_counts.update(version for version, ok in eligible.items() if not ok)
            with (args.artifacts / 'H_preprocessed_videos.jsonl').open('a') as file:
                file.write(json.dumps({'video_uid': futures[future], 'manifest': path,
                                       'eligible': eligible, 'seconds': seconds, 'reused': reused}) + '\n')
            if completed % 10 == 0 or completed == len(records):
                print(json.dumps(progress('RUNNING')), flush=True)
    write_json(args.artifacts / 'H_processed_manifest_paths.json', sorted(manifests))
    progress('PIXELS_COMPLETE_AUDITING_CONTENT_AND_SOURCE')
    report = finalize_h(sorted(manifests), args.artifacts)
    print(json.dumps(progress(report['status'])), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit')
    audit.add_argument('--output', type=Path, default=ARTIFACT_ROOT / 'data/H')
    audit.add_argument('--dataset-root', type=Path, default=DATA_ROOT)
    audit.add_argument('--split-root', type=Path, default=SPLIT_ROOT)
    qualify = sub.add_parser('qualify-source')
    qualify.add_argument('--manifest', type=Path, default=ARTIFACT_ROOT / 'data/H/H_manifest.json')
    qualify.add_argument('--output', type=Path, default=ARTIFACT_ROOT / 'qualification/H_source_geometry')
    qualify.add_argument('--per-cell', type=int, default=1)
    qualify.add_argument('--threads', type=int, default=2)
    freeze = sub.add_parser('freeze-source')
    freeze.add_argument('--report', type=Path, required=True)
    freeze.add_argument('--output', type=Path, default=ARTIFACT_ROOT / 'qualification/H_geometry_frozen.json')
    bulk = sub.add_parser('preprocess-all')
    bulk.add_argument('--manifest', type=Path, default=ARTIFACT_ROOT / 'data/H/H_manifest.json')
    bulk.add_argument('--exclusions', type=Path, default=ARTIFACT_ROOT / 'data/H/H_exclusions_metadata.json')
    bulk.add_argument('--freeze', type=Path, default=ARTIFACT_ROOT / 'qualification/H_geometry_frozen.json')
    bulk.add_argument('--artifacts', type=Path, default=ARTIFACT_ROOT / 'data/H')
    bulk.add_argument('--image-root', type=Path, default=Path('/Dataset/deepfake_detection/derived/pipeline_reliability_v2_20261023/H'))
    bulk.add_argument('--workers', type=int, default=4)
    bulk.add_argument('--threads', type=int, default=2)
    bulk.add_argument('--limit', type=int, default=0)
    args = parser.parse_args(argv)
    cv2.setNumThreads(2)
    if args.command == 'audit':
        records, mapping, exclusions, report = build_manifest(args.dataset_root, args.split_root)
        write_json(args.output / 'H_manifest.json', records)
        write_json(args.output / 'H_source_pair_partition.json', mapping)
        write_json(args.output / 'H_exclusions_metadata.json', exclusions)
        write_json(args.output / 'H_metadata_audit.json', report)
        print(json.dumps(report, indent=2))
    elif args.command == 'qualify-source':
        records = [row for row in json.loads(args.manifest.read_text()) if row['role'] in ('TRAIN', 'DEV')]
        print(json.dumps(qualify_source(records, args.output, SCRFD5(threads=args.threads), args.per_cell), indent=2))
    elif args.command == 'preprocess-all':
        preprocess_all(args)
    else:
        report = json.loads(args.report.read_text())
        if (report['status'] != 'SOURCE_QUALIFIED' or report['protocol_sha256'] != protocol_hash()
                or set(report['roles']) != {'TRAIN', 'DEV'} or report['videos'] < 10):
            raise ValueError('Source qualification does not satisfy geometry freeze gate')
        write_json(args.output, {'protocol_sha256': protocol_hash(), 'source_report': str(args.report),
                                'source_report_sha256': sha256_file(args.report),
                                'code_sha256': sha256_file(__file__), 'protocol': protocol(),
                                'status': 'FROZEN_BEFORE_TARGET_PREPROCESSING'})
        print(str(args.output))


if __name__ == '__main__':
    main()

