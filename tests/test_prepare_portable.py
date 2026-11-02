"""Portable preparation runs against tiny synthetic raw data, without old artifacts."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image
import pytest

from journal_detector import data_d
from journal_detector import prepare as prep


def test_audit_reads_current_bytes_without_prior_snapshot(tmp_path):
    image = tmp_path / 'frame.png'
    Image.new('RGB', (3, 2), 'red').save(image)
    out = tmp_path / 'audit'
    out.mkdir()
    frames = pd.DataFrame({'image_path': [str(image), str(image)]})
    audited, receipt = data_d.audit_images(frames, out, workers=1)
    assert len(audited) == 2
    assert receipt['candidate_unique_paths'] == receipt['fresh_rehashed'] == 1
    first = audited.iloc[0].rgb_sha256
    Image.new('RGB', (3, 2), 'blue').save(image)
    audited, _ = data_d.audit_images(frames, out, workers=1)
    assert audited.iloc[0].rgb_sha256 != first


def test_checked_in_metadata_is_complete_and_verified():
    definition = data_d.load_definition()
    assert len(definition['indices_sha256']) == 42
    assert len(definition['source_real_candidates']['videos']) == 140
    assert sum(len(v['frames']) for v in definition['source_real_candidates']['videos']) == 4477
    assert len(definition['cdf_real_split']['calibration_pool']) == 80
    assert len(definition['cdf_real_split']['heldout_test']) == 97
    assert set(definition['cdf_real_split']['calibration_pool']).isdisjoint(definition['cdf_real_split']['heldout_test'])


def test_registered_config_requires_no_weights_and_resolves_paths(tmp_path):
    config = prep.execution_config(prep.DEFAULT_CONFIG, tmp_path / 'output')
    assert config['artifact_root'] == str(tmp_path / 'output')
    assert len(config['runs']) == 15
    assert all(Path(v['weight_path']).is_absolute() for v in config['backbones'].values())
    raw_config=json.loads(prep.DEFAULT_CONFIG.read_text())
    base=prep.DEFAULT_CONFIG.parent.parent
    assert all(v['weight_path']==str((base/raw_config['backbones'][b]['weight_path']).resolve())
               for b,v in config['backbones'].items())
    assert all(v['weight_url'].startswith('https://') for v in config['backbones'].values())


def test_status_unprepared_is_read_only_from_any_directory(tmp_path):
    root = Path(__file__).resolve().parents[1]
    import os
    env = dict(os.environ, PYTHONPATH=str(root / 'src'))
    output = tmp_path / 'uncreated'
    result = subprocess.run([sys.executable, '-m', 'journal_detector', 'status', '--output-root', str(output)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, check=True)
    status = json.loads(result.stdout)
    assert status['status'] == 'NOT_PREPARED'
    assert status['formal_registered'] == 0
    assert status['planned_runs'] == 15
    assert not output.exists()


def test_index_hash_mismatch_is_rejected_before_parsing(tmp_path):
    raw = tmp_path / 'raw'
    folder = raw / 'dataset_json_update'
    folder.mkdir(parents=True)
    (folder / 'fsgan_ff.json').write_text('{}')
    definition = {'indices_sha256': {'fsgan_ff.json': '0' * 64}}
    with pytest.raises(ValueError, match='Source index hash mismatch'):
        data_d.rebuild_indices(raw, definition)


def test_complete_prepare_from_synthetic_42_indices_without_legacy_artifacts(tmp_path):
    """Exercise every build output and CDF closure with genuine tiny decoded images."""
    import shutil
    definitions = tmp_path / 'configs/data_protocol'
    shutil.copytree(data_d.DEFINITION_ROOT, definitions)
    definition = data_d.load_definition(definitions)
    roles, _ = data_d.pair_partition(definition['splits'])
    train_source = next(k for k, v in roles.items() if v == 'train')
    dev_source = next(k for k, v in roles.items() if v == 'dev')
    test_source = next(k for k, v in roles.items() if v == 'source_test')
    raw = tmp_path / 'raw'
    indexes = raw / 'dataset_json_update'
    indexes.mkdir(parents=True)
    counter = 0

    def image_paths(directory, count=8):
        nonlocal counter
        paths = []
        for i in range(count):
            counter += 1
            path = raw / directory / f'{i:03}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (2, 2), (counter % 256, (counter // 256) % 256, 77)).save(path)
            paths.append('deepfakes_detection_datasets/' + str(path.relative_to(raw)))
        return paths

    real_ff = {v: image_paths(f'FaceForensics++/frames/{v}') for v in [train_source, dev_source, test_source]}
    cdf_ids = definition['cdf_real_split']['calibration_pool'] + definition['cdf_real_split']['heldout_test']
    real_cdf = {v: image_paths(f'Celeb-real/frames/{v}') for v in cdf_ids}
    hashes = {}
    for method in data_d.METHODS:
        for domain in ['ff', 'cdf']:
            label_real, label_fake = method + '_Real', method + '_Fake'
            real_roles = {'train': {}, 'test': {}}
            fake_roles = {'train': {}, 'test': {}}
            if domain == 'ff':
                for video in [train_source, dev_source, test_source]:
                    role = 'test' if video == test_source else 'train'
                    real_roles[role][video] = {'label': label_real, 'frames': real_ff[video]}
                    prefix = 'DF40' if role == 'test' else 'DF40_train'
                    fake_roles[role][video] = {'label': label_fake,
                        'frames': image_paths(f'{prefix}/{method}/ff/{video}')}
                    # Test indices retain the released DF40 prefix, while raw files live under DF40_test.
                    if role == 'test':
                        source = raw / f'DF40/{method}/ff/{video}'
                        destination = raw / f'DF40_test/{method}/ff/{video}'
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        source.rename(destination)
            else:
                real_roles['test'] = {v: {'label': label_real, 'frames': frames} for v, frames in real_cdf.items()}
                fake_roles['test']['unrelated'] = {'label': label_fake,
                    'frames': image_paths(f'DF40/{method}/cdf/unrelated')}
                source = raw / f'DF40/{method}/cdf/unrelated'
                destination = raw / f'DF40_test/{method}/cdf/unrelated'
                destination.parent.mkdir(parents=True, exist_ok=True)
                source.rename(destination)
            path = indexes / f'{method}_{domain}.json'
            path.write_text(json.dumps({method: {label_real: real_roles, label_fake: fake_roles}}))
            hashes[path.name] = data_d.digest(path)
    cal_id = next(k for k, v in roles.items() if v == 'source_cal')
    image_paths(f'FaceForensics++/frames/{cal_id}')
    source_cal = {'candidate_paths': 8, 'videos': [{'source_id': cal_id,
        'directory': f'FaceForensics++/frames/{cal_id}', 'frames': [f'{i:03}.png' for i in range(8)]}]}
    data_d.save_json(definitions / 'source_real_candidates.json', source_cal)
    data_d.save_json(definitions / 'indices_sha256.json', hashes)
    metadata = json.loads((definitions / 'definition.json').read_text())
    metadata['files_sha256'] = {name: data_d.digest(definitions / name) for name in metadata['files_sha256']}
    data_d.save_json(definitions / 'definition.json', metadata)
    config_path = tmp_path / 'configs/journal.json'
    shutil.copyfile(prep.DEFAULT_CONFIG, config_path)
    output = tmp_path / 'output'
    receipt = prep.prepare(config_path, output, raw, definitions, workers=2)
    assert receipt['status'] == 'PREPARED_NOT_TRAINED'
    assert receipt['runs'] == 15
    assert not (tmp_path / 'weights').exists()
    train = pd.read_parquet(output / 'data/D/train.parquet')
    assert not train.rgb_sha256.duplicated().any()
    assert np.isclose(train.sampling_mass.sum(), 1)
    freeze = json.loads((output / 'data/D/freeze.json').read_text())
    assert freeze['cdf_calibration_scope']['cal_real_videos'] == 80
    assert freeze['cdf_calibration_scope']['test_real_videos'] == 77
    assert freeze['cdf_calibration_scope']['test_fake_videos'] == 21
    assert freeze['cdf_development_scope']['real_video_actual'] == 20
    assert freeze['cdf_development_scope']['status'] == 'NOT_USABLE'
    assert freeze['cdf_development_scope']['fake_video_actual'] == 0
    assert freeze['cache_audit']['fresh_rehashed'] == counter
    assert len(pd.read_parquet(output / 'data/D/source_cal.parquet')) == 8
    assert prep.status(config_path, output)['data_prepared'] is True
    assert prep.prepare(config_path, output, raw, definitions, workers=2) == receipt
    (output / 'data/D/counts.csv').write_text('tampered')
    with pytest.raises(ValueError, match='frozen file changed: counts.csv'):
        prep.prepare(config_path, output, raw, definitions, workers=2)
