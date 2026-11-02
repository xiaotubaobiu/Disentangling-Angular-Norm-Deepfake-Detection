"""Rebuild raw-data manifests and register 15 runs; never launches a model."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from . import data_d
from .data_d import digest, save_json
from .protocol import PROTOCOL, validate_config, validate_manifests

DEFAULT_CONFIG = data_d.REPOSITORY_ROOT / 'configs/journal.json'


def execution_config(config_path=DEFAULT_CONFIG, output_root=None):
    """Resolve repository-relative paths without loading or requiring weights."""
    path = Path(config_path).resolve()
    config = json.loads(path.read_text())
    base = path.parent.parent
    root = Path(output_root) if output_root is not None else Path(config['artifact_root'])
    if output_root is None and not root.is_absolute(): root = base/root
    config['artifact_root'] = str(root.resolve())
    for spec in config['backbones'].values():
        weight = Path(spec['weight_path'])
        spec['weight_path'] = str((weight if weight.is_absolute() else base/weight).resolve())
    validate_config(config)
    return config


def verify_freeze(root):
    data = Path(root)/'data/D'
    freeze = json.loads((data/'freeze.json').read_text())
    if freeze.get('status') != 'FROZEN' or freeze.get('method_count') != 21:
        raise ValueError('invalid data freeze')
    for name, expected in freeze['manifest_sha256'].items():
        if Path(name).name != name: raise ValueError('unsafe frozen filename')
        if digest(data/name) != expected: raise ValueError('frozen file changed: '+name)
    validate_manifests(root, pd.read_parquet(data/'train.parquet'), pd.read_parquet(data/'dev.parquet'))
    return freeze


def prepare(config_path=DEFAULT_CONFIG, output_root=None, data_root=data_d.DATA_ROOT,
            definition_root=data_d.DEFINITION_ROOT, workers=4):
    from .trainer import exclusive_file
    config = execution_config(config_path, output_root)
    root = Path(config['artifact_root'])
    raw = Path(data_root).resolve()
    if root == raw or raw in root.parents: raise ValueError('output must be outside raw data')
    with exclusive_file(root/'prepare.lock'):
        path = root/'configs/execution.json'
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError('existing registry differs')
        data = root/'data/D'
        if not (data/'freeze.json').exists():
            data_d.build(root, raw, definition_root, workers)
        freeze = verify_freeze(root)
        if freeze.get('data_definition_sha256') != digest(Path(definition_root)/'definition.json'):
            raise ValueError('frozen data definition differs')
        if freeze.get('raw_data_root') != str(raw): raise ValueError('frozen raw-data root differs')
        save_json(path, config)
        lockpath = root/'configs/requirements.execution.lock.txt'
        lock = subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        if lockpath.exists() and lockpath.read_text() != lock:
            raise ValueError('execution environment changed')
        lockpath.write_text(lock)
        receipt = {'status':'PREPARED_NOT_TRAINED','config':str(path),'runs':len(config['runs']),
            'raw_data_root':str(raw),'data_definition_sha256':freeze['data_definition_sha256'],
            'freeze_sha256':digest(data/'freeze.json'),'protocol':PROTOCOL,
            'initialization_weights_required_for_prepare':False}
        # Re-running prepare must not relabel an already executed experiment as untrained.
        existing = root/'prepared.json'
        if existing.exists(): return json.loads(existing.read_text())
        save_json(existing,receipt)
        return receipt


def status(config_path=DEFAULT_CONFIG, output_root=None):
    """Inventory only: no model, audit, output directory, or registry is created."""
    config = execution_config(config_path,output_root)
    root = Path(config['artifact_root'])
    ready = all((root/name).is_file() for name in ['prepared.json','data/D/freeze.json','configs/execution.json'])
    if not ready:
        return {'status':'NOT_PREPARED','artifact_root':str(root),'planned_runs':len(config['runs']),
                'formal_registered':0,'formal_caches_complete':0,'training_started':False}
    from .analysis_d import formal_cache_status
    result = formal_cache_status(root/'configs/execution.json')
    result.update(artifact_root=str(root), data_prepared=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--data-root',type=Path,default=data_d.DATA_ROOT)
    p.add_argument('--definition-root',type=Path,default=data_d.DEFINITION_ROOT)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    print(json.dumps(prepare(a.config,a.output_root,a.data_root,a.definition_root,a.workers),indent=2))
