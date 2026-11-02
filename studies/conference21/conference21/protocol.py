"""Strict conference21 protocol; earlier run configurations cannot be reused."""
import json
from pathlib import Path
import numpy as np
from .data_d import METHODS

PROTOCOL='conference21-ema-df40-20261102'
LRS={'clip_vit_b16':1e-5,'dinov2_vit_b14':5e-5,'efficientnet_b0':1e-4}
SEEDS=[7,42,123,2026,3407]


def validate_config(config):
    if config.get('protocol')!=PROTOCOL:raise ValueError('wrong training protocol')
    if config.get('effective_batch')!=128:raise ValueError('per-GPU batch must be128')
    if config.get('formal_seeds')!=SEEDS:raise ValueError('wrong five-seed registry')
    if set(config.get('backbones',{}))!=set(LRS):raise ValueError('wrong backbone registry')
    expected={(b,s) for b in LRS for s in SEEDS};found=[]
    for task in config['runs'].values():
        if (task['epochs']!=15 or task['microbatch']!=128 or task['lr']!=LRS[task['backbone']]
            or task.get('phase')!='formal' or task.get('recipe_frozen') is not True
            or task.get('precision')!='AMP' or task.get('route')!='D'
            or task.get('train_manifest')!='data/D/train.parquet' or task.get('dev_manifest')!='data/D/dev.parquet'):
            raise ValueError('run violates fixed conference21 recipe')
        found.append((task['backbone'],task['seed']))
    if len(found)!=15 or set(found)!=expected:raise ValueError('requires exactly three backbones x five seeds')


def validate_manifests(root,train,dev):
    from .trainer import digest
    root=Path(root);freeze=json.loads((root/'data/D/freeze.json').read_text())
    if freeze.get('status')!='FROZEN' or freeze.get('method_count')!=21:
        raise ValueError('requires frozen21-method manifests')
    registry=freeze['method_registry']['FS9']+freeze['method_registry']['FR12']
    if set(registry)!=set(METHODS):raise ValueError('wrong method registry')
    for role in ['train','dev']:
        path=root/f'data/D/{role}.parquet'
        if digest(path)!=freeze['manifest_sha256'][path.name]:raise ValueError('manifest hash mismatch')
    if train.rgb_sha256.duplicated().any():raise ValueError('training RGB not canonical')
    if set(train.rgb_sha256)&set(dev.rgb_sha256):raise ValueError('TRAIN/DEV RGB overlap')
    if set(train.group_id)&set(dev.group_id):raise ValueError('TRAIN/DEV source component overlap')
    if dev.groupby('video_uid').size().ne(8).any() or dev.groupby('video_uid').rgb_sha256.nunique().ne(8).any():
        raise ValueError('DEV must have8 distinct RGB per natural video')
    masses={}
    for row in train.itertuples():
        contribution=json.loads(row.method_contributions)
        if not np.isclose(sum(contribution.values()),row.sampling_mass):raise ValueError('row mass mismatch')
        for name,mass in contribution.items():masses[name]=masses.get(name,0.)+mass
    if set(masses)!=set(METHODS)|{'real'} or not np.isclose(masses['real'],.5):raise ValueError('wrong class sampling mass')
    if any(not np.isclose(masses[m],.5/21) for m in METHODS):raise ValueError('wrong fake method sampling mass')
