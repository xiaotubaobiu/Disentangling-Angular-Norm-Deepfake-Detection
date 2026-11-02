"""Strict journal_detector protocol; earlier run configurations cannot be reused."""
import json
from pathlib import Path
import numpy as np
from .data_d import METHODS

LEGACY_PROTOCOL='journal_detector-ema-df40-20261102'
PROTOCOL='journal_detector-target-dev-20261102-v2'
LRS={'clip_vit_b16':1e-5,'dinov2_vit_b14':5e-5,'efficientnet_b0':1e-4}
SEEDS=[7,42,123,2026,3407]
H_SEEDS=[7,42,123]
DEVELOPMENT_SEEDS=[1007,1042]


def validate_config(config):
    if config.get('protocol')==PROTOCOL:
        return validate_execution_config(config)
    if config.get('protocol')!=LEGACY_PROTOCOL:raise ValueError('wrong training protocol')
    if config.get('effective_batch')!=128:raise ValueError('per-GPU batch must be128')
    if config.get('formal_seeds')!=SEEDS:raise ValueError('wrong five-seed registry')
    if set(config.get('backbones',{}))!=set(LRS):raise ValueError('wrong backbone registry')
    expected={(b,s) for b in LRS for s in SEEDS};found=[]
    for task in config['runs'].values():
        if (task['epochs']!=15 or task['microbatch']!=128 or task['lr']!=LRS[task['backbone']]
            or task.get('phase')!='formal' or task.get('recipe_frozen') is not True
            or task.get('precision')!='AMP' or task.get('route')!='D'
            or task.get('train_manifest')!='data/D/train.parquet' or task.get('dev_manifest')!='data/D/dev.parquet'):
            raise ValueError('run violates fixed journal_detector recipe')
        found.append((task['backbone'],task['seed']))
    if len(found)!=15 or set(found)!=expected:raise ValueError('requires exactly three backbones x five seeds')


def validate_execution_config(config):
    if config.get('effective_batch')!=128:raise ValueError('actual per-GPU batch must be128')
    if config.get('formal_seeds')!=SEEDS or config.get('h_formal_seeds')!=H_SEEDS:
        raise ValueError('wrong formal seed registries')
    if config.get('development_seeds')!=DEVELOPMENT_SEEDS:
        raise ValueError('wrong development seed registry')
    if set(config.get('backbones',{}))!=set(LRS):raise ValueError('wrong backbone registry')
    if not config.get('runs'):raise ValueError('empty execution registry')
    identities=[]
    for task in config['runs'].values():
        backbone=task.get('backbone');route=task.get('route');phase=task.get('phase')
        if backbone not in LRS or task.get('lr')!=LRS[backbone]:
            raise ValueError('fixed meeting learning rate recipe violated')
        if (task.get('microbatch')!=128 or task.get('precision')!='AMP'
            or not isinstance(task.get('epochs'),int) or not 1<=task['epochs']<=60):
            raise ValueError('invalid fixed training recipe or epoch budget')
        if route not in ('D','H'):raise ValueError('unknown training route')
        if phase=='development':
            if task.get('seed') not in DEVELOPMENT_SEEDS:raise ValueError('unregistered development seed')
            if task.get('recipe_frozen') is not False:raise ValueError('development must not masquerade as formal')
        elif phase=='formal':
            expected=SEEDS if route=='D' else H_SEEDS
            if task.get('seed') not in expected:raise ValueError('unregistered formal seed')
            token=task.get('epoch_recipe_sha256','')
            if task.get('recipe_frozen') is not True or len(token)!=64 or any(c not in '0123456789abcdef' for c in token):
                raise ValueError('formal requires frozen epoch recipe hash')
        else:raise ValueError('unregistered training phase')
        if route=='D':
            if task.get('train_manifest')!='data/D/train.parquet' or task.get('dev_manifest')!='data/D/dev.parquet':
                raise ValueError('wrong D source manifest')
            cdf=task.get('cdf_development_manifest')
            if phase=='development' and cdf!='data/D/cdf_dev.parquet':
                raise ValueError('CDF development may use only isolated cdf_dev manifest')
            if phase=='formal' and cdf is not None:
                raise ValueError('formal trajectories cannot evaluate CDF development or TEST')
        else:
            history=task.get('history','B')
            if history not in ('B','AB') or task.get('train_manifest')!=f'data/H/{history}/train.parquet' or task.get('dev_manifest')!='data/H/B/dev.parquet':
                raise ValueError('wrong H paired source manifests')
            if task.get('cdf_development_manifest') is not None:raise ValueError('H never selects with CDF or DFD')
        key=(route,phase,backbone,task.get('seed'),task.get('history','B'),task['epochs'] if phase=='development' else None)
        if key in identities:raise ValueError('duplicate registered trajectory')
        identities.append(key)


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
