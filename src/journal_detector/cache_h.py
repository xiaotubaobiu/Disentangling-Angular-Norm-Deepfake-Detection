"""Frozen journal H paired-EMA FP32 feature caches, independent of D models."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd
import torch

from . import cache_d as arrays
from . import data_h, trainer

SCHEMA='journal_detector-H-paired-best-ema-fp32-cache-1'
ROLES=('B/source_cal','B/source_test','B/target_test','common/B/source_test',
       'common/B/target_test','common/AB/source_test','common/AB/target_test')
SEEDS=(7,42,123)
DIMENSIONS=arrays.DIMENSIONS
SELECTION='paired_source_dev_B_mean_probability_auc_ema'


def _required_hash(path,description):
    return arrays._required_hash(path,description)


def _registry(config,backbone):
    tasks={run:task for run,task in config.get('runs',{}).items()
        if task.get('route')=='H' and task.get('phase')=='formal' and task.get('backbone')==backbone}
    expected={(seed,history) for seed in SEEDS for history in ('B','AB')}
    if (len(tasks)!=6 or {(t['seed'],t.get('history')) for t in tasks.values()}!=expected
            or any(t.get('recipe_frozen') is not True for t in tasks.values())):
        raise ValueError('H requires six frozen formal runs: three exact B/AB seed pairs')
    return tasks


def _completed(root,run,task,config):
    out=root/'runs'/run
    checkpoint=out/'best_model.pth'
    if checkpoint.resolve().parent!=out.resolve() or out.resolve().parent!=(root/'runs').resolve():
        raise ValueError('H checkpoint must belong to the registered new local run')
    identity=trainer.training_identity(config,task)
    marker=out/'complete.json'
    sha=_required_hash(marker,'H formal training completion')
    receipt=json.loads(marker.read_text())
    if (receipt.get('status')!='TRAINING_COMPLETE' or receipt.get('route')!='H'
            or receipt.get('phase')!='formal' or receipt.get('history')!=task['history']
            or receipt.get('identity')!=identity or receipt.get('selection')!=SELECTION
            or receipt.get('target_used') is not False or receipt.get('target_development_used') is not False
            or receipt.get('epoch_recipe_sha256')!=task.get('epoch_recipe_sha256')):
        raise ValueError('H formal paired EMA identity/selection/development provenance mismatch')
    if (receipt.get('max_epochs')!=task['epochs'] or not 1<=receipt.get('epochs',0)<=task['epochs']
            or not 1<=receipt.get('best_epoch',0)<=receipt['epochs']):
        raise ValueError('H formal epoch identity is invalid')
    final=_required_hash(checkpoint,'H best paired EMA checkpoint')
    if receipt.get('best_sha256')!=final:
        raise ValueError('H best paired EMA checkpoint hash mismatch')
    pair=Path(receipt.get('pair_state_path','')).resolve()
    if not pair.is_relative_to(root/'pairs') or _required_hash(pair,'H paired recovery state')!=receipt.get('pair_state_sha256'):
        raise ValueError('H paired recovery state provenance mismatch')
    return receipt,identity,sha,final


def _context(config_path,run_id):
    from .protocol import PROTOCOL,validate_config
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id) or run_id in ('.','..'):
        raise ValueError('unsafe H run identity')
    path=Path(config_path).resolve();config=json.loads(path.read_text())
    root=Path(config['artifact_root']).resolve()
    task=config.get('runs',{}).get(run_id)
    if (not task or task.get('route')!='H' or task.get('phase')!='formal'
            or task.get('recipe_frozen') is not True):
        raise ValueError('only a frozen formal H run can export H target features')
    allowed={root/'configs/h_execution.json',root/f"configs/h_execution_{task['backbone']}.json"}
    if path not in allowed or config.get('protocol')!=PROTOCOL:
        raise ValueError('dedicated new frozen H execution config must match the selected backbone')
    validate_config(config)
    backbone=task['backbone'];tasks=_registry(config,backbone)
    data_h.validate_h_freeze(root,source_only=True)
    data_h.validate_h_freeze(root,source_only=False)
    partner_history='AB' if task['history']=='B' else 'B'
    partner=next(run for run,t in tasks.items() if t['seed']==task['seed'] and t['history']==partner_history)
    other=tasks[partner]
    for key in ('seed','backbone','epochs','lr','microbatch','precision','dev_manifest','epoch_recipe_sha256','epoch_recipe_path'):
        if task.get(key)!=other.get(key):raise ValueError('H paired frozen recipe differs: '+key)
    if task['train_manifest']!=f"data/H/{task['history']}/train.parquet" or task['dev_manifest']!='data/H/B/dev.parquet':
        raise ValueError('H paired source manifest identity mismatch')
    receipt,identity,complete_sha,best_sha=_completed(root,run_id,task,config)
    paired,_,paired_sha,paired_best=_completed(root,partner,other,config)
    for key in ('best_epoch','epochs','max_epochs','pair_state_path','pair_state_sha256'):
        if receipt.get(key)!=paired.get(key):raise ValueError('H models do not share the same selected paired epoch/state')
    recipe=Path(task['epoch_recipe_path'])
    if not recipe.is_absolute():recipe=root/recipe
    recipe=recipe.resolve()
    if not recipe.is_relative_to(root) or _required_hash(recipe,'frozen H epoch recipe')!=task['epoch_recipe_sha256']:
        raise ValueError('H frozen epoch recipe hash/path mismatch')
    proof=json.loads(recipe.read_text())
    if (proof.get('schema')!='EPOCH_RECIPE_V2' or proof.get('status')!='FROZEN'
            or proof.get('route')!='H' or proof.get('target_development_used') is not False
            or proof.get('paired_histories')!=['B','AB']):
        raise ValueError('H requires its source-only paired epoch recipe')
    base=dict(schema=SCHEMA,route='H',config_path=str(path),config_sha256=trainer.digest(path),
        artifact_root=str(root),run_id=run_id,backbone=backbone,history=task['history'],seed=task['seed'],
        dimension=DIMENSIONS[backbone],epochs=receipt['epochs'],best_epoch=receipt['best_epoch'],
        selection=SELECTION,target_development_used=False,formal_trajectory_target_used=False,
        training_identity=identity,checkpoint_path=str(root/'runs'/run_id/'best_model.pth'),
        checkpoint_sha256=best_sha,training_complete_sha256=complete_sha,
        partner_run_id=partner,partner_training_complete_sha256=paired_sha,partner_checkpoint_sha256=paired_best,
        source_data_freeze_sha256=trainer.digest(root/'data/H/train_dev_freeze.json'),
        full_data_freeze_sha256=trainer.digest(root/'data/H/freeze.json'),
        epoch_recipe_path=str(recipe),epoch_recipe_sha256=task['epoch_recipe_sha256'],
        cache_code_sha256=trainer.digest(__file__),cache_helpers_sha256=trainer.digest(arrays.__file__))
    return base,config,task


def _role_identity(base,role,manifest):
    root=Path(base['artifact_root']);manifest=Path(manifest).resolve()
    if role not in ROLES or manifest!=root/f'data/H/{role}.parquet':
        raise ValueError('unregistered H role manifest identity')
    freeze=json.loads((root/'data/H/freeze.json').read_text())
    sha=_required_hash(manifest,'frozen H role manifest')
    if freeze.get('tables',{}).get(role,{}).get('sha256')!=sha:
        raise ValueError('H role manifest differs from the full frozen cohort')
    return dict(base,role=role,source_manifest_path=str(manifest),source_manifest_sha256=sha)


def _validate_live(identity):
    base,_,_=_context(identity['config_path'],identity['run_id'])
    if identity!=_role_identity(base,identity['role'],identity['source_manifest_path']):
        raise ValueError('H cache live provenance identity/hash mismatch')


def _read_manifest(path,role):
    frame=arrays._read_manifest(path)
    required={'mother_id','mother_rgb_sha256','video_uid','video_id','method_id','group_id','domain','role','processing_version','rgb_sha256','content_sha256'}
    if not required.issubset(frame) or frame[list(required)].isna().any().any():
        raise ValueError('H frozen logical mother/input provenance is missing')
    leaf=role.split('/')[-1];version='AB' if role.startswith('common/AB/') else 'B'
    domain='target' if leaf=='target_test' else 'source'
    if (not frame.role.eq(leaf).all() or not frame.processing_version.eq(version).all()
            or not frame.domain.eq(domain).all() or frame.mother_id.duplicated().any()):
        raise ValueError('H role/version/domain or unique mother identity mismatch')
    sizes=frame.groupby('video_uid').size()
    if sizes.ne(8).any() or frame.groupby('video_uid').mother_id.nunique().ne(8).any() or frame.groupby('video_uid').rgb_sha256.nunique().ne(8).any():
        raise ValueError('H natural videos require exactly eight distinct frozen mothers/RGB frames')
    if leaf=='source_cal' and not frame.label.eq(0).all():raise ValueError('H source calibration is real-only')
    return frame


def _cache_role(root,manifest,identity,weight,bias,feature_batches,chunk_rows=4096):
    if not isinstance(chunk_rows,int) or chunk_rows<4096:raise ValueError('H chunks require at least 4096 rows')
    root=Path(root);manifest=Path(manifest);dimension=identity['dimension']
    weight,bias=arrays._validate_head(weight,bias,dimension)
    if trainer.digest(manifest)!=identity['source_manifest_sha256']:raise ValueError('H source manifest changed')
    frame=_read_manifest(manifest,identity['role']);root.mkdir(parents=True,exist_ok=True)
    header=root/'identity.json'
    if header.exists() and json.loads(header.read_text())!=identity:raise ValueError('H resumed cache identity mismatch')
    trainer.save_json(header,identity)
    saved=root/'manifest.parquet'
    if saved.exists():
        if trainer.digest(saved)!=identity['source_manifest_sha256']:raise ValueError('H cached manifest hash changed')
    else:
        temporary=root/'manifest.parquet.tmp';shutil.copyfile(manifest,temporary);os.replace(temporary,saved)
    head={}
    for key,value in (('weight',weight),('bias',bias)):
        path=root/f'head_{key}.npy'
        if path.exists():
            prior=np.load(path,allow_pickle=False)
            if prior.dtype!=np.float32 or not np.array_equal(prior,value):raise ValueError('H resumed original FP32 head changed')
        else:arrays._atomic_array(path,value)
        head[key+'_file']=path.name;head[key+'_sha256']=trainer.digest(path)
    chunks,position=arrays._read_chunks(root,frame,identity,chunk_rows)
    start=position;pending=[];count=0
    def commit(value):
        nonlocal position
        number=len(chunks);name=f'chunk_{number:06d}.npy'
        zeros=arrays._validate_features(value,len(value),dimension)
        item=dict(file=name,start=position,rows=len(value),dimension=dimension,dtype='float32',
            identity_sha256=arrays._json_hash(identity),row_sha256=arrays._row_hash(frame,position,len(value)),zero_norm_count=zeros)
        arrays._atomic_array(root/name,value);item['sha256']=trainer.digest(root/name)
        trainer.save_json(root/f'chunk_{number:06d}.json',item);chunks.append(item);position+=len(value)
    if start<len(frame):
        for value in feature_batches(start):
            value=np.asarray(value);arrays._validate_features(value,len(value),dimension)
            if not len(value) or position+count+len(value)>len(frame):raise ValueError('H feature iterator row mismatch')
            pending.append(value);count+=len(value)
            while count>=chunk_rows:
                block=np.concatenate(pending,axis=0);commit(block[:chunk_rows]);remaining=block[chunk_rows:]
                pending=[remaining] if len(remaining) else [];count=len(remaining)
        if position+count!=len(frame):raise ValueError('H feature iterator ended early')
        if count:commit(np.concatenate(pending,axis=0))
    if trainer.digest(manifest)!=identity['source_manifest_sha256']:raise ValueError('H manifest changed during extraction')
    meta=dict(schema=SCHEMA,status='CACHE_COMPLETE',identity=identity,rows=len(frame),dimension=dimension,
        precision='FP32',autocast=False,epsilon=arrays.EPSILON,chunk_rows=chunk_rows,
        zero_norm_count=sum(item['zero_norm_count'] for item in chunks),
        cached_manifest_sha256=trainer.digest(saved),head=head,chunks=chunks)
    trainer.save_json(root/'complete.json',meta)
    return meta


def verify_cache(root,identity=None,*,expected_run_id=None,expected_role=None,config_path=None):
    root=Path(root);marker=root/'complete.json'
    if not marker.is_file():raise ValueError('H feature cache is incomplete')
    meta=json.loads(marker.read_text());saved=meta.get('identity',{})
    if meta.get('schema')!=SCHEMA or meta.get('status')!='CACHE_COMPLETE' or saved.get('schema')!=SCHEMA:
        raise ValueError('unknown or incomplete journal H cache schema')
    if identity is not None and identity!=saved:raise ValueError('H requested cache identity mismatch')
    for actual,expected in ((saved.get('run_id'),expected_run_id),(saved.get('role'),expected_role)):
        if expected is not None and actual!=expected:raise ValueError('H requested run/role mismatch')
    if config_path is not None and Path(config_path).resolve()!=Path(saved['config_path']):raise ValueError('H config path mismatch')
    canonical=Path(saved['artifact_root'])/'features/H'/saved['run_id']/saved['role']
    if root.resolve()!=canonical.resolve():raise ValueError('H cache must occupy its canonical registered role path')
    _validate_live(saved)
    if json.loads((root/'identity.json').read_text())!=saved:raise ValueError('H cache identity header changed')
    manifest=root/'manifest.parquet'
    if trainer.digest(manifest)!=meta.get('cached_manifest_sha256') or meta['cached_manifest_sha256']!=saved['source_manifest_sha256']:
        raise ValueError('H cached manifest hash mismatch')
    frame=_read_manifest(manifest,saved['role']);chunk_rows=meta.get('chunk_rows')
    if not isinstance(chunk_rows,int) or chunk_rows<4096:raise ValueError('H chunk row budget is invalid')
    chunks,rows=arrays._read_chunks(root,frame,saved,chunk_rows)
    if (chunks!=meta.get('chunks') or rows!=len(frame) or rows!=meta.get('rows')
            or meta.get('dimension')!=saved['dimension'] or meta.get('precision')!='FP32'
            or meta.get('autocast') is not False or meta.get('epsilon')!=arrays.EPSILON
            or meta.get('zero_norm_count')!=sum(item['zero_norm_count'] for item in chunks)):
        raise ValueError('H cache rows/dtype/precision/chunks mismatch')
    head=[]
    for key in ('weight','bias'):
        item=meta['head']
        if item.get(key+'_file')!=f'head_{key}.npy':raise ValueError('unsafe H original head path')
        path=root/item[key+'_file']
        if trainer.digest(path)!=item.get(key+'_sha256'):raise ValueError('H original head hash mismatch')
        head.append(np.load(path,allow_pickle=False))
    arrays._validate_head(*head,saved['dimension'])
    return meta,frame


def load_features(root,identity=None,**expected):
    root=Path(root);meta,frame=verify_cache(root,identity,**expected)
    h=np.concatenate([np.load(root/item['file'],allow_pickle=False) for item in meta['chunks']],axis=0)
    return frame,h,np.load(root/meta['head']['weight_file'],allow_pickle=False),np.load(root/meta['head']['bias_file'],allow_pickle=False)


def extract(config_path,run_id,device='cuda'):
    from .models import build_model
    config=json.loads(Path(config_path).read_text());root=Path(config['artifact_root']).resolve()
    if not isinstance(run_id,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id) or run_id in ('.','..'):
        raise ValueError('unsafe H run identity')
    with trainer.exclusive_file(root/'features/H'/run_id/'extract.lock'):
        ownership=contextlib.nullcontext() if torch.device(device).type=='cpu' else trainer.owned_gpu(root,config)
        with ownership:
            base,config,task=_context(config_path,run_id)
            checkpoint=torch.load(base['checkpoint_path'],map_location='cpu',weights_only=False)
            if (checkpoint.get('identity')!=base['training_identity'] or checkpoint.get('weights')!='ema'
                    or checkpoint.get('epoch')!=base['best_epoch'] or checkpoint.get('selection')!=SELECTION):
                raise ValueError('H loaded paired best EMA identity/epoch/selection mismatch')
            W=checkpoint['model']['classifier.weight'].detach().cpu().numpy()
            b=checkpoint['model']['classifier.bias'].detach().cpu().numpy();arrays._validate_head(W,b,base['dimension'])
            batch=config.get('cache_batch_size',task['microbatch']);workers=config.get('cache_workers',config.get('workers',4))
            chunk_rows=config.get('cache_chunk_rows',4096)
            if not isinstance(batch,int) or batch<1 or not isinstance(workers,int) or workers<0:raise ValueError('invalid H cache loader settings')
            trainer.policy();model=None;outputs={};out=root/'features/H'/run_id
            for role in ROLES:
                manifest=root/f'data/H/{role}.parquet';identity=_role_identity(base,role,manifest);folder=out/role
                if (folder/'complete.json').is_file():meta,_=verify_cache(folder,identity)
                else:
                    if model is None:
                        model=build_model(task['backbone'],config['backbones'][task['backbone']]['weight_path'])
                        model.load_state_dict(checkpoint['model'],strict=True);model=model.to(device=device,dtype=torch.float32).eval()
                    frame=_read_manifest(manifest,role)
                    def features(start,frame=frame):
                        loader=trainer.loader(frame.iloc[start:].reset_index(drop=True),task['backbone'],batch,workers=workers)
                        with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,enabled=False):
                            for x,_ in loader:
                                h=model.encoder(x.to(device=device,dtype=torch.float32))
                                if h.dtype!=torch.float32:raise ValueError('H encoder must export original FP32 features')
                                yield h.cpu().numpy().copy()
                    meta=_cache_role(folder,manifest,identity,W,b,features,chunk_rows)
                    verify_cache(folder,identity)
                outputs[role]=dict(path=str(folder),rows=meta['rows'],complete_sha256=trainer.digest(folder/'complete.json'))
            final,_,_=_context(config_path,run_id)
            if final!=base:raise ValueError('H formal provenance changed during extraction')
            receipt=dict(status='CACHE_COMPLETE',route='H',run_id=run_id,identity=base,roles=outputs,
                target_development_used=False,formal_trajectory_target_used=False)
            trainer.save_json(out/'complete.json',receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    parser.add_argument('--run-id',required=True);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    print(json.dumps(extract(args.config,args.run_id,args.device),allow_nan=False))
