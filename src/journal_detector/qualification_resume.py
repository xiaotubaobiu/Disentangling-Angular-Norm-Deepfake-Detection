"""Bounded source-only GPU qualification of optimizer/RNG/EMA recovery.

This checks six continuous real-source updates against two saved updates plus
four restored updates. It is not a development or formal model and performs
no target evaluation or checkpoint selection. Paired H epoch/selection-state
recovery is covered separately by its production-loop CPU integration test.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime,timezone

import numpy as np
import pandas as pd
import torch

from . import trainer as t


def state_digest(value):
    def encode(item):
        if torch.is_tensor(item):
            tensor=item.detach().cpu().contiguous()
            return dict(dtype=str(tensor.dtype),shape=list(tensor.shape),
                sha256=hashlib.sha256(tensor.numpy().tobytes()).hexdigest())
        if isinstance(item,dict):
            return [[repr(key),encode(item[key])] for key in sorted(item,key=repr)]
        if isinstance(item,(list,tuple)):return [encode(x) for x in item]
        return item
    return hashlib.sha256(json.dumps(encode(value),sort_keys=True,allow_nan=False).encode()).hexdigest()


def _snapshots(model,optimizer,scaler,ema):
    return dict(model=state_digest(model.state_dict()),optimizer=state_digest(optimizer.state_dict()),
        scaler=state_digest(scaler.state_dict() if scaler is not None else None),
        ema=state_digest(ema.state_dict()))


def qualify_model(factory,batches,checkpoint_path,identity,*,seed,lr,device):
    if len(batches)!=6 or any(len(y)!=128 or len(x)!=128 for x,y in batches):
        raise ValueError('resume qualification requires six actual batch128 inputs')
    t.policy();t.set_random(seed)
    model=factory().to(device);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=5e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=128) if str(device).startswith('cuda') else None
    ema=t.EMA(model);events=[]
    for attempt,(x,y) in enumerate(batches):
        event=t.advance_step(model,optimizer,scaler,x,y,attempt,2,3,lr,128,device)
        if not event['success']:raise ValueError('resume qualification encountered an AMP skip')
        ema.update(model);events.append(event)
        if attempt==1:
            t.save_state(checkpoint_path,model,optimizer,scaler,1,identity,[],attempts=2,ema=ema)
    continuous=_snapshots(model,optimizer,scaler,ema)
    del model,optimizer,scaler,ema
    gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    # Prove restore, rather than relying on the same incidental live RNG state.
    t.set_random(seed+11111)
    model=factory().to(device);model.train()
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=5e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=128) if str(device).startswith('cuda') else None
    ema=t.EMA(model)
    saved=t.load_state(checkpoint_path,model,optimizer,scaler,identity)
    ema.load_state_dict(saved['ema']);resumed=[]
    for attempt in range(saved['attempts'],len(batches)):
        x,y=batches[attempt]
        event=t.advance_step(model,optimizer,scaler,x,y,attempt,2,3,lr,128,device)
        if not event['success']:raise ValueError('restored qualification encountered an AMP skip')
        ema.update(model);resumed.append(event)
    restored=_snapshots(model,optimizer,scaler,ema)
    matches={name:continuous[name]==restored[name] for name in continuous}
    matches['update_events']=events[2:]==resumed
    if not all(matches.values()):raise ValueError('resume matching state failed: '+str(matches))
    result=dict(status='QUALIFIED',device=str(device),continuous_success=len(events),resumed_success=len(resumed),
        matching_state=matches,continuous_state_sha256=continuous,restored_state_sha256=restored,
        checkpoint_sha256=t.digest(checkpoint_path),random_stream_perturbed_before_reload=True,
        continuous_learning_rates=[event['lr'] for event in events],target_used=False)
    del model,optimizer,scaler,ema,saved
    gc.collect()
    if str(device).startswith('cuda'):torch.cuda.empty_cache()
    return result


def source_batches(frame,backbone):
    reals=frame[frame.label.eq(0)].reset_index(drop=True)
    fakes=frame[frame.label.eq(1)].reset_index(drop=True)
    if reals.empty or fakes.empty or not frame.role.eq('train').all():
        raise ValueError('resume qualification accepts frozen two-class source TRAIN only')
    batches=[];input_hash=hashlib.sha256()
    for attempt in range(6):
        images=[];labels=[]
        for j in range(128):
            label=j%2;pool=fakes if label else reals
            row=pool.iloc[(attempt*64+j//2)%len(pool)]
            params=t.augmentation_parameters(7,attempt+1,str(row.sample_id),j)
            images.append(t.image_tensor(row.image_path,backbone,params));labels.append(label)
        x=torch.stack(images);y=torch.tensor(labels,dtype=torch.long)
        input_hash.update(x.numpy().tobytes());input_hash.update(y.numpy().tobytes());batches.append((x,y))
    return batches,input_hash.hexdigest()


def run(config_path):
    from .models import build_model
    from .protocol import LRS,validate_config
    config=json.loads(Path(config_path).read_text());root=Path(config['artifact_root']).resolve()
    validate_config(config)
    output=root/'qualification/gpu_resume';output.mkdir(parents=True,exist_ok=True)
    ready=root/'operations/gpu_resume_qualification.json'
    with t.exclusive_file(output/'qualification.lock'),t.owned_gpu(root,config):
        begun=time.monotonic()
        t.policy();source=root/'data/D/qualification_source.parquet'
        freeze=json.loads((root/'data/D/freeze.json').read_text())
        if t.digest(source)!=freeze['manifest_sha256'].get(source.name):
            raise ValueError('qualification source differs from frozen TRAIN evidence')
        frame=pd.read_parquet(source)
        result=dict(status='QUALIFICATION_RUNNING',target_used=False,
            started_utc=datetime.now(timezone.utc).isoformat(),gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
            scope='Source TRAIN core optimizer/RNG/EMA GPU recovery; not a model selection or formal trajectory.',
            source_sha256=t.digest(source),data_freeze_sha256=t.digest(root/'data/D/freeze.json'),
            code_sha256=t.digest(__file__),training_code_sha256=t.code_hash('D'),results={})
        try:
            for backbone in config['backbones']:
                spec=config['backbones'][backbone];weight=Path(spec['weight_path'])
                if t.digest(weight)!=spec['weight_sha256']:raise ValueError('qualification initialization hash mismatch')
                batches,input_hash=source_batches(frame,backbone)
                identity=dict(phase='GPU_RESUME_QUALIFICATION',backbone=backbone,
                    initialization_sha256=spec['weight_sha256'],input_sha256=input_hash,
                    data_freeze_sha256=result['data_freeze_sha256'],training_code_sha256=result['training_code_sha256'])
                checkpoint=output/backbone/'two_update_state.pt'
                receipt=qualify_model(lambda:build_model(backbone,weight),batches,checkpoint,identity,
                    seed=7,lr=LRS[backbone],device='cuda')
                receipt.update(identity=identity,batch_size=128)
                result['results'][backbone]=receipt;t.save_json(output/f'{backbone}.json',receipt)
                print(json.dumps(dict(backbone=backbone,status=receipt['status'],matching=receipt['matching_state'])),flush=True)
                del batches
            result['status']='QUALIFIED'
            result['elapsed_seconds']=time.monotonic()-begun
            t.save_json(ready,result)
        except Exception as exc:
            result.update(status='FAILED',error=repr(exc))
            t.save_json(root/'operations/gpu_resume_qualification_failed.json',result)
            raise
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();run(args.config)
