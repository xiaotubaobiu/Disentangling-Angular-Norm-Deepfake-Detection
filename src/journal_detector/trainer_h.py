"""Paired B/AB training with common source-B validation epoch and fixed recipe."""
from contextlib import ExitStack, nullcontext
from pathlib import Path
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from . import trainer as t
from .protocol import validate_config

H_SELECTION='paired_source_dev_B_mean_probability_auc_ema'
training_identity=t.training_identity


class PairedSelection(t.Selection):
    def observe(self,b_auc,ab_auc,epoch):
        return super().observe((b_auc+ab_auc)/2,epoch)

    @classmethod
    def from_state(cls,state):return cls(**state)


def make_pair_selection(btask,abtask):
    """Keep source-best EMA selection while optionally running the full budget."""
    left=btask.get('fixed_epoch_budget',False)
    right=abtask.get('fixed_epoch_budget',False)
    if type(left) is not bool or type(right) is not bool or left!=right:
        raise ValueError('H pair fixed epoch budget must be matching booleans')
    if btask['epochs']!=abtask['epochs']:
        raise ValueError('H pair epoch budget differs')
    return PairedSelection(patience=btask['epochs']+1 if left else 5)


def validate_pair_frames(b,ab):
    for name,frame in [('B',b),('AB',ab)]:
        if frame.empty or frame.mother_id.duplicated().any():
            raise ValueError('missing or duplicate paired mother: '+name)
    for key in ['mother_id','video_uid','method_id','label']:
        if b[key].tolist()!=ab[key].tolist():raise ValueError('paired mother order/metadata mismatch: '+key)
    if not np.array_equal(b.sampling_mass.to_numpy(),ab.sampling_mass.to_numpy()):
        raise ValueError('paired sampling masses differ')


def paired_step(models,optimizers,scalers,batches,attempt,steps,epochs,lr,micro,device,seed):
    events={}
    # Same backbone, same dropout/drop-path random stream despite sequential
    # execution. Image augmentation has its separate mother/draw keyed stream.
    for history in ['B','AB']:
        t.set_random(int(seed)+int(attempt)*100003)
        x,y=batches[history]
        events[history]=t.advance_step(models[history],optimizers[history],scalers[history],
            x,y,attempt,steps,epochs,lr,micro,device)
    return events


def _save_checkpoint(out,state,identity,selection):
    temporary=out/'best_model.pth.tmp'
    torch.save({'model':state,'identity':identity,'epoch':selection.best_epoch,
        'auc':selection.best_score,'weights':'ema','ema_decay':.999,'selection':H_SELECTION},temporary)
    os.replace(temporary,out/'best_model.pth')


def train_pair(config_path,b_run,ab_run,device='cuda'):
    config=json.loads(Path(config_path).read_text());root=Path(config['artifact_root'])
    with ExitStack() as stack:
        for run in sorted([b_run,ab_run]):stack.enter_context(t.exclusive_file(root/'runs'/run/'run.lock'))
        if str(device).startswith('cuda'):stack.enter_context(t.owned_gpu(root,config))
        return _train_pair(config_path,b_run,ab_run,device)


def _train_pair(config_path,b_run,ab_run,device='cuda'):
    from .models import build_model
    from .data_h import validate_h_freeze
    t.policy();config=json.loads(Path(config_path).read_text());validate_config(config)
    root=Path(config['artifact_root']);tasks={h:config['runs'][run] for h,run in [('B',b_run),('AB',ab_run)]}
    btask,abtask=tasks['B'],tasks['AB']
    if btask.get('history','B')!='B' or abtask.get('history')!='AB':raise ValueError('H histories must be B/AB')
    for key in ['backbone','seed','phase','lr','epochs','microbatch','precision','dev_manifest']:
        if btask[key]!=abtask[key]:raise ValueError('H pair recipe differs: '+key)
    if btask['route']!='H' or abtask['route']!='H':raise ValueError('H pair route required')
    freeze=validate_h_freeze(root,source_only=True)
    frame={h:pd.read_parquet(root/task['train_manifest']) for h,task in tasks.items()}
    validate_pair_frames(frame['B'],frame['AB'])
    dev=pd.read_parquet(root/'data/H/B/dev.parquet')
    ids={h:t.training_identity(config,task) for h,task in tasks.items()}
    outputs={h:root/'runs'/run for h,run in [('B',b_run),('AB',ab_run)]}
    for h,out in outputs.items():
        out.mkdir(parents=True,exist_ok=True);t.save_json(out/'identity.json',ids[h]);t.save_json(out/'resolved_config.json',tasks[h])
    completed=[(out/'complete.json').exists() for out in outputs.values()]
    if all(completed):
        results={h:json.loads((out/'complete.json').read_text()) for h,out in outputs.items()}
        for h,result in results.items():
            if result['identity']!=ids[h] or result['best_sha256']!=t.digest(outputs[h]/'best_model.pth'):
                raise ValueError('completed H model identity changed')
        return results
    backbone=btask['backbone'];seed=btask['seed'];epochs=btask['epochs'];lr=btask['lr'];micro=btask['microbatch']
    models={};opts={};scalers={};emas={}
    for h in ['B','AB']:
        t.set_random(seed)
        models[h]=build_model(backbone,config['backbones'][backbone]['weight_path']).to(device)
        opts[h]=torch.optim.AdamW(models[h].parameters(),lr=lr,weight_decay=5e-4)
        scalers[h]=torch.amp.GradScaler('cuda',init_scale=128) if str(device).startswith('cuda') else None
        emas[h]=t.EMA(models[h])
    draw_count=math.ceil(len(frame['B'])/128)*128;steps=draw_count//128
    selection=make_pair_selection(btask,abtask);best={};histories={h:[] for h in models};skips={h:0 for h in models}
    start=0;attempts=0
    pair_root=root/'pairs'/f"{backbone}_seed{seed}_{btask['phase']}_cap{epochs}";pair_root.mkdir(parents=True,exist_ok=True)
    state_path=pair_root/'state.pt';pair_identity={'B':ids['B'],'AB':ids['AB']}
    if state_path.exists():
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        if state['identity']!=pair_identity:raise ValueError('H paired resume identity mismatch')
        selection=PairedSelection.from_state(state['selection']);start=state['epoch'];attempts=state['attempts']
        best=state['best'];histories=state['histories'];skips=state['consecutive_skips']
        for h in models:
            models[h].load_state_dict(state[h]['model']);opts[h].load_state_dict(state[h]['optimizer'])
            if scalers[h] is not None:scalers[h].load_state_dict(state[h]['scaler'])
            emas[h].load_state_dict(state[h]['ema']);_save_checkpoint(outputs[h],best[h],ids[h],selection)
    for epoch in range(start,epochs):
        if selection.stopped:break
        begun=time.time();draws,methods=t.epoch_draws(frame['B'],seed,epoch+1,draw_count)
        for h in models:
            models[h].train();np.save(outputs[h]/f'epoch_{epoch+1:02d}_samples.npy',draws,allow_pickle=False)
            np.save(outputs[h]/f'epoch_{epoch+1:02d}_methods.npy',methods,allow_pickle=False)
        loaders={h:t.loader(frame[h],backbone,128,draws,seed,epoch+1,config.get('workers',4)) for h in models}
        successes={h:0 for h in models};losses={h:0. for h in models}
        with ExitStack() as stack:
            streams={h:stack.enter_context((outputs[h]/f'epoch_{epoch+1:02d}_updates.jsonl').open('w')) for h in models}
            for b_batch,ab_batch in zip(loaders['B'],loaders['AB'],strict=True):
                if not torch.equal(b_batch[1],ab_batch[1]):raise ValueError('H paired labels differ in batch')
                events=paired_step(models,opts,scalers,{'B':b_batch,'AB':ab_batch},attempts,steps,epochs,lr,micro,device,seed)
                for h,event in events.items():
                    if event['success']:emas[h].update(models[h])
                    successes[h]+=event['success'];losses[h]+=event['loss']
                    skips[h]=skips[h]+1 if event['skip'] else 0
                    event.update(epoch=epoch+1);streams[h].write(json.dumps(event,allow_nan=False)+'\n')
                    if skips[h]>=8:raise FloatingPointError('H 8 consecutive AMP skips: '+h)
                attempts+=1
        metrics={};scores={}
        with ExitStack() as stack:
            for h in models:stack.enter_context(emas[h].applied(models[h]))
            for h in models:metrics[h],scores[h]=t.source_metrics(models[h],dev,backbone,micro,device,config.get('workers',4))
            improved=selection.observe(metrics['B']['pooled_auc'],metrics['AB']['pooled_auc'],epoch+1)
            if improved:best={h:{k:v.detach().cpu().clone() for k,v in model.state_dict().items()} for h,model in models.items()}
        for h in models:
            scores[h].to_parquet(outputs[h]/f'epoch_{epoch+1:02d}_source_dev_B.parquet',index=False)
            row={'epoch':epoch+1,'lr':t.epoch_lr(epoch,epochs,lr),'train_loss':losses[h]/steps,
                'attempts':steps,'success':successes[h],'skip':steps-successes[h],'draw_count':draw_count,
                'dev':metrics[h],'paired_dev':metrics,'selection':selection.state_dict(),'seconds':time.time()-begun}
            histories[h].append(row);t.save_json(outputs[h]/'history.json',histories[h])
        state={'identity':pair_identity,'epoch':epoch+1,'attempts':attempts,'selection':selection.state_dict(),
               'best':best,'histories':histories,'consecutive_skips':skips}
        for h in models:state[h]={'model':models[h].state_dict(),'optimizer':opts[h].state_dict(),
            'scaler':scalers[h].state_dict() if scalers[h] is not None else None,'ema':emas[h].state_dict()}
        tmp=pair_root/'state.pt.tmp';torch.save(state,tmp);os.replace(tmp,state_path)
        if improved:
            for h in models:_save_checkpoint(outputs[h],best[h],ids[h],selection)
        print(json.dumps({'pair':str(pair_root),'epoch':epoch+1,'dev':metrics,'selection':selection.state_dict(),'seconds':time.time()-begun}),flush=True)
    if not best:raise ValueError('H pair produced no valid source-selected weights')
    results={}
    for h in models:
        result={'status':'TRAINING_COMPLETE','route':'H','history':h,'identity':ids[h],
            'phase':btask['phase'],'epochs':len(histories[h]),'max_epochs':epochs,'best_epoch':selection.best_epoch,
            'early_stopped':selection.stopped,'selection':H_SELECTION,'target_used':False,'target_development_used':False,
            'epoch_recipe_sha256':tasks[h].get('epoch_recipe_sha256'),'best_sha256':t.digest(outputs[h]/'best_model.pth'),
            'attempts':attempts,'success':sum(x['success'] for x in histories[h]),'skip':sum(x['skip'] for x in histories[h]),
            'pair_state_path':str(state_path),'pair_state_sha256':t.digest(state_path),'ema_decay':.999}
        t.save_json(outputs[h]/'complete.json',result);results[h]=result
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--b-run',required=True);p.add_argument('--ab-run',required=True)
    a=p.parse_args();print(json.dumps(train_pair(a.config,a.b_run,a.ab_run),indent=2))
