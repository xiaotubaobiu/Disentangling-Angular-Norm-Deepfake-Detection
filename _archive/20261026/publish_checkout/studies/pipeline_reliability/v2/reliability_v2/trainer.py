"""V2 fixed attempted-update training, keyed augmentation and exact epoch recovery.

This runner never manages services and never selects recipes with target scores.
"""
from pathlib import Path
from collections import Counter
import contextlib
import fcntl
import hashlib
import json
import math
import os
import random
import re
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def save_json(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.tmp')
    temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(temp,p)


def policy():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)


def set_random(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_initialized():torch.cuda.manual_seed_all(seed)


def attempt_lr(attempt,steps_per_epoch,epochs,base_lr):
    total=steps_per_epoch*epochs
    if not 0<=attempt<total:raise ValueError('attempt outside frozen clock')
    if attempt<steps_per_epoch:return base_lr*(attempt+1)/steps_per_epoch
    denominator=max(1,total-steps_per_epoch-1)
    phase=(attempt-steps_per_epoch)/denominator
    return base_lr*(.05+.95*.5*(1+math.cos(math.pi*phase)))


def epoch_draws(df,seed,epoch,draw_count=None):
    n=draw_count or math.ceil(len(df)/128)*128
    if n%128:raise ValueError('frozen draw count must be multiple of128')
    p=df.sampling_mass.to_numpy(dtype=np.float64)
    if not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1.):
        raise ValueError('invalid canonical sampling mass')
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,7301]))
    draws=rng.choice(len(df),size=n,replace=True,p=p)
    methods=[]
    for idx in draws:
        row=df.iloc[int(idx)]
        if 'method_contributions' in df:
            contrib=json.loads(row.method_contributions)
            names=sorted(contrib);w=np.asarray([contrib[k] for k in names],dtype=np.float64)
            if not np.isclose(w.sum(),row.sampling_mass):raise ValueError('method contribution mass mismatch')
            methods.append(str(rng.choice(names,p=w/w.sum())))
        else:methods.append(str(row.method_id))
    return draws,np.asarray(methods,dtype='U64')


def augmentation_parameters(seed,epoch,sample_key,draw_index):
    text=f'{seed}/{epoch}/{sample_key}/{draw_index}'.encode()
    key=int.from_bytes(hashlib.sha256(text).digest()[:16],'little')
    rng=np.random.default_rng(key)
    flip=bool(rng.random()<.5); perturb=bool(rng.random()<.5)
    return {'flip':flip,'brightness':float(rng.uniform(.9,1.1)) if perturb else 1.,
            'contrast':float(rng.uniform(.9,1.1)) if perturb else 1.}


def normalization(backbone):
    if backbone=='clip_vit_b16':return ((.48145466,.45762756,.40821073),(.26862954,.26130258,.27577711))
    return ((.485,.456,.406),(.229,.224,.225))


def image_tensor(path,backbone,params=None):
    with Image.open(path) as source: im=source.convert('RGB').resize((224,224),Image.Resampling.BICUBIC)
    if params:
        if params['flip']:im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        im=ImageEnhance.Brightness(im).enhance(params['brightness'])
        im=ImageEnhance.Contrast(im).enhance(params['contrast'])
    mean,std=normalization(backbone)
    rgb=(np.asarray(im,dtype=np.float32)/255-np.asarray(mean,dtype=np.float32))/np.asarray(std,dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1)))


def empirical_threshold(scores,alpha=.05):
    x=np.asarray(scores,dtype=np.float64)
    if len(x)==0 or not np.isfinite(x).all():raise ValueError('finite real calibration scores required')
    # Smallest feasible boundary with score>=threshold and exact observed ties.
    k=math.floor(alpha*len(x)+1e-12)
    descending=np.sort(x)[::-1]
    return float(np.nextafter(descending[k],np.inf))


def auc_independent(labels,scores):
    """Pair counting via ordered tie groups, independent of sklearn's ROC code."""
    y=np.asarray(labels);s=np.asarray(scores,dtype=np.float64)
    if not np.isfinite(s).all() or set(y)!={0,1}:raise ValueError('two classes and finite scores required for AUC')
    order=np.argsort(s,kind='stable');y=y[order];s=s[order]
    _,starts,counts=np.unique(s,return_index=True,return_counts=True)
    neg_before=0.;credit=0.
    for start,n in zip(starts,counts):
        p=float(y[start:start+n].sum());neg=n-p
        credit+=p*(neg_before+.5*neg);neg_before+=neg
    return float(credit/((y==0).sum()*(y==1).sum()))


def fp64_margin(features,weight,bias):
    w=weight.detach().double();b=bias.detach().double()
    return features.double()@(w[1]-w[0])+(b[1]-b[0])


def save_state(path,model,optimizer,scaler,epoch,identity,history,attempts=None,best_model=None,best_epoch=None,best_score=None,consecutive_skips=0):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
      'scaler':scaler.state_dict() if scaler is not None else None,
      'epoch':epoch,'attempts':epoch if attempts is None else attempts,
      'scheduler_clock':'attempt_count','identity':identity,'history':history,
      'best_model':best_model,'best_epoch':best_epoch,'best_score':best_score,
      'consecutive_skips':consecutive_skips,
      'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
      'torch_rng':torch.get_rng_state(),
      'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}
    tmp=p.with_name(p.name+'.tmp');torch.save(state,tmp);os.replace(tmp,p)


def load_state(path,model,optimizer,scaler,identity):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['identity']!=identity:raise ValueError('checkpoint identity mismatch: config/data/code/initialization')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    if scaler is not None:scaler.load_state_dict(state['scaler'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state


def repair_history(out,state,steps):
    if state['attempts']!=state['epoch']*steps:raise ValueError('checkpoint attempt clock mismatch')
    if [r['epoch'] for r in state['history']]!=list(range(1,state['epoch']+1)):
        raise ValueError('checkpoint history is incomplete')
    save_json(Path(out)/'history.json',state['history'])


@contextlib.contextmanager
def exclusive_file(path):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:yield stream
        finally:fcntl.flock(stream,fcntl.LOCK_UN)


@contextlib.contextmanager
def owned_gpu(root,config):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in config['allowed_gpu_uuids'] or config['allowed_gpus']!=[0,1,2,3]:
        raise ValueError('single approved GPU0-3 UUID required')
    with exclusive_file(Path(root)/'gpu_locks'/f'{visible}.lock'):
        query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
        occupied={row.split(', ')[0] for row in query.strip().splitlines() if row.strip()}
        if visible in occupied:raise RuntimeError('assigned GPU has an existing compute process; refuse to share')
        memory=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True)
        used={u:int(m) for u,m in [r.split(', ') for r in memory.strip().splitlines()]}
        if used[visible]>=100:raise RuntimeError('assigned GPU is not idle')
        yield


def clip_gradients(parameters,max_norm,allow_nonfinite):
    params=[p for p in parameters if p.grad is not None]
    # FP32 squared norms can overflow even when every gradient is finite.
    norms=[torch.linalg.vector_norm(p.grad.detach(),2,dtype=torch.float64) for p in params]
    norm=torch.linalg.vector_norm(torch.stack(norms),2) if norms else torch.tensor(0.,dtype=torch.float64)
    if not torch.isfinite(norm):
        if not allow_nonfinite:raise FloatingPointError('nonfinite gradient norm')
        return norm  # GradScaler already recorded elementwise nonfinite gradients.
    coef=min(1.,float(max_norm/(norm+1e-6)))
    for p in params:p.grad.mul_(coef)
    return norm


def advance_step(model,optimizer,scaler,x,y,attempt,steps_per_epoch,epochs,base_lr,microbatch,device):
    lr=attempt_lr(attempt,steps_per_epoch,epochs,base_lr)
    for group in optimizer.param_groups:group['lr']=lr
    optimizer.zero_grad(set_to_none=True);total=0.;n=len(y)
    amp=scaler is not None
    for j in range(0,n,microbatch):
        context=torch.autocast('cuda',dtype=torch.float16) if amp else contextlib.nullcontext()
        with context:
            loss=torch.nn.functional.cross_entropy(model(x[j:j+microbatch].to(device)),y[j:j+microbatch].to(device),reduction='sum')/n
        if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss; source protocol must be corrected before continuation')
        total+=float(loss.detach())
        if amp:scaler.scale(loss).backward()
        else:loss.backward()
    if amp:scaler.unscale_(optimizer)
    grad=clip_gradients(model.parameters(),1.,amp)
    scale_before=float(scaler.get_scale()) if amp else 1.
    if amp:scaler.step(optimizer);scaler.update()
    else:optimizer.step()
    scale_after=float(scaler.get_scale()) if amp else 1.
    skipped=bool(amp and scale_after<scale_before)
    return {'attempt':int(attempt),'lr':float(lr),'loss':total,
      'gradient_norm':float(grad) if torch.isfinite(grad) else None,
      'gradient_finite':bool(torch.isfinite(grad)),'scale_before':scale_before,
      'scale_after':scale_after,'success':not skipped,'skip':skipped}


class DrawDataset(torch.utils.data.Dataset):
    def __init__(self,frame,backbone,draws=None,seed=0,epoch=0):
        self.frame=frame;self.backbone=backbone;self.draws=draws
        self.seed=seed;self.epoch=epoch
    def __len__(self):return len(self.draws) if self.draws is not None else len(self.frame)
    def __getitem__(self,i):
        index=int(self.draws[i]) if self.draws is not None else i
        row=self.frame.iloc[index];params=None
        if self.draws is not None:
            key=str(row.get('mother_id',row.get('sample_id',row.rgb_sha256)))
            params=augmentation_parameters(self.seed,self.epoch,key,i)
        return image_tensor(row.image_path,self.backbone,params),int(row.label)


def loader(frame,backbone,batch,draws=None,seed=0,epoch=0,workers=4):
    # Use a private torch Generator: constructing an iterator must not consume
    # model/dropout RNG, and worker process scheduling cannot change augmentation.
    gen=torch.Generator().manual_seed(seed+epoch*100003)
    return torch.utils.data.DataLoader(DrawDataset(frame,backbone,draws,seed,epoch),
      batch_size=batch,shuffle=False,num_workers=workers,pin_memory=True,
      generator=gen,persistent_workers=False)


def source_metrics(model,frame,backbone,micro,device,workers=4):
    from sklearn.metrics import roc_auc_score
    model.eval();parts=[]
    with torch.inference_mode():
        for x,_ in loader(frame,backbone,micro,workers=workers):
            h=model.encoder(x.to(device)).float()
            # The cache/scoring boundary is explicit FP32 h and FP64 linear head.
            head=model.classifier
            parts.append(fp64_margin(h,head.weight,head.bias).cpu().numpy())
    out=frame[['video_uid','method_id','label']].copy();out['score']=np.concatenate(parts)
    video=out.groupby(['video_uid','method_id','label'],sort=True,as_index=False).score.mean()
    real=video[video.label.eq(0)]
    if len(real)==0 or not video.label.eq(1).any():raise ValueError('source DEV needs both classes')
    per={}
    for method in sorted(set(video.loc[video.label.eq(1),'method_id'])):
        q=pd.concat([real,video[video.method_id.eq(method)]])
        a=auc_independent(q.label,q.score);reference=roc_auc_score(q.label,q.score)
        if abs(a-reference)>1e-12:raise ValueError('independent source DEV AUC recomputation failed')
        per[method]=a
    return {'macro_auc':float(np.mean(list(per.values()))),'pooled_auc':auc_independent(video.label,video.score),'method_auc':per},video


def code_hash():
    h=hashlib.sha256()
    for name in ['trainer.py','models.py']:
        p=Path(__file__).parent/name;h.update(name.encode());h.update(p.read_bytes())
    return h.hexdigest()


def train(config_path,run_id,device='cuda'):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id) or run_id in ['.','..']:raise ValueError('unsafe run identity')
    config=json.loads(Path(config_path).read_text());root=Path(config['artifact_root'])
    with exclusive_file(root/'runs'/run_id/'run.lock'):
        context=owned_gpu(root,config) if str(device).startswith('cuda') else contextlib.nullcontext()
        with context:return _train(config_path,run_id,device)


def _train(config_path,run_id,device='cuda'):
    from .models import build_model
    policy();config=json.loads(Path(config_path).read_text());root=Path(config['artifact_root'])
    task=config['runs'][run_id]
    if task['phase'] not in ['development','formal']:raise ValueError('unregistered phase')
    backbone=task['backbone'];seed=task['seed'];epochs=task['epochs'];lr=task['lr'];micro=task['microbatch']
    if micro not in [8,16,32] or config['effective_batch']!=128:raise ValueError('unqualified micro/effective batch')
    if task['phase']=='formal' and not task.get('recipe_frozen',False):raise ValueError('formal recipe is not frozen')
    gate_path=root/task.get('qualification_gate','qualification/gate.json')
    gate=json.loads(gate_path.read_text())
    if gate.get('status')!='PASS' or gate['training_code_sha256']!=code_hash():raise ValueError('source qualification missing or stale')
    if task.get('route') not in gate['approved_routes']:raise ValueError('route lacks source/data qualification')
    trainpath=root/task['train_manifest'];devpath=root/task['dev_manifest']
    init=Path(config['backbones'][backbone]['weight_path'])
    identity={'config':hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest(),
      'train_manifest':digest(trainpath),'dev_manifest':digest(devpath),'initialization':digest(init),
      'environment':digest(root/'configs/requirements.execution.lock.txt'),'code':code_hash(),
      'qualification':digest(gate_path)}
    expected=config['backbones'][backbone]['weight_sha256']
    if identity['initialization']!=expected:raise ValueError('pretrained initialization hash mismatch')
    out=root/'runs'/run_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        complete=json.loads((out/'complete.json').read_text())
        if complete['identity']!=identity or digest(out/'final.pt')!=complete['final_sha256']:raise ValueError('completed model reuse identity mismatch')
        return complete
    save_json(out/'identity.json',identity);save_json(out/'resolved_config.json',task)
    frame=pd.read_parquet(trainpath);dev=pd.read_parquet(devpath)
    set_random(seed);model=build_model(backbone,init).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=5e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=128) if task.get('precision','AMP')=='AMP' else None
    draws_count=task.get('draw_count',math.ceil(len(frame)/128)*128);steps=draws_count//128
    start_epoch=0;history=[];attempts=0;best_model=None;best_epoch=None;best=-1.;consecutive_skips=0
    if (out/'state.pt').exists():
        state=load_state(out/'state.pt',model,optimizer,scaler,identity)
        start_epoch=state['epoch'];history=state['history'];attempts=state['attempts']
        repair_history(out,state,steps)
        best_model=state['best_model'];best_epoch=state['best_epoch'];best=state['best_score']
        consecutive_skips=state['consecutive_skips']
        if history and best_model is None:raise ValueError('checkpoint lacks recoverable source-best state')
        if best_model is not None:
            temp=out/'source_best.pt.tmp';torch.save({'model':best_model,'identity':identity,'epoch':best_epoch},temp);os.replace(temp,out/'source_best.pt')
    for epoch in range(start_epoch,epochs):
        start=time.time();model.train();draws,methods=epoch_draws(frame,seed,epoch+1,draws_count)
        np.save(out/f'epoch_{epoch+1:02d}_samples.npy',draws,allow_pickle=False)
        np.save(out/f'epoch_{epoch+1:02d}_methods.npy',methods,allow_pickle=False)
        # Keep the exact augmentation values, not only the RNG algorithm.
        aug=np.asarray([[p['flip'],p['brightness'],p['contrast']] for p in (
          augmentation_parameters(seed,epoch+1,str(frame.iloc[int(index)].get('mother_id',frame.iloc[int(index)].get('sample_id',frame.iloc[int(index)].rgb_sha256))),i)
          for i,index in enumerate(draws))],dtype=np.float64)
        np.save(out/f'epoch_{epoch+1:02d}_augmentations.npy',aug,allow_pickle=False)
        events=out/f'epoch_{epoch+1:02d}_updates.jsonl';success=0;loss=0.
        with events.open('w') as stream:
            for x,y in loader(frame,backbone,128,draws,seed,epoch+1,config.get('workers',4)):
                event=advance_step(model,optimizer,scaler,x,y,attempts,steps,epochs,lr,micro,device)
                event.update(epoch=epoch+1);stream.write(json.dumps(event,allow_nan=False)+'\n');stream.flush()
                success+=event['success'];loss+=event['loss'];attempts+=1
                consecutive_skips=consecutive_skips+1 if event['skip'] else 0
                if consecutive_skips>=8:raise FloatingPointError('8 consecutive AMP skips; source numeric protocol review required')
        metrics,scores=source_metrics(model,dev,backbone,micro,device,config.get('workers',4))
        scores.to_parquet(out/f'epoch_{epoch+1:02d}_source_dev.parquet',index=False)
        row={'epoch':epoch+1,'train_loss':loss/steps,'attempts':steps,'success':success,'skip':steps-success,
          'draw_count':draws_count,'method_counts':dict(Counter(methods.tolist())),
          'sample_sha256':digest(out/f'epoch_{epoch+1:02d}_samples.npy'),
          'augmentation_sha256':digest(out/f'epoch_{epoch+1:02d}_augmentations.npy'),
          'dev':metrics,'seconds':time.time()-start}
        if metrics['macro_auc']>best:
            best=metrics['macro_auc'];best_epoch=epoch+1
            best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        history.append(row);save_state(out/'state.pt',model,optimizer,scaler,epoch+1,identity,history,attempts,best_model,best_epoch,best,consecutive_skips)
        save_json(out/'history.json',history)
        if best_epoch==epoch+1:
            tmp=out/'source_best.pt.tmp';torch.save({'model':best_model,'identity':identity,'epoch':best_epoch},tmp);os.replace(tmp,out/'source_best.pt')
        print(json.dumps({'run':run_id,**row},allow_nan=False),flush=True)
    tmp=out/'final.pt.tmp';torch.save({'model':model.state_dict(),'identity':identity,'epoch':epochs},tmp);os.replace(tmp,out/'final.pt')
    result={'status':'TRAINING_COMPLETE','identity':identity,'final_sha256':digest(out/'final.pt'),
      'epochs':epochs,'attempts':attempts,'success':sum(h['success'] for h in history),
      'skip':sum(h['skip'] for h in history),'selection':'frozen_last','target_used':False}
    result.update(source_best_epoch=best_epoch,source_best_sha256=digest(out/'source_best.pt'))
    save_json(out/'complete.json',result);return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();train(args.config,args.run_id)
