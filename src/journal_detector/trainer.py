"""Conference training with epoch LR, parameter EMA, early stopping and recovery.

Only development trajectories may evaluate the isolated CDF development role.
Formal TEST/CAL roles are never read by this training module.
"""
from pathlib import Path
from collections import Counter
import contextlib
from dataclasses import dataclass, asdict
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


def epoch_lr(epoch,epochs,base_lr,warmup_epochs=1):
    if not 0<=epoch<epochs:raise ValueError('epoch outside training budget')
    if epoch<warmup_epochs:return base_lr*(epoch+1)/warmup_epochs
    progress=(epoch-warmup_epochs)/max(1,epochs-warmup_epochs)
    return base_lr*.5*(1+math.cos(math.pi*progress))


class EMA:
    """Conference semantics: trainable parameter EMA; use live model buffers."""
    def __init__(self,model,decay=.999):
        self.decay=decay
        self.shadow={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
    @torch.no_grad()
    def update(self,model):
        for k,p in model.named_parameters():
            if p.requires_grad:self.shadow[k].mul_(self.decay).add_(p.detach(),alpha=1-self.decay)
    def state_dict(self):return {k:v.detach().clone() for k,v in self.shadow.items()}
    def load_state_dict(self,state):
        if set(state)!=set(self.shadow):raise ValueError('EMA parameter identity mismatch')
        for k,v in state.items():self.shadow[k].copy_(v)
    @contextlib.contextmanager
    def applied(self,model):
        backup={k:p.detach().clone() for k,p in model.named_parameters() if p.requires_grad}
        try:
            with torch.no_grad():
                for k,p in model.named_parameters():
                    if p.requires_grad:p.copy_(self.shadow[k])
            yield
        finally:
            with torch.no_grad():
                for k,p in model.named_parameters():
                    if p.requires_grad:p.copy_(backup[k])


@dataclass
class Selection:
    best_score:float=-1.
    best_epoch:int=0
    wait:int=0
    patience:int=5
    def observe(self,score,epoch):
        if not math.isfinite(score):raise ValueError('nonfinite source selection AUC')
        improved=score>self.best_score
        if improved:self.best_score=score;self.best_epoch=epoch;self.wait=0
        else:self.wait+=1
        return improved
    @property
    def stopped(self):return self.wait>=self.patience
    def state_dict(self):return asdict(self)


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
    key=f'{seed}/{epoch}/{sample_key}/{draw_index}'.encode()
    return {'seed':int.from_bytes(hashlib.sha256(key).digest()[:4],'little')}


def normalization(backbone):
    if backbone=='clip_vit_b16':return ((.48145466,.45762756,.40821073),(.26862954,.26130258,.27577711))
    return ((.485,.456,.406),(.229,.224,.225))


def image_tensor(path,backbone,params=None):
    import cv2
    from .augmentation import augment
    with Image.open(path) as source:im=np.asarray(source.convert('RGB'))
    im=cv2.resize(im,(224,224),interpolation=cv2.INTER_CUBIC)
    if params is not None:im=augment(im,params['seed'])
    mean,std=normalization(backbone)
    rgb=(im.astype(np.float32)/255-np.asarray(mean,dtype=np.float32))/np.asarray(std,dtype=np.float32)
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


def save_state(path,model,optimizer,scaler,epoch,identity,history,attempts=None,best_model=None,best_epoch=None,best_score=None,consecutive_skips=0,ema=None,selection=None,development_stop=None,cdf_best_model=None):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),
      'scaler':scaler.state_dict() if scaler is not None else None,
      'epoch':epoch,'attempts':epoch if attempts is None else attempts,
      'scheduler_clock':'epoch','identity':identity,'history':history,
      'best_model':best_model,'best_epoch':best_epoch,'best_score':best_score,
      'consecutive_skips':consecutive_skips,
      'ema':ema.state_dict() if ema is not None else None,
      'selection':selection.state_dict() if selection is not None else None,
      'development_stop':development_stop.state_dict() if development_stop is not None else None,
      'cdf_best_model':cdf_best_model,
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
    lr=epoch_lr(attempt//steps_per_epoch,epochs,base_lr)
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
            parts.append(torch.sigmoid(fp64_margin(h,head.weight,head.bias)).cpu().numpy())
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


def code_hash(route='D'):
    h=hashlib.sha256();base=Path(__file__).resolve().parents[2]
    names=['trainer.py','models.py','augmentation.py','protocol.py','data_d.py','development.py']
    if route=='H':names+=['trainer_h.py','data_h.py','geometry_h.py']
    paths=[base/'src/journal_detector'/n for n in names if (base/'src/journal_detector'/n).is_file()]
    for folder in ['dinov2','df40']:paths+=list((base/'third_party'/folder).rglob('*.py'))
    for p in sorted(paths):
        h.update(str(p.relative_to(base)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def training_identity(config,task):
    root=Path(config['artifact_root'])
    init=Path(config['backbones'][task['backbone']]['weight_path'])
    identity={'config':hashlib.sha256(json.dumps(task,sort_keys=True).encode()).hexdigest(),
      'train_manifest':digest(root/task['train_manifest']),'dev_manifest':digest(root/task['dev_manifest']),
      'initialization':digest(init),'environment':digest(root/'configs/requirements.execution.lock.txt'),
      'code':code_hash(task.get('route','D')),'data_freeze':digest(root/('data/H/train_dev_freeze.json' if task.get('route')=='H' else 'data/D/freeze.json'))}
    if task.get('cdf_development_manifest'):
        cdf_path=root/task['cdf_development_manifest']
        if task.get('phase')!='development' or task['cdf_development_manifest']!='data/D/cdf_dev.parquet':
            raise ValueError('CDF role may be used only by registered development')
        actual=digest(cdf_path)
        freeze=json.loads((root/'data/D/freeze.json').read_text())
        if freeze.get('manifest_sha256',{}).get('cdf_dev.parquet')!=actual:
            raise ValueError('frozen CDF development manifest hash differs from data freeze')
        identity['cdf_development_manifest']=actual
    if task.get('epoch_recipe_sha256'):
        recipe_path=Path(task.get('epoch_recipe_path',''))
        if not recipe_path.is_absolute():recipe_path=root/recipe_path
        if not recipe_path.is_file() or digest(recipe_path)!=task['epoch_recipe_sha256']:
            raise ValueError('frozen epoch recipe provenance mismatch')
        recipe=json.loads(recipe_path.read_text())
        route=task.get('route','D')
        if recipe.get('route')!=route:raise ValueError('epoch recipe route differs from task route')
        if recipe.get('schema')!='EPOCH_RECIPE_V2' or recipe.get('status')!='FROZEN':
            raise ValueError('epoch recipe is not a frozen v2 decision')
        if recipe.get('data_freeze_sha256')!=identity['data_freeze']:
            raise ValueError('epoch recipe data freeze differs from actual training data')
        if recipe.get('target_development_used') is not (route=='D'):
            raise ValueError('epoch recipe target development scope differs from route')
        if route=='H' and recipe.get('paired_histories')!=['B','AB']:
            raise ValueError('H recipe must cover both paired histories')
        if recipe.get('backbone')!=task['backbone'] or recipe.get('chosen_epochs')!=task['epochs']:
            raise ValueError('task differs from frozen development epoch recipe')
        identity['epoch_recipe_sha256']=task['epoch_recipe_sha256']
    if identity['initialization']!=config['backbones'][task['backbone']]['weight_sha256']:
        raise ValueError('pretrained initialization hash mismatch')
    return identity


def train(config_path,run_id,device='cuda'):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',run_id) or run_id in ['.','..']:raise ValueError('unsafe run identity')
    config=json.loads(Path(config_path).read_text());root=Path(config['artifact_root'])
    with exclusive_file(root/'runs'/run_id/'run.lock'):
        context=owned_gpu(root,config) if str(device).startswith('cuda') else contextlib.nullcontext()
        with context:return _train(config_path,run_id,device)


def _train(config_path,run_id,device='cuda'):
    from .models import build_model
    from .protocol import validate_config, validate_manifests
    policy();config=json.loads(Path(config_path).read_text());validate_config(config)
    root=Path(config['artifact_root']);task=config['runs'][run_id]
    backbone=task['backbone'];seed=task['seed'];epochs=task['epochs'];lr=task['lr'];micro=task['microbatch']
    trainpath=root/task['train_manifest'];devpath=root/task['dev_manifest']
    identity=training_identity(config,task);out=root/'runs'/run_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        complete=json.loads((out/'complete.json').read_text())
        if complete['identity']!=identity or digest(out/'best_model.pth')!=complete['best_sha256']:
            raise ValueError('completed model reuse identity mismatch')
        return complete
    frame=pd.read_parquet(trainpath);dev=pd.read_parquet(devpath);validate_manifests(root,frame,dev)
    save_json(out/'identity.json',identity);save_json(out/'resolved_config.json',task)
    set_random(seed);model=build_model(backbone,config['backbones'][backbone]['weight_path']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=5e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=128) if str(device).startswith('cuda') else None
    ema=EMA(model);selection=Selection();best_model=None
    phase=task.get('phase','formal');cdf_frame=None;development_stop=None;cdf_best_model=None
    if task.get('cdf_development_manifest'):
        if phase!='development' or task['cdf_development_manifest']!='data/D/cdf_dev.parquet':
            raise ValueError('target evaluation is restricted to isolated development trajectories')
        cdf_frame=pd.read_parquet(root/task['cdf_development_manifest'])
        if cdf_frame.empty or set(cdf_frame.label)!={0,1}:
            raise ValueError('CDF development requires finite usable real and fake cohorts')
        if 'role' in cdf_frame and not cdf_frame.role.eq('cdf_dev').all():
            raise ValueError('CDF development manifest role mismatch')
        from .development import DevelopmentStop
        development_stop=DevelopmentStop()
    draws_count=math.ceil(len(frame)/128)*128;steps=draws_count//128
    start_epoch=0;history=[];attempts=0;consecutive_skips=0
    if (out/'state.pt').exists():
        state=load_state(out/'state.pt',model,optimizer,scaler,identity)
        start_epoch=state['epoch'];history=state['history'];attempts=state['attempts']
        repair_history(out,state,steps);ema.load_state_dict(state['ema']);selection=Selection(**state['selection'])
        best_model=state['best_model'];consecutive_skips=state['consecutive_skips']
        if best_model is None:raise ValueError('checkpoint lacks recoverable best EMA state')
        save_best(out,best_model,identity,selection)
        if development_stop is not None:
            if state.get('development_stop') is None:raise ValueError('resume lacks CDF development stopper')
            development_stop=DevelopmentStop.from_state(state['development_stop'])
            cdf_best_model=state['cdf_best_model']
            save_cdf_best(out,cdf_best_model,identity,development_stop.cdf)
    for epoch in range(start_epoch,epochs):
        if (development_stop.stopped if development_stop is not None else selection.stopped):break
        start=time.time();model.train();draws,methods=epoch_draws(frame,seed,epoch+1,draws_count)
        np.save(out/f'epoch_{epoch+1:02d}_samples.npy',draws,allow_pickle=False)
        np.save(out/f'epoch_{epoch+1:02d}_methods.npy',methods,allow_pickle=False)
        aug=np.asarray([augmentation_parameters(seed,epoch+1,str(frame.iloc[int(index)].get('sample_id',frame.iloc[int(index)].rgb_sha256)),i)['seed'] for i,index in enumerate(draws)],dtype=np.uint32)
        np.save(out/f'epoch_{epoch+1:02d}_augmentation_seeds.npy',aug,allow_pickle=False)
        events=out/f'epoch_{epoch+1:02d}_updates.jsonl';success=0;loss=0.
        with events.open('w') as stream:
            for x,y in loader(frame,backbone,128,draws,seed,epoch+1,config.get('workers',4)):
                event=advance_step(model,optimizer,scaler,x,y,attempts,steps,epochs,lr,micro,device)
                if event['success']:ema.update(model)
                event.update(epoch=epoch+1);stream.write(json.dumps(event,allow_nan=False)+'\n');stream.flush()
                success+=event['success'];loss+=event['loss'];attempts+=1
                consecutive_skips=consecutive_skips+1 if event['skip'] else 0
                if consecutive_skips>=8:raise FloatingPointError('8 consecutive AMP skips')
        with ema.applied(model):
            metrics,scores=source_metrics(model,dev,backbone,micro,device,config.get('workers',4))
            improved=selection.observe(metrics['pooled_auc'],epoch+1)
            if improved:best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            cdf_metrics=None;cdf_improved=False
            if cdf_frame is not None:
                cdf_metrics,cdf_scores=source_metrics(model,cdf_frame,backbone,micro,device,config.get('workers',4))
                cdf_improved=development_stop.observe(metrics['pooled_auc'],cdf_metrics['pooled_auc'],epoch+1)
                if cdf_improved:cdf_best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                cdf_scores.to_parquet(out/f'epoch_{epoch+1:02d}_cdf_dev.parquet',index=False)
        scores.to_parquet(out/f'epoch_{epoch+1:02d}_source_dev.parquet',index=False)
        row={'epoch':epoch+1,'lr':epoch_lr(epoch,epochs,lr),'train_loss':loss/steps,'attempts':steps,
          'success':success,'skip':steps-success,'draw_count':draws_count,'method_counts':dict(Counter(methods.tolist())),
          'sample_sha256':digest(out/f'epoch_{epoch+1:02d}_samples.npy'),
          'augmentation_sha256':digest(out/f'epoch_{epoch+1:02d}_augmentation_seeds.npy'),
          'dev':metrics,'selection':selection.state_dict(),'seconds':time.time()-start}
        if cdf_metrics is not None:
            row.update(cdf_dev=cdf_metrics,development_stop=development_stop.state_dict())
        history.append(row)
        save_state(out/'state.pt',model,optimizer,scaler,epoch+1,identity,history,attempts,
          best_model,selection.best_epoch,selection.best_score,consecutive_skips,ema,selection,development_stop,cdf_best_model)
        save_json(out/'history.json',history)
        if improved:save_best(out,best_model,identity,selection)
        if cdf_improved:save_cdf_best(out,cdf_best_model,identity,development_stop.cdf)
        print(json.dumps({'run':run_id,**row},allow_nan=False),flush=True)
    if not history or best_model is None:raise ValueError('no valid source-selected checkpoint')
    result={'status':'TRAINING_COMPLETE','identity':identity,'best_sha256':digest(out/'best_model.pth'),
      'epochs':len(history),'max_epochs':epochs,'best_epoch':selection.best_epoch,
      'early_stopped':development_stop.stopped if development_stop is not None else selection.stopped,'attempts':attempts,'success':sum(h['success'] for h in history),
      'skip':sum(h['skip'] for h in history),'selection':'source_dev_pooled_mean_probability_auc_ema',
      'target_used':cdf_frame is not None,'ema_decay':.999,'phase':phase,
      'target_development_used':bool(cdf_frame is not None or task.get('epoch_recipe_sha256')),
      'epoch_recipe_sha256':task.get('epoch_recipe_sha256')}
    if development_stop is not None:
        result.update(cdf_best_epoch=development_stop.cdf.best_epoch,
                      cdf_best_auc=development_stop.cdf.best_score,
                      cdf_best_sha256=digest(out/'cdf_best_model.pth'),
                      stopped_by='source_and_cdf_development_joint_plateau')
    save_json(out/'complete.json',result);return result


def save_best(out,model_state,identity,selection):
    tmp=out/'best_model.pth.tmp'
    torch.save({'model':model_state,'identity':identity,'epoch':selection.best_epoch,
      'auc':selection.best_score,'weights':'ema','ema_decay':.999},tmp)
    os.replace(tmp,out/'best_model.pth')


def save_cdf_best(out,model_state,identity,selection):
    if model_state is None:raise ValueError('missing recoverable CDF development checkpoint')
    tmp=out/'cdf_best_model.pth.tmp'
    torch.save({'model':model_state,'identity':identity,'epoch':selection.best_epoch,
        'auc':selection.best_score,'weights':'ema','ema_decay':.999,
        'selection':'isolated_cdf_dev_pooled_mean_probability_auc_ema'},tmp)
    os.replace(tmp,out/'cdf_best_model.pth')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--run-id',required=True)
    args=parser.parse_args();train(args.config,args.run_id)
