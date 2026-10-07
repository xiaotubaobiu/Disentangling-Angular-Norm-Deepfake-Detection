"""Fixed-epoch raw training with explicit epoch-boundary recovery."""
from pathlib import Path
import hashlib,json,os,random,time
import numpy as np
import pandas as pd
import torch
from .manifest import digest,save_json


def sampling_weights(df):
    mass=np.zeros(len(df),dtype=float)
    fake=sorted(set(df.loc[df.label==1,'method_id']))
    if not fake or not (df.label==0).any():raise ValueError('Both training classes required')
    for method in ['real']+fake:
        ids=np.flatnonzero(df.method_id.to_numpy()==method);g=df.iloc[ids]
        total=.5 if method=='real' else .5/len(fake)
        if (g.evaluation_unit=='image').all():mass[ids]=total/len(ids)
        else:
            counts=g.groupby('video_id').size();n=len(counts)
            mass[ids]=[total/n/counts[v] for v in g.video_id]
    if not np.isclose(mass.sum(),1):raise ValueError('Sampling mass mismatch')
    return mass


def save_checkpoint(path,model,optimizer,scaler,epoch,identity,history=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict() if scaler is not None else None,
      'epoch':epoch,'identity':identity,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
      'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,'resume_scope':'completed_epoch_boundary','history':history}
    if history is not None and [h['epoch'] for h in history]!=list(range(1,epoch+1)):raise ValueError('Checkpoint history/epoch mismatch')
    tmp=path.with_name(path.name+'.tmp');torch.save(state,tmp);os.replace(tmp,path)


def restore_checkpoint(path,model,optimizer,scaler,identity):
    # Only this project's own checkpoints, never arbitrary user downloads.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['identity']!=identity:raise ValueError('Checkpoint protocol/input/initialization mismatch')
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
    if scaler is not None:scaler.load_state_dict(state['scaler'])
    random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng']);torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
    return state['epoch']


def set_random(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_initialized():torch.cuda.manual_seed_all(seed)


def transform_image(path,backbone,augment=None):
    from PIL import Image
    with Image.open(path) as im:rgb=np.array(im.convert('RGB').resize((224,224),Image.Resampling.BICUBIC))
    if augment is not None:rgb=augment(image=rgb)['image']
    else:
        mean,std=normalization(backbone);rgb=(rgb.astype(np.float32)/255-np.array(mean,dtype=np.float32))/np.array(std,dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2,0,1))).float()


def normalization(backbone):
    if backbone=='clip_vit_b16':return ((.48145466,.45762756,.40821073),(.26862954,.26130258,.27577711))
    return ((.485,.456,.406),(.229,.224,.225))


def augmentation(backbone,seed):
    import albumentations as A
    mean,std=normalization(backbone)
    return A.Compose([A.HorizontalFlip(p=.5),A.Rotate(limit=(-10,10),p=.5),A.GaussianBlur(blur_limit=(3,7),p=.5),
      A.OneOf([A.RandomBrightnessContrast(brightness_limit=.1,contrast_limit=.1),A.FancyPCA(),A.HueSaturationValue()],p=.5),
      A.ImageCompression(quality_range=(40,100),p=.5),A.Normalize(mean=mean,std=std)],seed=seed)


def train(config,backbone,seed,device,root):
    from .models import build_model
    from .orchestration import require_gpu_gates
    root=Path(root);require_gpu_gates(root)
    manifest=root/'data/train.parquet';df=pd.read_parquet(manifest)
    identity=run_identity(config,backbone,seed,root);micro=identity['microbatch']
    run_id=f'{backbone}_seed{seed}';out=root/'runs'/run_id;out.mkdir(parents=True,exist_ok=True)
    complete=out/'complete.json'
    if complete.exists():
        old=json.loads(complete.read_text())
        if old['identity']!=identity or digest(out/'final.pt')!=old['final_sha256']:raise ValueError('Run reuse mismatch')
        return old
    set_random(seed);model=build_model(backbone,config['backbones'][backbone]['weight_path']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['backbones'][backbone]['learning_rate'],weight_decay=config['weight_decay'])
    scaler=torch.amp.GradScaler('cuda');epoch=0;state=out/'state.pt'
    history=[]
    if state.exists():
        epoch=restore_checkpoint(state,model,optimizer,scaler,identity)
        history=torch.load(state,map_location='cpu',weights_only=False).get('history')
        if history is None or [h['epoch'] for h in history]!=list(range(1,epoch+1)):raise ValueError('Incomplete checkpoint epoch evidence; cannot silently resume')
        save_json(out/'history.json',history)
    p=sampling_weights(df);pool=len(df)
    save_json(out/'identity.json',identity)
    for e in range(epoch,config['epochs']):
        start=time.time();model.train();aug=augmentation(backbone,seed+100003*e)
        sequence=np.random.default_rng(seed+100003*e).choice(pool,size=pool,replace=True,p=p)
        seqpath=out/f'epoch_{e+1:02d}_samples.npy';np.save(seqpath,sequence)
        loss_total=0.;slots=0
        for start_idx in range(0,pool,config['effective_batch']):
            batchids=sequence[start_idx:start_idx+config['effective_batch']];n=len(batchids);optimizer.zero_grad(set_to_none=True)
            for j in range(0,n,micro):
                rows=df.iloc[batchids[j:j+micro]]
                x=torch.stack([transform_image(path,backbone,aug) for path in rows.image_path]).to(device)
                y=torch.as_tensor(rows.label.to_numpy(),dtype=torch.long,device=device)
                with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(model(x),y,reduction='sum')/n
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                scaler.scale(loss).backward();loss_total+=float(loss.detach())*n;slots+=len(y)
            scaler.step(optimizer);scaler.update()
        history=[h for h in history if h['epoch']<=e]
        history.append({'epoch':e+1,'loss':loss_total/slots,'sample_slots':slots,'sample_sequence_sha256':digest(seqpath),'seconds':time.time()-start})
        save_checkpoint(state,model,optimizer,scaler,e+1,identity,history=history)
        save_json(out/'history.json',history);print(json.dumps({'run':run_id,**history[-1]}),flush=True)
    tmp=out/'final.pt.tmp';torch.save({'model':model.state_dict(),'identity':identity,'epoch':config['epochs']},tmp);os.replace(tmp,out/'final.pt')
    summary={'status':'TRAINING_COMPLETE','identity':identity,'epoch':config['epochs'],'final_sha256':digest(out/'final.pt'),'selection':'final_epoch_only'}
    save_json(complete,summary);return summary


def run_identity(config,backbone,seed,root):
    from .orchestration import code_hash
    root=Path(root);probe=json.loads((root/'probe_complete.json').read_text())
    config_sha=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    code_sha=code_hash(Path(__file__).resolve().parents[2])
    if probe.get('freeze_sha256')!=digest(root/'data/freeze_complete.json') or probe.get('config_identity')!=config_sha or probe.get('code_sha256')!=code_sha:
        raise ValueError('Probe does not match frozen data/config/source implementation')
    if backbone not in config['backbones'] or seed not in config['seeds']:raise ValueError('Unapproved model or seed')
    micro=probe['backbones'][backbone]['microbatch']
    if micro not in [16,32,64,128] or config['effective_batch']!=128:raise ValueError('Invalid fixed effective/microbatch protocol')
    return {'config':config_sha,'manifest':digest(root/'data/train.parquet'),
      'initialization':digest(config['backbones'][backbone]['weight_path']),'backbone':backbone,'seed':seed,'microbatch':micro,
      'code_sha256':code_sha,'probe_sha256':digest(root/'probe_complete.json')}


def extract(config,backbone,seed,device,root):
    from .models import build_model
    from .cache import verify_cache
    from .orchestration import require_gpu_gates
    root=Path(root);run_id=f'{backbone}_seed{seed}';out=root/'features'/run_id;out.mkdir(parents=True,exist_ok=True)
    frozen=require_gpu_gates(root);expected=run_identity(config,backbone,seed,root)
    checkpoint=root/'runs'/run_id/'final.pt';completion=json.loads((root/'runs'/run_id/'complete.json').read_text())
    if completion.get('status')!='TRAINING_COMPLETE' or completion['identity']!=expected or digest(checkpoint)!=completion['final_sha256']:
        raise ValueError('Unverified final training checkpoint')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['identity']!=expected or state['epoch']!=config['epochs']:raise ValueError('Final checkpoint identity/epoch mismatch')
    model=build_model(backbone,config['backbones'][backbone]['weight_path']);model.load_state_dict(state['model']);model.to(device).eval()
    roles=['train','ff_test','cdf_test']+(['calibration'] if frozen['calibration_available'] else [])
    for role in roles:
        p=root/'data'/f'{role}.parquet'
        df=pd.read_parquet(p);roleout=out/role;roleout.mkdir(exist_ok=True);identity={'checkpoint':digest(checkpoint),'manifest':digest(p)}
        marker=roleout/'complete.json'
        if marker.exists():
            verify_cache(roleout,identity)
            continue
        pieces=[]
        for start in range(0,len(df),4096):
            part=df.iloc[start:start+4096];cache=roleout/f'{start:09d}.npy';features=[]
            chunk_marker=cache.with_suffix('.json')
            if cache.exists() and chunk_marker.exists():
                saved=json.loads(chunk_marker.read_text())
                if saved.get('identity')!=identity or saved['start']!=start or saved['rows']!=len(part) or digest(cache)!=saved['sha256']:
                    raise ValueError('Interrupted feature chunk identity/hash mismatch')
                array=np.load(cache,mmap_mode='r',allow_pickle=False)
                if array.dtype!=np.float32 or array.ndim!=2 or len(array)!=len(part) or not np.isfinite(array).all():raise ValueError('Invalid interrupted feature chunk')
                pieces.append({k:saved[k] for k in ['start','rows','file','sha256']});continue
            with torch.inference_mode():
                for j in range(0,len(part),64):
                    rows=part.iloc[j:j+64];x=torch.stack([transform_image(path,backbone) for path in rows.image_path]).to(device)
                    features.append(model.encoder(x).float().cpu().numpy())
            array=np.concatenate(features);assert np.isfinite(array).all()
            tmp=cache.with_name(cache.name+'.tmp')
            with tmp.open('wb') as f:np.save(f,array)
            os.replace(tmp,cache);item={'start':start,'rows':len(part),'file':cache.name,'sha256':digest(cache)}
            save_json(chunk_marker,{'identity':identity,**item});pieces.append(item)
        df.to_parquet(roleout/'manifest.parquet',index=False)
        head={k:v.detach().cpu().numpy().tolist() for k,v in model.classifier.state_dict().items()}
        save_json(marker,{'identity':identity,'rows':len(df),'cached_manifest_sha256':digest(roleout/'manifest.parquet'),'chunks':pieces,'head':head,'backbone':backbone,'seed':seed,'encoding':'FP32 raw; no extra L2 normalization'})
        verify_cache(roleout,identity)
