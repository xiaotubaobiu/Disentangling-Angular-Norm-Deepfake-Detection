"""Source-only real-backbone GPU qualification; no service management."""
from pathlib import Path
import copy
import gc
import json
import os
import subprocess
import time
import numpy as np
import pandas as pd
import torch
from .trainer import (digest,save_json,policy,set_random,image_tensor,
    advance_step,save_state,load_state,code_hash,owned_gpu)


def source_probe_rows(frame):
    if not frame.domain.eq('ff').all() or not frame.role.str.upper().isin(['TRAIN','DEV']).all():
        raise ValueError('qualification requires source-only TRAIN/DEV rows')
    q=pd.concat([frame[frame.label.eq(0)].head(64),frame[frame.label.eq(1)].head(64)],ignore_index=True)
    if len(q)!=128:raise ValueError('128 class-balanced source inputs required')
    return q


def assert_equal_tree(a,b):
    if torch.is_tensor(a):assert torch.equal(a.detach().cpu(),b.detach().cpu())
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_equal_tree(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_equal_tree(x,y)
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b)
    else:assert a==b


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu_tree(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def make_probe(root):
    root=Path(root);old=Path('/home/z/project/deepfake_detector/artifacts/data')
    frame=pd.read_parquet(old/'training_associations.parquet')
    pairs=sorted([sorted(p) for p in json.loads((old/'source_splits/train.json').read_text())])
    train_pairs=np.random.default_rng(20261023).permutation(len(pairs))[:306]
    train_ids={v for i in train_pairs for v in pairs[int(i)]}
    selected=frame[frame.domain.eq('ff') & frame.source_ids.map(lambda s:bool(json.loads(s)) and set(json.loads(s))<=train_ids)]
    selected=selected[selected.method_type.isin(['FS','FR','real'])].copy();selected['role']='TRAIN'
    selected=source_probe_rows(selected)
    selected['video_uid']='DF40/ff/'+selected.method_id+'/'+selected.video_id
    for r in selected.itertuples():
        stat=Path(r.image_path).stat()
        if hasattr(r,'bytes') and (stat.st_size!=r.bytes or stat.st_mtime_ns!=r.mtime_ns):raise ValueError('probe source metadata changed')
        if digest(r.image_path)!=r.content_sha256:raise ValueError('probe source byte hash changed')
    path=root/'qualification/source_probe.parquet';selected.to_parquet(path,index=False)
    save_json(root/'qualification/source_probe_receipt.json',{
      'rows':len(selected),'label_counts':{str(k):int(v) for k,v in selected.label.value_counts().items()},
      'domain':'ff','role':'TRAIN','input_manifest_sha256':digest(old/'training_associations.parquet'),
      'source_probe_sha256':digest(path),'all_selected_bytes_verified':True,
      'purpose':'source-only runtime/microbatch/recovery; not a scientific model or new manifest acceptance'})
    return path


def worker(root,backbone,micro,benchmark_steps=100):
    root=Path(root);config=json.loads((root/'configs/base.json').read_text())
    with owned_gpu(root,config):return _worker(root,backbone,micro,benchmark_steps)


def _worker(root,backbone,micro,benchmark_steps=100):
    from .models import build_model
    root=Path(root);policy()
    if torch.cuda.device_count()!=1:raise ValueError('qualification requires one explicitly assigned GPU')
    config=json.loads((root/'configs/base.json').read_text());weight=config['backbones'][backbone]
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in config['allowed_gpu_uuids']:raise ValueError('GPU outside approved0-3 UUIDs')
    if digest(weight['weight_path'])!=weight['weight_sha256']:raise ValueError('pretrained hash mismatch')
    frame=source_probe_rows(pd.read_parquet(root/'qualification/source_probe.parquet'))
    x=torch.stack([image_tensor(p,backbone) for p in frame.image_path]);y=torch.as_tensor(frame.label.to_numpy(),dtype=torch.long)
    out=root/'qualification'/backbone;out.mkdir(parents=True,exist_ok=True)
    set_random(7007);model=build_model(backbone,weight['weight_path']).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=weight['learning_rate'],weight_decay=5e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=128)
    torch.cuda.reset_peak_memory_stats();start=time.time()
    identity={'code':code_hash(),'source_probe':digest(root/'qualification/source_probe.parquet'),
      'weight':weight['weight_sha256'],'backbone':backbone,'micro':micro,'epochs':2,'steps_per_epoch':1}
    events=[]
    events.append(advance_step(model,opt,scaler,x,y,0,1,2,weight['learning_rate'],micro,'cuda'))
    missing=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:raise ValueError('parameters without full-encoder gradient: '+str(missing))
    save_state(out/'resume_epoch1.pt',model,opt,scaler,1,identity,[{'epoch':1}],1)
    events.append(advance_step(model,opt,scaler,x,y,1,1,2,weight['learning_rate'],micro,'cuda'))
    target=cpu_tree(model.state_dict());target_opt=cpu_tree(opt.state_dict());target_scaler=scaler.state_dict()
    rng=torch.cuda.get_rng_state().clone();cpu_rng=torch.get_rng_state().clone()
    model.eval()
    with torch.inference_mode():features=model.encoder(x[:micro].cuda()).float().cpu().numpy()
    if features.shape[1]!={'clip_vit_b16':512,'dinov2_vit_b14':768,'efficientnet_b0':1280}[backbone]:raise ValueError('feature dimension mismatch')
    if not np.isfinite(features).all() or features.dtype!=np.float32:raise ValueError('invalid FP32 raw features')
    del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
    other=build_model(backbone,weight['weight_path']).cuda().train()
    oo=torch.optim.AdamW(other.parameters(),lr=weight['learning_rate'],weight_decay=5e-4)
    ss=torch.amp.GradScaler('cuda',init_scale=128)
    resumed=load_state(out/'resume_epoch1.pt',other,oo,ss,identity)
    events.append(advance_step(other,oo,ss,x,y,1,1,2,weight['learning_rate'],micro,'cuda'))
    assert_equal_tree(target,other.state_dict());assert_equal_tree(target_opt,oo.state_dict());assert_equal_tree(target_scaler,ss.state_dict())
    if not torch.equal(rng,torch.cuda.get_rng_state()) or not torch.equal(cpu_rng,torch.get_rng_state()):raise ValueError('RNG resume mismatch')
    other.eval()
    with torch.inference_mode():again=other.encoder(x[:micro].cuda()).float().cpu().numpy()
    if not np.array_equal(features,again):raise ValueError('FP32 feature export resume mismatch')
    rejected=False
    try:load_state(out/'resume_epoch1.pt',other,oo,ss,{'changed':'input/config/code'})
    except ValueError:rejected=True
    if not rejected:raise ValueError('changed checkpoint identity was accepted')
    np.save(out/'features_continuous.npy',features,allow_pickle=False)
    np.save(out/'features_resumed.npy',again,allow_pickle=False)
    other.train();torch.cuda.synchronize();bench_start=time.time()
    for step in range(benchmark_steps):
        events.append(advance_step(other,oo,ss,x,y,step,benchmark_steps,2,weight['learning_rate'],micro,'cuda'))
    torch.cuda.synchronize();seconds=time.time()-bench_start
    if any(e['skip'] for e in events):raise ValueError('source-only qualification AMP skip: inspect numeric trace')
    with (out/'updates.jsonl').open('w') as stream:
        for e in events:stream.write(json.dumps(e,allow_nan=False)+'\n')
    result={'status':'PASS','backbone':backbone,'microbatch':micro,'effective_batch':128,
      'device_uuid':visible,'device_name':torch.cuda.get_device_name(),'training_code_sha256':code_hash(),
      'identity':identity,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
      'resume_parameter_optimizer_scaler_rng_byte_exact':True,'resume_FP32_features_byte_exact':True,
      'changed_identity_rejected':True,'all_trainable_parameters_received_gradients':True,
      'features_dimension':features.shape[1],'benchmark_steps':benchmark_steps,'benchmark_seconds':seconds,
      'training_seconds_per128':seconds/benchmark_steps,'benchmark_scope':'real source inputs preloaded; excludes data IO and DEV evaluation',
      'elapsed_seconds':time.time()-start,'target_used':False}
    save_json(out/f'micro{micro}.json',result);print(json.dumps(result),flush=True)


def supervise(root,code):
    import sys
    root=Path(root);code=Path(code);config=json.loads((root/'configs/base.json').read_text())
    from .queue_d import gpu_memory,idle_device
    memory=gpu_memory()
    query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    occupied={r.split(', ')[0] for r in query.strip().splitlines() if r.strip()}
    for i in range(3):
        if memory[i]['uuid']!=config['allowed_gpu_uuids'][i] or not idle_device(memory[i],occupied):
            raise RuntimeError('qualification upfront check: approved GPU not idle, no workers started')
    save_json(root/'qualification/prelaunch_idle_verified.json',{'gpu0_2':{str(i):memory[i] for i in range(3)},'compute_processes':query,'no_assigned_compute_process':True,'unix_time':time.time()})
    if (root/'qualification/gpu_workers.json').exists():
        old=json.loads((root/'qualification/gpu_workers.json').read_text())
        for p in old['workers']:
            if Path(f'/proc/{p["pid"]}').exists():raise ValueError('existing qualification worker still alive')
    make_probe(root);jobs=[]
    for i,backbone in enumerate(config['backbones']):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=config['allowed_gpu_uuids'][i],
          CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(code))
        log=root/'logs'/f'qualify_{backbone}_micro32.log'
        with log.open('w') as stream:
            p=subprocess.Popen([str(root/'environment/bin/python'),'-u','-m','reliability_v2.qualify',
              'worker','--root',str(root),'--backbone',backbone,'--micro',str(32)],
              cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT)
        jobs.append((p,backbone,i,env))
    save_json(root/'qualification/gpu_workers.json',{'workers':[{'pid':p.pid,'backbone':b,'gpu':i} for p,b,i,_ in jobs]})
    passed={}
    for p,b,i,env in jobs:
        rc=p.wait()
        if rc:
            log=(root/'logs'/f'qualify_{b}_micro32.log').read_text()
            if 'out of memory' not in log.lower():raise RuntimeError(f'qualification failed {b}: {log[-3000:]}')
            for micro in [16,8]:
                with (root/'logs'/f'qualify_{b}_micro{micro}.log').open('w') as stream:
                    rc=subprocess.run([str(root/'environment/bin/python'),'-u','-m','reliability_v2.qualify',
                      'worker','--root',str(root),'--backbone',b,'--micro',str(micro)],cwd=code,env=env,stdout=stream,stderr=subprocess.STDOUT).returncode
                if rc==0:passed[b]=json.loads((root/'qualification'/b/f'micro{micro}.json').read_text());break
                log=(root/'logs'/f'qualify_{b}_micro{micro}.log').read_text()
                if 'out of memory' not in log.lower():raise RuntimeError(f'qualification failed {b}: {log[-3000:]}')
            else:raise RuntimeError('no allowed microbatch passed '+b)
        else:passed[b]=json.loads((root/'qualification'/b/'micro32.json').read_text())
    save_json(root/'qualification/gpu_gate.json',{'status':'PASS','backbones':passed,
      'training_code_sha256':code_hash(),'target_used':False,'note':'D/H data/source-DEV acceptance remains a separate gate'})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','supervise']);p.add_argument('--root',required=True)
    p.add_argument('--code');p.add_argument('--backbone');p.add_argument('--micro',type=int,default=32)
    a=p.parse_args()
    if a.action=='worker':worker(a.root,a.backbone,a.micro)
    else:supervise(a.root,a.code)
