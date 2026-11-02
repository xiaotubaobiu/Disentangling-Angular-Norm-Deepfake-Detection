"""GPU memory qualification and actual CUDA epoch-boundary resume evidence."""
from pathlib import Path
import gc,json,os,subprocess,sys,time
from .manifest import digest,save_json
from .orchestration import require_gpu_gates,exclusive_lock,llm_session,code_hash,validate_devices,termination_guard,owned_process
from .stages import config_identity

def apply_numeric_policy():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def validate_session(config,device_uuid,session_id):
    root=Path(config['artifact_root']);require_gpu_gates(root)
    session=json.loads((root/'service_state.json').read_text())
    if session['session_id']!=session_id or session['state'] not in ['STOPPED','ORIGINALLY_STOPPED']:
        raise ValueError('GPU task has no active owned service session')
    if not Path(f'/proc/{session["owner_pid"]}').exists() or device_uuid not in session['device_uuids'].values() or os.environ.get('CUDA_VISIBLE_DEVICES')!=device_uuid:
        raise ValueError('GPU task device/owner binding invalid')
    apply_numeric_policy()
    import torch
    if torch.cuda.device_count()!=1:raise ValueError('GPU task requires exactly one authorized visible GPU')
    return session

def cuda_resume_evidence(folder):
    import torch,numpy as np,random
    from .training import save_checkpoint,restore_checkpoint,set_random
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    set_random(7)
    model=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.2),torch.nn.Linear(8,2)).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
    scaler=torch.amp.GradScaler('cuda',init_scale=128)
    x=torch.arange(32,dtype=torch.float32,device='cuda').reshape(8,4)/32;y=torch.arange(8,device='cuda')%2
    def step(m,o,s):
        o.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16):loss=torch.nn.functional.cross_entropy(m(x),y)
        s.scale(loss).backward();s.step(o);s.update()
    step(model,opt,scaler)
    identity={'qualification':'CUDA optimizer AMP RNG stochastic forward'}
    save_checkpoint(folder/'resume.pt',model,opt,scaler,1,identity)
    step(model,opt,scaler);target={k:v.clone() for k,v in model.state_dict().items()}
    rng=torch.cuda.get_rng_state().clone();sc=scaler.state_dict()
    other=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.2),torch.nn.Linear(8,2)).cuda()
    oo=torch.optim.AdamW(other.parameters(),lr=1e-4);ss=torch.amp.GradScaler('cuda',init_scale=128)
    restore_checkpoint(folder/'resume.pt',other,oo,ss,identity);step(other,oo,ss)
    exact=all(torch.equal(v,other.state_dict()[k]) for k,v in target.items())
    if not exact or not torch.equal(rng,torch.cuda.get_rng_state()) or ss.state_dict()!=sc:
        raise ValueError('Actual CUDA optimizer/AMP/RNG resume failed equality')
    return {'parameter_byte_exact':True,'CUDA_RNG_exact':True,'AMP_scaler_exact':True,'resume_scope':'epoch_boundary'}

def probe_worker(config,backbone,microbatch,device_uuid,session_id,output):
    validate_session(config,device_uuid,session_id)
    import torch,pandas as pd
    from .models import build_model
    from .training import set_random,transform_image
    set_random(7);root=Path(config['artifact_root']);out=Path(output)
    resume=cuda_resume_evidence(out.parent/'cuda_resume')
    df=pd.read_parquet(root/'data/train.parquet')
    # Inputs are training-only, balanced by label; no target outcomes enter qualification.
    rows=pd.concat([df[df.label.eq(0)].head(64),df[df.label.eq(1)].head(64)],ignore_index=True)
    if len(rows)!=128:raise ValueError('Probe needs128 distinct class-balanced source inputs')
    x=torch.stack([transform_image(p,backbone) for p in rows.image_path]).cuda()
    y=torch.as_tensor(rows.label.to_numpy(),dtype=torch.long,device='cuda')
    model=build_model(backbone,config['backbones'][backbone]['weight_path']).cuda().train()
    opt=torch.optim.AdamW(model.parameters(),lr=config['backbones'][backbone]['learning_rate'],weight_decay=config['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',init_scale=128);torch.cuda.reset_peak_memory_stats();start=time.time();losses=[]
    for step in range(2):
        opt.zero_grad(set_to_none=True)
        for j in range(0,128,microbatch):
            with torch.autocast('cuda',dtype=torch.float16):
                loss=torch.nn.functional.cross_entropy(model(x[j:j+microbatch]),y[j:j+microbatch],reduction='sum')/128
            if not torch.isfinite(loss):raise ValueError('Nonfinite GPU qualification loss')
            scaler.scale(loss).backward();losses.append(float(loss.detach()))
        before=scaler.get_scale();scaler.step(opt);scaler.update()
        if scaler.get_scale()<before:raise ValueError('Qualification AMP overflow; no successful optimizer step')
    # Extraction uses the qualified microbatch, with FP32 model/inference.
    model.eval()
    with torch.inference_mode():h=model.encoder(x[:microbatch]).float()
    if not torch.isfinite(h).all():raise ValueError('Nonfinite raw inference features')
    torch.cuda.synchronize()
    save_json(out,{'status':'PROBE_PASS','backbone':backbone,'microbatch':microbatch,'effective_batch':128,
        'accumulation_steps':128//microbatch,'training_seconds':time.time()-start,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
        'device_name':torch.cuda.get_device_name(),'device_uuid':device_uuid,'losses':losses,'CUDA_resume':resume,
        'target_metrics_used':False,'inference_precision':'FP32','AMP':'float16','TF32':False})

def probe(config,project,config_path):
    project=Path(project);root=Path(config['artifact_root']);require_gpu_gates(root)
    expected={'freeze_sha256':digest(root/'data/freeze_complete.json'),'config_identity':config_identity(config),'code_sha256':code_hash(project)}
    complete=root/'probe_complete.json'
    if complete.exists():
        old=json.loads(complete.read_text())
        if any(old.get(k)!=v for k,v in expected.items()):raise ValueError('Existing probe identity mismatch')
        return old
    folder=root/'qualification';folder.mkdir(exist_ok=True)
    with termination_guard(),exclusive_lock(root/'queue.lock'),llm_session(root,validate_devices(config['physical_gpus'])) as session:
        device_uuid=session['device_uuids'][str(config['physical_gpus'][0])];records={}
        for backbone in config['backbones']:
            for micro in [128,64,32,16]:
                out=folder/f'{backbone}_micro{micro}.json';log=folder/f'{backbone}_micro{micro}.log'
                env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=device_uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
                args=[str(project/'.venv/bin/python'),'-B',str(project/'run.py'),'probe-worker','--config',str(config_path),
                    '--backbone',backbone,'--microbatch',str(micro),'--device-uuid',device_uuid,'--session-id',session['session_id'],'--output',str(out)]
                with log.open('w') as stream:p=owned_process(args,cwd=project,env=env,stdout=stream,stderr=subprocess.STDOUT)
                if p.returncode==0:
                    records[backbone]=json.loads(out.read_text());break
                text=log.read_text()
                if 'out of memory' not in text.lower():raise RuntimeError(f'GPU qualification failed: {log}')
            else:raise RuntimeError(f'No safe microbatch for {backbone}')
        result={'status':'GPU_PROBE_PASS',**expected,'backbones':records,'physical_device_uuids':session['device_uuids'],'target_selection_performed':False}
        save_json(complete,result);return result
