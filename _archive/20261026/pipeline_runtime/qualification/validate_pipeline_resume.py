"""Operational source-only check of the actual sampler/augmentation/loader resume path."""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import traceback

ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
CODE=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
GPU='GPU-626ec32f-1401-c959-74b0-db04ed63c646'
BACKBONES=['clip_vit_b16','dinov2_vit_b14','efficientnet_b0']
EXPECTED_CODE='336f3097434edf903a5ff402162e29afeca2442ca50f9c7d9883ca60136b8335'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temp,path)


def idle_gpu():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    if any(line.split(',')[0].strip()==GPU for line in apps.splitlines()):raise RuntimeError('Assigned GPU3 has a compute process')
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used','--format=csv,noheader,nounits'],text=True)
    row=[line.split(',') for line in text.splitlines() if GPU in line]
    if len(row)!=1:raise RuntimeError('Assigned GPU UUID not uniquely found')
    index,uuid,memory=[v.strip() for v in row[0]]
    if index!='3' or uuid!=GPU or int(memory)>=100:raise RuntimeError(f'Assigned GPU3 not idle below100MiB: {row}')
    return {'index':3,'uuid':uuid,'memory_used_MiB':int(memory)}


def worker(root,backbone):
    sys.path.insert(0,str(CODE))
    import numpy as np
    import pandas as pd
    import torch
    from reliability_v2 import trainer
    from reliability_v2.models import build_model

    trainer.policy();torch.set_num_threads(2)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU or torch.cuda.device_count()!=1:
        raise ValueError('Worker requires exactly assigned GPU3 UUID')
    out=root/'qualification'/f'pipeline_resume_{backbone}';out.mkdir(parents=True,exist_ok=True)
    if (out/'receipt.json').exists():raise ValueError('Refusing to overwrite completed pipeline-resume evidence')
    start=time.time();code_start=trainer.code_hash();script_start=digest(__file__)
    if code_start!=EXPECTED_CODE:raise ValueError('Final trainer hash differs from parent freeze')
    config=json.loads((root/'configs/base.json').read_text());weight=config['backbones'][backbone]
    if digest(weight['weight_path'])!=weight['weight_sha256']:raise ValueError('Initialization hash mismatch')
    source_path=root/'data/D/qualification_source.parquet'
    frame=pd.read_parquet(source_path)
    if not frame.domain.eq('ff').all() or not frame.role.eq('train').all() or set(frame.label)!={0,1}:
        raise ValueError('Only real D source TRAIN inputs with both classes are allowed')
    original_mass=float(frame.sampling_mass.sum())
    if not np.isfinite(original_mass) or original_mass<=0:raise ValueError('Invalid tiny subset mass')
    factor=1/original_mass
    frame=frame.copy();frame['sampling_mass']=frame.sampling_mass*factor
    frame['method_contributions']=frame.method_contributions.map(lambda value:json.dumps({k:v*factor for k,v in json.loads(value).items()},sort_keys=True))
    if not np.isclose(frame.sampling_mass.sum(),1):raise ValueError('Tiny source normalization failed')
    for row in frame.itertuples():
        if not np.isclose(sum(json.loads(row.method_contributions).values()),row.sampling_mass):raise ValueError('Tiny method masses not normalized by same factor')
        stat=Path(row.image_path).stat()
        if stat.st_size!=row.bytes or stat.st_mtime_ns!=row.mtime_ns or digest(row.image_path)!=row.content_sha256:
            raise ValueError('Source image changed since D freeze')
    manifest=out/'normalized_source.parquet';frame.to_parquet(manifest,index=False)
    seed=1007;lr=weight['learning_rate'];micro=32;batch=128;workers=4
    identity={'code':code_start,'script':script_start,'data':digest(manifest),'original_source':digest(source_path),
        'config':digest(root/'configs/base.json'),'environment':digest(root/'configs/requirements.execution.lock.txt'),
        'initialization':weight['weight_sha256'],'D_freeze':digest(root/'data/D/freeze.json'),
        'backbone':backbone,'microbatch':micro,'effective_batch':batch,'epochs':2,'attempts_per_epoch':1,'seed':seed,'workers':workers}
    save_json(out/'identity.json',identity)

    def tensor_bytes(value):
        tensor=value.detach().cpu().contiguous()
        return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()

    def add_tree(h,value):
        if torch.is_tensor(value):
            h.update(b'tensor'+str(value.dtype).encode()+repr(tuple(value.shape)).encode());h.update(tensor_bytes(value))
        elif isinstance(value,np.ndarray):h.update(b'numpy'+value.dtype.str.encode()+repr(value.shape).encode()+value.tobytes())
        elif isinstance(value,dict):
            h.update(b'dict')
            for key in sorted(value,key=lambda k:(type(k).__name__,repr(k))):add_tree(h,key);add_tree(h,value[key])
        elif isinstance(value,(tuple,list)):
            h.update(type(value).__name__.encode())
            for v in value:add_tree(h,v)
        elif isinstance(value,float):h.update(b'float'+struct.pack('!d',value))
        else:h.update(type(value).__name__.encode()+repr(value).encode())

    def tree_hash(value):
        h=hashlib.sha256();add_tree(h,value);return h.hexdigest()

    def assert_same(a,b,path=''):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.dtype!=b.dtype or a.shape!=b.shape or tensor_bytes(a)!=tensor_bytes(b):raise AssertionError('Tensor byte mismatch '+path)
        elif isinstance(a,np.ndarray):
            if not isinstance(b,np.ndarray) or a.dtype!=b.dtype or a.shape!=b.shape or a.tobytes()!=b.tobytes():raise AssertionError('Array byte mismatch '+path)
        elif isinstance(a,dict):
            if a.keys()!=b.keys():raise AssertionError('Dict keys mismatch '+path)
            for key in a:assert_same(a[key],b[key],path+'/'+str(key))
        elif isinstance(a,(tuple,list)):
            if type(a)!=type(b) or len(a)!=len(b):raise AssertionError('Sequence mismatch '+path)
            for i,(x,y) in enumerate(zip(a,b)):assert_same(x,y,path+'/'+str(i))
        elif isinstance(a,float):
            if struct.pack('!d',a)!=struct.pack('!d',b):raise AssertionError('Float byte mismatch '+path)
        elif a!=b:raise AssertionError('Value mismatch '+path)

    def run_epoch(model,opt,scaler,epoch,attempt,prefix):
        model.train();draws,methods=trainer.epoch_draws(frame,seed,epoch,128)
        aug=np.asarray([[p['flip'],p['brightness'],p['contrast']] for p in (
            trainer.augmentation_parameters(seed,epoch,str(frame.iloc[int(index)].get('mother_id',frame.iloc[int(index)].get('sample_id',frame.iloc[int(index)].rgb_sha256))),i)
            for i,index in enumerate(draws))],dtype=np.float64)
        for name,array in [('samples',draws),('methods',methods),('augmentations',aug)]:np.save(out/f'{prefix}_{name}.npy',array,allow_pickle=False)
        events=[];batch_hashes=[];loader_object=trainer.loader(frame,backbone,128,draws,seed,epoch,workers)
        if loader_object.num_workers!=4:raise AssertionError('Actual loader worker count differs')
        for x,y in loader_object:
            if len(y)!=128:raise AssertionError('Actual effective batch differs')
            batch_hashes.append({'inputs':tree_hash(x),'labels':tree_hash(y)})
            event=trainer.advance_step(model,opt,scaler,x,y,attempt,1,2,lr,micro,'cuda');event['epoch']=epoch
            if event['skip'] or not event['gradient_finite']:raise FloatingPointError('Source actual pipeline AMP update skipped/nonfinite')
            events.append(event);attempt+=1
        if len(events)!=1 or attempt!=epoch:raise AssertionError('Epoch attempt clock differs')
        result={'epoch':epoch,'attempts_after':attempt,'events':events,'batch_hashes':batch_hashes,
            'sampler_sha256':digest(out/f'{prefix}_samples.npy'),'methods_sha256':digest(out/f'{prefix}_methods.npy'),
            'augmentations_sha256':digest(out/f'{prefix}_augmentations.npy'),'loader_workers':4,'draw_count':128}
        save_json(out/f'{prefix}_record.json',result)
        print(json.dumps({'backbone':backbone,'phase':prefix,'event':events[0]}),flush=True)
        return result

    def export(model,path):
        model.eval();features=[]
        with torch.inference_mode():
            for x,y in trainer.loader(frame,backbone,32,workers=4):features.append(model.encoder(x.cuda()).float().cpu().numpy())
        array=np.concatenate(features)
        if array.dtype!=np.float32 or not np.isfinite(array).all() or array.shape[1]!={'clip_vit_b16':512,'dinov2_vit_b14':768,'efficientnet_b0':1280}[backbone]:
            raise AssertionError('Invalid FP32 raw feature export')
        np.save(path,array,allow_pickle=False)
        return array

    trainer.set_random(seed);model=build_model(backbone,weight['weight_path']).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=5e-4);scaler=torch.amp.GradScaler('cuda',init_scale=128)
    torch.cuda.reset_peak_memory_stats()
    first=run_epoch(model,opt,scaler,1,0,'epoch_01')
    trainer.save_state(out/'epoch_01_state.pt',model,opt,scaler,1,identity,[first],1)
    continuous=run_epoch(model,opt,scaler,2,1,'epoch_02_continuous')
    trainer.save_state(out/'epoch_02_continuous_state.pt',model,opt,scaler,2,identity,[first,continuous],2)
    raw=export(model,out/'features_continuous.npy')
    continuous_state=torch.load(out/'epoch_02_continuous_state.pt',map_location='cpu',weights_only=False)
    # Export through an independent four-worker loader must leave model RNG intact.
    assert_same(continuous_state['torch_rng'],torch.get_rng_state(),'continuous_export_CPU_rng')
    assert_same(continuous_state['cuda_rng'],torch.cuda.get_rng_state_all(),'continuous_export_CUDA_rng')
    del model,opt,scaler;gc.collect();torch.cuda.empty_cache()
    model=build_model(backbone,weight['weight_path']).cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=5e-4);scaler=torch.amp.GradScaler('cuda',init_scale=128)
    state=trainer.load_state(out/'epoch_01_state.pt',model,opt,scaler,identity)
    if state['attempts']!=1 or state['epoch']!=1:raise AssertionError('Epoch1 restore clock incorrect')
    resumed=run_epoch(model,opt,scaler,2,state['attempts'],'epoch_02_resumed')
    for key in ['sampler_sha256','methods_sha256','augmentations_sha256','batch_hashes','events','attempts_after']:
        assert_same(continuous[key],resumed[key],'epoch2/'+key)
    trainer.save_state(out/'epoch_02_resumed_state.pt',model,opt,scaler,2,identity,[first,resumed],2)
    resumed_state=torch.load(out/'epoch_02_resumed_state.pt',map_location='cpu',weights_only=False)
    exact={}
    for key in ['model','optimizer','scaler','python_rng','numpy_rng','torch_rng','cuda_rng','epoch','attempts','scheduler_clock','identity','history']:
        assert_same(continuous_state[key],resumed_state[key],key)
        exact[key]={'byte_exact':True,'continuous_sha256':tree_hash(continuous_state[key]),'resumed_sha256':tree_hash(resumed_state[key])}
    again=export(model,out/'features_resumed.npy');assert_same(raw,again,'FP32_features')
    assert_same(resumed_state['torch_rng'],torch.get_rng_state(),'resumed_export_CPU_rng')
    assert_same(resumed_state['cuda_rng'],torch.cuda.get_rng_state_all(),'resumed_export_CUDA_rng')
    rejected=[]
    for field in ['config','data','code','initialization']:
        changed={**identity,field:'intentional_changed_identity'}
        try:trainer.load_state(out/'epoch_01_state.pt',model,opt,scaler,changed)
        except ValueError as error:
            if 'identity mismatch' not in str(error):raise
            rejected.append(field)
        else:raise AssertionError('Changed identity accepted: '+field)
    torch.cuda.synchronize();code_end=trainer.code_hash()
    if code_start!=code_end or script_start!=digest(__file__):raise AssertionError('Code changed during actual pipeline resume check')
    receipt={'status':'PASS','backbone':backbone,'device_uuid':GPU,'device_name':torch.cuda.get_device_name(),
        'source_only':True,'target_used':False,'data_rows':len(frame),'original_source_sha256':digest(source_path),'normalized_source_sha256':digest(manifest),
        'mass_normalization_factor':factor,'method_contributions_same_normalization_factor':True,'all_source_input_byte_hashes_verified':True,
        'training_code_sha256_start':code_start,'training_code_sha256_end':code_end,'script_sha256':script_start,'identity':identity,
        'microbatch':32,'effective_batch':128,'loader_workers':4,'epochs':2,'attempts_per_epoch':1,'continuous_attempts':2,'resumed_attempts':2,
        'epoch2_sampler_and_augmentations_byte_exact':True,'epoch2_decoded_augmented_batches_byte_exact':True,
        'model_optimizer_scaler_all_RNG_byte_exact':True,'FP32_features_byte_exact':True,'feature_shape':list(raw.shape),
        'FP32_features_sha256':digest(out/'features_continuous.npy'),'changed_identity_fields_rejected':rejected,
        'state_comparisons':exact,'epoch1':first,'epoch2_continuous':continuous,'epoch2_resumed':resumed,
        'peak_memory_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.time()-start,
        'state_byte_comparison_scope':'Tensor/array dtype,shape,raw bytes; exact Python scalar/container state. Serialized torch zip file bytes need not match.',
        'evidence_sha256':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='receipt.json'}}
    save_json(out/'receipt.json',receipt);print(json.dumps({'status':'PASS','backbone':backbone,'elapsed_seconds':receipt['elapsed_seconds'],'receipt':str(out/'receipt.json'),'code_hash':code_end}),flush=True)


def supervise(root):
    root=Path(root);logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    initial=idle_gpu();script_hash=digest(__file__);records=[]
    save_json(logs/'pipeline_resume_supervisor.json',{'status':'RUNNING','pid':os.getpid(),'assigned_gpu':initial,'script_sha256':script_hash})
    for backbone in BACKBONES:
        before=idle_gpu();env={**os.environ,'PYTHONPATH':str(CODE),'CUDA_VISIBLE_DEVICES':GPU,'CUBLAS_WORKSPACE_CONFIG':':4096:8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'2','NUMEXPR_MAX_THREADS':'1'}
        command=[str(root/'environment/bin/python'),'-u',str(Path(__file__).resolve()),'worker','--root',str(root),'--backbone',backbone]
        log_path=logs/f'pipeline_resume_{backbone}.log'
        with log_path.open('w') as log:
            process=subprocess.Popen(command,cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
            record={'backbone':backbone,'pid':process.pid,'command':command,'log':str(log_path),'idle_check':before,'status':'RUNNING'};records.append(record)
            save_json(logs/'pipeline_resume_supervisor.json',{'status':'RUNNING','pid':os.getpid(),'assigned_gpu':initial,'script_sha256':script_hash,'workers':records})
            code=process.wait()
        record['returncode']=code;record['status']='PASS' if code==0 else 'FAILED'
        if code:
            save_json(logs/'pipeline_resume_supervisor.json',{'status':'FAILED','pid':os.getpid(),'script_sha256':script_hash,'workers':records})
            raise RuntimeError(f'Actual pipeline resume failed {backbone}; see {log_path}')
        print(json.dumps(record),flush=True)
    receipts={b:json.loads((root/'qualification'/f'pipeline_resume_{b}'/'receipt.json').read_text()) for b in BACKBONES}
    hashes={v['training_code_sha256_start'] for v in receipts.values()}|{v['training_code_sha256_end'] for v in receipts.values()}
    if len(hashes)!=1:raise RuntimeError('Trainer/model code changed between backbone qualifications')
    result={'status':'PASS','pid':os.getpid(),'workers':records,'training_code_sha256':next(iter(hashes)),'script_sha256':script_hash,'source_only':True,'target_used':False,'backbones':{b:{k:v[k] for k in ['status','elapsed_seconds','peak_memory_bytes','normalized_source_sha256','FP32_features_sha256']} for b,v in receipts.items()}}
    save_json(logs/'pipeline_resume_complete.json',result);save_json(logs/'pipeline_resume_supervisor.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['supervise','worker']);parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--backbone',choices=BACKBONES)
    args=parser.parse_args()
    try:
        if args.action=='worker':
            sys.path.insert(0,str(CODE))
            from reliability_v2 import trainer
            base=json.loads((args.root/'configs/base.json').read_text())
            with trainer.owned_gpu(args.root,base):worker(args.root,args.backbone)
        else:supervise(args.root)
    except BaseException as error:
        if args.action=='worker' and args.backbone:
            save_json(args.root/'qualification'/f'pipeline_resume_{args.backbone}'/'failure.json',{'status':'FAILED','error':repr(error),'traceback':traceback.format_exc(),'pid':os.getpid(),'source_only':True,'target_used':False})
        raise

