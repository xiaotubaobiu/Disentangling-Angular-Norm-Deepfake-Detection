"""Explicit finite plan/pilot/apply/restore CLI; never schedules or dispatches experiments."""
import argparse,pathlib,json,os,fcntl,hashlib,datetime,sys
if sys.flags.optimize:
 raise RuntimeError('Optimized Python disables verification; refusing before any file mutation')
import relocation as m
R=pathlib.Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
C=pathlib.Path('/Dataset/deepfake_detector_v2_cold_20261023/staged_copies')
M=R/'reports/D_closed_cold_copy_manifest_20261024T1510.json'
PLAN=R/'reports/closed_D_auxiliary_cold_relocation_plan_20261024T1845.json'
def prepare():
 assert not PLAN.exists(),'existing plan: inspect rather than overwrite'
 entries=[e for e in json.loads(M.read_text())['entries'] if m.PATTERN.fullmatch(e['relative'])]
 assert len(entries)==775 and sum(e['relative'].endswith('state.pt') for e in entries)==25
 closed={}
 for e in entries:
  m.check(R,C,e,hashes=False);run=e['relative'].split('/')[1];p=R/'runs'/run/'complete.json';comp=json.loads(p.read_text());assert comp['status']=='TRAINING_COMPLETE';closed[str(p)]=m.sha(p)
 doc={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PREPARED_NOT_EXECUTED_FRESH_HASHES_REQUIRED_ON_APPLY','root':str(R),'cold':str(C),'manifest_sha256':m.sha(M),'entries':entries,'closed_completion_sha256':closed,'logical_bytes':sum(e['bytes'] for e in entries),'file_count':len(entries),'state_files':25,'method_arrays':750,'protected_inputs_sha256':json.loads((R/'reports/immutable_inputs_20261024T0730.json').read_text())['protected_inputs_sha256'],'excluded':'qualification weights, final.pt, source_best.pt, features, images, H and live data','SSD_released_bytes':0,'validation':'Before mutation: original inode/size/ns and fresh source/cold SHA. During aliases: native state loader and NumPy reader. Backups retained until validation of every file. Rollback rematerializes cold bytes with original ns if commit interrupted.','restore_headroom_required_bytes':sum(e['bytes'] for e in entries)+50*2**30}
 m.save(PLAN,doc);print(json.dumps({'plan':str(PLAN),'sha256':m.sha(PLAN),'file_count':len(entries),'logical_GiB':doc['logical_bytes']/2**30,'SSD_released_bytes':0}))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','pilot','apply','restore']);ap.add_argument('--plan-sha256');ap.add_argument('--journal');a=ap.parse_args()
 if a.action=='prepare':prepare();return
 assert a.journal,'explicit unique journal is required';j=pathlib.Path(a.journal);assert j.parent.resolve()==R/'reports','journal must be in this study reports'
 qlock=(R/'queue.lock').open('a+');fcntl.flock(qlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if a.action=='restore':m.restore(j);print('RESTORED; fresh capacity remeasurement required');return
 assert a.plan_sha256 and m.sha(PLAN)==a.plan_sha256,'explicit reviewed plan hash required';plan=json.loads(PLAN.read_text());assert m.sha(M)==plan['manifest_sha256']
 for p,h in {**plan['closed_completion_sha256'],**plan['protected_inputs_sha256']}.items():assert m.sha(p)==h,'protected provenance changed'
 free=os.statvfs(R).f_bavail*os.statvfs(R).f_frsize;assert free>=plan['restore_headroom_required_bytes'],'insufficient full rollback plus reserve headroom'
 assert json.loads((R/'queue_status.json').read_text())['stage']=='P3_D_CACHE_COMPLETE_ANALYSIS_PENDING'
 entries=plan['entries'];records=[]
 import torch,numpy as np
 from reliability_v2 import trainer,models,cache_d
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(2);config=json.loads((R/'configs/execution.json').read_text());last_backbone=None;model=None
 def validate(paths):
  nonlocal last_backbone,model
  for p in paths:
   if p.suffix=='.npy':
    array=np.load(p,allow_pickle=False);assert len(array)==102272 and array.dtype==np.dtype('<U64');records.append({'path':str(p),'status':'PASS_NATIVE_NUMPY','rows':len(array)});del array
   else:
    run=p.parent.name;task=config['runs'][run];receipt=json.loads((p.parent/'complete.json').read_text());identity=receipt['identity'];assert identity['code']==trainer.code_hash()
    if task['phase']=='formal':assert cache_d._context(R/'configs/execution.json',run)[0]['training_identity']==identity
    if task['backbone']!=last_backbone:
     model=None;model=models.build_model(task['backbone'],config['backbones'][task['backbone']]['weight_path']);last_backbone=task['backbone']
    optimizer=torch.optim.AdamW(model.parameters(),lr=task['lr'],weight_decay=5e-4);scaler=torch.amp.GradScaler('cpu',init_scale=128) if task.get('precision','AMP')=='AMP' else None
    state=trainer.load_state(p,model,optimizer,scaler,identity)
    if scaler is not None and scaler.state_dict()!=state['scaler']:raise ValueError('restored scaler mismatch')
    if scaler is None and state['scaler'] is not None:raise ValueError('precision/scaler mismatch')
    success=sum(h['success'] for h in state['history']);skips=sum(h['skip'] for h in state['history']);assert (state['epoch'],state['attempts'],success,skips)==(receipt['epochs'],receipt['attempts'],receipt['success'],receipt['skip'])
    assert all(torch.equal(v,state['model'][k]) for k,v in model.state_dict().items());assert all(torch.isfinite(v).all() for v in model.state_dict().values() if torch.is_floating_point(v));steps=[int(x['step'].item()) for x in optimizer.state.values() if 'step' in x];assert len(set(steps))==1 and steps[0]==success
    records.append({'path':str(p),'status':'PASS_NATIVE_TRAINER_LOAD_STATE','epoch':state['epoch'],'attempts':state['attempts'],'successful_updates':success,'AMP_skips':skips,'CPU_model_optimizer_scaler_restored':True,'GPU_RNG_restore':False});del state,optimizer,scaler
  assert not torch.cuda.is_initialized()
  for p,h in plan['protected_inputs_sha256'].items():assert m.sha(p)==h
  m.save(j.with_name(j.stem+'.native_validation.json'),{'status':'PASS_ALL_SELECTED_NATIVE_READERS','records':records,'CUDA_initialized':False})
 result=m.relocate(R,C,entries,j,validate,commit=a.action=='apply');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':main()
