"""Source-CAL-only bridge qualification worker; no owner or production issuer.

The public GPU worker requires inherited genuine phase/native leases after the
original210-role stage. It never forwards a target input, selects a threshold,
trains or declares the45-model M evaluation complete. Full original provenance
validation may read frozen target hashes, exactly as cache.prepare_context does.
"""
from pathlib import Path
import fcntl,json,os,resource,shutil,time
import numpy as np
import torch
from . import source_best_cache as cache,source_best_files as sf,source_best_lease as lease,source_best_gpu as gpu,trainer
require=sf.require
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
PLAN_SCHEMA='M_SOURCE_CAL_ONLY_GPU_QUALIFICATION_PLAN_V1'

def qualification_runs():
 return tuple([f'D_formal_{b}_seed7' for b in BACKBONES]+[f'H_formal_{b}_{h}_seed7' for b in BACKBONES for h in ('B','AB')])
def codes():
 return {str(Path(m.__file__)):sf.digest(m.__file__) for m in (cache,sf,lease,gpu,trainer)}|{str(Path(__file__)):sf.digest(__file__)}

def source_calibration_forward(context,device,guard):
 """Low-level source-only exact FP32 transport test; CPU form is synthetic QA."""
 guard(True);context.check(full=True);route=context.source['route'];role='source_cal' if route=='D' else 'B/SOURCE_CAL';require(role in context.roles,'frozen source calibration role required')
 frame=context.legacy._read_manifest(context.tables[role]['path'],role) if route=='H' else context.legacy._read_manifest(context.tables[role]['path'])
 require(len(frame)==context.tables[role]['rows'] and len(frame)>0 and frame.label.eq(0).all(),'exact full real-only SOURCE_CAL table required')
 checkpoint=sf._load(context.source['checkpoint_path']);require(checkpoint.get('identity')==context.source['training_identity'] and checkpoint.get('epoch')==context.source['best']['epoch'],'selected source-best state changed')
 sf._tensor_check(checkpoint['model'],2*1024**3);w=checkpoint['model']['classifier.weight'].detach().cpu().numpy().copy();b=checkpoint['model']['classifier.bias'].detach().cpu().numpy().copy()
 require(cache._head_hash(w,b,context.base['dimension'])==context.base['original_head_sha256'],'selected original FP32 head mismatch')
 trainer.policy();task=context.task;config=context.config;model=cache._build_model(task['backbone'],config['backbones'][task['backbone']]['weight_path']);model.load_state_dict(checkpoint['model'],strict=True)
 for name,value in model.state_dict().items():require(torch.equal(value.cpu(),checkpoint['model'][name]),'model parameter bytes differ after strict source-state load')
 model=model.to(device=device,dtype=torch.float32).eval();batch=config.get('cache_batch_size',task['microbatch']);workers=config.get('cache_workers',config.get('workers',4));all_h=[]
 require(type(batch) is int and batch>0 and type(workers) is int and workers>=0,'original loader configuration required')
 with torch.inference_mode(),torch.autocast(device_type=torch.device(device).type,enabled=False):
  for x,_ in trainer.loader(frame,task['backbone'],batch,workers=workers):
   guard(False);x=x.to(device=device,dtype=torch.float32);a=model.encoder(x);reference=model.encoder(x)
   require(a.dtype==torch.float32 and a.shape==(len(x),context.base['dimension']) and bool(torch.isfinite(a).all()) and torch.equal(a,reference),'fresh repeated rawFP32 encoder outputs must be finite and bitwise equal')
   all_h.append(a.detach().cpu().numpy().copy());guard(False)
 h=np.concatenate(all_h);require(h.shape==(len(frame),context.base['dimension']) and h.nbytes<=64*1024**2,'bounded complete source-only feature matrix required')
 guard(True)
 def batches(offset):
  for i in range(offset,len(h),batch):yield h[i:i+batch]
 cache.write_role(context,role,w,b,batches);meta,saved_frame,saved,saved_w,saved_b=cache.load_role(context,role)
 require(np.array_equal(saved,h) and np.array_equal(saved_w,w) and np.array_equal(saved_b,b),'actual source cache/head exact bit readback failed')
 context.check(full=True);guard(True)
 return dict(parent_run_id=context.base['parent_run_id'],source_role=role,rows=len(frame),fresh_forward_rows=len(h),FP32_feature_bytes=h.nbytes,original_head_sha256=context.source['head_sha256'],source_best_epoch=context.source['best']['epoch'],checkpoint_sha256=context.source['checkpoint_sha256'],source_only_cache_complete_sha256=sf.digest(cache.role_path(context,role)/'complete.json'),repeated_forward_FP32_bitwise_equal=True,target_encoder_inference=False,threshold_or_test_metric_computed=False,optimizer_updates=0,GPU_executed=torch.device(device).type=='cuda')

def _plan(root,path,pin,input_path,input_sha):
 cache._pin(path,pin,{},root);plan=cache._json(path);require(plan.get('schema')==PLAN_SCHEMA and plan.get('artifact_root')==str(root) and plan.get('input_admission_sha256')==input_sha and plan.get('code_sha256')==codes(),'bound reviewed source-only qualification plan required')
 require(plan.get('target_encoder_inference_authorized') is False and type(plan.get('run_ids')) is list and tuple(plan['run_ids'])==qualification_runs() and plan.get('minimum_source_probe_rows')==416,'fixed9 SOURCE_CAL-only qualification scope required')
 cache._pin(input_path,input_sha,{},root);inp=cache._json(input_path);require(inp.get('schema')==cache.INPUT_SCHEMA and inp.get('artifact_root')==str(root) and inp.get('cache_code_sha256')==sf.digest(cache.__file__) and inp.get('target_inference_authorized') is False,'qualified source input capsule required')
 for name in ('actual45_source_result','independent_actual45_source_result'):
  ref=plan[name];cache._pin(ref['path'],ref['sha256'],{},root);proof=cache._json(ref['path']);require(proof.get('models')==45 and proof.get('target_inference_authorized') is False and proof.get('M_production_admitted') is False,'actual prior45source-only validation required')
 a=cache._json(plan['actual45_source_result']['path']);b=cache._json(plan['independent_actual45_source_result']['path']);require(a.get('status')=='PASS_SELECTED_ACTUAL_SOURCE_CHECKPOINT_BYTES_TENSORS_AND_ORIGINAL_FP32_HEADS_ONLY' and b.get('status')=='PASS_INDEPENDENT_D15_H30_SOURCE_SELECTION_FILES_FINITE_CPU_TENSORS_AND_ORIGINAL_FP32_HEADS_ONLY' and b.get('root_report_sha256')==plan['actual45_source_result']['sha256'] and b.get('admission_sha256')==inp['source_file_admission']['sha256'],'accepted independent45 source proof binding required')
 ref=plan['primary_remaining_budget'];require(Path(ref['path'])==root/'reports/full_capacity_check_pre_H_cache_20261025T0555.json' and ref['sha256']=='3ce2d435d51895b9cfe53ac2f4bde6df12db244a7a2365c89a3f31b418b502cd','original full remaining primary budget required');cache._pin(ref['path'],ref['sha256'],{},root);primary=cache._json(ref['path'])['normal_required_bytes']
 additional=plan['source_only_max_additional_write_bytes'];require(type(primary) is int and primary>0 and type(additional) is int and additional>=1024**3,'complete9-case source qualification write bound required')
 floor=plan['minimum_free_bytes'];require(type(floor) is int and floor>=max(50*1024**3,primary+additional) and shutil.disk_usage(root).free>=floor,'full remaining primary+source qualification+temp/reserve gate required')
 return plan,inp

def closed_primary_stage(root,inp):
 ref=inp['source_file_admission'];cache._pin(ref['path'],ref['sha256'],{},root);source=cache._json(ref['path']);expected=sorted(k for k,v in source['records'].items() if v['route']=='H')
 p=root/'reports/H_primary_cache_stage_complete.json';stage=cache._json(p);queue=cache._json(root/'H_queue_status.json')
 require(len(expected)==30 and stage.get('status')=='PRIMARY30_CACHE_WORKERS_RETURNED_210ROLE_MARKERS_PENDING_INDEPENDENT_ADMISSION' and type(stage.get('role_count')) is int and stage['role_count']==210 and type(stage.get('run_ids')) is list and len(stage['run_ids'])==30 and sorted(stage['run_ids'])==expected,'original210-role GPU cache stage must close before qualification')
 require(type(queue.get('active')) is list and queue['active']==[] and queue.get('stage')=='H_PRIMARY_CACHE_EXPORT_REQUIRES_INDEPENDENT_ROLE_ADMISSION_COMPLETE' and type(queue.get('owner_pid')) is int and queue['owner_pid']>0 and type(queue.get('detail')) is dict and queue['detail'].get('run_ids')==stage['run_ids'],'exact original GPU queue terminal/empty occupancy required')
 for path,pin in stage['bindings'].items():cache._pin(path,pin,{})
 return sf.digest(p)

def _contended_probe(root,relative):
 fd=lease._open_lock(root,relative)
 try:
  try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return True
  raise ValueError('expected phase/native exclusion is not held')
 finally:os.close(fd)

def qualification_worker(root,input_path,input_sha,run_id,*,plan_path,plan_sha,phase_proof,phase_fd):
 root=Path(root).absolute();lease.verify_worker(root,phase_proof,phase_fd,arm_parent_death=True)
 try:
  plan,inp=_plan(root,plan_path,plan_sha,input_path,input_sha);require(run_id in qualification_runs(),'unregistered GPU qualification endpoint');stage_sha=closed_primary_stage(root,inp)
  output=root/'reports/M_source_only_GPU_qualification'/phase_proof['nonce']/(run_id+'.json')
  with cache._lock(output.with_suffix('.lock'),root):
   require(not output.exists(),'qualification receipt already exists; do not repeat')
   context=cache.prepare_context(root,input_path,input_sha,run_id);role='source_cal' if context.source['route']=='D' else 'B/SOURCE_CAL';require(context.tables[role]['rows']>=416,'full actual source table must cover frozen416-row probe')
   def guard(full=False):
    lease.verify_worker(root,phase_proof,phase_fd);_plan(root,plan_path,plan_sha,input_path,input_sha);require(closed_primary_stage(root,inp)==stage_sha,'original phase completion changed')
   start=time.monotonic()
   with lease.native_gpu(root,context.config,phase_proof,phase_fd):
    visible=os.environ.get('CUDA_VISIBLE_DEVICES');require(visible and ',' not in visible,'one approved nativeGPU required')
    require(_contended_probe(root,'H_pipeline.lock') and _contended_probe(root,'gpu_locks/'+visible+'.lock'),'actual phase/native mutual exclusion probes required')
    with cache._lock(context.root/'features/source_best_M'/context.admission_sha/run_id/'extract.lock',context.root):result=source_calibration_forward(context,'cuda',guard)
   guard(True);receipt=dict(result,schema='M_ACTUAL_SOURCE_ONLY_GPU_QUALIFICATION_RETURN_V1',phase_owner_pid=phase_proof['owner_pid'],worker_pid=os.getpid(),phase_nonce=phase_proof['nonce'],approved_visible_GPU=visible,phase_native_mutual_exclusion_verified=True,primary_cache_stage_sha256=stage_sha,input_admission_sha256=input_sha,qualification_plan_sha256=plan_sha,code_sha256=codes(),seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,full_M_cache_or_statistics_qualified=False,M_production_admitted=False)
   cache._inside(output,root);cache._save_json(output,receipt);return receipt
 finally:os.close(phase_fd)
