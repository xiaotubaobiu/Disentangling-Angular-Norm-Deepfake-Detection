"""Recorded finite CPU command: existing full cache verification, then existing analysis.
No scheduler, GPU work, scientific parameter override or product-code change.
"""
import pathlib,json,os,sys,time,datetime,hashlib,traceback,shutil
import numpy as np
import torch
from reliability_v2 import cache_d,analysis_d
R=pathlib.Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
RUN='D_formal_dinov2_vit_b14_seed3407'
REPORT=R/'reports/D_formal_dinov2_vit_b14_seed3407_analysis_progress.json'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p): return analysis_d._digest(p)
state=dict(run_id=RUN,pid=os.getpid(),started_utc=now(),status='FULL_SEVEN_ROLE_ADMISSION_RUNNING',CPU_only=True,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),calibration_repeats=200,test_bootstrap_repeats=2000,roles={})
def save():
 state['utc']=now();analysis_d._save_json(REPORT,state)
 print(json.dumps(dict(utc=state['utc'],status=state['status'],roles_passed=list(state['roles'])),ensure_ascii=False),flush=True)
try:
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 torch.set_num_threads(2)
 assert not torch.cuda.is_initialized()
 save()
 base,config,task=cache_d._context(R/'configs/execution.json',RUN)
 root=R/'features/D'/RUN
 marker=json.loads((root/'complete.json').read_text())
 assert marker['status']=='CACHE_COMPLETE' and marker['identity']==base and set(marker['roles'])==set(cache_d.ROLES)
 freeze=json.loads((R/'data/D/freeze.json').read_text())
 assert freeze['status']=='FROZEN' and freeze['method_count']==21
 checkpoint=torch.load(base['checkpoint_path'],map_location='cpu',weights_only=False)
 assert checkpoint['identity']==base['training_identity'] and checkpoint['epoch']==task['epochs']
 W=checkpoint['model']['classifier.weight'].detach().cpu().numpy().copy()
 b=checkpoint['model']['classifier.bias'].detach().cpu().numpy().copy()
 del checkpoint
 cache_d._validate_head(W,b,base['dimension'])
 state['identity']=base;state['cache_completion_sha256']=digest(root/'complete.json');state['D_freeze_sha256']=digest(R/'data/D/freeze.json')
 for role in cache_d.ROLES:
  state['current_role']=role;save();started=time.monotonic()
  manifest=R/'data/D'/f'{role}.parquet'
  assert digest(manifest)==freeze['manifest_sha256'][manifest.name]
  identity=cache_d._role_identity(base,role,manifest)
  meta,frame=cache_d.verify_cache(root/role,identity=identity)
  assert digest(root/role/'complete.json')==marker['roles'][role]['complete_sha256']
  assert len(frame)==meta['rows']==marker['roles'][role]['rows']
  assert np.array_equal(np.load(root/role/'head_weight.npy',allow_pickle=False),W)
  assert np.array_equal(np.load(root/role/'head_bias.npy',allow_pickle=False),b)
  state['roles'][role]=dict(status='PASS_FULL_CURRENT_CACHE_AND_ORIGINAL_HEAD',rows=len(frame),dimension=meta['dimension'],chunk_count=len(meta['chunks']),zero_norm_count=meta['zero_norm_count'],source_manifest_sha256=identity['source_manifest_sha256'],cache_completion_sha256=digest(root/role/'complete.json'),seconds=time.monotonic()-started,utc=now())
  del frame,meta;save()
 assert cache_d._context(R/'configs/execution.json',RUN)[0]==base
 assert digest(root/'complete.json')==state['cache_completion_sha256']
 assert not torch.cuda.is_initialized()
 state['status']='FULL_SEVEN_ROLE_ADMISSION_PASSED';state['admission_passed_utc']=now();save()
 analysis_d._save_json(R/'reports/D_formal_dinov2_vit_b14_seed3407_analysis_admission.json',state.copy())
 assert shutil.disk_usage(R).free>=250*1024**3
 state['status']='PRODUCTION_ANALYSIS_RUNNING';state['analysis_started_utc']=now();save()
 result=analysis_d.analyze_run(R/'configs/execution.json',RUN)
 assert result['identity']['calibration_repeats']==200 and result['identity']['test_bootstrap_repeats']==2000
 assert not torch.cuda.is_initialized()
 state['status']='PRODUCTION_ANALYSIS_TERMINAL';state['analysis_result_status']=result['status'];state['analysis_completion_sha256']=digest(R/'analysis/D'/RUN/'complete.json');state['ended_utc']=now();save()
except BaseException as error:
 state['status']='FAILED';state['error_type']=type(error).__name__;state['error']=str(error);state['ended_utc']=now();save();traceback.print_exc();raise
