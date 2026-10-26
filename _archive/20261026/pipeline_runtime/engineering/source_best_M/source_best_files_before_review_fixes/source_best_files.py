"""Actual CPU source-best bytes/tensors/head staging for selected M.

Reuses accepted primary terminal proof and freshly verifies the selected file,
source identity and original FP32 head. No target data, encoder construction,
GPU/cache ownership or production inference authorization is provided here.
"""
from pathlib import Path
import gc,hashlib,json,re
import numpy as np
import torch
from . import cache_d,trainer,worker_h,source_best_policy as policy
BASE_SOURCE_PINS = {'reliability_v2/__init__.py': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'reliability_v2/analysis_d.py': 'a1d3be7bd43acb43995768f9167225b066ee9e6691c61c7b8bf28b00de53ba5e', 'reliability_v2/analysis_h.py': '051226f11737615bb3546ae1a301d621fd662943dc1c849e18085e610c865383', 'reliability_v2/bootstrap_fast.py': '7eaeaf0973d72cd1def838cf6bc82863c92c75fb35409553332015dec8b0e4e5', 'reliability_v2/cache_d.py': '8f5814646194c772dcf00f6751a35c2f792dd5f11008e0834fdba4eb5e329761', 'reliability_v2/cache_h.py': '9119200359f8d696f1366e6c0137bca533c63089c32e505971e69f84ebaf40fb', 'reliability_v2/coverage_h.py': 'f1b684903d2486107dc1ba2312502b7f34a66636b196662b0a3566a159475f9d', 'reliability_v2/data_d.py': '7d37fda956a5b5d0d9a4281a24b5967507ce28b892dd3a01486ec05280277cd1', 'reliability_v2/data_h.py': 'ad81f65c27d53159e6902bf8a20b0910cd129f75eb0105e5a5a053bcc2d2f45e', 'reliability_v2/geometry_h.py': '161ac4e837463ff16e7916ca42eeb5d13acc44aab9bcc1c397025b9426ad9532', 'reliability_v2/models.py': '5c4adb39314433c5f0d40311d7ada3ad5d30c7b239773be02080be120f5a2e60', 'reliability_v2/qualification_h.py': '5c70e9fb2a39bd974b0db278d519a8b7103f9553c3bfee32c7accb311edf0ec0', 'reliability_v2/qualify.py': '5e8b6e856f2396bf15fa4e464cb8228a043864bc7b0e90dfcbd9d20f847b06eb', 'reliability_v2/queue_d.py': '3db3d1ebb63fbe233fa19116934268ab999be94faaff59fe1f2d52fdbe1949b8', 'reliability_v2/queue_h.py': 'a2792361443e64622391fb6ae766ee5e6f887789d662afdfb6d809f9a5ea8176', 'reliability_v2/scoring_d.py': '9bddf066fcb8df1b9659d4cfe14aa4febdf850f7cc39c1c0c3d6050864263d1c', 'reliability_v2/stats_h.py': '5627cccc714c2fc59a3dadfed3ddeba32d1488020cfd550ca70d0c9be947dd74', 'reliability_v2/summary_d.py': '489330049b8b95c74db9bb3d191a38f7e7190107788144ff6c07aaec9946db93', 'reliability_v2/trainer.py': '56f0db0634a525a530fb53482732365ebf9693a1f5470b0a95ec0820cc614368', 'reliability_v2/worker_h.py': 'd0219aa52ebb1b6602602d5b09eb5fdfbbb9dc3aeb7b55dbfc0251ced66136f4'}
POLICY_SHA='e407059ce2def098ff46fd0ebf7e61c40f4ab5c81b39608925d501b6e3c07b39'
DIMENSIONS=cache_d.DIMENSIONS
REQUIRED_FILES=('identity.json','resolved_config.json','history.json','complete.json','source_best.pt')
def require(ok,why):
 if not ok:raise ValueError(why)
def signature(p):
 s=Path(p).stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
def digest(p):
 before=signature(p);h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 require(signature(p)==before,'file changed while hashing');return h.hexdigest()
def safe(p,root):
 p=Path(p).absolute();root=Path(root).absolute();require(p.is_relative_to(root) and p.resolve().is_relative_to(root.resolve()),'path escape')
 for item in (p,*p.parents):
  require(not item.is_symlink(),'symlink rejected')
  if item==root:break
 require(p.is_file(),'missing regular file:'+str(p));return p
def checked(p,pin,root,bound):
 p=safe(p,root);policy.digest_string(pin);require(digest(p)==pin,'accepted file hash drift:'+str(p));bound[str(p)]=pin;return p
def read(p):
 require(Path(p).stat().st_size<=32*1024**2,'metadata byte budget exceeded');return json.loads(Path(p).read_text())
def qualified_sources():
 base=Path(__file__).resolve().parent.parent;result={}
 for rel,pin in BASE_SOURCE_PINS.items():
  path=base/rel;require(not path.is_symlink() and digest(path)==pin,'qualified source changed:'+rel);result[str(path)]=pin
 require(digest(policy.__file__)==POLICY_SHA,'qualified source-best policy changed');result[str(Path(policy.__file__))]=POLICY_SHA;result[str(Path(__file__))]=digest(__file__);return result

def _proof_files(root,run,record,task,config_hash,bound):
 p=checked(root/record['proof_relative'],record['proof_sha256'],root,bound);require(p.parent==root/'reports','primary terminal proof location invalid');proof=read(p)
 if task['route']=='D':
  require(proof.get('status')=='PASS_ALL_FIFTEEN_D_FORMAL_TRAINING_TERMINALS_ONLY' and proof.get('all_D_formal_training_complete') is True and proof.get('no_target_performance') is True,'accepted D terminal manifest required')
  checked(proof['acceptance_audit'],proof['acceptance_audit_sha256'],root,bound)
  rows=[x for x in proof['completed_formal'] if x['run_id']==run];require(len(rows)==1,'missing/duplicate primary proof');row=rows[0]
  require(row.get('source_best_equal_saved_state_model') is True and row.get('final_equal_terminal_state_model') is True and row.get('source_DEV_tables_independently_accepted')==task['epochs'],'independent D best/final/sourceDEV proof required')
  core=row['core_file_inventory'];require(set(REQUIRED_FILES)<=set(core) and 'final.pt' in core,'D core file inventory missing')
  expected={name:core[name]['sha256'] for name in REQUIRED_FILES};return expected,core['final.pt']['sha256'],row['identity']
 require(proof.get('status')=='PASS_SELECTED_H_FORMAL_AMP_TERMINAL_FULL_CHECKPOINTS_SOURCE_DEV_AND_PAIRED_STREAMS' and proof.get('source_only') is True and proof.get('CUDA_initialized') is False,'accepted H independent terminal proof required')
 require(proof['inputs']['H_execution_sha256']==config_hash and run in proof['selected_run_ids'],'H terminal config identity changed')
 rows=[x for x in proof['records'] if x['run_id']==run];require(len(rows)==1,'missing/duplicate H terminal proof');row=rows[0]
 require(row.get('source_best_matches_best_state') is True and row.get('final_matches_complete_state') is True and row.get('source_DEV_tables_independently_recomputed')==task['epochs'] and row.get('source_only') is True and row.get('target_used') is False,'independent H best/final/sourceDEV proof required')
 sealp=checked(root/'runs'/run/'H_completion_seal.json',row['terminal_seal_sha256'],root,bound);seal=read(sealp)
 require(seal.get('status')=='SEALED' and seal.get('run_id')==run and seal.get('artifact_root')==str(root) and seal.get('config_path')==str(root/'configs/H_execution.json') and seal.get('target_used') is False,'H seal identity changed')
 require(set(seal['artifact_sha256'])==set(worker_h._completion_artifacts(task['epochs'])),'H complete seal inventory required')
 require(seal.get('H_data_freeze_sha256')==task['H_data_freeze_sha256'] and seal.get('h_code_bundle_sha256')==task['h_code_bundle_sha256'],'H seal frozen source binding changed')
 require(seal['artifact_sha256']['complete.json']==row['terminal_complete_sha256'] and seal['artifact_sha256']['identity.json']==row['identity_sha256'],'H independent/seal proof drift')
 return {name:seal['artifact_sha256'][name] for name in REQUIRED_FILES},seal['artifact_sha256']['final.pt'],None

def _source_context(config_path,run,route):
 if route=='D':
  base,config,task=cache_d._context(config_path,run);return config,task,base['training_identity']
 return worker_h.validate_task(config_path,run)

def _load(p):
 try:return torch.load(p,map_location='cpu',weights_only=True,mmap=True)
 except Exception as error:raise ValueError('invalid safe CPU checkpoint') from error

def _tensor_check(state,maximum):
 require(isinstance(state,dict) and len(state)>0,'nonempty model state required');total=0;dtypes={}
 for key,value in state.items():
  require(isinstance(key,str) and isinstance(value,torch.Tensor) and value.device.type=='cpu' and value.layout==torch.strided and not value.is_complex() and not value.is_quantized,'unsupported model tensor')
  total+=value.numel()*value.element_size();require(total<=maximum,'tensor byte resource budget exceeded')
  require(bool(torch.isfinite(value).all()),'nonfinite model tensor');dtypes[str(value.dtype)]=dtypes.get(str(value.dtype),0)+1
 return dict(tensor_count=len(state),tensor_bytes=total,tensor_dtype_counts=dtypes)

def validate_source_file(root,admission_path,admission_sha,run_id,*,max_checkpoint_bytes=2*1024**3):
 root=Path(root).absolute();require(root.resolve()==root,'canonical artifact root required')
 require(type(max_checkpoint_bytes) is int and 0<max_checkpoint_bytes<=2*1024**3,'checkpoint byte budget must be positive integer<=2GiB')
 require(isinstance(run_id,str) and re.fullmatch(r'[DH]_formal_[A-Za-z0-9_]+',run_id),'unsafe formal run ID')
 require(not torch.cuda.is_initialized(),'source staging must precede CUDA initialization')
 bound=qualified_sources();ap=checked(admission_path,admission_sha,root,bound);admission=read(ap)
 require(admission.get('schema')=='M_SOURCE_BEST_FILE_ADMISSION_V1' and admission.get('artifact_root')==str(root) and admission.get('target_inference_authorized') is False,'source-only input admission required')
 require(admission.get('endpoint_code_sha256')==digest(__file__),'endpoint code admission drift')
 require(run_id in admission.get('records',{}),'endpoint not admitted');record=admission['records'][run_id];route=record['route'];require(route in ('D','H'),'route invalid')
 config_rel='configs/execution.json' if route=='D' else 'configs/H_execution.json';require(record.get('config_relative')==config_rel,'canonical config required')
 config_path=checked(root/config_rel,record['config_sha256'],root,bound);config=read(config_path);task=config.get('runs',{}).get(run_id);require(isinstance(task,dict) and task.get('route')==route,'registered route/task required')
 expected,final_pin,proof_identity=_proof_files(root,run_id,record,task,record['config_sha256'],bound)
 folder=root/'runs'/run_id
 for name,pin in expected.items():
  path=folder/name
  if name=='source_best.pt':require(path.stat().st_size<=max_checkpoint_bytes,'source-best file exceeds byte budget')
  checked(path,pin,root,bound)
 completion=read(folder/'complete.json');history=read(folder/'history.json');identity=read(folder/'identity.json');require(read(folder/'resolved_config.json')==task and identity==completion.get('identity'),'current formal task/identity drift')
 require(completion.get('source_best_sha256')==expected['source_best.pt'] and completion.get('final_sha256')==final_pin,'accepted source-best/final receipt hash mismatch')
 live_config,live_task,live_identity=_source_context(config_path,run_id,route);require(live_config==config and live_task==task and live_identity==identity and (proof_identity is None or proof_identity==identity),'live source training identity changed')
 for field,path in [('train_manifest',root/task['train_manifest']),('dev_manifest',root/task['dev_manifest']),('initialization',Path(config['backbones'][task['backbone']]['weight_path'])),('environment',root/'configs/requirements.execution.lock.txt'),('qualification',root/task['qualification_gate'])]:
  # Initialization can be outside artifact root; original source context already validates its canonical config path/hash.
  if field=='initialization':require(path.is_file() and digest(path)==identity[field],'initialization changed');bound[str(path)]=identity[field]
  else:checked(path,identity[field],root,bound)
 H_registry=None
 if route=='H':
  registry=admission.get('H_registry');require(isinstance(registry,dict) and registry.get('relative')=='H_run_registry.json','H registry admission required');rp=checked(root/registry['relative'],registry['sha256'],root,bound);H_registry=read(rp)
 checkpoint= _load(folder/'source_best.pt');require(isinstance(checkpoint,dict),'checkpoint dictionary required')
 metadata=policy.metadata_endpoint(task,run_id,history,completion,checkpoint,H_registry=H_registry);state=checkpoint.get('model');summary=_tensor_check(state,max_checkpoint_bytes)
 try:w=state['classifier.weight'];b=state['classifier.bias']
 except KeyError as error:raise ValueError('original classifier head missing') from error
 require(w.dtype==b.dtype==torch.float32,'original head must remain FP32');weight=w.detach().cpu().numpy().copy();bias=b.detach().cpu().numpy().copy();cache_d._validate_head(weight,bias,DIMENSIONS[task['backbone']]);head_hash=hashlib.sha256(weight.tobytes(order='C')+bias.tobytes(order='C')).hexdigest()
 equality=None
 if metadata['best']['epoch']==task['epochs']:
  final=checked(folder/'final.pt',final_pin,root,bound);require(final.stat().st_size<=max_checkpoint_bytes,'final file exceeds byte budget');last=_load(final)
  require(last.get('identity')==identity and last.get('epoch')==task['epochs'],'terminal tensor identity drift');_tensor_check(last.get('model'),max_checkpoint_bytes)
  equality=set(last['model'])==set(state) and all(torch.equal(state[k],last['model'][k]) for k in state);require(equality,'best at last epoch differs from accepted final tensor state');del last
 for path,pin in bound.items():require(digest(path)==pin,'source staging input/code changed:'+path)
 require(not torch.cuda.is_initialized(),'CUDA unexpectedly initialized')
 result=dict(metadata,schema='M_SOURCE_BEST_FILE_TENSOR_CONTEXT_V1',status='SOURCE_BEST_BYTES_TENSORS_HEAD_PASS_NO_CACHE_AUTHORIZATION',checkpoint_path=str(folder/'source_best.pt'),source_best_bytes_tensors_head_verified=True,checkpoint_sha256=expected['source_best.pt'],head_sha256=head_hash,dimension=DIMENSIONS[task['backbone']],original_head_dtype='float32',best_final_tensor_equality_checked=equality is not None,best_equals_final_tensors=equality,actual_cache_context_qualified=False,target_inference_authorized=False,reuse_capacity_credit_bytes=0,terminal_optimizer_state_replayed_now=False,accepted_primary_sourceDEV_and_state_relation_proof_reused=True,input_sha256=bound,**summary)
 del checkpoint,state,w,b,weight,bias;gc.collect();return result
