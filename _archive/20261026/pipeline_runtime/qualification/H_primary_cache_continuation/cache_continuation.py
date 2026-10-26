"""Finite exclusive-owner primary H cache continuation, no training/M dispatch.
Operational receipts do not replace independent seven-role/scientific acceptance.
"""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,os,re,shutil,subprocess,sys,uuid
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
SCIENCE=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0');SEEDS=(7,42,123,2026,3407);HISTORIES=('B','AB')
ROLES=('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST','common/B/SOURCE_TEST','common/B/TARGET_TEST','common/AB/SOURCE_TEST','common/AB/TARGET_TEST')
QUEUE_SHA='a2792361443e64622391fb6ae766ee5e6f887789d662afdfb6d809f9a5ea8176'
def require(ok,reason):
 if not ok:raise ValueError(reason)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def effective_ids(reg):
 require(reg.get('schema')=='H_RUN_REGISTRY_V2' and reg.get('status')=='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING' and reg.get('target_export_allowed') is True and reg.get('formal_model_count')==30,'final global registry required')
 formal=[f'H_formal_{b}_{h}_seed{s}' for b in BACKBONES for s in SEEDS for h in HISTORIES]
 require(len(reg.get('formal_run_ids',[]))==30 and set(reg['formal_run_ids'])==set(formal),'formal30 registry identity')
 require(set(reg.get('effective_runs',{}))==set(BACKBONES),'all3 effective bones required')
 ids=[]
 for b in BACKBONES:
  item=reg['effective_runs'][b];precision=item.get('precision');require(precision in ('AMP','FP32'),'invalid effective precision')
  expected=[f'H_formal_{b}_{h}_seed{s}'+('_FP32' if precision=='FP32' else '') for s in SEEDS for h in HISTORIES]
  require(item.get('run_ids')==expected,'effective five-pair run inventory/order changed');ids+=expected
 return ids

def worker_receipt_path(root,nonce,run):
 require(re.fullmatch(r'[A-Za-z0-9_-]{1,64}',nonce) and re.fullmatch(r'[A-Za-z0-9_-]{1,160}',run),'unsafe receipt identifiers')
 return Path(root)/'reports/H_primary_cache_workers'/nonce/(run+'.json')

def verify_worker_receipt(root,run,nonce,bindings,missing_ok=False):
 root=Path(root);p=worker_receipt_path(root,nonce,run)
 if not p.exists() and missing_ok:return False
 require(p.is_file() and not p.is_symlink(),'missing or symlink worker receipt');d=read(p)
 require(d.get('status')=='ORIGINAL_EXTRACT_RETURNED' and d.get('nonce')==nonce and d.get('run_id')==run and d.get('bindings')==bindings,'worker receipt identity drift')
 folder=root/'features/H'/run;marker=folder/'complete.json';require(marker.is_file() and not marker.is_symlink() and sha(marker)==d['aggregate_marker_sha256'],'aggregate marker drift')
 complete=read(marker);require(complete.get('status')=='CACHE_COMPLETE' and complete.get('run_id')==run and complete.get('roles')==d['roles'] and set(d['roles'])==set(ROLES),'aggregate role inventory drift')
 for role,item in d['roles'].items():
  expected=folder/role/'complete.json';require(item.get('path')==str(expected.parent) and expected.is_file() and expected.resolve()==expected.absolute() and not expected.is_symlink(),'role path changed')
  require(sha(expected)==item['complete_sha256'],'role marker SHA drift')
 return True

def capacity_floor(free,required):
 require(isinstance(free,int) and not isinstance(free,bool) and isinstance(required,(int,float)) and not isinstance(required,bool) and free>=required>=0,'live free below immutable full primary bound')
 return dict(free_bytes=free,required_bytes=required,margin_bytes=free-required)

SOURCE_MANIFEST_SHA='977431237b901d5ac88172f98bc2591d3cd7992c35be50546cda34ad2276a747'
def qualified_source_bindings(source=SCIENCE):
 manifest=Path(__file__).parent/'qualified_source_manifest.json';raw=manifest.read_bytes()
 require(hashlib.sha256(raw).hexdigest()==SOURCE_MANIFEST_SHA,'qualified extraction source manifest changed')
 inventory=json.loads(raw);require(inventory['qualified_commit']=='5ef6d06bf7aeda02a5a11113e4c35a2c9909f215' and inventory['source_sha256']['reliability_v2/cache_h.py']=='9119200359f8d696f1366e6c0137bca533c63089c32e505971e69f84ebaf40fb','qualified original cache version changed')
 result={str(manifest):SOURCE_MANIFEST_SHA}
 for rel,pin in inventory['source_sha256'].items():
  path=Path(source)/rel;require(path.is_file() and not path.is_symlink() and sha(path)==pin,'qualified extractor/helper source changed:'+rel);result[str(path)]=pin
 acceptance=Path(inventory['original_H_posttraining_acceptance_path']);pin=inventory['original_H_posttraining_acceptance_sha256'];require(sha(acceptance)==pin,'original cache implementation acceptance changed');result[str(acceptance)]=pin
 return result

def bindings(root):
 root=Path(root);paths=[root/'configs/H_execution.json',root/'H_run_registry.json',Path(__file__),SCIENCE/'reliability_v2/cache_h.py',SCIENCE/'reliability_v2/queue_h.py']
 reg=read(root/'H_run_registry.json');effective_ids(reg)
 for b in BACKBONES:
  precision=reg['effective_runs'][b]['precision'];paths += [root/f'qualification/H_target_export_{b}.json',root/f'reports/H_{b}_{precision}_paired_ledger.json']
 require(sha(SCIENCE/'reliability_v2/queue_h.py')==QUEUE_SHA,'original resource/science queue drift')
 return dict(qualified_source_bindings(SCIENCE),**{str(p):sha(p) for p in paths})

def post_api():
 folder=ROOT/'qualification/H_post_AMP_validation';accept=folder/'component_acceptance.json'
 require(sha(accept)=='04820e0f635a0c0393d741d3b7e7d6067edb5dabf7478b88829658471d6ea0de','qualified post-AMP helper acceptance changed')
 p=folder/'continue_H_post_AMP.py';require(sha(p)==read(accept)['source_sha256'],'qualified budget helper changed')
 spec=importlib.util.spec_from_file_location('qualified_postamp_cache_budget',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def preflight(root,budget,budget_sha):
 root=Path(root);m=post_api();m.no_live_owner(root)
 config=read(root/'configs/H_execution.json');require(config.get('artifact_root')==str(root) and config.get('allowed_gpus')==[0,1,2,3] and config.get('effective_batch')==128,'config resource contract drift')
 reg=read(root/'H_run_registry.json');ids=effective_ids(reg);require(all(run in config['runs'] for run in ids),'unregistered effective run')
 for proc in Path('/proc').glob('[0-9]*/cmdline'):
  try:words=[x.decode(errors='replace') for x in proc.read_bytes().split(b'\0') if x]
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if ('reliability_v2.cache_h' in words or (str(Path(__file__)) in words and '--worker' in words)) and str(root/'configs/H_execution.json') in words:raise ValueError('matching H cache worker already active')
 cap=m.budget_check(root,budget,budget_sha);return dict(status='READ_ONLY_PRIMARY_CACHE_PREFLIGHT_PASS_ORIGINAL_VALIDATORS_PENDING',run_ids=ids,bindings=bindings(root),capacity=cap)

def worker(config,run,nonce,binding_json):
 require(Path(config)==ROOT/'configs/H_execution.json','canonical H config required');expected=json.loads(binding_json);require(bindings(ROOT)==expected and run in effective_ids(read(ROOT/'H_run_registry.json')),'worker provenance changed')
 sys.path.insert(0,str(SCIENCE));from reliability_v2 import cache_h
 require(tuple(cache_h.ROLES)==ROLES,'original seven roles changed')
 result=cache_h.extract(config,run,'cuda');require(bindings(ROOT)==expected,'context changed during original extract')
 folder=ROOT/'features/H'/run;p=worker_receipt_path(ROOT,nonce,run);p.parent.mkdir(parents=True,exist_ok=True)
 receipt=dict(status='ORIGINAL_EXTRACT_RETURNED',utc=utc(),nonce=nonce,run_id=run,worker_pid=os.getpid(),bindings=expected,aggregate_marker_sha256=sha(folder/'complete.json'),roles=result['roles'],independent_scientific_admission=False)
 with p.open('x') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
 verify_worker_receipt(ROOT,run,nonce,expected);return receipt

def queue_type(original,nonce,expected,required,budget,budget_sha):
 class PrimaryCacheQueue(original.HQueue):
  def _verified_completion(self,run):return verify_worker_receipt(self.root,run,nonce,expected,missing_ok=True)
  def _devices(self):
   require(sha(budget)==budget_sha and bindings(self.root)==expected,'dispatch admitted identity changed')
   cap=capacity_floor(shutil.disk_usage(self.root).free,required)
   p=self.root/'reports/H_primary_cache_workers'/nonce/'capacity_observations.jsonl';p.parent.mkdir(parents=True,exist_ok=True)
   with p.open('a') as f:f.write(json.dumps(dict(utc=utc(),**cap))+'\n')
   return super()._devices()
  def _launch(self,key,gpu,device_uuid,*,qualification=None):
   require(qualification is None and key in effective_ids(read(self.root/'H_run_registry.json')) and gpu in (0,1,2,3),'primary effective cache only')
   require(sha(budget)==budget_sha and bindings(self.root)==expected,'launch budget/context changed');capacity_floor(shutil.disk_usage(self.root).free,required)
   env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=device_uuid,CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',PYTHONPATH=str(self.code),PYTHONDONTWRITEBYTECODE='1')
   command=[str(self.root/'environment/bin/python'),'-B','-u',str(Path(__file__)),'--worker','--config',str(self.path),'--run-id',key,'--nonce',nonce,'--binding-json',json.dumps(expected,sort_keys=True)]
   with (self.logs/(key+'_primary_cache_'+nonce+'.log')).open('a') as f:
    require(sha(budget)==budget_sha and bindings(self.root)==expected,'budget/context changed immediately before dispatch')
    process=subprocess.Popen(command,cwd=self.code,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
   self.jobs[key]=dict(process=process,gpu=gpu,uuid=device_uuid,backbone=self.config['runs'][key]['backbone'])
 return PrimaryCacheQueue

def execute(root,budget,budget_sha):
 root=Path(root);require(root==ROOT and os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only canonical owner required');a=preflight(root,budget,budget_sha);m=post_api();sys.path.insert(0,str(SCIENCE))
 from reliability_v2 import queue_h,data_h,qualification_h
 with m.base.original_lock()(root/'H_pipeline.lock'):
  a=preflight(root,budget,budget_sha);data_h.validate_h_freeze(root);qualification_h.validate_qualification(root)
  nonce=uuid.uuid4().hex;kind=queue_type(queue_h,nonce,a['bindings'],a['capacity']['required_bytes'],budget,budget_sha);q=kind(root,SCIENCE)
  q.batch(a['run_ids'],'H_PRIMARY_CACHE_EXPORT_REQUIRES_INDEPENDENT_ROLE_ADMISSION')
  result=dict(status='PRIMARY30_CACHE_WORKERS_RETURNED_210ROLE_MARKERS_PENDING_INDEPENDENT_ADMISSION',utc=utc(),nonce=nonce,run_ids=a['run_ids'],role_count=210,bindings=a['bindings'],scientific_results_complete=False,independent_cache_admission=False)
  queue_h.save_json(root/'reports/H_primary_cache_stage_complete.json',result);return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--budget',type=Path);p.add_argument('--budget-sha');p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--config');p.add_argument('--run-id');p.add_argument('--nonce');p.add_argument('--binding-json');a=p.parse_args()
 if a.worker:result=worker(a.config,a.run_id,a.nonce,a.binding_json)
 else:
  require(a.budget is not None and a.budget_sha is not None,'fresh full primary budget required');result=execute(a.root,a.budget,a.budget_sha) if a.execute else preflight(a.root,a.budget,a.budget_sha)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
