"""Finite original-HQueue post-standard ->30 formal AMP handoff.

Default is read-only admission checking. --execute is for the root controller
alone. No FP32, paired-target gate, cache or analysis continuation is included.
"""
from pathlib import Path
import argparse,ast,contextlib,datetime,fcntl,hashlib,json,math,os,sys,traceback
import shutil
import numpy as np
sys.dont_write_bytecode=True

ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
SCIENCE=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2/reliability_v2')
G=1024**3
SOURCE_PINS={'queue_h.py':'a2792361443e64622391fb6ae766ee5e6f887789d662afdfb6d809f9a5ea8176',
 'queue_d.py':'3db3d1ebb63fbe233fa19116934268ab999be94faaff59fe1f2d52fdbe1949b8',
 'trainer.py':'56f0db0634a525a530fb53482732365ebf9693a1f5470b0a95ec0820cc614368',
 'worker_h.py':'d0219aa52ebb1b6602602d5b09eb5fdfbbb9dc3aeb7b55dbfc0251ced66136f4'}
AUDIT_PINS={
 'H_first4_terminal_acceptance_20261024T1630.json':'a59d0c4ed879f50c20dfe831f5243fb30f9052c455da2e4814282bfa54e25406',
 'H_DINO_lr1_terminal_acceptance_20261024T1800.json':'6c482a77ff04a235392dc1e17bd866344bf3870f3c3d167c74fa40bb2650a2cd',
 'H_DINO_lr3_terminal_acceptance_20261024T1925.json':'b9b273e4f8649bc84b460cb86d606ca4a524fcf10390552a35d0d80913e7d05d',
 'H_DINO_lr5_terminal_acceptance_20261024T2010.json':'2823215f2ec0b3011557f29c2a31b1e35c95e7fdcb658d23d5b54efa104d00b9',
 'H_EfficientNet_terminal_acceptance_20261024T2045.json':'eae9e20655cdea337cacadca660bf52bbc0ac82f0d6b696d296fc697a30b6c4c'}
PROGRESS='reports/H_source_and_standard_development_progress_20261024T1510.json'
COMPONENTS={'D_remaining_analysis','D_remaining_features','H_training','H_qualification','H_features','H_pair_analysis','future_logs_manifests_figures_audits'}

def require(test,message):
 if not test:raise ValueError(message)

def digest(path):
 path=Path(path);before=path.stat();h=hashlib.sha256()
 with path.open('rb') as stream:
  for b in iter(lambda:stream.read(1024*1024),b''):h.update(b)
 after=path.stat()
 require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'input changed while hashing')
 return h.hexdigest()

def safe(path,root):
 path=Path(path).absolute();root=Path(root).absolute()
 require(path.is_relative_to(root) and path.resolve().is_relative_to(root.resolve()),'path escape')
 for p in (path,*path.parents):
  require(not p.is_symlink(),'symlink rejected')
  if p==root:break
 require(path.is_file(),'missing regular file: '+str(path))
 return path

def source_tree(source,name):
 path=safe(Path(source)/name,source)
 require(digest(path)==SOURCE_PINS[name],'frozen source changed: '+name)
 return ast.parse(path.read_text())

def pure_api(source=SCIENCE):
 """Load unchanged pure/original I/O helpers, never import CUDA/training modules."""
 ns={'np':np,'Path':Path,'json':json,'os':os,'hashlib':hashlib,'fcntl':fcntl,
     'contextmanager':contextlib.contextmanager,'contextlib':contextlib}
 picks={'queue_d.py':{'choose_recipe','dino_extend'},
        'queue_h.py':{'development_id','registered_task','standard_development_tasks','formal_id','formal_tasks','choose_shared_recipe','extend_budget','capacity_check'},
        'trainer.py':{'digest','save_json','exclusive_file'},
        'worker_h.py':{'_completion_artifacts'}}
 for name,names in picks.items():
  tree=source_tree(source,name)
  if name=='queue_h.py':
   for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in {'BACKBONES','HISTORIES','DEVELOPMENT_SEEDS','FORMAL_SEEDS'} for t in n.targets):
     for t in n.targets:
      if isinstance(t,ast.Name):ns[t.id]=ast.literal_eval(n.value)
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
  require(len(nodes)==len(names),'original helper inventory changed')
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(Path(source)/name),'exec'),ns)
 return ns

def completion_inventory(source,epochs):
 return pure_api(source)['_completion_artifacts'](epochs)

def original_lock(source=SCIENCE):
 return pure_api(source)['exclusive_file']

def scientific_block(source=SCIENCE):
 """Exact original run() body from recipes={} through30 formal AMP batch."""
 tree=source_tree(source,'queue_h.py')
 q=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HQueue')
 run=next(n for n in q.body if isinstance(n,ast.FunctionDef) and n.name=='run')
 start=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='recipes' for t in n.targets))
 end=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='batch' and any(isinstance(a,ast.Constant) and a.value=='H_FORMAL_AMP_TRAINING' for a in n.value.args))
 require(start<end,'original block boundary changed')
 return ast.Module(body=run.body[start:end+1],type_ignores=[])

def number(value,name):
 require(not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value) and value>=0,'nonnegative finite '+name+' required')
 return value

def budget_check(root,path,expected,api,now,free):
 safe(path,root);require(isinstance(expected,str) and len(expected)==64 and digest(path)==expected,'immutable budget SHA mismatch')
 b=json.loads(Path(path).read_text())
 require(b.get('schema')=='FULL_REMAINING_PRIMARY_CAPACITY_V2' and b.get('status')=='REGISTERED_PRIMARY_REMAINING_COVERED','approved full remaining primary budget required')
 stamp=datetime.datetime.fromisoformat(b['utc']);require(stamp.tzinfo is not None,'budget UTC timezone required')
 age=(now-stamp).total_seconds();require(0<=age<=3600,'budget must be fresh<=1hour and not future')
 require(set(b.get('remaining_components',{}))==COMPONENTS,'all remaining primary components required')
 for k,v in b['remaining_components'].items():number(v,k)
 total=number(b['normal_remaining_write_bytes'],'remaining total')
 require(math.isclose(sum(b['remaining_components'].values()),total,abs_tol=.01,rel_tol=0),'remaining components disagree with total')
 temporary=number(b['temporary_bytes'],'temporary');reserve=number(b['reserve_bytes'],'reserve')
 require(temporary>=16*G and reserve>=50*G,'16GiB temporary/50GiB reserve floors required')
 require(math.isclose(number(b['normal_required_bytes'],'required'),total+temporary+reserve,abs_tol=.01,rel_tol=0),'budget required total inconsistent')
 require(b.get('source_best_active_budget') is None and b.get('compact_unvalidated_storage_savings_counted') is False and b.get('SSD_cold_release_bytes')==0,'source-best/compact/unreleased cold savings forbidden')
 require(b.get('D_individual_complete_count')==15 and b.get('D_remaining_analysis')==0,'D complete budget basis required')
 require(b['remaining_components']['D_remaining_analysis']==b['remaining_components']['D_remaining_features']==0,'D remaining must be zero')
 persistent=number(b['H_current_persistent_run_bytes'],'persistent H')
 working=number(b['H_persistent_training_budget_total_bytes'],'H training budget')
 require(working>=persistent and math.isclose(working-persistent,b['remaining_components']['H_training'],abs_tol=.01,rel_tol=0),'H persistent-only subtraction required')
 number(b['H_current_temporary_run_bytes'],'H transient');require(b.get('remaining_conditionals',{}).get('H8_DINO_extension_and_longer_formal_aux')==0,'budget presumes no DINO extension')
 base=safe(b['base_live_path'],root);require(digest(base)==b['base_live_sha256'],'budget base identity changed')
 # Original helper adds50GiB; honour any larger approved reserve and round UP.
 extra=math.ceil(total+temporary+reserve-50*G)
 cap=api['capacity_check'](free,initial=False,additional_bytes=extra)
 require(cap['allowed'],'live free capacity does not cover remaining primary+temporary+reserve')
 return b,dict(cap,budget_age_seconds=age,budget_sha256=expected,conditional_FP32_admitted=False)

def live_workers(root,proc_root):
 result=[];config=str(root/'configs/H_execution.json')
 for p in Path(proc_root).glob('[0-9]*'):
  try:args=(p/'cmdline').read_bytes().split(b'\0');words=[a.decode(errors='replace') for a in args if a]
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  if 'reliability_v2.worker_h' in words and (config in words or '--config='+config in words):result.append(int(p.name))
 return result

def preflight(root,budget_path,budget_sha,audit_pins=AUDIT_PINS,source=SCIENCE,proc_root=Path('/proc'),now=None,free_bytes=None,progress=None):
 """Read only: all sealed evidence, real owner state, fresh capacity and source recipe."""
 root=Path(root).absolute();api=pure_api(source)
 now=now or datetime.datetime.now(datetime.timezone.utc)
 require(not (Path(proc_root)/'1993186').exists(),'original owner1993186 must actually exit')
 config_path=safe(root/'configs/H_execution.json',root);config=json.loads(config_path.read_text())
 require(config.get('schema')=='H_EXECUTION_V2' and config.get('artifact_root')==str(root),'original H execution identity required')
 tasks=api['standard_development_tasks'](config);require(len(tasks)==20 and config['runs']==tasks,'exact20 original standard tasks, no existing formal/conditional registration')
 progress_path=safe(root/PROGRESS,root);original_progress=json.loads(progress_path.read_text())
 require(original_progress.get('pid')==1993186 and original_progress.get('status')=='H20_STANDARD_DEVELOPMENT_COMPLETE_REVIEW_BEFORE_CONDITIONALS' and len(original_progress.get('run_ids',[]))==20 and set(original_progress['run_ids'])==set(tasks),'original20-model successful progress required')
 status_path=safe(root/'H_queue_status.json',root);status=json.loads(status_path.read_text())
 require(status.get('stage')=='H_STANDARD_DEVELOPMENT_COMPLETE_CAPACITY_REVIEW_BEFORE_CONDITIONALS' and status.get('active')==[],'standard H queue must be complete and inactive')
 require(not live_workers(root,proc_root),'residual matching H worker active')
 dpath=safe(root/'queue_status.json',root);d=json.loads(dpath.read_text())
 require(d.get('stage')=='P3_D_CACHE_COMPLETE_ANALYSIS_PENDING' and d.get('active')==[],'D GPU stage still has priority')
 require(set(audit_pins)==set(AUDIT_PINS),'exact5 independent standard terminal audits required')
 records={};bound={str(p):digest(p) for p in [config_path,progress_path,status_path,dpath]}
 for name,pin in audit_pins.items():
  p=safe(root/'reports'/name,root);require(digest(p)==pin,'terminal acceptance SHA changed');a=json.loads(p.read_text());bound[str(p)]=pin
  require(a.get('status','').startswith('PASS_') and a.get('source_only') is True and a.get('CUDA_initialized') is False and a.get('production_processes_restarted') is False and len(a.get('records',[]))==4,'complete source-only independent terminal audit required')
  for row in a['records']:
   run=row['run_id'];require(run in tasks and run not in records,'independent terminal coverage duplicate/unknown');records[run]=row
 require(set(records)==set(tasks),'all20 independent terminal records required')
 histories={};inventory=set(api['_completion_artifacts'](30))
 for index,(run,task) in enumerate(tasks.items(),1):
  if progress:progress(f'Validating full standard seal {index}/20: {run}')
  folder=root/'runs'/run;rec=records[run];cp=safe(folder/'complete.json',root);sp=safe(folder/'H_completion_seal.json',root);hp=safe(folder/'history.json',root);state=safe(folder/'state.pt',root)
  require(digest(cp)==rec['terminal_complete_sha256'] and digest(sp)==rec['terminal_seal_sha256'] and digest(state)==rec['state_sha256'],'current complete/seal/state differs from independent acceptance')
  seal=json.loads(sp.read_text());complete=json.loads(cp.read_text())
  require(seal.get('schema')=='H_COMPLETION_SEAL_V1' and seal.get('status')=='SEALED' and seal.get('run_id')==run and seal.get('artifact_root')==str(root) and seal.get('config_path')==str(config_path) and seal.get('target_used') is False and seal.get('H_data_freeze_sha256')==config['H_data_freeze_sha256'] and seal.get('h_code_bundle_sha256')==config['h_code_bundle_sha256'],'seal identity changed')
  require(set(seal.get('artifact_sha256',{}))==inventory and rec.get('seal_current_artifacts_verified')==len(inventory),'full20 seal inventories required')
  for name,pin in seal['artifact_sha256'].items():
   p=safe(folder/name,root);require(digest(p)==pin,'sealed artifact changed: '+str(p));bound[str(p)]=pin
  bound[str(sp)]=rec['terminal_seal_sha256']
  require(rec.get('status')=='PASS_COMPLETE_ATOMIC_CHECKPOINT_CPU_SNAPSHOT' and rec.get('snapshot_epoch')==30 and rec.get('snapshot_attempts')==5730 and rec.get('source_DEV_tables_independently_recomputed')==30 and rec.get('target_used') is False and number(rec.get('source_DEV_max_auc_abs_error'),'DEV recomputation error')<=1e-12,'independent30-epoch source terminal acceptance required')
  require(complete.get('status')=='TRAINING_COMPLETE' and complete.get('epochs')==30 and complete.get('target_used') is False and complete.get('selection')=='frozen_last' and complete.get('attempts')==5730 and complete.get('success')==rec['snapshot_successful_updates'] and complete.get('skip')==rec['snapshot_AMP_skips'],'formal-last development terminal fields changed')
  h=json.loads(hp.read_text());require(len(h)==30 and [r['epoch'] for r in h]==list(range(1,31)),'complete source histories required');histories[run]=h
 decisions={}
 for bb in api['BACKBONES']:
  candidates=[]
  for lr in sorted({t['lr'] for t in tasks.values() if t['backbone']==bb}):
   ids=[api['development_id'](bb,v,s,lr,30) for s in api['DEVELOPMENT_SEEDS'] for v in api['HISTORIES']]
   candidates.append(dict(lr=lr,epochs=30,run_ids=ids,histories=[histories[r] for r in ids]))
  chosen=api['choose_shared_recipe'](candidates);ext=api['extend_budget'](chosen['histories'],30) if bb=='dinov2_vit_b14' else None
  require(ext is None,'DINO extension condition changed; independent source/capacity review required')
  decisions[bb]=dict(lr=chosen['lr'],epochs=30,extension=ext,run_ids=chosen['run_ids'],source_last5_mean=chosen['source_last5_mean'])
 free=shutil.disk_usage(root).free if free_bytes is None else free_bytes
 b,capacity=budget_check(root,budget_path,budget_sha,api,now,free)
 bound[str(Path(budget_path))]=budget_sha;bound[b['base_live_path']]=b['base_live_sha256']
 return dict(status='READ_ONLY_PREFLIGHT_PASS_PRODUCTION_LOCK_NOT_ACQUIRED_ORIGINAL_VALIDATORS_PENDING',accepted_standard_models=20,root=str(root),decisions=decisions,capacity=capacity,input_sha256=bound,science_SHA256=SOURCE_PINS)


def log_progress(message):
 print('['+datetime.datetime.now(datetime.timezone.utc).isoformat()+'] '+message,file=sys.stderr,flush=True)

def verify_snapshot(admission,progress=None):
 root=Path(admission['root']);last=None
 for path,expected in admission['input_sha256'].items():
  p=safe(path,root)
  if progress and p.is_relative_to(root/'runs'):
   run=p.relative_to(root/'runs').parts[0]
   if run!=last:progress('Rechecking immutable admitted evidence '+run);last=run
  require(digest(p)==expected,'admitted immutable input changed before source recipe construction')

def run_original_block(q,api,admission,check_before_batch):
 """Execute the exact original block, with rejection-only boundary guards."""
 ns=dict(api);ns['self']=q
 old_choose=api['choose_shared_recipe'];old_extend=api['extend_budget'];old_batch=q.batch
 def choose(candidates):
  result=old_choose(candidates)
  bb=q.config['runs'][result['run_ids'][0]]['backbone']
  expected=admission['decisions'][bb]
  require(result['lr']==expected['lr'] and result['epochs']==expected['epochs'] and result['run_ids']==expected['run_ids'],'source recipe changed after admission')
  return result
 def extend(histories,budget):
  value=old_extend(histories,budget);require(value is None,'DINO extension requires separate admission; no conditional dispatch');return value
 calls=[]
 def batch(ids,stage):
  require(not calls and stage=='H_FORMAL_AMP_TRAINING' and len(ids)==30,'only one original30-model AMP batch permitted')
  require(all(q.config['runs'][r]['phase']=='formal' and q.config['runs'][r]['precision']=='AMP' for r in ids),'only formal AMP models permitted')
  check_before_batch();log_progress('Final live capacity/worker gate passed: dispatching original30 formal AMP batch');calls.append(stage);return old_batch(ids,stage)
 ns.update(choose_shared_recipe=choose,extend_budget=extend);q.batch=batch
 try:exec(compile(scientific_block(SCIENCE),str(SCIENCE/'queue_h.py'),'exec'),ns)
 finally:q.batch=old_batch
 require(calls==['H_FORMAL_AMP_TRAINING'],'original formal batch boundary not reached')
 return ns

def execute(budget_path,budget_sha):
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','parent process must be CPU-only')
 for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
  require(os.environ.get(name) in ('1','2'),'bounded CPU thread environment required')
 log_progress('Read-only pre-lock admission; no production lock acquired')
 admission=preflight(ROOT,budget_path,budget_sha,progress=log_progress)
 sys.path.insert(0,str(SCIENCE.parent))
 from reliability_v2 import queue_h,data_h,qualification_h
 from reliability_v2.trainer import exclusive_file
 log_progress('Validating original H freeze and source qualification')
 data_h.validate_h_freeze(ROOT);qualification_h.validate_qualification(ROOT)
 log_progress('Attempting original nonblocking H_pipeline.lock; never stealing or unlinking')
 with exclusive_file(ROOT/'H_pipeline.lock'):
  log_progress('Original H_pipeline.lock acquired; repeating admission under ownership')
  # Recheck after ownership acquisition; no trusting the pre-lock observation.
  admission=preflight(ROOT,budget_path,budget_sha,progress=log_progress)
  log_progress('Revalidating original H freeze/qualification under ownership')
  data_h.validate_h_freeze(ROOT);qualification_h.validate_qualification(ROOT)
  log_progress('Original validation passed; constructing original HQueue')
  q=queue_h.HQueue(ROOT,SCIENCE.parent);q.config=queue_h.prepare_execution(ROOT)
  try:
   def before_batch():
    api=pure_api(SCIENCE)
    budget_check(ROOT,budget_path,budget_sha,api,datetime.datetime.now(datetime.timezone.utc),shutil.disk_usage(ROOT).free)
    require(not live_workers(ROOT,Path('/proc')),'unexpected H worker before formal dispatch')
   verify_snapshot(admission,progress=log_progress)
   before_batch()
   log_progress('Admitted source snapshot current; executing exact original recipes-to-formal AST block')
   ns=run_original_block(q,vars(queue_h),admission,before_batch)
   q.status('H_FORMAL_AMP_COMPLETE_INDEPENDENT_TERMINAL_AND_PAIRED_LEDGER_REVIEW_REQUIRED',
      dict(formal_run_ids=list(ns['formal']),independent_terminal_review_pending=True,paired_ledger_pending=True,
           conditional_FP32_not_admitted=True,target_export_allowed=False,cache_or_analysis_started=False))
   return dict(status='H30_FORMAL_AMP_BATCH_COMPLETE_INDEPENDENT_REVIEW_PENDING',formal_run_ids=list(ns['formal']),
               target_export_allowed=False,conditional_FP32_admitted=False,cache_started=False)
  except BaseException:
   q.status('H_POST_STANDARD_FORMAL_AMP_REVIEW_REQUIRED',dict(traceback=traceback.format_exc(),existing_workers_continue_until_own_completion=True));raise

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--budget',type=Path,required=True);p.add_argument('--budget-sha',required=True);p.add_argument('--execute',action='store_true')
 args=p.parse_args(argv)
 result=execute(args.budget,args.budget_sha) if args.execute else preflight(ROOT,args.budget,args.budget_sha,progress=log_progress)
 if not args.execute:
  # Original validators are read-only; check mode never acquires H_pipeline.lock.
  log_progress('Read-only original freeze/qualification validators; no production lock')
  sys.path.insert(0,str(SCIENCE.parent))
  from reliability_v2 import data_h,qualification_h
  data_h.validate_h_freeze(ROOT);qualification_h.validate_qualification(ROOT)
  result['status']='READ_ONLY_PREFLIGHT_AND_ORIGINAL_VALIDATORS_PASS_LOCK_NOT_ACQUIRED'
 bound=result.pop('input_sha256',None)
 if bound is not None:
  result['bound_input_count']=len(bound)
  result['bound_input_inventory_sha256']=hashlib.sha256(json.dumps(bound,sort_keys=True).encode()).hexdigest()
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
