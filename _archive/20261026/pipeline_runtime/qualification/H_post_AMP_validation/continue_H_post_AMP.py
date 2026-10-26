"""Finite original source-only post-AMP ledger/registry continuation; no GPU dispatch.
Default is read-only. Original owner must exit before either real preflight or execute.
"""
from pathlib import Path
import argparse,ast,datetime,hashlib,importlib.util,json,math,os,shutil,sys
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
SCIENCE=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2/reliability_v2')
BASE=ROOT/'qualification/continue_H_formal_AMP.py'
BASE_SHA='3955e3299f1bcdc2529cb196af17902df376ec2eaf685560cd08aba41b10dadc'
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==BASE_SHA,'original helper changed'
_spec=importlib.util.spec_from_file_location('original_H_formal_helper',BASE)
base=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(base)
G=1024**3
COMPONENTS=base.COMPONENTS
require=base.require
class UniformFP32ReviewRequired(RuntimeError):pass

def scientific_tail(source=SCIENCE):
 tree=base.source_tree(source,'queue_h.py')
 q=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='HQueue')
 run=next(n for n in q.body if isinstance(n,ast.FunctionDef) and n.name=='run')
 starts=[i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='effective' for t in n.targets)]
 require(len(starts)==1,'original tail boundary changed')
 return ast.Module(body=run.body[starts[0]:],type_ignores=[])

def no_live_owner(root,proc_root=Path('/proc')):
 require(not (Path(proc_root)/'2222605').exists(),'original finite owner2222605 must actually exit')
 require(not base.live_workers(root,proc_root),'matching H worker still active')

def budget_check(root,path,expected,*,now=None,free=None):
 root=Path(root);path=base.safe(path,root);require(base.digest(path)==expected,'budget SHA changed')
 b=json.loads(path.read_text());now=now or datetime.datetime.now(datetime.timezone.utc)
 require(b.get('schema')=='FULL_REMAINING_PRIMARY_CAPACITY_V2' and b.get('status')=='REGISTERED_PRIMARY_REMAINING_COVERED','full primary budget required')
 stamp=datetime.datetime.fromisoformat(b['utc']);require(stamp.tzinfo is not None and 0<=(now-stamp).total_seconds()<=3600,'budget age must be0..3600sec')
 require(set(b.get('remaining_components',{}))==COMPONENTS,'all remaining components required')
 n=base.number
 for k,v in b['remaining_components'].items():n(v,k)
 total=n(b['normal_remaining_write_bytes'],'remaining total');temporary=n(b['temporary_bytes'],'temp');reserve=n(b['reserve_bytes'],'reserve')
 require(math.isclose(total,sum(b['remaining_components'].values()),abs_tol=.01,rel_tol=0),'component sum drift')
 require(temporary>=16*G and reserve>=50*G,'16GiB temp50GiB reserve required')
 required=n(b['normal_required_bytes'],'normal required');require(math.isclose(required,total+temporary+reserve,abs_tol=.01,rel_tol=0),'required total drift')
 require(b.get('source_best_active_budget') is None and b.get('compact_unvalidated_storage_savings_counted') is False and b.get('SSD_cold_release_bytes')==0,'no M/compact/cold savings credit')
 require(b.get('D_individual_complete_count')==15 and b.get('D_remaining_analysis')==0,'D15complete budget basis required')
 require(b['remaining_components']['D_remaining_analysis']==b['remaining_components']['D_remaining_features']==0,'D remaining must be0')
 persistent=n(b['H_current_persistent_run_bytes'],'persistent');working=n(b['H_persistent_training_budget_total_bytes'],'working');n(b['H_current_temporary_run_bytes'],'transient')
 require(math.isclose(max(0,working-persistent),b['remaining_components']['H_training'],abs_tol=.01,rel_tol=0),'clamped persistent-only subtraction required')
 require(b.get('remaining_conditionals',{}).get('H8_DINO_extension_and_longer_formal_aux')==0,'DINO extension budget changed')
 require(base.digest(base.safe(b['base_live_path'],root))==b['base_live_sha256'],'base budget changed')
 free=shutil.disk_usage(root).free if free is None else n(free,'livefree');require(free>=required,'full remaining capacity failed')
 return dict(required_bytes=required,free_bytes=free,margin_bytes=free-required,budget_sha256=expected,conditional_FP32_admitted=False)

def preflight(root,budget,budget_sha,audit_list,audit_list_sha,*,proc_root=Path('/proc')):
 root=Path(root);no_live_owner(root,proc_root);api=base.pure_api(SCIENCE)
 config_path=base.safe(root/'configs/H_execution.json',root);config=json.loads(config_path.read_text())
 require(config.get('artifact_root')==str(root) and config.get('schema')=='H_EXECUTION_V2','config identity drift')
 require(config.get('allowed_gpus')==[0,1,2,3] and config.get('effective_batch')==128,'GPU/effective batch contract drift')
 recipe_path=base.safe(root/'configs/H_frozen_recipes.json',root);recipe=json.loads(recipe_path.read_text());recipes=recipe['recipes']
 for key,path in [('H_data_freeze_sha256',root/'data/H/freeze.json'),('H_qualification_sha256',root/'qualification/H_gate.json')]:require(base.digest(base.safe(path,root))==config[key],'freeze/qualification changed')
 require(recipe['H_data_freeze_sha256']==config['H_data_freeze_sha256'] and recipe['h_code_bundle_sha256']==config['h_code_bundle_sha256'] and recipe['target_scores_used'] is False,'source recipe binding drift')
 formal=api['formal_tasks'](config,recipes,recipe_sha256=base.digest(recipe_path));standard=api['standard_development_tasks'](config)
 require(len(formal)==30 and config['runs']==dict(standard,**formal),'exact original20standard+30formal registration required; extra branches require separate admission')
 require(all(t['epochs']==30 for t in formal.values()),'unexpected formal epoch budget')
 registry_path=base.safe(root/'H_run_registry.json',root);registry=json.loads(registry_path.read_text())
 require(set(registry['formal_run_ids'])==set(formal) and not registry.get('effective_runs') and registry.get('target_export_allowed') is False,'already promoted or changed registry')
 qpath=base.safe(root/'H_queue_status.json',root);q=json.loads(qpath.read_text())
 require(q.get('owner_pid')==2222605 and q.get('active')==[] and q.get('stage')=='H_FORMAL_AMP_COMPLETE_INDEPENDENT_TERMINAL_AND_PAIRED_LEDGER_REVIEW_REQUIRED','original finite owner successful inactive terminal status required')
 d=json.loads(base.safe(root/'queue_status.json',root).read_text());require(d.get('active')==[] and d.get('stage')=='P3_D_CACHE_COMPLETE_ANALYSIS_PENDING','D GPU priority not released')
 audit_list=base.safe(audit_list,root);require(base.digest(audit_list)==audit_list_sha,'audit inventory SHA drift');inventory=json.loads(audit_list.read_text());require(isinstance(inventory,dict) and len(inventory)>0,'audit inventory required')
 records={};bound={str(p):base.digest(p) for p in [config_path,recipe_path,registry_path,qpath,audit_list]}
 accepted_status={'PASS_SELECTED_H_FORMAL_AMP_TERMINAL_FULL_CHECKPOINTS_SOURCE_DEV_AND_PAIRED_STREAMS','SELECTED_TERMINAL_CHECKS_PASS_AMP_PAIR_SKIP_ASYMMETRY_ALL5_SOURCE_REVIEW_REQUIRED'}
 for path,pin in inventory.items():
  p=base.safe(path,root);require(p.parent==root/'reports' and base.digest(p)==pin,'independent report SHA changed');a=json.loads(p.read_text());bound[str(p)]=pin
  require(a.get('status') in accepted_status and a.get('source_only') is True and a.get('CUDA_initialized') is False and a.get('production_processes_restarted') is False,'independent terminal audit required')
  require(a.get('inputs',{}).get('H_execution_sha256')==base.digest(config_path),'terminal audit config binding drift')
  require(set(a['selected_run_ids'])=={x['run_id'] for x in a['records']},'report selected/record inventory drift')
  for row in a['records']:
   run=row['run_id'];require(run in formal and run not in records,'duplicate/unknown terminal audit');records[run]=row
 require(set(records)==set(formal),'all30 independent terminal models required')
 names=set(api['_completion_artifacts'](30));require(len(names)==158,'sealed inventory changed')
 for run,t in formal.items():
  row=records[run];folder=root/'runs'/run;seal_path=base.safe(folder/'H_completion_seal.json',root);seal=json.loads(seal_path.read_text())
  require(base.digest(seal_path)==row['terminal_seal_sha256'] and seal.get('status')=='SEALED' and seal.get('run_id')==run,'seal identity drift')
  require(seal.get('target_used') is False and seal.get('artifact_root')==str(root) and seal.get('config_path')==str(config_path),'seal root/target identity drift')
  require(seal.get('H_data_freeze_sha256')==config['H_data_freeze_sha256'] and seal.get('h_code_bundle_sha256')==config['h_code_bundle_sha256'],'seal frozen binding drift')
  require(set(seal.get('artifact_sha256',{}))==names and row.get('seal_current_artifacts_verified')==158,'full158 seal required')
  require(row.get('snapshot_epoch')==30 and row.get('snapshot_attempts')==5730 and row.get('source_DEV_tables_independently_recomputed')==30 and row.get('source_DEV_metric_checks')==180 and row.get('target_used') is False and base.number(row.get('source_DEV_max_auc_abs_error'),'DEV error')<=1e-12,'independent full terminal metrics required')
  for name,pin in seal['artifact_sha256'].items():
   path=base.safe(folder/name,root);require(base.digest(path)==pin,'terminal artifact changed:'+run+'/'+name)
  require(base.digest(folder/'complete.json')==row['terminal_complete_sha256'] and base.digest(folder/'state.pt')==row['state_sha256'],'independent terminal SHA drift')
  complete=json.loads((folder/'complete.json').read_text());require(complete.get('status')=='TRAINING_COMPLETE' and complete.get('epochs')==30 and complete.get('attempts')==5730 and complete.get('success')==row['snapshot_successful_updates'] and complete.get('skip')==row['snapshot_AMP_skips'] and complete.get('target_used') is False and complete.get('selection')=='frozen_last','complete fields changed')
  bound[str(seal_path)]=base.digest(seal_path)
 cap=budget_check(root,budget,budget_sha);return dict(status='READ_ONLY_POST_AMP_PREFLIGHT_PASS_ORIGINAL_VALIDATORS_PENDING',formal_models=30,capacity=cap,input_sha256=bound)

def reject_new_training(free,**kwargs):
 exc=UniformFP32ReviewRequired('Frozen all5 ledger requires uniformFP32. Stop before registration/dispatch until fresh full branch capacity and independentFP32 terminal review path are qualified; no selective reruns or target promotion.')
 exc.capacity_observation=dict(live_free_bytes=free,projected_uniform_branch_bytes=kwargs.get('additional_bytes'),full_remaining_branch_admission_complete=False)
 raise exc

def run_original_tail(q,api):
 ns=dict(api);ns.update(self=q,formal={k:v for k,v in q.config.get('runs',{}).items() if v.get('phase')=='formal' and v.get('precision')=='AMP'},recipes=json.loads((q.root/'configs/H_frozen_recipes.json').read_text()).get('recipes',{}),recipe_path=q.root/'configs/H_frozen_recipes.json',registry_path=q.root/'H_run_registry.json',shutil=shutil,json=json,capacity_check=reject_new_training)
 exec(compile(scientific_tail(),str(SCIENCE/'queue_h.py'),'exec'),ns)
 return ns['effective']

def execute(budget,budget_sha,audit_list,audit_list_sha):
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only controller required')
 require(all(os.environ.get(k) in ('1','2') for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']),'bounded CPU environment required')
 admission=preflight(ROOT,budget,budget_sha,audit_list,audit_list_sha)
 sys.path.insert(0,str(SCIENCE.parent));from reliability_v2 import queue_h,data_h,qualification_h
 data_h.validate_h_freeze(ROOT);qualification_h.validate_qualification(ROOT)
 with base.original_lock()(ROOT/'H_pipeline.lock'):
  admission=preflight(ROOT,budget,budget_sha,audit_list,audit_list_sha)
  data_h.validate_h_freeze(ROOT);qualification_h.validate_qualification(ROOT)
  budget_check(ROOT,budget,budget_sha)
  q=queue_h.HQueue(ROOT,SCIENCE.parent)
  try:effective=run_original_tail(q,vars(queue_h))
  except UniformFP32ReviewRequired as exc:
   evidence=dict(status='FROZEN_UNIFORM_FP32_BRANCH_PENDING_FULL_REMAINING_CAPACITY_AND_INDEPENDENT_TERMINAL_REVIEW',reason=str(exc),capacity_observation=exc.capacity_observation,target_export_allowed=False,training_or_cache_dispatched=False,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
   queue_h.save_json(ROOT/'reports/H_uniform_FP32_admission_review_required.json',evidence)
   q.status('H_UNIFORM_FP32_SOURCE_BRANCH_ADMISSION_REVIEW_REQUIRED',evidence);raise
 return dict(status='ORIGINAL_ALL3_EFFECTIVE_REGISTRY_COMPLETE_NO_CACHE_DISPATCHED',effective_runs=effective,cache_started=False,training_started=False)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--budget',type=Path,required=True);p.add_argument('--budget-sha',required=True);p.add_argument('--audit-list',type=Path,required=True);p.add_argument('--audit-list-sha',required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
 result=execute(args.budget,args.budget_sha,args.audit_list,args.audit_list_sha) if args.execute else preflight(ROOT,args.budget,args.budget_sha,args.audit_list,args.audit_list_sha)
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
