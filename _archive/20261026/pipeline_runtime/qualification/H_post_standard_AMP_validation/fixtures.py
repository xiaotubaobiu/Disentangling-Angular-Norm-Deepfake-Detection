"""Synthetic admission evidence; no models or real jobs."""
from pathlib import Path
import hashlib,json,datetime
G=1024**3
NAMES=['H_first4_terminal_acceptance_20261024T1630.json','H_DINO_lr1_terminal_acceptance_20261024T1800.json','H_DINO_lr3_terminal_acceptance_20261024T1925.json','H_DINO_lr5_terminal_acceptance_20261024T2010.json','H_EfficientNet_terminal_acceptance_20261024T2045.json']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x)+'\n')
def make(root,module):
 source=module.SCIENCE
 api=module.pure_api(source);config=dict(schema='H_EXECUTION_V2',artifact_root=str(root),allowed_gpus=[0,1,2,3],allowed_gpu_uuids=['g0','g1','g2','g3'],H_microbatches={b:{'AMP':32,'FP32':32} for b in api['BACKBONES']},H_training_draw_count=24448,H_training_rows=24424,H_data_freeze_sha256='df',h_code_bundle_sha256='hc',runs={})
 tasks=api['standard_development_tasks'](config);config['runs']=tasks;js(root/'configs/H_execution.json',config)
 groups=[[],[],[],[],[]]
 inventory=module.completion_inventory(source,30)
 for run,t in tasks.items():
  out=root/'runs'/run;out.mkdir(parents=True)
  for name in inventory:(out/name).write_text('synthetic '+run+' '+name)
  auc=.90 if t['lr']==1e-5 else .80 if t['lr']==3e-5 else .70
  history=[dict(epoch=i,dev={'macro_auc':auc}) for i in range(1,31)]
  js(out/'history.json',history);js(out/'complete.json',dict(status='TRAINING_COMPLETE',epochs=30,attempts=5730,success=5730,skip=0,target_used=False,selection='frozen_last'))
  seal=dict(schema='H_COMPLETION_SEAL_V1',status='SEALED',run_id=run,artifact_root=str(root),config_path=str(root/'configs/H_execution.json'),H_data_freeze_sha256='df',h_code_bundle_sha256='hc',target_used=False,artifact_sha256={n:sha(out/n) for n in inventory});js(out/'H_completion_seal.json',seal)
  row=dict(run_id=run,status='PASS_COMPLETE_ATOMIC_CHECKPOINT_CPU_SNAPSHOT',snapshot_epoch=30,snapshot_attempts=5730,snapshot_successful_updates=5730,snapshot_AMP_skips=0,state_sha256=sha(out/'state.pt'),terminal_complete_sha256=sha(out/'complete.json'),terminal_seal_sha256=sha(out/'H_completion_seal.json'),seal_current_artifacts_verified=len(inventory),source_DEV_tables_independently_recomputed=30,source_DEV_max_auc_abs_error=0.,target_used=False)
  index=0 if t['backbone']=='clip_vit_b16' else 4 if t['backbone']=='efficientnet_b0' else {1e-5:1,3e-5:2,5e-5:3}[t['lr']];groups[index].append(row)
 pins={}
 for name,records in zip(NAMES,groups):
  p=root/'reports'/name;js(p,dict(status='PASS_SYNTHETIC_TERMINAL',source_only=True,CUDA_initialized=False,production_processes_restarted=False,records=records));pins[name]=sha(p)
 ids=list(tasks);js(root/'reports/H_source_and_standard_development_progress_20261024T1510.json',dict(pid=1993186,status='H20_STANDARD_DEVELOPMENT_COMPLETE_REVIEW_BEFORE_CONDITIONALS',run_ids=ids))
 js(root/'H_queue_status.json',dict(owner_pid=1993186,stage='H_STANDARD_DEVELOPMENT_COMPLETE_CAPACITY_REVIEW_BEFORE_CONDITIONALS',active=[]))
 js(root/'queue_status.json',dict(stage='P3_D_CACHE_COMPLETE_ANALYSIS_PENDING',active=[]))
 (root/'proc').mkdir()
 base=root/'reports/base_capacity.json';js(base,{'synthetic':True})
 components=dict(D_remaining_analysis=0,D_remaining_features=0,H_training=20*G,H_qualification=0,H_features=10*G,H_pair_analysis=10*G,future_logs_manifests_figures_audits=5*G)
 stamp=datetime.datetime.now(datetime.timezone.utc)
 budget=dict(schema='FULL_REMAINING_PRIMARY_CAPACITY_V2',status='REGISTERED_PRIMARY_REMAINING_COVERED',utc=stamp.isoformat(),base_live_path=str(base),base_live_sha256=sha(base),D_individual_complete_count=15,D_remaining_analysis=0,H_current_persistent_run_bytes=10*G,H_current_temporary_run_bytes=0,H_persistent_training_budget_total_bytes=30*G,remaining_components=components,normal_remaining_write_bytes=sum(components.values()),temporary_bytes=16*G,reserve_bytes=50*G,normal_required_bytes=111*G,source_best_active_budget=None,compact_unvalidated_storage_savings_counted=False,SSD_cold_release_bytes=0,remaining_conditionals={'H8_DINO_extension_and_longer_formal_aux':0,'H30_uniform_FP32_fallback':40*G})
 path=root/'reports/budget.json';js(path,budget)
 return dict(root=root,budget_path=path,budget_sha=sha(path),audit_pins=pins,source=source,proc_root=root/'proc',now=stamp,free_bytes=200*G)
def edit_budget(f,update):
 a=json.loads(f['budget_path'].read_text());update(a);js(f['budget_path'],a);f['budget_sha']=sha(f['budget_path'])
def rebind_run(f,run):
 root=f['root'];out=root/'runs'/run;sp=out/'H_completion_seal.json';seal=json.loads(sp.read_text());seal['artifact_sha256']={n:sha(out/n) for n in seal['artifact_sha256']};js(sp,seal)
 for name in NAMES:
  p=root/'reports'/name;a=json.loads(p.read_text())
  for row in a['records']:
   if row['run_id']==run:row.update(state_sha256=sha(out/'state.pt'),terminal_complete_sha256=sha(out/'complete.json'),terminal_seal_sha256=sha(sp))
  js(p,a);f['audit_pins'][name]=sha(p)
