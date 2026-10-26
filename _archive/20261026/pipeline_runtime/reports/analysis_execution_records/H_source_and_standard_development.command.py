"""Finite original HQueue qualification +20 standard development, no conditional branch."""
from pathlib import Path
import json,os,datetime,shutil,traceback
from reliability_v2 import queue_h,data_h
from reliability_v2.trainer import exclusive_file,save_json
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
V=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
PROGRESS=R/'reports/H_source_and_standard_development_progress_20261024T1510.json'
def status(name,**kw):save_json(PROGRESS,dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),status=name,**kw))
with exclusive_file(R/'H_pipeline.lock'):
 terminal=json.loads((R/'reports/H_first4_source_qualification_terminal_20261024T1510.json').read_text())
 assert terminal['status']=='FIRST4_SOURCE_QUALIFICATION_RECEIPTS_READY_FRESH_ACCEPTANCE_PENDING' and all(r['returncode']==0 for r in terminal['results'])
 budget=json.loads((R/'reports/full_capacity_projection_20261024T1430.json').read_text())
 base=budget['normal_total_write_bytes']-budget['normal_remaining_write_bytes']['D_remaining_features']
 written_D=sum(sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) for folder in (R/'analysis/D').iterdir() if folder.is_dir() and folder.name!='D_formal_clip_vit_b16_seed7')
 written_Hqual=sum(p.stat().st_size for p in (R/'qualification/H').rglob('*') if p.is_file())
 remaining=max(0,base-written_D-written_Hqual)
 eff=budget['D_observed_run_size_maxima']['efficientnet_b0']['state_bytes']
 extra_eff_oom=2*12*(eff*1.02+16*1024**2)
 required=remaining+budget['temporary_peak_additional_bytes']+budget['free_reserve_bytes']+extra_eff_oom
 capacity=dict(normal_remaining_bytes=remaining,remaining_Eff_two_extra_OOM_attempt_set_bytes=extra_eff_oom,temporary_bytes=budget['temporary_peak_additional_bytes'],reserve_bytes=budget['free_reserve_bytes'],required_bytes=required,free_bytes=shutil.disk_usage(R).free,large_backbone_qualification_retries_remaining=0,before_conditional_development_or_formal_stop=True)
 save_json(R/'reports/H_source_and_standard_development_capacity_20261024T1510.json',capacity)
 if capacity['free_bytes']<=required:status('BLOCKED_CAPACITY_BEFORE_REMAINING_SOURCE_ATTEMPTS',capacity=capacity);raise RuntimeError('Updated remaining matrix + all possible Eff retries + temporary +reserve does not fit; cold migration first')
 status('ORIGINAL_HQUEUE_FRESH_QUALIFICATION_ACCEPTANCE_AND_REMAINING_EFF',capacity=capacity)
 data_h.validate_h_freeze(R)
 owner=queue_h.HQueue(R,V)
 try:
  owner.qualification()
  status('H_FULL_SOURCE_QUALIFICATION_ACCEPTED_REGISTERING_STANDARD_DEVELOPMENT')
  owner.config=queue_h.prepare_execution(R)
  tasks=queue_h.standard_development_tasks(owner.config);assert len(tasks)==20
  owner._register(tasks)
  status('ORIGINAL_HQUEUE_20_STANDARD_DEVELOPMENT_RUNNING',run_ids=list(tasks),automatic_conditional_extensions=False,automatic_formal_training=False)
  owner.batch(list(tasks),'H_STANDARD_DEVELOPMENT')
  owner.status('H_STANDARD_DEVELOPMENT_COMPLETE_CAPACITY_REVIEW_BEFORE_CONDITIONALS',dict(run_ids=list(tasks),next='Refresh capacity/cold archive, then original source-only recipe selection and prescribed conditional extensions/formals.'))
  status('H20_STANDARD_DEVELOPMENT_COMPLETE_REVIEW_BEFORE_CONDITIONALS',run_ids=list(tasks))
 except BaseException:
  owner.status('H_SOURCE_OR_STANDARD_DEVELOPMENT_REVIEW_REQUIRED',dict(traceback=traceback.format_exc(),existing_workers_continue_until_own_completion=True));status('REVIEW_REQUIRED',traceback=traceback.format_exc());raise
