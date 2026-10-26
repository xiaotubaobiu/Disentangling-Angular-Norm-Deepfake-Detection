"""Finite first4 H source qualification attempts using the original HQueue/worker.
No automatic OOM retry or development dispatch before source/capacity review.
"""
from pathlib import Path
import json,datetime,traceback
from reliability_v2 import queue_h,qualification_h,data_h
from reliability_v2.trainer import exclusive_file,save_json
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
V=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
with exclusive_file(R/'H_pipeline.lock'):
 data_h.validate_h_freeze(R)
 qualification_h.prepare_probe(R)
 owner=queue_h.HQueue(R,V)
 specs=[('clip_vit_b16','AMP',32),('clip_vit_b16','FP32',32),('dinov2_vit_b14','AMP',32),('dinov2_vit_b14','FP32',32)]
 for b,precision,micro in specs:
  planned=qualification_h.next_attempt(R,b,precision)
  assert planned['status']!='COMPLETE' and planned['microbatch']==micro
 devices,boundary=owner._devices()
 assert boundary['ready'] and len(devices)==4
 try:
  for (b,precision,micro),(gpu,uuid) in zip(specs,devices):
   key=f'H_qual_{b}_{precision}_micro{micro}'
   owner._launch(key,gpu,uuid,qualification=(b,precision,micro))
  owner.status('H_SOURCE_QUALIFICATION_FIRST4_FROZEN_ATTEMPTS',dict(pending_after_this_batch=[['efficientnet_b0','AMP',32],['efficientnet_b0','FP32',32]],conditional_retry_policy='Pause before conditional retry for full remaining-capacity review; preserve every failed attempt.'))
  results=[]
  for key,item in list(owner.jobs.items()):
   rc=item['process'].wait();spec=next(s for s in specs if key==f'H_qual_{s[0]}_{s[1]}_micro{s[2]}');b,precision,micro=spec;folder=R/f'qualification/H/{b}/{precision}/micro{micro}'
   result=dict(run_id=key,pid=item['process'].pid,returncode=rc,receipt_exists=(folder/'receipt.json').exists(),failure=json.loads((folder/'failure.json').read_text()) if (folder/'failure.json').exists() else None)
   results.append(result);del owner.jobs[key]
   if rc==0 and result['receipt_exists']:owner.completed.append(key)
   owner.status('H_SOURCE_QUALIFICATION_FIRST4_RUNNING',dict(results=results,pending_after_this_batch=[['efficientnet_b0','AMP',32],['efficientnet_b0','FP32',32]],no_automatic_retry=True))
  passed=all(r['returncode']==0 and r['receipt_exists'] for r in results)
  terminal=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='FIRST4_SOURCE_QUALIFICATION_RECEIPTS_READY_FRESH_ACCEPTANCE_PENDING' if passed else 'SOURCE_QUALIFICATION_REVIEW_REQUIRED',results=results,remaining_standard_specs=[['efficientnet_b0','AMP',32],['efficientnet_b0','FP32',32]],H_gate_created=False,source_only=True,target_used=False,automatic_development_started=False,conditional_retry_started=False)
  save_json(R/'reports/H_first4_source_qualification_terminal_20261024T1510.json',terminal)
  owner.status(terminal['status'],terminal)
  print(json.dumps(terminal,ensure_ascii=False),flush=True)
 except BaseException:
  owner.status('H_FIRST4_COMMAND_REVIEW_REQUIRED',dict(traceback=traceback.format_exc(),existing_workers_continue_until_their_own_completion=True));raise
