"""Read-only numeric ledger, process ownership and phase followup. No training launch."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json,math,os,subprocess,time,shutil
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
def utc():return datetime.now(timezone.utc).isoformat()
def save(p,j):
 q=p.with_name(p.name+'.monitor.tmp');q.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def proc(pid):
 p=Path('/proc')/str(pid)
 try:
  argv=(p/'cmdline').read_bytes().split(b'\0');argv=[x.decode() for x in argv if x]
  stat=(p/'stat').read_text().rsplit(')',1)[1].split();parent=int(stat[1]);ticks=int(stat[11])+int(stat[12])
  return {'pid':int(pid),'parent_pid':parent,'argv':argv,'CPU_ticks':ticks}
 except (OSError,ValueError):return None
def logs(run):
 result=[];files=[];partial=0;out=R/'runs'/run
 for p in sorted(out.glob('epoch_*_updates.jsonl')):
  with p.open('rb') as stream:
   size=os.fstat(stream.fileno()).st_size;raw=stream.read(size)
  records=[]
  lines=raw.splitlines(keepends=True)
  for i,line in enumerate(lines):
   try:records.append(json.loads(line))
   except json.JSONDecodeError:
    assert i==len(lines)-1 and not line.endswith(b'\n');partial+=1
  result+=records;files.append({'name':p.name,'observed_bytes':size,'records':len(records)})
 assert [x['attempt'] for x in result]==list(range(len(result)))
 streak=0;maxstreak=0;skip=[]
 for n,x in enumerate(result):
  assert math.isfinite(x['loss']) and x['success']!=x['skip'] and x['scale_before']>0 and x['scale_after']>0
  if x['skip']:
   streak+=1;assert not x['gradient_finite'] and x['gradient_norm'] is None and x['scale_after']<x['scale_before']
   nxt=result[n+1] if n+1<len(result) else None
   if nxt:assert nxt['success'] and nxt['gradient_finite'] and math.isfinite(nxt['loss'])
   skip.append({'event':x,'next_success':nxt,'next_event_status':'SUCCESS' if nxt else 'PENDING_NEXT_LIVE_EVENT'})
  else:
   streak=0;assert x['gradient_finite'] and math.isfinite(x['gradient_norm'])
  maxstreak=max(maxstreak,streak)
 assert maxstreak<8
 return result,skip,maxstreak,files,partial

def cpu_sample():
    return [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
def io_sample(pid):
    return {k:int(v) for k,v in (l.split(':',1) for l in Path('/proc',str(pid),'io').read_text().splitlines())}
def digest_file(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
 started=utc();q0=json.loads((R/'queue_status.json').read_text());p0={x['pid']:proc(x['pid']) for x in q0['active']};samples=[]
 deep_launch=json.loads((R/'reports/H_production_data_inspection_launch.json').read_text());deep_pid=deep_launch['pid'];cpu0=cpu_sample();wall0=time.perf_counter()
 for i in range(8):
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
  samples.append({'utc':utc(),'csv':raw})
  if i<7:time.sleep(2)
 wall1=time.perf_counter();cpu1=cpu_sample()
 q=json.loads((R/'queue_status.json').read_text());cfg=json.loads((R/'configs/execution.json').read_text());cp=json.loads((R/'reports/checkpoint_health_20261024T0100.json').read_text());prior=json.loads((R/'reports/check_20261024T0000Z.json').read_text())
 previous={x['run_id']:x for x in prior['rows']};cps={x['run_id']:x for x in cp['checks']}
 assert len(set(x['run_id'] for x in q['active']))==len(q['active']) and 0<=len(q['active'])<=4
 assert len(set(x['gpu'] for x in q['active']))==len(q['active']) and set(x['gpu'] for x in q['active'])<={0,1,2,3}
 assert len(set(x['pid'] for x in q['active']))==len(q['active'])
 live={x['run_id']:x for x in q['active']};processes={};roots=[];loader_children=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  x=proc(p.name)
  if not x:continue
  a=x['argv']
  if '-m' in a and a[a.index('-m')+1]=='reliability_v2.trainer' and '--config' in a and a[a.index('--config')+1]==str(R/'configs/execution.json'):
   processes[x['pid']]=x
 for x in processes.values():
  if x['parent_pid'] in processes:loader_children.append({'pid':x['pid'],'parent_pid':x['parent_pid']})
  else:roots.append(x)
 assert {x['pid'] for x in roots}=={x['pid'] for x in q['active']},'unowned or duplicate trainer root'
 ownership=[];now_ticks=os.sysconf(os.sysconf_names['SC_CLK_TCK'])
 locks=Path('/proc/locks').read_text().splitlines();lockproof=[]
 def held(p,pid):
  st=p.stat();key=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
  rows=[line for line in locks if key in line.split()]
  assert len(rows)==1 and int(rows[0].split()[4])==pid,(str(p),pid,rows)
  return {'path':str(p),'expected_pid':pid,'kernel_lock':rows[0]}
 lockproof.append(held(R/'queue.lock',q['owner_pid']))
 for item in q['active']:
  x=processes[item['pid']];assert x['parent_pid']==q['owner_pid'];a=x['argv'];assert a[a.index('--run-id')+1]==item['run_id']
  env=dict(e.split(b'=',1) for e in (Path('/proc')/str(x['pid'])/'environ').read_bytes().split(b'\0') if b'=' in e)
  visible=env.get(b'CUDA_VISIBLE_DEVICES',b'').decode();assert visible==item['uuid'] and visible in cfg['allowed_gpu_uuids']
  ownership.append({**item,'parent_pid':x['parent_pid'],'CUDA_VISIBLE_DEVICES':visible,'CPU_ticks_at_end':x['CPU_ticks'],'CPU_ticks_at_start':p0.get(x['pid'],{}).get('CPU_ticks')})
  lockproof.append(held(R/'runs'/item['run_id']/'run.lock',item['pid']));lockproof.append(held(R/'gpu_locks'/f'{item["uuid"]}.lock',item['pid']))
 complete=[run for run in cfg['runs'] if (R/'runs'/run/'complete.json').exists()]
 pending=[run for run in cfg['runs'] if run not in complete and run not in live]
 rows=[]
 for run,task in cfg['runs'].items():
  out=R/'runs'/run
  if not (out/'identity.json').exists():
   rows.append({'run_id':run,'status':'NOT_STARTED','live_records':None,'live_skips':None});continue
  events,skip,maxstreak,files,partial=logs(run);old=previous.get(run)
  pinned=cps.get(run,{})
  h=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
  c=json.loads((out/'complete.json').read_text()) if (out/'complete.json').exists() else None
  if c:assert len(events)==c['attempts']==23970 and len(skip)==c['skip']
  row={'run_id':run,'status':'COMPLETE' if c else 'RUNNING','observed_utc':utc(),'gpu':live.get(run,{}).get('gpu'),
   'live_records':len(events),'live_successful_update_records':sum(x['success'] for x in events),'live_skips':len(skip),'skip_positions':[x['event']['attempt'] for x in skip],
   'skip_evidence':skip,'max_consecutive_skips':maxstreak,'previous_live_skips':(old['live_skips'] or 0) if old else 0,
   'new_live_skips_since_previous_cutoff':len(skip)-((old['live_skips'] or 0) if old else 0),
   'baseline_scope':'Prior20261024T0000live numeric receipt' if old and old.get('live_skips') is not None else 'Newly started registered run baseline0;priorNOT_STARTEDwasNA',
   'pinned_checkpoint_inspection_status':pinned.get('status'),'pinned_checkpoint_epoch':pinned.get('checkpoint_epoch'),
   'pinned_checkpoint_attempts':pinned.get('attempts',pinned.get('committed_attempts')),'pinned_checkpoint_success':pinned.get('successful_updates'),'pinned_checkpoint_skips':pinned.get('AMP_skips'),
   'current_history_epoch_metadata_only':len(h),'latest_live_event':events[-1] if events else None,'logfiles':files,'trailing_partial_lines_ignored':partial,
   'recent3epoch_mean_seconds':sum(x['seconds'] for x in h[-3:])/min(3,len(h)) if h else None}
  rows.append(row)
 cpnotactive=[x for x in cps if x not in live and x not in complete];assert not cpnotactive
 protected=[{'pid':p,'alive':Path('/proc',str(p)).exists()} for p in [1532179,1532180,1532181,1532182]];assert all(x['alive'] for x in protected)
 receipt={'utc':utc(),'started_utc':started,'status':'PASS_NO_DUPLICATE_OR_SUSTAINED_NUMERIC_FAILURE','queue_start':q0,'queue_end':q,'completed_run_ids':complete,'pending_run_ids':pending,
 'trainer_root_ownership':ownership,'dataloader_child_processes':loader_children,'kernel_lock_proofs':lockproof,'protected_processes':protected,
 'rows':rows,'live_total_skips':sum(x['live_skips'] or 0 for x in rows),'new_live_skips_since0000cutoff':sum(x.get('new_live_skips_since_previous_cutoff',0) for x in rows),
 'GPU_samples':samples,'H_progress':json.loads((R/'data/H/H_preprocessing_progress.json').read_text()),'capacity':{'workspace_free_GiB':shutil.disk_usage(R).free/1024**3,'Dataset_free_GiB':shutil.disk_usage('/Dataset').free/1024**3},
 'protocol':'Keep fixed23970attempts per30epoch run; allisolatedskips retained; no补步/retrain or configurationchanges','H_freeze_created':False,'experiment_process_restarted':False,'automation_retried':False}
 for row in rows:
     if row['status']=='NOT_STARTED':continue
     out=R/'runs'/row['run_id'];h=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
     assert [x['epoch'] for x in h]==list(range(1,len(h)+1)) and all(x['attempts']==799 and x['draw_count']==102272 and x['success']+x['skip']==799 for x in h)
     row['current_committed_history_attempts']=sum(x['attempts'] for x in h);row['current_committed_history_success']=sum(x['success'] for x in h);row['current_committed_history_skips']=sum(x['skip'] for x in h)
     assert row['current_committed_history_attempts']<=row['live_records'] and row['current_committed_history_skips']<=row['live_skips']
     row['latest_two_committed_array_hashes']=[]
     for x in h[-2:]:
         for field,suffix in [('sample_sha256','samples'),('augmentation_sha256','augmentations')]:
             p=out/f'epoch_{x["epoch"]:02d}_{suffix}.npy';assert digest_file(p)==x[field];row['latest_two_committed_array_hashes'].append({'epoch':x['epoch'],'kind':suffix,'sha256':x[field]})
     sec=row['recent3epoch_mean_seconds'];row['draw_normalized_inputs_per_second']=102272/sec if sec else None
 full=json.loads((R/'data/H/H_full_candidate_audit_receipt.json').read_text());launch=json.loads((R/'reports/H_fresh_full_audit_launch_1450.json').read_text());integrity=json.loads((R/'data/H/H_artifact_integrity_receipt.json').read_text())
 assert launch['returncode']==0 and launch['status']=='FRESH_FULL_AUDIT_COMMAND_COMPLETE' and full['artifact_integrity_status']=='PASSED' and full['candidate_videos']==full['processed_manifest_count']==8431
 assert full['status']=='AUDIT_COMPLETE_READY_FOR_H_DATA_FREEZE_REVIEW' and integrity['status']=='PASSED' and integrity['source_video_byte_hashes_verified']==8431
 assert full['fresh_code_sha256']==launch['geometry_code_sha256']==integrity['fresh_code_sha256']
 for key,name in [('candidate_manifest_sha256','H_candidates_all.json'),('processed_paths_sha256','H_processed_manifest_paths.json'),('content_source_audit_sha256','H_content_source_audit.json'),('freeze_candidate_sha256','H_data_freeze_candidate.json'),('artifact_integrity_receipt_sha256','H_artifact_integrity_receipt.json')]:assert digest_file(R/'data/H'/name)==full[key]
 assert len(full['tables'])==13
 for key,x in full['tables'].items():assert digest_file(x['path'])==x['sha256']
 assert deep_launch['status']=='READY_FOR_INDEPENDENT_REVIEW' and not proc(deep_pid)
 deep_report=json.loads((R/'reports/H_production_data_inspection.json').read_text());assert deep_report['status']=='READY_FOR_INDEPENDENT_REVIEW' and deep_report['blockers']==[] and not deep_report['target_performance_used']
 assert digest_file(R/'reports/H_production_data_inspection.json')==deep_launch['inspection_report_sha256']
 for item in [*deep_report['identities'].values(),*deep_report['tables'].values()]:assert digest_file(item['path'])==item['sha256']
 assert (R/'data/H/freeze.json').exists() and not (R/'qualification/H_gate.json').exists()
 frozen_validation=json.loads((R/'reports/H_freeze_receipt_validation_1930.json').read_text());assert frozen_validation['status']=='PASS_FROZEN_RECEIPT_AND_OSRC0_SHALLOW_CURRENT_BINDINGS' and frozen_validation['OS_returncode']==0
 assert digest_file(R/'data/H/freeze.json')==frozen_validation['freeze_sha256']
 receipt['H_freeze_binding_validation']=frozen_validation
 receipt['H_freeze_created']=True
 d=[b-a for a,b in zip(cpu0,cpu1)];total=sum(d)
 receipt.update(status='PASS_D_RUNTIME_H_DATA_FROZEN_CURRENT_BINDINGS_GPU_QUALIFICATION_PENDING',nominal_window='2026-10-24T01:00:00Z',
     CPU_busy_percent=100*(total-d[3]-d[4])/total,CPU_iowait_percent=100*d[4]/total,retired_deep_process_CPU_cores_over_sample=None,retired_deep_process_IO_delta=None,
     D_complete=len(complete),D_active=len(q['active']),D_pending=len(pending),current_committed_history_attempts=sum(x.get('current_committed_history_attempts',0) for x in rows),current_committed_history_success=sum(x.get('current_committed_history_success',0) for x in rows),current_committed_history_skips=sum(x.get('current_committed_history_skips',0) for x in rows),
     current_full_tensor_inspection_performed=True,current_full_tensor_receipt='reports/checkpoint_health_20261024T0100.json',H_fresh_audit_complete=True,H_fresh_audit_passed=True,H_fresh_audit=full,H_artifact_integrity=integrity,H_fresh_audit_completion_utc=launch['ended_utc'],H_fresh_audit_returncode=0,
     H_deep_inspection={'pid':deep_pid,'launch_utc':deep_launch['utc'],'status':deep_launch['status'],'native_session_id_retired':95551,'ended_utc':deep_launch['ended_utc'],'elapsed_seconds':(datetime.fromisoformat(deep_launch['ended_utc'])-datetime.fromisoformat(deep_launch['utc'])).total_seconds(),'progress_percentage':None,'progress_scope':'Deep inspection completed and current report/table/code bindings verified. Independent terminal review and freeze are separate subsequent stages.','does_not_freeze_or_train':True},
     H_data_frozen=True,H_GPU_qualification_or_training_started=False,next_recommended_check_utc='2026-10-24T01:20:00Z',scientific_results_complete=False,journal_manuscripts_complete=False)
 receipt.update(D_development_complete=sum(cfg['runs'][r]['phase']=='development' for r in complete),D_formal_complete=sum(cfg['runs'][r]['phase']=='formal' for r in complete),D_formal_active=sum(cfg['runs'][r]['phase']=='formal' for r in live),D_formal_pending=sum(cfg['runs'][r]['phase']=='formal' for r in pending),registered_D_runs=len(cfg['runs']),current_full_tensor_checkpoint_count=sum(x['status']=='PASS' for x in cp['checks']),first_epoch_recovery_pending_run_ids=[x['run_id'] for x in cp['checks'] if x['status']=='PENDING_FIRST_EPOCH_RECOVERY_POINT'],D_phase=q['stage'])
 save(R/'reports/check_20261024T0100Z.json',receipt)
 print(json.dumps({k:v for k,v in receipt.items() if k not in ['rows','dataloader_child_processes','GPU_samples','queue_start','queue_end']},ensure_ascii=False),flush=True)
 print('ROWS',json.dumps([{k:x[k] for k in ['run_id','status','live_records','live_skips','new_live_skips_since_previous_cutoff','pinned_checkpoint_epoch'] if k in x} for x in rows],ensure_ascii=False))
if __name__=='__main__':main()
