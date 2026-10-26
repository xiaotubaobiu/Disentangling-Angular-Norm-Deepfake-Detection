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
    return {k:int(v) for k,v in (line.split(':',1) for line in Path('/proc',str(pid),'io').read_text().splitlines())}
def digest_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
 started=utc();q0=json.loads((R/'queue_status.json').read_text());p0={x['pid']:proc(x['pid']) for x in q0['active']};samples=[]
 cpu0=cpu_sample();audit0=proc(152000);io0=io_sample(152000);wall0=time.perf_counter()
 for i in range(8):
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
  samples.append({'utc':utc(),'csv':raw})
  if i<7:time.sleep(2)
 wall1=time.perf_counter();cpu1=cpu_sample();audit1=proc(152000);io1=io_sample(152000)
 q=json.loads((R/'queue_status.json').read_text());cfg=json.loads((R/'configs/execution.json').read_text());cp=json.loads((R/'reports/checkpoint_health_1450.json').read_text());prior=json.loads((R/'reports/D_numeric_queue_followup_1450.json').read_text())
 previous={x['run_id']:x for x in prior['rows']};cps={x['run_id']:x for x in cp['checks']}
 assert len(set(x['run_id'] for x in q['active']))==len(q['active'])==4
 assert len(set(x['gpu'] for x in q['active']))==4 and set(x['gpu'] for x in q['active'])=={0,1,2,3}
 assert len(set(x['pid'] for x in q['active']))==4
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
   'baseline_scope':'Prior1450live numeric receipt' if old and old.get('live_skips') is not None else 'Newly started registered run baseline0;priorNOT_STARTEDwasNA',
   'pinned_checkpoint_inspection_status':pinned.get('status'),'pinned_checkpoint_epoch':pinned.get('checkpoint_epoch'),
   'pinned_checkpoint_attempts':pinned.get('attempts',pinned.get('committed_attempts')),'pinned_checkpoint_success':pinned.get('successful_updates'),'pinned_checkpoint_skips':pinned.get('AMP_skips'),
   'current_history_epoch_metadata_only':len(h),'latest_live_event':events[-1] if events else None,'logfiles':files,'trailing_partial_lines_ignored':partial,
   'recent3epoch_mean_seconds':sum(x['seconds'] for x in h[-3:])/min(3,len(h)) if h else None}
  rows.append(row)
 cpnotactive=[x for x in cps if x not in live and x not in complete];assert not cpnotactive
 protected=[{'pid':p,'alive':Path('/proc',str(p)).exists()} for p in [1532179,1532180,1532181,1532182]];assert all(x['alive'] for x in protected)
 receipt={'utc':utc(),'started_utc':started,'status':'PASS_NO_DUPLICATE_OR_SUSTAINED_NUMERIC_FAILURE','queue_start':q0,'queue_end':q,'completed_run_ids':complete,'pending_run_ids':pending,
 'trainer_root_ownership':ownership,'dataloader_child_processes':loader_children,'kernel_lock_proofs':lockproof,'protected_processes':protected,
 'rows':rows,'live_total_skips':sum(x['live_skips'] or 0 for x in rows),'new_live_skips_since1450cutoff':sum(x.get('new_live_skips_since_previous_cutoff',0) for x in rows),
 'GPU_samples':samples,'H_progress':json.loads((R/'data/H/H_preprocessing_progress.json').read_text()),'capacity':{'workspace_free_GiB':shutil.disk_usage(R).free/1024**3,'Dataset_free_GiB':shutil.disk_usage('/Dataset').free/1024**3},
 'protocol':'Keep fixed23970attempts per30epoch run; allisolatedskips retained; no补步/retrain or configurationchanges','H_freeze_created':False,'experiment_process_restarted':False,'automation_retried':False}
 for row in rows:
     if row['status']=='NOT_STARTED':continue
     out=R/'runs'/row['run_id'];h=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
     assert [x['epoch'] for x in h]==list(range(1,len(h)+1))
     assert all(x['attempts']==799 and x['draw_count']==102272 and x['success']+x['skip']==799 for x in h)
     row['current_committed_history_attempts']=sum(x['attempts'] for x in h)
     row['current_committed_history_success']=sum(x['success'] for x in h)
     row['current_committed_history_skips']=sum(x['skip'] for x in h)
     assert row['current_committed_history_attempts']<=row['live_records']
     assert row['current_committed_history_skips']<=row['live_skips']
     row['latest_two_committed_array_hashes']=[]
     for x in h[-2:]:
         for field,suffix in [('sample_sha256','samples'),('augmentation_sha256','augmentations')]:
             p=out/f'epoch_{x["epoch"]:02d}_{suffix}.npy';assert digest_file(p)==x[field]
             row['latest_two_committed_array_hashes'].append({'epoch':x['epoch'],'kind':suffix,'sha256':x[field]})
     row['state_file_mtime_utc']=datetime.fromtimestamp((out/'state.pt').stat().st_mtime,timezone.utc).isoformat()
     row['history_inspection_scope']='Committed history/log/array integrity only; latest full model/optimizer/RNG proof remains checkpoint_health_1450'
     sec=row['recent3epoch_mean_seconds'];row['draw_normalized_inputs_per_second']=102272/sec if sec else None
 log=R/'logs/H_fresh_full_audit_1450.log';logtext=log.read_text();snapshot=R/'reports/H_fresh_full_audit_log_snapshot_1515.jsonl';snapshot.write_text(logtext)
 progress=[json.loads(x) for x in logtext.splitlines() if x.startswith('{')][-1]
 earlier=json.loads((R/'reports/check_20261023T1450Z.json').read_text())['H_fresh_audit']
 t0=datetime.fromisoformat(earlier['utc']);t1=datetime.now(timezone.utc);delta_seconds=(t1-t0).total_seconds()
 added=progress['verified_videos']-earlier['latest_log_progress']['verified_videos'];rate=added/delta_seconds if delta_seconds>0 and added>0 else None
 initial_launch=json.loads((R/'reports/H_fresh_full_audit_launch_1450.json').read_text());state=json.loads((R/'data/H/H_full_candidate_audit_receipt.json').read_text())
 assert initial_launch['status']=='FRESH_FULL_AUDIT_RUNNING' and state['status']=='VALIDATING_ARTIFACTS'
 assert audit0 and audit1 and audit0['pid']==audit1['pid']==152000
 d=[b-a for a,b in zip(cpu0,cpu1)];total=sum(d)
 receipt.update(status='PASS_INCREMENTAL_D_RUNTIME_H_FRESH_AUDIT_RUNNING',nominal_window='2026-10-23T15:15:00Z',
     CPU_busy_percent=100*(total-d[3]-d[4])/total,CPU_iowait_percent=100*d[4]/total,
     audit_process_CPU_cores_over_sample=(audit1['CPU_ticks']-audit0['CPU_ticks'])/now_ticks/(wall1-wall0),audit_process_IO_delta={k:io1[k]-io0[k] for k in io0},
     H_audit={'pid':152000,'launcher_pid':initial_launch['launcher_pid'],'status':'VALIDATING_ARTIFACTS','latest_progress':progress,'observed_utc':t1.isoformat(),'baseline_progress':earlier['latest_log_progress'],'baseline_utc':earlier['utc'],'added_verified_videos':added,'interval_seconds':delta_seconds,'rolling_videos_per_minute':60*rate if rate else None,'rolling_remaining_seconds':(8431-progress['verified_videos'])/rate if rate else None,'immutable_log_snapshot':str(snapshot),'immutable_log_snapshot_sha256':digest_file(snapshot),'ETA_scope':'Rolling video-count estimate only; heterogeneous source/PNG sizes may change rate; excludes finalgraph/table work and mandatorylaterdeepinspection/freezevalidation'},
     current_committed_history_attempts=sum(x.get('current_committed_history_attempts',0) for x in rows),current_committed_history_success=sum(x.get('current_committed_history_success',0) for x in rows),current_committed_history_skips=sum(x.get('current_committed_history_skips',0) for x in rows),
     D_complete=len(complete),D_active=len(q['active']),D_pending=len(pending),current_full_tensor_inspection_performed=False,prior_full_tensor_receipt='reports/checkpoint_health_1450.json',
     H_fresh_audit_complete=False,H_data_frozen=False,H_GPU_qualification_or_training_started=False,next_recommended_check_utc='2026-10-23T16:00:00Z',
     resource_dependency='CurrentfourDINOworkers own allGPU0-3. Existingqueue willassign pendingEfficientNetwhenDslotsfree. H waitsdatafreeze plusDterminalGPU/cachepriority;nohandoffnow;GPU4-7protected.',scientific_results_complete=False,journal_manuscripts_complete=False)
 save(R/'reports/check_20261023T1515Z.json',receipt)
 print(json.dumps({k:v for k,v in receipt.items() if k not in ['rows','dataloader_child_processes','GPU_samples','queue_start','queue_end']},ensure_ascii=False),flush=True)
 print('ROWS',json.dumps([{k:x[k] for k in ['run_id','status','live_records','live_skips','new_live_skips_since_previous_cutoff','pinned_checkpoint_epoch'] if k in x} for x in rows],ensure_ascii=False))
if __name__=='__main__':main()
