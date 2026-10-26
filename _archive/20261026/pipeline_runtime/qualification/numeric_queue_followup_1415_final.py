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

def main():
 started=utc();q0=json.loads((R/'queue_status.json').read_text());p0={x['pid']:proc(x['pid']) for x in q0['active']};samples=[]
 for i in range(8):
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
  samples.append({'utc':utc(),'csv':raw})
  if i<7:time.sleep(2)
 q=json.loads((R/'queue_status.json').read_text());cfg=json.loads((R/'configs/execution.json').read_text());cp=json.loads((R/'reports/checkpoint_health_1415.json').read_text());prior=json.loads((R/'reports/D_numeric_queue_followup_1315.json').read_text())
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
   'baseline_scope':'Prior1315live numeric receipt' if old and old.get('live_skips') is not None else 'Newly started registered run baseline0;priorNOT_STARTEDwasNA',
   'pinned_checkpoint_inspection_status':pinned.get('status'),'pinned_checkpoint_epoch':pinned.get('checkpoint_epoch'),
   'pinned_checkpoint_attempts':pinned.get('attempts',pinned.get('committed_attempts')),'pinned_checkpoint_success':pinned.get('successful_updates'),'pinned_checkpoint_skips':pinned.get('AMP_skips'),
   'current_history_epoch_metadata_only':len(h),'latest_live_event':events[-1] if events else None,'logfiles':files,'trailing_partial_lines_ignored':partial,
   'recent3epoch_mean_seconds':sum(x['seconds'] for x in h[-3:])/min(3,len(h)) if h else None}
  rows.append(row)
 cpnotactive=[x for x in cps if x not in live and x not in complete];assert not cpnotactive
 protected=[{'pid':p,'alive':Path('/proc',str(p)).exists()} for p in [1532179,1532180,1532181,1532182]];assert all(x['alive'] for x in protected)
 receipt={'utc':utc(),'started_utc':started,'status':'PASS_NO_DUPLICATE_OR_SUSTAINED_NUMERIC_FAILURE','queue_start':q0,'queue_end':q,'completed_run_ids':complete,'pending_run_ids':pending,
 'trainer_root_ownership':ownership,'dataloader_child_processes':loader_children,'kernel_lock_proofs':lockproof,'protected_processes':protected,
 'rows':rows,'live_total_skips':sum(x['live_skips'] or 0 for x in rows),'new_live_skips_since1315cutoff':sum(x.get('new_live_skips_since_previous_cutoff',0) for x in rows),
 'GPU_samples':samples,'H_progress':json.loads((R/'data/H/H_preprocessing_progress.json').read_text()),'capacity':{'workspace_free_GiB':shutil.disk_usage(R).free/1024**3,'Dataset_free_GiB':shutil.disk_usage('/Dataset').free/1024**3},
 'protocol':'Keep fixed23970attempts per30epoch run; allisolatedskips retained; no补步/retrain or configurationchanges','H_freeze_created':False,'experiment_process_restarted':False,'automation_retried':False}
 save(R/'reports/D_numeric_queue_followup_1415_final.json',receipt)
 print(json.dumps({k:v for k,v in receipt.items() if k not in ['rows','dataloader_child_processes','GPU_samples','queue_start','queue_end']},ensure_ascii=False),flush=True)
 print('ROWS',json.dumps([{k:x[k] for k in ['run_id','status','live_records','live_skips','new_live_skips_since_previous_cutoff','pinned_checkpoint_epoch'] if k in x} for x in rows],ensure_ascii=False))
if __name__=='__main__':main()
