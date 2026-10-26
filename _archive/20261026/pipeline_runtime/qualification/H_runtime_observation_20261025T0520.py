from pathlib import Path
import datetime,json,subprocess,time,statistics
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
q0=json.loads((R/'H_queue_status.json').read_text());samples=[];stream=R/'reports/GPU_samples_20261025T0520.jsonl'
with stream.open('x') as f:
 for i in range(21):
  raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True);row=dict(utc=utc(),gpu=[list(map(int,x.split(', '))) for x in raw.splitlines()]);samples.append(row);f.write(json.dumps(row)+'\n');f.flush()
  if i<20:time.sleep(2)
q1=json.loads((R/'H_queue_status.json').read_text());active=[]
for item in q1['active']:
 a=dict(item);p=R/'runs'/a['run_id'];cp=Path('/proc')/str(a['pid'])/'cmdline'
 try:a['process_alive_and_matches']=a['run_id'] in cp.read_bytes().replace(b'\0',b' ').decode()
 except (FileNotFoundError,ProcessLookupError):a['process_alive_and_matches']=False;a['process_observation']='ended_during_read'
 hp=p/'history.json';hist=json.loads(hp.read_text()) if hp.exists() else []
 a.update(history_epochs=len(hist),history_attempts=sum(x['attempts'] for x in hist),history_success=sum(x['success'] for x in hist),history_skips=sum(x['skip'] for x in hist),recent5_draw_equivalent_per_second=24448/statistics.mean(x['seconds'] for x in hist[-5:]) if hist else None)
 events=[];partial=0
 for path in sorted(p.glob('epoch_*_updates.jsonl')):
  for line in path.read_text().splitlines():
   try:events.append(json.loads(line))
   except json.JSONDecodeError:partial+=1
 a.update(live_complete_update_events=len(events),live_skip_events=sum(bool(x['skip']) for x in events),partial_inflight_json_lines=partial);active.append(a)
ended=[dict(x,observation='not_in_end_queue') for x in q0['active'] if x['run_id'] not in {a['run_id'] for a in q1['active']}]
out=dict(utc=utc(),status='READONLY_GPU_WINDOW_AND_END_QUEUE_METADATA',window_start=samples[0]['utc'],window_end=samples[-1]['utc'],GPU_samples=samples,GPU0_3_mean_util_percent=[statistics.mean(s['gpu'][i][1] for s in samples) for i in range(4)],queue_start=q0,queue_end=q1,active_observation=active,ended_or_replaced_during_window=ended,scope='21samples2sec spacing. Phase window only, not full-run average. End-queue workers and ended start workers are distinct; no restart/intervention. GPU sample JSONL is flushed incrementally.')
p=R/'reports/runtime_health_20261025T0520.json'
with p.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2)
print('COMPLETE',p,flush=True)
