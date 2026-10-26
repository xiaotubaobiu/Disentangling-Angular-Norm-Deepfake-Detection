import pathlib,json,time,datetime,subprocess,statistics,os
R=pathlib.Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
start=datetime.datetime.now(datetime.timezone.utc).isoformat();q=json.loads((R/'H_queue_status.json').read_text());samples=[]
for index in range(61):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
 samples.append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':[list(map(int,line.split(', '))) for line in raw.splitlines()]})
 if index<60:time.sleep(2)
active=[]
for a in q['active']:
 p=R/'runs'/a['run_id'];hist=json.loads((p/'history.json').read_text());events=[];partial=0
 for f in sorted(p.glob('epoch_*_updates.jsonl')):
  for line in f.read_text().splitlines():
   try:events.append(json.loads(line))
   except json.JSONDecodeError:partial+=1
 active.append(dict(a,history_epochs=len(hist),history_attempts=sum(x['attempts'] for x in hist),history_success=sum(x['success'] for x in hist),history_skips=sum(x['skip'] for x in hist),live_complete_update_events=len(events),live_skip_events=sum(bool(e['skip']) for e in events),partial_inflight_json_lines=partial,recent5_draw_equivalent_per_second=24448/statistics.mean(x['seconds'] for x in hist[-5:]),actual_cmdline=pathlib.Path('/proc')/str(a['pid'])/'cmdline'))
for a in active:
 p=a['actual_cmdline'];a['actual_cmdline']=p.read_bytes().replace(bytes([0]),b' ').decode();assert a['run_id'] in a['actual_cmdline']
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'window_start':start,'status':'ORIGINAL_H_QUEUE_READONLY_RUNTIME_OBSERVATION','queue_start':q,'queue_end':json.loads((R/'H_queue_status.json').read_text()),'GPU_samples':samples,'GPU0_3_mean_util_percent':[statistics.mean(s['gpu'][i][1] for s in samples) for i in range(4)],'active_observation':active,'scope':'61 samples separated by2 seconds, about130 seconds including nvidia-smi calls, slightly shorter than a DINO training epoch. Phase window only. epoch.seconds includes training/sourceDEV/logging, excludes subsequent checkpoint save. No GPU intervention.'}
report=R/'reports/runtime_health_20261025T0445.json';assert not report.exists();report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('COMPLETE',report,flush=True)
