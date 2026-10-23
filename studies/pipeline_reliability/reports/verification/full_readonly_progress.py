from pathlib import Path
import json,datetime,fcntl,shutil,subprocess
root=Path('/home/z/project/deepfake_detector')
e=json.loads((root/'artifacts/execution_status.json').read_text());q=json.loads((root/'artifacts/queue_status.json').read_text())
cfg=json.loads((root/'configs/v1.1.json').read_text())
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return {'PID':pid,'alive':False}
 try:
  args=[a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
  script=next((a for a in args if a.endswith('/run.py')),None)
  fields=(p/'stat').read_text().split(') ',1)[1].split()
  return {'PID':pid,'alive':True,'PPID':int(fields[1]),'start_ticks':int(fields[19]),'cwd':str((p/'cwd').resolve()),'script':script,'stage':args[args.index(script)+1] if script and len(args)>args.index(script)+1 else None}
 except FileNotFoundError:return {'PID':pid,'alive':False,'exited_during_read':True}
rows=[]
for backbone in cfg['backbones']:
 for seed in cfg['seeds']:
  rid=f'{backbone}_seed{seed}';r=q['runs'].get(rid,{})
  out=root/'artifacts/runs'/rid
  history=json.loads((out/'history.json').read_text()) if (out/'history.json').exists() else []
  cache=[role for role in ['train','ff_test','cdf_test','calibration'] if (root/'artifacts/features'/rid/role/'complete.json').exists()]
  row={'run_id':rid,'state':r.get('state','PENDING'),'GPU':r.get('physical_device'),'PID':r.get('pid'),'last_epoch':history[-1]['epoch'] if history else 0,'training_complete':(out/'complete.json').exists(),'cache_complete_roles':cache,'error':r.get('error',r.get('failure'))}
  if r.get('pid'):row['process']=proc(r['pid'])
  if row['state']=='RUNNING' and row['training_complete']:
   row['cache_chunks']={role:len(list((root/'artifacts/features'/rid/role).glob('*.npy'))) for role in ['train','ff_test','cdf_test','calibration']}
  rows.append(row)
locks={}
for name in ['execution.lock','queue.lock']:
 with (root/'artifacts'/name).open('r') as f:
  try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:locks[name]='HELD'
  else:locks[name]='NOT_HELD';fcntl.flock(f.fileno(),fcntl.LOCK_UN)
fmt='{"ID":{{json .Id}},"Running":{{json .State.Running}},"RestartPolicy":{{json .HostConfig.RestartPolicy.Name}}}'
cp=subprocess.run(['docker','inspect','--format',fmt,'onecat-vllm'],capture_output=True,text=True,timeout=20)
service=json.loads(cp.stdout) if cp.returncode==0 else {'inspect_returncode':cp.returncode}
cache_proof=json.loads((root/'reports/completed_run_cache_receipts.json').read_text())
receipt={'checked_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'supervisor':proc(e['owner_pid']),'original_supervisor_PID':3984143,'status':e['status'],'phase':e.get('phase'),'queue_status':q['status'],'locks':locks,'planned_runs':15,'training_complete_models':sum(r['training_complete'] for r in rows),'all_four_cache_markers_present':sum(len(r['cache_complete_roles'])==4 for r in rows),'full_runs_complete':sum(r['state']=='COMPLETE' for r in rows),'independently_verified_full_runs':len(cache_proof['completed_runs']),'runs':rows,'artifact_SSD_free_bytes':shutil.disk_usage(root/'artifacts').free,'original_service_state':service,'execution_error':e.get('failure',e.get('error')),'queue_error':q.get('failure',q.get('error')),'duplicate_jobs_started':False,'new_schedules_created':False,'settings_changed':False,'read_only_remote_check':True}
print(json.dumps(receipt,ensure_ascii=False,indent=2))
