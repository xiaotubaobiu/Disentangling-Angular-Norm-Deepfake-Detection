from pathlib import Path
import subprocess,json,datetime,time,hashlib,os
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');P=R/'reports';RUN='D_formal_dinov2_vit_b14_seed42'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
cmd=P/'analysis_execution_records'
rank=cmd/(RUN+'.saved_rank_audit.command.py');terminal=cmd/(RUN+'.terminal_audit.command.py')
r=subprocess.run([str(R/'environment/bin/python'),str(rank)],capture_output=True,text=True)
(P/(RUN+'_saved_rank_audit_process.json')).write_text(json.dumps({'utc':now(),'returncode':r.returncode,'command_sha256':sha(rank),'stdout':r.stdout[-3000:],'stderr':r.stderr[-3000:]},indent=2)+'\n')
assert r.returncode==0
start=now();t=time.monotonic();log=P/(RUN+'_terminal_audit.stdout.log')
with log.open('xb') as out:
 child=subprocess.Popen([str(R/'environment/bin/python'),'-u',str(terminal)],stdout=out,stderr=subprocess.STDOUT)
 (P/(RUN+'_terminal_audit_progress.json')).write_text(json.dumps({'utc':now(),'status':'RUNNING','pid':child.pid,'supervisor_pid':os.getpid(),'command_sha256':sha(terminal)},indent=2)+'\n')
 rc=child.wait()
(P/(RUN+'_terminal_audit_process.json')).write_text(json.dumps({'started_utc':start,'ended_utc':now(),'run_id':RUN,'pid':child.pid,'returncode':rc,'seconds':time.monotonic()-t,'command_sha256':sha(terminal),'template_sha256':'8453de3803c346231096bac8efa0cf8bd774b02102997f9c7cbdba730eb7118d','stdout':str(log)},indent=2)+'\n')
assert rc==0
print('INDEPENDENT_AUDITS_COMPLETE',RUN,flush=True)
