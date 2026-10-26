"""One-shot supervisor/worker for the existing, reviewed H freeze protocol."""
from pathlib import Path
from datetime import datetime,timezone
import fcntl,hashlib,json,os,subprocess,sys,traceback
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
REVIEW=R/'reports/H_production_data_independent_review_1450.json'
DEEP=R/'reports/H_production_data_inspection.json'
LAUNCH=R/'reports/H_production_data_freeze_launch.json'
EXIT=R/'reports/H_production_data_freeze_exit.json'
FREEZE=R/'data/H/freeze.json'
def utc():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):
 t=p.with_name(p.name+'.freeze1800.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def worker():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 assert all(os.environ.get(k)=='2' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'])
 with (R/'H_production_data_freeze.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not FREEZE.exists() and not FREEZE.is_symlink() and not LAUNCH.exists()
  from reliability_v2 import data_h
  assert sha(data_h.__file__)=='ad81f65c27d53159e6902bf8a20b0910cd129f75eb0105e5a5a053bcc2d2f45e'
  deep=json.loads(DEEP.read_text());launch=json.loads((R/'reports/H_production_data_inspection_launch.json').read_text())
  assert sha(DEEP)==launch['inspection_report_sha256']=='c319ee7a6a6ab332b85f5ecd4a0b56bebeedd073228e6d8227ad187a09806b2d'
  assert launch['status']==deep['status']=='READY_FOR_INDEPENDENT_REVIEW' and deep['blockers']==[] and not deep['target_performance_used']
  assert not Path('/proc/200931').exists()
  independent=data_h._review(REVIEW,deep)
  data_h._unchanged(deep)
  acceptance=json.loads((R/'reports/H_fresh_full_audit_acceptance_1630.json').read_text())
  for path,hsh in acceptance['protected_code_config_plan_manuscripts_sha256'].items():assert sha(path)==hsh,path
  assert not (R/'qualification/H_gate.json').exists()
  rec={'utc':utc(),'status':'RUNNING','operation':'data_h.freeze_h_data mandatory full integrity revalidation then exclusive freeze','pid':os.getpid(),'supervisor_pid':os.getppid(),'CPU_only':True,'OMP_MKL_OPENBLAS_threads':2,'deep_report_path':str(DEEP),'deep_report_sha256':sha(DEEP),'independent_review':independent,'data_gate_code_sha256':sha(data_h.__file__),'does_not_train_or_use_GPU':True,'freeze_created':False}
  save(LAUNCH,rec);print(json.dumps(rec,ensure_ascii=False),flush=True)
  try:
   receipt=data_h.freeze_h_data(R,REVIEW)
   assert json.loads(FREEZE.read_text())==receipt and receipt['status']=='FROZEN'
   assert receipt['independent_review']['sha256']==independent['sha256'] and sha(REVIEW)==independent['sha256']
   data_h._unchanged(receipt)
   rec.update(status='FROZEN',ended_utc=utc(),freeze_created=True,freeze_path=str(FREEZE),freeze_sha256=sha(FREEZE),blockers=[])
   save(LAUNCH,rec);print(json.dumps(rec,ensure_ascii=False),flush=True)
  except BaseException as exc:
   rec.update(status='FAILED',ended_utc=utc(),freeze_created=FREEZE.exists(),error_type=type(exc).__name__,error=str(exc));save(LAUNCH,rec);traceback.print_exc();raise

def main():
 if len(sys.argv)==2 and sys.argv[1]=='--worker':worker();return
 assert len(sys.argv)==1
 with (R/'H_production_data_freeze_supervisor.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not LAUNCH.exists() and not EXIT.exists() and not FREEZE.exists()
  review=json.loads(REVIEW.read_text());assert review['schema']=='H_DATA_INDEPENDENT_REVIEW_V1' and review['status']=='PASS' and review['blockers']==[] and review['reviewer']['independent'] is True
  started=utc();command=[sys.executable,'-u',str(Path(__file__).resolve()),'--worker']
  child=subprocess.Popen(command,env=os.environ.copy());rc=child.wait()
  exit_receipt={'utc':utc(),'started_utc':started,'supervisor_pid':os.getpid(),'worker_pid':child.pid,'returncode':rc,'command':command,'freeze_exists':FREEZE.exists(),'freeze_sha256':sha(FREEZE) if FREEZE.exists() else None,'launch_receipt_sha256':sha(LAUNCH) if LAUNCH.exists() else None}
  save(EXIT,exit_receipt);print(json.dumps(exit_receipt,ensure_ascii=False),flush=True)
  raise SystemExit(rc)
if __name__=='__main__':main()
