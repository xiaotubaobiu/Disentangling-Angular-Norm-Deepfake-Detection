"""Explicit one-shot H deep inspection AFTER fresh audit; never freezes or trains."""
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,fcntl
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,j):
 q=p.with_name(p.name+'.inspection.tmp');q.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 from reliability_v2 import data_h,geometry_h
 d=R/'data/H';l=json.loads((R/'reports/H_fresh_full_audit_launch_1450.json').read_text());full=json.loads((d/'H_full_candidate_audit_receipt.json').read_text())
 assert l.get('returncode')==0 and l['status']=='FRESH_FULL_AUDIT_COMMAND_COMPLETE','Fresh audit still running or failed; do not inspect/freeze yet'
 assert full['status'] in ['AUDIT_COMPLETE_READY_FOR_H_DATA_FREEZE_REVIEW','AUDIT_CONFLICTS_REVIEW_REQUIRED'] and full['artifact_integrity_status']=='PASSED' and full['candidate_videos']==8431
 assert full['fresh_code_sha256']==sha(Path(geometry_h.__file__))==l['geometry_code_sha256']
 assert not (d/'freeze.json').exists(),'Freeze already exists; inspect existing accepted state separately'
 result_path=R/'reports/H_production_data_inspection.json';assert not result_path.exists(),'Existing inspection report must be examined, never silently overwritten'
 with (R/'H_production_data_inspection.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  started={'utc':datetime.now(timezone.utc).isoformat(),'status':'RUNNING','pid':os.getpid(),'CPU_only':True,'operation':'data_h.inspect_h_data full8431source/PNGintegrity+cohortreconstruction','fresh_full_audit_sha256':sha(d/'H_full_candidate_audit_receipt.json'),'data_gate_code_sha256':sha(Path(data_h.__file__)),'does_not_freeze_or_train':True}
  save(R/'reports/H_production_data_inspection_launch.json',started);print(json.dumps(started),flush=True)
  report=data_h.inspect_h_data(R);save(result_path,report)
  ended={**started,'status':report['status'],'ended_utc':datetime.now(timezone.utc).isoformat(),'inspection_report_sha256':sha(result_path),'blockers':report['blockers']};save(R/'reports/H_production_data_inspection_launch.json',ended);print(json.dumps(ended),flush=True)
  assert report['status']=='READY_FOR_INDEPENDENT_REVIEW' and not report['blockers'],'Production H deep inspection blocked; retain evidence and resolve before freeze'
if __name__=='__main__':main()
