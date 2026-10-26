
"""Saved genuine-direct-child callback harness executed in memory on5ee987.
Materialized after execution; do not rerun during later cap-only confirmation.
All child work is temporary CPU phase-proof validation, never a GPU worker.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,tempfile,time
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M';W=E/'code/studies/pipeline_reliability/v2'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(W));sys.dont_write_bytecode=True;sys.path.insert(0,str(W))
P=E/'review_snapshots/owner_5ee987/source_best_owner.py'
assert hashlib.sha256(P.read_bytes()).hexdigest()=='5ee98738aad0596d197c4ea79bc8f444852e5a0c18c61b361a1601063a5abddb'
spec=importlib.util.spec_from_file_location('owner_fixed_callback_confirmation',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);results={}
for kind in ('progress','wait','poll'):
 with tempfile.TemporaryDirectory(prefix='owner_callback_confirmation_') as td:
  root=Path(td);children=[];starts=[];verified=[]
  with m.lease.PhaseLease(root) as phase:
   class Wrapped:
    def __init__(self,p):self.p=p;self.pid=p.pid;self.once=True
    def poll(self):
     if kind=='poll' and self.once:self.once=False;raise OSError('synthetic poll error')
     return self.p.poll()
    def wait(self):return self.p.wait()
   def start(run,device):
    code='import sys,json,time;from reliability_v2 import source_best_lease as l;l.verify_worker(sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3]),arm_parent_death=True);print("INHERITED",flush=True);time.sleep(.2)'
    p=subprocess.Popen([sys.executable,'-B','-c',code,str(root),json.dumps(phase.proof()),str(phase.fd)],pass_fds=(phase.fd,),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);assert p.stdout.readline().strip()=='INHERITED';children.append(p);starts.append(run);return Wrapped(p)
   def progress(*a):
    if kind=='progress':raise OSError('synthetic progress error')
   def wait():
    if kind=='wait':raise OSError('synthetic wait error')
    time.sleep(.01)
   answer=m.schedule(['a','b','never'],lambda:[{'index':i,'uuid':'GPU-'+str(i)} for i in range(2)],start,lambda run,*a:verified.append(run) or {},progress,wait)
   reaped=all(p.returncode==0 for p in children)
   for p in children:p.communicate(timeout=2)
   results[kind]={'failed_terminal':answer['status']=='FAILED_WORKER_REQUIRES_REPAIR_NO_RETRY','only_two_original_children_started':starts==['a','b'],'both_healthy_verified':set(verified)=={'a','b'},'pending_never_preserved':answer['pending']==['never'],'all_children_reaped_exit0_before_schedule_return':reaped,'genuine_direct_inherited_phase_passed':True}
print(json.dumps(results));assert all(all(x.values()) for x in results.values())
