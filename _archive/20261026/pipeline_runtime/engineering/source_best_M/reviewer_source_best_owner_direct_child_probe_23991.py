
"""Temporary-only reproduction of owner progress-failure drain bug (old23991).
Run with project Python -B; optionally pass the exact preserved23991 source path.
No real GPU, nvidia-smi, R lock or existing PID is used.
"""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,sys,tempfile,time
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');W=R/'engineering/source_best_M/code/studies/pipeline_reliability/v2'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(W));sys.dont_write_bytecode=True;sys.path.insert(0,str(W))
P=Path(sys.argv[1]) if len(sys.argv)>1 else W/'tools/source_best_owner.py'
assert hashlib.sha256(P.read_bytes()).hexdigest()=='23991aa298d068563a1c7db652b6ab9df0ba82d19de3a5f3cec8f4b98f0e8e55','Reproduction is bound to original23991 source'
spec=importlib.util.spec_from_file_location('owner_direct_child_reproduction',P);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix='owner_direct_child_reproduction_') as td:
 root=Path(td);children=[]
 with m.lease.PhaseLease(root) as phase:
  def start(run,device):
   source='import sys,json,time;from reliability_v2 import source_best_lease as l;l.verify_worker(sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3]),arm_parent_death=True);print("INHERITED",flush=True);time.sleep(.35)'
   p=subprocess.Popen([sys.executable,'-B','-c',source,str(root),json.dumps(phase.proof()),str(phase.fd)],pass_fds=(phase.fd,),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);assert p.stdout.readline().strip()=='INHERITED';children.append(p);return p
  def broken(*a):raise OSError('synthetic status storage failure')
  escaped=False
  try:m.schedule(['a','b','never'],lambda:[{'index':i,'uuid':'GPU-'+str(i)} for i in range(2)],start,lambda *a:{},broken,lambda:time.sleep(.01))
  except OSError:escaped=True
  live_at_escape=sum(p.poll() is None for p in children)
  for p in children:p.communicate(timeout=3)
  result={'escaped_instead_of_terminal_failure':escaped,'healthy_children_still_running_when_schedule_escaped':live_at_escape,'all_temporary_children_manually_drained_exit0':all(p.returncode==0 for p in children),'direct_inherited_phase_verification_passed':True}
print(json.dumps(result));assert escaped and live_at_escape>0 and result['all_temporary_children_manually_drained_exit0']
