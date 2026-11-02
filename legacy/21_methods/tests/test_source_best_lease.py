import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import pytest
from reliability_v2 import source_best_lease as m

UUIDS=['GPU-'+str(n)*8+'-'+str(n)*4+'-'+str(n)*4+'-'+str(n)*4+'-'+str(n)*12 for n in range(5)]
def child(phase,script,*,extra_fds=(),env=None):
 payload=json.dumps(phase.proof())
 return subprocess.Popen([sys.executable,'-c',script,str(phase.root),payload,str(phase.fd)],pass_fds=(phase.fd,*extra_fds),env=env,text=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
VERIFY='from reliability_v2 import source_best_lease as m;import sys,json; m.verify_worker(sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3]));print("PASS",flush=True);sys.stdin.read(1)'
def finish(p):
 if p.poll() is None:p.stdin.write('x');p.stdin.flush()
 return p.communicate(timeout=10)

def test_phase_contends_with_existing_original_flock(tmp_path):
 p=tmp_path/'H_pipeline.lock'
 with p.open('a+') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  with pytest.raises(BlockingIOError):
   with m.PhaseLease(tmp_path):pass
 assert p.read_bytes()==b''

def test_own_phase_proof_and_duplicate_supervisor_refusal(tmp_path):
 with m.PhaseLease(tmp_path) as phase:
  phase.check();assert phase.proof()['owner_pid']==os.getpid()
  with pytest.raises(BlockingIOError):
   with m.PhaseLease(tmp_path):pass
  p=child(phase,VERIFY)
  assert p.stdout.readline().strip()=='PASS';finish(p);assert p.returncode==0

@pytest.mark.parametrize('kind',['symlink','nonempty'])
def test_phase_foreign_lock_preserved(tmp_path,kind):
 p=tmp_path/'H_pipeline.lock';outside=tmp_path.parent/(tmp_path.name+'_outside')
 if kind=='symlink':p.symlink_to(outside)
 else:p.write_bytes(b'preserve')
 with pytest.raises((OSError,ValueError)):
  with m.PhaseLease(tmp_path):pass
 assert not outside.exists()
 if kind=='nonempty':assert p.read_bytes()==b'preserve'

def test_inherited_fd_retains_phase_after_owner_context_closes(tmp_path):
 phase=m.PhaseLease(tmp_path)
 with phase:
  p=child(phase,VERIFY);assert p.stdout.readline().strip()=='PASS'
 with (tmp_path/'H_pipeline.lock').open('r') as other:
  with pytest.raises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
 finish(p);assert p.returncode==0
 with m.PhaseLease(tmp_path):pass

def test_worker_rejects_forged_parent_start_time(tmp_path):
 with m.PhaseLease(tmp_path) as phase:
  proof=phase.proof();proof['owner_start_ticks']+=1
  code='from reliability_v2 import source_best_lease as m;import sys,json;m.verify_worker(sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3]))'
  p=subprocess.run([sys.executable,'-c',code,str(tmp_path),json.dumps(proof),str(phase.fd)],pass_fds=(phase.fd,),capture_output=True,text=True)
  assert p.returncode!=0

def test_owner_death_signal_releases_inherited_lock(tmp_path):
 owner=r"""import sys,subprocess,json
from reliability_v2 import source_best_lease as m
with m.PhaseLease(sys.argv[1]) as phase:
 code='from reliability_v2 import source_best_lease as m;import sys,json,time;m.verify_worker(sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3]),arm_parent_death=True);print("READY",flush=True);time.sleep(20)'
 p=subprocess.Popen([sys.executable,'-c',code,sys.argv[1],json.dumps(phase.proof()),str(phase.fd)],pass_fds=(phase.fd,),stdout=subprocess.PIPE,text=True)
 assert p.stdout.readline().strip()=='READY'
 print('READY',flush=True);sys.stdin.read(1)
"""
 p=subprocess.Popen([sys.executable,'-c',owner,str(tmp_path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 assert p.stdout.readline().strip()=='READY';p.stdin.write('x');p.stdin.flush();p.communicate(timeout=10);assert p.returncode==0
 deadline=time.monotonic()+5
 while True:
  try:
   with m.PhaseLease(tmp_path):break
  except BlockingIOError:
   assert time.monotonic()<deadline;time.sleep(.05)

GPU=r"""
import json,sys,os
from reliability_v2 import source_best_lease as m
root,proof,fd=sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3])
uuids=json.loads(os.environ['TEST_UUIDS']);config=dict(allowed_gpus=[0,1,2,3],allowed_gpu_uuids=uuids[:4])
def query(args):
 if '--query-compute-apps=gpu_uuid,pid' in args:return os.environ.get('TEST_OCCUPIED','')
 return '\n'.join(f'{n}, {u}, {os.environ.get("TEST_USED","0")}' for n,u in enumerate(uuids))
m._query=query
with m.native_gpu(root,config,proof,fd):print('GPU_HELD',flush=True);sys.stdin.read(1)
"""
def gpu_env(index=0,**extra):return dict(os.environ,CUDA_VISIBLE_DEVICES=UUIDS[index],TEST_UUIDS=json.dumps(UUIDS),**extra)
def test_shared_native_gpu_lock_prevents_second_worker(tmp_path):
 with m.PhaseLease(tmp_path) as phase:
  a=child(phase,GPU,env=gpu_env());assert a.stdout.readline().strip()=='GPU_HELD'
  b=child(phase,GPU,env=gpu_env());b.communicate(timeout=10);assert b.returncode!=0
  with (tmp_path/'gpu_locks'/f'{UUIDS[0]}.lock').open('r') as original:
   with pytest.raises(BlockingIOError):fcntl.flock(original,fcntl.LOCK_EX|fcntl.LOCK_NB)
  finish(a);assert a.returncode==0

@pytest.mark.parametrize('index,extra',[(4,{}),(0,{'TEST_USED':'100'}),(0,{'TEST_OCCUPIED':UUIDS[0]+', 12345'})])
def test_protected_or_nonidle_gpu_refused(tmp_path,index,extra):
 with m.PhaseLease(tmp_path) as phase:
  p=child(phase,GPU,env=gpu_env(index,**extra));p.communicate(timeout=10);assert p.returncode!=0
 assert not (tmp_path/'gpu_locks'/f'{UUIDS[4]}.lock').exists()


def test_separately_opened_same_inode_fd_is_not_inherited_lease(tmp_path):
 with m.PhaseLease(tmp_path) as phase:
  with (tmp_path/'H_pipeline.lock').open('r') as separate:
   code='from reliability_v2 import source_best_lease as m;import sys,json;m.verify_worker(sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3]))'
   p=subprocess.run([sys.executable,'-c',code,str(tmp_path),json.dumps(phase.proof()),str(separate.fileno())],pass_fds=(separate.fileno(),),capture_output=True,text=True)
   assert p.returncode!=0

def test_failed_phase_postlock_check_does_not_leak_lock(tmp_path,monkeypatch):
 phase=m.PhaseLease(tmp_path);monkeypatch.setattr(m,'_held',lambda *a:False)
 try:
  with pytest.raises(ValueError):phase.__enter__()
  with (tmp_path/'H_pipeline.lock').open('r') as independent:fcntl.flock(independent,fcntl.LOCK_EX|fcntl.LOCK_NB)
 finally:
  if phase.fd is not None:os.close(phase.fd);phase.fd=None

def test_gpu_uuid_mapping_to_protected_index_never_creates_protected_lock(tmp_path):
 code=GPU.replace("f'{n}, {u},", "f'{4 if n==0 else 0 if n==4 else n}, {u},")
 with m.PhaseLease(tmp_path) as phase:
  p=child(phase,code,env=gpu_env());p.communicate(timeout=10);assert p.returncode!=0
 assert not (tmp_path/'gpu_locks'/f'{UUIDS[0]}.lock').exists()

def test_worker_refuses_owner_fd_loss(tmp_path):
 code=VERIFY.replace("sys.stdin.read(1)","sys.stdin.read(1);m.verify_worker(sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3]))")
 with m.PhaseLease(tmp_path) as phase:
  p=child(phase,code);assert p.stdout.readline().strip()=='PASS'
 finish(p);assert p.returncode!=0


@pytest.mark.parametrize('failure',['start','nonce'])
def test_review_all_phase_initialization_exceptions_close_fd(tmp_path,monkeypatch,failure):
 phase=m.PhaseLease(tmp_path)
 def broken(*a):raise RuntimeError('synthetic initialization fault')
 if failure=='start':monkeypatch.setattr(m,'_start',broken)
 else:monkeypatch.setattr(m.uuid,'uuid4',broken)
 try:
  with pytest.raises(RuntimeError):phase.__enter__()
  with (tmp_path/'H_pipeline.lock').open('r') as other:fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
 finally:
  if phase.fd is not None:os.close(phase.fd);phase.fd=None

def test_review_parent_death_cannot_be_ignored(tmp_path):
 owner=r"""import sys,subprocess,json
from reliability_v2 import source_best_lease as m
with m.PhaseLease(sys.argv[1]) as phase:
 code='from reliability_v2 import source_best_lease as m;import sys,json,time,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);m.verify_worker(sys.argv[1],json.loads(sys.argv[2]),int(sys.argv[3]),arm_parent_death=True);print("READY",flush=True);sys.stdin.read(1)'
 p=subprocess.Popen([sys.executable,'-c',code,sys.argv[1],json.dumps(phase.proof()),str(phase.fd)],pass_fds=(phase.fd,),stdin=sys.stdin,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
 assert p.stdout.readline().strip()=='READY'
 print('READY',flush=True);sys.stdin.read(1)
"""
 # A private pipe permits bounded cleanup of this self-owned CPU child even
 # against the old buggy SIGTERM implementation, without signaling any PID.
 read_fd,write_fd=os.pipe()
 owner=owner.replace('stdin=sys.stdin','stdin='+str(read_fd))
 p=subprocess.Popen([sys.executable,'-c',owner,str(tmp_path)],pass_fds=(read_fd,),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 os.close(read_fd)
 try:
  assert p.stdout.readline().strip()=='READY';p.stdin.write('x');p.stdin.flush();p.communicate(timeout=10);assert p.returncode==0
  deadline=time.monotonic()+2
  while True:
   try:
    with m.PhaseLease(tmp_path):break
   except BlockingIOError:
    assert time.monotonic()<deadline,'orphan ignored death cleanup';time.sleep(.05)
 finally:
  try:os.write(write_fd,b'x')
  except BrokenPipeError:pass
  os.close(write_fd)
