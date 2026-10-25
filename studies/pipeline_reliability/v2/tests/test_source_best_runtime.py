import os,json,hashlib
from pathlib import Path
import pytest
from reliability_v2 import source_best_runtime as m,source_best_sink as sink

@pytest.fixture
def case(tmp_path):
 out=tmp_path/'analysis/source_best_M'/('a'*64)/'job'
 return tmp_path,out

def test_same_primary_CPU_slot_excludes_all_secondary_work(case):
 root,out=case
 with m.cpu_slot(root,0):
  with pytest.raises(BlockingIOError):
   with m.cpu_slot(root,0):pass
  with m.cpu_slot(root,1):pass
 with m.cpu_slot(root,0):pass

def test_other_process_Cannot_duplicate_CPU_slot(case):
 root,_=case
 with m.cpu_slot(root,0):
  pid=os.fork()
  if pid==0:
   try:
    with m.cpu_slot(root,0):os._exit(2)
   except BlockingIOError:os._exit(0)
  assert os.waitpid(pid,0)[1]==0

def test_two_live_reservations_exclude_oversubscription(case):
 root,out=case;other=out.with_name('other')
 with m.TemporaryPool(root,out,10*1024**2,16*1024**2):
  with pytest.raises(ValueError):
   with m.TemporaryPool(root,other,10*1024**2,16*1024**2):pass
 with m.TemporaryPool(root,other,10*1024**2,16*1024**2):pass

def test_duplicate_job_claim_refused(case):
 root,out=case
 with m.TemporaryPool(root,out,4*1024**2,16*1024**2):
  with pytest.raises(ValueError):
   with m.TemporaryPool(root,out,4*1024**2,16*1024**2):pass

def test_failed_staging_remains_charged_after_release(case):
 root,out=case;stage=out/'.staging';stage.mkdir(parents=True)
 with m.TemporaryPool(root,out,12*1024**2,16*1024**2):
  (stage/'failed').write_bytes(b'x'*(10*1024**2))
 before=hashlib.sha256((stage/'failed').read_bytes()).hexdigest()
 with pytest.raises(ValueError):
  with m.TemporaryPool(root,out.with_name('next'),8*1024**2,16*1024**2):pass
 assert hashlib.sha256((stage/'failed').read_bytes()).hexdigest()==before

def test_dead_claim_recovery_only_frees_unused_reservation(case):
 root,out=case;pid=os.fork()
 if pid==0:
  pool=m.TemporaryPool(root,out,12*1024**2,16*1024**2);pool.__enter__();p=out/'.staging';p.mkdir(parents=True);(p/'failed').write_bytes(b'x'*(10*1024**2));os._exit(0)
 assert os.waitpid(pid,0)[1]==0
 with pytest.raises(ValueError):
  with m.TemporaryPool(root,out.with_name('next'),8*1024**2,16*1024**2):pass
 with m.TemporaryPool(root,out.with_name('next'),4*1024**2,16*1024**2):pass
 assert (out/'.staging/failed').exists()

def test_resource_symlinks_and_forged_release_refused(case):
 root,out=case
 with m.TemporaryPool(root,out,4*1024**2,16*1024**2) as pool:claim=pool.claim_path
 release=claim.with_suffix('.released.json');d=json.loads(release.read_text());d['claim_sha256']='0'*64;release.write_text(json.dumps(d))
 with pytest.raises(ValueError):
  with m.TemporaryPool(root,out.with_name('next'),4*1024**2,16*1024**2):pass

def test_fork_child_cannot_release_parent_claim(case):
 root,out=case
 with m.TemporaryPool(root,out,4*1024**2,16*1024**2) as pool:
  pid=os.fork()
  if pid==0:
   try:pool.__exit__(None,None,None);os._exit(2)
   except ValueError:os._exit(0)
  assert os.waitpid(pid,0)[1]==0


def test_child_context_close_preserves_parent_CPU_lease(case):
 root,_=case;lease=m.cpu_slot(root,0);lease.__enter__();pid=os.fork()
 if pid==0:
  lease.__exit__(None,None,None);os._exit(0)
 os.waitpid(pid,0)
 try:
  with pytest.raises(BlockingIOError):
   with m.cpu_slot(root,0):pass
 finally:lease.__exit__(None,None,None)

def test_constructed_pool_cannot_enter_in_fork_child(case):
 root,out=case;pool=m.TemporaryPool(root,out,4*1024**2,16*1024**2);pid=os.fork()
 if pid==0:
  try:pool.__enter__();os._exit(2)
  except ValueError:os._exit(0)
 assert os.waitpid(pid,0)[1]==0

def test_zombie_does_not_hold_unused_reservation(case):
 import time
 root,out=case;pid=os.fork()
 if pid==0:
  pool=m.TemporaryPool(root,out,12*1024**2,16*1024**2);pool.__enter__();os._exit(0)
 try:
  for _ in range(200):
   if Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()[0]=='Z':break
   time.sleep(.01)
  assert m.live({'pid':pid,'birth':m.lease._start(pid)}) is False
  with m.TemporaryPool(root,out.with_name('next'),12*1024**2,16*1024**2):pass
 finally:os.waitpid(pid,0)

def test_dangling_staging_link_is_not_silently_uncharged(case):
 root,out=case;out.mkdir(parents=True);(out/'.staging').symlink_to(root/'missing',target_is_directory=True)
 with pytest.raises(ValueError):
  with m.TemporaryPool(root,out,4*1024**2,16*1024**2):pass
