import unittest,tempfile,pathlib,hashlib,os,json
import relocation as m
class RelocationTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup);p=pathlib.Path(self.t.name);self.r=p/'ssd';self.c=p/'cold';self.r.mkdir();self.c.mkdir();self.s=self.r/'runs/D_formal_fixture_seed7/state.pt';self.s.parent.mkdir(parents=True);self.s.write_bytes(b'original-evidence');self.dst=self.c/'runs/D_formal_fixture_seed7/state.pt';self.dst.parent.mkdir(parents=True);self.dst.write_bytes(self.s.read_bytes());st=self.s.stat();self.original=(st.st_ino,st.st_mtime_ns);self.e={'relative':str(self.s.relative_to(self.r)),'staged_path':str(self.dst),'bytes':st.st_size,'source_inode':st.st_ino,'mtime_ns':st.st_mtime_ns,'sha256':hashlib.sha256(self.s.read_bytes()).hexdigest()};self.j=self.r/'journal.json'
 def restored(self):
  self.assertFalse(self.s.is_symlink());self.assertEqual(self.s.read_bytes(),b'original-evidence');self.assertEqual(self.s.stat().st_mtime_ns,self.original[1])
 def test_pilot_returns_exact_original(self):
  m.relocate(self.r,self.c,[self.e],self.j,lambda paths:self.assertTrue(paths[0].is_symlink()),commit=False);self.restored();self.assertEqual(self.s.stat().st_ino,self.original[0])
 def test_validator_exception_rolls_back(self):
  def fail(paths):raise RuntimeError('inject')
  with self.assertRaisesRegex(RuntimeError,'inject'):m.relocate(self.r,self.c,[self.e],self.j,fail,commit=True)
  self.restored();self.assertEqual(self.s.stat().st_ino,self.original[0])
 def test_corrupt_cold_rejected_before_change(self):
  self.dst.write_bytes(b'corrupt');
  with self.assertRaises(ValueError):m.relocate(self.r,self.c,[self.e],self.j,lambda p:None,True)
  self.restored()
 def test_changed_source_metadata_rejected(self):
  self.e['mtime_ns']-=1
  with self.assertRaises(ValueError):m.relocate(self.r,self.c,[self.e],self.j,lambda p:None,True)
  self.restored()
 def test_boundary_rejected(self):
  self.e['relative']='../outside/state.pt'
  with self.assertRaises(ValueError):m.relocate(self.r,self.c,[self.e],self.j,lambda p:None,True)
  self.restored()
 def test_primary_weight_excluded(self):
  self.e['relative']='runs/D_formal_fixture_seed7/final.pt'
  with self.assertRaises(ValueError):m.relocate(self.r,self.c,[self.e],self.j,lambda p:None,True)
  self.restored()
 def test_commit_and_materialize_restore_preserve_bytes_and_ns(self):
  m.relocate(self.r,self.c,[self.e],self.j,lambda p:None,True);self.assertTrue(self.s.is_symlink());self.assertEqual(self.s.read_bytes(),b'original-evidence');self.assertTrue(self.dst.exists());m.restore(self.j);self.restored();self.assertTrue(self.dst.exists())
 def test_duplicate_transaction_refused(self):
  m.relocate(self.r,self.c,[self.e],self.j,lambda p:None,False)
  with self.assertRaises(FileExistsError):m.relocate(self.r,self.c,[self.e],self.j,lambda p:None,False)
  self.restored()

 def test_busy_lock_refused(self):
  import fcntl
  with (self.r/'.cold_auxiliary_migration.lock').open('a+') as f:
   fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
   with self.assertRaises(BlockingIOError):m.relocate(self.r,self.c,[self.e],self.j,lambda p:None,False)
  self.restored()
 def test_preexisting_backup_preserved(self):
  b=self.s.with_name('.state.pt.cold-original.'+self.j.stem);b.write_bytes(b'unknown-evidence')
  with self.assertRaises(ValueError):m.relocate(self.r,self.c,[self.e],self.j,lambda p:None,True)
  self.assertEqual(b.read_bytes(),b'unknown-evidence');self.restored()
 def test_process_exit_during_validation_recovers(self):
  import subprocess,sys
  code="import relocation as m,sys,json,os; m.relocate(sys.argv[1],sys.argv[2],json.loads(sys.argv[3]),sys.argv[4],lambda p:os._exit(77),True)"
  p=subprocess.run([sys.executable,'-c',code,str(self.r),str(self.c),json.dumps([self.e]),str(self.j)])
  self.assertEqual(p.returncode,77);self.assertTrue(self.s.is_symlink());m.restore(self.j);self.restored();self.assertEqual(self.s.stat().st_ino,self.original[0])
 def test_process_exit_after_partial_commit_restores_all(self):
  import subprocess,sys
  s2=self.s.with_name('epoch_01_methods.npy');s2.write_bytes(b'other-evidence');t2=self.dst.with_name(s2.name);t2.write_bytes(s2.read_bytes());st=s2.stat();e2=dict(self.e,relative=str(s2.relative_to(self.r)),staged_path=str(t2),bytes=st.st_size,source_inode=st.st_ino,mtime_ns=st.st_mtime_ns,sha256=hashlib.sha256(s2.read_bytes()).hexdigest())
  code="""import relocation as m,sys,json,os,pathlib
original=pathlib.Path.unlink
def kill_after_unlink(self,*a,**kw):
 original(self,*a,**kw)
 if '.cold-original.' in self.name:os._exit(78)
pathlib.Path.unlink=kill_after_unlink
m.relocate(sys.argv[1],sys.argv[2],json.loads(sys.argv[3]),sys.argv[4],lambda p:None,True)
"""
  p=subprocess.run([sys.executable,'-c',code,str(self.r),str(self.c),json.dumps([self.e,e2]),str(self.j)])
  self.assertEqual(p.returncode,78);m.restore(self.j);self.restored();self.assertFalse(s2.is_symlink());self.assertEqual(s2.read_bytes(),b'other-evidence');self.assertEqual(s2.stat().st_mtime_ns,e2['mtime_ns']);self.assertTrue(self.dst.exists() and t2.exists())


 def test_missing_cold_during_validation_recovers_existing_original(self):
  def fail(paths):self.dst.unlink();raise RuntimeError('cold disappeared')
  with self.assertRaisesRegex(RuntimeError,'cold disappeared'):m.relocate(self.r,self.c,[self.e],self.j,fail,True)
  self.restored();self.assertEqual(self.s.stat().st_ino,self.original[0])

if __name__=='__main__':unittest.main()
