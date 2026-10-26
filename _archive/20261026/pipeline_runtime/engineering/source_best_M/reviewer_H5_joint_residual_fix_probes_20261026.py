from pathlib import Path
import os,sys,json,hashlib,importlib.util,tempfile,threading,subprocess,select
import pytest
W=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/engineering/source_best_M/code/studies/pipeline_reliability/v2')
SOURCE_SHA='01e625dc771daa56442364ff49ec8aa819c4ec215b2d467e6ce52621e45d0a57'
TEST_SHA='f24831342399e0c3d9baf3e93cbdba68f14a5e8aa28ad72069a966cd1c077919'
assert hashlib.sha256((W/'tools/h_saved_five_seed_summary.py').read_bytes()).hexdigest()==SOURCE_SHA
assert hashlib.sha256((W/'tests/test_h_saved_five_seed_summary.py').read_bytes()).hexdigest()==TEST_SHA
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
spec=importlib.util.spec_from_file_location('reviewer_h5_joint_fixture',W/'tests/test_h_saved_five_seed_summary.py');T=importlib.util.module_from_spec(spec);spec.loader.exec_module(T)
base=Path(tempfile.mkdtemp(prefix='H5_joint_residual_fix_'));results={}
def snapshots(out):
 return {p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),inode=p.stat().st_ino,mtime_ns=p.stat().st_mtime_ns) for p in out.iterdir()}
# Replay the old two cooperative9MiB publisher schedule: B pauses after
# prediction, A attempts publication, and B is then allowed to stage/commit.
with pytest.MonkeyPatch.context() as mp:
 m,rr,br,request,entries,paths,audits,dump=T.fixture(base/'concurrency',mp)
 out=m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary';payloads={n:b'x' for n in m.OUTPUTS};payloads[m.OUTPUTS[0]]=b'x'*(9*1024**2)
 predicted=threading.Event();allow=threading.Event();doneA=threading.Event();outcomes={};errors={};mkdir=Path.mkdir
 def hook(p,*args,**kw):
  if threading.current_thread().name=='private_B' and p.name.startswith('.'+out.name+'.stage_'):
   predicted.set();assert allow.wait(10),'B barrier timeout'
  return mkdir(p,*args,**kw)
 def run(label):
  try:outcomes[label]=m.publish(out,payloads,rr['sha256'],{},m.INITIAL_SHA,br['path'],br['sha256'],1)
  except BaseException as e:errors[label]=str(e)
  finally:
   if label=='A':doneA.set()
 mp.setattr(Path,'mkdir',hook)
 b=threading.Thread(target=run,args=('B',),name='private_B');a=threading.Thread(target=run,args=('A',),name='private_A');b.start()
 try:
  assert predicted.wait(10);a.start();assert doneA.wait(10);assert 'writer busy' in errors.get('A','')
  stages_before_release=list(out.parent.glob('.'+out.name+'.stage_*'));assert not stages_before_release
 finally:
  allow.set();a.join(15);b.join(15)
 assert not a.is_alive() and not b.is_alive() and set(outcomes)=={'B'} and len(errors)==1
 allocated=m.namespace_allocation(out);assert allocated<=m.CAP and outcomes['B']['writer_reservation_release_confirmed'] is True
 before=snapshots(out);again=m.publish(out,payloads,rr['sha256'],{},m.INITIAL_SHA,br['path'],br['sha256'],0)
 assert again['reused'] is True and again['writer_reservation_release_confirmed'] is True and snapshots(out)==before
 assert not entries
 results['HSUM003_R1_original_concurrent_schedule']=dict(success=outcomes,errors=errors,allocated_bytes=allocated,cap_bytes=m.CAP,busy_before_any_stage=True,reentry_same7_bytes_SHA_inode_mtime=True,shared_slot_calls=len(entries))
# Separate process holds the real abstract socket reservation, then exits
# inside its context via os._exit. No lock files or slots are used.
with pytest.MonkeyPatch.context() as mp:
 m,rr,br,request,entries,paths,audits,dump=T.fixture(base/'process',mp);out=m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary'
 childcode="from pathlib import Path;import importlib.util,sys,os;sp=importlib.util.spec_from_file_location('private_h5_reservation_child',sys.argv[1]);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)\nwith m.writer_reservation(Path(sys.argv[2])):\n print('HELD',flush=True)\n sys.stdin.readline()\n os._exit(0)\n"
 child=subprocess.Popen([sys.executable,'-B','-c',childcode,str(T.P),str(out)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:
  assert select.select([child.stdout],[],[],10)[0],'child readiness timeout';assert child.stdout.readline().strip()=='HELD'
  try:
   with m.writer_reservation(out):raise AssertionError('cross-process reservation bypass')
  except RuntimeError as e:assert 'writer busy' in str(e)
  with m.writer_reservation(out.parent/'different_summary'):pass
 finally:
  stdout,stderr=child.communicate('exit\n',timeout=10)
 assert child.returncode==0,(stdout,stderr)
 with m.writer_reservation(out) as release:pass
 assert release['release_confirmed'] is True and not out.exists() and not entries
 results['HSUM003_R1_process_exit']=dict(busy_while_child_holds=True,released_after_os_exit=True,different_output_independent=True,disk_output_absent=True,child_exit=child.returncode,shared_slot_calls=0)
# Exact historical raw-path alias counterexample, same-audit variant, and
# coherent aliases. Reject every attempt to digest or open a historical PNG.
for mode in ('cross_seed_conflict','same_audit_conflict','coherent_alias'):
 with pytest.MonkeyPatch.context() as mp:
  m,rr,br,request,entries,paths,audits,dump=T.fixture(base/mode,mp);canonical=str(m.ROOT/'data/H/shared_historical.png');alias=str(m.ROOT)+'/data/H/./shared_historical.png';first=json.loads(audits[0].read_text());first['current_pins_sha256'][canonical]='a'*64
  if mode=='same_audit_conflict':first['current_pins_sha256'][alias]='b'*64
  request['pairs'][0]['independent_numeric_audit']=dump(audits[0],first)
  if mode!='same_audit_conflict':
   second=json.loads(audits[1].read_text());second['current_pins_sha256'][alias]=('a' if mode=='coherent_alias' else 'b')*64;request['pairs'][1]['independent_numeric_audit']=dump(audits[1],second)
  rr=dump(rr['path'],request);original_digest=m.digest;original_open=Path.open;attempts=[]
  def no_pixel_digest(path):
   if Path(path)==Path(canonical):attempts.append('digest');raise AssertionError('historical pixel must not be read')
   return original_digest(path)
  def no_pixel_open(p,*args,**kw):
   if Path(p)==Path(canonical):attempts.append('open');raise AssertionError('historical pixel must not be opened')
   return original_open(p,*args,**kw)
  mp.setattr(m,'digest',no_pixel_digest);mp.setattr(Path,'open',no_pixel_open)
  if mode=='coherent_alias':
   run=T.execute(m,rr,br);out=Path(run['output']);identity=json.loads((out/'identity.json').read_text());assert identity['historical_nonconsumed_source_sha256']=={canonical:'a'*64};assert canonical not in identity['fresh_checked_input_sha256']
   before=snapshots(out);again=m.main(['--execute','--request',rr['path'],'--request-sha',rr['sha256'],'--budget',br['path'],'--budget-sha',br['sha256'],'--slot','0']);assert again['reused'] is True and again['sha256']==run['sha256'] and before==snapshots(out)
   results['HSUM004_R1_'+mode]=dict(canonical_identity={canonical:'a'*64},historical_pixel_read_attempts=attempts,reentry_same7_bytes_SHA_inode_mtime=True,same_receipt_SHA=True,slot1_to0=True,shared_slot_calls=len(entries))
  else:
   try:T.execute(m,rr,br)
   except AssertionError as e:
    expected='within accepted audit' if mode=='same_audit_conflict' else 'across five seeds';assert expected in str(e),str(e);error=str(e)
   else:raise AssertionError('historical alias conflict accepted')
   assert not (m.ROOT/'analysis/H/clip_vit_b16/five_seed_saved_result_summary').exists()
   results['HSUM004_R1_'+mode]=dict(rejected=True,reason=error,historical_pixel_read_attempts=attempts,output_absent=True,shared_slot_calls=len(entries))
  assert not attempts and not Path(canonical).exists()
assert hashlib.sha256((W/'tools/h_saved_five_seed_summary.py').read_bytes()).hexdigest()==SOURCE_SHA
assert hashlib.sha256((W/'tests/test_h_saved_five_seed_summary.py').read_bytes()).hexdigest()==TEST_SHA
print('JOINT_RESIDUAL_CONFIRMATION '+json.dumps(dict(source_sha256=SOURCE_SHA,test_sha256=TEST_SHA,root=str(base),CPU_affinity_count=len(os.sched_getaffinity(0)),real_or_synthetic_flock_calls=0,actual_summary_executed=False,source_code_mutations_or_copies=0,results=results),sort_keys=True))
