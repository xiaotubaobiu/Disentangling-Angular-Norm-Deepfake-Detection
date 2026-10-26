import hashlib,json,types
from pathlib import Path
import pytest
from cpu_ledger_adapter import installed_fresh_audit

def fixture_context(tmp_path):
 config=tmp_path/'config.json';config.write_text('{}');code=tmp_path/'code.py';code.write_text('pass')
 ref={'utc':'old','status':'MATCHED','pairs':list(range(10))};reference=tmp_path/'ledger.json';reference.write_text(json.dumps(ref))
 pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [config,code,reference]};calls=[]
 ids=[str(i) for i in range(10)]
 def verify(c,r):assert Path(c)==config;calls.append(r);return {'run':r}
 def original(c,b,*,precision='AMP',verify=None):
  assert b=='bb' and precision=='AMP'
  for r in ids:assert verify(c,r)=={'run':r}
  return dict(ref,utc='now')
 queue=types.SimpleNamespace(audit_paired_ledgers=original)
 return queue,original,config,code,reference,pins,calls,verify,ids

def enter(s):
 q,old,c,code,reference,pins,calls,verify,ids=s
 return installed_fresh_audit(q,c,verify,lambda b,p:ids,lambda b,p:reference,pins)

def test_exact_output_all10_fresh_and_repeated_not_memoized(tmp_path):
 s=fixture_context(tmp_path);q,old,c,_,_,_,calls,_,_=s
 with enter(s) as receipts:
  assert q.audit_paired_ledgers(c,'bb')['status']=='MATCHED';assert sorted(calls)==sorted(str(i) for i in range(10))
  q.audit_paired_ledgers(c,'bb');assert len(calls)==20 and len(receipts)==2
 assert q.audit_paired_ledgers is old

def test_foreign_config_and_caller_verifier_rejected_before_work(tmp_path):
 s=fixture_context(tmp_path);q,_,c,_,_,_,calls,_,_=s
 with enter(s):
  with pytest.raises(ValueError):q.audit_paired_ledgers(tmp_path/'foreign','bb')
  with pytest.raises(ValueError):q.audit_paired_ledgers(c,'bb',verify=lambda *a:None)
 assert not calls

def test_pin_drift_at_entry_and_inside_fresh_verify_rejects(tmp_path):
 s=fixture_context(tmp_path);s[3].write_text('changed')
 with pytest.raises(ValueError):
  with enter(s):pass
 s=fixture_context(tmp_path);oldverify=s[7]
 def mutate(c,r):s[3].write_text('midflight');return oldverify(c,r)
 s=list(s);s[7]=mutate
 with pytest.raises(ValueError):
  with enter(s):s[0].audit_paired_ledgers(s[2],'bb')
 assert s[0].audit_paired_ledgers is s[1]

def test_original_ledger_mismatch_rejects_and_restores(tmp_path):
 s=list(fixture_context(tmp_path));old=s[1]
 def bad(*a,**k):r=old(*a,**k);r['status']='changed';return r
 s[0].audit_paired_ledgers=bad
 with pytest.raises(ValueError):
  with enter(s):s[0].audit_paired_ledgers(s[2],'bb')
 assert s[0].audit_paired_ledgers is bad

def test_worker_failure_restores_without_result(tmp_path):
 s=list(fixture_context(tmp_path));old=s[1]
 def fail(*a):raise RuntimeError('fresh failure')
 s[7]=fail
 with pytest.raises(RuntimeError):
  with enter(s) as receipts:s[0].audit_paired_ledgers(s[2],'bb')
 assert s[0].audit_paired_ledgers is old and receipts==[]

def test_ledger_must_consume_each_exact_receipt_once(tmp_path):
 s=list(fixture_context(tmp_path))
 def bad(c,b,*,precision='AMP',verify=None):
  verify(c,'0');verify(c,'0');return {'utc':'now','status':'MATCHED','pairs':list(range(10))}
 s[0].audit_paired_ledgers=bad
 with pytest.raises(ValueError):
  with enter(s):s[0].audit_paired_ledgers(s[2],'bb')

def test_nested_install_rejected_and_restores(tmp_path):
 s=fixture_context(tmp_path)
 with enter(s):
  with pytest.raises(ValueError):
   with enter(s):pass
 assert s[0].audit_paired_ledgers is s[1]

def test_exit_waits_for_fresh_audit_and_holds_installation_registration(tmp_path):
 import threading,time
 s=list(fixture_context(tmp_path));started=threading.Event();release=threading.Event();exited=threading.Event();errors=[];threads=[];base=s[7]
 def slow(c,r):
  started.set()
  if not release.wait(3):raise RuntimeError('probe release timeout')
  return base(c,r)
 s[7]=slow
 def audit(q):
  try:q.audit_paired_ledgers(s[2],'bb')
  except BaseException as e:errors.append(e)
 def scope():
  try:
   with enter(s):
    t=threading.Thread(target=audit,args=(s[0],));threads.append(t);t.start();assert started.wait(1)
  except BaseException as e:errors.append(e)
  finally:exited.set()
 outer=threading.Thread(target=scope);outer.start();assert started.wait(1)
 try:
  assert not exited.wait(.05),'scope exited with a fresh audit still active'
  with pytest.raises(ValueError):
   with enter(s):pass
 finally:
  release.set();outer.join(3)
  for t in threads:t.join(3)
 assert not outer.is_alive() and not any(t.is_alive() for t in threads)
 assert not errors and s[0].audit_paired_ledgers is s[1]
