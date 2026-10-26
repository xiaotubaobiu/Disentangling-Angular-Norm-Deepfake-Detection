from pathlib import Path
import importlib.util,hashlib,threading,time
import pytest

P=Path(__file__).resolve().parents[1]/'parallel_source.py'
def api(name):
 spec=importlib.util.spec_from_file_location('parallel_source_test_subject',P)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 obj=getattr(m,name,None)
 assert callable(obj),'Missing candidate behavior: '+name
 return obj

def test_each_invocation_observes_current_file_bytes(tmp_path):
 # Returning previously accepted receipts would hide a changed input.
 collect=api('collect_fresh');f=tmp_path/'weights';f.write_bytes(b'abc')
 def verify(config,run):return {'run':run,'hash':hashlib.sha256(f.read_bytes()).hexdigest()}
 one=collect(tmp_path/'config',['a'],verify,workers=1)
 f.write_bytes(b'changed')
 two=collect(tmp_path/'config',['a'],verify,workers=1)
 assert one['a']['hash']=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
 assert two['a']['hash']!=one['a']['hash']

def test_two_fresh_verifications_really_overlap_and_remain_bounded(tmp_path):
 # A serial implementation or an unbounded pool violates this behavior.
 collect=api('collect_fresh');lock=threading.Lock();barrier=threading.Barrier(2);active=0;peak=0
 def verify(config,run):
  nonlocal active,peak
  with lock:active+=1;peak=max(peak,active)
  barrier.wait(timeout=2)
  time.sleep(.01)
  with lock:active-=1
  return {'run':run,'value':int(run)}
 result=collect(tmp_path/'config',['0','1','2','3'],verify,workers=2)
 assert result=={'0':{'run':'0','value':0},'1':{'run':'1','value':1},'2':{'run':'2','value':2},'3':{'run':'3','value':3}}
 assert peak==2 and active==0

@pytest.mark.parametrize('workers',[0,3,True,'2'])
def test_rejects_invalid_concurrency_before_work(tmp_path,workers):
 collect=api('collect_fresh');touched=[]
 def verify(c,r):touched.append(r);return {'run':r}
 with pytest.raises(ValueError):collect(tmp_path/'config',['a'],verify,workers=workers)
 assert touched==[]

@pytest.mark.parametrize('runs',[[],['a','a']])
def test_rejects_empty_or_duplicate_model_selection(tmp_path,runs):
 collect=api('collect_fresh');touched=[]
 def verify(c,r):touched.append(r);return {'run':r}
 with pytest.raises(ValueError):collect(tmp_path/'config',runs,verify,workers=2)
 assert touched==[]

def test_failed_complete_verification_propagates_no_partial_result(tmp_path):
 collect=api('collect_fresh')
 def verify(c,r):
  if r=='bad':raise RuntimeError('sealed artifact mismatch')
  return {'run':r}
 with pytest.raises(RuntimeError,match='sealed artifact mismatch'):
  collect(tmp_path/'config',['good','bad'],verify,workers=2)

def test_lookup_is_context_bound_and_each_fresh_receipt_is_consumed_once(tmp_path):
 lookup=api('FreshLookup')(tmp_path/'config',{'a':{'value':1},'b':{'value':2}})
 assert lookup(tmp_path/'config','a')=={'value':1}
 with pytest.raises(ValueError):lookup(tmp_path/'other','b')
 with pytest.raises(ValueError):lookup(tmp_path/'config','unknown')
 with pytest.raises(ValueError):lookup(tmp_path/'config','a')
 with pytest.raises(ValueError):lookup.finish()
 assert lookup(tmp_path/'config','b')=={'value':2}
 lookup.finish()

def test_lookup_returns_isolated_receipt_values(tmp_path):
 receipts={'a':{'nested':{'success':2}}};lookup=api('FreshLookup')(tmp_path/'config',receipts)
 result=lookup(tmp_path/'config','a');result['nested']['success']=99
 assert receipts=={'a':{'nested':{'success':2}}}
 lookup.finish()

def test_ledger_comparison_ignores_only_top_level_utc():
 compare=api('assert_same_ledger')
 reference={'utc':'old','status':'MATCHED','pairs':[{'seed':7,'sha':'abc','success':5730}]}
 compare(reference,{'utc':'new','status':'MATCHED','pairs':[{'seed':7,'sha':'abc','success':5730}]})
 with pytest.raises(ValueError):compare(reference,{'utc':'new','status':'MATCHED','pairs':[{'seed':7,'sha':'abc','success':5729}]})
 with pytest.raises(ValueError):compare(reference,{'utc':'new','status':'MATCHED','pairs':[{'seed':7,'sha':'changed','success':5730}]})
 with pytest.raises(ValueError):compare(reference,{'utc':'new','status':'MATCHED','pairs':[{'seed':7,'sha':'abc','success':5730}],'extra':True})

def test_nested_utc_is_not_exempt_from_ledger_comparison():
 compare=api('assert_same_ledger')
 with pytest.raises(ValueError):compare({'utc':'old','pairs':[{'utc':'bound'}]},{'utc':'new','pairs':[{'utc':'changed'}]})

@pytest.mark.parametrize("runs",[[""],[None]])
def test_rejects_invalid_model_identifiers_before_work(tmp_path,runs):
 collect=api("collect_fresh");seen=[]
 def verify(c,r):seen.append(r);return {}
 with pytest.raises(ValueError):collect(tmp_path/"config",runs,verify,workers=2)
 assert seen==[]

def test_numeric_type_drift_is_not_hidden_by_python_equality():
 compare=api("assert_same_ledger")
 with pytest.raises(ValueError):compare({"utc":"old","value":True},{"utc":"new","value":1})
 with pytest.raises(ValueError):compare({"utc":"old","value":5730},{"utc":"new","value":5730.0})

def test_both_ledger_observation_timestamps_are_required():
 compare=api("assert_same_ledger")
 with pytest.raises(ValueError):compare({"utc":"old","status":"OK"},{"status":"OK"})


def entry_api(tmp_path):
 path=P.parent/"benchmark_source.py"
 assert path.exists(),"Missing explicit source-benchmark entry behavior"
 spec=importlib.util.spec_from_file_location("source_benchmark_test_subject",path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.ROOT=tmp_path
 return module

def test_default_entry_is_inert_and_creates_nothing(tmp_path,capsys):
 module=entry_api(tmp_path)
 result=module.main([])
 assert result["executed"] is False and result["production_integration"] is False
 assert list(tmp_path.iterdir())==[]
 assert "CHECK_ONLY" in capsys.readouterr().out

def test_real_entry_rejects_nonempty_GPU_environment_before_IO(tmp_path,monkeypatch):
 module=entry_api(tmp_path);monkeypatch.setenv("CUDA_VISIBLE_DEVICES","GPU-protected")
 with pytest.raises(ValueError):module.main(["--execute"])
 assert list(tmp_path.iterdir())==[]

def test_real_entry_rejects_existing_output_without_overwrite(tmp_path):
 module=entry_api(tmp_path);out=tmp_path/module.REPORT_RELATIVE
 out.parent.mkdir(parents=True);out.write_text("preserve")
 with pytest.raises(FileExistsError):module.main(["--execute"])
 assert out.read_text()=="preserve"

def test_execute_flag_reaches_real_reference_read_not_a_synthetic_success(tmp_path):
 module=entry_api(tmp_path)
 with pytest.raises(FileNotFoundError):module.main(["--execute"])
 assert not (tmp_path/module.REPORT_RELATIVE).exists()


@pytest.mark.parametrize("filename",["worker_h.py","data_h.py","qualification_h.py","queue_d.py"])
def test_entry_rejects_late_code_drift_beyond_queue_h(tmp_path,filename):
 module=entry_api(tmp_path);operation=getattr(module,"with_frozen_sources",None)
 assert callable(operation),"Missing full frozen-source bracketing"
 code=tmp_path/"code";code.mkdir();(code/"queue_h.py").write_text("stable queue")
 changed=code/filename;changed.write_text("original")
 def collect():
  changed.write_text("modified after fresh collection")
  return {"status":"MATCHED"}
 with pytest.raises(ValueError):operation(code,[],collect)
 assert (code/"queue_h.py").read_text()=="stable queue"

def test_entry_frozen_sources_preserve_unchanged_real_operation(tmp_path):
 module=entry_api(tmp_path);operation=getattr(module,"with_frozen_sources",None)
 assert callable(operation),"Missing full frozen-source bracketing"
 code=tmp_path/"code";code.mkdir();(code/"queue_h.py").write_text("queue");worker=code/"worker_h.py";worker.write_text("worker")
 value,hashes=operation(code,[],lambda:{"value":10})
 assert value=={"value":10} and str(worker.resolve()) in hashes

def test_publisher_commits_complete_json_and_removes_success_temporary(tmp_path):
 publish=api("publish_json_new");out=tmp_path/"result.json"
 publish(out,{"value":10})
 import json
 assert json.loads(out.read_text())=={"value":10}
 assert list(tmp_path.iterdir())==[out]

def test_publisher_never_overwrites_existing_result(tmp_path):
 publish=api("publish_json_new");out=tmp_path/"result.json";out.write_text("old")
 with pytest.raises(FileExistsError):publish(out,{"value":10})
 assert out.read_text()=="old"

def test_publisher_serialization_failure_creates_no_final_or_temporary(tmp_path):
 publish=api("publish_json_new");out=tmp_path/"result.json"
 with pytest.raises(TypeError):publish(out,{"bad":object()})
 assert not out.exists() and list(tmp_path.iterdir())==[]

def test_publisher_storage_sync_failure_preserves_partial_evidence_without_final(tmp_path,monkeypatch):
 publish=api("publish_json_new");out=tmp_path/"result.json"
 import os
 def fail(fd):raise OSError("simulated storage sync failure")
 monkeypatch.setattr(os,"fsync",fail)
 with pytest.raises(OSError,match="simulated storage"):publish(out,{"value":10})
 assert not out.exists()
 assert len(list(tmp_path.glob("*.partial")))==1
