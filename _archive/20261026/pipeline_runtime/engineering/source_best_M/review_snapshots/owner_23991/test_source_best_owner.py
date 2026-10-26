import importlib.util,sys
from pathlib import Path
import pytest
P=Path(__file__).parents[1]/'tools/source_best_owner.py'
spec=importlib.util.spec_from_file_location('selected_m_owner_under_test',P);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

class Process:
 def __init__(self,pid,code=0,delay=1):self.pid=pid;self.code=code;self.delay=delay
 def poll(self):
  self.delay-=1
  return self.code if self.delay<0 else None
 def wait(self):return self.code

def test_finite_queue_keeps_four_distinct_cards_without_duplicate_jobs():
 starts=[];running={};peaks=[]
 def start(run,device):
  assert device['uuid'] not in running.values();running[run]=device['uuid'];starts.append((run,device['uuid']));peaks.append(len(running));return Process(len(starts)+100)
 def verify(run,p,device):running.pop(run);return {'run':run}
 result=m.schedule([str(i) for i in range(9)],lambda:[{'index':i,'uuid':'GPU-'+str(i)} for i in range(4)],start,verify,lambda *a:None,lambda:None)
 assert result['status']=='ALL_REQUESTED_WORKERS_VERIFIED_OPERATIONALLY_ONLY'
 assert len(starts)==len(set(x[0] for x in starts))==9 and max(peaks)==4 and result['pending']==[]

def test_failure_drains_healthy_children_and_never_retries():
 starts=[];verified=[]
 def start(run,device):starts.append(run);return Process(len(starts),code=1 if run=='0' else 0,delay=1)
 result=m.schedule([str(i) for i in range(9)],lambda:[{'index':i,'uuid':'GPU-'+str(i)} for i in range(4)],start,lambda run,*a:verified.append(run) or {},lambda *a:None,lambda:None)
 assert result['status']=='FAILED_WORKER_REQUIRES_REPAIR_NO_RETRY' and len(starts)==4 and set(verified)=={'1','2','3'} and '0' in result['failures'] and len(result['pending'])==5

def test_no_idle_device_returns_finite_blocked_state():
 called=[]
 result=m.schedule(['a'],lambda:[],lambda *a:called.append(a),lambda *a:None,lambda *a:None,lambda:None)
 assert result['status']=='BLOCKED_NO_IDLE_APPROVED_GPU' and not called

def test_duplicate_run_or_protected_device_never_starts():
 for ids,devices in [(['a','a'],[{'index':0,'uuid':'GPU-0'}]),(['a'],[{'index':4,'uuid':'GPU-4'}])]:
  called=[]
  with pytest.raises(ValueError):m.schedule(ids,lambda:devices,lambda *a:called.append(a),lambda *a:None,lambda *a:None,lambda:None)
  assert not called

def test_verification_failure_is_not_operational_completion():
 def bad(*a):raise ValueError('receipt mismatch')
 result=m.schedule(['a'],lambda:[{'index':0,'uuid':'GPU-0'}],lambda *a:Process(123,delay=0),bad,lambda *a:None,lambda:None)
 assert result['status']=='FAILED_WORKER_REQUIRES_REPAIR_NO_RETRY' and result['completed']=={}
