import copy
from pathlib import Path
import numpy as np,pytest
from test_source_best_cache import project
from reliability_v2 import source_best_source_qualification as q,source_best_cache as cache

def test_exact_nine_source_only_qualification_endpoints():
 runs=q.qualification_runs();assert len(runs)==9 and len(set(runs))==9
 assert sum(r.startswith('D_') for r in runs)==3 and sum(r.startswith('H_') for r in runs)==6
 assert all(r.endswith('_seed7') and '_formal_' in r for r in runs)

def test_qualification_forwards_source_cal_only_and_original_head(project):
 c=cache.prepare_context(project['root'],project['admission'],project['admission_sha'],project['run']);before=c.roles;calls=[]
 result=q.source_calibration_forward(c,'cpu',lambda full=False:calls.append(full))
 role='source_cal' if project['route']=='D' else 'B/SOURCE_CAL';assert result['source_role']==role and result['rows']==c.tables[role]['rows'] and result['repeated_forward_FP32_bitwise_equal'] is True
 assert result['target_encoder_inference'] is False and c.roles==before
 assert result['original_head_sha256']==c.source['head_sha256']
 for r in c.roles:assert (cache.role_path(c,r)/'complete.json').exists()==(r==role)
 assert len(calls)>2
 # A real qualification cannot pass by merely reusing an existing cache.
 result2=q.source_calibration_forward(c,'cpu',lambda full=False:None);assert result2['fresh_forward_rows']==result['rows']

@pytest.mark.parametrize('mutation',['revoked','wrong_head','not_source'])
def test_source_qualification_fails_closed(project,monkeypatch,mutation):
 c=cache.prepare_context(project['root'],project['admission'],project['admission_sha'],project['run'])
 def guard(full=False):
  if mutation=='revoked':raise ValueError('lease revoked')
 if mutation=='wrong_head':
  c.source['head_sha256']='0'*64
 if mutation=='not_source':
  role='source_cal' if project['route']=='D' else 'B/SOURCE_CAL';c.tables[role]['rows']+=1
 with pytest.raises(ValueError):q.source_calibration_forward(c,'cpu',guard)


@pytest.fixture
def closed_case(tmp_path):
 import json
 root=tmp_path;ids=[f'H_formal_{b}_{h}_seed{s}' for b in q.BACKBONES for s in (7,42,123,2026,3407) for h in ('B','AB')];src=root/'source.json';src.write_text(json.dumps({'records':{k:{'route':'H'} for k in ids}}));inp={'source_file_admission':{'path':str(src),'sha256':q.sf.digest(src)}}
 stage={'status':'PRIMARY30_CACHE_WORKERS_RETURNED_210ROLE_MARKERS_PENDING_INDEPENDENT_ADMISSION','role_count':210,'run_ids':ids,'bindings':{}}
 queue={'owner_pid':123,'stage':'H_PRIMARY_CACHE_EXPORT_REQUIRES_INDEPENDENT_ROLE_ADMISSION_COMPLETE','active':[],'detail':{'run_ids':ids}}
 (root/'reports').mkdir();return root,inp,stage,queue

def save_closed(case):
 import json
 root,inp,stage,queue=case;(root/'reports/H_primary_cache_stage_complete.json').write_text(json.dumps(stage));(root/'H_queue_status.json').write_text(json.dumps(queue));return root,inp

def test_closed_primary_gate_matches_all30_models_and210roles(closed_case):
 root,inp=save_closed(closed_case);assert len(q.closed_primary_stage(root,inp))==64

@pytest.mark.parametrize('mutation',['role_count','role_float','active','active_missing','duplicate','queue_stage','queue_model'])
def test_primary_stage_cannot_be_bypassed(closed_case,mutation):
 root,inp,stage,queue=closed_case
 if mutation=='role_count':stage['role_count']=209
 elif mutation=='role_float':stage['role_count']=210.0
 elif mutation=='active':queue['active']=[{'pid':123}]
 elif mutation=='active_missing':queue.pop('active')
 elif mutation=='duplicate':stage['run_ids']=stage['run_ids'][:-1]+stage['run_ids'][:1]
 elif mutation=='queue_stage':queue['stage']='running'
 else:queue['detail']['run_ids']=queue['detail']['run_ids'][:-1]
 root,inp=save_closed(closed_case)
 with pytest.raises(ValueError):q.closed_primary_stage(root,inp)


def test_public_worker_owns_receipt_directory_before_finite_work(tmp_path,monkeypatch):
 import os,json,contextlib
 from types import SimpleNamespace
 root=tmp_path;proof={'nonce':'synthetic','owner_pid':123};run=q.qualification_runs()[0]
 context=SimpleNamespace(root=root,source={'route':'D'},tables={'source_cal':{'rows':1120}},config={},admission_sha='a'*64)
 monkeypatch.setattr(q.lease,'verify_worker',lambda *a,**k:None)
 @contextlib.contextmanager
 def native(*a,**k):yield
 monkeypatch.setattr(q.lease,'native_gpu',native);monkeypatch.setenv('CUDA_VISIBLE_DEVICES','GPU-synthetic')
 monkeypatch.setattr(q,'_plan',lambda *a:({},{}));monkeypatch.setattr(q,'closed_primary_stage',lambda *a:'b'*64);monkeypatch.setattr(q.cache,'prepare_context',lambda *a:context);monkeypatch.setattr(q,'_contended_probe',lambda *a:True)
 monkeypatch.setattr(q,'source_calibration_forward',lambda *a:{'scope':'SYNTHETIC_STUB_NO_ACTUAL_GPU'})
 result=q.qualification_worker(root,'input','a'*64,run,plan_path='plan',plan_sha='c'*64,phase_proof=proof,phase_fd=os.open('/dev/null',os.O_RDONLY))
 out=root/'reports/M_source_only_GPU_qualification/synthetic'/(run+'.json');assert out.exists() and result['M_production_admitted'] is False
 before=out.read_bytes()
 with pytest.raises(ValueError):q.qualification_worker(root,'input','a'*64,run,plan_path='plan',plan_sha='c'*64,phase_proof=proof,phase_fd=os.open('/dev/null',os.O_RDONLY))
 assert out.read_bytes()==before


@pytest.mark.parametrize('mutation',['signed_zero','second_dtype','cache_signed_zero'])
def test_bitwise_qualification_rejects_value_equal_but_byte_different(project,monkeypatch,mutation):
 import torch
 from test_source_best_cache import Encoder
 c=cache.prepare_context(project['root'],project['admission'],project['admission_sha'],project['run']);calls=[0]
 def forward(self,x):
  calls[0]+=1;out=torch.zeros((len(x),512),dtype=torch.float32)
  if mutation=='signed_zero' and calls[0]%2==0:return -out
  if mutation=='second_dtype' and calls[0]%2==0:return out.double()
  return out
 if mutation=='cache_signed_zero':
  monkeypatch.setattr(Encoder,'forward',lambda self,x:-torch.zeros((len(x),512),dtype=torch.float32));q.source_calibration_forward(c,'cpu',lambda full=False:None)
 monkeypatch.setattr(Encoder,'forward',forward)
 with pytest.raises(ValueError):q.source_calibration_forward(c,'cpu',lambda full=False:None)
