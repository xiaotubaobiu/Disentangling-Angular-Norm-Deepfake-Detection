from pathlib import Path
import ast,copy,contextlib,hashlib,importlib.util,json,os
import pytest
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');P=Path(__file__).parents[1]/'tools/h_last_resumed_result_audit.py'
def module():
 assert P.is_file(),'resume-aware consumer is not implemented'
 s=importlib.util.spec_from_file_location('resume_consumer_test',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def real_refs(L):
 cmd=L['command'];q=cmd[cmd.index('--qualification')+1];c=cmd[cmd.index('--admission')+1]
 return dict(primary_launch=dict(path=str(R/f'reports/H_primary_analysis_{L["backbone"]}_seed{L["seed"]}_resume_20261026T013643Z_launch.json'),sha256={'clip_vit_b16':'0d3a89013786d51824a4b8679f3e2ee57c0678c2370faef330c487591b6cf91b','dinov2_vit_b14':'ea89f76b78671e6f942862d4247dd9524d61a8332151db9992a4ed6bb83ea113'}[L['backbone']]),primary_entry=dict(path=L['entry'],sha256=L['entry_sha256']),CPU_qualification=dict(path=q,sha256=L['qualification_sha256']),cache_admission=dict(path=c,sha256=L['admission_sha256']),cache_audit=dict(path=L['independent_cache_audit'],sha256=L['independent_cache_audit_sha256']),primary_stdout=dict(path=L['stdout'],sha256='a'*64),primary_time_v=dict(path=L['time_v'],sha256='b'*64))

@pytest.mark.parametrize('bone,seed',[('clip_vit_b16',3407),('dinov2_vit_b14',7)])
def test_actual_resumed_JSON_explicitly_qualified_without_field_synthesis(bone,seed):
 m=module();path=R/f'reports/H_primary_analysis_{bone}_seed{seed}_resume_20261026T013643Z_launch.json';L=json.loads(path.read_text());before=copy.deepcopy(L);info=m.validate_resume_launch(L,bone,seed,real_refs(L));assert L==before and 'M_statistics_started' not in L;assert info['M_statistics_started_field_present'] is False and info['missing_field_not_synthesized'] is True;assert len(info['evidence_sha256'])>=4

@pytest.mark.parametrize('change',['status','M_false_inserted','M_true','command','slot','bootstrap','auth','terminal_resume','fresh41','path','rawSHA'])
def test_bound_resume_record_changes_never_authorized(change):
 m=module();bone,seed='dinov2_vit_b14',7;L=json.loads((R/f'reports/H_primary_analysis_{bone}_seed{seed}_resume_20261026T013643Z_launch.json').read_text());refs=real_refs(L)
 if change=='status':L['status']='LAUNCHED_ORIGINAL_PRIMARY_H_PAIR_STATISTICS_AFTER_INDEPENDENT_CACHE_ADMISSION'
 elif change=='M_false_inserted':L['M_statistics_started']=False
 elif change=='M_true':L['M_statistics_started']=True
 elif change=='command':L['command'][-1]='0'*64
 elif change=='slot':L['CPU_slot']=1
 elif change=='bootstrap':L['bootstrap_repeats']=1999
 elif change=='auth':L['resume_authorization']['sha256']='0'*64
 elif change=='terminal_resume':L['resume_existing_terminal']=True
 elif change=='fresh41':L['expected_fresh_original_ledger_invocations']=41
 elif change=='path':refs['primary_launch']['path']=refs['primary_launch']['path'].replace('013643Z','013644Z')
 else:refs['primary_launch']['sha256']='0'*64
 with pytest.raises((AssertionError,KeyError,ValueError)):m.validate_resume_launch(L,bone,seed,refs)

def test_derived_adapter_keeps_numeric_orchestration_AST():
 m=module();assert m.ADAPTATION_SITES=={'validate_chain_launch_gate':1,'load_spec_launch_path_gate':1,'metadata_evidence_return':1,'main_publication_source_guard':1};assert m.ORIGINAL_AUDIT_AST==ast.dump(next(n for n in ast.parse(m.BB_PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='audit_owned'),include_attributes=False);assert m.main([])['production_audit_started'] is False

def fixture(tmp_path,monkeypatch,case='baseline',resume=False):
 path=R/'engineering/source_best_M/audits/independent_h_last_matrix_public_fixture_v2.py';s=importlib.util.spec_from_file_location('old_public_fixture',path);r=importlib.util.module_from_spec(s);s.loader.exec_module(r);r.OUT=tmp_path;r.SRC=P;f=r.Fixture(case,'dinov2_vit_b14');m=f.m
 source=m.BB_PATH;copied=f.root/'tools/original_bb299e.py';copied.write_bytes(source.read_bytes());m.BB_PATH=copied
 entries=[]
 @contextlib.contextmanager
 def counted_no_lock(*args):entries.append(tuple(map(str,args)));yield
 monkeypatch.setattr(m.base,'cpu_slot',counted_no_lock)
 if resume:
  stamp='20261026T013643Z';stem=f'H_primary_analysis_{f.bone}_seed123_resume_{stamp}';L=m._json(f.refs['primary_launch']);L.update(status='RESUMED_ORIGINAL_PRIMARY_H_PAIR_STATISTICS_FROM_UNFINISHED_USER_PAUSE',utc='2026-10-26T01:36:44+00:00',expected_fresh_original_ledger_invocations=42,resume_existing_terminal=False,unfinished_partial_identity_preserved=True,independent_numeric_bootstrap_acceptance_pending=True);L.pop('M_statistics_started')
  f.refs['primary_launch']=f.root/'reports'/f'{stem}_launch.json'
  for key in ('primary_stdout','primary_time_v'):
   dest=f.root/'reports'/(stem+('_time_v.log' if key=='primary_time_v' else '.log'));dest.write_bytes(f.refs[key].read_bytes());f.refs[key]=dest
  L.update(stdout=str(f.refs['primary_stdout']),time_v=str(f.refs['primary_time_v']));f.primary_log=f.refs['primary_stdout'];f.primary_time=f.refs['primary_time_v']
  pause=f.root/'reports/pause.json';marker=f.root/'reports/retained_pause.json';auth=f.root/'resume_authorization.json'
  pause_body=dict(schema='DETECTOR_V2_USER_PAUSED_VERIFIED_V1',state='PAUSED_BY_USER_NO_AUTOMATIC_RESUME',user_instruction_utc='2026-10-25T23:51:00Z',observed_utc='2026-10-25T23:57:12+00:00',dispatch_allowed=False,automatic_recovery_allowed=False,resume_requires_new_explicit_user_authorization=True,live_task_work_PIDs=[],in_flight_workers=0,retained_progress=dict(H_CPU_statistical_partial_outputs={f'{f.bone}/seed123':dict(complete_exists=False,no_files_removed=True)}),CPU_interrupt_logs={'DINO123':dict(retained_partial_only=True,interrupted_by_user_pause=True,scientifically_accepted=False,stdout=dict(path=str(f.root/'reports/old.log'),sha256=r.dump(f.root/'reports/old.log',{'interrupted':True})),time_v=dict(path=str(f.root/'reports/old_time.log'),sha256=r.dump(f.root/'reports/old_time.log',{'exit':'interrupted'})))})
  m.PAUSE_REF=dict(path=str(pause),sha256=r.dump(pause,pause_body));m.RETAINED_PAUSE_REF=dict(path=str(marker),sha256=r.dump(marker,dict(schema='DETECTOR_V2_USER_PAUSE_MARKER_V1',state='PAUSED_BY_USER_NO_AUTOMATIC_RESUME',no_new_dispatch=True,no_automatic_resume=True,requires_new_explicit_user_authorization=True,evidence=m.PAUSE_REF)))
  m.AUTHORIZATION_REF=dict(path=str(auth),sha256=r.dump(auth,dict(schema='DETECTOR_V2_EXPLICIT_RESUME_V1',utc='2026-10-26T01:36:43+00:00',scope='DETECTOR_ONLY_ORIGINAL_FULL_EXPERIMENT_AND_CN_EN_PAPERS',prior_pause_retained=m.RETAINED_PAUSE_REF,pause_verification=m.PAUSE_REF,user_explicit_resume_UTC='2026-10-26T01:33:00Z',JEV_GLM_MMTT_pair_investigation_remain_paused=True,GPUs_allowed=[0,1,2,3],new_scheduler_created=False,state='RESUMING_INDEPENDENT_H_CPU_FIRST_M_BUDGET_RECHECK_PENDING')))
  L['resume_authorization']=m.AUTHORIZATION_REF;L['command']=['/usr/bin/time','-v','-o',L['time_v'],str(f.root/'environment/bin/python'),'-B','-u',L['entry'],'--execute','--backbone',f.bone,'--seed','123','--slot',str(L['CPU_slot']),'--qualification',str(f.refs['CPU_qualification']),'--qualification-sha',L['qualification_sha256'],'--admission',str(f.refs['cache_admission']),'--admission-sha',L['admission_sha256']];r.dump(f.refs['primary_launch'],L);m.RESUME_CASES={(f.bone,123):dict(path=str(f.refs['primary_launch']),sha256=r.H(f.refs['primary_launch']),slot=0,pause_tag='DINO123')}
  b=m._json(f.bp)
  for key in ('primary_launch','primary_stdout','primary_time_v'):b['proofs'][key]=dict(path=str(f.refs[key]),sha256=r.H(f.refs[key]))
  r.dump(f.bp,b)
 return f,entries

@pytest.mark.parametrize('resume',[False,True])
def test_public_normal_and_explicit_resume_preserve_all8_numeric_cohorts(tmp_path,monkeypatch,resume):
 f,entries=fixture(tmp_path,monkeypatch,resume=resume);out=f.execute();d=f.m._json(out['output']);assert len(entries)==1 and len(f.calls)==8 and len(d['cohorts'])==8;assert d['consumer_source_sha256']==f.m.ENTRY_SHA and d['original_bb299e_source_sha256']==f.m.BB_SHA;assert d['launch_fact_route']==('explicit_bound_user_resume' if resume else 'original_normal_launch')

@pytest.mark.parametrize('case',['producer_count41','producer_complete_first','producer_exit1_only','missing_P_base_pin','wrong_Q_actual_evidence'])
def test_public_resumed_route_preserves_original_terminal_and_proof_failures(tmp_path,monkeypatch,case):
 f,entries=fixture(tmp_path,monkeypatch,case,resume=True)
 with pytest.raises(AssertionError):f.execute()
 assert not (f.m.HERE/'H_LAST_dinov2_vit_b14_seed123_independent_audit.json').exists()


# Safe replacement for the rejected file-mutation probe: these tests do not call
# the filesystem-copy fixture and never write __file__, ENTRY_PATH or BB_PATH.
# Only digest reads and in-memory private-module callbacks are mocked.
@pytest.mark.parametrize('which',['consumer','original_bb'])
def test_mock_late_audit_digest_drift_rejected_without_source_writes(monkeypatch,which):
 m=module();actual=m.sha;late=[];target=Path(m.__file__) if which=='consumer' else m.BB_PATH
 monkeypatch.setattr(m,'_derived_audit_owned',lambda *args:dict(status='SYNTHETIC_ONLY'))
 def route(spec):late.append(True);return dict(launch_fact_route='original_normal_launch')
 monkeypatch.setattr(m,'_route_info',route)
 def digest(path):return '0'*64 if late and Path(path)==target else actual(path)
 monkeypatch.setattr(m,'sha',digest)
 with pytest.raises(AssertionError,match='source.*(drift|digest)'):m.audit_owned({},'synthetic_terminal',1)
 assert actual(Path(m.__file__))==m.ENTRY_SHA and actual(m.BB_PATH)==m.BB_SHA

@pytest.mark.parametrize('which',['consumer','original_bb'])
def test_mock_publication_digest_drift_rejected_without_source_writes(tmp_path,monkeypatch,which):
 m=module();actual=m.sha;target=Path(m.__file__) if which=='consumer' else m.BB_PATH;binding=tmp_path/'binding.json';binding.write_text('{}');pin=actual(binding);late=[];entries=[];m.HERE=tmp_path
 monkeypatch.setattr(m,'load_spec',lambda *args:dict(backbone='dinov2_vit_b14',seed=123))
 monkeypatch.setattr(m,'audit_owned',lambda *args:dict(status='SYNTHETIC_ONLY'))
 @contextlib.contextmanager
 def no_lock(*args):entries.append(True);yield
 monkeypatch.setattr(m.base,'cpu_slot',no_lock)
 for k,v in [('CUDA_VISIBLE_DEVICES',''),('OMP_NUM_THREADS','1'),('MKL_NUM_THREADS','1'),('OPENBLAS_NUM_THREADS','1')]:monkeypatch.setenv(k,v)
 def digest(path):
  if Path(path)==binding:late.append(True);return pin
  return '0'*64 if late and Path(path)==target else actual(path)
 monkeypatch.setattr(m,'sha',digest)
 with pytest.raises(AssertionError,match='source.*(drift|digest)'):m.main(['--execute','--binding-spec',str(binding),'--binding-spec-sha',pin,'--terminal-sha','a'*64,'--slot','1'])
 assert len(entries)==1 and not (tmp_path/'H_LAST_dinov2_vit_b14_seed123_independent_audit.json').exists()
 assert actual(Path(m.__file__))==m.ENTRY_SHA and actual(m.BB_PATH)==m.BB_SHA


def test_source_attestation_binds_initial_and_observed_digests_without_source_writes(monkeypatch):
 m=module();monkeypatch.setattr(m,'_derived_audit_owned',lambda *args:dict(status='SYNTHETIC_ONLY'));monkeypatch.setattr(m,'_route_info',lambda *args:dict(launch_fact_route='original_normal_launch'));d=m.audit_owned({},'synthetic_terminal',1)
 assert d['source_attestation_initial']==d['source_attestation_after_audit']
 assert d['source_attestation_initial']['consumer']==dict(path=str(Path(m.__file__)),initial_sha256=m.ENTRY_SHA,observed_sha256=m.ENTRY_SHA)
 assert d['source_attestation_initial']['original_bb']==dict(path=str(m.BB_PATH),initial_sha256=m.BB_SHA,observed_sha256=m.BB_SHA)
