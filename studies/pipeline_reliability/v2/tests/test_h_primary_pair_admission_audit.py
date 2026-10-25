import copy,hashlib,importlib.util,json,os
from pathlib import Path
import pytest
P=Path(__file__).parents[1]/'tools/h_primary_pair_admission_audit.py'
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
def module():
 assert P.is_file(), 'new finite metadata adapter is not implemented'
 s=importlib.util.spec_from_file_location('primary_pair_metadata_test',P);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def real_family(m,bone):
 old=m.helpers.old;ref=m.helpers.read(old.REFERENCE);pins=m.helpers.reconstruct_pins(bone,ref,m.helpers.read(old.MANIFEST),m.helpers.read(R/'H_run_registry.json'),m.helpers.read(old.ALL3_ACCEPTED));return ref,pins
def real_Q_inputs(m,bone):
 spec=m.QUALIFICATIONS[bone];Q=m.read(spec['path']);C=m.read(Q['candidate_report']);A=m.read(Q['independent_actual_audit']);ER=m.read(Q['primary_entry_review']);ref,pins=real_family(m,bone);return Q,C,A,ER,ref,pins

def test_preparation_has_no_authority():
 m=module();d=m.main([]);assert d['preparation_only'] is True and d['production_admission_started'] is False;assert d['future_wrapper_format']=='H_PRIMARY_PAIR_ADMISSION_TEMPLATE_V1';assert d['qualified_backbones']==['clip_vit_b16','dinov2_vit_b14']

@pytest.mark.parametrize('bone,seed,mode,allowed',[(b,s,'qualified_pair',b!='efficientnet_b0') for b in ('clip_vit_b16','dinov2_vit_b14','efficientnet_b0') for s in (7,42,123,2026,3407)]+[('dinov2_vit_b14',7,'actual_qualification_prototype',True),('efficientnet_b0',7,'actual_qualification_prototype',True),('clip_vit_b16',7,'actual_qualification_prototype',False),('dinov2_vit_b14',42,'actual_qualification_prototype',False),('efficientnet_b0',True,'actual_qualification_prototype',False)])
def test_finite_mode_scope(bone,seed,mode,allowed):
 m=module()
 if allowed:assert m.validate_mode(bone,seed,mode) is None
 else:
  with pytest.raises((AssertionError,ValueError)):m.validate_mode(bone,seed,mode)

@pytest.mark.parametrize('bone,label',[('clip_vit_b16','CLIP'),('dinov2_vit_b14','DINO'),('efficientnet_b0','EFF')])
@pytest.mark.parametrize('seed',[7,42,123,2026,3407])
@pytest.mark.parametrize('slot',[0,1])
def test_template_is_exact_finite_substitution(bone,label,seed,slot):
 m=module();base=m.read(R/'reports/H_primary_DINO_CPU_qualification_candidate_seed7_slot0_20261025T2030_launch.json')['parameterized_entry_code'];expected=base.replace('dinov2_vit_b14',bone).replace('DINO',label).replace('seed7',f'seed{seed}').replace("'--seed','7'",f"'--seed','{seed}'").replace('entry.cpu_slot(0)',f'entry.cpu_slot({slot})');assert m.generated_wrapper_code(bone,seed,slot)==expected

@pytest.mark.parametrize('bone,seed,slot',[('clip_vit_b16',True,0),('dinov2_vit_b14',8,0),('efficientnet_b0',7,True),('dinov2_vit_b14',7,2),('foreign',7,0)])
def test_invalid_template_parameter_rejected(bone,seed,slot):
 m=module()
 with pytest.raises((AssertionError,ValueError)):m.generated_wrapper_code(bone,seed,slot)

@pytest.mark.parametrize('name',['H_primary_admit_clip_seed3407_slot1_20261025T2000_launch.json','H_primary_DINO_CPU_qualification_candidate_seed7_slot0_20261025T2030_launch.json','H_primary_EFF_CPU_qualification_candidate_seed7_slot0_20261025T2155_launch.json'])
def test_existing_three_actual_launch_metadata(name):
 m=module();L=m.read(R/'reports'/name);assert m.validate_launch(L,L['backbone'],L['seed'],R/'reports'/name) is None

def future_launch(m,bone='dinov2_vit_b14',seed=123,slot=1):
 stem=f'H_primary_admit_{bone}_seed{seed}_slot{slot}_20261025T230000Z';code=m.generated_wrapper_code(bone,seed,slot);return dict(wrapper_format='H_PRIMARY_PAIR_ADMISSION_TEMPLATE_V1',attempt_utc='20261025T230000Z',backbone=bone,seed=seed,CPU_slot=slot,pid=999999991,parameterized_entry_code=code,parameterized_code_sha256=hashlib.sha256(code.encode()).hexdigest(),template_original_DINO_launch=str(R/'reports/H_primary_DINO_CPU_qualification_candidate_seed7_slot0_20261025T2030_launch.json'),template_original_DINO_launch_sha256='d77d73e3ab0c4f3ec6d9338f842990df9db3b85044074b3443fdce968fa1a46a',entry=str(R/'qualification/H_parallel_fresh_validation/cpu_ledger_admission.py'),entry_sha256='bd202e85a4990f9246b96a7081815d07cf364041e5d3498f49bfe41d8779b213',stdout=str(R/'reports'/f'{stem}.log'),time_v=str(R/'reports'/f'{stem}_time_v.log'),candidate=str(R/f'reports/H_{bone}_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json'),new_dispatcher=False,repeat_on_failure=False,production_or_F_files_changed=False,primary_scientific_scoring_started=False)

def test_future_launch_contract_positive():
 m=module();L=future_launch(m);m.validate_launch(L,'dinov2_vit_b14',123,R/'reports/H_primary_admit_dinov2_vit_b14_seed123_slot1_20261025T230000Z_launch.json')

@pytest.mark.parametrize('field,value',[('seed',42),('CPU_slot',0),('CPU_slot',True),('parameterized_entry_code','print(1)'),('template_original_DINO_launch_sha256','0'*64),('new_dispatcher',True),('candidate','/tmp/foreign.json')])
def test_future_launch_parameter_or_code_drift_rejected(field,value):
 m=module();L=future_launch(m);L[field]=value
 with pytest.raises((AssertionError,ValueError)):m.validate_launch(L,'dinov2_vit_b14',123,R/'reports/H_primary_admit_dinov2_vit_b14_seed123_slot1_20261025T230000Z_launch.json')

@pytest.mark.parametrize('bone',['clip_vit_b16','dinov2_vit_b14'])
def test_real_own_qualification_chain_metadata_only(bone):
 m=module();Q,C,A,ER,ref,pins=real_Q_inputs(m,bone);m.validate_qualification_body(bone,Q,C,A,ER,ref,pins);extra={};got=m.validate_qualification(bone,dict(path=m.QUALIFICATIONS[bone]['path'],sha256=m.QUALIFICATIONS[bone]['sha256']),ref,pins,extra);assert got==Q;assert len(extra)>=4;assert all(m.sha(p)==h for p,h in extra.items())

@pytest.mark.parametrize('mutation',['wrongbone','bool140','baduse','badsource','badimpl','FAILED_aux','CLIP_aux','bad_identity','bool_audit140','bad_pin','bad_entry'])
def test_semantic_qualification_bad_bodies_rejected(mutation):
 m=module();Q,C,A,ER,ref,pins=real_Q_inputs(m,'dinov2_vit_b14')
 if mutation=='wrongbone':Q['allowed_backbones']=['clip_vit_b16']
 elif mutation=='bool140':Q['fresh_default_source_checks']=True
 elif mutation=='baduse':Q['allowed_use']=['independent_role_admission']
 elif mutation=='badsource':Q['original_source_manifest_sha256']='0'*64
 elif mutation=='badimpl':Q['implementation_sha256']['parallel_source.py']='0'*64
 elif mutation=='FAILED_aux':A['status']='FAILED'
 elif mutation=='CLIP_aux':A['backbone']='clip_vit_b16'
 elif mutation=='bad_identity':A['original_primary_pair_identity']['seed']=42
 elif mutation=='bool_audit140':A['fresh_default_source_checks_in_ledgers']=True
 elif mutation=='bad_pin':A['pinned_sha256'].pop(next(iter(A['pinned_sha256'])))
 else:ER['engineering_entry_review_passed']=False
 with pytest.raises((AssertionError,ValueError,KeyError)):m.validate_qualification_body('dinov2_vit_b14',Q,C,A,ER,ref,pins)

@pytest.mark.parametrize('delta',[-1,1])
def test_nearby_birth_tick_never_proves_live_exit(delta):
 m=module();pid=os.getpid();tick=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]);import datetime
 w=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_process_tree=[dict(pid=pid,start_ticks=tick+delta)])
 with pytest.raises(AssertionError):m.helpers.validate_exit(dict(pid=pid),w)


# Public surrogate-root tests. Source/F/harness code bytes are retained; only
# fixture metadata anchors and absolute namespaces are retargeted. Every slot
# context is explicitly a non-acquiring counter. No models/images/chunks exist.
def public_fixture(tmp_path,monkeypatch,bone='dinov2_vit_b14',mode='qualified_pair',future=False):
 import contextlib,datetime
 archive=R/'engineering/source_best_M/audits/independent_h_parallel_matrix_admission_public_fixture.py'
 text=archive.read_text().split("\nif __name__=='__main__':")[0]
 text=text.replace('sp.loader.exec_module(m)','sp.loader.exec_module(m);m.HARNESS[("efficientnet_b0",7)]=("261e6ecebc24990098ecaad3d961304c5db5bfe262ced047f8e01c3f9b532057",0,"H_primary_EFF_CPU_qualification_candidate_seed7_slot0_20261025T2155")')
 ns=dict(__file__=str(archive),__name__='fixture_only');exec(compile(text,str(archive),'exec'),ns);ns['OUT']=tmp_path;ns['SOURCE']=R/'engineering/source_best_M/qualification_snapshots/H_matrix_admission_a65b198602e66662de005c6162563f7286041808c9cafd3b4630b1755c0e3196/entry.py';seed=3407 if bone=='clip_vit_b16' else 7
 h,root,b,_,names=ns['fixture']('generic_fixture',bone,seed);g=module();g.ROOT=root;g.E=root/'engineering/source_best_M';g.HERE=g.E/'audits';g.helpers=h;g.old=h.old
 # Fresh isolated module objects only; no production globals or files are changed.
 original_helper=g.HELPER_PATH;g.HELPER_PATH=root/'qualified_a65/entry.py';g.HELPER_PATH.parent.mkdir();g.HELPER_PATH.write_bytes(original_helper.read_bytes());original_bb=g.BB_PATH;g.BB_PATH=root/'qualified_bb.py';g.BB_PATH.write_bytes(original_bb.read_bytes());g.TEMPLATE_PATH=root/'reports/H_primary_DINO_CPU_qualification_candidate_seed7_slot0_20261025T2030_launch.json'
 if not g.TEMPLATE_PATH.exists():g.TEMPLATE_PATH.write_bytes((R/'reports'/g.TEMPLATE_PATH.name).read_bytes())
 g.TEMPLATE_SHA=g.sha(g.TEMPLATE_PATH)
 digest=g.sha(P);snapshot=g.E/f'qualification_snapshots/H_primary_pair_admission_{digest}/entry.py';snapshot.parent.mkdir(parents=True);snapshot.write_bytes(P.read_bytes());g.__file__=str(snapshot)
 C=g.read(names['candidate']);ref=g.read(h.old.REFERENCE);expected=C['pinned_sha256'];baseC=copy.deepcopy(C)
 if future:
  seed=123;newmarkers={}
  for history,run in C['identity']['run_ids'].items():
   newrun=f'H_formal_{bone}_{history}_seed{seed}';rr=root/'runs'/newrun;seal=g.read(rr/'H_completion_seal.json')
   for role in h.old.ROLES:
    d=g.read(root/'features/H'/run/role/'complete.json');v=d['identity'];v.update(run_id=newrun,checkpoint_path=str(rr/'final.pt'),checkpoint_sha256=seal['artifact_sha256']['final.pt'],training_complete_sha256=expected[str(rr/'complete.json')],completion_seal_sha256=expected[str(rr/'H_completion_seal.json')]);dest=root/'features/H'/newrun/role/'complete.json';newmarkers[history+'/'+role]=ns['dump'](dest,d)
  C.update(seed=seed,identity=h.expected_identity(bone,seed,expected,newmarkers,ref));code=g.generated_wrapper_code(bone,seed,1);stamp='20261025T230000Z';stem=f'H_primary_admit_{bone}_seed{seed}_slot1_{stamp}';L=g.read(names['launch']);L.update(wrapper_format='H_PRIMARY_PAIR_ADMISSION_TEMPLATE_V1',attempt_utc=stamp,seed=seed,CPU_slot=1,parameterized_entry_code=code,parameterized_code_sha256=hashlib.sha256(code.encode()).hexdigest(),template_original_DINO_launch=str(g.TEMPLATE_PATH),template_original_DINO_launch_sha256=g.TEMPLATE_SHA,entry=str(h.old.F/'cpu_ledger_admission.py'),entry_sha256=g.CPU_HASHES['cpu_ledger_admission.py'],primary_scientific_scoring_started=False)
  names=dict(launch=root/'reports'/f'{stem}_launch.json',candidate=root/f'reports/H_{bone}_seed{seed}_independent_7role_parallel_candidate_20261025T0645.json',stdout=root/'reports'/f'{stem}.log',time_v=root/'reports'/f'{stem}_time_v.log',process_witness=g.HERE/f'{stem}_actual_producer_process_witness.json');L.update(stdout=str(names['stdout']),time_v=str(names['time_v']),candidate=str(names['candidate']));ns['dump'](names['launch'],L);ns['dump'](names['candidate'],C);b=g.HERE/f'{stem}_matrix_admission_bindings.json'
  lines=[f'BEGIN_FRESH_ORIGINAL_LEDGER {i} {bone} AMP models10_no_receipt_reuse' for i in range(1,15)]+['PASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT '+str(names['candidate'])];names['stdout'].write_text('\n'.join(lines)+'\n');names['time_v'].write_text('Elapsed (wall clock) time (h:mm:ss or m:ss): 0:20.1\nMaximum resident set size (kbytes): 10000\nExit status: 0\n');boot=int(next(x.split()[1] for x in Path('/proc/stat').read_text().splitlines() if x.startswith('btime ')));tick=int((datetime.datetime.now(datetime.timezone.utc).timestamp()-boot)*os.sysconf('SC_CLK_TCK'));L['utc']=datetime.datetime.fromtimestamp(boot+tick/os.sysconf('SC_CLK_TCK'),datetime.timezone.utc).isoformat();ns['dump'](names['launch'],L);child=str(root/'environment/bin/python')+' -B -u -c '+code+' ';ns['dump'](names['process_witness'],dict(schema='H_FINITE_PRODUCER_ACTUAL_PROCESS_WITNESS_V1',read_only=True,launch=str(names['launch']),launch_sha256=g.sha(names['launch']),wrapper_pid=L['pid'],utc=L['utc'],observed_process_tree=[dict(pid=L['pid'],start_ticks=tick,state='S',cmdline='/usr/bin/time -v -o '+L['time_v']+' '+child),dict(pid=99999102,start_ticks=tick,state='R',cmdline=child)]))
 elif bone=='efficientnet_b0':
  L=g.read(names['launch']);L['template_original_DINO_launch']=str(g.TEMPLATE_PATH);L['template_original_DINO_launch_sha256']=g.TEMPLATE_SHA;ns['dump'](names['launch'],L);w=g.read(names['process_witness']);w['launch_sha256']=g.sha(names['launch']);ns['dump'](names['process_witness'],w)
 refs={k:dict(path=str(v),sha256=g.sha(v)) for k,v in names.items()}
 if mode=='qualified_pair':
  spec=copy.deepcopy(g.QUALIFICATIONS[bone]);proto=spec['prototype_seed'];Cq=copy.deepcopy(baseC);Cq.update(seed=proto,identity=h.expected_identity(bone,proto,expected,baseC['identity']['cache_completion_sha256'],ref));cqp=root/f'reports/H_{bone}_seed{proto}_independent_7role_parallel_candidate_20261025T0645.json';ns['dump'](cqp,Cq)
  aqp=h.old.F/('independent_actual_CPU_parallel_pair_admission_audit.json' if bone=='clip_vit_b16' else 'independent_actual_CPU_parallel_pair_admission_audit_dinov2_vit_b14.json');Aq=copy.deepcopy(ref);Aq.update(backbone=bone,seed=proto,receipt_path=str(cqp),receipt_sha256=g.sha(cqp),receipt_status=Cq['status'],original_primary_pair_identity=Cq['identity'],pinned_sha256=expected,functional_allowed_backbones=[bone],intended_functional_validator_use=['independent_role_admission','primary_pair_analysis'],implementation_sha256=copy.deepcopy(g.CPU_HASHES));ns['dump'](aqp,Aq)
  if aqp==h.old.REFERENCE:h.old.REFERENCE_SHA=g.sha(aqp)
  erp=h.old.F/'independent_cpu_primary_analysis_entry_fix_confirmation.json';erp.write_bytes((R/'qualification/H_parallel_fresh_validation'/erp.name).read_bytes());qpath=root/'reports'/Path(spec['path']).name;Q=g.read(spec['path']);Q.update(candidate_report=str(cqp),candidate_report_sha256=g.sha(cqp),independent_actual_audit=str(aqp),independent_actual_audit_sha256=g.sha(aqp),primary_entry_review=str(erp),primary_entry_review_sha256=g.sha(erp));ns['dump'](qpath,Q);spec.update(path=str(qpath),sha256=g.sha(qpath),candidate_sha256=g.sha(cqp),audit_path=str(aqp),audit_sha256=g.sha(aqp));g.QUALIFICATIONS=copy.deepcopy(g.QUALIFICATIONS);g.QUALIFICATIONS[bone]=spec;refs['CPU_qualification']=dict(path=str(qpath),sha256=g.sha(qpath));refs['candidate']['sha256']=g.sha(names['candidate'])
 b=g.HERE/f'{names["launch"].name.removesuffix("_launch.json")}_{mode}_matrix_admission_bindings.json'
 ns['dump'](b,dict(schema='H_MATRIX_ADMISSION_EXTERNAL_BINDINGS_V1',root=str(root),backbone=bone,seed=seed,proofs=refs))
 entries=[]
 @contextlib.contextmanager
 def no_lock(*args):entries.append(tuple(map(str,args)));yield
 monkeypatch.setattr(h.old.SHARED,'cpu_slot',no_lock)
 return g,b,names,entries,ns['dump']

def run_public(g,b,mode):return g.main(['--execute','--binding',str(b),'--binding-sha',g.sha(b),'--mode',mode,'--slot','1'])

@pytest.mark.parametrize('bone,mode,future',[('clip_vit_b16','qualified_pair',False),('dinov2_vit_b14','qualified_pair',False),('dinov2_vit_b14','qualified_pair',True),('dinov2_vit_b14','actual_qualification_prototype',False),('efficientnet_b0','actual_qualification_prototype',False)])
def test_public_main_full_metadata_surrogate_positive(tmp_path,monkeypatch,bone,mode,future):
 g,b,names,entries,dump=public_fixture(tmp_path,monkeypatch,bone,mode,future);result=run_public(g,b,mode);d=g.read(result['output']);assert len(entries)==1 and len(d['canonical_shared107_pin_sha256'])==107 and len(d['role_marker_checks'])==14 and len(d['role_manifest_footer_metadata'])==7;assert result['CPU_qualification_issued'] is False and d['source_checks_in_ledger_receipts']==140;assert d['bracketed_readonly_evidence_sha256'][g.__file__]==g.sha(g.__file__);assert d['production_GPU_allowed'] is False and d['M_authorized'] is False
 if mode=='qualified_pair':assert g.QUALIFICATIONS[bone]['path'] in d['bracketed_readonly_evidence_sha256']
 if future:assert Path(d['producer_process_witness_path']).name.startswith('H_primary_admit_dinov2_vit_b14_seed123_slot1_20261025T230000Z')

@pytest.mark.parametrize('case',['missing_Q','wrong_Q_SHA','foreign_Q_path','wrongbone_Q','FAILED_aux','CLIP_aux','bool_aux140','foreign_aux_namespace','bool140','marker_dimension','marker_head','frame_count','source_manifest','footer','conflicting_pin','late_PID','before_output_drift','wrong_snapshot','wrong_attempt','witness_wrong_code','stdout_early_complete','time_exit1'])
def test_public_main_fails_closed_and_no_positive_output(tmp_path,monkeypatch,case):
 qualified=case in ('missing_Q','wrong_Q_SHA','foreign_Q_path','wrongbone_Q','FAILED_aux','CLIP_aux','bool_aux140','foreign_aux_namespace');mode='qualified_pair' if qualified else 'actual_qualification_prototype';g,b,names,entries,dump=public_fixture(tmp_path,monkeypatch,mode=mode);bd=g.read(b);out=g.HERE/('H_DINO_seed7_parallel_cache_admission_independent_audit.json' if qualified else 'H_DINO_seed7_actual_CPU_qualification_metadata_audit.json')
 if qualified:
  qr=bd['proofs']['CPU_qualification'];Q=g.read(qr['path']);spec=g.QUALIFICATIONS['dinov2_vit_b14']
  if case=='missing_Q':bd['proofs'].pop('CPU_qualification')
  elif case=='wrong_Q_SHA':qr['sha256']='0'*64
  elif case=='foreign_Q_path':qr['path']=str(tmp_path/'outside.json')
  else:
   if case=='wrongbone_Q':Q['allowed_backbones']=['clip_vit_b16']
   elif case=='foreign_aux_namespace':Q['independent_actual_audit']=str(tmp_path/'outside.json')
   else:
    aux=g.read(Q['independent_actual_audit'])
    if case=='FAILED_aux':aux['status']='FAILED'
    elif case=='CLIP_aux':aux['backbone']='clip_vit_b16'
    else:aux['fresh_default_source_checks_in_ledgers']=True
    dump(Q['independent_actual_audit'],aux);Q['independent_actual_audit_sha256']=g.sha(Q['independent_actual_audit']);spec['audit_sha256']=Q['independent_actual_audit_sha256']
   dump(qr['path'],Q);qr['sha256']=g.sha(qr['path']);spec['sha256']=qr['sha256']
 elif case=='bool140':
  C=g.read(names['candidate']);C['all_source_completions_fresh']=True;dump(names['candidate'],C);bd['proofs']['candidate']['sha256']=g.sha(names['candidate'])
 elif case in ('marker_dimension','marker_head','frame_count','source_manifest'):
  C=g.read(names['candidate']);run=C['identity']['run_ids']['B'];role='B/SOURCE_CAL';p=g.ROOT/'features/H'/run/role/'complete.json';d=g.read(p)
  if case=='marker_dimension':d['dimension']=512
  elif case=='marker_head':d['identity']['original_head_sha256']='0'*64
  elif case=='frame_count':d['rows']+=1;d['identity']['accepted_table_rows']+=1
  else:d['identity']['source_manifest_path']='/tmp/foreign.parquet'
  C['identity']['cache_completion_sha256']['B/'+role]=dump(p,d);dump(names['candidate'],C);bd['proofs']['candidate']['sha256']=g.sha(names['candidate'])
 elif case=='footer':
  import pyarrow as pa,pyarrow.parquet as pq
  pq.write_table(pa.table({'video_uid':[0]*88}),g.ROOT/'data/H/B/SOURCE_CAL.parquet')
 elif case=='conflicting_pin':
  r=g.read(g.old.REFERENCE);r['protected_review_sha256'][str(g.old.F/'cpu_primary_analysis.py')]='0'*64;dump(g.old.REFERENCE,r);g.old.REFERENCE_SHA=g.sha(g.old.REFERENCE)
 elif case=='late_PID':
  real=g.helpers.validate_exit;calls=[]
  def late(*args):
   calls.append(True)
   if len(calls)>1:raise AssertionError('late original producer alive')
   return real(*args)
  monkeypatch.setattr(g.helpers,'validate_exit',late)
 elif case=='before_output_drift':
  real=g.json.dumps
  def drift(value,*args,**kwargs):
   text=real(value,*args,**kwargs)
   if isinstance(value,dict) and value.get('schema')=='INDEPENDENT_COMPLETED_H_PARALLEL_PAIR_ADMISSION_METADATA_AUDIT_V1':names['candidate'].write_bytes(names['candidate'].read_bytes()+b'\n')
   return text
  monkeypatch.setattr(g.json,'dumps',drift)
 elif case=='wrong_snapshot':g.__file__=str(P)
 elif case=='wrong_attempt':
  L=g.read(names['launch']);L['wrapper_format']='H_PRIMARY_PAIR_ADMISSION_TEMPLATE_V1';L['attempt_utc']='20261399T999999Z';dump(names['launch'],L);bd['proofs']['launch']['sha256']=g.sha(names['launch'])
 elif case=='witness_wrong_code':
  w=g.read(names['process_witness']);w['observed_process_tree'][1]['cmdline']='foreign';dump(names['process_witness'],w);bd['proofs']['process_witness']['sha256']=g.sha(names['process_witness'])
 elif case=='stdout_early_complete':
  lines=names['stdout'].read_text().splitlines();names['stdout'].write_text('\n'.join([lines[-1],*lines[:-1]])+'\n');bd['proofs']['stdout']['sha256']=g.sha(names['stdout'])
 else:names['time_v'].write_text(names['time_v'].read_text().replace('Exit status: 0','Exit status: 1'));bd['proofs']['time_v']['sha256']=g.sha(names['time_v'])
 dump(b,bd)
 with pytest.raises((AssertionError,ValueError,KeyError,FileNotFoundError)):run_public(g,b,mode)
 assert not out.exists()


def test_EFF_formal_rejected_before_even_fake_slot(tmp_path,monkeypatch):
 g,b,names,entries,dump=public_fixture(tmp_path,monkeypatch,'efficientnet_b0','actual_qualification_prototype')
 with pytest.raises(AssertionError):run_public(g,b,'qualified_pair')
 assert entries==[]


def test_last_footer_cannot_change_proof_after_final_hash_check(tmp_path,monkeypatch):
 g,b,names,entries,dump=public_fixture(tmp_path,monkeypatch,mode='actual_qualification_prototype');real=g.old.manifest_footer;calls=[]
 def last_footer(path):
  value=real(path);calls.append(str(path))
  if len(calls)==28:names['candidate'].write_bytes(names['candidate'].read_bytes()+b'\n')
  return value
 monkeypatch.setattr(g.old,'manifest_footer',last_footer)
 with pytest.raises(AssertionError,match='metadata pin'):run_public(g,b,'actual_qualification_prototype')
 assert not (g.HERE/'H_DINO_seed7_actual_CPU_qualification_metadata_audit.json').exists()


# HPA001/HPA002 are exact independent-review trigger boundaries. Their original
# probe and report remain byte-identical; rejection must leave no positive JSON.
def test_snapshot_replacement_inside_fake_slot_preserves_early_digest(tmp_path,monkeypatch):
 import contextlib
 g,b,names,entries,dump=public_fixture(tmp_path,monkeypatch,mode='actual_qualification_prototype');snapshot=Path(g.__file__);initial=g.sha(snapshot)
 @contextlib.contextmanager
 def nonlocking_counted_slot(*args):
  entries.append(tuple(map(str,args)));snapshot.write_text('raise RuntimeError("these are unexecuted replacement bytes")\n');yield
 monkeypatch.setattr(g.old.SHARED,'cpu_slot',nonlocking_counted_slot)
 with pytest.raises(AssertionError,match='snapshot|source digest'):run_public(g,b,'actual_qualification_prototype')
 assert len(entries)==1 and g.sha(snapshot)!=initial and initial in str(snapshot.parent)
 assert not (g.HERE/'H_DINO_seed7_actual_CPU_qualification_metadata_audit.json').exists()


def test_last_footer_earlier_manifest_rewrite_cannot_publish_old_metadata(tmp_path,monkeypatch):
 import pyarrow as pa,pyarrow.parquet as pq
 g,b,names,entries,dump=public_fixture(tmp_path,monkeypatch,mode='actual_qualification_prototype');real=g.old.manifest_footer;calls=[];first=[]
 def rewrite_earlier(path):
  value=real(path);calls.append(str(path))
  if not first:first.append(Path(path))
  if len(calls)==28:
   table=pq.read_table(first[0]);pq.write_table(pa.Table.from_pydict({name:[0]*81 for name in table.column_names}),first[0])
  return value
 monkeypatch.setattr(g.old,'manifest_footer',rewrite_earlier)
 with pytest.raises(AssertionError,match='manifest.*(fingerprint|drift|changed)'):run_public(g,b,'actual_qualification_prototype')
 assert len(entries)==1 and len(calls)==28 and real(first[0])['rows']==81
 assert not (g.HERE/'H_DINO_seed7_actual_CPU_qualification_metadata_audit.json').exists()


def test_manifest_metadata_fingerprint_detects_same_size_inode_replacement(tmp_path):
 m=module();p=tmp_path/'synthetic.parquet';p.write_bytes(b'1234');first=m.manifest_fingerprint(p);assert set(first)=={'dev','inode','size','mtime_ns','ctime_ns'}
 q=tmp_path/'replacement.parquet';q.write_bytes(b'5678');s=p.stat();os.utime(q,ns=(s.st_atime_ns,s.st_mtime_ns));os.replace(q,p);second=m.manifest_fingerprint(p)
 assert first['size']==second['size'] and first['mtime_ns']==second['mtime_ns'] and first['inode']!=second['inode'] and first!=second



def test_same_legacy_producer_prototype_then_qualified_uses_distinct_bindings(tmp_path,monkeypatch):
 # Build one completed synthetic DINO7 producer and its accepted own-Q fixture.
 # Publish prototype before making that Q/qualified binding available. This does
 # not create a second launch or touch the original historical five-ref binding.
 g,legacy,names,entries,dump=public_fixture(tmp_path,monkeypatch,mode='qualified_pair');historical=g.HERE/'H_dinov2_vit_b14_seed7_matrix_admission_bindings.json';historical_sha=g.sha(historical);assert len(g.read(historical)['proofs'])==5;six=g.read(legacy);Qref=six['proofs']['CPU_qualification'];qpath=Path(Qref['path']);qbytes=qpath.read_bytes();qpath.unlink();legacy.unlink()
 stem=names['launch'].name.removesuffix('_launch.json');pb=g.HERE/f'{stem}_actual_qualification_prototype_matrix_admission_bindings.json';qb=g.HERE/f'{stem}_qualified_pair_matrix_admission_bindings.json';five=copy.deepcopy(six);five['proofs'].pop('CPU_qualification');dump(pb,five);prototype_sha=g.sha(pb)
 facts={str(p):g.sha(p) for p in names.values()};proto=run_public(g,pb,'actual_qualification_prototype');assert proto['CPU_qualification_issued'] is False and len(entries)==1
 # A separate acceptance step supplies the prebuilt synthetic own-Q bytes. The
 # qualified binding adds only Q; every producer fact is the same SHA as before.
 qpath.write_bytes(qbytes);dump(qb,six);qualified=run_public(g,qb,'qualified_pair');assert qualified['CPU_qualification_issued'] is False and len(entries)==2 and proto['output']!=qualified['output']
 assert g.sha(pb)==prototype_sha and g.sha(historical)==historical_sha and all(g.sha(p)==h for p,h in facts.items());assert {k:v for k,v in g.read(qb)['proofs'].items() if k!='CPU_qualification'}==g.read(pb)['proofs'];assert len(g.read(pb)['proofs'])==5 and len(g.read(qb)['proofs'])==6
 before=len(entries)
 for path,mode in [(pb,'actual_qualification_prototype'),(qb,'qualified_pair')]:
  with pytest.raises(AssertionError,match='preserve independent outputs'):run_public(g,path,mode)
 assert len(entries)==before


def test_future_binding_contract_names_each_mode(tmp_path,monkeypatch):
 g,b,names,entries,dump=public_fixture(tmp_path,monkeypatch,mode='qualified_pair',future=True);L=g.read(names['launch']);stem=names['launch'].name.removesuffix('_launch.json')
 for mode in ('actual_qualification_prototype','qualified_pair'):
  _,binding=g.launch_layout(L,'dinov2_vit_b14',123,mode=mode);assert binding==g.HERE/f'{stem}_{mode}_matrix_admission_bindings.json'
 assert g.main([])['future_binding']=='<stem>_<mode>_matrix_admission_bindings.json'
