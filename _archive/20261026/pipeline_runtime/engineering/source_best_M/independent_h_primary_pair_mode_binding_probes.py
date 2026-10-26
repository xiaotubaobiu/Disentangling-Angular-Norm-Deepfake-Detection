"""Independent mode-binding transition probes. No actual producer/slot work.
All new producer fixtures/output files are /tmp. Real DINO facts are read-only;
its unpublished six-ref binding is served as a SHA-checked in-memory document.
"""
from pathlib import Path
import ast,copy,hashlib,json,pytest
import test_h_primary_pair_admission_audit as reviewed
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M'

def fingerprint(m,paths):return {str(p):(m.sha(p),Path(p).stat().st_mtime_ns) for p in paths}

def test_transition_same_DINO_producer_retains_five_facts_and_distinct_two_outputs(tmp_path,monkeypatch):
 g,qb,names,entries,dump=reviewed.public_fixture(tmp_path,monkeypatch,mode='qualified_pair');six=g.read(qb);stem=names['launch'].name.removesuffix('_launch.json');historical=g.HERE/'H_dinov2_vit_b14_seed7_matrix_admission_bindings.json';old_fact=fingerprint(g,[historical,*names.values()]);assert len(g.read(historical)['proofs'])==5
 qpath=Path(six['proofs']['CPU_qualification']['path']);qbytes=qpath.read_bytes();qpath.unlink();qb.unlink()
 five=copy.deepcopy(six);five['proofs'].pop('CPU_qualification');pb=g.HERE/f'{stem}_actual_qualification_prototype_matrix_admission_bindings.json';dump(pb,five);proto_fact=fingerprint(g,[pb])
 prototype=reviewed.run_public(g,pb,'actual_qualification_prototype');assert len(entries)==1 and prototype['CPU_qualification_issued'] is False
 # Only own-Q and a separate6-ref binding become available. Producer5facts never change.
 qpath.write_bytes(qbytes);dump(qb,six);qualified=reviewed.run_public(g,qb,'qualified_pair');assert len(entries)==2 and qualified['CPU_qualification_issued'] is False
 assert prototype['output']!=qualified['output'] and fingerprint(g,[historical,*names.values()])==old_fact and fingerprint(g,[pb])==proto_fact
 assert g.read(pb)['proofs']=={k:v for k,v in g.read(qb)['proofs'].items() if k!='CPU_qualification'}
 A5=g.read(prototype['output']);A6=g.read(qualified['output']);assert A5['candidate_identity']==A6['candidate_identity'] and A5['role_marker_checks']==A6['role_marker_checks'];assert A5['bracketed_readonly_evidence_sha256'][str(pb)]==g.sha(pb) and A6['bracketed_readonly_evidence_sha256'][str(qb)]==g.sha(qb)
 preserved=fingerprint(g,[pb,qb,historical,Path(prototype['output']),Path(qualified['output']),*names.values()]);before=len(entries)
 for binding,mode in [(pb,'actual_qualification_prototype'),(qb,'qualified_pair')]:
  with pytest.raises(AssertionError,match='preserve independent outputs'):reviewed.run_public(g,binding,mode)
 for binding,mode in [(pb,'qualified_pair'),(qb,'actual_qualification_prototype')]:
  with pytest.raises(AssertionError,match='exact5/6 external proof inventory'):reviewed.run_public(g,binding,mode)
 assert len(entries)==before and fingerprint(g,[Path(x) for x in preserved])==preserved
 print('SAME_PRODUCER_DINO_TRANSITION',json.dumps({'distinct_bindings':True,'facts_5_unchanged':True,'old_historical_binding_unchanged':True,'two_outputs':True,'repeat_and_mode_swap_refused_before_fake_slot':True}))


def test_Eff_same_producer_synthetic_prototype_and_qualified_boundary(tmp_path,monkeypatch):
 g,pb,names,entries,dump=reviewed.public_fixture(tmp_path,monkeypatch,bone='efficientnet_b0',mode='actual_qualification_prototype');five=g.read(pb);stem=names['launch'].name.removesuffix('_launch.json');historical=g.HERE/'H_efficientnet_b0_seed7_matrix_admission_bindings.json';before=fingerprint(g,[pb,historical,*names.values()])
 proto=reviewed.run_public(g,pb,'actual_qualification_prototype');assert len(entries)==1
 qpath=g.ROOT/'reports/H_parallel_CPU_actual_qualification_efficientnet_b0.json';dump(qpath,dict(schema='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1',synthetic_only=True,allowed_backbones=['efficientnet_b0']))
 qb=g.HERE/f'{stem}_qualified_pair_matrix_admission_bindings.json';six=copy.deepcopy(five);six['proofs']['CPU_qualification']={'path':str(qpath),'sha256':g.sha(qpath)};dump(qb,six)
 # Layout is mode-distinct; it does NOT grant current formal Eff permission.
 L=g.read(names['launch']);_,expected_pb=g.launch_layout(L,'efficientnet_b0',7,mode='actual_qualification_prototype');_,expected_qb=g.launch_layout(L,'efficientnet_b0',7,mode='qualified_pair');assert expected_pb==pb and expected_qb==qb and pb!=qb
 with pytest.raises(AssertionError,match='own actual qualification required; EFF formal disabled'):reviewed.run_public(g,qb,'qualified_pair')
 with pytest.raises(AssertionError,match='preserve independent outputs'):reviewed.run_public(g,pb,'actual_qualification_prototype')
 assert len(entries)==1 and fingerprint(g,[Path(x) for x in before])==before and not (g.HERE/'H_EFF_seed7_parallel_cache_admission_independent_audit.json').exists()
 print('EFF_MODE_BOUNDARY',json.dumps({'synthetic_only':True,'prototype_ok':True,'qualified_refused':True,'five_facts_and_old_binding_unchanged':True,'no_actual_Eff_C_read':True}))


def test_real_DINO_old_chain_readonly_and_unpublished_six_ref_layout_load(tmp_path,monkeypatch):
 m=reviewed.module();report=json.loads((E/'H_primary_pair_admission_audit_mode_binding_fix_report.json').read_text());bindings=report['historical_DINO_evidence_and_same_producer_facts_unchanged'];before=fingerprint(m,[Path(x) for x in bindings]);assert all(m.sha(p)==h for p,h in bindings.items())
 old_path=E/'audits/H_dinov2_vit_b14_seed7_matrix_admission_bindings.json';old=m.read(old_path);assert len(old['proofs'])==5 and m.sha(old_path)=='d47351498b66010c22ae215c94736549dea70e4b4c02a1865704a27f1e160626'
 A=m.read(E/'audits/H_DINO_seed7_actual_CPU_qualification_metadata_audit.json');assert A['bracketed_readonly_evidence_sha256'][str(old_path)]==m.sha(old_path)
 launch=old['proofs']['launch'];L=m.read(launch['path']);names,pb=m.launch_layout(L,'dinov2_vit_b14',7,mode='actual_qualification_prototype');_,qb=m.launch_layout(L,'dinov2_vit_b14',7,mode='qualified_pair');assert pb!=qb and old_path not in (pb,qb)
 assert {k:str(v) for k,v in names.items()}=={k:r['path'] for k,r in old['proofs'].items()}
 six=copy.deepcopy(old);six['proofs']['CPU_qualification']={'path':m.QUALIFICATIONS['dinov2_vit_b14']['path'],'sha256':m.QUALIFICATIONS['dinov2_vit_b14']['sha256']};raw=(json.dumps(six,indent=2)+'\n').encode();sixsha=hashlib.sha256(raw).hexdigest();existed=qb.exists()
 if existed:
  # If root independently prepared it, only a read-only protocol check occurs.
  parsed,got,_=m.load_binding({'path':str(qb),'sha256':m.sha(qb)},'qualified_pair');assert parsed['proofs']==six['proofs']
 else:
  # Only the absent new external file is virtual. Canonical path/mode/launch and
  # real unchanged source facts still use the actual functions and real bytes.
  checks=m.check_pins;read=m.read;virtual=[]
  def pin(pins):
   remaining={}
   for p,h in pins.items():
    if p==str(qb):assert h==sixsha;virtual.append(p)
    else:remaining[p]=h
   return checks(remaining)
  def get(path):return copy.deepcopy(six) if str(path)==str(qb) else read(path)
  monkeypatch.setattr(m,'check_pins',pin);monkeypatch.setattr(m,'read',get)
  parsed,got,_=m.load_binding({'path':str(qb),'sha256':sixsha},'qualified_pair');assert virtual and parsed['proofs']==six['proofs']
  assert not qb.exists()
 assert got==L and fingerprint(m,[Path(x) for x in bindings])==before
 assert all(m.sha(p)==h for p,h in bindings.items())
 print('REAL_DINO_READONLY_MODE_LAYOUT',json.dumps({'historical5_SHA':m.sha(old_path),'existing_prototype_A_SHA':m.sha(E/'audits/H_DINO_seed7_actual_CPU_qualification_metadata_audit.json'),'own_Q_SHA':m.sha(m.QUALIFICATIONS['dinov2_vit_b14']['path']),'new6_canonical_path':str(qb),'new6_was_real_file':existed,'new6_unpublished_template_SHA':sixsha,'actual_main_or_slots_executed':False,'actual_files_changed':False}))


def test_only_mode_functions_changed_and_race_functions_AST_unchanged():
 before=json.loads((E/'independent_h_primary_pair_admission_audit_fix_confirmation.json').read_text());old=Path(before['reviewed_source']['snapshot']);new=reviewed.P
 def functions(p):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef)}
 a,b=functions(old),functions(new);changed={n for n in set(a)|set(b) if a.get(n)!=b.get(n)};assert changed=={'launch_layout','load_binding','main'}
 print('AST_CHANGED_ONLY',sorted(changed))
