"""Independent Eff metadata-anchor review. All public main fixtures /tmp with
explicit no-lock counters. No actual slots, producer execution or heavy payloads.
"""
from pathlib import Path
import ast,copy,json,pytest
import test_h_primary_pair_admission_audit as reviewed
import independent_h_primary_pair_admission_fix_probes as previous
E=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/engineering/source_best_M')

def test_anchor_only_AST_and_preserved_previous_maps():
 m=reviewed.module();prior=json.loads((E/'independent_h_primary_pair_admission_audit_mode_binding_fix_confirmation.json').read_text());old=Path(prior['reviewed_source']['snapshot']);assert m.sha(old)==prior['reviewed_source']['sha256']
 a,b=ast.parse(old.read_text()),ast.parse(reviewed.P.read_text())
 def functions(tree):return {n.name:n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
 fa,fb=functions(a),functions(b);assert set(fa)==set(fb)
 for name in fa:
  x,y=copy.deepcopy(fa[name]),copy.deepcopy(fb[name])
  if name=='validate_mode':
   # Only the obsolete error message changes; admission expression is identical.
   assert x.body[1].value.args[1].value=='own actual qualification required; EFF formal disabled'
   assert y.body[1].value.args[1].value=='fixed own actual qualification required'
   x.body[1].value.args[1]=y.body[1].value.args[1]
  assert ast.dump(x,include_attributes=False)==ast.dump(y,include_attributes=False),name
 def assignments(tree):return {n.targets[0].id:n.value for n in tree.body if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
 ga,gb=assignments(a),assignments(b);assert set(ga)==set(gb)
 assert [k for k in ga if ast.dump(ga[k])!=ast.dump(gb[k])]==['QUALIFICATIONS']
 ma,mb=ga['QUALIFICATIONS'],gb['QUALIFICATIONS'];assert [k.value for k in ma.keys]==['clip_vit_b16','dinov2_vit_b14'];assert [k.value for k in mb.keys]==['clip_vit_b16','dinov2_vit_b14','efficientnet_b0']
 assert all(ast.dump(x)==ast.dump(y) for x,y in zip(ma.values,mb.values))
 print('ONLY_EFF_FIXED_ANCHOR_PLUS_DOC_AND_ERROR_MESSAGE')

def test_actual_Eff_own_Q_aux_metadata_compatibility():
 m=reviewed.module();bone='efficientnet_b0';Q,C,A,ER,ref,pins=reviewed.real_Q_inputs(m,bone);spec=m.QUALIFICATIONS[bone]
 assert C['seed']==7 and A['backbone']==bone and Q['allowed_backbones']==[bone]
 m.validate_qualification_body(bone,Q,C,A,ER,ref,pins);extra={};assert m.validate_qualification(bone,dict(path=spec['path'],sha256=spec['sha256']),ref,pins,extra)==Q
 assert len(pins)==107 and len(extra)==4 and all(m.sha(p)==h for p,h in extra.items())
 print('ACTUAL_EFF_METADATA_ONLY_FIXED_Q_C_AQ_ER',json.dumps(extra,sort_keys=True))

@pytest.mark.parametrize('future',[False,True])
def test_Eff_public_output_is_exact_bb299e_A_chain(tmp_path,monkeypatch,future):
 def eff_fixture(tmp,mp,qualified=False,future=False):
  mode='qualified_pair' if qualified else 'actual_qualification_prototype';g,b,names,entries,dump=reviewed.public_fixture(tmp,mp,'efficientnet_b0',mode,future)
  assert g.ROOT.is_relative_to(tmp) and Path(g.__file__).is_relative_to(tmp) and entries==[]
  return g,b,names,entries,dump,mode
 monkeypatch.setattr(previous,'fixture',eff_fixture)
 previous.test_normal_public_qualified_output_matches_exact_bb299e_A_chain(tmp_path,monkeypatch,future)
 print('EFF_QUALIFIED_BB299E_COMPATIBLE',future)

def test_Eff_same_producer_prototype_to_qualified_and_duplicate_refusal(tmp_path,monkeypatch):
 g,qb,names,entries,dump=reviewed.public_fixture(tmp_path,monkeypatch,'efficientnet_b0','qualified_pair');assert g.ROOT.is_relative_to(tmp_path) and entries==[]
 historical=g.HERE/'H_efficientnet_b0_seed7_matrix_admission_bindings.json';historical_sha=g.sha(historical);six=g.read(qb);qpath=Path(six['proofs']['CPU_qualification']['path']);qbytes=qpath.read_bytes();qpath.unlink();qb.unlink()
 stem=names['launch'].name.removesuffix('_launch.json');pb=g.HERE/f'{stem}_actual_qualification_prototype_matrix_admission_bindings.json';assert qb==g.HERE/f'{stem}_qualified_pair_matrix_admission_bindings.json'
 five=copy.deepcopy(six);five['proofs'].pop('CPU_qualification');dump(pb,five);proto_sha=g.sha(pb);facts={str(p):g.sha(p) for p in names.values()}
 proto=reviewed.run_public(g,pb,'actual_qualification_prototype');assert proto['CPU_qualification_issued'] is False and not qpath.exists() and len(entries)==1
 qpath.write_bytes(qbytes);dump(qb,six);qualified=reviewed.run_public(g,qb,'qualified_pair');assert qualified['CPU_qualification_issued'] is False and len(entries)==2 and proto['output']!=qualified['output']
 assert g.sha(pb)==proto_sha and g.sha(historical)==historical_sha and all(g.sha(p)==h for p,h in facts.items())
 assert {k:v for k,v in g.read(qb)['proofs'].items() if k!='CPU_qualification'}==g.read(pb)['proofs'] and len(g.read(pb)['proofs'])==5 and len(g.read(qb)['proofs'])==6
 for binding,mode in [(pb,'actual_qualification_prototype'),(qb,'qualified_pair')]:
  with pytest.raises(AssertionError,match='preserve independent outputs'):reviewed.run_public(g,binding,mode)
 assert len(entries)==2
 print('EFF_SINGLE_PRODUCER_5_TO_6_REFS_NO_OVERWRITE_DUPLICATES_REFUSED')

@pytest.mark.parametrize('case',['Q_bool140','A_bool140','A_bool107','A_failed','A_wrongbone','A_false_independent','Q_wrong_aux_namespace','Q_new_self_declaration'])
def test_Eff_public_sha_correct_bad_semantics_or_unaccepted_anchor(tmp_path,monkeypatch,case):
 g,b,names,entries,dump=reviewed.public_fixture(tmp_path,monkeypatch,'efficientnet_b0','qualified_pair');assert g.ROOT.is_relative_to(tmp_path) and entries==[];bd=g.read(b);qr=bd['proofs']['CPU_qualification'];Q=g.read(qr['path']);spec=g.QUALIFICATIONS['efficientnet_b0'];match=None
 if case=='Q_bool140':Q['fresh_default_source_checks']=True;match='qualified typed14/140 boundaries'
 elif case=='Q_wrong_aux_namespace':Q['independent_actual_audit']=str(tmp_path/'foreign_Aq.json');match='canonical own Q three auxiliary paths/SHA'
 elif case=='Q_new_self_declaration':Q['utc']='unaccepted_reissued_Q'
 else:
  A=g.read(Q['independent_actual_audit'])
  if case=='A_bool140':A['fresh_default_source_checks_in_ledgers']=True
  elif case=='A_bool107':A['current_pin_count']=True;match='qualification audit typed inventory'
  elif case=='A_failed':A['status']='FAILED'
  elif case=='A_wrongbone':A['backbone']='dinov2_vit_b14'
  else:A['all107_current_hashes_match_receipt_and_accepted_expectations']=False;match='accepted independent real metadata body'
  dump(Q['independent_actual_audit'],A);Q['independent_actual_audit_sha256']=g.sha(Q['independent_actual_audit']);spec['audit_sha256']=Q['independent_actual_audit_sha256']
 dump(qr['path'],Q);qr['sha256']=g.sha(qr['path'])
 if case!='Q_new_self_declaration':spec['sha256']=qr['sha256']
 else:match='external canonical fixed own Q'
 dump(b,bd)
 with pytest.raises((AssertionError,ValueError,KeyError),match=match) as exc:reviewed.run_public(g,b,'qualified_pair')
 assert not (g.HERE/'H_EFF_seed7_parallel_cache_admission_independent_audit.json').exists() and not (g.HERE/'H_EFF_seed7_actual_CPU_qualification_metadata_audit.json').exists()
 print('EFF_REJECT',case,str(exc.value))
