"""Independent public metadata surrogates; all generated files /tmp only.
Explicit non-acquiring counting slot substitutes throughout; no real slot lock IO,
producer context/model/weights/images/GPU, or unfinished Eff C is accessed.
"""
from pathlib import Path
import ast,contextlib,copy,hashlib,json,pytest
import pyarrow as pa,pyarrow.parquet as pq
import test_h_primary_pair_admission_audit as reviewed

def fixture(tmp_path,monkeypatch,qualified=False,future=False):
 mode='qualified_pair' if qualified else 'actual_qualification_prototype'
 g,b,names,entries,dump=reviewed.public_fixture(tmp_path,monkeypatch,'dinov2_vit_b14',mode,future)
 assert g.ROOT.is_relative_to(tmp_path) and Path(g.__file__).is_relative_to(tmp_path) and entries==[]
 return g,b,names,entries,dump,mode

@pytest.mark.parametrize('future',[False,True])
def test_normal_public_qualified_output_matches_exact_bb299e_A_chain(tmp_path,monkeypatch,future):
 g,b,names,entries,dump,mode=fixture(tmp_path,monkeypatch,qualified=True,future=future)
 result=reviewed.run_public(g,b,mode);A=g.read(result['output']);C=g.read(names['candidate']);I=C['identity'];bone=I['backbone'];seed=I['seed'];Q=g.read(g.QUALIFICATIONS[bone]['path']);F=g.old.F
 refs=dict(cache_admission={'path':str(names['candidate']),'sha256':g.sha(names['candidate'])},cache_audit={'path':result['output'],'sha256':g.sha(result['output'])},primary_entry={'path':str(F/'cpu_primary_analysis.py'),'sha256':g.CPU_HASHES['cpu_primary_analysis.py']},CPU_qualification={'path':g.QUALIFICATIONS[bone]['path'],'sha256':g.QUALIFICATIONS[bone]['sha256']},cache_audit_entry={'path':g.__file__,'sha256':g.sha(g.__file__)})
 pinned=dict(C['pinned_sha256']);pinned.update({refs[k]['path']:refs[k]['sha256'] for k in ('CPU_qualification','cache_admission','primary_entry')});assert len(pinned)==110
 L=dict(status='LAUNCHED_ORIGINAL_PRIMARY_H_PAIR_STATISTICS_AFTER_INDEPENDENT_CACHE_ADMISSION',backbone=bone,seed=seed,bootstrap_repeats=2000,GPUs_allowed=False,original_F_unchanged=True,M_statistics_started=False,admission_sha256=refs['cache_admission']['sha256'],qualification_sha256=refs['CPU_qualification']['sha256'],independent_cache_audit=refs['cache_audit']['path'],independent_cache_audit_sha256=refs['cache_audit']['sha256'],entry=refs['primary_entry']['path'],entry_sha256=refs['primary_entry']['sha256'],CPU_slot=1)
 terminal='a'*64
 P=dict(status='ORIGINAL_PRIMARY_H_ANALYSIS_EXECUTION_COMPLETE_REQUIRES_INDEPENDENT_METRIC_BOOTSTRAP_AUDIT',backbone=bone,seed=seed,original_analysis_identity=I,original_analysis_complete_sha256=terminal,pair_admission_sha256=refs['cache_admission']['sha256'],source_qualification_sha256=refs['CPU_qualification']['sha256'],pinned_sha256=pinned,CPU_slot=1,GPU_launch_attempted=False,primary_cache_owner_modified=False,M_results_generated=False,fresh_ledger_invocations=[copy.deepcopy(C['fresh_ledger_invocations'][0]) for _ in range(42)])
 # Extract only accepted pure JSON functions, never import/run scientific BB kernel.
 nodes=[n for n in ast.parse(g.BB_PATH.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('_int','validate_ledgers','validate_chain')];assert len(nodes)==3
 ns=dict(Path=Path,require=g.require,BACKBONES=tuple(g.LABELS),SEEDS=g.SEEDS,CPU_HASHES=g.CPU_HASHES)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(g.BB_PATH),'exec'),ns)
 assert ns['validate_chain'](g.ROOT,bone,seed,refs,I,C,A,L,P,Q,terminal_sha=terminal) is True
 assert len(entries)==1
 print('BB299E_A_CHAIN_COMPATIBLE',future,len(A['canonical_shared107_pin_sha256']),len(A['role_marker_checks']))


def test_snapshot_change_after_initial_namespace_check_must_reject(tmp_path,monkeypatch):
 g,b,names,entries,dump,mode=fixture(tmp_path,monkeypatch)
 snapshot=Path(g.__file__);initial=g.sha(snapshot)
 @contextlib.contextmanager
 def nonlocking_counted_slot(*args):
  entries.append(tuple(map(str,args)))
  snapshot.write_text('raise RuntimeError("these are unexecuted replacement bytes")\n')
  yield
 monkeypatch.setattr(g.old.SHARED,'cpu_slot',nonlocking_counted_slot)
 with pytest.raises(AssertionError,match='audit source digest changed after initial snapshot check'):
  reviewed.run_public(g,b,mode)
 assert initial!=g.sha(snapshot) and initial in str(snapshot.parent) and len(entries)==1
 assert not (g.HERE/'H_DINO_seed7_actual_CPU_qualification_metadata_audit.json').exists()
 print('FIX_REJECT original_initial_snapshot_to_owned_race')


def test_last_footer_earlier_manifest_drift_must_reject(tmp_path,monkeypatch):
 g,b,names,entries,dump,mode=fixture(tmp_path,monkeypatch);footer=g.old.manifest_footer;calls=[];first=[]
 def drift(path):
  value=footer(path);calls.append(str(path))
  if not first:first.append(Path(path))
  if len(calls)==28:
   table=pq.read_table(first[0]);pq.write_table(pa.Table.from_pydict({name:[0]*81 for name in table.column_names}),first[0])
  return value
 monkeypatch.setattr(g.old,'manifest_footer',drift)
 with pytest.raises(AssertionError,match='manifest fingerprint drift during audit'):
  reviewed.run_public(g,b,mode)
 assert len(calls)==28 and footer(first[0])['rows']==81 and len(entries)==1
 assert not (g.HERE/'H_DINO_seed7_actual_CPU_qualification_metadata_audit.json').exists()
 print('FIX_REJECT original_final_footer_to_earlier_manifest_race')


def test_final_proof_scan_manifest_inode_drift_also_refuses(tmp_path,monkeypatch):
 import os
 g,b,names,entries,dump,mode=fixture(tmp_path,monkeypatch);footer=g.old.manifest_footer;counter=[];first=[];changed=[]
 def observe(path):
  value=footer(path);counter.append(str(path))
  if not first:first.append(Path(path))
  return value
 check=g.check_pins
 def replace_after_proof_scan(pins):
  answer=check(pins)
  if len(counter)==28 and not changed:
   p=first[0];oldstat=p.stat();replacement=p.with_suffix('.replacement');replacement.write_bytes(p.read_bytes());os.utime(replacement,ns=(oldstat.st_atime_ns,oldstat.st_mtime_ns));os.replace(replacement,p);changed.append((oldstat.st_ino,p.stat().st_ino))
  return answer
 monkeypatch.setattr(g.old,'manifest_footer',observe);monkeypatch.setattr(g,'check_pins',replace_after_proof_scan)
 with pytest.raises(AssertionError,match='manifest fingerprint drift during audit'):
  reviewed.run_public(g,b,mode)
 assert changed and changed[0][0]!=changed[0][1] and len(entries)==1
 assert not (g.HERE/'H_DINO_seed7_actual_CPU_qualification_metadata_audit.json').exists()
 print('FIX_REJECT same_bytes_same_mtime_inode_drift_after_other_proof_scan')


def test_snapshot_drift_after_owned_before_publication_refuses(tmp_path,monkeypatch):
 g,b,names,entries,dump,mode=fixture(tmp_path,monkeypatch);owned=g.audit_owned;initial=g.sha(g.__file__);observed=[]
 def drift(*args,**kwargs):
  assert kwargs['source_sha']==initial
  d=owned(*args,**kwargs);assert d['audit_source_sha256']==initial and d['bracketed_readonly_evidence_sha256'][g.__file__]==initial
  observed.append(True);Path(g.__file__).write_text('raise RuntimeError("late unexecuted source bytes")\n');return d
 monkeypatch.setattr(g,'audit_owned',drift)
 with pytest.raises(AssertionError,match='audit source digest changed after initial snapshot check'):
  reviewed.run_public(g,b,mode)
 assert observed==[True] and len(entries)==1
 assert not (g.HERE/'H_DINO_seed7_actual_CPU_qualification_metadata_audit.json').exists()
 print('FIX_REJECT fixed_owned_digest_to_publish_boundary')


def test_owned_wrong_explicit_early_digest_refuses_before_binding_reads(tmp_path,monkeypatch):
 g,b,names,entries,dump,mode=fixture(tmp_path,monkeypatch)
 with pytest.raises(AssertionError,match='audit source digest changed after initial snapshot check'):
  g.audit_owned({'path':str(b),'sha256':g.sha(b)},mode,1,source_sha='0'*64)
 assert entries==[]
 assert not (g.HERE/'H_DINO_seed7_actual_CPU_qualification_metadata_audit.json').exists()
 print('FIX_REJECT explicit_incorrect_source_digest')
