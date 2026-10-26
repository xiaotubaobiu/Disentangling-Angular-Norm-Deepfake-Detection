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
 snapshot=Path(g.__file__);initial=g.sha(snapshot);changes=[]
 @contextlib.contextmanager
 def nonlocking_counted_slot(*args):
  entries.append(tuple(map(str,args)))
  # Simulate source drift after main's require_snapshot but before audit pins it.
  snapshot.write_text('raise RuntimeError("these are unexecuted replacement bytes")\n');changes.append(g.sha(snapshot))
  yield
 monkeypatch.setattr(g.old.SHARED,'cpu_slot',nonlocking_counted_slot)
 result=reviewed.run_public(g,b,mode);A=g.read(result['output']);recorded=A['bracketed_readonly_evidence_sha256'][str(snapshot)]
 assert initial!=recorded==g.sha(snapshot) and initial in str(snapshot.parent) and recorded not in str(snapshot.parent)
 print('SNAPSHOT_FALSE_ATTESTATION',json.dumps({'loaded_original_SHA':initial,'recorded_unexecuted_SHA':recorded,'snapshot_namespace':str(snapshot),'status':A['status'],'no_real_lock_calls':True}))
 assert not Path(result['output']).exists(),'BLOCKER: public main accepted unexecuted replacement code as its source after snapshot namespace was validated only once'


def test_last_footer_earlier_manifest_drift_must_reject(tmp_path,monkeypatch):
 g,b,names,entries,dump,mode=fixture(tmp_path,monkeypatch);footer=g.old.manifest_footer;calls=[];first=[]
 def drift(path):
  value=footer(path);calls.append(str(path))
  if not first:first.append(Path(path))
  if len(calls)==28:
   # All synthetic Parquet files only. Same required columns, changed row count.
   table=pq.read_table(first[0]);pq.write_table(pa.Table.from_pydict({name:[0]*81 for name in table.column_names}),first[0])
  return value
 monkeypatch.setattr(g.old,'manifest_footer',drift)
 result=reviewed.run_public(g,b,mode);A=g.read(result['output']);declared=next(x for x in A['role_manifest_footer_metadata'] if x['path']==str(first[0]));current=footer(first[0]);assert declared!=current and declared['rows']==80 and current['rows']==81
 assert len(entries)==1
 print('STALE_FOOTER_PUBLISHED',json.dumps({'declared_rows':declared['rows'],'current_rows':current['rows'],'footer_calls_before_postcheck':len(calls),'status':A['status'],'no_real_lock_calls':True}))
 assert not Path(result['output']).exists(),'BLOCKER: final footer sweep can mutate earlier manifest before publication while final pin hashes do not bind those footers'
