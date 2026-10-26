import importlib.util,sys,copy
from pathlib import Path
import pytest
P=Path(__file__).parents[1]/'tools/h_last_matrix_result_audit.py';s=importlib.util.spec_from_file_location('matrix_H_audit_under_test',P);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)

def packet(root,bone='clip_vit_b16',seed=123):
 roles={h+'/'+r:'b'*64 for h in ('B','AB') for r in m.base.ROLES};runs={h:f'H_formal_{bone}_{h}_seed{seed}' for h in ('B','AB')};ledger=str(root/f'reports/H_{bone}_AMP_paired_ledger.json');identity=dict(schema='H_PAIR_ANALYSIS_IDENTITY_V1',backbone=bone,seed=seed,effective_precision='AMP',bootstrap_repeats=2000,resampling_seed=20261023,run_ids=runs,inputs_sha256={ledger:'d'*64},cache_completion_sha256=roles,code_sha256={})
 refs=m.canonical_refs(root,bone,seed);refs={k:{'path':str(p),'sha256':str(i+1)*64 if i<9 else 'f'*64} for i,(k,p) in enumerate(refs.items())};pins={str(root/f'proof{i}.json'):'c'*64 for i in range(107)}
 ld=dict(backbone=bone,precision='AMP',fresh_source_completions=10,reference_sha256='d'*64,exact_original_ledger_except_top_utc=True,cross_invocation_receipt_reuse=False)
 C=dict(status='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS',identity=copy.deepcopy(identity),models=2,roles=14,pinned_sha256=pins,fresh_ledger_invocations=[copy.deepcopy(ld) for _ in range(14)],all_source_completions_fresh=140,cross_invocation_receipt_reuse=False)
 A=dict(schema='INDEPENDENT_COMPLETED_H_PARALLEL_PAIR_ADMISSION_METADATA_AUDIT_V1',status='PASS_COMPLETED_QUALIFIED_ORIGINAL_PAIR_ADMISSION_METADATA_HASH_REBIND',backbone=bone,seed=seed,candidate_identity=copy.deepcopy(identity),canonical_shared107_pin_sha256=copy.deepcopy(pins),reference_pin_inventory_independently_reconstructed=True,qualified_original_code_count=20,canonical_proof_count=37,models=2,roles=14,fresh_ledger_invocations=14,source_checks_per_ledger=10,source_checks_in_ledger_receipts=140,lifetime_CPU_flock_held=True,bracketed_readonly_evidence_sha256={refs[k]['path']:refs[k]['sha256'] for k in ('cache_admission','primary_entry','CPU_qualification','cache_audit_entry')},role_marker_checks=[dict(history=h,role=r,run_id=runs[h],complete_sha256=roles[h+'/'+r]) for h in ('B','AB') for r in m.base.ROLES])
 impl={name:m.CPU_HASHES[name] for name in m.CPU_HASHES};Q=dict(schema='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1',status='PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE',allowed_backbones=[bone],allowed_use=['independent_role_admission','primary_pair_analysis'],all_original_ledger_checks_exact_except_utc=True,independent_review_pass=True,production_GPU_allowed=False,original_role_checks=14,fresh_default_source_checks=140,implementation_sha256=impl)
 L=dict(status='LAUNCHED_ORIGINAL_PRIMARY_H_PAIR_STATISTICS_AFTER_INDEPENDENT_CACHE_ADMISSION',backbone=bone,seed=seed,CPU_slot=1,entry=refs['primary_entry']['path'],entry_sha256=impl['cpu_primary_analysis.py'],independent_cache_audit=refs['cache_audit']['path'],independent_cache_audit_sha256=refs['cache_audit']['sha256'],admission_sha256=refs['cache_admission']['sha256'],qualification_sha256=refs['CPU_qualification']['sha256'],bootstrap_repeats=2000,GPUs_allowed=False,original_F_unchanged=True,M_statistics_started=False)
 P=dict(status='ORIGINAL_PRIMARY_H_ANALYSIS_EXECUTION_COMPLETE_REQUIRES_INDEPENDENT_METRIC_BOOTSTRAP_AUDIT',backbone=bone,seed=seed,CPU_slot=1,source_qualification_sha256=refs['CPU_qualification']['sha256'],pair_admission_sha256=refs['cache_admission']['sha256'],original_analysis_identity=copy.deepcopy(identity),original_analysis_complete_sha256='e'*64,fresh_ledger_invocations=[copy.deepcopy(ld) for _ in range(42)],pinned_sha256={**pins,**{str(root/'qualification/H_parallel_fresh_validation'/n):h for n,h in impl.items()}},GPU_launch_attempted=False,primary_cache_owner_modified=False,M_results_generated=False)
 return root,bone,seed,refs,identity,C,A,L,P,Q

def validate(p):return m.validate_chain(*p,terminal_sha='e'*64)

@pytest.mark.parametrize('bone',['clip_vit_b16','dinov2_vit_b14','efficientnet_b0'])
@pytest.mark.parametrize('seed',[7,42,123,2026,3407])
def test_all_registered_fifteen_parallel_pair_chains(tmp_path,bone,seed):validate(packet(tmp_path,bone,seed))

@pytest.mark.parametrize('change',['seed','best','Qbone','Quse','C140bool','C10bool','Cmissing','reuse','Aidentity','Apins','Laudit','PCsha','PQsha','P14only','Pidentity','Pterminal','Qstatus','Qimpl','GPU','M'])
def test_incomplete_or_misbound_chain_never_accepts(tmp_path,change):
 p=packet(tmp_path);root,bone,seed,refs,I,C,A,L,P,Q=p
 if change=='seed':I['seed']=True
 elif change=='best':I['run_ids']['B']='H_source_best_clip_B_seed123'
 elif change=='Qbone':Q['allowed_backbones']=['efficientnet_b0']
 elif change=='Quse':Q['allowed_use']=['independent_role_admission']
 elif change=='C140bool':C['all_source_completions_fresh']=True
 elif change=='C10bool':C['fresh_ledger_invocations'][0]['fresh_source_completions']=True
 elif change=='Cmissing':C['fresh_ledger_invocations'].pop()
 elif change=='reuse':C['cross_invocation_receipt_reuse']=True
 elif change=='Aidentity':A['candidate_identity']['seed']=42
 elif change=='Apins':A['canonical_shared107_pin_sha256'].pop(next(iter(A['canonical_shared107_pin_sha256'])))
 elif change=='Laudit':L['independent_cache_audit_sha256']='f'*64
 elif change=='PCsha':P['pair_admission_sha256']=refs['cache_audit']['sha256']
 elif change=='PQsha':P['source_qualification_sha256']='f'*64
 elif change=='P14only':P['fresh_ledger_invocations']=P['fresh_ledger_invocations'][:14]
 elif change=='Pidentity':P['original_analysis_identity']['seed']=42
 elif change=='Pterminal':P['original_analysis_complete_sha256']='f'*64
 elif change=='Qstatus':Q['status']='CANDIDATE'
 elif change=='Qimpl':Q['implementation_sha256']['cpu_primary_analysis.py']='f'*64
 elif change=='GPU':P['GPU_launch_attempted']=True
 else:P['M_results_generated']=True
 with pytest.raises(AssertionError):validate(p)
