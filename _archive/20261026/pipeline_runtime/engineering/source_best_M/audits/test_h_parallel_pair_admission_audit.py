import copy,json,unittest,hashlib,tempfile
from unittest.mock import patch
from pathlib import Path
import h_parallel_pair_admission_audit as a

NAMES=('__init__','analysis_d','analysis_h','bootstrap_fast','cache_d','cache_h','coverage_h','data_d','data_h','geometry_h','models','qualification_h','qualify','queue_d','queue_h','scoring_d','stats_h','summary_d','trainer','worker_h')
def fixture(seed=123):
    reg=dict(status='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING',target_export_allowed=True,effective_runs={b:dict(precision='AMP',run_ids=[f'H_formal_{b}_{h}_seed{s}' for s in a.SEEDS for h in ('B','AB')]) for b in a.BACKBONES})
    manifest=dict(qualified_commit='5ef6d06bf7aeda02a5a11113e4c35a2c9909f215',source_sha256={f'reliability_v2/{n}.py':'1'*64 for n in NAMES})
    proof={str(a.ROOT/'H_run_registry.json'):a.REGISTRY_SHA}
    for b,e in reg['effective_runs'].items():
        proof.update({str(a.ROOT/'runs'/run/'H_completion_seal.json'):'2'*64 for run in e['run_ids']})
        proof[str(a.ROOT/f'qualification/H_target_export_{b}.json')]='2'*64;proof[str(a.ROOT/f'reports/H_{b}_AMP_paired_ledger.json')]='2'*64
    accepted=dict(current_canonical_proof_sha256=proof)
    pins={str(a.CODE/n):h for n,h in manifest['source_sha256'].items()};pins.update({str(a.F/n):h for n,h in a.ADMISSION_IMPL.items()});pins[str(a.MANIFEST)]=a.MANIFEST_SHA;pins.update(proof);pins[str(a.ALL3_ACCEPTED)]=a.ALL3_SHA
    for p in a.extra_paths(reg):pins.setdefault(str(p),'5'*64)
    code={n+'.py':'1'*64 for n in ('analysis_h','cache_h','stats_h','coverage_h','analysis_d','scoring_d','bootstrap_fast','cache_d')}
    inputs={str(a.ROOT/p):pins[str(a.ROOT/p)] for p in ('configs/H_execution.json','H_run_registry.json','qualification/H_target_export_clip_vit_b16.json','reports/H_clip_vit_b16_AMP_paired_ledger.json','data/H/freeze.json')}
    ident=dict(schema='H_PAIR_ANALYSIS_IDENTITY_V1',backbone='clip_vit_b16',seed=42,effective_precision='AMP',run_ids={h:f'H_formal_clip_vit_b16_{h}_seed42' for h in ('B','AB')},inputs_sha256=inputs,code_sha256=code,cache_completion_sha256={h+'/'+r:'6'*64 for h in ('B','AB') for r in a.ROLES},bootstrap_repeats=2000,resampling_seed=20261023,source_threshold='each_model_allB_SOURCE_CAL_empiricalFPR5',primary='DFD_allB_Raw_mean_logit_H10_minus_H00',test_cohorts='allB_and_common_reported_separately')
    ref=dict(schema='H_FRESH_PARALLEL_CPU_REAL_RECEIPT_AUDIT_V1',status='PASS',independent_review_pass=True,backbone='clip_vit_b16',seed=42,current_pin_count=107,qualified_original_source_count=20,canonical_accepted_proof_count=37,pinned_sha256=pins,original_primary_pair_identity=ident,source_training_h_code_bundle_sha256='7'*64,source_training_code_sha256='8'*64)
    i=copy.deepcopy(ident);i.update(seed=seed,run_ids={h:f'H_formal_clip_vit_b16_{h}_seed{seed}' for h in ('B','AB')})
    receipt=dict(status='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS',backbone='clip_vit_b16',seed=seed,models=2,roles=14,identity=i,pinned_sha256=dict(pins),all_source_completions_fresh=140,cross_invocation_receipt_reuse=False,production_scoring_executed=False,GPU_launch_attempted=False,live_cache_owner_changed=False,elapsed_seconds=100.,fresh_ledger_invocations=[dict(backbone='clip_vit_b16',precision='AMP',fresh_source_completions=10,exact_original_ledger_except_top_utc=True,reference_sha256='2'*64,cross_invocation_receipt_reuse=False,elapsed_seconds=1.) for _ in range(14)])
    return reg,manifest,accepted,ref,receipt

def marker(seed=123,role='B/SOURCE_CAL',history='B'):
    _,_,_,ref,r=fixture(seed);i=r['identity'];run=i['run_ids'][history];runidentity=dict(config='a',code='8'*64)
    seal=dict(schema='H_COMPLETION_SEAL_V1',status='SEALED',run_id=run,identity=runidentity,target_used=False,artifact_sha256={'final.pt':'9'*64},h_code_bundle_sha256='7'*64)
    mi=dict(run_id=run,history=history,role=role,backbone='clip_vit_b16',artifact_root=str(a.ROOT),config_path=str(a.ROOT/'configs/H_execution.json'),config_sha256=i['inputs_sha256'][str(a.ROOT/'configs/H_execution.json')],training_precision='AMP',effective_precision='AMP',epochs=30,dimension=512,checkpoint_path=str(a.ROOT/'runs'/run/'final.pt'),checkpoint_sha256='9'*64,training_identity=runidentity,training_complete_sha256=ref['pinned_sha256'][str(a.ROOT/'runs'/run/'complete.json')],completion_seal_sha256='2'*64,H_run_registry_sha256=a.REGISTRY_SHA,effective_run_ids=[f'H_formal_clip_vit_b16_{h}_seed{s}' for s in a.SEEDS for h in ('B','AB')],paired_ledger_path=str(a.ROOT/'reports/H_clip_vit_b16_AMP_paired_ledger.json'),paired_ledger_sha256='2'*64,target_export_gate_sha256='2'*64,H_data_freeze_sha256=i['inputs_sha256'][str(a.ROOT/'data/H/freeze.json')],h_code_bundle_sha256='7'*64,training_code_sha256='8'*64,cache_code_sha256='1'*64,cache_helper_code_sha256='1'*64,original_head_sha256='a'*64,source_manifest_path=str(a.ROOT/'data/H'/f'{role}.parquet'),source_manifest_sha256='b'*64,accepted_table_rows=80,accepted_table_videos=10)
    m=dict(schema='pipeline-reliability-v2-H-fp32-cache-1',status='CACHE_COMPLETE',identity=mi,precision='FP32',autocast=False,dimension=512,rows=80,cached_manifest_sha256='b'*64,input_validation='FRESH_PROCESSED_IMAGE_BYTE_AND_DECODED_RGB_HASHES',chunks=[{}])
    return ref,r,m,seal,runidentity

class Inventory(unittest.TestCase):
    def test_exact107_reconstructed_from20_and37(self):
        reg,m,ac,ref,r=fixture();p=a.reconstruct_pins(ref,m,reg,ac);self.assertEqual(len(p),107)
    def test_extra_candidate_pin_rejects(self):
        reg,m,ac,ref,r=fixture();r['pinned_sha256']['/tmp/foreign']='0'*64
        with self.assertRaises(AssertionError):a.validate_candidate(r,123,ref,a.reconstruct_pins(ref,m,reg,ac))
    def test_missing_pin_rejects(self):
        reg,m,ac,ref,r=fixture();ref['pinned_sha256'].pop(next(iter(ref['pinned_sha256'])))
        with self.assertRaises(AssertionError):a.reconstruct_pins(ref,m,reg,ac)
    def test_source19_rejects(self):
        reg,m,ac,ref,r=fixture();m['source_sha256'].pop('reliability_v2/worker_h.py')
        with self.assertRaises(AssertionError):a.reconstruct_pins(ref,m,reg,ac)
    def test_proof36_rejects(self):
        reg,m,ac,ref,r=fixture();ac['current_canonical_proof_sha256'].pop(next(iter(ac['current_canonical_proof_sha256'])))
        with self.assertRaises(AssertionError):a.reconstruct_pins(ref,m,reg,ac)
    def test_noncanonical_all5_run_ids_rejects(self):
        reg,m,ac,ref,r=fixture();reg['effective_runs']['clip_vit_b16']['run_ids'].reverse()
        with self.assertRaises(AssertionError):a.reconstruct_pins(ref,m,reg,ac)

class Receipt(unittest.TestCase):
    def check(self,r,seed=123):
        reg,m,ac,ref,_=fixture(seed);return a.validate_candidate(r,seed,ref,a.reconstruct_pins(ref,m,reg,ac))
    def test123_accepts(self):self.check(fixture()[4])
    def test2026_accepts_same107(self):self.check(fixture(2026)[4],2026)
    def bad(self,mutate):
        r=fixture()[4];mutate(r)
        with self.assertRaises(AssertionError):self.check(r)
    def test_wrong_seed_rejects(self):self.bad(lambda r:r.update(seed=42))
    def test_boolean_total_rejects(self):self.bad(lambda r:r.update(all_source_completions_fresh=True))
    def test_float_total_rejects(self):self.bad(lambda r:r.update(all_source_completions_fresh=140.))
    def test_only13_ledgers_rejects(self):self.bad(lambda r:r['fresh_ledger_invocations'].pop())
    def test_typed_count10_rejects_bool(self):self.bad(lambda r:r['fresh_ledger_invocations'][0].update(fresh_source_completions=True))
    def test_ledger_SHA_rejects(self):self.bad(lambda r:r['fresh_ledger_invocations'][0].update(reference_sha256='0'*64))
    def test_precision_rejects(self):self.bad(lambda r:r['fresh_ledger_invocations'][0].update(precision='FP32'))
    def test_reuse_rejects(self):self.bad(lambda r:r['fresh_ledger_invocations'][0].update(cross_invocation_receipt_reuse=True))
    def test_scoring_authority_rejects(self):self.bad(lambda r:r.update(production_scoring_executed=True))
    def test_protocol_repeats_rejects(self):self.bad(lambda r:r['identity'].update(bootstrap_repeats=1999))
    def test_source_best_runid_rejects(self):self.bad(lambda r:r['identity']['run_ids'].update(B='M_source_best'))
    def test_partial14_marker_inventory_rejects(self):self.bad(lambda r:r['identity']['cache_completion_sha256'].pop(next(iter(r['identity']['cache_completion_sha256']))))

class Marker(unittest.TestCase):
    def check(self,mutate=None):
        ref,r,m,seal,ri=marker()
        if mutate:mutate(m,seal)
        return a.validate_marker(m,123,'B','B/SOURCE_CAL',r['identity'],ref['pinned_sha256'],ref,seal,ri)
    def test_original_LAST_FP32_header_accepts(self):self.check()
    def bad(self,mutate):
        with self.assertRaises(AssertionError):self.check(mutate)
    def test_source_best_path_rejects(self):self.bad(lambda m,s:m['identity'].update(checkpoint_path='/tmp/source_best.pt'))
    def test_checkpoint_SHA_vs_seal_rejects(self):self.bad(lambda m,s:m['identity'].update(checkpoint_sha256='0'*64))
    def test_FP16_rejects(self):self.bad(lambda m,s:m.update(precision='FP16'))
    def test_autocast_rejects(self):self.bad(lambda m,s:m.update(autocast=True))
    def test_wrong_role_rejects(self):self.bad(lambda m,s:m['identity'].update(role='B/TARGET_TEST'))
    def test_foreign_manifest_rejects(self):self.bad(lambda m,s:m['identity'].update(source_manifest_path='/tmp/role.parquet'))
    def test_stale_registry_rejects(self):self.bad(lambda m,s:m['identity'].update(H_run_registry_sha256='0'*64))
    def test_changed_original_head_code_rejects(self):self.bad(lambda m,s:m['identity'].update(cache_code_sha256='0'*64))
    def test_manifest_footer_rows_rejects(self):
        with self.assertRaises(AssertionError):a.validate_footer(dict(rows=79,columns=list(a.IDENTITY_COLUMNS)),80)
    def test_manifest_footer_columns_rejects(self):
        with self.assertRaises(AssertionError):a.validate_footer(dict(rows=80,columns=['video_uid']),80)

class LogTime(unittest.TestCase):
    def test14_begin_pass_exit0(self):
        log='\n'.join(f'BEGIN_FRESH_ORIGINAL_LEDGER {n} clip_vit_b16 AMP models10_no_receipt_reuse' for n in range(1,15))+'\nPASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT '+str(a.candidate_path(123))+'\n'
        a.validate_log_time(log,'Elapsed (wall clock) time (h:mm:ss or m:ss): 1:40.01\nMaximum resident set size (kbytes): 1000\nExit status: 0\n',123)
    def test_partial_log_rejects(self):
        with self.assertRaises(AssertionError):a.validate_log_time('BEGIN_FRESH_ORIGINAL_LEDGER 1','Exit status: 0',123)
    def test_default_does_no_actual_audit(self):self.assertEqual(a.main([]),0)


def context():
    rows={123:dict(launch=str(a.ROOT/'reports/H_primary_admit_clip_seed123_slot1_launch_20261025T1355.json'),stdout=str(a.ROOT/'reports/H_primary_admit_clip_seed123_slot1_20261025T1355.log'),time_v=str(a.ROOT/'reports/H_primary_admit_clip_seed123_slot1_time_v_20261025T1355.log'),producer_slot=1,producer_pids=[99999991,99999992]),2026:dict(launch=str(a.ROOT/'reports/H_primary_admit_clip_seed2026_slot0_20261025T1410_launch.json'),stdout=str(a.ROOT/'reports/H_primary_admit_clip_seed2026_slot0_20261025T1410.log'),time_v=str(a.ROOT/'reports/H_primary_admit_clip_seed2026_slot0_20261025T1410_time_v.log'),producer_slot=0,producer_pids=[99999993,99999994])}
    for seed,x in rows.items():x.update(candidate=str(a.candidate_path(seed)),launch_sha256='b'*64)
    return dict(schema='H_PARALLEL_PAIR_ADMISSION_AUDIT_CONTEXT_V1',root=str(a.ROOT),backbone='clip_vit_b16',seeds={str(k):v for k,v in rows.items()},shared_source=str(a.SHARED_PATH),shared_source_sha256=a.SHARED_SHA,wrapper_template=str(a.ROOT/'engineering/source_best_M/primary_admit_clip_seed123_slot1.py'),wrapper_template_sha256='cd33c85b28dcf82ea0307c0786603a49cf67014f1406ef0d92668a28a6151fa7')
def launch(seed,c):
    x=c['seeds'][str(seed)];template="output='seed123'\nwith entry.cpu_slot(1):\n print('SHARED_CPU_SLOT_1')\n argv=['--seed','123']\n"
    l=dict(pid=x['producer_pids'][0],CPU_slot=x['producer_slot'],stdout=x['stdout'],time_v=x['time_v'],new_dispatcher=False,repeat_on_failure=False)
    if seed==123:l.update(entry=c['wrapper_template'],entry_sha256=c['wrapper_template_sha256'],original_production_and_F_edited=False)
    else:
        code=template.replace('seed123','seed2026').replace('entry.cpu_slot(1)','entry.cpu_slot(0)').replace('SHARED_CPU_SLOT_1','SHARED_CPU_SLOT_0').replace("'--seed','123'","'--seed','2026'")
        l.update(parameterized_entry_code=code,parameterized_code_sha256=hashlib.sha256(code.encode()).hexdigest(),entry_template=c['wrapper_template'],entry_template_sha256=c['wrapper_template_sha256'],production_or_F_files_changed=False)
    return l,template

class ContextLaunch(unittest.TestCase):
    def load(self,c):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'H_parallel_pair_admission_audit_context.json';p.write_text(json.dumps(c));h=hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(a,'HERE',Path(t)):return a.load_context(h)
    def test_canonical_external_context_accepts(self):self.load(context())
    def test_cross_seed_launch_path_rejects(self):
        c=context();c['seeds']['123']['launch']=c['seeds']['2026']['launch']
        with self.assertRaises(AssertionError):self.load(c)
    def test_foreign_log_path_rejects(self):
        c=context();c['seeds']['123']['stdout']='/tmp/foreign.log'
        with self.assertRaises(AssertionError):self.load(c)
    def test_template123_accepts(self):
        c=context();l,t=launch(123,c)
        with patch.object(Path,'read_text',return_value=t):a.validate_launch(l,123,c)
    def test_exact2026_seed_slot_substitutions_accept(self):
        c=context();l,t=launch(2026,c)
        with patch.object(Path,'read_text',return_value=t):a.validate_launch(l,2026,c)
    def test2026_extra_functionality_rejects(self):
        c=context();l,t=launch(2026,c);l['parameterized_entry_code']+='extra_call()\n';l['parameterized_code_sha256']=hashlib.sha256(l['parameterized_entry_code'].encode()).hexdigest()
        with patch.object(Path,'read_text',return_value=t):
            with self.assertRaises(AssertionError):a.validate_launch(l,2026,c)
    def test_time_nonzero_exit_rejects(self):
        log='\n'.join(f'BEGIN_FRESH_ORIGINAL_LEDGER {n} clip_vit_b16 AMP models10_no_receipt_reuse' for n in range(1,15))+'\nPASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT '+str(a.candidate_path(123))+'\n'
        with self.assertRaises(AssertionError):a.validate_log_time(log,'Exit status: 1\n',123)
    def test_tiny_real_parquet_footer_only(self):
        import pyarrow as pa,pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'tiny.parquet';pq.write_table(pa.table({k:list(range(4)) for k in a.IDENTITY_COLUMNS}),p)
            meta=a.manifest_footer(p);a.validate_footer(meta,4);self.assertEqual(meta['rows'],4)

class MockOwned(unittest.TestCase):
    def fixture(self,seed=123):
        reg,m,ac,ref,r=fixture(seed);c=context();l,template=launch(seed,c);x=c['seeds'][str(seed)]
        ref['protected_review_sha256']={str(a.F/'independent_cpu_ledger_adapter_fix_confirmation.json'):'c'*64}
        data={str(a.REFERENCE):json.dumps(ref),str(a.MANIFEST):json.dumps(m),str(a.ROOT/'H_run_registry.json'):json.dumps(reg),str(a.ALL3_ACCEPTED):json.dumps(ac),x['candidate']:json.dumps(r),x['launch']:json.dumps(l),c['wrapper_template']:template}
        log='\n'.join(f'BEGIN_FRESH_ORIGINAL_LEDGER {n} clip_vit_b16 AMP models10_no_receipt_reuse' for n in range(1,15))+'\nPASS_ACTUAL_CANONICAL_ORIGINAL_PAIR_CONTEXT '+str(a.candidate_path(seed))+'\n'
        data[x['stdout']]=log;data[x['time_v']]='Elapsed (wall clock) time (h:mm:ss or m:ss): 1:40.01\nMaximum resident set size (kbytes): 1000\nExit status: 0\n'
        hashes=dict(ref['pinned_sha256']);hashes.update({str(a.REFERENCE):a.REFERENCE_SHA,str(a.MANIFEST):a.MANIFEST_SHA,str(a.ALL3_ACCEPTED):a.ALL3_SHA,str(a.ROOT/'H_run_registry.json'):a.REGISTRY_SHA,str(a.SHARED_PATH):a.SHARED_SHA,str(Path(a.__file__)):'d'*64,str(a.HERE/'H_parallel_pair_admission_audit_context.json'):'e'*64,x['candidate']:'f'*64,x['time_v']:'a'*64,x['stdout']:'a'*64,x['launch']:'b'*64,c['wrapper_template']:c['wrapper_template_sha256'],str(a.F/'cpu_primary_analysis.py'):'c081c6ffeb14dc6bb0219851fd3fd11df25b28d2eed26ac4b7e06ef1e6e4c6ce',str(a.ROOT/'reports/H_parallel_CPU_actual_qualification_clip_vit_b16_20261025T0830.json'):'0ae309b3366de860cbd77f539c24a05aae3ec1180dbbd84bbbff95ed3111d7f7',**ref['protected_review_sha256']})
        for history in ('B','AB'):
            for role in a.ROLES:
                _,_,mm,seal,ri=marker(seed,role,history);run=r['identity']['run_ids'][history];path=a.ROOT/'features/H'/run/role/'complete.json'
                data[str(path)]=json.dumps(mm);hashes[str(path)]='6'*64
                data[str(a.ROOT/'runs'/run/'identity.json')]=json.dumps(ri);data[str(a.ROOT/'runs'/run/'H_completion_seal.json')]=json.dumps(seal)
        return c,data,hashes
    def invoke(self,c,data,hashes,seed=123):
        with patch.object(a,'load_context',return_value=c),patch.object(a,'sha',side_effect=lambda p:hashes[str(p)]),patch.object(Path,'read_text',autospec=True,side_effect=lambda p,*args,**kw:data[str(p)]),patch.object(a,'manifest_footer',side_effect=lambda p:dict(path=str(p),rows=80,columns=list(a.IDENTITY_COLUMNS),size=123,mtime_ns=456)):
            return a.audit_owned(seed,'e'*64,'f'*64,'a'*64,'a'*64)
    def test_full107_14headers_7footers_receipt_chain123(self):
        c,d,h=self.fixture();r=self.invoke(c,d,h);self.assertEqual(len(r['role_marker_checks']),14);self.assertEqual(len(r['role_manifest_footer_metadata']),7)
    def test_full_same_path2026(self):
        c,d,h=self.fixture(2026);r=self.invoke(c,d,h,2026);self.assertEqual(r['seed'],2026)
    def test_changed_pin_bytes_rejects(self):
        c,d,h=self.fixture();h[str(a.CODE/'reliability_v2/cache_h.py')]='0'*64
        with self.assertRaises(AssertionError):self.invoke(c,d,h)
    def test_changed_marker_bytes_rejects(self):
        c,d,h=self.fixture();h[str(a.ROOT/'features/H/H_formal_clip_vit_b16_B_seed123/B/SOURCE_CAL/complete.json')]='0'*64
        with self.assertRaises(AssertionError):self.invoke(c,d,h)

if __name__=='__main__':unittest.main()

