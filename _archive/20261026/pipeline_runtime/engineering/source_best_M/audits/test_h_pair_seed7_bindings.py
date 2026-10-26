"""Pure/mock metadata preparation tests only; never acquire real locks or read real results."""
import copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import h_pair_result_audit as a

PATHS=dict(identity=a.ROOT/'analysis/H/clip_vit_b16/seed7/identity.json',admission=a.ROOT/'reports/H_clip_seed7_independent_7role_admission_20261025T0750.json',admission_acceptance=a.ROOT/'reports/H_clip_seed7_independent_cache_admission_acceptance_20261025T0950.json',admission_launch=a.ROOT/'reports/H_clip_seed7_independent_7role_admission_launch_20261025T0750.json',admission_entry=a.ROOT/'qualification/H_primary_cache_independent_admission/admit_clip_seed7_20261025T0750.py',primary_launch=a.ROOT/'reports/H_primary_analysis_clip_vit_b16_seed7_launch_20261025T0950.json',primary_execution=a.ROOT/'reports/H_primary_analysis_execution_clip_vit_b16_seed7.json',source_manifest=a.ROOT/'qualification/H_primary_cache_continuation/qualified_source_manifest.json',CPU_qualification=a.ROOT/'reports/H_parallel_CPU_actual_qualification_clip_vit_b16_20261025T0830.json',primary_entry=a.ROOT/'qualification/H_parallel_fresh_validation/cpu_primary_analysis.py')
def spec():return dict(schema='H_LAST_PAIR_AUDIT_BINDINGS_SERIAL7_V1',root=str(a.ROOT),backbone='clip_vit_b16',seed=7,evidence_route='ORIGINAL_SERIAL14_ACCEPTED',source_checkpoint='original_LAST_final.pt',bootstrap_repeats=2000,resampling_seed=20261023,proofs={k:dict(path=str(p),sha256='1'*64) for k,p in PATHS.items()})
def receipts():
    b=spec();i=dict(backbone='clip_vit_b16',seed=7,effective_precision='AMP',inputs_sha256={str(a.ROOT/'reports/H_clip_vit_b16_AMP_paired_ledger.json'):'2'*64})
    ad=dict(status='PASS_ORIGINAL_FULL_14_ROLE_PAIR_CONTEXT_INDEPENDENT_OF_CACHE_WORKERS',identity=i,models=2,roles=14,backbone='clip_vit_b16',seed=7,analysis_executed=False,GPU_launch_attempted=False,qualified_source_manifest_sha256='1'*64)
    ac=dict(status='PASS_ORIGINAL_SERIAL7_PAIR_FULL14ROLE_COMPLETION_CURRENT_CODE_INPUT_MARKER_REBIND',models=2,roles=14,backbone='clip_vit_b16',seed=7,independent_from_cache_workers=True,source_report=str(PATHS['admission']),source_report_sha256='1'*64,complete_marker_hashes_checked=14,primary_statistics_or_metrics_generated=False)
    al=dict(entry=str(PATHS['admission_entry']),entry_sha256='1'*64,science_function='unmodified canonical analysis_h._pair_context',backbone='clip_vit_b16',seed=7,production_scoring_started=False,GPU_launch_attempted=False)
    pl=dict(entry=str(PATHS['primary_entry']),entry_sha256='1'*64,backbone='clip_vit_b16',seed=7,independent_pair_admission=str(PATHS['admission']),admission_sha256='1'*64,qualification=str(PATHS['CPU_qualification']),qualification_sha256='1'*64,bootstrap_repeats=2000,GPUs_allowed=False,M_statistics_or_cache_started=False)
    ex=dict(status='ORIGINAL_PRIMARY_H_ANALYSIS_EXECUTION_COMPLETE_REQUIRES_INDEPENDENT_METRIC_BOOTSTRAP_AUDIT',backbone='clip_vit_b16',seed=7,pair_admission_sha256='1'*64,source_qualification_sha256='1'*64,original_analysis_identity=i,original_analysis_complete_sha256='3'*64,GPU_launch_attempted=False,primary_cache_owner_modified=False,M_results_generated=False,scientific_final_acceptance=False,
        fresh_ledger_invocations=[dict(backbone='clip_vit_b16',precision='AMP',fresh_source_completions=10,exact_original_ledger_except_top_utc=True,reference_sha256='2'*64,cross_invocation_receipt_reuse=False) for _ in range(42)])
    q=dict(schema='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1',status='PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE',allowed_backbones=['clip_vit_b16'],independent_review_pass=True,production_GPU_allowed=False,original_source_manifest_sha256='1'*64)
    return b,i,dict(admission=ad,admission_acceptance=ac,admission_launch=al,primary_launch=pl,primary_execution=ex,CPU_qualification=q)

class BindingSpec(unittest.TestCase):
    def load(self,b,wrong_sha=False):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'H_LAST_clip_seed7_audit_bindings.json';p.write_text(json.dumps(b));h=hashlib.sha256(p.read_bytes()).hexdigest()
            with patch.object(a,'HERE',Path(t)):return a.load_bindings(p,'0'*64 if wrong_sha else h)
    def test_actual_shaped_serial_spec_accepts(self):self.assertEqual(self.load(spec())['seed'],7)
    def test_external_sha_mismatch_rejects(self):
        with self.assertRaises(AssertionError):self.load(spec(),True)
    def test_noncanonical_root_rejects(self):
        b=spec();b['root']='/tmp/foreign'
        with self.assertRaises(AssertionError):self.load(b)
    def test_seed42_evidence_cannot_be_renamed7(self):
        b=spec();b['proofs']['admission']['path']=str(a.ADMISSION)
        with self.assertRaises(AssertionError):self.load(b)
    def test_unsupported_seed_rejects(self):
        b=spec();b['seed']=42
        with self.assertRaises(AssertionError):self.load(b)
    def test_boolean_seed_rejects(self):
        b=spec();b['seed']=True
        with self.assertRaises(AssertionError):self.load(b)
    def test_parallel_schema_cannot_be_silent_serial(self):
        b=spec();b['evidence_route']='PARALLEL14'
        with self.assertRaises(AssertionError):self.load(b)
    def test_missing_or_extra_proof_rejects(self):
        for extra in (False,True):
            b=spec()
            if extra:b['proofs']['foreign']=dict(path='/tmp/x',sha256='1'*64)
            else:del b['proofs']['admission_entry']
            with self.assertRaises(AssertionError):self.load(b)
    def test_nonsha_proof_rejects(self):
        b=spec();b['proofs']['identity']['sha256']='x'
        with self.assertRaises(AssertionError):self.load(b)

class SerialEvidence(unittest.TestCase):
    def test_actual_shaped_serial_no_parallel_fields_accepts(self):
        b,i,d=receipts();self.assertNotIn('pinned_sha256',d['admission']);self.assertNotIn('fresh_ledger_invocations',d['admission']);a.validate_serial7_evidence(b,i,d,'3'*64)
    def reject(self,change):
        b,i,d=receipts();change(b,i,d)
        with self.assertRaises(AssertionError):a.validate_serial7_evidence(b,i,d,'3'*64)
    def test_wrong_serial_status_rejects(self):self.reject(lambda b,i,d:d['admission'].update(status='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS'))
    def test_wrong_admission_sha_rejects(self):self.reject(lambda b,i,d:d['admission_acceptance'].update(source_report_sha256='9'*64))
    def test_wrong_serial_entry_sha_rejects(self):self.reject(lambda b,i,d:d['admission_launch'].update(entry_sha256='9'*64))
    def test_missing_cache_independence_rejects(self):self.reject(lambda b,i,d:d['admission_acceptance'].update(independent_from_cache_workers=False))
    def test_only13_role_markers_rejects(self):self.reject(lambda b,i,d:d['admission_acceptance'].update(complete_marker_hashes_checked=13))
    def test_execution_wrong_identity_rejects(self):self.reject(lambda b,i,d:d['primary_execution'].update(original_analysis_identity={'seed':42}))
    def test_execution_wrong_terminal_rejects(self):self.reject(lambda b,i,d:d['primary_execution'].update(original_analysis_complete_sha256='9'*64))
    def test_execution_only41_ledgers_rejects(self):self.reject(lambda b,i,d:d['primary_execution']['fresh_ledger_invocations'].pop())
    def test_execution_boolean10_rejects(self):self.reject(lambda b,i,d:d['primary_execution']['fresh_ledger_invocations'][0].update(fresh_source_completions=True))
    def test_execution_changed_ledger_hash_rejects(self):self.reject(lambda b,i,d:d['primary_execution']['fresh_ledger_invocations'][0].update(reference_sha256='9'*64))
    def test_execution_changed_precision_rejects(self):self.reject(lambda b,i,d:d['primary_execution']['fresh_ledger_invocations'][0].update(precision='FP32'))
    def test_execution_reuse_rejects(self):self.reject(lambda b,i,d:d['primary_execution']['fresh_ledger_invocations'][0].update(cross_invocation_receipt_reuse=True))
    def test_GPU_authority_does_not_leak(self):self.reject(lambda b,i,d:d['primary_execution'].update(GPU_launch_attempted=True))
    def test_M_authority_does_not_leak(self):self.reject(lambda b,i,d:d['primary_execution'].update(M_results_generated=True))
    def test_mismatched_backbone_rejects(self):self.reject(lambda b,i,d:d['primary_launch'].update(backbone='dinov2_vit_b14'))


class SerialPinIntegration(unittest.TestCase):
    def fixture(self):
        from test_h_pair_result_admission_schema import AdmissionSchema
        i,_,old_data,_=AdmissionSchema().fixture()
        i=json.loads(json.dumps(i).replace('_seed42','_seed7'));i['seed']=7
        b,_,d=receipts();b['_spec_path']=str(a.HERE/'H_LAST_clip_seed7_audit_bindings.json');b['_spec_sha256']='8'*64
        b['proofs']['source_manifest']['sha256']='977431237b901d5ac88172f98bc2591d3cd7992c35be50546cda34ad2276a747'
        d['admission']['identity']=i;d['admission']['qualified_source_manifest_sha256']=b['proofs']['source_manifest']['sha256']
        d['CPU_qualification']['original_source_manifest_sha256']=b['proofs']['source_manifest']['sha256']
        d['primary_execution']['original_analysis_identity']=i
        for row in d['primary_execution']['fresh_ledger_invocations']:row['reference_sha256']='1'*64
        names=('__init__','analysis_d','analysis_h','bootstrap_fast','cache_d','cache_h','coverage_h','data_d','data_h','geometry_h','models','qualification_h','qualify','queue_d','queue_h','scoring_d','stats_h','summary_d','trainer','worker_h')
        source={f'reliability_v2/{n}.py':('2'*64 if n=='analysis_h' else '6'*64) for n in names}
        d['source_manifest']=dict(qualified_commit='5ef6d06bf7aeda02a5a11113e4c35a2c9909f215',source_sha256=source,
            original_H_posttraining_acceptance_path=str(a.ROOT/'reports/H_posttraining_implementation_acceptance.json'),original_H_posttraining_acceptance_sha256='9'*64)
        ep={str(a.ORIGINAL.parent/n):h for n,h in source.items()};ep.update({str(a.ROOT/f'reports/synthetic_pin{n}.json'):'7'*64 for n in range(90)})
        d['primary_execution']['pinned_sha256']=ep;d['identity']=i
        data={b['proofs'][k]['path']:v for k,v in d.items()};pins={v['path']:v['sha256'] for v in b['proofs'].values()}
        pins.update(ep);pins[b['_spec_path']]=b['_spec_sha256'];pins.update(i['inputs_sha256'])
        pins[d['source_manifest']['original_H_posttraining_acceptance_path']]='9'*64
        for path,m in old_data.items():
            if '/features/H/' not in path:continue
            path=path.replace('_seed42','_seed7');m=json.loads(json.dumps(m).replace('_seed42','_seed7'))
            m['dimension']=512;m['identity'].update(backbone='clip_vit_b16',dimension=512,accepted_table_rows=80)
            data[path]=m;pins[path]='3'*64;pins[m['identity']['source_manifest_path']]='4'*64
        return b,i,data,pins
    def invoke(self,b,i,data,pins):
        with patch.object(a,'sha',side_effect=lambda p:pins[str(p)]),patch.object(a,'read_json',side_effect=lambda p:data[str(p)]):return a.pin_metadata(i,b,'3'*64)
    def test_full_mock_serial20sources110pins14markers_accepts(self):
        b,i,d,p=self.fixture();pins,records=self.invoke(b,i,d,p);self.assertEqual(len(records),14);self.assertIn(b['_spec_path'],pins)
    def reject(self,change):
        b,i,d,p=self.fixture();change(b,i,d,p)
        with self.assertRaises(AssertionError):self.invoke(b,i,d,p)
    def test_bound_spec_drift_rejects(self):self.reject(lambda b,i,d,p:p.update({b['_spec_path']:'9'*64}))
    def test_raw_proof_drift_rejects(self):self.reject(lambda b,i,d,p:p.update({b['proofs']['admission']['path']:'9'*64}))
    def test_original_source_drift_rejects(self):self.reject(lambda b,i,d,p:p.update({str(a.ORIGINAL/'analysis_h.py'):'9'*64}))
    def test_execution_pin_inventory_missing_rejects(self):self.reject(lambda b,i,d,p:d[b['proofs']['primary_execution']['path']]['pinned_sha256'].pop(str(a.ROOT/'reports/synthetic_pin0.json')))
    def test_source_manifest_19_rejects(self):self.reject(lambda b,i,d,p:d[b['proofs']['source_manifest']['path']]['source_sha256'].pop('reliability_v2/worker_h.py'))
    def test_marker_sha_drift_rejects(self):self.reject(lambda b,i,d,p:p.update({str(a.ROOT/'features/H/H_formal_clip_vit_b16_B_seed7/B/SOURCE_CAL/complete.json'):'9'*64}))
    def test_source_best_checkpoint_cannot_replace_LAST(self):
        self.reject(lambda b,i,d,p:d[str(a.ROOT/'features/H/H_formal_clip_vit_b16_B_seed7/B/SOURCE_CAL/complete.json')]['identity'].update(checkpoint_path=str(a.ROOT/'runs/H_formal_clip_vit_b16_B_seed7/source_best.pt')))
    def test_wrong_role_header_rejects(self):
        self.reject(lambda b,i,d,p:d[str(a.ROOT/'features/H/H_formal_clip_vit_b16_B_seed7/B/SOURCE_CAL/complete.json')]['identity'].update(role='B/TARGET_TEST'))
    def test_FP16_cache_rejects(self):
        self.reject(lambda b,i,d,p:d[str(a.ROOT/'features/H/H_formal_clip_vit_b16_B_seed7/B/SOURCE_CAL/complete.json')].update(precision='FP16'))
    def test_head_drift_across_roles_rejects(self):
        self.reject(lambda b,i,d,p:d[str(a.ROOT/'features/H/H_formal_clip_vit_b16_B_seed7/B/SOURCE_CAL/complete.json')]['identity'].update(original_head_sha256='9'*64))

if __name__=='__main__':unittest.main()

