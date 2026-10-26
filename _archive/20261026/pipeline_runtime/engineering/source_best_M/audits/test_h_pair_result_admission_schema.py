"""Mock-only pinned admission schema integration; no real files/slots/data read."""
import copy,unittest
from unittest.mock import patch
import h_pair_result_audit as a

class AdmissionSchema(unittest.TestCase):
    def fixture(self):
        ledger=a.ROOT/'reports/H_clip_vit_b16_AMP_paired_ledger.json'
        identity=dict(backbone='clip_vit_b16',effective_precision='AMP',seed=42,
            run_ids={h:f'H_formal_clip_vit_b16_{h}_seed42' for h in ('B','AB')},
            inputs_sha256={str(ledger):'1'*64},code_sha256={'analysis_h.py':'2'*64},cache_completion_sha256={})
        pins={str(a.ADMISSION):a.ADMISSION_SHA,str(a.ADMISSION_AUDIT):a.ADMISSION_AUDIT_SHA,
            str(a.LAUNCH):a.LAUNCH_SHA,str(a.PAIR/'identity.json'):a.IDENTITY_SHA,
            str(ledger):'1'*64,str(a.ORIGINAL/'analysis_h.py'):'2'*64}
        ledgers=[dict(backbone='clip_vit_b16',precision='AMP',fresh_source_completions=10,
            exact_original_ledger_except_top_utc=True,reference_sha256='1'*64,elapsed_seconds=200.,cross_invocation_receipt_reuse=False) for _ in range(14)]
        ad=dict(status='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS',identity=identity,
            roles=14,models=2,fresh_ledger_invocations=ledgers,all_source_completions_fresh=140,
            cross_invocation_receipt_reuse=False,pinned_sha256={})
        audit=dict(independent_review_pass=True,actual_CLIP_original14role_fresh140ledger_source_receipt_audit_passed=True,
            receipt_sha256=a.ADMISSION_SHA,original_primary_pair_identity=identity,pinned_sha256={},current_role_marker_checks=[])
        launch=dict(admission_sha256=a.ADMISSION_SHA,backbone='clip_vit_b16',seed=42,bootstrap_repeats=2000,GPUs_allowed=False)
        data={str(a.ADMISSION):ad,str(a.ADMISSION_AUDIT):audit,str(a.LAUNCH):launch}
        for history in ('B','AB'):
            run=identity['run_ids'][history]
            for role in a.ROLES:
                key=history+'/'+role;digest='3'*64;identity['cache_completion_sha256'][key]=digest
                path=a.ROOT/'features/H'/run/role/'complete.json';manifest=a.ROOT/'data/H'/f'{role}.parquet'
                pins[str(path)]=digest;pins[str(manifest)]='4'*64
                audit['current_role_marker_checks'].append(dict(history=history,run_id=run,role=role,
                    complete_sha256=digest,original_head_sha256='5'*64,rows=80,accepted_videos=10))
                data[str(path)]=dict(status='CACHE_COMPLETE',precision='FP32',autocast=False,
                    input_validation='FRESH_PROCESSED_IMAGE_BYTE_AND_DECODED_RGB_HASHES',rows=80,
                    identity=dict(run_id=run,role=role,history=history,effective_precision='AMP',
                        checkpoint_path=str(a.ROOT/'runs'/run/'final.pt'),original_head_sha256='5'*64,accepted_table_videos=10,
                        source_manifest_path=str(manifest),source_manifest_sha256='4'*64))
        return identity,ad,data,pins
    def invoke(self,identity,data,pins):
        with patch.object(a,'sha',side_effect=lambda p:pins[str(p)]),patch.object(a,'read_json',side_effect=lambda p:data[str(p)]):return a.pin_metadata(identity)
    def test_actual_producer_integer140_and14_times10_accepts(self):
        i,ad,data,pins=self.fixture();out,records=self.invoke(i,data,pins);self.assertEqual(len(records),14)
    def test_boolean_true_is_not_integer140(self):
        i,ad,data,pins=self.fixture();ad['all_source_completions_fresh']=True
        with self.assertRaises(AssertionError):self.invoke(i,data,pins)
    def test_float140_is_not_typed_integer(self):
        i,ad,data,pins=self.fixture();ad['all_source_completions_fresh']=140.
        with self.assertRaises(AssertionError):self.invoke(i,data,pins)
    def test_incorrect_total139_rejects(self):
        i,ad,data,pins=self.fixture();ad['all_source_completions_fresh']=139
        with self.assertRaises(AssertionError):self.invoke(i,data,pins)
    def test_each_ledger_typed10_required(self):
        for bad in (9,10.,True,'10'):
            with self.subTest(bad=bad):
                i,ad,data,pins=self.fixture();ad['fresh_ledger_invocations'][0]['fresh_source_completions']=bad
                with self.assertRaises(AssertionError):self.invoke(i,data,pins)
    def test_exact14_invocations_required(self):
        i,ad,data,pins=self.fixture();ad['fresh_ledger_invocations'].pop()
        with self.assertRaises(AssertionError):self.invoke(i,data,pins)
    def test_original_ledger_hash_required(self):
        i,ad,data,pins=self.fixture();ad['fresh_ledger_invocations'][0]['reference_sha256']='9'*64
        with self.assertRaises(AssertionError):self.invoke(i,data,pins)
    def test_effective_precision_required(self):
        i,ad,data,pins=self.fixture();ad['fresh_ledger_invocations'][0]['precision']='FP32'
        with self.assertRaises(AssertionError):self.invoke(i,data,pins)
    def test_actual_backbone_required(self):
        i,ad,data,pins=self.fixture();ad['fresh_ledger_invocations'][0]['backbone']='dinov2_vit_b14'
        with self.assertRaises(AssertionError):self.invoke(i,data,pins)
    def test_exact_original_ledger_flag_required(self):
        i,ad,data,pins=self.fixture();ad['fresh_ledger_invocations'][0]['exact_original_ledger_except_top_utc']=False
        with self.assertRaises(AssertionError):self.invoke(i,data,pins)
    def test_cross_invocation_reuse_rejects_per_ledger(self):
        i,ad,data,pins=self.fixture();ad['fresh_ledger_invocations'][0]['cross_invocation_receipt_reuse']=True
        with self.assertRaises(AssertionError):self.invoke(i,data,pins)
    def test_original_pair_code_pin_still_enforced(self):
        i,ad,data,pins=self.fixture();pins[str(a.ORIGINAL/'analysis_h.py')]='9'*64
        with self.assertRaises(AssertionError):self.invoke(i,data,pins)

if __name__=='__main__':unittest.main()
