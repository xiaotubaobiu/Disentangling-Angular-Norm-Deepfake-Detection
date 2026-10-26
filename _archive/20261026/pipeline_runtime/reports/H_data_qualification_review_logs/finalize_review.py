import json, pathlib, re, sys, hashlib, datetime, os
V=pathlib.Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
R=pathlib.Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
sys.path.insert(0,str(V))
from reliability_v2 import trainer, worker_h
def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
main_files=['reliability_v2/data_h.py','reliability_v2/qualification_h.py','tests/test_data_h.py','tests/test_qualification_h.py']
deps=['reliability_v2/geometry_h.py','reliability_v2/trainer.py','reliability_v2/models.py','reliability_v2/worker_h.py','reliability_v2/queue_h.py','reliability_v2/queue_d.py']
hashes={n:sha(V/n) for n in main_files+deps}
assert hashes['reliability_v2/data_h.py']=='ad81f65c27d53159e6902bf8a20b0910cd129f75eb0105e5a5a053bcc2d2f45e'
assert hashes['reliability_v2/qualification_h.py']=='4581ece963c8196422b718425f035d1e96b35b35c1e0d29025c7b2065bc23c18'
assert hashes['reliability_v2/geometry_h.py']=='161ac4e837463ff16e7916ca42eeb5d13acc44aab9bcc1c397025b9426ad9532'
focused=R/'reports/H_data_qualification_review_final_focused.log'
independent=R/'reports/H_data_qualification_review_final_independent.log'
assert re.search(r'\b10 passed\b',focused.read_text())
checks=json.loads(independent.read_text())
assert len(checks['tests'])==5 and all(x['status']=='PASS' for x in checks['tests'])
assert checks['cuda_visible_devices']=='' and checks['cuda_initialized'] is False
for name,h in checks['code_hashes'].items():assert hashes['reliability_v2/'+name]==h
assert not (R/'data/H/freeze.json').exists()
assert not (R/'qualification/H_gate.json').exists()
receipt={
 'schema':'H_DATA_QUALIFICATION_IMPLEMENTATION_REVIEW_V1',
 'status':'PASS_CODE_REVIEW_ONLY',
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'reviewer':{'identity':'/root/h_data_qual_review','independent':True},
 'scope':'Read-only implementation/source-qualification review and CPU-only synthetic checks after parent announced code freeze. This is not the H_DATA_INDEPENDENT_REVIEW_V1 production data approval.',
 'blockers':[],
 'reviewed_files_sha256':{n:hashes[n] for n in main_files},
 'dependency_snapshot_sha256':{n:hashes[n] for n in deps},
 'training_code_sha256':trainer.code_hash(),
 'h_code_bundle_sha256':worker_h.h_code_hash(),
 'frozen_plan_sha256':sha(R/'configs/plan_frozen_input.md'),
 'resolved_findings':[
  {'id':'H-DQ-001','severity':'Important','location':'reliability_v2/qualification_h.py:_source',
   'trigger':'Canonical geometry FFPP table rows use domain=source; old qualifier required domain=ff.',
   'consequence':'prepare_probe rejected all actual accepted source TRAIN/DEV/CAL tables.',
   'resolution':'Qualifier and fixtures use canonical source domain; geometry-to-qualification interface regression added and independent final check passes.'},
  {'id':'H-DQ-002','severity':'Important','location':'reliability_v2/qualification_h.py:_verify_small_evidence and validate_qualification',
   'trigger':'Earlier shallow proof validation used only size/mtime for saved small DEV/CAL/metrics/array evidence.',
   'consequence':'Same-size byte mutation with restored mtime could escape fresh default validation.',
   'resolution':'Default now cryptographically rehashes every proof file up to 8 MiB and checks full file inventory, while receipt hashes and accepted stat fingerprints remain bound. Same-size metrics mutation with restored mtime is rejected by final regression.'}
 ],
 'verified_implementation_contracts':[
  'Production data gate requires exactly 8431 complete candidate manifests and exactly 560 SOURCE_UNUSED records, fresh current geometry code/protocol audit hashes, source-video byte hashes, processed metadata inventory and mother/crop/mask artifact inventory.',
  'Full inspection reconstructs SOURCE_UNUSED-inclusive source/RGB graph closure and quarantine, reconstructs all thirteen active tables exactly, checks audit disposition, active-role source/RGB isolation, and excludes quarantined/SOURCE_UNUSED videos from every active table.',
  'Common TRAIN/DEV B and AB tables retain identical ordered logical mothers, labels, method contributions and .5 real/.125 per fake mass with equal video/frame weights; asymmetric dedup is rejected.',
  'All-B deployment and common-B/common-AB CAL/source/target cohorts are separately bound; common-B is an exact all-B subset.',
  'Production freeze requires external independent hash-bound PASS and explicit quarantine disposition when historical conflicts exist; this review does not supply that approval.',
  'Source probes select 8 real videos plus 2 per fake method, with 8 mothers each: 128 TRAIN rows, 64 real/64 fake. DEV is source-only common cohort, B SOURCE_CAL is real-only and never enters two-class AUC.',
  'Actual weighted sampler, four-worker loader, mother-keyed augmentation, continuous two epochs versus one epoch plus resume compare parameters, optimizer, scaler/RNG/state/history, draws/methods/augmentation arrays/batches and FP32 Raw features byte-for-byte.',
  'Expected Raw feature dimensions are 512/768/1280; all encoder/head parameters must be trainable and receive gradients. CPU runtime tests exercise a toy 512-dimensional encoder in both histories; production pretrained AMP/FP32 proofs remain separate.',
  'Config/data/code/initialization identity changes reject checkpoint recovery; qualification acceptance requires six current two-history receipts for three backbones and AMP/FP32, with 32->16->8 fallback permitted only after bound OOM evidence.',
  'validate_h_freeze default performs strict fresh metadata/code/review/audit/table/processed-manifest hashing, while deep=True rechecks all actual source/PNG bytes and reconstructs cohorts. Ordinary workers do not perform a full production source-video/PNG scan.',
  'Qualification acceptance deep-verifies H data and all proof artifacts; default runtime qualification rehashes receipts, selected source PNGs and small proofs, checks accepted large-proof size/mtime, and does not deserialize checkpoints.'
 ],
 'evidence':[
  {'path':str(focused),'sha256':sha(focused),'result':'10 passed in 32.48s after announced final code freeze'},
  {'path':str(independent),'sha256':sha(independent),'result':'5 independent checks PASS; CUDA_VISIBLE_DEVICES empty and torch CUDA uninitialized'},
  {'path':str(R/'reports/H_data_qualification_review_pytest_run3.log'),
   'sha256':sha(R/'reports/H_data_qualification_review_pytest_run3.log'),
   'result':'Supporting prior snapshot only: 52 passed in 107.00s; final patch coverage is the focused log above.'}
 ],
 'independent_checks':checks['tests'],
 'earlier_runs_not_used_for_final_pass':[
  {'path':str(R/'reports/H_data_qualification_review_pytest.log'),'reason':'Review harness initially omitted parent temporary directory; no implementation conclusion drawn.'},
  {'path':str(R/'reports/H_data_qualification_review_pytest_run2.log'),'reason':'48 passed, one correctly rejected data_h.py changing mid-inspection; superseded by final frozen-code checks.'}
 ],
 'production_requirements_still_separate':[
  'Wait for all H pixel preprocessing to complete, then run a fresh process full 8431-candidate audit using current immutable geometry code.',
  'Perform a new full production artifact/cohort review and bind its exact candidate/audit/table/code/protocol hashes before any H data freeze.',
  'Only after that data freeze may real source-only H AMP and FP32 qualification run for all three pretrained backbones; all required receipts must pass before H gate acceptance.'
 ],
 'limitations':[
  'This report does not certify the production PNG population or complete DFD ancestry; DFD filename/source proxies remain conditional.',
  'No production H inspect/freeze/review PASS, production H gate or GPU qualification was run by this reviewer.',
  'Default validation of proof files larger than 8 MiB is accepted size/mtime freshness, not fresh cryptographic byte verification; deep validation rehashes and verifies those files.'
 ],
 'production_state_at_report':{'H_data_freeze_exists':False,'H_gate_exists':False},
 'execution_safety':{'source_code_edits':False,'production_config_edits':False,'GPU_allocation':False,'GPU4_7_access':False,
  'D_jobs_changed':False,'H_pixel_jobs_changed':False,'Jev_or_automation_changed':False,'writes_only_authorized_report_and_review_logs':True}
}
path=R/'reports/H_data_qualification_independent_review.json'
path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
assert json.loads(path.read_text())==receipt
print(json.dumps({'report':str(path),'sha256':sha(path),'status':receipt['status'],'blockers':receipt['blockers'],
 'reviewed_files_sha256':receipt['reviewed_files_sha256'],'h_code_bundle_sha256':receipt['h_code_bundle_sha256']},indent=2))

