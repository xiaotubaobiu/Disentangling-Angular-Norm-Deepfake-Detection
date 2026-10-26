from pathlib import Path
import json,hashlib,datetime,sys
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M';V=E/'code/studies/pipeline_reliability/v2';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
output=Path(sys.argv[1]);assert output.is_absolute() and output.parent==E and not output.exists()
registry_path=R/'H_run_registry.json';reg=json.loads(registry_path.read_text());assert reg['target_export_allowed'] is True and reg['status']=='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING'
dproof=R/'reports/D_formal_terminal_manifest_current.json';d=json.loads(dproof.read_text());assert d['status']=='PASS_ALL_FIFTEEN_D_FORMAL_TRAINING_TERMINALS_ONLY' and sha(d['acceptance_audit'])==d['acceptance_audit_sha256']
records={};dcfg=R/'configs/execution.json';hc=R/'configs/H_execution.json';configD=json.loads(dcfg.read_text());configH=json.loads(hc.read_text())
for row in d['completed_formal']:
 run=row['run_id'];task=configD['runs'][run];assert task['phase']=='formal' and task['route']=='D' and row['source_best_equal_saved_state_model'] and row['final_equal_terminal_state_model'];folder=R/'runs'/run
 for name in ['identity.json','resolved_config.json','history.json','complete.json']:assert sha(folder/name)==row['core_file_inventory'][name]['sha256']
 complete=json.loads((folder/'complete.json').read_text());assert complete['source_best_sha256']==row['core_file_inventory']['source_best.pt']['sha256'] and complete['identity']==row['identity']
 records[run]=dict(route='D',config_relative='configs/execution.json',config_sha256=sha(dcfg),proof_relative=str(dproof.relative_to(R)),proof_sha256=sha(dproof))
inv=R/'reports/H30_independent_terminal_audit_inventory_20261025T0520.json';inventory=json.loads(inv.read_text());heffective={x for item in reg['effective_runs'].values() for x in item['run_ids']}
for proof_path,pin in inventory.items():
 proof=Path(proof_path);assert sha(proof)==pin;a=json.loads(proof.read_text());assert a['status']=='PASS_SELECTED_H_FORMAL_AMP_TERMINAL_FULL_CHECKPOINTS_SOURCE_DEV_AND_PAIRED_STREAMS'
 for row in a['records']:
  run=row['run_id']
  if run not in heffective:continue
  task=configH['runs'][run];assert task['phase']=='formal' and task['route']=='H' and row['source_best_matches_best_state'] and row['final_matches_complete_state'];folder=R/'runs'/run
  sealp=folder/'H_completion_seal.json';assert sha(sealp)==row['terminal_seal_sha256'];seal=json.loads(sealp.read_text());assert seal['h_code_bundle_sha256']==task['h_code_bundle_sha256'] and seal['H_data_freeze_sha256']==task['H_data_freeze_sha256']
  for name in ['identity.json','resolved_config.json','history.json','complete.json']:assert sha(folder/name)==seal['artifact_sha256'][name]
  complete=json.loads((folder/'complete.json').read_text());assert complete['source_best_sha256']==seal['artifact_sha256']['source_best.pt']
  assert run not in records;records[run]=dict(route='H',config_relative='configs/H_execution.json',config_sha256=sha(hc),proof_relative=str(proof.relative_to(R)),proof_sha256=pin)
assert len(records)==45 and sum(x['route']=='D' for x in records.values())==15 and sum(x['route']=='H' for x in records.values())==30
payload=dict(schema='M_SOURCE_BEST_FILE_ADMISSION_V1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifact_root=str(R),target_inference_authorized=False,endpoint_code_sha256=sha(V/'reliability_v2/source_best_files.py'),H_registry=dict(relative='H_run_registry.json',sha256=sha(registry_path)),records=records,scope='Actual accepted45 primary source-only checkpoints staging. This input list binds producer proofs and selected files; current bytes/tensors/head validation still required. No target inference/cache authorization or resource savings.')
with output.open('x') as f:json.dump(payload,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status='ACTUAL45_SOURCE_INPUT_ADMISSION_PREPARED_FILE_TENSOR_VALIDATION_PENDING',path=str(output),sha256=sha(output),models=45,endpoint_code_sha256=payload['endpoint_code_sha256'])))
