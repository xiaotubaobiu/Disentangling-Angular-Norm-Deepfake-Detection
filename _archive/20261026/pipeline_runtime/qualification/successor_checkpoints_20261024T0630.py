from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,os
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
spec=importlib.util.spec_from_file_location('accepted_health',R/'qualification/checkpoint_health_20261024T0630_terminal.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.torch.cuda.is_initialized()
config=json.loads((R/'configs/execution.json').read_text());queue=json.loads((R/'queue_status.json').read_text());basepath=R/'reports/checkpoint_health_20261024T0630_terminal.json';base=json.loads(basepath.read_text());auditpath=R/'reports/D_completed_artifact_audit_20261024T0630_terminal.json';audit=json.loads(auditpath.read_text());accepted={x['run_id']:x for x in audit['completed']};assert audit['status']=='PASS'
hashes={k:m.digest(p) for k,p in {'train':R/'data/D/train.parquet','dev':R/'data/D/dev.parquet','environment':R/'configs/requirements.execution.lock.txt','gate':R/'qualification/D_gate.json',**{k:Path(v['weight_path']) for k,v in config['backbones'].items()}}.items()}
results=[];cutoffs={};inherited=[];fresh=[];stat_bindings=[]
for row in base['checks']:
 if row['lifecycle_at_inspection']!='COMPLETE':continue
 run=row['run_id'];out=R/'runs'/run;prior=accepted[run];assert m.digest(out/'complete.json')==prior['core_file_inventory']['complete.json']['sha256'];assert row['checkpoint_sha256']==prior['core_file_inventory']['state.pt']['sha256']
 st=(out/'state.pt').stat();assert st.st_ino==row['checkpoint_inode'] and st.st_size==row['checkpoint_bytes']
 row={**row,'inspection_scope_this_round':'INHERITED_TERMINAL_CPU_PROOF_BOUND_TO_CURRENT_ACCEPTED_AUDIT_AND_UNCHANGED_STATE_INODE_SIZE_COMPLETION_SHA'};results.append(row);inherited.append(run);cutoffs[run]=base['inspection_cutoffs_utc'][run];stat_bindings.append({'run_id':run,'current_state_inode':st.st_ino,'current_state_bytes':st.st_size,'current_complete_sha256':prior['core_file_inventory']['complete.json']['sha256'],'full_state_sha_last_verified_in_terminal_audit':prior['core_file_inventory']['state.pt']['sha256']})
for x in queue['active']:
 item=dict(**x,process_required=True,lifecycle_at_inspection='RUNNING');assert (R/'runs'/x['run_id']/'state.pt').is_file(),'FIRST_EPOCH_STILL_PENDING:'+x['run_id']
 row=m.inspect_run(item,config,hashes);row['inspection_scope_this_round']='FRESH_FULL_CPU_CURRENT_ACTIVE_RECOVERY_POINT';results.append(row);fresh.append(x['run_id']);cutoffs[x['run_id']]=datetime.now(timezone.utc).isoformat();print(json.dumps({k:row[k] for k in ['run_id','status','checkpoint_epoch','attempts','successful_updates','AMP_skips']}),flush=True)
assert len(inherited)==18 and len(fresh)==4 and all(x['status']=='PASS' for x in results)
receipt=dict(utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,status='PASS',checks=results,queue_snapshot=queue,scope='Fresh4current full CPU proofs including3automatic successors;18terminalproofs inherited using independent terminal audit and unchanged stat/completion bindings. No live restore; explicit mixed cutoffs.',CUDA_initialized=m.torch.cuda.is_initialized(),inspection_cutoffs_utc=cutoffs,inherited_accepted_terminal_run_ids=inherited,fresh_full_CPU_run_ids=fresh,prior_receipt=str(basepath),prior_receipt_sha256=m.digest(basepath),terminal_audit=str(auditpath),terminal_audit_sha256=m.digest(auditpath),inherited_terminal_stat_bindings=stat_bindings)
m.trainer.save_json(R/'reports/checkpoint_health_20261024T0630_followup.json',receipt);assert not receipt['CUDA_initialized'];print('ALL22_FULL_CPU_PROOFS_ACCEPTED_FRESH4_INHERITED18',flush=True)
