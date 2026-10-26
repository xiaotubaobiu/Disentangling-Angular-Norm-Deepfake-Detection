from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,os
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
spec=importlib.util.spec_from_file_location('accepted_health',R/'qualification/checkpoint_health_20261024T1315.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.torch.cuda.is_initialized()
basepath=R/'reports/checkpoint_health_20261024T1300.json';base=json.loads(basepath.read_text());assert base['status']=='PASS';config=json.loads((R/'configs/execution.json').read_text());queue=json.loads((R/'queue_status.json').read_text());run='D_formal_efficientnet_b0_seed3407';out=R/'runs'/run
hashes={k:m.digest(p) for k,p in {'train':R/'data/D/train.parquet','dev':R/'data/D/dev.parquet','environment':R/'configs/requirements.execution.lock.txt','gate':R/'qualification/D_gate.json',**{k:Path(v['weight_path']) for k,v in config['backbones'].items()}}.items()}
def item_now():
 completed=(out/'complete.json').is_file();return dict(run_id=run,pid=None if completed else 1489240,process_required=not completed,lifecycle_at_inspection='COMPLETE' if completed else 'RUNNING')
transition=None
try: fresh=m.inspect_run(item_now(),config,hashes)
except (FileNotFoundError,AssertionError) as e:
 if not (out/'complete.json').is_file():raise
 transition=dict(exception_type=type(e).__name__,message=str(e),scope='fresh terminal evidence appeared during inspection; readonly terminal retry, no worker restart')
 fresh=m.inspect_run(item_now(),config,hashes)
assert fresh['status']=='PASS';cutoff=datetime.now(timezone.utc).isoformat();fresh['inspection_scope_this_round']='FRESH_FULL_CPU_STATE_PROOF_LAST_D_FORMAL_ONLY';rows=[];cutoffs={};inherited=[]
for old in base['checks']:
 if old['run_id']==run:continue
 assert old['lifecycle_at_inspection']=='COMPLETE' and m.digest(R/'runs'/old['run_id']/'state.pt')==old['checkpoint_sha256'];rows.append({**old,'inspection_scope_this_round':'INHERITED_ACCEPTED_TERMINAL_CURRENT_STATE_SHA_REVALIDATED'});cutoffs[old['run_id']]=base['inspection_cutoffs_utc'][old['run_id']];inherited.append(old['run_id'])
rows.append(fresh);cutoffs[run]=cutoff;assert len(rows)==len({x['run_id'] for x in rows})==25 and not m.torch.cuda.is_initialized()
receipt=dict(utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,status='PASS',checks=rows,queue_snapshot=queue,scope='Phase-agnostic explicit last D training endpoint; cache worker runIDs are never re-inspected as active trainers. No GPUwork or live restore.',CUDA_initialized=False,inspection_cutoffs_utc=cutoffs,inherited_accepted_terminal_run_ids=inherited,fresh_inspected_run_ids=[run],fresh_full_CPU_run_ids=[run],prior_receipt=str(basepath),prior_receipt_sha256=m.digest(basepath),transition_terminal_retry=transition)
m.trainer.save_json(R/'reports/checkpoint_health_20261024T1315.json',receipt);print(json.dumps({k:fresh.get(k) for k in ['run_id','checkpoint_epoch','attempts','successful_updates','AMP_skips','lifecycle_at_inspection']}),flush=True)
