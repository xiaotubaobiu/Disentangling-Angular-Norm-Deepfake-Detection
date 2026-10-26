from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,os,hashlib
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
spec=importlib.util.spec_from_file_location('accepted_health',R/'qualification/checkpoint_health_20261024T1030.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.torch.cuda.is_initialized()
basepath=R/'reports/checkpoint_health_20261024T1000.json';base=json.loads(basepath.read_text());assert base['status'] in ('PASS','PASS_WITH_EXPLICIT_FIRST_EPOCH_PENDING')
config=json.loads((R/'configs/execution.json').read_text());queue=json.loads((R/'queue_status.json').read_text())
hashes={k:m.digest(p) for k,p in {'train':R/'data/D/train.parquet','dev':R/'data/D/dev.parquet','environment':R/'configs/requirements.execution.lock.txt','gate':R/'qualification/D_gate.json',**{k:Path(v['weight_path']) for k,v in config['backbones'].items()}}.items()}
results=[];cutoffs={};inherited=[];fresh=[]
for row in base['checks']:
 if row['lifecycle_at_inspection']!='COMPLETE':continue
 run=row['run_id'];assert (R/'runs'/run/'complete.json').is_file();assert m.digest(R/'runs'/run/'state.pt')==row['checkpoint_sha256']
 row={**row,'inspection_scope_this_round':'INHERITED_ACCEPTED_TENSOR_PROOF_CURRENT_STATE_SHA_REVALIDATED_NO_FULL_RELOAD'};results.append(row);inherited.append(run);cutoffs[run]=base['inspection_cutoffs_utc'][run]
items=[dict(**x,process_required=True,lifecycle_at_inspection='RUNNING') for x in queue['active']]
for run in config['runs']:
 if (R/'runs'/run/'complete.json').is_file() and run not in inherited and run not in [x['run_id'] for x in items]:items.append(dict(run_id=run,pid=None,process_required=False,lifecycle_at_inspection='COMPLETE'))
for item in items:
 out=R/'runs'/item['run_id']
 if (out/'complete.json').exists():item={**item,'process_required':False,'lifecycle_at_inspection':'COMPLETE'}
 try: row=m.inspect_run(item,config,hashes) if (out/'state.pt').exists() else m.first_epoch(item,config,hashes)
 except Exception as e:row={**item,'status':'FAILED_INSPECTION','exception_type':type(e).__name__,'error':str(e)}
 row['inspection_scope_this_round']='FRESH_FULL_CPU_STATE_PROOF' if row['status']=='PASS' else 'FIRST_EPOCH_PENDING_NO_FULL_STATE_PROOF';results.append(row);fresh.append(item['run_id']);cutoffs[item['run_id']]=datetime.now(timezone.utc).isoformat()
 print(json.dumps({k:row[k] for k in ['run_id','status','checkpoint_epoch','attempts','successful_updates','AMP_skips','error'] if k in row}),flush=True)
ok=all(x['status'] in ['PASS','PENDING_FIRST_EPOCH_RECOVERY_POINT'] for x in results)
receipt=dict(utc=datetime.now(timezone.utc).isoformat(),CPU_only=True,status='PASS' if ok and all(x['status']=='PASS' for x in results) else 'PASS_WITH_EXPLICIT_FIRST_EPOCH_PENDING' if ok else 'INSPECTION_REQUIRES_REVIEW',checks=results,queue_snapshot=queue,scope='Fresh active CPU proof; unchanged terminal proofs inherited with current state hash binding. Explicit mixed cutoffs, no live restore or unchanged full terminal audit.',CUDA_initialized=m.torch.cuda.is_initialized(),inspection_cutoffs_utc=cutoffs,inherited_accepted_terminal_run_ids=inherited,fresh_inspected_run_ids=fresh,fresh_full_CPU_run_ids=[x['run_id'] for x in results if x['run_id'] in fresh and x['status']=='PASS' and x.get('checkpoint_epoch',0)>0],prior_receipt=str(basepath),prior_receipt_sha256=m.digest(basepath))
m.trainer.save_json(R/'reports/checkpoint_health_20261024T1030.json',receipt);assert ok and not receipt['CUDA_initialized']
print('FRESH',len(fresh),'INHERITED',len(inherited),'STATUS',receipt['status'],flush=True)
