from pathlib import Path
import importlib.util,json,datetime,os
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
spec=importlib.util.spec_from_file_location('accepted_checkpoint_inspector',R/'qualification/checkpoint_health_2100.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not m.torch.cuda.is_initialized()
config=json.loads((R/'configs/execution.json').read_text());q=json.loads((R/'queue_status.json').read_text());assert q['stage']=='P3_D_FORMAL_TRAINING' and len(q['active'])==4
hashes={key:m.digest(path) for key,path in {'train':R/'data/D/train.parquet','dev':R/'data/D/dev.parquet','environment':R/'configs/requirements.execution.lock.txt','gate':R/'qualification/D_gate.json',**{key:Path(v['weight_path']) for key,v in config['backbones'].items()}}.items()}
checks=[]
for item in q['active']:
 assert config['runs'][item['run_id']]['phase']=='formal' and (R/'runs'/item['run_id']/'state.pt').exists()
 report=m.inspect_run(dict(**item,process_required=True,lifecycle_at_inspection='RUNNING'),config,hashes);assert report['status']=='PASS' and report['checkpoint_epoch']>=1
 checks.append(report);print(json.dumps({k:report[k] for k in ['run_id','status','checkpoint_epoch','attempts','successful_updates','AMP_skips','recent_effective_input_rate_per_second']}),flush=True)
receipt=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_ALL4_FORMAL_FIRST_COMMITTED_EPOCH_FULL_CPU_RECOVERIES',CPU_only=True,checks=checks,queue_snapshot=q,CUDA_initialized=m.torch.cuda.is_initialized(),scope='Newfirstformalcheckpoints only; fullpinnedtensors/Adam/RNG/scaler/attemptLR/identity/committedlogs/samplingarrays. No live restore or training modification.')
assert not receipt['CUDA_initialized'];m.trainer.save_json(R/'reports/D_formal_first_checkpoint_acceptance_2100.json',receipt)
