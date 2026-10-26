from pathlib import Path
from datetime import datetime,timezone
import os,json,importlib.util
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');spec=importlib.util.spec_from_file_location('readonly_health',R/'qualification/checkpoint_health_20261024T0850.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
assert os.environ['CUDA_VISIBLE_DEVICES']=='' and not m.torch.cuda.is_initialized()
base=json.loads((R/'reports/checkpoint_health_20261024T0850.json').read_text());targets={x['run_id'] for x in base['checks'] if x['status']=='PENDING_FIRST_EPOCH_RECOVERY_POINT'};assert targets=={'D_formal_efficientnet_b0_seed123','D_formal_efficientnet_b0_seed2026'}
config=json.loads((R/'configs/execution.json').read_text());queue=json.loads((R/'queue_status.json').read_text());hashes={k:m.digest(p) for k,p in {'train':R/'data/D/train.parquet','dev':R/'data/D/dev.parquet','environment':R/'configs/requirements.execution.lock.txt','gate':R/'qualification/D_gate.json',**{k:Path(v['weight_path']) for k,v in config['backbones'].items()}}.items()};checks=[];cutoffs={}
for item in queue['active']:
 if item['run_id'] not in targets:continue
 assert (R/'runs'/item['run_id']/'state.pt').is_file();row=m.inspect_run(dict(**item,process_required=True,lifecycle_at_inspection='RUNNING'),config,hashes);assert row['status']=='PASS' and row['checkpoint_epoch']>=1;checks.append(row);cutoffs[item['run_id']]=datetime.now(timezone.utc).isoformat();print(json.dumps({k:row[k] for k in ['run_id','status','checkpoint_epoch','attempts','successful_updates','AMP_skips']}),flush=True)
assert len(checks)==2 and not m.torch.cuda.is_initialized();record=dict(utc=datetime.now(timezone.utc).isoformat(),status='PASS_BOTH_AUTOMATIC_SUCCESSORS_FIRST_COMPLETE_CPU_RECOVERY_POINTS',checks=checks,inspection_cutoffs_utc=cutoffs,CPU_only=True,CUDA_initialized=False,scope='Targeted followup of two explicit first-epoch pending entries; no training/restart/live restore. Initial0850receipt remains unchanged; its fresh_full_CPU_run_ids includes inspected-but-pending entries, so count only statusPASS with checkpoint_epoch>0 as full tensor proof.')
m.trainer.save_json(R/'reports/D_successor_recovery_acceptance_20261024T0850.json',record)
