from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,os
from reliability_v2 import data_h
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
p=R/'data/H/freeze.json';before=sha(p);receipt=data_h.validate_h_freeze(R,deep=False);after=sha(p)
launch=json.loads((R/'reports/H_production_data_freeze_launch.json').read_text());ex=json.loads((R/'reports/H_production_data_freeze_exit.json').read_text())
assert before==after==launch['freeze_sha256']==ex['freeze_sha256']=='11ac3398b8bd2b347da24efbe45fb74e890681d2d063933b01a65d5fdba0675d'
assert launch['status']=='FROZEN' and launch['blockers']==[] and ex['returncode']==0 and ex['launch_receipt_sha256']==sha(R/'reports/H_production_data_freeze_launch.json')
assert not Path('/proc/246169').exists() and not Path('/proc/246168').exists()
assert receipt['candidate_videos']==8431 and receipt['source_unused_candidate_videos']==560 and receipt['training_draw_count']==24448 and receipt['logical_train_mother_rows']==24424 and not receipt['target_performance_used']
assert not (R/'qualification/H_gate.json').exists()
out={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS_FROZEN_RECEIPT_AND_OSRC0_SHALLOW_CURRENT_BINDINGS','freeze_path':str(p),'freeze_sha256':before,'freeze_utc':receipt['utc'],'freeze_completed_utc':launch['ended_utc'],'elapsed_seconds':(datetime.fromisoformat(launch['ended_utc'])-datetime.fromisoformat(launch['utc'])).total_seconds(),'OS_returncode':ex['returncode'],'retired_worker_pid':246169,'retired_supervisor_pid':246168,'validation_call':'data_h.validate_h_freeze(root,deep=False)','validation_scope':'SHALLOW_METADATA_BINDING: current code/review/audit/all13tables/8431metadata; no sourcevideo/PNG bytes rescanned, no detection or new cohort. Original freeze event performed mandatory full validation.','receipt':receipt,'launch_receipt_sha256':sha(R/'reports/H_production_data_freeze_launch.json'),'exit_receipt_sha256':sha(R/'reports/H_production_data_freeze_exit.json'),'H_data_frozen':True,'H_GPU_qualification_or_training_started':False,'remaining_H_dependency':'Original D_PRIORITY_UNTIL_D_GPU_CACHE_STAGE_COMPLETE plus sharedGPUlocks and actual source GPU qualification; data freeze alone is not GPU acceptance.'}
with (R/'reports/H_freeze_receipt_validation_1930.json').open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write(chr(10))
print(json.dumps({k:v for k,v in out.items() if k!='receipt'},ensure_ascii=False))
