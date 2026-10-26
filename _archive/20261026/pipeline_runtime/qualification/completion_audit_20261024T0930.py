"""Append-only CPU terminal audit using the accepted independent metric implementation."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,os
import pandas as pd
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 spec=importlib.util.spec_from_file_location('terminal_audit_accepted',R/'qualification/completion_audit_1415.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 basepath=R/'reports/D_completed_artifact_audit_20261024T0850.json';base=json.loads(basepath.read_text());assert base['status']=='PASS'
 rows=list(base['completed']);known={x['run_id'] for x in rows};config=json.loads((R/'configs/execution.json').read_text());dev=pd.read_parquet(R/'data/D/dev.parquet');keys=dev.groupby(['video_uid','method_id','label'],sort=True,as_index=False).size()[['video_uid','method_id','label']];trainlen=len(pd.read_parquet(R/'data/D/train.parquet'));new=[]
 for run,task in config['runs'].items():
  if run in known or not (R/'runs'/run/'complete.json').is_file():continue
  result=m.inspect(run,task,keys,trainlen);assert result['status']=='PASS_COMPLETED_FROZEN_LAST_MODEL';rows.append(result);new.append(run);print('NEW_TERMINAL_FULL_AUDIT_PASS',run,flush=True)
 for row in base['completed']:
  for name,entry in row['core_file_inventory'].items():assert m.sha(R/'runs'/row['run_id']/name)==entry['sha256']
 receipt={**base,'utc':datetime.now(timezone.utc).isoformat(),'completed':rows,'newly_appended_completed_run_ids':new,'prior_completion_receipt_sha256':m.sha(basepath),'CUDA_initialized':m.torch.cuda.is_initialized(),'independent_completed_DEV_tables':sum(len(x['epochs_inventory']) for x in rows),'independent_completed_DEV_metric_comparisons':sum(sum(e['method_count']+2 for e in x['epochs_inventory']) for x in rows)}
 for row in rows:row['registered_phase']=config['runs'][row['run_id']]['phase']
 receipt['completed_by_registered_phase']={phase:sum(row['registered_phase']==phase for row in rows) for phase in ['development','formal']}
 receipt['independent_DEV_tables_by_registered_phase']={phase:sum(len(row['epochs_inventory']) for row in rows if row['registered_phase']==phase) for phase in ['development','formal']}
 receipt['scope']='CompletedDdevelopmentandformalrunsannotatedbyregisteredphase;no targetevaluation,traininglaunchorclaimofwholestudycompletion'
 assert not receipt['CUDA_initialized'];m.save(R/'reports/D_completed_artifact_audit_20261024T0930.json',receipt);print('COMPLETED_AUDIT_COUNT',len(rows),'NEW',len(new),'DEV_TABLES',receipt['independent_completed_DEV_tables'],flush=True)
if __name__=='__main__':main()
