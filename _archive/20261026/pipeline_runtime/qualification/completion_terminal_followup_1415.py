"""Read-only append-only terminal audit for models completing during the health window."""
from pathlib import Path
from datetime import datetime,timezone
import importlib.util,json,os,hashlib
import pandas as pd
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 m=load('terminal_audit_base',R/'qualification/completion_audit_1415.py');base=json.loads((R/'reports/D_completed_artifact_audit_1415.json').read_text());assert base['status']=='PASS'
 rows=list(base['completed']);known={x['run_id'] for x in rows};cfg=json.loads((R/'configs/execution.json').read_text())
 dev=pd.read_parquet(R/'data/D/dev.parquet');keys=dev.groupby(['video_uid','method_id','label'],sort=True,as_index=False).size()[['video_uid','method_id','label']];trainlen=len(pd.read_parquet(R/'data/D/train.parquet'))
 new=[]
 for run,task in cfg['runs'].items():
  if run not in known and (R/'runs'/run/'complete.json').is_file():
   result=m.inspect(run,task,keys,trainlen);rows.append(result);new.append(run);print('NEW_COMPLETED_AUDIT_PASS',run,flush=True)
 # Freshly recheck core inventory hashes for prior already-audited terminal models.
 for x in base['completed']:
  for name,entry in x['core_file_inventory'].items():assert sha(R/'runs'/x['run_id']/name)==entry['sha256']
 receipt={**base,'utc':datetime.now(timezone.utc).isoformat(),'completed':rows,'newly_appended_completed_run_ids':new,'prior_completion_receipt_sha256':sha(R/'reports/D_completed_artifact_audit_1415.json')}
 (R/'reports/D_completed_artifact_audit_1415_final.json').write_text(json.dumps(receipt,indent=2)+'\n')
 cp=load('checkpoint_audit_base',R/'qualification/checkpoint_health_1415.py');hashes={key:cp.digest(path) for key,path in {'train':R/'data/D/train.parquet','dev':R/'data/D/dev.parquet','environment':R/'configs/requirements.execution.lock.txt','gate':R/'qualification/D_gate.json',**{k:Path(v['weight_path']) for k,v in cfg['backbones'].items()}}.items()}
 previous=json.loads((R/'reports/checkpoint_health_1415.json').read_text());was={x['run_id']:x for x in previous['checks']};extension=[]
 for row in rows:
  run=row['run_id']
  if was.get(run,{}).get('checkpoint_epoch',0)<cfg['runs'][run]['epochs']:
   result=cp.inspect_run({'run_id':run,'pid':None,'process_required':False,'lifecycle_at_inspection':'COMPLETE'},cfg,hashes);assert result['status']=='PASS';extension.append(result);print('TERMINAL_CHECKPOINT_PASS',run,flush=True)
 ext={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS','checks':extension,'scope':'Only new completed terminal checkpoints beyond initial1415pinned snapshots; CPU-only no live restore','CUDA_initialized':m.torch.cuda.is_initialized(),'CPU_only':True}
 assert not ext['CUDA_initialized'];(R/'reports/checkpoint_terminal_extension_1415.json').write_text(json.dumps(ext,indent=2)+'\n');print('FINAL_COMPLETED_COUNT',len(rows),flush=True)
if __name__=='__main__':main()
