"""Independent saved source-DEV checks at pinned epochs; CPU only."""
import pathlib,json,hashlib,datetime,importlib.util
import numpy as np,pandas as pd
R=pathlib.Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('independent_completion_check',R/'qualification/completion_audit_1415.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cp=json.loads((R/'reports/checkpoint_health_1415.json').read_text());dev=pd.read_parquet(R/'data/D/dev.parquet');expected=dev.groupby(['video_uid','method_id','label'],sort=True,as_index=False).size()[['video_uid','method_id','label']]
rows=[]
for c in cp['checks']:
 assert c['status']=='PASS';run=c['run_id'];out=R/'runs'/run;h=json.loads((out/'history.json').read_text())[:c['checkpoint_epoch']];assert len(h)==c['checkpoint_epoch']
 entries=[];maxerr=0.;comparisons=0
 for row in h:
  p=out/f'epoch_{row["epoch"]:02d}_source_dev.parquet';digest=sha(p);f=pd.read_parquet(p);assert sha(p)==digest
  assert list(f)==['video_uid','method_id','label','score'] and f[['video_uid','method_id','label']].equals(expected)
  assert np.isfinite(f.score).all() and not f.video_uid.duplicated().any()
  per={};real=f[f.label==0]
  for method,x in f[f.label==1].groupby('method_id',sort=True):
   z=pd.concat([real,x],ignore_index=True);per[method]=m.auc(z.label,z.score)
  assert set(per)==set(row['dev']['method_auc'])
  errors=[abs(m.auc(f.label,f.score)-row['dev']['pooled_auc']),abs(float(np.mean(list(per.values())))-row['dev']['macro_auc'])]+[abs(v-row['dev']['method_auc'][k]) for k,v in per.items()]
  assert max(errors)<1e-12;maxerr=max(maxerr,max(errors));comparisons+=len(errors)
  entries.append({'epoch':row['epoch'],'rows':len(f),'methods':len(per),'sha256':digest,'max_abs_error':max(errors)})
 rows.append({'run_id':run,'pinned_epoch':c['checkpoint_epoch'],'checked_DEV_files':len(entries),'metric_comparisons':comparisons,'max_abs_error':maxerr,'files':entries})
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'PASS_INDEPENDENT_SAVED_SOURCE_DEV_RECOMPUTATION','scope':'Actual SOURCE_DEV only through each pinned epoch; no target performance, recipe change, training or target cache evaluation. Integer concordance routine independent of trainer/sklearn used by completion audit.',
'rows':rows,'total_files':sum(x['checked_DEV_files'] for x in rows),'metric_comparisons':sum(x['metric_comparisons'] for x in rows),'max_abs_error':max(x['max_abs_error'] for x in rows),'checkpoint_receipt_sha256':sha(R/'reports/checkpoint_health_1415.json')}
(R/'reports/D_saved_source_DEV_independent_1415.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k!='rows'}),flush=True)
