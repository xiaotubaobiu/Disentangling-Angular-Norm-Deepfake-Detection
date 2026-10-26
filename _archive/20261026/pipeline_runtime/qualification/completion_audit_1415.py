"""Read-only completion and independent source-DEV integrity audit. No CUDA use."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import gc,hashlib,json,math,os
import numpy as np
import pandas as pd
import torch
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,j):
 q=p.with_name(p.name+'.audit.tmp');q.write_text(json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n');q.replace(p)
def auc(y,s):
 y=np.asarray(y);s=np.asarray(s,dtype=np.float64);assert np.isfinite(s).all()
 neg=np.sort(s[y==0]);pos=s[y==1];assert len(neg)>0 and len(pos)>0
 twice_credit=int(np.sum(np.searchsorted(neg,pos,side='left')+np.searchsorted(neg,pos,side='right'),dtype=np.int64))
 return twice_credit/(2*len(neg)*len(pos))
def same_model(a,b):
 assert set(a)==set(b)
 for k in a:
  assert a[k].device.type==b[k].device.type=='cpu'
  assert torch.equal(a[k],b[k]),'weight mismatch:'+k
  if a[k].is_floating_point():assert torch.isfinite(a[k]).all()
def inspect(run,task,devkeys,trainlen):
 out=R/'runs'/run;c=json.loads((out/'complete.json').read_text());ident=json.loads((out/'identity.json').read_text());hist=json.loads((out/'history.json').read_text())
 assert c['status']=='TRAINING_COMPLETE' and c['identity']==ident
 assert c['epochs']==task['epochs']==30 and c['attempts']==30*799==23970
 assert c['success']+c['skip']==c['attempts'] and c['selection']=='frozen_last' and c['target_used'] is False
 assert sha(out/'final.pt')==c['final_sha256'] and sha(out/'source_best.pt')==c['source_best_sha256']
 st=torch.load(out/'state.pt',map_location='cpu',weights_only=False)
 assert st['identity']==ident and st['epoch']==30 and st['attempts']==23970 and st['history']==hist
 final=torch.load(out/'final.pt',map_location='cpu',weights_only=False)
 assert final['identity']==ident and final['epoch']==30;same_model(final['model'],st['model']);del final;gc.collect()
 best=torch.load(out/'source_best.pt',map_location='cpu',weights_only=False)
 assert best['identity']==ident and best['epoch']==c['source_best_epoch']==st['best_epoch'];same_model(best['model'],st['best_model']);del best;gc.collect()
 assert [x['epoch'] for x in hist]==list(range(1,31))
 assert sum(x['attempts'] for x in hist)==c['attempts'] and sum(x['success'] for x in hist)==c['success'] and sum(x['skip'] for x in hist)==c['skip']
 expected_names={'identity.json','resolved_config.json','state.pt','history.json','final.pt','source_best.pt','complete.json','run.lock'}
 files={};errmax=0.;epochs=[];events=[]
 for row in hist:
  e=row['epoch'];assert row['attempts']==799 and row['draw_count']==102272 and row['success']+row['skip']==799
  names={k:out/f'epoch_{e:02d}_{k}' for k in ['samples.npy','methods.npy','augmentations.npy','updates.jsonl','source_dev.parquet']}
  for p in names.values():assert p.is_file();expected_names.add(p.name);files[p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
  assert files[names['samples.npy'].name]['sha256']==row['sample_sha256']
  assert files[names['augmentations.npy'].name]['sha256']==row['augmentation_sha256']
  draws=np.load(names['samples.npy'],allow_pickle=False);methods=np.load(names['methods.npy'],allow_pickle=False);aug=np.load(names['augmentations.npy'],allow_pickle=False)
  assert draws.shape==(102272,) and draws.min()>=0 and draws.max()<trainlen
  assert methods.shape==(102272,) and dict(Counter(methods.tolist()))==row['method_counts']
  assert aug.shape==(102272,3) and np.isfinite(aug).all() and np.isin(aug[:,0],[0,1]).all()
  assert ((aug[:,1:]>=.9)&(aug[:,1:]<=1.1)).all()
  f=pd.read_parquet(names['source_dev.parquet']);assert list(f)==['video_uid','method_id','label','score']
  assert f[['video_uid','method_id','label']].equals(devkeys) and np.isfinite(f.score.to_numpy()).all()
  assert not f.video_uid.duplicated().any()
  pooled=auc(f.label,f.score);real=f[f.label==0];per={}
  for method,z in f[f.label==1].groupby('method_id',sort=True):
   part=pd.concat([real,z],ignore_index=True);per[method]=auc(part.label,part.score)
  assert set(per)==set(row['dev']['method_auc'])
  errors=[abs(pooled-row['dev']['pooled_auc']),abs(float(np.mean(list(per.values())))-row['dev']['macro_auc'])]+[abs(v-row['dev']['method_auc'][k]) for k,v in per.items()]
  assert max(errors)<1e-12;errmax=max(errmax,max(errors))
  log=[json.loads(l) for l in names['updates.jsonl'].read_text().splitlines()];assert len(log)==799
  assert [v['attempt'] for v in log]==list(range((e-1)*799,e*799));events+=log
  epochs.append({'epoch':e,'files':5,'DEV_video_rows':len(f),'method_count':len(per),'recomputation_max_abs_error':max(errors)})
 actual_names={p.name for p in out.iterdir() if p.is_file()};assert expected_names.issubset(actual_names)
 assert not any(n.endswith('.tmp') for n in actual_names)
 extra=sorted(actual_names-expected_names)
 skip=[x for x in events if x['skip']];evidence=[]
 for x in skip:
  n=x['attempt'];assert math.isfinite(x['loss']) and not x['gradient_finite'] and x['gradient_norm'] is None and x['scale_after']<x['scale_before']
  following=events[n+1] if n+1<len(events) else None
  if following:assert following['success'] and following['gradient_finite'] and math.isfinite(following['loss'])
  evidence.append({'attempt':n,'epoch':x['epoch'],'scale_before':x['scale_before'],'scale_after':x['scale_after'],'next_successful_attempt':following['attempt'] if following else None})
 assert len(skip)==c['skip']
 del st;gc.collect()
 inventory={n:dict(bytes=(out/n).stat().st_size,sha256=sha(out/n)) for n in ['identity.json','resolved_config.json','state.pt','history.json','final.pt','source_best.pt','complete.json']}
 return {'run_id':run,'status':'PASS_COMPLETED_FROZEN_LAST_MODEL','epochs':30,'attempts':c['attempts'],'successful_updates':c['success'],'AMP_skips':c['skip'],'selection':'frozen_last','target_used':False,'final_sha256':c['final_sha256'],'source_best_sha256':c['source_best_sha256'],'final_equal_terminal_state_model':True,'source_best_equal_saved_state_model':True,'epochs_inventory':epochs,'epoch_file_inventory':files,'core_file_inventory':inventory,'unexpected_extra_files':extra,'independent_DEV_recomputation':'integer concordance via sorted real scores and searchsorted, without trainer/scikit-learn metric routines','independent_DEV_recomputation_max_abs_error':errmax,'skip_evidence':evidence}
def main():
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
 config=json.loads((R/'configs/execution.json').read_text());dev=pd.read_parquet(R/'data/D/dev.parquet');devkeys=dev.groupby(['video_uid','method_id','label'],sort=True,as_index=False).size()[['video_uid','method_id','label']]
 trainlen=len(pd.read_parquet(R/'data/D/train.parquet'));rows=[]
 for run,task in config['runs'].items():
  if (R/'runs'/run/'complete.json').exists():
   try:rep=inspect(run,task,devkeys,trainlen)
   except Exception as e:rep={'run_id':run,'status':'FAILED_COMPLETION_AUDIT','exception_type':type(e).__name__,'error':str(e)}
   rows.append(rep);print(json.dumps({k:v for k,v in rep.items() if k not in ['epochs_inventory','epoch_file_inventory','core_file_inventory']},ensure_ascii=False),flush=True)
 receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS' if rows and all(x['status']=='PASS_COMPLETED_FROZEN_LAST_MODEL' for x in rows) else 'REQUIRES_REVIEW','completed':rows,'CPU_only':True,'CUDA_initialized':torch.cuda.is_initialized(),'scope':'Completed development runs only; no target evaluation, new training or claim of complete study'}
 save(R/'reports/D_completed_artifact_audit_1415.json',receipt);assert receipt['status']=='PASS'
if __name__=='__main__':main()
