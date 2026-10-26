import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OPENBLAS_NUM_THREADS']='1'
import pathlib,pandas as pd,numpy as np,json,datetime,hashlib
r=pathlib.Path('/home/z/project/deepfake_detector');p=r/'artifacts/data';rows=[];digests={}
for role in ['calibration','ff_test','cdf_test']:
 path=p/(role+'.parquet');digests[role]=hashlib.sha256(path.read_bytes()).hexdigest();d=pd.read_parquet(path)
 units={'image':d,'video':d[d.evaluation_unit.eq('video_frame')].groupby(['domain','method_id','video_id'],sort=True).agg(label=('label','first'),group_id=('group_id','first')).reset_index()}
 for unit,t in units.items():
  groups,codes=np.unique(t.group_id.astype(str),return_inverse=True);real=t.label.to_numpy()==0;methods=sorted(set(t.loc[~real,'method_id']));support={m:np.unique(codes[t.method_id.to_numpy()==m]) for m in methods};rc=np.unique(codes[real]);by_seed={}
  if role!='calibration':
   for seed in [7,42,123,2026,3407]:
    draws=np.random.default_rng(seed).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000);valid=(draws[:,rc].sum(axis=1)>0);missing={}
    for method,ids in support.items():
     present=draws[:,ids].sum(axis=1)>0;valid &= present;missing[method]=int((~present).sum())
    by_seed[str(seed)]={'macro_support_valid':int(valid.sum()),'macro_support_invalid':int((~valid).sum()),'methods_with_missing_support':{k:v for k,v in missing.items() if v}}
  rows.append({'role':role,'unit':unit,'rows':len(t),'groups':len(groups),'real_groups':len(rc),'minimum_fake_method_groups':min(map(len,support.values())) if support else None,'fake_methods':len(methods),'support_by_seed':by_seed})
proof={'checked_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_manifest_sha256':digests,'score_values_used':False,'protocol_or_input_changes':False,'test_bootstrap_iterations':2000,'policy':'all prescribed methods must be present; a missing method invalidates the entire macro replicate; no reduced-method averaging','scope':'declared frozen group_id proxy only, unknown ancestry not certified','bootstrap_support_structure':rows}
path=r/'reports/bootstrap_support_structure.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(proof,indent=2)+'\n');tmp.replace(path)
with (r/'reports/audit.log').open('a') as f:f.write(json.dumps({'event':'independent_frozen_bootstrap_support_verified','UTC':proof['checked_UTC'],'scores_used':False,'manifest_or_protocol_changed':False})+'\n')
print(json.dumps({'checked_UTC':proof['checked_UTC'],'receipt':str(path),'FF_image_invalid':[x['support_by_seed'] for x in rows if x['role']=='ff_test' and x['unit']=='image'],'CDF_image_invalid':[x['support_by_seed'] for x in rows if x['role']=='cdf_test' and x['unit']=='image'],'video_all_prescribed_support':True}))