from pathlib import Path
import datetime,hashlib,json,resource,time,os
import numpy as np
import pandas as pd
from reliability_v2 import stats_h,analysis_d,multiplicity_codec as codec,sensitivity_io as transport
E=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023/engineering/source_best_M')
O=E/'storage_profiles/H_synthetic_frozen_kernel_v1';O.mkdir(exist_ok=False)
start=time.monotonic();rows=[]
for domain,methods in [('source',('Deepfakes','Face2Face','FaceSwap','NeuralTextures')),('target',('DeepFakeDetection',))]:
 q=[]
 for group in range(20):
  for m in ['real']+list(methods):q.append(dict(video_uid=f'{domain}_g{group}_{m}',label=int(m!='real'),method_id=m,group_id=f'{domain}_g{group}',score=(group%7-3)*.17+int(m!='real')*.31))
 for aggregation in ['mean_logit','mean_probability']:
  first=pd.DataFrame(q);second=first.copy();second['score']=second.score*.71-.03
  cal=pd.DataFrame(dict(video_uid=[f'cal_{i}' for i in range(40)],group_id=[f'calg_{i}' for i in range(40)],label=0,method_id='real',score=np.linspace(-.5,.5,40)))
  if aggregation=='mean_probability':
   for a in [first,second,cal]:a['score']=1/(1+np.exp(-a.score))
  result=stats_h.analyze_cohort({'H00':first,'H10':second},{'B':cal,'AB':cal.copy()},fixed_methods=methods,aggregation=aggregation,repeats=2000,seed=20261023)
  value=analysis_d._jsonable(result);raw=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False).encode();bundle=codec.encode_counts(value,max_cells=1000000)
  tag=domain+'_'+aggregation;npz=O/(tag+'.npz');np.savez_compressed(npz,**bundle['arrays']);meta={k:v for k,v in bundle.items() if k!='arrays'};mp=O/(tag+'.metadata.json.gz');receipt=transport.write_json(mp,meta)
  restored=transport.read_json(mp,compressed_sha256=receipt['compressed_sha256'],uncompressed_sha256=receipt['uncompressed_sha256'],max_uncompressed_bytes=64*1024**2)
  with np.load(npz,allow_pickle=False) as archive:restored['arrays']={k:archive[k] for k in archive.files}
  decoded=codec.decode_counts(restored,max_cells=1000000);decoded_raw=json.dumps(decoded,ensure_ascii=False,indent=2,allow_nan=False).encode();assert decoded_raw==raw
  assert len(decoded['bootstrap']['draws'])==2000
  assert decoded['bootstrap']['repeats']==2000 and all(sum(row['group_draw_counts'].values())==20 for row in decoded['bootstrap']['draws'])
  rows.append(dict(domain=domain,aggregation=aggregation,groups=20,draws=2000,source_bytes=len(raw),packed_bytes=npz.stat().st_size+mp.stat().st_size,original_raw_sha256=hashlib.sha256(raw).hexdigest(),decoded_raw_sha256=hashlib.sha256(decoded_raw).hexdigest(),point_checks=len(result['independent_checks']),exact_all_JSON_values_and_order=True))
  print(tag,'PASS2000',flush=True)
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_SYNTHETIC_H_ACTUAL_FROZEN_KERNEL_FORMAT_ONLY',cases=rows,seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scope='Synthetic known scores/groups passed through original H kernel at unchanged2000draws/nativeB two-cell comparisons/both aggregations/source4methods and target1method. No real H data/cache/model/target inference; not scientific results or a realH resource estimate.',real_H_profile_complete=False,M_production_admitted=False,capacity_savings_credited=False)
(E/'H_synthetic_frozen_kernel_storage_probe.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print('COMPLETE',report['seconds'],flush=True)
