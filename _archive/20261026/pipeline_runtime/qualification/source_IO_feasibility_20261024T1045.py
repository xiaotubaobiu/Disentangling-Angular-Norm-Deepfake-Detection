from pathlib import Path
from datetime import datetime,timezone
import os,io,json,time,hashlib,statistics
import pandas as pd
from PIL import Image
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');manifest=R/'data/D/train.parquet';manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest();q=pd.read_parquet(manifest);assert 'image_path' in q;paths=sorted(set(q.image_path));selected=[paths[i*len(paths)//32] for i in range(32)];rows=[];byte_total=0
for repeat in range(2):
 for path in selected:
  if byte_total>=32*1024**2:break
  t0=time.perf_counter();fd=os.open(path,os.O_RDONLY);t1=time.perf_counter();size=os.fstat(fd).st_size;t2=time.perf_counter()
  if size>1024**2:
   os.close(fd);rows.append(dict(pass_index=repeat,path_sha256=hashlib.sha256(path.encode()).hexdigest(),status='SKIPPED_LARGER_THAN1MiB',bytes=size));continue
  with os.fdopen(fd,'rb') as stream:data=stream.read()
  t3=time.perf_counter();assert len(data)==size;byte_total+=size
  with Image.open(io.BytesIO(data)) as image:image.load();dimensions=image.size
  t4=time.perf_counter();rows.append(dict(pass_index=repeat,path_sha256=hashlib.sha256(path.encode()).hexdigest(),file_sha256=hashlib.sha256(data).hexdigest(),status='READ_DECODED',bytes=size,dimensions=dimensions,open_ms=(t1-t0)*1000,fstat_ms=(t2-t1)*1000,read_close_ms=(t3-t2)*1000,decode_ms=(t4-t3)*1000))
summary=[]
for repeat in range(2):
 sample=[x for x in rows if x['pass_index']==repeat and x['status']=='READ_DECODED'];summary.append(dict(pass_index=repeat,count=len(sample),**{k:dict(median=statistics.median(x[k] for x in sample),mean=statistics.mean(x[k] for x in sample),max=max(x[k] for x in sample)) for k in ['open_ms','fstat_ms','read_close_ms','decode_ms']}))
queue=json.loads((R/'queue_status.json').read_text());affinity=[]
for a in queue['active']:
 try:affinity.append(dict(pid=a['pid'],run_id=a['run_id'],allowed_cpu_count=len(os.sched_getaffinity(a['pid']))))
 except ProcessLookupError:affinity.append(dict(pid=a['pid'],run_id=a['run_id'],process_changed_during_read=True))
assert manifest_hash==hashlib.sha256(manifest.read_bytes()).hexdigest();out=dict(utc=datetime.now(timezone.utc).isoformat(),status='BOUNDED_SOURCE_ONLY_IO_MITIGATION_FEASIBILITY_PROBE',manifest_sha256=manifest_hash,source_role='DTRAINONLY_no_target_scores',selected_path_hashes=[hashlib.sha256(x.encode()).hexdigest() for x in selected],maximum_read_budget_bytes=32*1024**2,actual_read_bytes=byte_total,rows=rows,timing_ms=summary,trainer_affinity=affinity,scope='32deterministicexistingTRAINimages,twoordinaryreadonlypasses; timings are diagnostic notdetectorresults. No data/config/RNG/model/process/otherproject changes.',mitigation_candidates=[dict(action='change live dataloaderworkers or replace paths withSSDmirror',admitted=False,reason='no supported hot interface; would touchfrozenconfig/paths or require restart'),dict(action='global filesystem tuning or otherproject scheduling',admitted=False,reason='outside thisproject no interventionauthorized'),dict(action='full project page-cache warmup',admitted=False,reason='benefitnotproven by boundedprobe; additionalsharedIO mayworsen contention; alreadyboundedreadsonly')],GPU_launch_attempted=False,training_restarted=False)
(R/'reports/IO_mitigation_feasibility_20261024T1045.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print('SOURCE_IO_PROBE',out['utc'],'BYTES',byte_total,'SUMMARY',summary,'AFFINITY',affinity,flush=True)
