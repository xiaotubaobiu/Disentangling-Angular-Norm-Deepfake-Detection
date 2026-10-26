from pathlib import Path
import gc,hashlib,json,os,resource,time,datetime,traceback
from reliability_v2 import source_best_cache as m
import torch
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M'
resource.setrlimit(resource.RLIMIT_AS,(64*1024**3,64*1024**3))
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized()
admission=E/'actual_M_cache_input_context_candidate.json';pin=hashlib.sha256(admission.read_bytes()).hexdigest();assert json.loads(admission.read_text())['cache_code_sha256']==hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
assert not (R/'features/source_best_M').exists(),'read-only context probe must start without M cache artifacts'
records=[];started=time.monotonic();report=E/'actual_M_cache_input_context_probe.json';assert not report.exists()
try:
 for run in ['D_formal_clip_vit_b16_seed7','H_formal_clip_vit_b16_B_seed7']:
  print('BEGIN_ACTUAL_CACHE_CONTEXT',run,flush=True);tick=time.monotonic();c=m.prepare_context(R,admission,pin,run)
  records.append(dict(run_id=run,route=c.source['route'],base=c.base,roles=c.tables,selected_rows=sum(x['rows'] for x in c.tables.values()),input_sha256=c.pins,seconds=time.monotonic()-tick,target_inference_authorized=False))
  print('PASS_ACTUAL_CACHE_CONTEXT',run,records[-1]['selected_rows'],records[-1]['seconds'],flush=True);del c;gc.collect()
 assert not torch.cuda.is_initialized() and not (R/'features/source_best_M').exists()
 result=dict(status='PASS_TWO_REPRESENTATIVE_ACTUAL_M_CACHE_INPUT_CONTEXTS_READ_ONLY_NO_ENCODER',records=records,source_input_admission_sha256=pin,actual_model_contexts_checked=2,all45_cache_contexts_qualified=False,production_M_admitted=False,target_inference_executed=False,cache_artifacts_written=False,selected_M_statistics_implemented=False,seconds=time.monotonic()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
except BaseException as error:
 result=dict(status='FAILED_READ_ONLY_M_CONTEXT_PROBE_NO_PRODUCTION_AUTHORIZATION',records=records,source_input_admission_sha256=pin,error=repr(error),traceback=traceback.format_exc(),production_M_admitted=False,target_inference_executed=False)
 with report.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
 raise
with report.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print('ACTUAL_M_CONTEXT_PROBE_COMPLETE',str(report),flush=True)
