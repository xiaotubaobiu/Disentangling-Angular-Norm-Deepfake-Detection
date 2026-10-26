from pathlib import Path
import argparse,datetime,json,hashlib,os,time,gc,resource
import torch
from reliability_v2 import source_best_files
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023');E=R/'engineering/source_best_M'
def main():
 p=argparse.ArgumentParser();p.add_argument('--admission',required=True);p.add_argument('--admission-sha',required=True);p.add_argument('--report',required=True);p.add_argument('--run-id',action='append');args=p.parse_args()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k) in ('1','2') for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'));torch.set_num_threads(2);assert not torch.cuda.is_initialized()
 output=Path(args.report);assert output.parent==E and not output.exists();admission=json.loads(Path(args.admission).read_text());ids=args.run_id or list(admission['records']);assert len(set(ids))==len(ids)
 start=time.monotonic();records=[]
 for i,run in enumerate(ids,1):
  print('BEGIN_ACTUAL_SOURCE_FILE_TENSORS',i,len(ids),run,flush=True);tick=time.monotonic();item=source_best_files.validate_source_file(R,args.admission,args.admission_sha,run);item['validation_seconds']=time.monotonic()-tick;records.append(item);print('PASS_ACTUAL_SOURCE_FILE_TENSORS',run,'seconds',item['validation_seconds'],'tensor_count',item['tensor_count'],'head',item['head_sha256'],flush=True);gc.collect()
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_SELECTED_ACTUAL_SOURCE_CHECKPOINT_BYTES_TENSORS_AND_ORIGINAL_FP32_HEADS_ONLY',models=len(records),expected_models=len(ids),records=records,elapsed_seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,CUDA_initialized=torch.cuda.is_initialized(),actual_cache_context_qualified=False,target_inference_authorized=False,M_production_admitted=False,reuse_capacity_credit_bytes=0,scope='Actual source-only checkpoint files plus source identity/proofs revalidated; no GPU or target score/cache extraction, encoder construction, M statistics or end-to-end scientific claim.')
 with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
 print('COMPLETE_ACTUAL_SOURCE_FILE_VALIDATION',output,flush=True)
if __name__=='__main__':main()
