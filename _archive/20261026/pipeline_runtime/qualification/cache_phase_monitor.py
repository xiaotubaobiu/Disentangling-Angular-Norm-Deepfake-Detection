"""Read-only D cache-phase ownership, committed-chunk and throughput snapshot."""
from pathlib import Path
import argparse,json,os,time,datetime,subprocess,shutil,importlib.util
import numpy as np
import pandas as pd
import torch
from reliability_v2 import cache_d
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text())
def proc(item):
 root=Path('/proc')/str(item['pid']);argv=[x.decode() for x in (root/'cmdline').read_bytes().split(b'\0') if x];ppid=int((root/'stat').read_text().rsplit(')',1)[1].split()[1]);env=dict(x.split(b'=',1) for x in (root/'environ').read_bytes().split(b'\0') if b'=' in x);assert ppid==3942476 and argv[argv.index('-m')+1]=='reliability_v2.cache_d' and argv[argv.index('--run-id')+1]==item['run_id'] and argv[argv.index('--config')+1]==str(R/'configs/execution.json') and env[b'CUDA_VISIBLE_DEVICES'].decode()==item['uuid'];return dict(**item,parent_pid=ppid,module='reliability_v2.cache_d',CUDA_VISIBLE_DEVICES=env[b'CUDA_VISIBLE_DEVICES'].decode())
def held(path,pid):
 st=Path(path).stat();key=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}';lines=[l for l in Path('/proc/locks').read_text().splitlines() if key in l.split()];assert len(lines)==1 and int(lines[0].split()[4])==pid;return dict(path=str(path),expected_pid=pid,kernel_lock=lines[0])
def progress(run):
 roles=[]
 for role in cache_d.ROLES:
  root=R/'features/D'/run/role;markers=sorted(root.glob('chunk_*.json'));chunks=[read(p) for p in markers];times=[p.stat().st_mtime for p in markers];recent=[]
  for i in range(max(1,len(chunks)-3),len(chunks)):
   if times[i]>times[i-1]:recent.append(chunks[i]['rows']/(times[i]-times[i-1]))
  roles.append(dict(role=role,committed_chunks=len(chunks),committed_rows=sum(x['rows'] for x in chunks),role_marker_complete=(root/'complete.json').is_file(),recent_commit_rows_per_second=recent))
 return dict(run_id=run,utc=utc(),roles=roles,committed_rows=sum(x['committed_rows'] for x in roles))
def cpu():return [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
def main(stamp):
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized();output=R/f'reports/cache_runtime_{stamp}.json';assert not output.exists();start=utc();q=read(R/'queue_status.json');assert q['stage']=='P3_D_FP32_FEATURE_CACHE' and q['owner_pid']==3942476 and len(q['active'])<=4;active=q['active'];assert len(set(x['gpu'] for x in active))==len(active) and {x['gpu'] for x in active}<={0,1,2,3};ownership=[proc(x) for x in active];locks=[held(R/'queue.lock',3942476)]
 for x in active:locks.extend([held(R/'features/D'/x['run_id']/'extract.lock',x['pid']),held(R/'gpu_locks'/f"{x['uuid']}.lock",x['pid'])])
 first={x['run_id']:progress(x['run_id']) for x in active};wall0=time.perf_counter();cpu0=cpu();samples=[];checks=[];cfg=read(R/'configs/execution.json')
 for item in active:
  base,config,task=cache_d._context(R/'configs/execution.json',item['run_id']);checkpoint=torch.load(base['checkpoint_path'],map_location='cpu',weights_only=False);w=checkpoint['model']['classifier.weight'].numpy();b=checkpoint['model']['classifier.bias'].numpy();roles=[]
  for role in cache_d.ROLES:
   root=R/'features/D'/item['run_id']/role
   if not (root/'identity.json').exists():continue
   if not all((root/name).is_file() for name in ['manifest.parquet','head_weight.npy','head_bias.npy']):
    roles.append(dict(role=role,status='PENDING_NEW_ROLE_HEADER_NOT_YET_VERIFIABLE',verified_chunks=0,verified_rows=0));continue
   expected=cache_d._role_identity(base,role,R/f'data/D/{role}.parquet');assert read(root/'identity.json')==expected and cache_d.trainer.digest(root/'manifest.parquet')==expected['source_manifest_sha256'];frame=pd.read_parquet(expected['source_manifest_path']);chunks,position=cache_d._read_chunks(root,frame,expected,config.get('cache_chunk_rows',4096));weight=np.load(root/'head_weight.npy',allow_pickle=False);bias=np.load(root/'head_bias.npy',allow_pickle=False);assert weight.dtype==bias.dtype==np.float32 and np.array_equal(weight,w) and np.array_equal(bias,b);complete=(root/'complete.json').exists()
   roles.append(dict(role=role,status='COMPLETE_MARKER_AND_ARRAYS_SNAPSHOT_ONLY' if complete else 'PASS_COMMITTED_PARTIAL_CHUNKS_ONLY',verified_chunks=len(chunks),verified_rows=position,expected_rows=len(frame),dimension=base['dimension'],original_FP32_head_equal_final_checkpoint=True,zero_norm_count=sum(c['zero_norm_count'] for c in chunks)))
  checks.append(dict(run_id=item['run_id'],roles=roles,context_checkpoint_sha256=base['checkpoint_sha256']));print('PARTIAL_CACHE_VERIFIED',item['run_id'],[(x['role'],x['verified_rows']) for x in roles],flush=True);del checkpoint
 for i in range(8):
  samples.append(dict(utc=utc(),csv=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True)))
  if i<7:time.sleep(2)
 remaining=45-(time.perf_counter()-wall0)
 if remaining>0:time.sleep(min(45,remaining))
 wall1=time.perf_counter();cpu1=cpu();last={x['run_id']:progress(x['run_id']) for x in active};qend=read(R/'queue_status.json');assert qend['owner_pid']==3942476 and qend['stage']=='P3_D_FP32_FEATURE_CACHE';means=[]
 for gpu in range(4):values=[int(l.split(',')[1]) for s in samples for l in s['csv'].strip().splitlines() if int(l.split(',')[0])==gpu];means.append(sum(values)/len(values))
 rates=[]
 for x in active:
  run=x['run_id'];assert last[run]['committed_rows']>=first[run]['committed_rows'];rates.append(dict(run_id=run,gpu=x['gpu'],committed_rows_start=first[run]['committed_rows'],committed_rows_end=last[run]['committed_rows'],elapsed_seconds=wall1-wall0,committed_rows_per_wall_second=(last[run]['committed_rows']-first[run]['committed_rows'])/(wall1-wall0),quantized_by_chunk_rows=cfg.get('cache_chunk_rows',4096)))
 spec=importlib.util.spec_from_file_location('accepted_training_logs',R/'qualification/runtime_monitor_20261024T1300.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);prior={x['run_id']:set(x.get('skip_positions',[])) for x in read(R/'reports/check_20261024T1300Z.json')['rows']};counts=dict(development=0,formal=0,total=0,attempts=0,successful_updates=0);events=[];new=[]
 for run,task in cfg['runs'].items():
  records,skip,streak,files,partial=m.logs(run);assert len(records)==23970 and not partial and (R/'runs'/run/'complete.json').is_file();counts[task['phase']]+=len(skip);counts['total']+=len(skip);counts['attempts']+=len(records);counts['successful_updates']+=sum(x['success'] for x in records)
  for e in skip:
   events.append(dict(run_id=run,**e))
   if e['event']['attempt'] not in prior[run]:new.append(dict(run_id=run,epoch=e['event']['epoch'],attempt=e['event']['attempt'],next_attempt=e['next_success']['attempt']))
 assert counts==dict(development=32,formal=57,total=89,attempts=599250,successful_updates=599161) and all(e['next_event_status']=='SUCCESS' for e in events);delta=[b-a for a,b in zip(cpu0,cpu1)];total=sum(delta);formal=[r for r,t in cfg['runs'].items() if t['phase']=='formal' and t.get('route')=='D'];capacity=dict(workspace_free_GiB=shutil.disk_usage(R).free/1024**3,Dataset_free_GiB=shutil.disk_usage('/Dataset').free/1024**3);assert capacity['workspace_free_GiB']>250
 record=dict(utc=utc(),started_utc=start,status='PASS_ORIGINAL_CACHE_WORKERS_COMMITTED_ARRAYS_QUICK_SNAPSHOT_ONLY',queue_start=q,queue_end=qend,ownership=ownership,kernel_lock_proofs=locks,cache_checks=checks,progress_start=first,progress_end=last,committed_throughput=rates,GPU_samples=samples,GPU_mean_util_percent=means,CPU_busy_percent=100*(total-delta[3]-delta[4])/total,CPU_iowait_percent=100*delta[4]/total,capacity=capacity,working_set_estimate_GiB=200,reserve_GiB=50,raw_D_feature_floor_is_not_total=True,model_cache_complete_markers=sum((R/'features/D'/r/'complete.json').is_file() for r in formal),role_cache_complete_markers=sum((R/'features/D'/r/role/'complete.json').is_file() for r in formal for role in cache_d.ROLES),expected_model_role_caches=105,training_final_counts=counts,training_skip_events=events,new_training_skips_since1300=new,protected_processes=[dict(pid=p,alive=Path('/proc',str(p)).exists()) for p in [1532179,1532180,1532181,1532182]],CUDA_initialized=torch.cuda.is_initialized(),production_processes_restarted=False,new_dispatcher_created=False,H_GPU_started=False,full_role_scientific_verification_performed=False,live_source_image_stat_or_byte_reaudit_performed=False,scope='Read-only original cache ownership and committed chunks at mixed explicit cutoffs. Partial chunk proof is not complete-model or seven-role scientific acceptance. Throughput is quantized committed rows, not pure GPU kernel speed. No training-only worker assertion is used.')
 assert not record['CUDA_initialized'] and all(x['alive'] for x in record['protected_processes']);output.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=record['status'],output=str(output),GPU_mean_util_percent=means,capacity=capacity,committed_throughput=rates,model_cache_complete_markers=record['model_cache_complete_markers'],role_cache_complete_markers=record['role_cache_complete_markers'],training_final_counts=counts,new_skips=new)),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stamp',required=True);args=p.parse_args();assert args.stamp.isalnum();main(args.stamp)
