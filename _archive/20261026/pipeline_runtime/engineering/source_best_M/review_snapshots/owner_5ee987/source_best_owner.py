"""Finite selected-M source-qualification/cache owner. No automation or issuer.

One original H_pipeline lease, at most one direct child per approved GPU0-3.
Failures stop new launches and drain healthy children. Resume needs explicit
SHA-bound prior receipts; no automatic restart or protected-GPU intervention.
"""
from pathlib import Path
import argparse,csv,hashlib,json,os,re,subprocess,sys,time
from types import SimpleNamespace
CODE=Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:sys.path.insert(0,str(CODE))
from reliability_v2 import source_best_lease as lease,source_best_cache as cache,source_best_source_qualification as qualification,source_best_gpu as gpu,source_best_analysis as analysis,source_best_runtime as runtime
require=runtime.require
SCHEMA='M_FINITE_OWNER_PLAN_V1'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def proof(root,ref):
 p=cache._pin(ref['path'],ref['sha256'],{},root);require(p.stat().st_size<=1024**2,'bounded owner proof required');return cache._json(p)

def schedule(run_ids,devices,start,verify,progress,wait):
 require(type(run_ids) is list and len(run_ids)==len(set(run_ids)) and bool(run_ids),'unique finite jobs required')
 pending=list(run_ids);active={};completed={};failures={}
 while pending or active:
  if not failures:
   try:
    available=devices();require(len(available)<=4 and len({d['uuid'] for d in available})==len(available) and len({d['index'] for d in available})==len(available) and all(type(d['index']) is int and d['index'] in (0,1,2,3) for d in available),'only four distinct approved GPU0-3 indexes and UUIDs')
   except Exception as error:
    if not active:raise
    failures['__owner__']=str(error)[:2000];available=[]
   used={v['device']['uuid'] for v in active.values()}
   for device in available:
    if not pending or failures:break
    if device['uuid'] in used:continue
    run=pending.pop(0)
    try:
     process=start(run,device);active[run]=dict(process=process,device=device);used.add(device['uuid'])
    except Exception as error:failures[run]=str(error)[:2000]
  changed=False
  for run,item in list(active.items()):
   try:code=item['process'].poll()
   except Exception as error:
    failures.setdefault('__poll__:'+run,str(error)[:2000])
    try:code=item['process'].wait()
    except Exception as wait_error:
     failures.setdefault('__reap__:'+run,str(wait_error)[:2000]);continue
   if code is None:continue
   try:item['process'].wait()
   except Exception as error:
    failures.setdefault('__reap__:'+run,str(error)[:2000]);continue
   changed=True
   if code!=0:failures[run]=dict(returncode=code)
   else:
    try:completed[run]=verify(run,item['process'],item['device'])
    except Exception as error:failures[run]=str(error)[:2000]
   del active[run]
  try:progress(pending,{r:dict(pid=v['process'].pid,**v['device']) for r,v in active.items()},completed,failures)
  except Exception as error:failures.setdefault('__progress__',str(error)[:2000])
  if not active:
   if failures:return dict(status='FAILED_WORKER_REQUIRES_REPAIR_NO_RETRY',completed=completed,pending=pending,failures=failures)
   if pending and not changed:return dict(status='BLOCKED_NO_IDLE_APPROVED_GPU',completed=completed,pending=pending,failures={})
  if active and not changed:
   try:wait()
   except Exception as error:
    failures.setdefault('__wait__',str(error)[:2000]);time.sleep(.01)
 return dict(status='ALL_REQUESTED_WORKERS_VERIFIED_OPERATIONALLY_ONLY',completed=completed,pending=[],failures={})

def idle_devices(config):
 require(config.get('allowed_gpus')==[0,1,2,3] and len(config.get('allowed_gpu_uuids',[]))==4,'frozen four-card configuration required')
 rows=list(csv.reader(lease._query(['--query-gpu=index,uuid,memory.used']).splitlines(),skipinitialspace=True));occupied={r[0].strip() for r in csv.reader(lease._query(['--query-compute-apps=gpu_uuid,pid']).splitlines(),skipinitialspace=True) if r};devices=[]
 for index,uuid,memory in rows:
  if uuid in config['allowed_gpu_uuids']:
   require(int(index) in (0,1,2,3),'configured UUID maps to protectedGPU')
   if uuid not in occupied and int(memory)<100:devices.append(dict(index=int(index),uuid=uuid))
 return sorted(devices,key=lambda d:d['index'])

def validate_plan(root,path,pin,*,initial=True):
 plan=proof(root,dict(path=str(path),sha256=pin));require(plan.get('schema')==SCHEMA and plan.get('artifact_root')==str(root) and plan.get('owner_code_sha256')==digest(__file__) and plan.get('code_root')==str(CODE),'bound canonical finite owner plan required')
 require(plan.get('mode') in ('source_qualification','cache'),'fixed finite mode required');inp=proof(root,plan['input']);require(inp.get('schema')==cache.INPUT_SCHEMA and inp.get('artifact_root')==str(root) and inp.get('cache_code_sha256')==digest(cache.__file__) and inp.get('target_inference_authorized') is False,'source input capsule required')
 expected=list(qualification.qualification_runs()) if plan['mode']=='source_qualification' else list(analysis.expected_feature_runs())
 require(type(plan.get('run_ids')) is list and plan['run_ids']==expected,'exact9source or45full registered models required')
 source=proof(root,inp['source_file_admission']);require(set(analysis.expected_feature_runs())==set(source['records']),'exact45 staged formal checkpoints required')
 require(plan.get('worker_code_sha256')==(qualification.codes() if plan['mode']=='source_qualification' else {str(Path(m.__file__)):digest(m.__file__) for m in (gpu,lease,cache)}),'worker source family changed')
 stage_sha=qualification.closed_primary_stage(root,inp)
 if plan['mode']=='source_qualification':qualification._plan(root,plan['worker_plan']['path'],plan['worker_plan']['sha256'],plan['input']['path'],plan['input']['sha256'])
 else:
  permit=proof(root,plan['cache_permit']);require(permit.get('schema')==cache.PERMIT_SCHEMA and permit.get('status')=='PASS_PRODUCTION_CACHE_ADMISSION' and permit.get('artifact_root')==str(root) and permit.get('input_admission_sha256')==plan['input']['sha256'] and permit.get('cache_code_sha256')==digest(cache.__file__) and permit.get('target_inference_authorized') is True and permit.get('four_core_comparisons_only') is True and len(permit.get('parent_run_ids',[]))==45 and set(permit['parent_run_ids'])==set(expected),'full fixed cache permit required')
  for name in ('cache_qualification','actual_D_storage','actual_H_storage','joint_capacity'):
   q=proof(root,permit[name]);require(q.get('M_production_cache_admission') is True and q.get('cache_code_sha256')==digest(cache.__file__) and q.get('input_admission_sha256')==plan['input']['sha256'],'unaccepted cache gate:'+name)
  gpu._bridge_gate(SimpleNamespace(root=root,admission_path=Path(plan['input']['path'])),permit,full=True)
  budget=proof(root,plan['budget']);require(budget['runtime_floor_bytes']>=50*1024**3,'50GiB runtime reserve required')
  import shutil
  require(shutil.disk_usage(root).free>=(analysis.remaining_required(root,budget) if initial else budget['runtime_floor_bytes']),'full initial or runtime resource budget failed')
 return plan,inp,stage_sha

def verify_return(root,plan,run,path,pin,*,expected_nonce=None,expected_pid=None,expected_owner=None,device=None):
 result=proof(root,dict(path=str(path),sha256=pin));nonce=result.get('phase_nonce');require(type(nonce) is str and re.fullmatch('[0-9a-f]{32}',nonce) is not None,'worker phase nonce required')
 expected=root/('reports/M_source_only_GPU_qualification' if plan['mode']=='source_qualification' else 'reports/M_source_best_workers')/nonce/(run+'.json');require(Path(path)==expected,'canonical worker receipt required')
 require(type(result.get('worker_pid')) is int and result['worker_pid']>0 and type(result.get('phase_owner_pid')) is int and result['phase_owner_pid']>0,'positive worker and phase owner identity required')
 if expected_nonce is not None:require(nonce==expected_nonce and result.get('worker_pid')==expected_pid and result.get('phase_owner_pid')==expected_owner,'current direct-child receipt identity mismatch')
 require(result.get('parent_run_id')==run and result.get('input_admission_sha256')==plan['input']['sha256'],'worker source endpoint/input mismatch')
 if plan['mode']=='source_qualification':
  require(result.get('primary_cache_stage_sha256')==qualification.closed_primary_stage(root,proof(root,plan['input'])),'qualification original primary stage binding changed')
  require(result.get('schema')=='M_ACTUAL_SOURCE_ONLY_GPU_QUALIFICATION_RETURN_V1' and result.get('qualification_plan_sha256')==plan['worker_plan']['sha256'] and result.get('code_sha256')==plan['worker_code_sha256'] and result.get('GPU_executed') is True and result.get('phase_native_mutual_exclusion_verified') is True and result.get('repeated_forward_FP32_bitwise_equal') is True and result.get('target_encoder_inference') is False and result.get('threshold_or_test_metric_computed') is False and type(result.get('optimizer_updates')) is int and result['optimizer_updates']==0,'actual source-onlyGPU proof required')
  role='source_cal' if run.startswith('D_') else 'B/SOURCE_CAL';require(result.get('source_role')==role and type(result.get('rows')) is int and result['rows']>=416 and result['fresh_forward_rows']==result['rows'],'complete source probe required');require(digest(root/'features/source_best_M'/plan['input']['sha256']/run/role/'complete.json')==result['source_only_cache_complete_sha256'],'source cache marker drift')
  if device is not None:require(result.get('approved_visible_GPU')==device['uuid'],'worker GPU mismatch')
 else:
  require(result.get('schema')=='M_SOURCE_BEST_OPERATION_RETURN_V1' and result.get('status')=='ALL_SELECTED_CANDIDATE_ROLES_VERIFIED_REQUIRES_INDEPENDENT_ADMISSION' and result.get('permit_sha256')==plan['cache_permit']['sha256'] and result.get('source_sha256')==plan['worker_code_sha256'],'exact cache worker return required')
  route='D' if run.startswith('D_') else 'H';require(set(result['roles'])==set(cache.ROLES[route]),'exact selected role inventory')
  for role,info in result['roles'].items():
   folder=root/'features/source_best_M'/plan['input']['sha256']/run/role;require(info['path']==str(folder) and digest(folder/'complete.json')==info['complete_sha256'],'cache worker role drift')
 return dict(path=str(path),sha256=pin)

def _status(path,value):
 runtime.sink._dirs(Path(value['artifact_root']),path.parent)
 raw=runtime.sink.typed(value)+b'\n';require(len(raw)<=1024**2,'bounded owner status required');tmp=path.with_suffix('.partial')
 with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path)

def execute_owner(root,plan_path,plan_sha):
 root=Path(root).absolute();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'owner must remainCPU-only/2threads')
 plan,inp,stage=validate_plan(root,plan_path,plan_sha);prior={}
 for ref in plan.get('resume_receipts',[]):
  require(ref['run_id'] in plan['run_ids'] and ref['run_id'] not in prior,'duplicate/unregistered resume endpoint');prior[ref['run_id']]=verify_return(root,plan,ref['run_id'],ref['path'],ref['sha256'],expected_nonce=ref['phase_nonce'],expected_pid=ref['worker_pid'],expected_owner=ref['phase_owner_pid'])
 config=cache._json(root/'configs/execution.json');handles=[]
 with lease.PhaseLease(root) as owned:
  validate_plan(root,plan_path,plan_sha);phase=owned.proof();home=root/'reports/M_finite_owner'/plan['mode']/plan_sha/phase['nonce'];runtime.sink._dirs(root,home);status=home/'status.json';start=time.monotonic()
  def guard():
   owned.check();require(validate_plan(root,plan_path,plan_sha,initial=False)[2]==stage,'original stage binding changed')
  def devices():guard();return idle_devices(config)
  def launch(run,device):
   guard();log=home/(run+'.log');error=home/(run+'.stderr.log');x=log.open('xb');y=error.open('xb');handles.extend([x,y]);env=dict(os.environ,CUDA_VISIBLE_DEVICES=device['uuid'],PYTHONPATH=str(CODE),PYTHONDONTWRITEBYTECODE='1')
   args=[str(root/'environment/bin/python'),'-B','-u',str(Path(__file__).resolve()),'--worker','--root',str(root),'--plan',str(plan_path),'--plan-sha',plan_sha,'--run-id',run,'--phase-proof',json.dumps(phase,separators=(',',':')),'--phase-fd',str(owned.fd)]
   return subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=x,stderr=y,pass_fds=(owned.fd,))
  def verify(run,process,device):
   folder='M_source_only_GPU_qualification' if plan['mode']=='source_qualification' else 'M_source_best_workers';p=root/'reports'/folder/phase['nonce']/(run+'.json');return verify_return(root,plan,run,p,digest(p),expected_nonce=phase['nonce'],expected_pid=process.pid,expected_owner=phase['owner_pid'],device=device)
  def progress(pending,active,completed,failures):
   _status(status,dict(schema='M_FINITE_OWNER_PROGRESS_V1',artifact_root=str(root),owner_pid=os.getpid(),owner_start_ticks=owned.birth,phase_nonce=phase['nonce'],mode=plan['mode'],owner_plan_sha256=plan_sha,primary_cache_stage_sha256=stage,completed=dict(prior,**completed),active=active,pending=pending,failures=failures,scientific_admission=False,seconds=time.monotonic()-start))
  try:result=schedule([run for run in plan['run_ids'] if run not in prior],devices,launch,verify,progress,lambda:time.sleep(1)) if len(prior)<len(plan['run_ids']) else dict(status='ALL_REQUESTED_WORKERS_VERIFIED_OPERATIONALLY_ONLY',completed={},pending=[],failures={})
  finally:
   for handle in handles:handle.close()
  result['completed']=dict(prior,**result['completed']);result.update(schema='M_FINITE_OWNER_TERMINAL_V1',artifact_root=str(root),mode=plan['mode'],owner_pid=os.getpid(),owner_plan_path=str(plan_path),owner_plan_sha256=plan_sha,owner_code_sha256=digest(__file__),phase_nonce=phase['nonce'],primary_cache_stage_sha256=stage,run_ids=plan['run_ids'],input_admission_sha256=plan['input']['sha256'],scientific_admission=False,M_production_qualified_here=False,seconds=time.monotonic()-start)
  guard();target=home/'complete.json';runtime.publish(target,result);return dict(terminal=str(target),terminal_sha256=digest(target),status=result['status'])

def worker(root,plan_path,plan_sha,run,phase,fd):
 root=Path(root);plan=proof(root,dict(path=plan_path,sha256=plan_sha));require(plan.get('owner_code_sha256')==digest(__file__) and run in plan['run_ids'],'worker wrapper code/endpoint changed')
 if plan['mode']=='source_qualification':return qualification.qualification_worker(root,plan['input']['path'],plan['input']['sha256'],run,plan_path=plan['worker_plan']['path'],plan_sha=plan['worker_plan']['sha256'],phase_proof=phase,phase_fd=fd)
 return gpu.extract_worker(root,plan['input']['path'],plan['input']['sha256'],run,permit_path=plan['cache_permit']['path'],permit_sha256=plan['cache_permit']['sha256'],phase_proof=phase,phase_fd=fd)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--worker',action='store_true')
 for k in ('root','plan','plan-sha','run-id','phase-proof'):p.add_argument('--'+k)
 p.add_argument('--phase-fd',type=int);a=p.parse_args(argv)
 if a.worker:return worker(a.root,a.plan,a.plan_sha,a.run_id,json.loads(a.phase_proof),a.phase_fd)
 if not a.execute:return dict(execute=False,scope='Finite approved-GPU owner only after closed originalH phase',production_admitted=False)
 if not all((a.root,a.plan,a.plan_sha)):p.error('root and external SHA-bound owner plan required')
 return execute_owner(a.root,a.plan,a.plan_sha)
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
