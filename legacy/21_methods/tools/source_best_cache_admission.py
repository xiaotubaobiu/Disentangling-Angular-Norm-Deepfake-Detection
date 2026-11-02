"""Fresh M cache-job admission and separate format/ownership collectors.

Data admission uses the unchanged full source/cache validators in a shared CPU
slot. Capacity closure checks owned formats/producer receipts, not image or
scientific acceptance. No target score, threshold or model training is produced.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,math,os,re,stat,sys,time
import numpy as np
import pyarrow.parquet as pq
CODE=Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:sys.path.insert(0,str(CODE))
if str(Path(__file__).parent) not in sys.path:sys.path.insert(0,str(Path(__file__).parent))
from reliability_v2 import source_best_cache as cache,source_best_analysis as analysis,source_best_runtime as runtime,source_best_source_qualification as qualification,source_best_gpu as gpu,source_best_lease as lease
import source_best_owner as owner
require=runtime.require
CLOSURE_SCHEMA='M_SELECTED_FEATURE_CAPACITY_CLOSURE_V1'

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as stream:
  for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def known_runs():return {run:dict(job=analysis.job(route,b,s),label=label) for route in ('D','H') for b in analysis.BACKBONES for s in analysis.SEEDS for label,run in analysis.job(route,b,s)['runs'].items()}

def _code_snapshot():
 return analysis.implementations()|{str(Path(__file__)):digest(__file__),str(Path(owner.__file__)):digest(owner.__file__)}


def _bounded_role(root,folder):
 root=Path(root).absolute();folder=cache._inside(folder,root);parts=folder.relative_to(root/'features/source_best_M').parts
 require(len(parts)>=3,'canonical owned role path required');input_sha,run=parts[:2];role='/'.join(parts[2:]);records=known_runs();require(run in records,'registered owned source endpoint required');spec=records[run]['job'];route=spec['route'];require(role in cache.ROLES[route],'owned selected role required')
 source=cache._inside(root/f'data/{route}'/(role+'.parquet'),root,file=True);rows=pq.ParquetFile(source).metadata.num_rows;dimension=cache.cache_d.DIMENSIONS[spec['backbone']]
 context=SimpleNamespace(root=root,admission_sha=input_sha,base={'parent_run_id':run,'dimension':dimension},roles=cache.ROLES[route],tables={role:{'rows':rows}})
 cache._model_inventory(context);cache._inventory(context,role)
 for name,shape in [('head_weight.npy',(2,dimension)),('head_bias.npy',(2,))]:cache._npy(folder/name,shape)
 manifest=folder/'manifest.parquet';require(manifest.is_file() and manifest.stat().st_size==source.stat().st_size,'exact bounded copied manifest size required')
 return context

def _derived_snapshot(root,folder,*,content_checksums):
 folder=cache._inside(folder,root);require(folder.is_dir() and not folder.is_symlink(),'regular derived role directory required');_bounded_role(root,folder);snapshot={}
 for path in sorted(folder.iterdir()):
  cache._inside(path,root,file=True);info=path.lstat();require(stat.S_ISREG(info.st_mode),'regular derived role file required')
  if re.fullmatch(r'chunk_[0-9]{6}\.npy',path.name) and not content_checksums:
   with path.open('rb') as stream:
    require(np.lib.format.read_magic(stream)==(1,0),'NPY v1 header required');np.lib.format.read_array_header_1_0(stream,max_header_size=4096);length=stream.tell();require(length<=4096,'bounded feature header required');stream.seek(0);pin=hashlib.sha256(stream.read(length)).hexdigest()
  else:pin=digest(path)
  snapshot[path.name]=(info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,pin)
 return snapshot

def _same_derived(root,snapshots):
 for folder,full,prior in snapshots:require(_derived_snapshot(root,folder,content_checksums=full)==prior,'derived role changed before aggregate admission publication')

def owned_role(root,input_sha,run,role,marker_sha,*,content_checksums=False):
 root=Path(root).absolute();records=known_runs();require(run in records,'registered source-best endpoint required');spec=records[run]['job'];route=spec['route'];dimension=cache.cache_d.DIMENSIONS[spec['backbone']];require(role in cache.ROLES[route],'selected native role required')
 folder=cache._inside(root/'features/source_best_M'/input_sha/run/role,root);marker=folder/'complete.json';require(marker.is_file() and marker.stat().st_size<=1024**2 and digest(marker)==marker_sha,'owned marker SHA mismatch');meta=cache._json(marker);ident=meta.get('identity',{});source=root/f'data/{route}'/(role+'.parquet');cache._inside(source,root,file=True)
 rows=pq.ParquetFile(source).metadata.num_rows;source_sha=digest(source)
 require(meta.get('status')=='CACHE_COMPLETE' and meta.get('precision')=='FP32' and meta.get('autocast') is False and type(meta.get('rows')) is int and meta['rows']==rows and meta.get('dimension')==dimension and meta.get('chunk_rows')==4096,'exact complete FP32 role format required')
 require(ident.get('checkpoint_selection')=='source_best' and ident.get('artifact_root')==str(root) and ident.get('run_id')==run and ident.get('parent_run_id')==run and ident.get('role')==role and ident.get('route')==route and ident.get('dimension')==dimension and ident.get('input_admission_sha256')==input_sha and ident.get('cache_code_sha256')==digest(cache.__file__) and ident.get('checkpoint_path')==str(root/'runs'/run/'source_best.pt') and ident.get('source_manifest_path')==str(source) and ident.get('source_manifest_sha256')==source_sha and ident.get('accepted_table_rows')==rows,'selected source/role/manifest ownership mismatch')
 require(type(ident.get('source_best_epoch')) is int and 1<=ident['source_best_epoch']<=60,'frozen source-best epoch disclosure required')
 c=SimpleNamespace(root=root,admission_sha=input_sha,base={'parent_run_id':run,'dimension':dimension},roles=cache.ROLES[route],tables={role:{'rows':rows}});cache._inventory(c,role)
 require(cache._json(folder/'identity.json')==ident and digest(folder/'manifest.parquet')==source_sha==meta['cached_manifest_sha256'],'exact cached role manifest/identity required')
 w=cache._npy(folder/'head_weight.npy',(2,dimension));b=cache._npy(folder/'head_bias.npy',(2,));head=cache._head_hash(w,b,dimension);require(head==ident['original_head_sha256'],'original source-best head bytes changed')
 chunks=meta['chunks'];require(type(chunks) is list and len(chunks)==math.ceil(rows/4096),'complete contiguous chunk inventory required');position=0
 for i,item in enumerate(chunks):
  count=min(4096,rows-position);require(item['file']==f'chunk_{i:06d}.npy' and item['start']==position and item['rows']==count and item['dimension']==dimension and item['dtype']=='float32','frozen contiguous chunk header contract')
  cache._npy(folder/item['file'],(count,dimension));require(cache._json(folder/f'chunk_{i:06d}.json')==item,'chunk metadata drift')
  if content_checksums:require(digest(folder/item['file'])==item['sha256'],'feature payload bytes changed')
  position+=count
 require(digest(marker)==marker_sha,'marker changed during ownership validation')
 return dict(rows=rows,dimension=dimension,head_sha256=head,complete_sha256=marker_sha,source_manifest_sha256=source_sha,owned_FP32_format_verified=True,all_feature_payload_checksums_recomputed=content_checksums,fresh_source_image_validation=False,source_best_epoch=ident['source_best_epoch'],checkpoint_sha256=ident['checkpoint_sha256'])

def admit_contexts(spec,contexts):
 require(spec==analysis.job(spec['route'],spec['backbone'],spec['seed']) and set(contexts)==set(spec['runs']),'exact D single-model or H matched pair required');markers={};heads={};pins={};snapshots=[];code=_code_snapshot()
 for label,c in contexts.items():
  require(c.base['parent_run_id']==spec['runs'][label] and c.source['route']==spec['route'] and c.task['seed']==spec['seed'] and c.task['backbone']==spec['backbone'],'requested source cache job mismatch');c.check(full=True)
  require(tuple(c.roles)==cache.ROLES[spec['route']],'all selected role checks required')
  for role in c.roles:
   folder=cache.role_path(c,role);before=_derived_snapshot(c.root,folder,content_checksums=True);snapshots.append((folder,True,before));meta,frame=cache.verify_role(c,role);_same_derived(c.root,[(folder,True,before)]);w=cache._npy(folder/'head_weight.npy',(2,c.base['dimension']));b=cache._npy(folder/'head_bias.npy',(2,));head=cache._head_hash(w,b,c.base['dimension']);require(head==c.source['head_sha256'],'fresh original selected head validation failed');heads[label+'/'+role]=head;markers[label+'/'+role]=digest(folder/'complete.json')
  c.check(full=True)
  for path,pin in c.pins.items():
   require(path not in pins or pins[path]==pin,'conflicting source proof');pins[path]=pin
 require(len(markers)==6,'exact D6 or H2x3 admitted roles required')
 for c in contexts.values():c.check()
 _same_derived(next(iter(contexts.values())).root,snapshots);require(_code_snapshot()==code,'source or issuer implementation changed during admission')
 roots={str(c.root) for c in contexts.values()};inputs={c.admission_sha for c in contexts.values()};require(len(roots)==len(inputs)==1,'matched root/input context required')
 return dict(schema=analysis.ADMISSION_SCHEMA,status='PASS',artifact_root=roots.pop(),job=spec,input_admission_sha256=inputs.pop(),cache_code_sha256=digest(cache.__file__),source_contexts={label:c.base for label,c in contexts.items()},cache_completion_sha256=markers,original_head_sha256_by_role=heads,pinned_source_proof_sha256=pins,independent_role_and_source_best_head_validation_passed=True,fresh_original_source_context_and_all_selected_role_checks=True,scientific_metric_or_bootstrap_acceptance=False,source_best_only=True)

def admit_job(root,input_ref,permit_ref,route,backbone,seed,slot):
 root=Path(root).absolute();require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU-only qualified2threads required');spec=analysis.job(route,backbone,seed);owner.proof(root,input_ref);owner.proof(root,permit_ref);start=time.monotonic()
 with runtime.cpu_slot(root,slot):
  contexts={}
  for label,run in spec['runs'].items():
   c=cache.prepare_context(root,input_ref['path'],input_ref['sha256'],run);cache._permit(c,permit_ref['path'],permit_ref['sha256']);contexts[label]=c
  result=admit_contexts(spec,contexts)
  for c in contexts.values():cache._permit(c,permit_ref['path'],permit_ref['sha256'])
  result.update(issuer_code_sha256=digest(__file__),CPU_slot=slot,seconds=time.monotonic()-start,GPU_launch_attempted=False)
  stable=hashlib.sha256(runtime.sink.typed({k:result[k] for k in ('job','input_admission_sha256','source_contexts','cache_completion_sha256')})).hexdigest();out=root/'qualification/M_independent_cache_jobs'/input_ref['sha256']/spec['job_id']/(stable+'.json');runtime.sink._dirs(root,out.parent)
  with cache._lock(out.with_suffix('.lock'),root):
   if out.exists():
    previous=cache._json(out);require(all(previous.get(k)==result[k] for k in ('job','input_admission_sha256','source_contexts','cache_completion_sha256','cache_code_sha256')),'prior admitted endpoint drift');return dict(path=str(out),sha256=digest(out))
   # Proof may contain >64KiB pinned original contexts; use bounded append-only
   # cache evidence writer, not the small reservation-record helper.
   cache._save_json(out,result)
  return dict(path=str(out),sha256=digest(out))

def read_owner_stage(root,stage_ref,mode):
 stage=owner.proof(root,stage_ref);require(stage.get('schema')=='M_FINITE_OWNER_TERMINAL_V1' and stage.get('status')=='ALL_REQUESTED_WORKERS_VERIFIED_OPERATIONALLY_ONLY' and stage.get('mode')==mode and stage.get('owner_code_sha256')==digest(owner.__file__),'qualified finite owner terminal required');plan,inp,primary=owner.validate_plan(root,stage['owner_plan_path'],stage['owner_plan_sha256'],initial=False)
 require(plan['mode']==mode and stage['run_ids']==plan['run_ids'] and stage['input_admission_sha256']==plan['input']['sha256'] and stage['primary_cache_stage_sha256']==primary and set(stage['completed'])==set(plan['run_ids']),'complete owner endpoint inventory required')
 expected=root/'reports/M_finite_owner'/mode/stage['owner_plan_sha256']/stage['phase_nonce']/'complete.json';require(Path(stage_ref['path'])==expected,'canonical owner stage path required')
 for run,ref in stage['completed'].items():owner.verify_return(root,plan,run,ref['path'],ref['sha256'])
 return stage,plan

def collect_source_qualification(root,stage_ref,slot):
 root=Path(root).absolute();stage,plan=read_owner_stage(root,stage_ref,'source_qualification');results={};snapshots=[];code=_code_snapshot()
 # Only the existing sourceCAL payloads are checked here. The producer's full
 # provenance/GPU forward is rebound, not repeated for all45 checkpoints.
 with runtime.cpu_slot(root,slot):
  for run,ref in stage['completed'].items():
   rec=owner.proof(root,ref);role='source_cal' if run.startswith('D_') else 'B/SOURCE_CAL';folder=root/'features/source_best_M'/plan['input']['sha256']/run/role;before=_derived_snapshot(root,folder,content_checksums=True);snapshots.append((folder,True,before));owned=owned_role(root,plan['input']['sha256'],run,role,rec['source_only_cache_complete_sha256'],content_checksums=True)
   require(owned['rows']==rec['rows'] and owned['head_sha256']==rec['original_head_sha256'] and owned['source_best_epoch']==rec['source_best_epoch'] and owned['checkpoint_sha256']==rec['checkpoint_sha256'],'source-only GPU/cache ownership binding failed');results[run]=owned
  require(set(results)==set(qualification.qualification_runs()),'all9 architectures/history probes required')
  require(read_owner_stage(root,stage_ref,'source_qualification')==(stage,plan),'source qualification owner stage changed');_same_derived(root,snapshots);require(_code_snapshot()==code,'source collector implementation changed')
  report=dict(schema='M_GPU_BRIDGE_QUALIFICATION_V1',status='PASS',input_admission_sha256=plan['input']['sha256'],real_source_only_GPU_qualification_passed=True,phase_native_mutual_exclusion_verified=True,source_sha256={str(Path(m.__file__)):digest(m.__file__) for m in (gpu,lease,cache)},qualification_worker_code_sha256=digest(qualification.__file__),collector_code_sha256=digest(__file__),owner_stage=stage_ref,results=results,actual_target_encoder_inference=False,fresh_45_checkpoint_tensor_validation_repeated=False,fresh_source_image_validation_repeated=False,independent_feature_activations_recomputed=False,full_M_production_permit_issued=False,M_production_cache_admission=False,scientific_M_results_accepted=False)
  out=root/'qualification/M_source_only_GPU_bridge'/plan['input']['sha256']/'accepted.json';runtime.sink._dirs(root,out.parent)
  with cache._lock(out.with_suffix('.lock'),root):cache._save_json(out,report)
  return dict(path=str(out),sha256=digest(out))

def _feature_capacity_report(root,stage_ref):
 root=Path(root).absolute();stage,plan=read_owner_stage(root,stage_ref,'cache');roles={};snapshots=[];code=_code_snapshot()
 for run,ref in stage['completed'].items():
  rec=owner.proof(root,ref)
  for role,item in rec['roles'].items():
   folder=root/'features/source_best_M'/plan['input']['sha256']/run/role;before=_derived_snapshot(root,folder,content_checksums=False);snapshots.append((folder,False,before));roles[run+'/'+role]=owned_role(root,plan['input']['sha256'],run,role,item['complete_sha256'],content_checksums=False)
 require(len(roles)==180 and set(stage['completed'])==set(analysis.expected_feature_runs()),'exact45models/180roles ownership closure required')
 require(read_owner_stage(root,stage_ref,'cache')==(stage,plan),'feature owner stage changed');_same_derived(root,snapshots);require(_code_snapshot()==code,'capacity collector implementation changed')
 report=dict(schema=CLOSURE_SCHEMA,status='PASS_OWNED_45MODEL_180ROLE_FORMATS_ZERO_SCHEDULED_FEATURE_WRITES',input_admission_sha256=plan['input']['sha256'],cache_code_sha256=digest(cache.__file__),collector_code_sha256=digest(__file__),owner_stage=stage_ref,roles=roles,run_ids=list(analysis.expected_feature_runs()),future_scheduled_feature_write_bytes=0,fresh_all_feature_payload_checksums_recomputed=False,fresh_source_image_validation=False,independent_data_or_scientific_admission=False,M_production_cache_admission=False)
 return report

def validate_feature_capacity(root,ref):
 root=Path(root).absolute();saved=owner.proof(root,ref);input_sha=saved.get('input_admission_sha256');require(type(input_sha) is str and len(input_sha)==64,'fixed input hash required')
 expected=root/'reports/M_feature_capacity_closure'/input_sha/'complete.json';require(Path(ref['path'])==expected and saved.get('collector_code_sha256')==digest(__file__),'canonical current collector closure required')
 fresh=_feature_capacity_report(root,saved['owner_stage']);require(saved==fresh,'owner stage or current owned feature formats changed')
 return 0

def collect_feature_capacity(root,stage_ref):
 root=Path(root).absolute();report=_feature_capacity_report(root,stage_ref)
 out=root/'reports/M_feature_capacity_closure'/report['input_admission_sha256']/'complete.json';runtime.sink._dirs(root,out.parent)
 with cache._lock(out.with_suffix('.lock'),root):cache._save_json(out,report)
 return dict(path=str(out),sha256=digest(out))

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--mode',choices=('job','source-qualification','feature-capacity'))
 for k in ('root','input','input-sha','permit','permit-sha','route','backbone','owner-stage','owner-stage-sha'):p.add_argument('--'+k)
 p.add_argument('--seed',type=int);p.add_argument('--slot',type=int,choices=(0,1));a=p.parse_args(argv)
 if not a.execute:return dict(execute=False,scope='Fresh cache job admission; separate sourceGPU/ownership collectors',M_production_admitted=False)
 if not a.root or not a.mode:p.error('explicit root and mode required')
 if a.mode=='job' and not all((a.input,a.input_sha,a.permit,a.permit_sha,a.route,a.backbone,a.seed is not None,a.slot is not None)):p.error('job requires bound inputs/permit/registered endpoint/shared CPU slot')
 if a.mode!='job' and not all((a.owner_stage,a.owner_stage_sha)):p.error('collector requires SHA-bound finite-owner terminal')
 if a.mode=='source-qualification' and a.slot is None:p.error('source qualification collector requires shared CPU slot')
 if a.mode=='job':return admit_job(a.root,dict(path=a.input,sha256=a.input_sha),dict(path=a.permit,sha256=a.permit_sha),a.route,a.backbone,a.seed,a.slot)
 ref=dict(path=a.owner_stage,sha256=a.owner_stage_sha)
 return collect_source_qualification(a.root,ref,a.slot) if a.mode=='source-qualification' else collect_feature_capacity(a.root,ref)
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
