"""Fail-closed selected-M CPU entry. No permit issuer or automatic dispatcher."""
from pathlib import Path
import argparse,copy,hashlib,json,os,resource,shutil,time
from . import source_best_cache as cache,source_best_stats as stats,source_best_sink as sink,source_best_runtime as runtime,source_best_files as sf

SEEDS=(7,42,123,2026,3407)
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
PLAN_SCHEMA='M_FIXED_REMAINING_WRITE_PLAN_V1'
ADMISSION_SCHEMA='M_SOURCE_BEST_JOB_CACHE_INDEPENDENT_ADMISSION_V1'
PERMIT_SCHEMA='M_PRODUCTION_STATISTIC_ADMISSION_V1'
CANARY_SCHEMA='M_BOUNDED_STATISTIC_CANARY_ADMISSION_V1'
CANARY_JOBS=('D_clip_vit_b16_seed7','H_clip_vit_b16_seed7')
require=runtime.require

def job(route,backbone,seed):
 require(route in ('D','H') and backbone in BACKBONES and type(seed) is int and seed in SEEDS,'registered fixed selected-M job required')
 runs={'D':f'D_formal_{backbone}_seed{seed}'} if route=='D' else {h:f'H_formal_{backbone}_{h}_seed{seed}' for h in ('B','AB')}
 return dict(job_id=f'{route}_{backbone}_seed{seed}',route=route,backbone=backbone,seed=seed,runs=runs)
def expected_jobs():return tuple(job(r,b,s)['job_id'] for r in ('D','H') for b in BACKBONES for s in SEEDS)
def expected_feature_runs():return tuple(run for r in ('D','H') for b in BACKBONES for s in SEEDS for run in job(r,b,s)['runs'].values())

def implementations():
 modules=(cache,stats,sink,runtime,sf,sink.io,sink.bundle,sink.bundle.codec,sink.bundle.io)
 return {str(Path(m.__file__)):sf.digest(m.__file__) for m in modules}|{str(Path(__file__)):sf.digest(__file__)}

def identity(spec,input_sha,contexts,markers):
 return dict(schema='M_STATISTIC_STATIC_IDENTITY_V1',scope='SELECTED_FOUR_CORE_SOURCE_BEST_DESCRIPTIVE_SENSITIVITY',job=copy.deepcopy(spec),input_admission_sha256=input_sha,source_contexts=copy.deepcopy(contexts),cache_completion_sha256=copy.deepcopy(markers),code_sha256=implementations(),checkpoint_selection='source_best',CAL_repeats=200,TEST_repeats=2000,resampling_seed=20261023,update_matched_primary_claim=False)

def _integer(value,name,minimum=0):require(type(value) is int and value>=minimum,name+' strict integer bound required');return value

def remaining_required(root,plan):
 """Remaining WRITE bounds; consumed bytes are not scientific acceptance.

 A registered append-only output has a proven total allocation cap, so bytes
 already occupying that cap reduce future writes. No compression-ratio,
 cache-reuse, snapshot, deletion or scientific credit is inferred here.
 """
 root=Path(root).absolute();require(plan.get('schema')==PLAN_SCHEMA,'fixed complete remaining write plan required')
 for k in ('base_future_and_reserve_bytes','runtime_floor_bytes','global_M_temporary_bytes','feature_cap_bytes'):_integer(plan[k],k,1)
 require(plan['runtime_floor_bytes']<=plan['base_future_and_reserve_bytes'] and plan['global_M_temporary_bytes']<=plan['base_future_and_reserve_bytes'],'reserve/temp accounting invalid')
 require(type(plan['feature_runs']) is list and len(plan['feature_runs'])==45 and set(plan['feature_runs'])==set(expected_feature_runs()),'exact45 feature endpoints required')
 require(type(plan['jobs']) is dict and set(plan['jobs'])==set(expected_jobs()),'all30 statistic jobs and all5seeds required')
 sf_hash=plan['input_admission_sha256'];sink.bundle.io._hash_identity(sf_hash)
 feature=root/'features/source_best_M'/sf_hash;cache._inside(feature,root);used=0
 if feature.exists():
  require(all(p.is_dir() and p.name in plan['feature_runs'] for p in feature.iterdir()),'foreign feature namespace cannot receive byte accounting')
  used=sink._usage(feature)
 required=plan['base_future_and_reserve_bytes']+max(0,plan['feature_cap_bytes']-used)
 for name,item in plan['jobs'].items():
  require(set(item)=={'cap_bytes','output'},'exact job capacity fields required');cap=_integer(item['cap_bytes'],'job cap',1);consumed=0
  if item['output'] is not None:
   output=runtime.output_path(root,item['output']);require(output.name==name,'registered output job mismatch')
   if output.exists():
    claim=output/'identity.json';require(claim.exists(),'unclaimed outputs cannot receive bounded-write accounting')
    require(claim.stat().st_size<=1024**2,'bounded sink identity claim required');saved=cache._json(claim);require(saved.get('schema')==sink.SCHEMA and saved.get('limits',{}).get('max_total_bytes')==cap and saved.get('implementations')==sink._codes(),'output allocation contract drift')
    consumed=sink._usage(output)-sink._usage(output/'.staging');require(consumed<=cap,'bounded output already exceeds cap')
  required+=max(0,cap-consumed)
 staged=sum(runtime.staging_usage(root).values())
 require(staged<=plan['global_M_temporary_bytes'],'retained global staging exceeds pool')
 return required-min(staged,plan['global_M_temporary_bytes'])

def _proof(root,path,pin):
 p=cache._pin(path,pin,{},root);require(p.stat().st_size<=1024**2,'bounded production proof required');d=cache._json(p);sink.typed(d);return d

def qualified_resources(permit,qualified,job_id):
 require(all(qualified.get(k) is True for k in ('actual_D_transport_accepted','actual_H_terminal_bound_transport_accepted','independent_entry_review_passed')),'real D/H transport and independent entry review required')
 if permit.get('schema')==CANARY_SCHEMA:
  require(type(permit.get('job_ids')) is list and len(permit['job_ids'])==2 and set(permit['job_ids'])==set(CANARY_JOBS) and job_id in CANARY_JOBS,'only the two fixed first planned jobs may qualify resource use')
  require(qualified.get('actual_entry_resource_qualified') is False and qualified.get('bounded_canary_resource_plan_independently_accepted') is True,'bounded canary plan must independently precede actual resource qualification')
 else:
  require(permit.get('schema')==PERMIT_SCHEMA and type(permit.get('job_ids')) is list and len(permit['job_ids'])==30 and set(permit['job_ids'])==set(expected_jobs()) and job_id in permit['job_ids'] and qualified.get('actual_entry_resource_qualified') is True,'full matrix needs actual canary resource acceptance')

def _preflight(root,input_path,input_sha,admission_path,admission_sha,permit_path,permit_sha):
 root=Path(root).absolute();require(root.resolve()==root,'canonical artifact root required')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='2' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'CPU-only emptyCUDA and qualified2threads required')
 inp=_proof(root,input_path,input_sha);require(inp.get('schema')==cache.INPUT_SCHEMA and inp.get('artifact_root')==str(root) and inp.get('cache_code_sha256')==sf.digest(cache.__file__) and inp.get('source_input_scope')=='SELECTED_M_D6_H3_FIVE_SEEDS' and inp.get('target_inference_authorized') is False,'accepted input context capsule required')
 ad=_proof(root,admission_path,admission_sha);spec=ad['job'];require(spec==job(spec['route'],spec['backbone'],spec['seed']),'exact registered job specification required')
 require(ad.get('schema')==ADMISSION_SCHEMA and ad.get('artifact_root')==str(root) and ad.get('status')=='PASS' and ad.get('independent_role_and_source_best_head_validation_passed') is True and ad.get('input_admission_sha256')==input_sha and ad.get('cache_code_sha256')==sf.digest(cache.__file__),'independent selected-M exact cache admission required')
 permit=_proof(root,permit_path,permit_sha);require(permit.get('schema') in (PERMIT_SCHEMA,CANARY_SCHEMA) and permit.get('status')=='PASS' and permit.get('artifact_root')==str(root) and permit.get('input_admission_sha256')==input_sha and permit.get('four_core_comparisons_only') is True and permit.get('all_five_seeds_required') is True,'separate M statistic permit required')
 require(permit.get('code_sha256')==implementations(),'fixed whole implementation inventory required')
 q=permit['qualification'];qualified=_proof(root,q['path'],q['sha256']);require(qualified.get('schema')=='M_STATISTIC_RESOURCE_AND_FORMAT_ACCEPTANCE_V1' and qualified.get('status')=='PASS' and qualified.get('code_sha256')==implementations() and qualified.get('input_admission_sha256')==input_sha,'bound format/entry engineering acceptance required');qualified_resources(permit,qualified,spec['job_id'])
 ref=permit['capacity'];plan=_proof(root,ref['path'],ref['sha256']);require(plan.get('input_admission_sha256')==input_sha and plan['runtime_floor_bytes']>=50*1024**3,'50GiB reserve and same input budget required');required=remaining_required(root,plan);require(shutil.disk_usage(root).free>=required,'full remaining write budget+temp+reserve not covered')
 return inp,ad,permit,plan

def execute(root,input_path,input_sha,admission_path,admission_sha,permit_path,permit_sha,*,slot):
 start=time.monotonic();root=Path(root).absolute();inp,ad,permit,plan=_preflight(root,input_path,input_sha,admission_path,admission_sha,permit_path,permit_sha);spec=ad['job'];initial_code=implementations();contexts={};data={};markers={}
 with runtime.cpu_slot(root,slot):
  # Slot held before expensive original source/file or feature verification.
  for label,run in spec['runs'].items():
   c=cache.prepare_context(root,input_path,input_sha,run);require(c.source['route']==spec['route'] and c.task['seed']==spec['seed'] and c.task['backbone']==spec['backbone'] and (spec['route']=='D' or c.task['history']==label),'source context job mismatch');cache._permit(c,permit['cache_permit']['path'],permit['cache_permit']['sha256']);contexts[label]=c
  bases={label:c.base for label,c in contexts.items()};require(sink.typed(bases)==sink.typed(ad['source_contexts']),'independently admitted source-best contexts changed')
  for label,c in contexts.items():
   for role in c.roles:
    p=cache.role_path(c,role)/'complete.json';markers[label+'/'+role]=sf.digest(p)
  require(markers==ad['cache_completion_sha256'] and len(markers)==6,'exact6 selected roles (D6 or H2x3) required')
  ident=identity(spec,input_sha,bases,markers);out=root/'analysis/source_best_M'/hashlib.sha256(sink.typed(ident)).hexdigest()/spec['job_id'];record=plan['jobs'][spec['job_id']];limits=permit['limits'][spec['route']]
  require(record['output']==str(out) and record['cap_bytes']==limits['max_total_bytes'],'capacity plan must bind this exact source/code output')
  require(limits['max_temporary_bytes']<=plan['global_M_temporary_bytes'],'temporary reservation exceeds global pool')
  with runtime.TemporaryPool(root,out,limits['max_temporary_bytes'],plan['global_M_temporary_bytes']):
   require(shutil.disk_usage(root).free>=remaining_required(root,plan),'remaining capacity changed before compute')
   def guard(full=False):
    require(implementations()==initial_code,'entry/scientific implementation drift')
    _proof(root,admission_path,admission_sha);_proof(root,permit_path,permit_sha)
    require(shutil.disk_usage(root).free>=plan['runtime_floor_bytes'],'runtime reserve floor crossed')
    for label,c in contexts.items():
     c.check(full=full) if full else None
     for role in c.roles:require(sf.digest(cache.role_path(c,role)/'complete.json')==markers[label+'/'+role],'admitted cache role marker drift')
   guard(True)
   with sink.CompactSink(root,out,ident,inventory=sink.declared_inventory(spec['route']),limits=limits) as emit:
    if not emit.completed:
     for label,c in contexts.items():
      data[label]={}
      for role in c.roles:
       meta,frame,h,w,b=cache.load_role(c,role);data[label][role]=(frame,h,w.copy(),b.copy())
     def saved(name,kind,value):guard(False);result=emit(name,kind,value);guard(False);return result
     selected={label:dict(c.source,registered_terminal_epochs=c.task['epochs']) for label,c in contexts.items()}
     if spec['route']=='D':stats.d_driver(data['D'],selected['D'],cache._json(root/'data/D/freeze.json'),saved)
     else:stats.h_driver(data,selected,saved)
    # Revalidate every saved cache payload and original source context after
    # computation, before the transport completion marker can be committed.
    for c in contexts.values():
     for role in c.roles:cache.verify_role(c,role)
    guard(True);result=emit.finish()
   receipt=dict(schema='M_STATISTIC_EXECUTION_RETURN_V1',status='COMPLETE_DECLARED_TRANSPORT_PENDING_INDEPENDENT_SAVED_RESULT_AUDIT',job=spec,output=str(out),complete_sha256=sf.digest(out/'complete.json'),identity_sha256=hashlib.sha256(sink.typed(ident)).hexdigest(),independent_cache_admission_sha256=admission_sha,production_permit_sha256=permit_sha,CPU_slot=slot,seconds=time.monotonic()-start,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,scientific_final_acceptance=False,all_five_seed_conclusions_available=False,GPU_or_model_training_started=False,source_best_only=True,bounded_resource_canary=permit['schema']==CANARY_SCHEMA)
   folder=root/'reports/M_statistic_workers';sink._dirs(root,folder);target=folder/(spec['job_id']+'_'+receipt['complete_sha256']+'.json')
   with cache._lock(target.with_suffix('.lock'),root):
    if target.exists():
     prior=runtime.read(target);require(prior['complete_sha256']==receipt['complete_sha256'] and prior['identity_sha256']==receipt['identity_sha256'],'existing execution receipt drift');return prior
    runtime.publish(target,receipt)
   return receipt

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true')
 for k in ('root','input','input-sha','admission','admission-sha','permit','permit-sha'):p.add_argument('--'+k)
 p.add_argument('--slot',type=int,choices=(0,1));a=p.parse_args(argv)
 if not a.execute:return dict(execute=False,production_admitted=False,scope='SelectedM gated CPU entry; no permit issuer')
 if any(getattr(a,k) is None for k in ('root','input','input_sha','admission','admission_sha','permit','permit_sha','slot')):p.error('execute requires all external SHA-bound proofs and slot')
 return execute(a.root,a.input,a.input_sha,a.admission,a.admission_sha,a.permit,a.permit_sha,slot=a.slot)
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
