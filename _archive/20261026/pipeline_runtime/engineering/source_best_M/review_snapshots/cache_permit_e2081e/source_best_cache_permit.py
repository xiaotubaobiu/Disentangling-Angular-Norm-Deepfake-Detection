"""Issue fixed-M cache/owner documents only from four independent actual gates.
No qualification is invented here and no GPU, analysis or dispatcher is started.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,shutil,sys
CODE=Path(__file__).resolve().parents[1]
for path in (CODE,Path(__file__).parent):
 if str(path) not in sys.path:sys.path.insert(0,str(path))
import source_best_owner as owner
from reliability_v2 import source_best_cache as cache,source_best_analysis as analysis,source_best_gpu as gpu,source_best_lease as lease,source_best_source_qualification as qualification,source_best_runtime as runtime
require=runtime.require
GATES=('cache_qualification','actual_D_storage','actual_H_storage','joint_capacity')

def assemble(root,input_ref,gates,bridge,budget_ref):
 root=Path(root).absolute();require(root.resolve()==root,'canonical artifact root required');inp=owner.proof(root,input_ref)
 require(inp.get('schema')==cache.INPUT_SCHEMA and inp.get('artifact_root')==str(root) and inp.get('cache_code_sha256')==owner.digest(cache.__file__) and inp.get('source_input_scope')=='SELECTED_M_D6_H3_FIVE_SEEDS' and inp.get('target_inference_authorized') is False,'fixed source-only input capsule required')
 require(type(gates) is dict and set(gates)==set(GATES) and len({gates[k]['path'] for k in GATES})==4,'four distinct exact independently accepted gate documents required')
 accepted={}
 for name in GATES:
  d=owner.proof(root,gates[name]);require(d.get('gate')==name and d.get('M_production_cache_admission') is True and d.get('independent_acceptance') is True and d.get('evidence_kind')=='actual' and d.get('input_admission_sha256')==input_ref['sha256'] and d.get('cache_code_sha256')==owner.digest(cache.__file__),'missing independent actual gate:'+name);accepted[name]=d
  if name!='joint_capacity':require(d.get('qualification')==bridge.get('qualification'),'actual storage and GPU gate qualification must match')
 q=accepted['cache_qualification'];require(q.get('real_source_only_GPU_qualification_passed') is True and q.get('run_ids')==list(qualification.qualification_runs()),'all9 real source-onlyGPU cases must be independently accepted')
 for route in ('D','H'):
  d=accepted['actual_'+route+'_storage'];require(d.get('route')==route and d.get('run_ids')==[r for r in qualification.qualification_runs() if r.startswith(route+'_')],'complete actual three-architecture storage group required')
 budget=owner.proof(root,budget_ref);require(budget.get('input_admission_sha256')==input_ref['sha256'],'same-input complete future write plan required');required=analysis.remaining_required(root,budget);capacity=accepted['joint_capacity']
 require(capacity.get('initial_remaining_plan')==budget_ref and type(capacity.get('initial_required_remaining_bytes')) is int and capacity['initial_required_remaining_bytes']>=required,'independent full initial capacity acceptance required')
 require(type(capacity.get('required_remaining_bytes')) is int and capacity['required_remaining_bytes']>=50*1024**3 and budget['runtime_floor_bytes']>=50*1024**3 and capacity['required_remaining_bytes']>=budget['runtime_floor_bytes'],'full runtime reserve gate required')
 require(shutil.disk_usage(root).free>=max(required,capacity['initial_required_remaining_bytes']),'actual free capacity below complete joint remaining budget')
 permit=dict(schema=cache.PERMIT_SCHEMA,status='PASS_PRODUCTION_CACHE_ADMISSION',artifact_root=str(root),input_admission_sha256=input_ref['sha256'],cache_code_sha256=owner.digest(cache.__file__),target_inference_authorized=True,four_core_comparisons_only=True,parent_run_ids=list(analysis.expected_feature_runs()),gpu_bridge=bridge,**gates)
 gpu._bridge_gate(SimpleNamespace(root=root,admission_path=Path(input_ref['path'])),permit,full=True)
 plan=dict(schema=owner.SCHEMA,artifact_root=str(root),owner_code_sha256=owner.digest(owner.__file__),code_root=str(CODE),mode='cache',input=input_ref,run_ids=list(analysis.expected_feature_runs()),worker_code_sha256={str(Path(m.__file__)):owner.digest(m.__file__) for m in (gpu,lease,cache)},budget=budget_ref,resume_receipts=[])
 return permit,plan

def issue(root,input_ref,gates,bridge,budget_ref):
 root=Path(root).absolute();before=analysis.implementations()|{str(Path(__file__)):owner.digest(__file__),str(Path(owner.__file__)):owner.digest(owner.__file__)};permit,plan=assemble(root,input_ref,gates,bridge,budget_ref);key=hashlib.sha256(runtime.sink.typed(dict(input=input_ref,gates=gates,bridge=bridge,budget=budget_ref,source=before))).hexdigest();folder=root/'qualification/M_cache_production'/input_ref['sha256']/key
 runtime.sink._dirs(root,folder)
 def publish(path,value):
  if path.exists():require(runtime.read(path)==value,'append-only permit namespace collision')
  else:runtime.publish(path,value)
  return dict(path=str(path),sha256=owner.digest(path))
 with cache._lock(folder/'issuer.lock',root):
  require(assemble(root,input_ref,gates,bridge,budget_ref)==(permit,plan),'gates changed before permit publication');require(analysis.implementations()|{str(Path(__file__)):owner.digest(__file__),str(Path(owner.__file__)):owner.digest(owner.__file__)}==before,'issuer implementation changed')
  permit_ref=publish(folder/'cache_permit.json',permit);plan['cache_permit']=permit_ref;plan_ref=publish(folder/'cache_owner_plan.json',plan)
  # The already qualified owner is the final immutable-plan/phase/budget gate.
  owner.validate_plan(root,plan_ref['path'],plan_ref['sha256']);return dict(cache_permit=permit_ref,owner_plan=plan_ref,GPU_or_dispatcher_started=False,statistics_permit_issued=False)

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--root');p.add_argument('--request');p.add_argument('--request-sha');a=p.parse_args(argv)
 if not a.execute:return dict(execute=False,qualification_or_budget_invented=False,GPU_started=False)
 if not all((a.root,a.request,a.request_sha)):p.error('root and external SHA-bound request required')
 root=Path(a.root).absolute();d=owner.proof(root,dict(path=a.request,sha256=a.request_sha));require(d.get('schema')=='M_CACHE_PERMIT_REQUEST_V1' and d.get('artifact_root')==str(root),'fixed external permit request required')
 return issue(root,d['input'],d['gates'],d['gpu_bridge'],d['initial_remaining_plan'])
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
