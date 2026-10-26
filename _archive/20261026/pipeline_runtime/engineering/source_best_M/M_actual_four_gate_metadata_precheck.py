"""Bounded, read-only metadata PREPARATION entry. Never issues a gate or permit.
No reliability_v2 imports, models, payload reads, slots, locks, PID queries or jobs.
A zero exit only means the preparation document's small bindings were consistent.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,datetime
MAX=1024**2

def require(value,message):
 if not value:raise ValueError(message)

def bounded(path,root,pin=None):
 p=Path(path).absolute();root=Path(root).absolute()
 require(p.is_relative_to(root) and p.resolve().is_relative_to(root),'metadata path escape')
 for q in (p,*p.parents):
  require(not q.is_symlink(),'metadata symlink refused')
  if q==root:break
 require(p.is_file() and p.stat().st_size<=MAX,'small regular metadata/code required')
 with p.open('rb') as f:raw=f.read(MAX+1)
 require(len(raw)<=MAX,'metadata/code grew beyond bound')
 digest=hashlib.sha256(raw).hexdigest()
 if pin is not None:require(digest==pin,'metadata/code SHA drift:'+str(p))
 return raw,digest

def parse(raw):
 def pairs(items):
  out={}
  for key,value in items:
   require(key not in out,'duplicate metadata key');out[key]=value
  return out
 return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite metadata')))

def main(argv=None):
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('--review',required=True);a.add_argument('--review-sha',required=True);args=a.parse_args(argv)
 review_path=Path(args.review).absolute();root=review_path.parents[2]
 raw,_=bounded(review_path,root,args.review_sha);d=parse(raw)
 require(d.get('schema')=='M_ACTUAL_FOUR_GATE_REVIEW_PREPARATION_V1' and d.get('status')=='PREPARATION_ONLY_NO_GATE_OR_PERMIT','preparation document required')
 require(d.get('artifact_root')==str(root) and d.get('M_production_cache_admission') is False and d.get('production_admitted') is False,'preparation cannot authorize production')
 checked=[]
 for ref in d['small_immutable_metadata_bindings']:
  bounded(ref['path'],root,ref['sha256']);checked.append(ref['path'])
 for path,pin in d['fixed_code_sha256'].items():bounded(path,root,pin)
 inp=parse(bounded(d['input_ref']['path'],root,d['input_ref']['sha256'])[0]);plan=parse(bounded(d['initial_plan_ref']['path'],root,d['initial_plan_ref']['sha256'])[0])
 require(plan['input_admission_sha256']==d['input_ref']['sha256'],'plan/input mismatch')
 source=parse(bounded(inp['source_file_admission']['path'],root,inp['source_file_admission']['sha256'])[0]);runs=set(d['fixed_inventory']['feature_runs']);jobs=set(d['fixed_inventory']['job_ids'])
 require(len(runs)==45 and set(source['records'])==runs and set(plan['feature_runs'])==runs,'exact45 metadata inventory required')
 require(all(record['route']==run[0] for run,record in source['records'].items()),'source route metadata mismatch')
 require(len(jobs)==30 and set(plan['jobs'])==jobs and all(x['output'] is None for x in plan['jobs'].values()),'initial full30 no-credit plan required')
 total=plan['base_future_and_reserve_bytes']+plan['feature_cap_bytes']+sum(x['cap_bytes'] for x in plan['jobs'].values())
 require(total==d['initial_budget']['required_initial_bytes']==165195057321,'initial arithmetic drift')
 require(plan['runtime_floor_bytes']==50*1024**3 and plan['global_M_temporary_bytes']==6*1024**3,'floor/temp metadata drift')
 result=dict(status='METADATA_PREPARATION_ONLY_NOT_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checked_small_bindings=len(checked),checked_code_files=len(d['fixed_code_sha256']),required_initial_bytes=total,required_plus_publication_bytes=total+MAX,live_free_bytes=shutil.disk_usage(root).free,capacity_reserved=False,payload_validated=False,CPU_slot_taken=False,gate_or_permit_written=False,M_production_cache_admission=False,production_admitted=False,compression_or_feature_credit_bytes=0)
 for name,path in d['future_completion_paths'].items():
  p=Path(path);item={'path':str(p),'present':p.exists(),'accepted_by_this_entry':False}
  if p.exists():
   value,pin=bounded(p,root);meta=parse(value);item.update(sha256=pin,schema=meta.get('schema'),status=meta.get('status'))
  result[name]=item
 print(json.dumps(result,ensure_ascii=False,indent=2));return 0

if __name__=='__main__':
 try:sys.exit(main())
 except Exception as exc:
  print(json.dumps(dict(status='METADATA_PREPARATION_INPUT_ERROR',error=str(exc),gate_or_permit_written=False,production_admitted=False)));sys.exit(2)
