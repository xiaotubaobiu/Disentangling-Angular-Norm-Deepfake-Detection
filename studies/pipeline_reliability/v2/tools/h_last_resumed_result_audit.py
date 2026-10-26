"""Explicit resume-aware DERIVED consumer of fixed bb299e orchestration.

This is new consumer code, not an assertion that unmodified bb299e ran. Original
launch dictionaries are never normalized or given invented missing fields.
Only the two fixed20261026T013643Z resume records are compatible, backed by
fixed Detector-only authorization and original unfinished-user-pause evidence.
Normal launches retain the exact old gates. Four declared AST adaptation sites:
launch gate, launch-path gate, metadata return (additional provenance pins), and
a final source-attestation guard immediately before result publication.
The numeric orchestration, terminal72 inventory,42ledgers,8cohorts and original
6cd3 independent2000draw/NA/metric kernel are reused without algorithm changes.
Default preparation only. Actual audit still needs terminal SHA, external11refs,
producer true exit/log/time and the existing shared lifetime CPU slot.
"""
from pathlib import Path
import ast,copy,hashlib,json
ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
BB_PATH=ROOT/'engineering/source_best_M/code/studies/pipeline_reliability/v2/tools/h_last_matrix_result_audit.py'
BB_SHA='bb299e512e355fa0a8c9b25f901b1788cee6a2b445a4eb1ddb67b30cb4c3a0e2'
ENTRY_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
AUTHORIZATION_REF=dict(path=str(ROOT/'DETECTOR_RESUME_AUTHORIZED_20261026T0133.json'),sha256='3c820360f0e9e554fbcf7d941f5cfa996a84d2c82f59cef448521cdb4e00ade5')
PAUSE_REF=dict(path=str(ROOT/'reports/user_pause_20261025T2351/pause_verified.json'),sha256='3e2696907c8199c103c271366b5264c364f81869b7f8f2bca4282d21f1ff93a2')
RETAINED_PAUSE_REF=dict(path=str(ROOT/'reports/user_pause_20261025T2351/PAUSED_BY_USER.resolved_detector_20261026T0133.json'),sha256='62d47b959ba56d8293d83f98f98a2d4edf3dcbf5602ee2a42803365df79ad6e4')
RESUME_CASES={('clip_vit_b16',3407):dict(path=str(ROOT/'reports/H_primary_analysis_clip_vit_b16_seed3407_resume_20261026T013643Z_launch.json'),sha256='0d3a89013786d51824a4b8679f3e2ee57c0678c2370faef330c487591b6cf91b',slot=1,pause_tag='CLIP3407'),('dinov2_vit_b14',7):dict(path=str(ROOT/'reports/H_primary_analysis_dinov2_vit_b14_seed7_resume_20261026T013643Z_launch.json'),sha256='ea89f76b78671e6f942862d4247dd9524d61a8332151db9992a4ed6bb83ea113',slot=0,pause_tag='DINO7')}
RESUME_STATUS='RESUMED_ORIGINAL_PRIMARY_H_PAIR_STATISTICS_FROM_UNFINISHED_USER_PAUSE'

def _bound_json(ref):
 require(type(ref) is dict and set(ref)=={'path','sha256'} and type(ref['sha256']) is str and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'fixed JSON evidence reference')
 path=_safe(ref['path']);require(sha(path)==ref['sha256'],'bound JSON evidence drift '+str(path));return _json(path)

def validate_resume_launch(L,bone,seed,refs):
 require((bone,seed) in RESUME_CASES and type(seed) is int,'only two fixed real user-resume records')
 case=RESUME_CASES[(bone,seed)];ref=refs['primary_launch'];require(ref==dict(path=case['path'],sha256=case['sha256']),'exact resume launch path/rawSHA allowlist');require(_bound_json(ref)==L,'original resumed launch JSON must remain unmodified')
 require(L.get('status')==RESUME_STATUS and L.get('backbone')==bone and _int(L.get('seed'),seed) and _int(L.get('CPU_slot'),case['slot']) and type(L.get('pid')) is int and L['pid']>0,'fixed resumed launch identity')
 require(_int(L.get('bootstrap_repeats'),2000) and _int(L.get('expected_fresh_original_ledger_invocations'),42) and L.get('GPUs_allowed') is False and L.get('original_F_unchanged') is True and L.get('resume_existing_terminal') is False and L.get('unfinished_partial_identity_preserved') is True and L.get('independent_numeric_bootstrap_acceptance_pending') is True,'unfinished original42/2000 CPU resume only')
 require('M_statistics_started' not in L,'fixed resume schema absence must not be synthesized asFalse')
 require(L['entry']==refs['primary_entry']['path']==str(F/'cpu_primary_analysis.py') and L['entry_sha256']==refs['primary_entry']['sha256']==CPU_HASHES['cpu_primary_analysis.py'],'unchanged qualified original H-only CPU entry')
 stem=Path(case['path']).name.removesuffix('_launch.json');require(L['stdout']==refs['primary_stdout']['path']==str(ROOT/'reports'/f'{stem}.log') and L['time_v']==refs['primary_time_v']['path']==str(ROOT/'reports'/f'{stem}_time_v.log'),'exact resumed stdout/time paths')
 expected=['/usr/bin/time','-v','-o',L['time_v'],str(ROOT/'environment/bin/python'),'-B','-u',L['entry'],'--execute','--backbone',bone,'--seed',str(seed),'--slot',str(case['slot']),'--qualification',refs['CPU_qualification']['path'],'--qualification-sha',refs['CPU_qualification']['sha256'],'--admission',refs['cache_admission']['path'],'--admission-sha',refs['cache_admission']['sha256']]
 require(L['command']==expected,'exact bounded original F resume CLI without M/GPU/alternate entry')
 require(L['resume_authorization']==AUTHORIZATION_REF,'explicit bound resume authorization');auth=_bound_json(AUTHORIZATION_REF);pause=_bound_json(PAUSE_REF);retained=_bound_json(RETAINED_PAUSE_REF)
 require(auth['schema']=='DETECTOR_V2_EXPLICIT_RESUME_V1' and auth['scope']=='DETECTOR_ONLY_ORIGINAL_FULL_EXPERIMENT_AND_CN_EN_PAPERS' and auth['pause_verification']==PAUSE_REF and auth['prior_pause_retained']==RETAINED_PAUSE_REF and auth['JEV_GLM_MMTT_pair_investigation_remain_paused'] is True and auth['GPUs_allowed']==[0,1,2,3] and auth['new_scheduler_created'] is False,'Detector-only explicit resume scope')
 require(pause['schema']=='DETECTOR_V2_USER_PAUSED_VERIFIED_V1' and pause['state']=='PAUSED_BY_USER_NO_AUTOMATIC_RESUME' and pause['dispatch_allowed'] is False and pause['automatic_recovery_allowed'] is False and pause['resume_requires_new_explicit_user_authorization'] is True and pause['live_task_work_PIDs']==[] and _int(pause['in_flight_workers'],0),'verified prior user pause')
 require(retained['schema']=='DETECTOR_V2_USER_PAUSE_MARKER_V1' and retained['state']=='PAUSED_BY_USER_NO_AUTOMATIC_RESUME' and retained['no_new_dispatch'] is True and retained['no_automatic_resume'] is True and retained['requires_new_explicit_user_authorization'] is True and retained['evidence']==PAUSE_REF,'preserved original pause marker')
 partial=pause['retained_progress']['H_CPU_statistical_partial_outputs'][f'{bone}/seed{seed}'];interrupted=pause['CPU_interrupt_logs'][case['pause_tag']]
 require(partial['complete_exists'] is False and partial['no_files_removed'] is True and interrupted['retained_partial_only'] is True and interrupted['interrupted_by_user_pause'] is True and interrupted['scientifically_accepted'] is False,'exact pair unfinished at user pause; no prior terminal acceptance')
 parse=lambda x:datetime.datetime.fromisoformat(x.replace('Z','+00:00'))
 require(parse(pause['user_instruction_utc'])<=parse(pause['observed_utc'])<parse(auth['user_explicit_resume_UTC'])<=parse(auth['utc'])<=parse(L['utc']),'pause/authorization/launch chronology')
 pins={x['path']:x['sha256'] for x in (ref,AUTHORIZATION_REF,PAUSE_REF,RETAINED_PAUSE_REF)}
 for key in ('stdout','time_v'):
  r=interrupted[key];require(set(r)=={'path','sha256'} and re.fullmatch('[0-9a-f]{64}',r['sha256']),'bound original interrupt witness');_safe(r['path']);pins[r['path']]=r['sha256']
 return dict(launch_fact_route='explicit_bound_user_resume',evidence_sha256=pins,M_statistics_started_field_present=False,missing_field_not_synthesized=True,missing_field_alternative_evidence='Exact original c081 H-only entry and immutable bounded CLI, plus unchanged downstream terminal M_results_generated=False guard; no global M inactivity claim.')

def _launch_gate(L,bone,seed,refs):
 if L.get('status')==RESUME_STATUS or any(refs['primary_launch']['path']==x['path'] for x in RESUME_CASES.values()):validate_resume_launch(L,bone,seed,refs)
 else:_normal_launch_gate(L,bone,seed)

def _launch_ref_gate(q,bone,seed,ref):
 if (bone,seed) in RESUME_CASES and str(q)==RESUME_CASES[(bone,seed)]['path']:
  case=RESUME_CASES[(bone,seed)];require(ref==dict(path=case['path'],sha256=case['sha256']),'fixed resumed launch reference')
 else:_normal_path_gate(q,bone,seed)

def _route_info(spec):
 refs=spec['proofs'];L=_json(refs['primary_launch']['path'])
 if L.get('status')==RESUME_STATUS:return validate_resume_launch(L,spec['backbone'],spec['seed'],refs)
 return dict(launch_fact_route='original_normal_launch',evidence_sha256={},M_statistics_started_field_present=True,missing_field_not_synthesized=True)

def _check_consumer_sources():
 consumer_observed=sha(_safe(Path(__file__)));bb_observed=sha(_safe(BB_PATH))
 require(consumer_observed==ENTRY_SHA and bb_observed==BB_SHA,'consumer/orchestration source digest drift')
 return dict(consumer=dict(path=str(Path(__file__)),initial_sha256=ENTRY_SHA,observed_sha256=consumer_observed),original_bb=dict(path=str(BB_PATH),initial_sha256=BB_SHA,observed_sha256=bb_observed))

def _publication_source_guard(result):
 proof=_check_consumer_sources();require(result.get('source_attestation_initial')==proof,'source attestation drift before publication');result['source_attestation_before_publication']=proof

def _evidence_return(spec,I,pins):
 _check_consumer_sources()
 extra={str(BB_PATH):BB_SHA,str(Path(__file__)):ENTRY_SHA};extra.update(_route_info(spec)['evidence_sha256'])
 for p,h in extra.items():require(p not in pins or pins[p]==h,'conflicting consumer/resume provenance');pins[p]=h
 for p,h in extra.items():require(sha(_safe(p))==h,'consumer/resume provenance drift '+p)
 return I,pins

# Explicitly derive a NEW namespace from the hash-pinned source. Never import or
# mutate a live bb299e module, never modify its file, never synthesize launch facts.
_bb_bytes=BB_PATH.read_bytes();assert hashlib.sha256(_bb_bytes).hexdigest()==BB_SHA,'fixed bb299e source drift'
_tree=ast.parse(_bb_bytes);_functions={n.name:n for n in _tree.body if isinstance(n,ast.FunctionDef)}
ORIGINAL_AUDIT_AST=ast.dump(_functions['audit_owned'],include_attributes=False)
ADAPTATION_SITES={'validate_chain_launch_gate':0,'load_spec_launch_path_gate':0,'metadata_evidence_return':0,'main_publication_source_guard':0}
class _GuardAdapter(ast.NodeTransformer):
 def __init__(self,label,name,args,site):self.label=label;self.name=name;self.args=args;self.site=site;self.original=None
 def visit_Expr(self,node):
  value=node.value
  if isinstance(value,ast.Call) and isinstance(value.func,ast.Name) and value.func.id=='require' and len(value.args)==2 and isinstance(value.args[1],ast.Constant) and value.args[1].value==self.label:
   self.original=copy.deepcopy(node);ADAPTATION_SITES[self.site]+=1;return ast.Expr(value=ast.Call(func=ast.Name(id=self.name,ctx=ast.Load()),args=[ast.Name(id=x,ctx=ast.Load()) for x in self.args],keywords=[]))
  return self.generic_visit(node)
_launch_adapter=_GuardAdapter('actual original primary launch','_launch_gate',['L','bone','seed','refs'],'validate_chain_launch_gate');_launch_adapter.visit(_functions['validate_chain'])
_path_adapter=_GuardAdapter('canonical launch','_launch_ref_gate',['q','bone','seed','r'],'load_spec_launch_path_gate');_path_adapter.visit(_functions['load_spec'])
for node in _functions['_metadata'].body:
 if isinstance(node,ast.Return):node.value=ast.Call(func=ast.Name(id='_evidence_return',ctx=ast.Load()),args=[ast.Name(id=x,ctx=ast.Load()) for x in ('spec','I','pins')],keywords=[]);ADAPTATION_SITES['metadata_evidence_return']+=1
class _PublicationGuard(ast.NodeTransformer):
 def visit_With(self,node):
  node=self.generic_visit(node)
  if len(node.items)==1 and isinstance(node.items[0].context_expr,ast.Call):
   call=node.items[0].context_expr;fn=call.func
   if isinstance(fn,ast.Attribute) and isinstance(fn.value,ast.Name) and fn.value.id=='out' and fn.attr=='open':
    ADAPTATION_SITES['main_publication_source_guard']+=1
    return [ast.Expr(value=ast.Call(func=ast.Name(id='_publication_source_guard',ctx=ast.Load()),args=[ast.Name(id='result',ctx=ast.Load())],keywords=[])),node]
  return node
_PublicationGuard().visit(_functions['main'])
assert ADAPTATION_SITES=={'validate_chain_launch_gate':1,'load_spec_launch_path_gate':1,'metadata_evidence_return':1,'main_publication_source_guard':1},'unexpected frozen orchestration structure'
def _normal_guard(name,args,statement):
 return ast.FunctionDef(name=name,args=ast.arguments(posonlyargs=[],args=[ast.arg(arg=x) for x in args],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[statement],decorator_list=[])
_guards=ast.Module(body=[_normal_guard('_normal_launch_gate',['L','bone','seed'],_launch_adapter.original),_normal_guard('_normal_path_gate',['q','bone','seed'],_path_adapter.original)],type_ignores=[])
exec(compile(ast.fix_missing_locations(_guards),str(BB_PATH),'exec'),globals())
# Keep this module's explicit consumer documentation and defer CLI until wrappers.
_tree.body=[n for n in _tree.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) and not (isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')]
exec(compile(ast.fix_missing_locations(_tree),str(BB_PATH),'exec'),globals())
_derived_audit_owned=audit_owned;_derived_main=main

def audit_owned(spec,terminal_sha,slot):
 initial=_check_consumer_sources();result=_derived_audit_owned(spec,terminal_sha,slot);info=_route_info(spec);final=_check_consumer_sources()
 result.update(consumer_source_path=initial['consumer']['path'],consumer_source_sha256=ENTRY_SHA,original_bb299e_source_path=initial['original_bb']['path'],original_bb299e_source_sha256=BB_SHA,source_attestation_initial=initial,source_attestation_after_audit=final,consumer_implementation='Explicit four-site resume-aware derived consumer; unmodified6cd independent numeric kernel',orchestration_adaptation_sites=ADAPTATION_SITES,**info)
 return result

def main(argv=None):
 result=_derived_main(argv)
 if result.get('production_audit_started') is False:result.update(resume_aware_consumer=True,original_bb299e_source_sha256=BB_SHA,bound_resume_pairs=[list(x) for x in RESUME_CASES])
 return result
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False))
