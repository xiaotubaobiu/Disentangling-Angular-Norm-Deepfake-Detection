"""Summarize five independently accepted ORIGINAL H LAST saved results.

Engineering entry only: no models, pixels, cache contexts, score/bootstrapping reruns,
or original summarize_backbone execution. --execute requires an external request,
a separate fresh inclusive operational budget, and an existing shared CPU slot.
The new five_seed_saved_result_summary namespace never impersonates the original
five_seed_summary. The16MiB cap charges allocated/rounded files, directories and
all retained sibling staging transactions, rather than logical payload bytes.
The complete receipt is deterministic; runtime slot/budget facts are stdout only. Freeze this exact source before production request creation.
"""
from pathlib import Path
import argparse,ast,contextlib,csv,ctypes,datetime,hashlib,io,json,math,os,re,socket,stat,statistics,uuid

ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
HERE=ROOT/'engineering/source_best_M/audits'
ENGINEERING=ROOT/'engineering/source_best_M'
ORIGINAL=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
SEEDS=(7,42,123,2026,3407)
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
ROLES=('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST','common/B/SOURCE_TEST','common/B/TARGET_TEST','common/AB/SOURCE_TEST','common/AB/TARGET_TEST')
METRICS=('pooled_auc','macro_auc','fpr','tpr','pooled_ba','macro_tpr','macro_ba')
ANALYSIS_CODE_NAMES=('analysis_h.py','cache_h.py','stats_h.py','coverage_h.py','analysis_d.py','scoring_d.py','bootstrap_fast.py','cache_d.py')
KERNEL_SHA='6cd3bff403cc2f7c27a870d092023e4c8677f5d9386668abbc439829eadaa4e5'
LEGACY42_SHA='12ed094960dc7082215bf6261093d54962ccfb80d557c4ba3e0136baf467b32f'
MATRIX_SHA='bb299e512e355fa0a8c9b25f901b1788cee6a2b445a4eb1ddb67b30cb4c3a0e2'
RESUMED_SHA='0d33952bcb640ecfbe6d2cb4a0a2ac132089f485f43046d4096f0e2cdb83412b'
def ref(path,sha):return dict(path=str(path),sha256=sha)
CONSUMERS={KERNEL_SHA:ref(HERE/'h_pair_result_audit.py',KERNEL_SHA),LEGACY42_SHA:ref(HERE/'h_pair_result_audit_before_seed7_binding.py',LEGACY42_SHA),MATRIX_SHA:ref(ENGINEERING/'code/studies/pipeline_reliability/v2/tools/h_last_matrix_result_audit.py',MATRIX_SHA),RESUMED_SHA:ref(ENGINEERING/f'qualification_snapshots/H_last_resumed_result_audit_{RESUMED_SHA}/entry.py',RESUMED_SHA)}
DEFINITIONS={
 'plan':ref('/home/z/obsidian/paper/deepfake-detector/新实验设计/20261023/新实验计划_本机适配版.md','b018b01915a62c645fc3718134970cd49dc0651ecfd02792220f3a0086c6594d'),
 'analysis_h':ref(ORIGINAL/'reliability_v2/analysis_h.py','051226f11737615bb3546ae1a301d621fd662943dc1c849e18085e610c865383'),
 'stats_h':ref(ORIGINAL/'reliability_v2/stats_h.py','5627cccc714c2fc59a3dadfed3ddeba32d1488020cfd550ca70d0c9be947dd74'),
 'independent_D_reducer':ref(ROOT/'qualification/D_summary_independent_validation/audit.py','9ed097d6d38c9d0a84324f9788c63b63cda2a66b63502c1078a439d9d910a6a9'),
 'shared_slot':CONSUMERS[KERNEL_SHA],
 'qualified_manifest':ref(ROOT/'qualification/H_primary_cache_continuation/qualified_source_manifest.json','977431237b901d5ac88172f98bc2591d3cd7992c35be50546cda34ad2276a747')}
OUTPUTS=('identity.json','per_seed_metrics.csv','per_seed_comparisons.csv','five_seed_metrics.csv','five_seed_comparisons.csv','limitations.json')
CAP=16*1024**2
# Current real reference packet only; future refs must come from the externally
# SHA-bound budget, never from hashing live files to issue new expected hashes.
CAPACITY_REFERENCE_EXAMPLE={
 'plan':ref(ENGINEERING/'plans/M_two_canary_remaining_write_plan_after_H6_20261026_candidate.json','c0f5080192f20203dd63f0133aeeb21afd24807fbf7b1af9119945baa2abef07'),
 'acceptance':ref(ENGINEERING/'M_remaining_after_H6_capacity_acceptance_20261026.json','d78bc0f1a35d954564581110f835bb4781351343dcf9ea97e5436339a36d668a'),
 'independent_review':ref(ENGINEERING/'independent_M_capacity_after_H6_reassessment_20261026.json','a5fcdd16ef564117dbe8df1da72cfbdf3dad8185327885fb1afe2bd645531210')}
CAPACITY_BREAKDOWN=dict(remaining_H_primary9_full_caps=9*1024**3,primary_logs_manifests_figures_audits=16*1024**3,primary_temporary_peak=16*1024**3,primary_reserve=50*1024**3,M_temporary=6*1024**3,M_extra=5*1024**3,all_M45_feature_cap=18413684905,all_M30_statistic_caps=20132659200)
CAPACITY_TOTAL=148068010153
CAPACITY_ADMISSION='8d8e25e767fe5ee396c2d6773d4c65d002268e19bbdfa4151b0764e6f6ca475a'
FLAGS=('fresh_pixel_or_source_context_validation','fresh_feature_dot_product_recompute','weight_or_model_loading','M_source_best_used','M_production_authorized','GPU_or_target_inference_started','snapshot_promoted')

def require(value,message):
 if not value:raise AssertionError(message)
def sha(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
INITIAL_SHA=digest(__file__)

def safe_path(value):
 p=Path(value);require(p.is_absolute() and '..' not in p.parts and str(p)==str(p.absolute()),'absolute canonical path required')
 for part in (p,*p.parents):require(not part.is_symlink(),'symlink input/output rejected')
 require(p.resolve()==p,'real canonical path required')
 return p

def allowed_path(value):
 p=safe_path(value)
 external={x['path'] for x in DEFINITIONS.values()}|{x['path'] for x in CONSUMERS.values()}|{str(Path(__file__).resolve())}
 require(p.is_relative_to(ROOT) or p.is_relative_to(ORIGINAL) or str(p) in external,'input path outside authorized evidence namespaces')
 require(p.suffix.lower() not in ('.pt','.pth','.npy','.npz','.jpg','.jpeg','.png','.mp4'),'model/pixel/feature payload is outside saved summary scope')
 return p

def merge(pins,path,value):
 p=allowed_path(path);require(sha(value),'SHA256 required')
 require(str(p) not in pins or pins[str(p)]==value,'conflicting evidence pins')
 pins[str(p)]=value

def check_ref(value,pins,expected=None):
 require(type(value) is dict and set(value)=={'path','sha256'},'exact external path/SHA ref required')
 if expected is not None:require(value==expected,'canonical external proof path/SHA mismatch')
 merge(pins,value['path'],value['sha256']);return allowed_path(value['path'])

def recheck(pins):
 for p,h in sorted(pins.items()):require(digest(allowed_path(p))==h,'evidence/source SHA drift: '+p)

def load_json(path):
 p=allowed_path(path);require(p.stat().st_size<=32*1024**2,'bounded JSON input required')
 def pairs(items):
  d={}
  for k,v in items:require(k not in d,'duplicate JSON key');d[k]=v
  return d
 return json.loads(p.read_text(),object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))

def extract_functions(reference,names,namespace):
 source=Path(reference['path']).read_bytes();require(hashlib.sha256(source).hexdigest()==reference['sha256'],'qualified pure helper source changed')
 tree=ast.parse(source);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
 require({n.name for n in nodes}==set(names),'pure helper inventory mismatch')
 exec(compile(ast.Module(body=nodes,type_ignores=[]),reference['path'],'exec'),namespace)
 return namespace

_REDUCER=extract_functions(DEFINITIONS['independent_D_reducer'],('scalar','seed_stats'),dict(math=math,statistics=statistics,SEEDS=SEEDS))
_SLOT=extract_functions(DEFINITIONS['shared_slot'],('cpu_slot',),dict(contextlib=contextlib,contextmanager=contextlib.contextmanager,Path=Path,os=os,stat=stat,require=require))
cpu_slot=_SLOT['cpu_slot']

def reduce_seed_comparisons(rows,group_columns=None):
 rows=list(rows);require(bool(rows),'nonempty five-seed estimates required')
 required={'seed','comparison','threshold_kind','metric','estimate'}
 require(all(required<=set(r) for r in rows),'estimate columns missing')
 candidates=('backbone','domain','cohort','cohort_kind','aggregation','comparison','threshold_kind','metric')
 known=[k for k in candidates if any(k in r for r in rows)]
 groups=known if group_columns is None else list(group_columns)
 require(set(known)<=set(groups) and 'seed' not in groups and len(groups)==len(set(groups)),'cannot collapse cohort/domain/aggregation strata')
 grouped={}
 for r in rows:
  require(type(r['seed']) is int and r['seed'] in SEEDS,'exact typed registered seed required')
  require(all(k in r for k in groups),'stratum key missing')
  v=r['estimate'];require(v is None or (not isinstance(v,bool) and isinstance(v,(int,float))),'numeric estimate or NA required')
  grouped.setdefault(tuple(r[k] for k in groups),[]).append((r['seed'],v))
 out=[]
 for key,pairs in sorted(grouped.items()):
  stats=_REDUCER['seed_stats'](pairs);stats['expected_seeds']=SEEDS
  out.append(dict(zip(groups,key),**stats))
 return out

def pair_inventory():
 out={'coverage.json','identity.json','independent_saved_score_recompute.json','limitations.json'}
 for agg in ('mean_logit','mean_probability'):
  out.update(f'{agg}/source_cal_{h}.csv' for h in ('B','AB'))
  for dom in ('SOURCE_TEST','TARGET_TEST'):
   for co,cells in [('all_B',('H00','H10')),('common',('H00','H01','H10','H11'))]:
    pre=f'{agg}/{dom}/{co}/';out.update(pre+n for n in ('metrics.csv','comparisons.csv','conditional_bootstrap.json','independent_kernel_checks.json','thresholds.json'));out.update(pre+f'{c}_scores.csv' for c in cells)
 return out

def table(path):
 require(Path(path).stat().st_size<=CAP,'bounded result table required')
 with Path(path).open(newline='') as f:
  reader=csv.DictReader(f);columns=reader.fieldnames;require(columns and len(set(columns))==len(columns),'CSV header required');rows=list(reader)
 require(rows and all(None not in r and all(v is not None for v in r.values()) for r in rows),'CSV row inventory malformed')
 return columns,rows

def number(value):
 if value is None or value=='':return None
 require(not isinstance(value,bool),'boolean estimate rejected');v=float(value)
 return v if math.isfinite(v) else None

def same_number(left,right):
 l=number(left);r=number(right)
 return l is None and r is None or l is not None and r is not None and math.isclose(l,r,rel_tol=1e-12,abs_tol=1e-12)

def validate_audit(a,pair,bone,seed,pins):
 require(a['schema']=='INDEPENDENT_ORIGINAL_H_LAST_PAIR_RESULT_AUDIT_V1' and a['status']=='SAVED_RESULT_NUMERICS_AND_BOUND_PRIOR_ROLE_ADMISSION_PASS','independent numeric PASS required')
 require(a['backbone']==bone and type(a['seed']) is int and a['seed']==seed and a['source_checkpoint']=='original_LAST_final.pt','independent audit original LAST identity mismatch')
 require(a['artifact_count']==72 and type(a['artifact_count']) is int and type(a['cpu_slot']) is int and a['cpu_slot'] in (0,1) and a['lifetime_shared_cpu_slot_held'] is True,'accepted lifetime-slot audit required')
 require(all(a[k] is False for k in FLAGS),'explicit accepted audit scope flags required')
 require(a['terminal_sha256']==pair['complete']['sha256'] and a['identity_sha256']==pair['identity']['sha256'],'audit terminal/identity binding mismatch')
 h=a['audit_code_sha256'];require(h in CONSUMERS and pair['audit_source']==CONSUMERS[h],'accepted consumer source binding mismatch');check_ref(pair['audit_source'],pins,CONSUMERS[h])
 missing=[]
 if h==LEGACY42_SHA:
  require((bone,seed)==('clip_vit_b16',42),'legacy42 only exact original pair')
  missing=['evidence_route','independent_numeric_kernel_sha256','all_five_seed_conclusions_available','binding_spec_sha256']
  require(not any(k in a for k in missing),'historical42 schema mismatch')
 elif h==KERNEL_SHA:
  require((bone,seed)==('clip_vit_b16',7) and a['evidence_route']=='ORIGINAL_SERIAL14_ACCEPTED','legacy7 only exact original pair')
  missing=['independent_numeric_kernel_sha256','all_five_seed_conclusions_available'];require(not any(k in a for k in missing),'historical7 schema mismatch')
 else:
  require(a['independent_numeric_kernel_sha256']==KERNEL_SHA and a['all_five_seed_conclusions_available'] is False and a['evidence_route']=='ACCEPTED_PARALLEL14_METADATA_REBOUND_ORIGINAL42FRESH_LEDGER_PRODUCER','accepted matrix audit schema mismatch')
  if h==RESUMED_SHA:
   require(a['consumer_source_path']==CONSUMERS[h]['path'] and a['consumer_source_sha256']==h and a['original_bb299e_source_path']==CONSUMERS[MATRIX_SHA]['path'] and a['original_bb299e_source_sha256']==MATRIX_SHA,'resumed actual source provenance mismatch')
   for stage in ('initial','after_audit','before_publication'):
    proof=a['source_attestation_'+stage]
    for key,expected in [('consumer',CONSUMERS[h]),('original_bb',CONSUMERS[MATRIX_SHA])]:
     require(proof[key]==dict(path=expected['path'],initial_sha256=expected['sha256'],observed_sha256=expected['sha256']),'resumed initial/final source attestation mismatch')
   require(a['launch_fact_route']=='explicit_bound_user_resume' and a['M_statistics_started_field_present'] is False and a['missing_field_not_synthesized'] is True,'explicit actual resume facts required')
   check_ref(CONSUMERS[MATRIX_SHA],pins)
 require(type(a['current_pins_sha256']) is dict and a['current_pins_sha256'],'current accepted source/dependency pins required')
 historical={}
 for p,h in a['current_pins_sha256'].items():
  if Path(p).suffix.lower() in ('.pt','.pth','.npy','.npz','.jpg','.jpeg','.png','.mp4'):
   canonical=safe_path(p);require(canonical.is_relative_to(ROOT) or canonical.is_relative_to(ORIGINAL),'historical source pin outside original evidence namespaces');require(sha(h),'historical SHA required');key=str(canonical)
   require(key not in historical or historical[key]==h,'conflicting historical source identity within accepted audit');historical[key]=h
  else:merge(pins,p,h)
 return dict(source_sha256=a['audit_code_sha256'],explicit_historical_absent_fields=missing,historical_nonconsumed_source_sha256=historical)

def read_pair(pair,bone,pins,manifest):
 require(set(pair)=={'seed','identity','complete','independent_numeric_audit','audit_source'},'exact pair request inventory')
 seed=pair['seed'];require(type(seed) is int and seed in SEEDS,'typed registered pair seed required')
 folder=ROOT/f'analysis/H/{bone}/seed{seed}'
 ip=check_ref(pair['identity'],pins);cp=check_ref(pair['complete'],pins);ap=check_ref(pair['independent_numeric_audit'],pins)
 require(ip==folder/'identity.json' and cp==folder/'complete.json','canonical pair files required')
 label='clip' if bone=='clip_vit_b16' else bone
 require(ap==HERE/f'H_LAST_{label}_seed{seed}_independent_audit.json','canonical independent audit path required')
 I=load_json(ip);C=load_json(cp);A=load_json(ap);profile=validate_audit(A,pair,bone,seed,pins)
 require(set(I)=={'schema','backbone','seed','effective_precision','run_ids','inputs_sha256','cache_completion_sha256','code_sha256','bootstrap_repeats','resampling_seed','source_threshold','primary','test_cohorts'},'exact original pair identity fields required')
 require(I['schema']=='H_PAIR_ANALYSIS_IDENTITY_V1' and I['backbone']==bone and type(I['seed']) is int and I['seed']==seed and I['effective_precision']=='AMP','original formal H identity required')
 require(I['run_ids']=={h:f'H_formal_{bone}_{h}_seed{seed}' for h in ('B','AB')},'original LAST run IDs required')
 require(I['bootstrap_repeats']==2000 and type(I['bootstrap_repeats']) is int and I['resampling_seed']==20261023 and type(I['resampling_seed']) is int,'frozen sampling definition mismatch')
 require(I['source_threshold']=='each_model_allB_SOURCE_CAL_empiricalFPR5' and I['primary']=='DFD_allB_Raw_mean_logit_H10_minus_H00' and I['test_cohorts']=='allB_and_common_reported_separately','frozen comparison definition mismatch')
 expected_inputs={str(ROOT/p) for p in ('configs/H_execution.json','H_run_registry.json',f'qualification/H_target_export_{bone}.json',f'reports/H_{bone}_AMP_paired_ledger.json','data/H/freeze.json')}
 require(set(I['inputs_sha256'])==expected_inputs,'exact five frozen input paths required')
 for p,h in I['inputs_sha256'].items():merge(pins,p,h)
 registry=load_json(ROOT/'H_run_registry.json');require(registry['target_export_allowed'] is True and registry['status']=='H_FORMAL_COMPLETE_TARGET_CACHE_PENDING' and set(registry['effective_runs'])==set(BACKBONES),'final canonical all3 precision registry required')
 for b,r in registry['effective_runs'].items():require(r['precision']=='AMP' and len(r['run_ids'])==10 and set(r['run_ids'])=={f'H_formal_{b}_{h}_seed{s}' for s in SEEDS for h in ('B','AB')},'all3 complete effective model family required')
 require(set(I['cache_completion_sha256'])=={h+'/'+r for h in ('B','AB') for r in ROLES} and all(sha(h) for h in I['cache_completion_sha256'].values()),'original14 role identity bindings required')
 require(I['code_sha256']=={n:manifest['source_sha256']['reliability_v2/'+n] for n in ANALYSIS_CODE_NAMES},'accepted producer source/HEAD mismatch')
 for n,h in I['code_sha256'].items():merge(pins,ORIGINAL/'reliability_v2'/n,h)
 require(C['schema']=='H_PAIR_ANALYSIS_COMPLETE_V1' and C['status'] in ('ANALYSIS_COMPLETE','ANALYSIS_COMPLETE_WITH_NUMERIC_NA') and C['identity']==I and C['backbone']==bone and type(C['seed']) is int and C['seed']==seed and C['effective_precision']=='AMP' and C['independent_recompute']=='PASSED_FOR_ALL_REPORTABLE_CELLS','exact completed saved pair required')
 require(set(C['artifact_sha256'])==pair_inventory(),'complete exact72 artifact inventory required')
 require({str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}==pair_inventory()|{'complete.json'},'pair directory inventory differs from completed72')
 for n,h in C['artifact_sha256'].items():merge(pins,folder/n,h)
 require(C['artifact_sha256']['identity.json']==pair['identity']['sha256'],'identity artifact SHA differs')
 require(type(A['cohorts']) is list and len(A['cohorts'])==8,'all8 independent cohort audits required')
 cohorts={};metrics=[];comparisons=[]
 for a in A['cohorts']:
  key=(a['aggregation'],a['domain'],a['cohort']);require(key not in cohorts,'duplicate audited cohort');cohorts[key]=a
 expected={(g,d,c) for g in ('mean_logit','mean_probability') for d in ('SOURCE_TEST','TARGET_TEST') for c in ('all_B','common')};require(set(cohorts)==expected,'exact8 cohort routing required')
 metric_columns=None
 for agg,dom,co in sorted(expected):
  a=cohorts[(agg,dom,co)];require(type(a['draws_checked']) is int and ((a['status'] in ('ok','no_valid_draws') and a['draws_checked']==2000) or (a['status']=='insufficient_source_groups' and a['draws_checked']==0 and a['intervals_checked']==0)),'applicable2000 draws or explicit insufficient-group NA audit required')
  point=a['independent_point_estimates'];base=folder/agg/dom/co;mc,mrows=table(base/'metrics.csv');cc,crows=table(base/'comparisons.csv');require(set(cc)=={'comparison','threshold_kind','metric','estimate'},'original comparison table columns required')
  require({'cell','model_history','threshold_kind',*METRICS}<=set(mc),'original metric columns required')
  require(metric_columns is None or metric_columns==mc,'consistent lossless metric columns required');metric_columns=mc
  cells=('H00','H10') if co=='all_B' else ('H00','H01','H10','H11');names=('H10-H00',) if co=='all_B' else ('H10-H00','H01-H00','H11-H01','H11-H10','(H11-H10)-(H01-H00)')
  require(len(mrows)==2*len(cells) and {(r['cell'],r['threshold_kind']) for r in mrows}=={(c,k) for c in cells for k in ('default','source_real')},'complete cell/threshold table required')
  require(len(crows)==len(names)*2*len(METRICS) and {(r['comparison'],r['threshold_kind'],r['metric']) for r in crows}=={(c,k,v) for c in names for k in ('default','source_real') for v in METRICS},'complete signed comparison table required')
  meta=dict(seed=seed,backbone=bone,domain=dom,cohort=co,aggregation=agg)
  require(not(set(mc)&set(meta)) and not(set(cc)&set(meta)),'producer columns must not overwrite routing metadata')
  for r in mrows:
   for v in METRICS:require(same_number(r[v],point[f"{r['threshold_kind']}:cell:{r['cell']}:{v}"]),'metric differs from accepted independent numerical estimate')
   metrics.append(dict(r,**meta))
  for r in crows:
   require(same_number(r['estimate'],point[f"{r['threshold_kind']}:comparison:{r['comparison']}:{r['metric']}"]),'comparison differs from accepted independent numerical estimate')
   comparisons.append(dict(r,**meta))
 return metrics,comparisons,dict(seed=seed,identity=pair['identity'],complete=pair['complete'],independent_numeric_audit=pair['independent_numeric_audit'],**profile)

def json_bytes(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def csv_bytes(rows):
 require(rows,'nonempty output table');columns=list(rows[0]);require(all(set(r)==set(columns) for r in rows),'lossless output column inventory mismatch')
 f=io.StringIO(newline='');writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows);return f.getvalue().encode()

def source_guard(expected):
 require(expected==INITIAL_SHA and digest(__file__)==expected,'initial executing summary source SHA drift')

def load_request(path,expected):
 pins={};require(sha(expected),'external request SHA required');check_ref(ref(path,expected),pins)
 request=load_json(path);bone=request['backbone'];require(bone in BACKBONES,'registered backbone required')
 require(Path(path)==HERE/f'H_SAVED_FIVE_SEED_{bone}_request.json','canonical summary request path required')
 require(request['schema']=='H_SAVED_FIVE_SEED_SUMMARY_REQUEST_V1' and request['root']==str(ROOT) and request['source_checkpoint']=='original_LAST_final.pt' and request['seeds']==list(SEEDS) and all(type(s) is int for s in request['seeds']),'exact original five seeds request required')
 require(request['summary_source']==ref(str(Path(__file__).resolve()),INITIAL_SHA),'actual executing summary source ref required');check_ref(request['summary_source'],pins)
 require(request['definitions']==DEFINITIONS,'exact frozen statistical/source definitions required')
 for r in DEFINITIONS.values():check_ref(r,pins)
 require(type(request['pairs']) is list and [p['seed'] for p in request['pairs']]==list(SEEDS),'exact ordered five pair inputs required')
 recheck(pins);return request,pins

def live_free_bytes():
 s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize

def check_capacity_chain(b,pins):
 anchors=b['capacity_evidence'];require(type(anchors) is dict and set(anchors)=={'plan','acceptance','independent_review'},'exact externally SHA-bound accepted capacity chain refs required')
 base_namespace=ROOT/'engineering/source_best_M'
 for key,r in anchors.items():
  require(type(r) is dict and set(r)=={'path','sha256'} and sha(r['sha256']),'external capacity ref expected SHA required')
  path=safe_path(r['path']);require(path.parent==(base_namespace/'plans' if key=='plan' else base_namespace) and path.suffix=='.json','canonical Detector capacity proof namespace required')
  require(('write_plan' in path.name and path.name.startswith('M_')) if key=='plan' else (path.name.startswith('independent_M_') if key=='independent_review' else path.name.startswith('M_') and 'acceptance' in path.name),'capacity proof filename family mismatch')
 require(type(b['evidence']) is list and len(b['evidence'])==3 and {tuple(sorted(r.items())) for r in b['evidence']}=={tuple(sorted(r.items())) for r in anchors.values()},'exact plan/independent-review/root-acceptance evidence required')
 for r in anchors.values():check_ref(r,pins)
 recheck(pins)
 plan=load_json(anchors['plan']['path']);acceptance=load_json(anchors['acceptance']['path']);review=load_json(anchors['independent_review']['path'])
 require(plan['schema']=='M_FIXED_REMAINING_WRITE_PLAN_V1' and sha(plan['input_admission_sha256']),'same admitted complete write plan required')
 runs={f'D_formal_{bone}_seed{seed}' for bone in BACKBONES for seed in SEEDS}|{f'H_formal_{bone}_{h}_seed{seed}' for bone in BACKBONES for seed in SEEDS for h in ('B','AB')}
 require(type(plan['feature_runs']) is list and len(plan['feature_runs'])==45 and set(plan['feature_runs'])==runs,'full45 future feature runs retained')
 jobs={f'{d}_{bone}_seed{seed}':(1024**3 if d=='D' else 1024**3//4) for d in ('D','H') for bone in BACKBONES for seed in SEEDS}
 require(set(plan['jobs'])==set(jobs),'all30 future statistic job caps retained')
 for key,cap in jobs.items():
  job=plan['jobs'][key];require(set(job)=={'cap_bytes','output'} and type(job['cap_bytes']) is int and job['cap_bytes']==cap,'fixed full job cap required without completion credit')
  if job['output'] is not None:require(type(job['output']) is str and safe_path(job['output']).is_relative_to(ROOT/'analysis/source_best_M') and Path(job['output']).name==key,'canonical existing job namespace required')
 require(plan['feature_cap_bytes']==CAPACITY_BREAKDOWN['all_M45_feature_cap'] and type(plan['feature_cap_bytes']) is int and plan['global_M_temporary_bytes']==6*1024**3 and type(plan['global_M_temporary_bytes']) is int and plan['runtime_floor_bytes']==50*1024**3 and type(plan['runtime_floor_bytes']) is int,'full feature and simultaneous peaks/reserve retained')
 parts=review['budget_breakdown_bytes'];hkeys=[k for k in parts if k.startswith('remaining_H_primary') and k.endswith('_full_caps')]
 require(len(hkeys)==1,'one complete remaining primary family cap required')
 match=re.fullmatch(r'remaining_H_primary(\d+)_full_caps',hkeys[0]);require(match is not None and 0<=int(match[1])<=15,'bounded remaining primary pair count required');primary_count=int(match[1])
 fixed={k:v for k,v in CAPACITY_BREAKDOWN.items() if not k.startswith('remaining_H_primary')}
 require(set(parts)==set(fixed)|{hkeys[0]} and all(type(v) is int for v in parts.values()) and all(parts[k]==v for k,v in fixed.items()) and parts[hkeys[0]]==primary_count*1024**3,'full retained reserve/peaks/misc/features/jobs decomposition required')
 require(type(review['remaining_primary_future_pair_count']) is int and review['remaining_primary_future_pair_count']==primary_count and type(review['H_independently_accepted_pair_count']) is int and review['H_independently_accepted_pair_count']+primary_count==15,'reviewed primary closure and remaining scope must total15')
 total=sum(parts.values());base=sum(v for k,v in parts.items() if k not in ('all_M45_feature_cap','all_M30_statistic_caps'))
 require(type(plan['base_future_and_reserve_bytes']) is int and plan['base_future_and_reserve_bytes']==base,'full base future write decomposition required')
 require(review['schema']=='INDEPENDENT_M_CAPACITY_AFTER_H6_REASSESSMENT_V1' and review['status']=='PASS_CLOSED_PRIMARY_FUTURE_WRITE_REASSESSMENT_ONLY' and review['bindings']['new_plan']==anchors['plan'],'independently accepted own plan required')
 require(review['all45_feature_runs_preserved'] is True and review['all30_job_caps_preserved'] is True and review['required_total_bytes']==total and type(review['required_total_bytes']) is int,'independent full remaining bound required')
 for key in ('partial_M_feature_credit_bytes','M_statistic_credit_bytes','physical_SSD_release_bytes'):require(type(review[key]) is int and review[key]==0,'no partial/compression/statistic or physical release credit')
 require(b['decomposition_bytes']==parts and all(type(v) is int for v in b['decomposition_bytes'].values()),'exact independently reviewed full decomposition required')
 require(acceptance['schema']=='M_REMAINING_FUTURE_WRITE_AFTER_H6_CAPACITY_ACCEPTANCE_V1' and acceptance['status']=='PASS_REVIEWED_FIXED_FUTURE_WRITE_PLAN_NOT_PRODUCTION_PERMISSION' and acceptance['artifact_root']==str(ROOT) and acceptance['input_admission_sha256']==plan['input_admission_sha256'] and acceptance['write_plan']==anchors['plan'] and acceptance['independent_review']==anchors['independent_review'],'root acceptance must bind own reviewed plan/admission')
 require(acceptance['actual_H6_acceptance']==review['bindings']['new_H_acceptance'],'accepted completed primary credit proof must agree')
 closure=load_json(check_ref(acceptance['actual_H6_acceptance'],pins))
 require(closure['schema']=='H_RESUMED_ORIGINAL_LAST_TWO_PAIR_INDEPENDENT_ACCEPTANCE_V1' and closure['status']=='PASS_ACTUAL_TERMINAL_AND_INDEPENDENT_NUMERIC_BOOTSTRAP_AUDIT' and type(closure['H_total_independently_accepted_pairs']) is int and closure['H_total_independently_accepted_pairs']==review['H_independently_accepted_pair_count'],'root primary closure acceptance count/status must match reviewed remaining cap')
 require(type(closure['pairs']) is list and closure['pairs'] and len(closure['pairs'])<=closure['H_total_independently_accepted_pairs'],'explicit own completed primary evidence required')
 closed=set()
 for pair in closure['pairs']:
  bone=pair['backbone'];seed=pair['seed'];require(bone in BACKBONES and type(seed) is int and seed in SEEDS and (bone,seed) not in closed,'unique formal primary closure identity required');closed.add((bone,seed))
  require(type(pair['artifact_count']) is int and pair['artifact_count']==72 and type(pair['independently_recomputed_cohorts']) is int and pair['independently_recomputed_cohorts']==8 and type(pair['applicable_draws_recomputed']) is int and 0<=pair['applicable_draws_recomputed']<=16000 and pair['native_source_checkpoints']=='original_LAST_final.pt' and pair['source_best_or_M_used'] is False,'original LAST scientifically accepted primary closure metadata required')
  label='clip' if bone=='clip_vit_b16' else bone
  require(pair['complete']['path']==str(ROOT/f'analysis/H/{bone}/seed{seed}/complete.json') and pair['independent_audit']['path']==str(ROOT/f'engineering/source_best_M/audits/H_LAST_{label}_seed{seed}_independent_audit.json'),'closure pair own canonical result/audit proof required')
  for key in ('independent_audit','complete','producer_execution','producer_launch','producer_time_v','audit_launch','audit_true_exit'):check_ref(pair[key],pins)
 for mapping in (review['bindings'],review.get('fresh_reference_bindings',{})):
  for r in mapping.values():check_ref(r,pins)
 for path,h in review.get('current_python_code_hashes_verified',{}).items():merge(pins,path,h)
 require(acceptance['summary_small_outputs_included_in_existing_primary16GiB_logs_figures_audits_and_M5GiB_extra'] is True,'summary allocation must already be covered by reviewed misc bound')
 checks={'full_required_remaining_with_peak_and_reserve_bytes':total,'primary_temporary_peak_retained_bytes':16*1024**3,'M_temporary_peak_retained_bytes':6*1024**3,'runtime_floor_bytes':50*1024**3,'full_M_feature_bound_retained_bytes':CAPACITY_BREAKDOWN['all_M45_feature_cap'],'all_M30_job_caps_retained_bytes':CAPACITY_BREAKDOWN['all_M30_statistic_caps'],'partial_M_feature_credit_bytes':0,'physical_SSD_released_bytes':0}
 require(all(type(acceptance[k]) is int and acceptance[k]==v for k,v in checks.items()),'acceptance full future/peak/reserve decomposition mismatch')
 require(total==base+plan['feature_cap_bytes']+sum(j['cap_bytes'] for j in plan['jobs'].values()),'complete inclusive plan arithmetic mismatch')
 require(b['full_remaining_with_peaks_reserve_bytes']==total and b['reserve_already_included_bytes']==50*1024**3 and b['primary_temporary_already_included_bytes']==16*1024**3 and b['M_temporary_already_included_bytes']==6*1024**3 and b['all_M45_feature_cap_bytes']==CAPACITY_BREAKDOWN['all_M45_feature_cap'] and b['all_M30_statistic_caps_bytes']==CAPACITY_BREAKDOWN['all_M30_statistic_caps'],'operational bound must retain entire accepted plan without duplicated peaks or new credits')


def check_budget(path,expected):
 pins={};check_ref(ref(path,expected),pins);recheck(pins);b=load_json(path)
 require(b['schema']=='H_SAVED_SUMMARY_OPERATIONAL_BUDGET_V1' and b['root']==str(ROOT) and b['status']=='VERIFIED_FULL_REMAINING_PRIMARY_WITH_PEAKS_RESERVE','verified inclusive operational budget required')
 stamp=datetime.datetime.fromisoformat(b['utc']);require(stamp.tzinfo is not None and 0<=(datetime.datetime.now(datetime.timezone.utc)-stamp).total_seconds()<=900,'fresh15minute full remaining budget required')
 for k in ('full_remaining_with_peaks_reserve_bytes','reserve_already_included_bytes','primary_temporary_already_included_bytes','M_temporary_already_included_bytes','all_M45_feature_cap_bytes','all_M30_statistic_caps_bytes','summary_output_cap_covered_bytes','unverified_compression_credit_bytes','partial_feature_credit_bytes'):require(type(b[k]) is int and b[k]>=0,'typed nonnegative budget bytes required')
 require(b['summary_output_cap_covered_bytes']==CAP and b['summary_cap_already_included'] is True and b['unverified_compression_credit_bytes']==0 and b['partial_feature_credit_bytes']==0,'reviewed16MiB allocated summary cap already included; no added credits')
 check_capacity_chain(b,pins)
 recheck(pins);require(live_free_bytes()>=b['full_remaining_with_peaks_reserve_bytes'],'fresh free space below already inclusive full remaining bound')
 return pins

def build(request,pins,request_sha):
 manifest=load_json(DEFINITIONS['qualified_manifest']['path']);require(manifest['qualified_commit']=='5ef6d06bf7aeda02a5a11113e4c35a2c9909f215' and len(manifest['source_sha256'])==20,'accepted original20 source commit required')
 metrics=[];comparisons=[];profiles=[];historical={}
 for pair in request['pairs']:
  a,b,p=read_pair(pair,request['backbone'],pins,manifest)
  for path,h in p['historical_nonconsumed_source_sha256'].items():
   require(path not in historical or historical[path]==h,'conflicting historical source identity across five seeds');historical[path]=h
   require(path not in pins or pins[path]==h,'historical/current source identity conflict')
  metrics.extend(a);comparisons.extend(b);profiles.append(p)
 mlong=[]
 for r in metrics:
  for metric in METRICS:mlong.append({**{k:r[k] for k in ('seed','backbone','domain','cohort','aggregation','threshold_kind')},'comparison':r['cell'],'metric':metric,'estimate':number(r[metric])})
 clong=[dict(r,estimate=number(r['estimate'])) for r in comparisons]
 sm=reduce_seed_comparisons(mlong);sc=reduce_seed_comparisons(clong)
 sm=[{('cell' if k=='comparison' else k):v for k,v in r.items()} for r in sm]
 identity=dict(schema='H_SAVED_FIVE_SEED_SUMMARY_IDENTITY_V1',route='SHA_BOUND_ACCEPTED_SAVED_RESULT_REDUCTION',backbone=request['backbone'],source_checkpoint='original_LAST_final.pt',seeds=list(SEEDS),request_sha256=request_sha,summary_source=request['summary_source'],definitions=DEFINITIONS,pairs=profiles,fresh_checked_input_sha256=dict(sorted(pins.items())),historical_nonconsumed_source_sha256=dict(sorted(historical.items())),historical_pin_policy='Prior non-consumed original model/pixel/feature source pins retained from accepted evidence; not freshly byte-rehashed here',statistics='arithmetic_mean_and_sample_SD_ddof1_only_if_all5_finite;sign_tolerance1e-10')
 unavailable=[{k:r[k] for k in ('backbone','domain','cohort','aggregation','threshold_kind','metric')}|dict(series=r.get('cell',r.get('comparison')),valid_count=r['n_valid_seeds'],NA_count=5-r['n_valid_seeds']) for r in sm+sc if r['n_valid_seeds']<5]
 limitations=dict(route=identity['route'],original_summarize_backbone_executed=False,source_pixels_or_context_revalidated=False,fresh_feature_dot_reconstructed=False,models_or_GPU_loaded=False,saved_scores_recomputed_here=False,all_fresh_checked_metadata_and_artifact_pins_rehashed=True,historical_nonconsumed_source_bytes_freshly_verified=False,prior_independent_saved_score_numerics_reused=True,M_source_best_used=False,pooled_15_model_IID_claim=False,bootstrap_draws_are_training_seeds=False,new_p_values_generated=False,cross_cohort_subtraction_performed=False,NA_strata=unavailable,all_strata_have_explicit_n_valid_and_sign_counts=True,scope='Descriptive across exactly five independent training seeds; within-model saved-score uncertainty remains in original pair outputs')
 return {'identity.json':json_bytes(identity),'per_seed_metrics.csv':csv_bytes(metrics),'per_seed_comparisons.csv':csv_bytes(comparisons),'five_seed_metrics.csv':csv_bytes(sm),'five_seed_comparisons.csv':csv_bytes(sc),'limitations.json':json_bytes(limitations)}

def publication_guard(pins,source_sha,budget_path,budget_sha):
 source_guard(source_sha);recheck(pins);check_budget(budget_path,budget_sha)

def complete_bytes(payloads,request_sha,source_sha):
 # Fully deterministic statistical receipt. Runtime slot/budget are stdout-only
 # operational facts, never mutable fields in the append-only statistical receipt.
 return json_bytes(dict(schema='H_SAVED_FIVE_SEED_SUMMARY_COMPLETE_V1',status='SAVED_RESULT_SUMMARY_COMPLETE',request_sha256=request_sha,summary_source_sha256=source_sha,artifact_sha256={n:hashlib.sha256(b).hexdigest() for n,b in payloads.items()},route='DETERMINISTIC_SAVED_RESULT_RECEIPT_RUNTIME_FACTS_IN_OPERATION_LOG'))


def fragment_bytes(path):
 value=os.statvfs(path).f_frsize;require(type(value) is int and value>0,'filesystem allocation fragment required');return value

def rounded(value,block):return ((value+block-1)//block)*block


def allocated_tree(path,block):
 path=safe_path(path);s=path.lstat();require(stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode),'summary namespace regular files/directories only')
 if stat.S_ISREG(s.st_mode):require(s.st_nlink==1,'single-link summary payload required')
 charge=max(rounded(s.st_size,block),s.st_blocks*512,block if stat.S_ISDIR(s.st_mode) else 0)
 if stat.S_ISDIR(s.st_mode):charge+=sum(allocated_tree(child,block) for child in path.iterdir())
 return charge


def namespace_allocation(out):
 # Charge the whole output, every retained sibling transaction, and its parent
 # directory. Unknown remnants are preserved and charged, never removed/reused.
 parent=safe_path(out.parent);block=fragment_bytes(parent);s=parent.stat();total=max(block,rounded(s.st_size,block),s.st_blocks*512)
 for child in parent.iterdir():
  if child.name==out.name or child.name.startswith('.'+out.name+'.stage_'):total+=allocated_tree(child,block)
 return total


def check_allocation(out):
 value=namespace_allocation(out);require(value<=CAP,'summary allocated namespace/staging peak exceeds16MiB cap');return value


def verify_existing(out,payloads,request_sha,source_sha):
 check_allocation(out);require({p.name for p in out.iterdir()}==set(OUTPUTS)|{'complete.json'},'existing summary must have exact complete inventory')
 expected=complete_bytes(payloads,request_sha,source_sha)
 require(safe_path(out/'complete.json').read_bytes()==expected,'existing deterministic complete receipt bytes changed')
 for n,b in payloads.items():require(digest(safe_path(out/n))==hashlib.sha256(b).hexdigest(),'existing summary content/input/reducer mismatch')
 check_allocation(out)


def atomic_commit(stage,out):
 # Same-filesystem atomic whole-directory publish with no replacement, including
 # an existing empty destination. No post-commit cleanup can invalidate success.
 libc=ctypes.CDLL(None,use_errno=True);rename=libc.renameat2;rename.argtypes=(ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint);rename.restype=ctypes.c_int
 if rename(-100,os.fsencode(stage),-100,os.fsencode(out),1):
  error=ctypes.get_errno();raise OSError(error,os.strerror(error),str(out))


@contextlib.contextmanager
def writer_reservation(out):
 # Atomic Linux kernel reservation, per canonical output and across processes.
 # No filesystem lock/slot, listener, traffic or daemon. It cannot leave stale
 # disk locks or require postcommit unlink; process exit releases the socket.
 canonical=safe_path(out);key=b'\0H_saved_five_seed_writer_v1_'+hashlib.sha256(os.fsencode(str(canonical))).hexdigest().encode()
 lease=socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM);state=dict(release_confirmed=None)
 try:
  try:lease.bind(key)
  except OSError as error:
   if error.errno==98:raise RuntimeError('summary writer busy; no wait or retry') from error
   raise
  yield state
 finally:
  try:lease.close();state['release_confirmed']=True
  except OSError:state['release_confirmed']=False


def publish(out,payloads,request_sha,pins,source_sha,budget_path,budget_sha,slot):
 # Entire prediction/stage/retention/commit/reentry sequence has one owner.
 with writer_reservation(out) as reservation:
  result=_publish_reserved(out,payloads,request_sha,pins,source_sha,budget_path,budget_sha,slot)
 return dict(result,writer_reservation_release_confirmed=reservation['release_confirmed'])


def _publish_reserved(out,payloads,request_sha,pins,source_sha,budget_path,budget_sha,slot):
 require(set(payloads)==set(OUTPUTS),'exact summary payload inventory required')
 all_payloads={**payloads,'complete.json':complete_bytes(payloads,request_sha,source_sha)}
 publication_guard(pins,source_sha,budget_path,budget_sha)
 if out.exists():
  verify_existing(out,payloads,request_sha,source_sha);publication_guard(pins,source_sha,budget_path,budget_sha);check_allocation(out);return dict(reused=True,directory_fsync_confirmed=None)
 safe_path(out);safe_path(out.parent);block=fragment_bytes(out.parent)
 # Include file rounding, a new stage/output directory and conservative parent
 # directory growth; existing stage remnants are included by namespace_allocation.
 require(namespace_allocation(out)+2*block+sum(rounded(len(b),block) for b in all_payloads.values())<=CAP,'summary allocated predicted staging peak exceeds16MiB cap')
 stage=out.parent/('.'+out.name+'.stage_'+uuid.uuid4().hex);stage.mkdir()
 for n,b in all_payloads.items():
  with (stage/n).open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
 fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:os.fsync(fd)
 finally:os.close(fd)
 check_allocation(out);publication_guard(pins,source_sha,budget_path,budget_sha);check_allocation(out)
 atomic_commit(stage,out)
 # Statistics are already atomically committed. Failure to confirm directory
 # durability is a truthful operational flag, not a false failed transaction.
 confirmed=True
 try:
  fd=os.open(out.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
  try:os.fsync(fd)
  finally:os.close(fd)
 except OSError:confirmed=False
 return dict(reused=False,directory_fsync_confirmed=confirmed)

def main(argv=None):
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--request',required=True);parser.add_argument('--request-sha',required=True);parser.add_argument('--budget');parser.add_argument('--budget-sha');parser.add_argument('--slot',type=int,choices=(0,1));args=parser.parse_args(argv)
 source_guard(INITIAL_SHA);request,pins=load_request(args.request,args.request_sha)
 if not args.execute:return dict(status='READ_ONLY_REQUEST_BOUND_NO_SUMMARY_EXECUTED',backbone=request['backbone'])
 require(args.budget and args.budget_sha and args.slot is not None,'explicit budget and shared slot required')
 require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')),'empty CUDA and single CPU thread environment required')
 check_budget(args.budget,args.budget_sha)
 with cpu_slot(ROOT/'analysis/H/.cpu_slots',args.slot):
  publication_guard(pins,INITIAL_SHA,args.budget,args.budget_sha);payloads=build(request,pins,args.request_sha);publication_guard(pins,INITIAL_SHA,args.budget,args.budget_sha)
  out=ROOT/f"analysis/H/{request['backbone']}/five_seed_saved_result_summary";receipt_sha=hashlib.sha256(complete_bytes(payloads,args.request_sha,INITIAL_SHA)).hexdigest();operation=publish(out,payloads,args.request_sha,pins,INITIAL_SHA,args.budget,args.budget_sha,args.slot)
  return dict(output=str(out),sha256=receipt_sha,status='SAVED_RESULT_SUMMARY_COMPLETE',cpu_slot=args.slot,lifetime_shared_cpu_slot_held=True,operational_budget_ref=ref(args.budget,args.budget_sha),budget_excluded_from_statistical_identity=True,**operation)

if __name__=='__main__':print(json.dumps(main(),sort_keys=True))
