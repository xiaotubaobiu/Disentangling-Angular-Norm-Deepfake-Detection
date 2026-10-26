"""Independent stdlib verification of D five-seed summaries; no pipeline imports."""
from pathlib import Path
import csv,json,math,statistics,hashlib,datetime,collections,sys,ast
SEEDS=(7,42,123,2026,3407)
METRICS=('pooled_auc','macro_auc','fpr','tpr','pooled_ba','macro_tpr','macro_ba','false_positives','true_positives')
ERRORS=[]
def scalar(v):
 if v is None or v=='':return None
 if isinstance(v,bool):return v
 if isinstance(v,str) and v in ('True','False'):return v=='True'
 try:
  n=float(v)
  return None if math.isnan(n) else n
 except (TypeError,ValueError):return v

def seed_stats(pairs):
 if len(pairs)!=5 or {int(x[0]) for x in pairs}!=set(SEEDS) or len({int(x[0]) for x in pairs})!=5:raise ValueError('exact five unique registered seeds required')
 values=[scalar(x[1]) for x in pairs];valid=[v for v in values if isinstance(v,(int,float)) and math.isfinite(v)];complete=len(valid)==5
 return dict(n_seeds=5,expected_seeds=list(SEEDS),n_valid_seeds=len(valid),complete_five_seeds=True,all_estimates_available=complete,
  estimate_mean=statistics.mean(valid) if complete else None,estimate_sample_sd=statistics.stdev(valid) if complete else None,
  n_positive=sum(v>1e-10 for v in valid),n_negative=sum(v< -1e-10 for v in valid),n_zero_within_tolerance=sum(abs(v)<=1e-10 for v in valid),
  status='ok' if complete else 'unavailable_seed_estimate',model_count=5,inference_scope='descriptive_across_five_training_seeds')

def subset_stats(values,statuses,expected):
 if len(values)!=expected or len(statuses)!=expected:raise ValueError('all frozen subsets required')
 values=[scalar(v) for v in values];valid=[v for v,s in zip(values,statuses) if isinstance(v,(int,float)) and math.isfinite(v) and s=='ok'];complete=len(valid)==expected
 out=dict(estimate=statistics.mean(valid) if complete else None,n_subsets=expected,n_valid_subsets=len(valid),subset_sample_sd=None,subset_q025=None,subset_q975=None,status='ok' if complete else 'unavailable_subset_estimate')
 if complete and expected>1:
  ordered=sorted(valid)
  def quantile(p):
   position=(expected-1)*p;i=int(math.floor(position));f=position-i
   return ordered[i]+(ordered[min(i+1,expected-1)]-ordered[i])*f
  out.update(subset_sample_sd=statistics.stdev(valid),subset_q025=quantile(.025),subset_q975=quantile(.975))
 return out

def check_rows(actual,expected,keys):
 def index(rows):
  result={}
  for row in rows:
   key=tuple(scalar(row[k]) for k in keys);assert key not in result,('duplicate stratum',key);result[key]=row
  return result
 a,e=index(actual),index(expected);assert set(a)==set(e),('stratum set differs',len(a),len(e),list(set(a)^set(e))[:2]);checks=0
 for key,row in e.items():
  assert set(a[key])==set(row),('schema differs',key,set(a[key])^set(row))
  for name,want in row.items():
   got=scalar(a[key][name]);want=scalar(want);checks+=1
   if isinstance(want,bool):assert isinstance(got,bool) and got==want,(key,name,got,want)
   elif isinstance(want,(int,float)):
    assert isinstance(got,(int,float)) and not isinstance(got,bool),(key,name,got,want)
    if math.isinf(want):assert got==want,(key,name,got,want)
    else:
     assert math.isclose(got,want,rel_tol=2e-12,abs_tol=3e-12),(key,name,got,want)
     ERRORS.append(abs(got-want))
   elif isinstance(want,(list,dict,tuple)):
    parsed=ast.literal_eval(got) if isinstance(got,str) else got
    assert parsed==want,(key,name,parsed,want)
   else:assert got==want,(key,name,got,want)
 return checks

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def run(root,backbone):
 root=Path(root);output=root/'analysis/D'/backbone/'five_seed_summary';marker=output/'complete.json';marker_sha=digest(marker);receipt=json.loads(marker.read_text())
 assert receipt['status'] in ('SUMMARY_COMPLETE','SUMMARY_COMPLETE_WITH_NUMERIC_NA') and receipt['training_seed_count']==5 and receipt['backbone']==backbone
 assert receipt['identity']==json.loads((output/'identity.json').read_text()) and receipt['identity']['seeds']==list(SEEDS)
 inventory=set(receipt['artifact_sha256']);assert inventory=={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}-{'complete.json'}
 assert len(inventory)==15 and all(not (output/n).is_symlink() and digest(output/n)==h for n,h in receipt['artifact_sha256'].items())
 config=root/'configs/execution.json';assert digest(config)==receipt['identity']['config_sha256'];cfg=json.loads(config.read_text())
 code_root=Path(cfg['code_root']) if 'code_root' in cfg else Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
 for name,h in receipt['identity']['code_sha256'].items():assert digest(code_root/'reliability_v2'/name)==h
 support=json.loads((output/'supporting_artifacts.json').read_text());assert support['calibration_repeats']==200 and support['test_bootstrap_repeats']==2000
 source_hashes={};data=collections.defaultdict(list);calraw=[]
 def read_csv(p):return list(csv.DictReader(p.open(newline='')))
 for seed in SEEDS:
  run_id=f'D_formal_{backbone}_seed{seed}';base=root/'analysis/D'/run_id;complete=base/'complete.json';j=json.loads(complete.read_text())
  assert j['seed']==seed and j['backbone']==backbone and digest(complete)==receipt['identity']['analysis_complete_sha256'][str(seed)]
  external=root/'reports'/('D_first_terminal_acceptance_20261024T1430.json' if seed==7 and backbone=='clip_vit_b16' else run_id+'_terminal_acceptance.json')
  accepted=json.loads(external.read_text());assert accepted['status'].startswith('PASS_') and accepted['completion_sha256']==digest(complete)
  source_hashes[str(complete)]=digest(complete);source_hashes[str(external)]=digest(external)
  bound=support['models'][str(seed)];assert bound['run_id']==run_id and bound['analysis_complete_sha256']==digest(complete) and set(bound['artifacts'])==set(j['artifact_sha256'])
  for n,h in j['artifact_sha256'].items():assert bound['artifacts'][n]=={'path':str(base/n),'sha256':h}
  def source(n,kind='csv'):
   p=base/n;assert not p.is_symlink() and digest(p)==j['artifact_sha256'][n];source_hashes[str(p)]=digest(p)
   return read_csv(p) if kind=='csv' else json.loads(p.read_text())
  for role in ('source_test','target_full'):
   for label,file in [('metrics','metrics.csv'),('effects','effects.csv'),('numerical','numerical_audits.csv')]:
    for row in source(role+'/'+file):data[label].append(dict(row,seed=seed,backbone=backbone,domain=role))
  for rep in ('raw','l2'):
   for agg in ('mean_logit','mean_probability'):
    flow='rq4/'+rep+'_'+agg+'/'
    for row in source(flow+'calibration.csv'):calraw.append(dict(row,seed=seed,backbone=backbone,domain='target_test',representation=rep,aggregation=agg))
    oracle=source(flow+'oracle.json','json')
    if isinstance(oracle,dict):
     assert oracle['status']=='UNRESOLVED_NUMERIC_ORDERING';oracle=[dict(criterion=c,status='unresolved_numeric_ordering',threshold=None,**{m:None for m in METRICS}) for c in ('max_ba','fpr_le_0.05','fpr_le_0.10')]
    for row in oracle:data['oracles'].append(dict(row,seed=seed,backbone=backbone,domain='target_test',representation=rep,aggregation=agg))
 def groups(rows,keys):
  out=collections.defaultdict(list)
  for r in rows:out[tuple(scalar(r[k]) for k in keys)].append(r)
  return out
 def seed_reduce(rows,keys):
  return [dict(zip(keys,k),**seed_stats([(r['seed'],r['estimate']) for r in q])) for k,q in groups(rows,keys).items()]
 mk=['seed','backbone','domain','representation','aggregation'];points=[]
 for k,q in groups(calraw,mk+['calibration_kind','budget']).items():
  identity=dict(zip(mk+['calibration_kind','budget'],k));kind=identity['calibration_kind'];expected=200 if kind=='target_real' else 1
  assert len(q)==expected and set(scalar(r['repeat']) for r in q)==(set(range(200)) if expected==200 else {-1})
  for metric in METRICS:points.append(dict(identity,comparison=kind,threshold_kind=kind,metric=metric,**subset_stats([r.get(metric) for r in q],[r['status'] for r in q],expected)))
 comparisons=[]
 for _,q in groups(points,mk+['metric']).items():
  source=[r for r in q if r['calibration_kind']=='source_real'];assert len(source)==1
  for r in q:
   if r['calibration_kind']=='target_real':comparisons.append(dict(r,comparison='target_real_minus_source_real',threshold_kind='target_vs_source',estimate=None if r['estimate'] is None or source[0]['estimate'] is None else r['estimate']-source[0]['estimate'],primary_budget=r['budget']==20,scope='within-seed CAL mean minus same-seed source operating metric'))
 expected={'per_seed_metrics':data['metrics'],'per_seed_effects':data['effects'],'per_seed_numerical':data['numerical'],'per_seed_oracles':data['oracles'],'per_seed_calibration':points,'per_seed_calibration_comparisons':comparisons}
 metriclong=[dict({k:r[k] for k in ['seed','backbone','domain','grouping','representation','aggregation']},metric=m,estimate=r[m],comparison='cell',threshold_kind='ranking') for r in data['metrics'] for m in ('pooled_auc','macro_auc')]
 expected['five_seed_metrics']=seed_reduce(metriclong,['backbone','domain','grouping','representation','aggregation','comparison','threshold_kind','metric'])
 calkeys=['backbone','domain','representation','aggregation','budget','comparison','threshold_kind','metric']
 expected['five_seed_calibration']=seed_reduce(points,calkeys);expected['five_seed_calibration_comparisons']=seed_reduce(comparisons,calkeys)
 oraclelong=[dict({k:r[k] for k in mk},metric=m,estimate=r.get(m),comparison=r['criterion'],threshold_kind='oracle_diagnostic') for r in data['oracles'] for m in METRICS]
 expected['five_seed_oracles']=seed_reduce(oraclelong,['backbone','domain','representation','aggregation','comparison','threshold_kind','metric'])
 effects=[]
 for k,q in groups(data['effects'],['backbone','domain','aggregation']).items():
  assert len(q)==5 and {int(r['seed']) for r in q}==set(SEEDS);valid=[r for r in q if scalar(r['valid']) is True];complete=len(valid)==5
  item=dict(zip(['backbone','domain','aggregation'],k),n_seeds=5,complete_five_seeds=complete,n_valid_seeds=len(valid),n_seed_reversals=sum(scalar(r['reversal']) is True for r in q),reversal_count_denominator=len(valid))
  for name in ('delta_correct','delta_collision','interaction'):
   values=[scalar(r[name]) for r in valid];item[name+'_mean']=statistics.mean(values) if complete else None;item[name+'_sample_sd']=statistics.stdev(values) if complete else None
  a,b=item['delta_correct_mean'],item['delta_collision_mean'];item['mean_reversal']=None if not complete else ((a>1e-10 and b< -1e-10) or (a< -1e-10 and b>1e-10));effects.append(item)
 expected['five_seed_effects']=effects
 nk=['backbone','domain','grouping','representation','aggregation','scale'];numerical=[]
 for k,q in groups(data['numerical'],nk).items():
  assert len(q)==5 and {int(r['seed']) for r in q}==set(SEEDS)
  numerical.append(dict(zip(nk,k),n_seeds=5,n_ranking_usable=sum(scalar(r['ranking_usable']) is True for r in q),n_precision_triggered=sum(scalar(r['triggered']) is True for r in q),n_negative_control_evaluable=sum(scalar(r['negative_control_order_preserved']) is not None for r in q),n_negative_control_preserved=sum(scalar(r['negative_control_order_preserved']) is True for r in q),scope='descriptive training-seed counts, not significance'))
 expected['five_seed_numerical']=numerical
 keys={'per_seed_metrics':['seed','backbone','domain','grouping','representation','aggregation'],'per_seed_effects':['seed','backbone','domain','aggregation'],'per_seed_numerical':['seed']+nk,'per_seed_oracles':mk+['criterion'],'per_seed_calibration':mk+['calibration_kind','budget','metric'],'per_seed_calibration_comparisons':mk+['calibration_kind','budget','metric'],'five_seed_metrics':['backbone','domain','grouping','representation','aggregation','comparison','threshold_kind','metric'],'five_seed_effects':['backbone','domain','aggregation'],'five_seed_numerical':nk,'five_seed_calibration':calkeys,'five_seed_calibration_comparisons':calkeys,'five_seed_oracles':['backbone','domain','representation','aggregation','comparison','threshold_kind','metric']}
 checks={}
 for name,want in expected.items():
  actual=read_csv(output/(name+'.csv'));checks[name]={'rows':len(actual),'fields_verified':check_rows(actual,want,keys[name])}
 for p,h in source_hashes.items():assert digest(p)==h
 assert digest(marker)==marker_sha and all(digest(output/n)==h for n,h in receipt['artifact_sha256'].items())
 report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_INDEPENDENT_REAL_FIVE_SEED_SUMMARY_STDLIB_RECOMPUTATION',backbone=backbone,training_seed_count=5,seeds=list(SEEDS),summary_completion_sha256=marker_sha,summary_artifacts=len(inventory),tables=checks,field_checks=sum(v['fields_verified'] for v in checks.values()),max_absolute_numeric_error=max(ERRORS,default=0),source_files_bound_sha256=source_hashes,source_files_count=len(source_hashes),pipeline_modules_imported=False,full_input_artifact_hashes_repeated=False,scope='All12 aggregate/per-seed tables independently rebuilt from committed seed-level metric/effect/numerical/CAL/oracle files; CAL200 reduced within seed, sampleSD ddof1, missing-seed estimates preserved. Raw scoring and bootstrap kernels remain covered by original individual audits; no new effect significance or NP guarantee.')
 out=root/'reports'/(backbone+'_five_seed_summary_independent_acceptance_20261024T1630.json');assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({k:report[k] for k in ('status','training_seed_count','summary_artifacts','field_checks','max_absolute_numeric_error')}))
 return report
if __name__=='__main__':run(sys.argv[1],sys.argv[2])
