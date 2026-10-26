"""Independent finite audit of already saved rank tables. No pipeline imports.
Exact Decimal comparisons plus bisect concordance, distinct from grouped-sort implementation.
Partial role audit only; no claim that full model analysis is complete.
"""
from pathlib import Path
from decimal import Decimal
from fractions import Fraction
from bisect import bisect_left,bisect_right
import csv,json,hashlib,datetime,time,math,os
R=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
ROOT=R/'analysis/D/D_formal_efficientnet_b0_seed42'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
freeze=R/'data/D/freeze.json';f=json.loads(freeze.read_text());methods=sorted(f['method_registry']['FS9']+f['method_registry']['FR12'])
inputs={str(freeze):sha(freeze)};records=[];checks=0

def close(actual,saved):
 global checks
 checks+=1
 if actual is None:
  assert saved=='' or saved.lower()=='nan',('Expected NA',saved)
 else:
  assert math.isfinite(float(saved)) and abs(float(actual)-float(saved))<=1e-10,(actual,saved)

def exact_auc(reals,fakes):
 assert reals and fakes
 total=sum(bisect_left(reals,x)+bisect_right(reals,x) for x in fakes)
 return Fraction(total,2*len(reals)*len(fakes))

for role in ['source_test','target_full']:
 scorefile=ROOT/role/'video_scores.csv';metricfile=ROOT/role/'metrics.csv'
 for p in [scorefile,metricfile]:inputs[str(p)]=sha(p)
 cells={}
 with scorefile.open(newline='') as stream:
  for row in csv.DictReader(stream):
   key=tuple(row[k] for k in ['grouping','representation','aggregation'])
   value=Decimal(row['rank_score_decimal']);assert value.is_finite()
   assert row['ranking_usable'] in ['True','False']
   cells.setdefault(key,[]).append(dict(label=int(row['label']),method=row['method_id'],uid=row['video_uid'],rank=value,usable=row['ranking_usable']=='True'))
 with metricfile.open(newline='') as stream:metrics=list(csv.DictReader(stream))
 assert len(metrics)==len(cells)==8
 seen=set()
 for m in metrics:
  key=tuple(m[k] for k in ['grouping','representation','aggregation']);assert key not in seen;seen.add(key);rows=cells[key]
  assert len({x['uid'] for x in rows})==len(rows) and {x['label'] for x in rows}=={0,1}
  reals=sorted(x['rank'] for x in rows if x['label']==0);fakes=[x['rank'] for x in rows if x['label']==1];usable=all(x['usable'] for x in rows)
  assert usable==(m['ranking_usable']=='True')
  for field,value in [('n_real',len(reals)),('n_fake',len(fakes)),('real_mass',len(reals)),('fake_mass',len(fakes))]:close(value,m[field])
  auc=exact_auc(reals,fakes) if usable else None;close(auc,m['pooled_auc'])
  missing=[method for method in methods if not any(x['label']==1 and x['method']==method for x in rows)]
  assert json.loads(m['missing_methods'])==missing
  macro_valid=key[0]=='correct' and usable and not missing
  assert macro_valid==(m['macro_valid']=='True')
  macro=[exact_auc(reals,[x['rank'] for x in rows if x['label']==1 and x['method']==method]) for method in methods] if macro_valid else []
  macro_value=sum(float(x) for x in macro)/len(macro) if macro else None
  close(macro_value,m['macro_auc']);close(None,m['macro_tpr']);close(None,m['macro_ba'])
  records.append(dict(role=role,grouping=key[0],representation=key[1],aggregation=key[2],ranking_usable=usable,n_real=len(reals),n_fake=len(fakes),pooled_concordance_numerator=auc.numerator if auc is not None else None,pooled_concordance_denominator=auc.denominator if auc is not None else None,pooled_auc_abs_error=abs(float(auc)-float(m['pooled_auc'])) if auc is not None else None,macro_valid=macro_valid,macro_auc_abs_error=abs(macro_value-float(m['macro_auc'])) if macro_value is not None else None,status='PASS' if usable else 'PASS_PRESERVED_NUMERIC_NA'))
for name,value in inputs.items(): assert sha(Path(name))==value,'Input changed during audit: '+name
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_SAVED_TWO_ROLE_RANK_METRICS_ONLY_FULL_ANALYSIS_STILL_PENDING',run_id='D_formal_efficientnet_b0_seed42',algorithm='Python Decimal exact-order bisect twice-concordance, Fraction ratio; no pipeline/scoring/analysis imports',CPU_only=True,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),source_input_sha256=inputs,cells=len(records),numeric_scalar_checks=checks,records=records,max_pooled_auc_abs_error=max(x['pooled_auc_abs_error'] or 0 for x in records),max_macro_auc_abs_error=max(x['macro_auc_abs_error'] or 0 for x in records),full_analysis_accepted=False,calibration_bootstrap_resource_terminal_acceptance_pending=True)
p=R/'reports/D_formal_efficientnet_b0_seed42_saved_rank_recompute.json';assert not p.exists();p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['status','cells','numeric_scalar_checks','max_pooled_auc_abs_error','max_macro_auc_abs_error','full_analysis_accepted']}))
