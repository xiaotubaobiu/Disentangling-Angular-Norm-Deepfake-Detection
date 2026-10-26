"""Small hand-checkable synthetic D packets; never scientific results."""
from pathlib import Path
import csv, hashlib, json, random

BB=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
SEEDS=(7,42,123,2026,3407)
TABLES=('per_seed_metrics','five_seed_metrics','per_seed_effects','five_seed_effects',
'per_seed_calibration','five_seed_calibration','per_seed_calibration_comparisons',
'five_seed_calibration_comparisons','per_seed_oracles','five_seed_oracles','per_seed_numerical','five_seed_numerical')
CODE=('analysis_d.py','scoring_d.py','stats_h.py','cache_d.py','bootstrap_fast.py','summary_d.py')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,sort_keys=True)+'\n')
def csvwrite(p,rows):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def csvrows(path):
 with Path(path).open() as f:return list(csv.DictReader(f))
def config(root):
 return json.loads((root/'configs/execution.json').read_text())
def rebind(root,bb,seed=None):
 if seed is not None:
  run=f'D_formal_{bb}_seed{seed}';p=root/'analysis/D'/run/'complete.json';a=json.loads(p.read_text())
  a['artifact_sha256']={str(x.relative_to(p.parent)):sha(x) for x in p.parent.rglob('*') if x.is_file() and x!=p};js(p,a)
 summary=root/'analysis/D'/bb/'five_seed_summary';p=summary/'complete.json';a=json.loads(p.read_text())
 a['identity']['analysis_complete_sha256']={str(s):sha(root/'analysis/D'/f'D_formal_{bb}_seed{s}'/'complete.json') for s in SEEDS if (root/'analysis/D'/f'D_formal_{bb}_seed{s}'/'complete.json').exists()}
 js(summary/'identity.json',a['identity'])
 a['artifact_sha256']={str(x.relative_to(summary)):sha(x) for x in summary.iterdir() if x.is_file() and x!=p};js(p,a)
 audit=root/'reports'/f'{bb}_five_seed_summary_independent_acceptance_fixture.json'
 source={str(root/'analysis/D'/f'D_formal_{bb}_seed{s}'/'complete.json'):sha(root/'analysis/D'/f'D_formal_{bb}_seed{s}'/'complete.json') for s in SEEDS if (root/'analysis/D'/f'D_formal_{bb}_seed{s}'/'complete.json').exists()}
 js(audit,dict(status='PASS_INDEPENDENT_REAL_FIVE_SEED_SUMMARY_STDLIB_RECOMPUTATION',backbone=bb,training_seed_count=5,seeds=list(SEEDS),summary_completion_sha256=sha(p),tables={n:dict(rows=len(csvrows(summary/(n+'.csv'))),fields_verified=1) for n in TABLES},source_files_bound_sha256=source))
def make(root,pool=80):
 root=Path(root);js(root/'SYNTHETIC_FIXTURE.json',dict(status='SYNTHETIC_TEST_INPUTS_NOT_RESULTS'))
 for name in CODE:(root/'science').mkdir(exist_ok=True);(root/'science'/name).write_text('# synthetic identity\n')
 cfg=dict(artifact_root=str(root),formal_seeds=list(SEEDS),runs={f'D_formal_{b}_seed{s}':dict(route='D',phase='formal',backbone=b,seed=s,recipe_frozen=True) for b in BB for s in SEEDS});js(root/'configs/execution.json',cfg);js(root/'data/D/freeze.json',{'synthetic':True})
 ids=[f'cal{i:03d}' for i in range(pool)];perms=[]
 for repeat in range(200):
  q=ids.copy();random.Random(100+repeat).shuffle(q);perms.append(q)
 testids=['testreal','testfake'];testhash=hashlib.sha256('\n'.join(sorted(testids)).encode()).hexdigest()
 for bi,b in enumerate(BB):
  for seed in SEEDS:
   run=f'D_formal_{b}_seed{seed}';folder=root/'analysis/D'/run
   for role,domain in [('source_test','ff'),('target_full','cdf')]:
    rows=[]
    for rep in ['raw','l2']:
     for uid,label,value in [('r',0,1),('f1',1,2),('f2',1,2)]:
      rows.append(dict(domain=domain,role=role,label=label,video_uid=role+uid,member_video_uids=json.dumps([role+uid]),grouping='correct',representation=rep,aggregation='mean_logit',score=str(value+bi),rank_score_decimal=str(900+value)))
    csvwrite(folder/role/'video_scores.csv',rows)
   for rep in ['raw','l2']:
    leaf=folder/'rq4'/f'{rep}_mean_logit'
    def vrow(uid,label,role):return dict(domain='cdf',role=role,label=label,video_uid=uid,member_video_uids=json.dumps([uid]),grouping='correct',representation=rep,aggregation='mean_logit',score='0',rank_score_decimal='0')
    csvwrite(leaf/'target_cal_scores.csv',[vrow(i,0,'target_cal') for i in ids] or [vrow('unused',0,'target_cal')])
    csvwrite(leaf/'target_test_scores.csv',[vrow(testids[0],0,'target_test'),vrow(testids[1],1,'target_test')])
    cs=[]
    for kind,budget in [('default',0),('source_real',140)]:
     cs.append(dict(calibration_kind=kind,budget=budget,repeat=-1,status='ok',test_id_hash=testhash,calibration_ids='',unique_set_count='',fpr='0.1',tpr='0.7',pooled_ba='0.8',operating_score_field='score'))
    for budget in [10,20,40,80]:
     sets=[p[:budget] if pool>=budget else [] for p in perms];count=len({tuple(sorted(s)) for s in sets}) if pool>=budget else 0
     for repeat,subset in enumerate(sets):
      cs.append(dict(calibration_kind='target_real',budget=budget,repeat=repeat,status='ok' if pool>=budget else 'unavailable',test_id_hash=testhash,calibration_ids=json.dumps(subset),unique_set_count=count,fpr=str(0.2 if budget==pool else (repeat%5)/10) if subset else '',tpr='0.6' if subset else '',pooled_ba=str(0.7 if budget==pool else [0.8,0.75,0.7,0.65,0.6][repeat%5]) if subset else '',operating_score_field='score' if subset else ''))
    csvwrite(leaf/'calibration.csv',cs)
   js(folder/'complete.json',dict(status='ANALYSIS_COMPLETE',run_id=run,seed=seed,backbone=b,identity=dict(calibration_repeats=200,test_bootstrap_repeats=2000,D_freeze_sha256=sha(root/'data/D/freeze.json'),cache_context=dict(config_sha256=sha(root/'configs/execution.json'))),artifact_sha256={}))
  summary=root/'analysis/D'/b/'five_seed_summary';summary.mkdir(parents=True)
  for name in TABLES:
   if name.startswith('per_seed'):
    values=[dict(backbone=b,seed=s,domain='source_test',estimate='' if s==7 else str(s),status='structural_NA' if s==7 else 'ok',CI_reference='unchanged.json') for s in SEEDS]
   else:
    values=[dict(backbone=b,n_seeds=5,expected_seeds=json.dumps(SEEDS),estimate_mean='',estimate_sample_sd='',status='unavailable_seed_estimate',inference_scope='descriptive_across_five_training_seeds')]
   csvwrite(summary/(name+'.csv'),values)
  js(summary/'identity.json',{'synthetic':True})
  js(summary/'complete.json',dict(schema='D_FIVE_SEED_SUMMARY_COMPLETE_V1',status='SUMMARY_COMPLETE',backbone=b,training_seed_count=5,identity=dict(backbone=b,seeds=list(SEEDS),config_sha256=sha(root/'configs/execution.json'),analysis_complete_sha256={},code_sha256={n:sha(root/'science'/n) for n in CODE}),artifact_sha256={}))
  for seed in SEEDS:rebind(root,b,seed)
 return root
