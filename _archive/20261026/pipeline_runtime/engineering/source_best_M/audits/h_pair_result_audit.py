"""Independent original H LAST saved-result audit. Default is preparation only.
Never imports producer analysis/scoring kernels, torch, encoders or source-context validators.
Real use requires a later explicit terminal authorization, terminal hash and lifetime shared CPU slot.
"""
from __future__ import annotations
import argparse, contextlib, csv, hashlib, json, math, os, re, stat, sys, time
from decimal import Decimal
from pathlib import Path
import numpy as np

ROOT=Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
E=ROOT/'engineering/source_best_M'
HERE=E/'audits'
ORIGINAL=Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2/reliability_v2')
PAIR=ROOT/'analysis/H/clip_vit_b16/seed42'
ADMISSION=ROOT/'reports/H_clip_vit_b16_seed42_independent_7role_parallel_candidate_20261025T0645.json'
ADMISSION_SHA='3e6c6366089ac131804e26d260e40f7af7c0bac92c87ee2b763d8a507c8c48f7'
ADMISSION_AUDIT=ROOT/'qualification/H_parallel_fresh_validation/independent_actual_CPU_parallel_pair_admission_audit.json'
ADMISSION_AUDIT_SHA='a7d29d89b9dcfc9fea48c6e6488e1632273fd3980a907161ed3e7de045165bf3'
LAUNCH=ROOT/'reports/H_primary_analysis_clip_vit_b16_seed42_launch_20261025T0830.json'
LAUNCH_SHA='cb63d1f452ef8240c3af9fbaa49780528078a1c43ddcbc303d86fbd316f1f3ec'
IDENTITY_SHA='b0200cf4495865f7fd716ce4f0b7666a4afe6690b8dcc9e85335ba5d8b34fb1f'
ROLES=('B/SOURCE_CAL','B/SOURCE_TEST','B/TARGET_TEST','common/B/SOURCE_TEST','common/B/TARGET_TEST','common/AB/SOURCE_TEST','common/AB/TARGET_TEST')
KINDS=('default','source_real')
METRICS=('pooled_auc','macro_auc','fpr','tpr','pooled_ba','macro_tpr','macro_ba')
COUNTS=('n_real','n_fake','false_positives','true_positives')
HISTORY={'H00':'B','H01':'B','H10':'AB','H11':'AB'}
IDS=('video_uid','label','method_id','group_id')
METHODS=('Deepfakes','Face2Face','FaceSwap','NeuralTextures')


def sha(path):
    p=Path(path)
    if p.is_symlink() or not p.is_file():raise ValueError(f'nonregular/symlink file: {p}')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def read_json(p):return json.loads(Path(p).read_text())

def require(test,message):
    if not test:raise AssertionError(message)

def close(actual,expected,label='number'):
    def missing(x):return x is None or x=='' or (isinstance(x,(float,np.floating)) and math.isnan(x)) or (isinstance(x,str) and x.lower()=='nan')
    if missing(expected):require(missing(actual),f'{label}: expected NA, got {actual}');return
    require(not missing(actual),f'{label}: unexpected NA')
    x,y=float(actual),float(expected)
    exact=isinstance(expected,(int,np.integer)) and not isinstance(expected,(bool,np.bool_))
    require(math.isfinite(x) and math.isfinite(y) and (x==y if exact else abs(x-y)<=1e-10),f'{label}: {actual} != {expected}')

def parse_rank(kind,text):
    if kind=='float64':x=Decimal.from_float(float(text))
    elif kind=='mpmath':x=Decimal(text)
    else:raise ValueError('unknown original precision rank kind')
    if not x.is_finite():raise ValueError('nonfinite original rank')
    return x

def boolean(x):
    if x is True or x=='True':return True
    if x is False or x=='False':return False
    raise ValueError('strict boolean required')

def source_threshold(scores):
    x=sorted(float(v) for v in scores)
    if not x or not all(math.isfinite(v) for v in x):raise ValueError('nonempty finite CAL required')
    allowed=math.floor(.05*len(x)+1e-12)
    return float(np.nextafter(x[-allowed-1],np.inf))

def percentile(values,q):return float(np.quantile(np.asarray(values,dtype=float),q,method='linear')) if values else float('nan')

def read_scores(path,aggregation,cal=False):
    with Path(path).open(newline='') as f:
        reader=csv.DictReader(f); fields=reader.fieldnames or []
        require(set(IDS+('score','rank_score_kind','rank_score_decimal','rank_score_export_digits','ranking_usable'))<=set(fields),'score columns')
        require(not {'weight','draw_id'}&set(fields),'score table already weighted')
        out=[]
        for row in reader:
            require(row.get('representation')=='raw' and row.get('grouping')=='correct' and row.get('aggregation')==aggregation,'score representation/aggregation')
            require(int(row['rank_score_export_digits'])>=350,'rank serialization precision')
            y=int(row['label']);s=float(row['score'])
            require(y in (0,1) and math.isfinite(s),'finite binary scores')
            require(all(row[k] for k in ('video_uid','method_id','group_id')),'nonempty identities')
            require((row['method_id']=='real')==(y==0),'label/method agreement')
            if cal:require(y==0,'CAL must be real only')
            out.append(dict(video_uid=row['video_uid'],label=y,method_id=row['method_id'],group_id=row['group_id'],
                score=s,rank=parse_rank(row['rank_score_kind'],row['rank_score_decimal']),ranking_usable=boolean(row['ranking_usable'])))
    require(out and len({x['video_uid'] for x in out})==len(out),'nonempty unique natural video UID')
    return sorted(out,key=lambda r:r['video_uid'])

def identities(q):return [tuple(r[k] for k in IDS) for r in sorted(q,key=lambda x:x['video_uid'])]

class IndependentCell:
    """Integer pair-concordance using sorted REAL prefixes and fake insertion bounds.
    No producer rank-bin or PreparedCell code is imported or executed.
    """
    def __init__(self,rows,methods,group_indices):
        self.rows=rows;self.methods=tuple(methods);self.g=np.asarray(group_indices,dtype=np.int64)
        self.y=np.array([r['label'] for r in rows]);self.real=np.flatnonzero(self.y==0);self.fake=np.flatnonzero(self.y==1)
        require(len(rows)==len(self.g),'group alignment')
        self.usable=all(r['ranking_usable'] for r in rows)
        self.s=np.array([r['score'] for r in rows],dtype=float)
        self.order=np.array(sorted(self.real,key=lambda i:rows[i]['rank']),dtype=np.int64)
        real_values=np.array([rows[i]['rank'] for i in self.order],dtype=object)
        fv=np.array([rows[i]['rank'] for i in self.fake],dtype=object)
        self.less=np.searchsorted(real_values,fv,side='left');self.leq=np.searchsorted(real_values,fv,side='right')
        self.method_masks=[np.array([rows[i]['method_id']==m for i in self.fake]) for m in methods]
    def evaluate(self,dense,tau):
        d=np.asarray(dense)
        if d.dtype.kind not in 'iu' or np.any(d<0):raise ValueError('integer nonnegative multiplicities required')
        w=d[self.g].astype(np.int64);nr=int(w[self.real].sum());nf=int(w[self.fake].sum())
        if not nr or not nf:raise ValueError('two-class AUC requires positive mass')
        require(nr*nf<2**61,'integer pair sum bound exceeded')
        fw=w[self.fake];masses=[int(fw[m].sum()) for m in self.method_masks]
        op=bool(self.methods and all(masses));macro=op and self.usable
        auc=ma=float('nan')
        if self.usable:
            cumulative=np.concatenate(([0],np.cumsum(w[self.order],dtype=np.int64)))
            twice=cumulative[self.less]+cumulative[self.leq]
            numerator=int(np.dot(fw,twice));auc=numerator/(2*nr*nf)
            if macro:ma=float(np.mean([int(np.dot(fw[m],twice[m]))/(2*nr*n) for m,n in zip(self.method_masks,masses)]))
        pred=self.s>=tau;fp=int(w[self.real][pred[self.real]].sum());tp=int(w[self.fake][pred[self.fake]].sum())
        mt=float(np.mean([int(fw[m & pred[self.fake]].sum())/n for m,n in zip(self.method_masks,masses)])) if op else float('nan')
        return dict(pooled_auc=auc,macro_auc=ma,n_real=nr,n_fake=nf,false_positives=fp,true_positives=tp,
            fpr=fp/nr,tpr=tp/nf,pooled_ba=(1-fp/nr+tp/nf)/2,macro_tpr=mt,macro_ba=(1-fp/nr+mt)/2,
            macro_valid=macro,macro_operating_valid=op,ranking_usable=self.usable,threshold=tau)

def terms(four):
    t={'H10-H00':(('H10',1),('H00',-1))}
    if four:t.update({'H01-H00':(('H01',1),('H00',-1)),'H11-H10':(('H11',1),('H10',-1)),
        '(H11-H10)-(H01-H00)':(('H11',1),('H10',-1),('H01',-1),('H00',1)),'H11-H01':(('H11',1),('H01',-1))})
    return t

def flatten(values):
    d={f'{kind}:cell:{cell}:{m}':v[m] for cell,kinds in values.items() for kind,v in kinds.items() for m in METRICS+COUNTS if m in v}
    d.update({f'{kind}:comparison:{name}:{m}':float(sum(coef*values[cell][kind][m] for cell,coef in expression))
        for name,expression in terms(len(values)==4).items() for kind in KINDS for m in METRICS})
    return d

def prepare(cells,cal,methods,aggregation):
    if set(cells) not in ({'H00','H10'},{'H00','H01','H10','H11'}):raise ValueError('exact 2/4 cells required')
    cells={k:sorted(v,key=lambda r:r['video_uid']) for k,v in sorted(cells.items())}
    ref=cells['H00'];refid=identities(ref)
    for q in cells.values():
        if identities(q)!=refid or len({r['video_uid'] for r in q})!=len(q):raise ValueError('cohort identities differ')
        if not {r['method_id'] for r in q if r['label']==1}<=set(methods):raise ValueError('unregistered method')
    if set(cal)!={'B','AB'} or identities(cal['B'])!=identities(cal['AB']):raise ValueError('matched B real-only CAL required')
    for q in cal.values():
        if not q or any(r['label']!=0 for r in q) or {r['video_uid'] for r in q}&{r['video_uid'] for r in ref}:raise ValueError('CAL real-only / TEST disjoint')
    groups=sorted({r['group_id'] for r in ref});idx={g:i for i,g in enumerate(groups)}
    gi=np.array([idx[r['group_id']] for r in ref],dtype=np.int64)
    prepared={k:IndependentCell(q,methods,gi) for k,q in cells.items()}
    thresholds={h:source_threshold(r['score'] for r in q) for h,q in cal.items()}
    default=0. if aggregation=='mean_logit' else .5
    def evaluate(d):return {k:{t:c.evaluate(d,default if t=='default' else thresholds[HISTORY[k]]) for t in KINDS} for k,c in prepared.items()}
    return cells,groups,gi,thresholds,evaluate

def recompute_cohort(cells,cal,methods,aggregation,*,repeats=2000,seed=20261023):
    cells,groups,gi,thresholds,evaluate=prepare(cells,cal,methods,aggregation)
    values=evaluate(np.ones(len(groups),dtype=np.int64));point=flatten(values)
    sizes=np.bincount(gi,minlength=len(groups));frac=float(sizes.max()/sizes.sum())
    doc=dict(n_source_groups=len(groups),source_groups=groups,largest_group_fraction=frac,repeats=repeats,seed=seed,
        min_groups=20,max_group_fraction=.5,macro_invalid_repeats=0,pooled_invalid_repeats=0,ranking_invalid_repeats=0,
        inference_scope='conditional_on_fixed_models_B_source_cal_and_caller_known_or_proxy_TEST_groups',ancestry_certified=False,
        source_cal_resampled=False,training_seeds_resampled=False,point_estimates=point,draws=[],intervals=None)
    if len(groups)<20 or frac>.5:doc['status']='insufficient_source_groups';return doc
    rng=np.random.default_rng(seed);storedbytes=0
    for repeat in range(repeats):
        dense=np.bincount(rng.choice(len(groups),size=len(groups),replace=True),minlength=len(groups))
        multiplicity={g:int(v) for g,v in zip(groups,dense) if v};storedbytes+=8*len(multiplicity)
        row=dict(repeat=repeat,group_draw_counts=multiplicity)
        try:v=evaluate(dense)
        except ValueError:
            doc['pooled_invalid_repeats']+=1;doc['macro_invalid_repeats']+=1
            row.update(pooled_valid=False,macro_valid=False,macro_operating_valid=False,ranking_valid=False)
        else:
            mv=all(x['macro_valid'] for t in v.values() for x in t.values());ov=all(x['macro_operating_valid'] for t in v.values() for x in t.values());rv=all(x['ranking_usable'] for t in v.values() for x in t.values())
            doc['macro_invalid_repeats']+=int(not mv);doc['ranking_invalid_repeats']+=int(not rv)
            row.update(pooled_valid=True,macro_valid=mv,macro_operating_valid=ov,ranking_valid=rv,**flatten(v))
        doc['draws'].append(row)
    doc['status']='ok' if any(r['pooled_valid'] for r in doc['draws']) else 'no_valid_draws'
    intervals=[]
    for key,estimate in point.items():
        kind,target_kind,target,metric=key.split(':',3)
        finite=[r[key] for r in doc['draws'] if key in r and math.isfinite(r[key])]
        intervals.append(dict(threshold_kind=kind,target_kind=target_kind,target=target,metric=metric,estimate=estimate,
            lower=percentile(finite,.025),upper=percentile(finite,.975),valid_repeats=len(finite),interval_type='pointwise_conditional_percentile'))
    doc.update(intervals=intervals,saved_multiplicity_array_bytes=storedbytes)
    return doc

def check_bootstrap(saved,cells,cal,methods,aggregation,repeats=2000,seed=20261023):
    expected=recompute_cohort(cells,cal,methods,aggregation,repeats=repeats,seed=seed)
    for k,v in expected.items():
        if k in ('draws','intervals','point_estimates'):continue
        if isinstance(v,float):close(saved.get(k),v,k)
        else:require(saved.get(k)==v,f'bootstrap metadata {k}')
    require(set(saved['point_estimates'])==set(expected['point_estimates']),'point estimate inventory')
    for k,v in expected['point_estimates'].items():close(saved['point_estimates'][k],v,k)
    require(isinstance(saved['draws'],list) and len(saved['draws'])==len(expected['draws']),'draw count')
    for actual,wanted in zip(saved['draws'],expected['draws']):
        for k in ('repeat','group_draw_counts','pooled_valid','macro_valid','macro_operating_valid','ranking_valid'):
            require(actual.get(k)==wanted[k],f'draw {wanted["repeat"]} {k}')
        require(all(type(x) is int and x>0 for x in actual['group_draw_counts'].values()),'integer positive sparse multiplicities')
        for k in expected['point_estimates']:close(actual.get(k),wanted.get(k),f'draw {wanted["repeat"]}:{k}')
        require(set(actual)<=set(wanted)|set(expected['point_estimates']),'unexpected draw column')
    if expected['intervals'] is None:require(saved['intervals'] is None,'insufficient groups must have NA intervals')
    else:
        key=lambda x:tuple(x[k] for k in ('threshold_kind','target_kind','target','metric'))
        intervals={key(r):r for r in saved['intervals']}
        require(len(intervals)==len(saved['intervals'])==len(expected['intervals']),'interval inventory')
        for wanted in expected['intervals']:
            actual=intervals[key(wanted)]
            for k in ('estimate','lower','upper'):close(actual[k],wanted[k],f'interval {key(wanted)} {k}')
            require(actual['valid_repeats']==wanted['valid_repeats'] and actual['interval_type']==wanted['interval_type'],'interval NA/type')
    return dict(draws_checked=len(expected['draws']),point_values_checked=len(expected['point_estimates']),intervals_checked=len(expected['intervals'] or []),status=expected['status'])

def inventory():
    names=['identity.json','coverage.json','independent_saved_score_recompute.json','limitations.json']
    for aggregation in ('mean_logit','mean_probability'):
        names.extend(f'{aggregation}/source_cal_{h}.csv' for h in ('B','AB'))
        for leaf in ('SOURCE_TEST','TARGET_TEST'):
            for cohort,cells in (('all_B',('H00','H10')),('common',('H00','H01','H10','H11'))):
                pre=f'{aggregation}/{leaf}/{cohort}'
                names.extend(f'{pre}/{cell}_scores.csv' for cell in cells)
                names.extend(f'{pre}/{name}' for name in ('metrics.csv','comparisons.csv','thresholds.json','conditional_bootstrap.json','independent_kernel_checks.json'))
    return names

def verify_terminal(folder,identity,terminal_sha):
    require(sha(folder/'complete.json')==terminal_sha,'externally observed terminal SHA')
    c=read_json(folder/'complete.json')
    require(c.get('schema')=='H_PAIR_ANALYSIS_COMPLETE_V1' and c.get('status') in ('ANALYSIS_COMPLETE','ANALYSIS_COMPLETE_WITH_NUMERIC_NA'),'actual H terminal')
    require(c.get('identity')==identity and all(c.get(k)==identity[k] for k in ('backbone','seed','effective_precision')),'terminal identity')
    require(c.get('independent_recompute')=='PASSED_FOR_ALL_REPORTABLE_CELLS','producer saved-score guard')
    require(not any(p.is_symlink() for p in folder.rglob('*')),'symlink in result inventory')
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p!=folder/'complete.json'}
    require(set(c['artifact_sha256'])==actual==set(inventory()),'complete artifact inventory')
    for name,h in c['artifact_sha256'].items():require(sha(folder/name)==h,f'artifact SHA {name}')
    return c

@contextlib.contextmanager
def cpu_slot(folder,slot):
    import fcntl
    if slot not in (0,1):raise ValueError('shared CPU slot0 or1 required')
    path=Path(folder)/f'slot{slot}.lock'
    fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW)
    try:
        s=os.fstat(fd)
        require(stat.S_ISREG(s.st_mode) and s.st_nlink==1,'canonical single-link CPU lock')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('CPU analysis slot busy; no wait or retry') from error
        yield fd
    finally:os.close(fd)

def serial7_paths():
    return dict(identity=ROOT/'analysis/H/clip_vit_b16/seed7/identity.json',admission=ROOT/'reports/H_clip_seed7_independent_7role_admission_20261025T0750.json',admission_acceptance=ROOT/'reports/H_clip_seed7_independent_cache_admission_acceptance_20261025T0950.json',admission_launch=ROOT/'reports/H_clip_seed7_independent_7role_admission_launch_20261025T0750.json',admission_entry=ROOT/'qualification/H_primary_cache_independent_admission/admit_clip_seed7_20261025T0750.py',primary_launch=ROOT/'reports/H_primary_analysis_clip_vit_b16_seed7_launch_20261025T0950.json',primary_execution=ROOT/'reports/H_primary_analysis_execution_clip_vit_b16_seed7.json',source_manifest=ROOT/'qualification/H_primary_cache_continuation/qualified_source_manifest.json',CPU_qualification=ROOT/'reports/H_parallel_CPU_actual_qualification_clip_vit_b16_20261025T0830.json',primary_entry=ROOT/'qualification/H_parallel_fresh_validation/cpu_primary_analysis.py')

def load_bindings(path,expected_sha):
    path=Path(path)
    require(path==HERE/'H_LAST_clip_seed7_audit_bindings.json','canonical audit-local binding spec path')
    require(type(expected_sha) is str and re.fullmatch('[0-9a-f]{64}',expected_sha) and sha(path)==expected_sha,'external binding spec SHA')
    b=read_json(path)
    require(set(b)=={'schema','root','backbone','seed','evidence_route','source_checkpoint','bootstrap_repeats','resampling_seed','proofs'},'binding field inventory')
    require(b['schema']=='H_LAST_PAIR_AUDIT_BINDINGS_SERIAL7_V1' and b['root']==str(ROOT) and b['backbone']=='clip_vit_b16'
        and type(b['seed']) is int and b['seed']==7 and b['evidence_route']=='ORIGINAL_SERIAL14_ACCEPTED','supported canonical serial7 scope')
    require(b['source_checkpoint']=='original_LAST_final.pt' and type(b['bootstrap_repeats']) is int and b['bootstrap_repeats']==2000
        and type(b['resampling_seed']) is int and b['resampling_seed']==20261023,'frozen original LAST numerical protocol')
    paths=serial7_paths();require(type(b['proofs']) is dict and set(b['proofs'])==set(paths),'exact serial7 proof inventory')
    for key,p in paths.items():
        proof=b['proofs'][key]
        require(type(proof) is dict and set(proof)=={'path','sha256'} and proof['path']==str(p)
            and type(proof['sha256']) is str and re.fullmatch('[0-9a-f]{64}',proof['sha256']),'canonical proof path/SHA '+key)
    return dict(b,_spec_path=str(path),_spec_sha256=expected_sha)

def validate_serial7_evidence(b,identity,d,terminal_sha):
    """Validate the SERIAL receipt as written, never fabricate parallel-admission fields."""
    proofs=b['proofs'];ad=d['admission'];ac=d['admission_acceptance'];al=d['admission_launch'];pl=d['primary_launch'];ex=d['primary_execution'];q=d['CPU_qualification']
    require(identity['backbone']==b['backbone'] and type(identity['seed']) is int and identity['seed']==b['seed'],'serial7 identity pair')
    for row in (ad,ac,al,pl,ex):
        require(row.get('backbone')==b['backbone'] and type(row.get('seed')) is int and row['seed']==b['seed'],'serial7 receipt actual pair')
    require(ad.get('status')=='PASS_ORIGINAL_FULL_14_ROLE_PAIR_CONTEXT_INDEPENDENT_OF_CACHE_WORKERS'
        and ad.get('identity')==identity and type(ad.get('models')) is int and ad['models']==2 and type(ad.get('roles')) is int and ad['roles']==14
        and ad.get('analysis_executed') is False and ad.get('GPU_launch_attempted') is False
        and ad.get('qualified_source_manifest_sha256')==proofs['source_manifest']['sha256'],'original serial14 admission')
    require('pinned_sha256' not in ad and 'fresh_ledger_invocations' not in ad,'serial evidence cannot masquerade as parallel')
    require(ac.get('status')=='PASS_ORIGINAL_SERIAL7_PAIR_FULL14ROLE_COMPLETION_CURRENT_CODE_INPUT_MARKER_REBIND'
        and ac.get('independent_from_cache_workers') is True and type(ac.get('models')) is int and ac['models']==2
        and type(ac.get('roles')) is int and ac['roles']==14 and type(ac.get('complete_marker_hashes_checked')) is int and ac['complete_marker_hashes_checked']==14
        and ac.get('source_report')==proofs['admission']['path'] and ac.get('source_report_sha256')==proofs['admission']['sha256']
        and ac.get('primary_statistics_or_metrics_generated') is False,'accepted serial14 source report binding')
    require(al.get('entry')==proofs['admission_entry']['path'] and al.get('entry_sha256')==proofs['admission_entry']['sha256']
        and al.get('science_function')=='unmodified canonical analysis_h._pair_context' and al.get('production_scoring_started') is False and al.get('GPU_launch_attempted') is False,'original serial entry binding')
    require(pl.get('entry')==proofs['primary_entry']['path'] and pl.get('entry_sha256')==proofs['primary_entry']['sha256']
        and pl.get('independent_pair_admission')==proofs['admission']['path'] and pl.get('admission_sha256')==proofs['admission']['sha256']
        and pl.get('qualification')==proofs['CPU_qualification']['path'] and pl.get('qualification_sha256')==proofs['CPU_qualification']['sha256']
        and type(pl.get('bootstrap_repeats')) is int and pl['bootstrap_repeats']==2000 and pl.get('GPUs_allowed') is False and pl.get('M_statistics_or_cache_started') is False,'original primary launch binding')
    require(q.get('schema')=='H_FRESH_PARALLEL_CPU_VALIDATION_ACCEPTANCE_V1' and q.get('status')=='PASS_ACTUAL_ORIGINAL14ROLE_FRESH140SOURCE_EXACT_LEDGER_EQUIVALENCE'
        and q.get('allowed_backbones')==['clip_vit_b16'] and q.get('independent_review_pass') is True and q.get('production_GPU_allowed') is False
        and q.get('original_source_manifest_sha256')==proofs['source_manifest']['sha256'],'accepted original CPU validation qualification')
    require(ex.get('status')=='ORIGINAL_PRIMARY_H_ANALYSIS_EXECUTION_COMPLETE_REQUIRES_INDEPENDENT_METRIC_BOOTSTRAP_AUDIT'
        and ex.get('pair_admission_sha256')==proofs['admission']['sha256'] and ex.get('source_qualification_sha256')==proofs['CPU_qualification']['sha256']
        and ex.get('original_analysis_identity')==identity and ex.get('original_analysis_complete_sha256')==terminal_sha
        and all(ex.get(k) is False for k in ('GPU_launch_attempted','primary_cache_owner_modified','M_results_generated','scientific_final_acceptance')),'original primary execution terminal binding')
    rows=ex.get('fresh_ledger_invocations');require(type(rows) is list and len(rows)==42,'exact42 execution ledger invocations')
    ledger=str(ROOT/'reports'/f"H_{identity['backbone']}_{identity['effective_precision']}_paired_ledger.json")
    reference=identity['inputs_sha256'][ledger]
    for row in rows:
        require(type(row) is dict and type(row.get('fresh_source_completions')) is int and row['fresh_source_completions']==10
            and row.get('backbone')==identity['backbone'] and row.get('precision')==identity['effective_precision']
            and row.get('reference_sha256')==reference and row.get('exact_original_ledger_except_top_utc') is True
            and row.get('cross_invocation_receipt_reuse') is False,'execution ledger exact original identity/fresh10/no reuse')

def pin_serial7_metadata(identity,b,terminal_sha):
    pins={b['_spec_path']:b['_spec_sha256']}
    def merge(more):
        for path,h in more.items():
            require(path not in pins or pins[path]==h,'conflicting proof SHA '+path);pins[path]=h
    merge({v['path']:v['sha256'] for v in b['proofs'].values()})
    for path,h in pins.items():require(sha(path)==h,'bound serial7 proof '+path)
    d={k:read_json(v['path']) for k,v in b['proofs'].items() if not k.endswith('_entry')}
    require(d['identity']==identity,'raw identity proof')
    validate_serial7_evidence(b,identity,d,terminal_sha)
    manifest=d['source_manifest'];require(b['proofs']['source_manifest']['sha256']=='977431237b901d5ac88172f98bc2591d3cd7992c35be50546cda34ad2276a747'
        and manifest['qualified_commit']=='5ef6d06bf7aeda02a5a11113e4c35a2c9909f215' and len(manifest['source_sha256'])==20,'original qualified20 source manifest')
    source_pins={str(ORIGINAL.parent/name):h for name,h in manifest['source_sha256'].items()}
    require(all(Path(path).parent==ORIGINAL and Path(path).suffix=='.py' for path in source_pins),'canonical original20 source paths')
    execution_pins=d['primary_execution']['pinned_sha256'];require(type(execution_pins) is dict and len(execution_pins)==110,'actual serial7 execution110 pin inventory')
    require(all(execution_pins.get(path)==h for path,h in source_pins.items()),'original20 code/actual execution binding')
    for path in execution_pins:
        p=Path(path);require(p.is_absolute() and p==Path(os.path.normpath(str(p))) and (p.is_relative_to(ROOT) or p.is_relative_to(ORIGINAL.parent))
            and p.suffix in ('.json','.py','.txt'),'metadata-only canonical execution pin')
    merge(execution_pins);merge(source_pins);merge(identity['inputs_sha256'])
    merge({str(ORIGINAL/name):h for name,h in identity['code_sha256'].items()})
    merge({manifest['original_H_posttraining_acceptance_path']:manifest['original_H_posttraining_acceptance_sha256']})
    expected={h+'/'+role for h in ('B','AB') for role in ROLES};require(set(identity['cache_completion_sha256'])==expected,'exact14 accepted serial cache markers')
    records={};heads={}
    for key,h in identity['cache_completion_sha256'].items():
        history,role=key.split('/',1);run=identity['run_ids'][history];path=ROOT/'features/H'/run/role/'complete.json';merge({str(path):h});require(sha(path)==h,'accepted serial marker SHA')
        m=read_json(path);i=m['identity']
        require(m.get('status')=='CACHE_COMPLETE' and m.get('precision')=='FP32' and m.get('autocast') is False
            and m.get('input_validation')=='FRESH_PROCESSED_IMAGE_BYTE_AND_DECODED_RGB_HASHES','accepted serial FP32 cache')
        require(i['run_id']==run and i['role']==role and i['history']==history and i['backbone']==identity['backbone']
            and i['effective_precision']==identity['effective_precision'] and i['checkpoint_path']==str(ROOT/'runs'/run/'final.pt')
            and i['dimension']==m['dimension']==512,'accepted original LAST serial cache header')
        require(m['rows']==i['accepted_table_rows'] and type(m['rows']) is int and m['rows']>0
            and type(i['accepted_table_videos']) is int and 0<i['accepted_table_videos']<=m['rows'],'accepted serial row/video metadata')
        require(history not in heads or heads[history]==i['original_head_sha256'],'one original head across7 roles');heads[history]=i['original_head_sha256']
        source=ROOT/'data/H'/f'{role}.parquet';require(i['source_manifest_path']==str(source),'canonical accepted role table');merge({str(source):i['source_manifest_sha256']})
        records[key]=dict(history=history,run_id=run,role=role,complete_sha256=h,original_head_sha256=i['original_head_sha256'],rows=m['rows'],accepted_videos=i['accepted_table_videos'])
    for path,h in pins.items():require(sha(path)==h,'current serial7 pin '+path)
    return pins,records

def pin_metadata(identity,bindings=None,terminal_sha=None):
    if bindings is not None:return pin_serial7_metadata(identity,bindings,terminal_sha)
    pins={str(ADMISSION):ADMISSION_SHA,str(ADMISSION_AUDIT):ADMISSION_AUDIT_SHA,str(LAUNCH):LAUNCH_SHA,str(PAIR/'identity.json'):IDENTITY_SHA}
    for p,h in list(pins.items()):require(sha(p)==h,f'accepted proof pin {p}')
    ad=read_json(ADMISSION);audit=read_json(ADMISSION_AUDIT);launch=read_json(LAUNCH)
    require(ad['status']=='PASS_ORIGINAL_FULL14ROLE_PAIR_CONTEXT_WITH_FRESH_PARALLEL_ORIGINAL_LEDGERS' and ad['identity']==identity and ad['roles']==14 and ad['models']==2,'independent14 role admission')
    require(audit['independent_review_pass'] is True and audit['actual_CLIP_original14role_fresh140ledger_source_receipt_audit_passed'] is True and audit['receipt_sha256']==ADMISSION_SHA and audit['original_primary_pair_identity']==identity,'accepted admission audit')
    require(launch['admission_sha256']==ADMISSION_SHA and launch['backbone']=='clip_vit_b16' and launch['seed']==42 and launch['bootstrap_repeats']==2000 and launch['GPUs_allowed'] is False,'actual original CPU launch binding')
    ledgers=ad.get('fresh_ledger_invocations')
    require(type(ledgers) is list and len(ledgers)==14,'exact14 prior independent ledger invocations')
    require(type(ad.get('all_source_completions_fresh')) is int and ad['all_source_completions_fresh']==140,
        'fresh prior independent source-completion total must be typed integer140')
    require(ad.get('cross_invocation_receipt_reuse') is False,'prior independent ledger receipt reuse forbidden')
    ledger_path=str(ROOT/'reports'/f"H_{identity['backbone']}_{identity['effective_precision']}_paired_ledger.json")
    reference_sha=identity['inputs_sha256'][ledger_path]
    for index,row in enumerate(ledgers):
        require(type(row) is dict and type(row.get('fresh_source_completions')) is int and row['fresh_source_completions']==10,
            f'prior independent ledger {index} requires typed integer10 source completions')
        require(row.get('backbone')==identity['backbone'] and row.get('precision')==identity['effective_precision']
            and row.get('reference_sha256')==reference_sha and row.get('exact_original_ledger_except_top_utc') is True
            and row.get('cross_invocation_receipt_reuse') is False,
            f'prior independent ledger {index} exact original identity/SHA/freshness')
    require(sum(row['fresh_source_completions'] for row in ledgers)==ad['all_source_completions_fresh'],
        'all14 source-completion counts must sum to140')
    pins.update(ad['pinned_sha256']);pins.update(identity['inputs_sha256']);pins.update({str(ORIGINAL/n):h for n,h in identity['code_sha256'].items()})
    require(audit['pinned_sha256']==ad['pinned_sha256'],'accepted pin inventory agreement')
    expected={h+'/'+r for h in ('B','AB') for r in ROLES};require(set(identity['cache_completion_sha256'])==expected,'14 roles exact')
    records={x['history']+'/'+x['role']:x for x in audit['current_role_marker_checks']};require(set(records)==expected and len(audit['current_role_marker_checks'])==14,'accepted role header inventory')
    for key,h in identity['cache_completion_sha256'].items():
        history,role=key.split('/',1);run=identity['run_ids'][history];p=ROOT/'features/H'/run/role/'complete.json';require(records[key]['complete_sha256']==h,'accepted role SHA')
        pins[str(p)]=h;m=read_json(p);i=m['identity'];require(m['status']=='CACHE_COMPLETE' and m['precision']=='FP32' and m['autocast'] is False and m['input_validation']=='FRESH_PROCESSED_IMAGE_BYTE_AND_DECODED_RGB_HASHES','FP32 accepted marker')
        require(i['run_id']==run and i['role']==role and i['history']==history and i['effective_precision']==identity['effective_precision'] and i['checkpoint_path']==str(ROOT/'runs'/run/'final.pt'),'original LAST model binding')
        require(i['original_head_sha256']==records[key]['original_head_sha256'] and m['rows']==records[key]['rows'] and i['accepted_table_videos']==records[key]['accepted_videos'],'accepted head/row/header')
        path=Path(i['source_manifest_path']);require(path==ROOT/'data/H'/f'{role}.parquet','canonical accepted role table');pins[str(path)]=i['source_manifest_sha256']
    for p,h in pins.items():require(sha(p)==h,f'current metadata/source pin {p}')
    return pins,records

def manifest_ids(role,run):
    # Only accepted row metadata, never processed pixels, feature arrays or model bytes.
    import pyarrow.parquet as pq
    marker=read_json(ROOT/'features/H'/run/role/'complete.json');p=Path(marker['identity']['source_manifest_path'])
    q=pq.read_table(p,columns=list(IDS)).to_pydict();ids={}
    for values in zip(*(q[k] for k in IDS)):
        row=tuple(int(x) if k=='label' else str(x) for k,x in zip(IDS,values))
        require(row[0] not in ids or ids[row[0]]==row,'role video identity inconsistent across frames');ids[row[0]]=row
    return sorted(ids.values())

def check_tables(base,cells,cal,methods,aggregation):
    _,_,_,thresholds,evaluate=prepare(cells,cal,methods,aggregation)
    saved_thresholds=read_json(base/'thresholds.json');require(set(saved_thresholds)=={'B','AB'},'two frozen source CAL thresholds')
    for h,t in thresholds.items():close(saved_thresholds[h],t,f'CAL boundary {h}')
    groups=sorted({r['group_id'] for r in cells['H00']});values=evaluate(np.ones(len(groups),dtype=np.int64));point=flatten(values)
    with (base/'metrics.csv').open(newline='') as f:metrics=list(csv.DictReader(f))
    require(len(metrics)==len(cells)*2 and {(x['cell'],x['threshold_kind']) for x in metrics}=={(c,k) for c in cells for k in KINDS},'point metric row inventory')
    for row in metrics:
        value=values[row['cell']][row['threshold_kind']]
        for k in METRICS+COUNTS+('threshold',):close(row[k],value[k],f'{base}:{row["cell"]}:{k}')
        require(row['model_history']==HISTORY[row['cell']],'threshold source model')
        for k in ('macro_valid','macro_operating_valid','ranking_usable'):require(boolean(row[k])==value[k],f'point validity {k}')
    with (base/'comparisons.csv').open(newline='') as f:comp=list(csv.DictReader(f))
    expected={k:v for k,v in point.items() if ':comparison:' in k};require(len(comp)==len(expected),'effect row count')
    seen=set()
    for row in comp:
        k=f'{row["threshold_kind"]}:comparison:{row["comparison"]}:{row["metric"]}'
        require(k in expected and k not in seen,'effect inventory');seen.add(k);close(row['estimate'],expected[k],k)
    checks=read_json(base/'independent_kernel_checks.json')
    require(len(checks)==len(cells)*2 and {(x['cell'],x['threshold_kind']) for x in checks}=={(c,k) for c in cells for k in KINDS},'kernel checks inventory')
    for check in checks:
        usable=values[check['cell']][check['threshold_kind']]['ranking_usable']
        require(check['status']==('PASSED' if usable else 'OPERATING_PASSED_RANKING_UNRESOLVED'),'kernel ranking NA declaration')
        require(math.isfinite(check['max_abs_error']) and 0<=check['max_abs_error']<=1e-10,'producer numeric check bound')
    summary=check_bootstrap(read_json(base/'conditional_bootstrap.json'),cells,cal,methods,aggregation)
    summary['numeric_NA_cells']=[c for c in cells if not values[c]['default']['ranking_usable']]
    summary['independent_point_estimates']={k:float(v) if math.isfinite(v) else None for k,v in point.items()}
    summary['source_real_thresholds']=thresholds
    summary['source_cal']={h:dict(n_real=len(q),false_positives=sum(r['score']>=thresholds[h] for r in q),
        allowed_false_positives=math.floor(.05*len(q)+1e-12),
        calibration_uid_sha256=hashlib.sha256('\n'.join(sorted(r['video_uid'] for r in q)).encode()).hexdigest()) for h,q in cal.items()}
    return summary

def _execute_owned(terminal_sha,slot,bindings=None):
    pair=PAIR if bindings is None else ROOT/f"analysis/H/{bindings['backbone']}/seed{bindings['seed']}"
    expected_seed=42 if bindings is None else bindings['seed']
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','audit must have empty CUDA')
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):require(os.environ.get(key) in ('1','2'),f'bounded CPU {key}')
    started=time.monotonic();identity=read_json(pair/'identity.json')
    require(identity['schema']=='H_PAIR_ANALYSIS_IDENTITY_V1' and identity['backbone']=='clip_vit_b16' and type(identity['seed']) is int and identity['seed']==expected_seed and identity['bootstrap_repeats']==2000 and identity['resampling_seed']==20261023,'frozen pair identity')
    require(identity['run_ids']=={h:f'H_formal_clip_vit_b16_{h}_seed{expected_seed}' for h in ('B','AB')},'frozen effective original runs')
    terminal=verify_terminal(pair,identity,terminal_sha);pins,records=pin_metadata(identity,bindings,terminal_sha)
    results=[];table_ids={role:manifest_ids(role,identity['run_ids']['B']) for role in ROLES}
    for aggregation in ('mean_logit','mean_probability'):
        cal={h:read_scores(pair/aggregation/f'source_cal_{h}.csv',aggregation,True) for h in ('B','AB')}
        for q in cal.values():require(identities(q)==table_ids['B/SOURCE_CAL'],'saved CAL row identities')
        for domain in ('SOURCE_TEST','TARGET_TEST'):
            for cohort,names in (('all_B',('H00','H10')),('common',('H00','H01','H10','H11'))):
                base=pair/aggregation/domain/cohort;cells={c:read_scores(base/f'{c}_scores.csv',aggregation) for c in names}
                for cell,q in cells.items():
                    version='AB' if cell in ('H01','H11') else 'B';role=f'B/{domain}' if cohort=='all_B' else f'common/{version}/{domain}'
                    require(identities(q)==table_ids[role],f'saved score exact accepted role {cell}/{role}')
                result=check_tables(base,cells,cal,METHODS if domain=='SOURCE_TEST' else ('DeepFakeDetection',),aggregation)
                results.append(dict(aggregation=aggregation,domain=domain,cohort=cohort,**result))
    saved_checks=read_json(pair/'independent_saved_score_recompute.json')
    require(len(saved_checks)==8,'saved recompute cohort inventory')
    expected_na=[];check_groups=set()
    for result in results:
        key=(result['aggregation'],result['domain'],result['cohort']);check_groups.add(key)
        matching=[x for x in saved_checks if (x['aggregation'],x['role'],x['cohort'])==key];require(len(matching)==1,'saved recompute exact cohort')
        names=('H00','H10') if result['cohort']=='all_B' else ('H00','H01','H10','H11')
        checks=matching[0]['checks'];require(len(checks)==len(names)*2 and {(x['cell'],x['threshold_kind']) for x in checks}=={(c,k) for c in names for k in KINDS},'saved recompute cell inventory')
        for row in checks:
            status='PASS_OPERATING_ONLY_NUMERIC_AUC_NA' if row['cell'] in result['numeric_NA_cells'] else 'PASS'
            require(row['status']==status,'saved numeric NA status')
            if status!='PASS':expected_na.append(dict(role=result['domain'],cohort=result['cohort'],aggregation=result['aggregation'],**row))
    require(len(check_groups)==8,'eight independent cohorts')
    limitations=read_json(pair/'limitations.json');require(limitations['numerical_NA']==expected_na,'numerical NA declarations')
    require(terminal['status']==('ANALYSIS_COMPLETE_WITH_NUMERIC_NA' if expected_na else 'ANALYSIS_COMPLETE'),'terminal numerical NA status')
    require(limitations['training_seed_count']==1 and limitations['completed_model_pairs']==1 and limitations['five_seed_summary_pending'] is True,'single original pair inference')
    # The producer's report/coverage and numeric checks are inventoried and pinned, not used as an independent numeric oracle.
    for p,h in pins.items():require(sha(p)==h,f'pin changed during audit {p}')
    require(verify_terminal(pair,identity,terminal_sha)==terminal,'terminal changed during audit')
    snapshot_four={
        'mean_logit/SOURCE_TEST/all_B/conditional_bootstrap.json':'ddc4b28fbc134c6fe03a6220699c86f21b3c98f3dc5855203ad9d2028ce27be5',
        'mean_logit/TARGET_TEST/all_B/conditional_bootstrap.json':'90295899175f91bf1e6a779a26574f1ffcc3009d4521fc04b7d33e65fefd1f79',
        'mean_probability/SOURCE_TEST/all_B/conditional_bootstrap.json':'2884b9c456888ea4f5306022ac0e9619c36890f8dfd67d3902ab022a5d9b0c9d',
        'mean_probability/TARGET_TEST/all_B/conditional_bootstrap.json':'f1e1a0e5b39d59c3ad7e8e139494a1b5d8d434f2952f8aa2cc4cf1adc820797c'}
    snapshot_match=(sha(pair/'identity.json')==IDENTITY_SHA and all(terminal['artifact_sha256'].get(n)==h for n,h in snapshot_four.items())) if bindings is None else None
    return dict(schema='INDEPENDENT_ORIGINAL_H_LAST_PAIR_RESULT_AUDIT_V1',status='SAVED_RESULT_NUMERICS_AND_BOUND_PRIOR_ROLE_ADMISSION_PASS',
        backbone=identity['backbone'],seed=identity['seed'],source_checkpoint='original_LAST_final.pt',terminal_sha256=terminal_sha,
        identity_sha256=sha(pair/'identity.json'),binding_spec_sha256=bindings['_spec_sha256'] if bindings else None,evidence_route=bindings['evidence_route'] if bindings else 'ACCEPTED_PARALLEL14_SEED42',audit_code_sha256=sha(__file__),artifact_count=72,current_pins_sha256=pins,
        cohorts=results,cpu_slot=slot,lifetime_shared_cpu_slot_held=True,elapsed_seconds=time.monotonic()-started,
        snapshot_four_artifacts_and_identity_match=snapshot_match,snapshot_promoted=False,
        fresh_pixel_or_source_context_validation=False,fresh_feature_dot_product_recompute=False,weight_or_model_loading=False,
        independent_scientific_scope='saved natural-video scores, point metrics, fixed B-source-CAL thresholds, signed effects and every applicable shared-group draw/NA/interval; prior14 cache admission rebound by SHA',
        pending=['Independent frame-to-video/feature-dot-product reconstruction is outside this saved-score audit; immutable accepted cache admission is reused.','coverage.json is inventoried/SHA-bound; fresh independent geometry/coverage reconstruction is not performed by this saved-score audit.','Single seed-pair result only; five-seed summary remains pending.','Existing snapshot promotion requires separate review/acceptance; no encoding or resampling performed.'],
        M_source_best_used=False,M_production_authorized=False,GPU_or_target_inference_started=False)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--terminal-sha');p.add_argument('--slot',type=int,choices=(0,1));p.add_argument('--output-name');p.add_argument('--binding-spec');p.add_argument('--binding-spec-sha')
    args=p.parse_args(argv)
    if not args.execute:
        print('PREPARATION_ONLY: no production reads, CPU-slot locks or real audit. Await explicit terminal authorization.');return 0
    if not args.terminal_sha or len(args.terminal_sha)!=64 or args.slot is None:p.error('execute requires externally observed terminal SHA and shared CPU slot')
    if bool(args.binding_spec)!=bool(args.binding_spec_sha):p.error('binding spec requires external SHA and vice versa')
    bindings=load_bindings(args.binding_spec,args.binding_spec_sha) if args.binding_spec else None
    expected_output=f"H_LAST_clip_seed{bindings['seed'] if bindings else 42}_independent_audit.json"
    args.output_name=args.output_name or expected_output
    require(args.output_name==expected_output,'canonical pair audit output name')
    require(Path(args.output_name).name==args.output_name and args.output_name.endswith('.json'),'audit-local JSON filename required')
    dest=HERE/args.output_name;require(not dest.exists(),'preserve prior audit output')
    with cpu_slot(ROOT/'analysis/H/.cpu_slots',args.slot):
        result=_execute_owned(args.terminal_sha,args.slot,bindings)
        with dest.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=result['status'],output=str(dest),sha256=sha(dest))));return 0

if __name__=='__main__':raise SystemExit(main())
