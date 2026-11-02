"""CPU-only assembly of five accepted D analyses; no inference or training.

Training-seed variation, CAL-subset spread and conditional TEST intervals remain
separate. Inputs must be the new source-DEV-selected best EMA endpoints and all seven role
caches, never development/historical artifacts or a silent valid-seed subset.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import fcntl
import hashlib
import os
import stat
import tempfile
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import analysis_d as io
from . import scoring_d as score
from . import stats_h
from .trainer import digest

SEEDS=(7,42,123,2026,3407)
BACKBONES=('clip_vit_b16','dinov2_vit_b14','efficientnet_b0')
ROLES=('source_test','target_full')
AGGREGATIONS=('mean_logit','mean_probability')
REPRESENTATIONS=('raw','l2')
METRICS=('pooled_auc','macro_auc','fpr','tpr','pooled_ba','macro_tpr','macro_ba','false_positives','true_positives')
MODEL_KEYS=['seed','backbone','domain','representation','aggregation']


def seed_summary(rows, group_columns):
    """Reuse the existing strict five-model mean/sample-SD implementation."""
    return stats_h.summarize_seed_comparisons(rows,group_columns=group_columns)


def effect_summary(rows):
    result=[]
    for (backbone,domain),part in rows.groupby(['backbone','domain'],sort=False):
        for aggregation,q in part.groupby('aggregation',sort=False):
            if len(q)!=5 or set(q.seed)!=set(SEEDS) or q.seed.duplicated().any():
                raise ValueError('exact five registered seeds required in D effects')
        table=score.summarize_seed_effects(part,expected_seeds=SEEDS)
        for row in table.to_dict('records'):
            row.update(backbone=backbone,domain=domain,reversal_count_denominator=row['n_valid_seeds'])
            if not row['complete_five_seeds']:
                for key in row:
                    if key.endswith('_mean') or key.endswith('_sample_sd'):row[key]=float('nan')
                row['mean_reversal']=None
            result.append(row)
    return pd.DataFrame(result)


def calibration_tables(rows):
    """Reduce 200 subsets within each model before any five-model summary."""
    keys=MODEL_KEYS+['calibration_kind','budget'];points=[]
    if rows.empty or not set(keys+['repeat','status'])<=set(rows):
        raise ValueError('nonempty D calibration rows required')
    for values,q in rows.groupby(keys,sort=False,dropna=False):
        identity=dict(zip(keys,values));kind=identity['calibration_kind'];budget=identity['budget']
        expected=200 if kind=='target_real' else 1
        if kind not in ('default','source_real','target_real') or len(q)!=expected:
            raise ValueError('calibration requires default/source plus200subsets per target budget')
        if kind=='target_real':
            if budget not in (10,20,40,80) or set(q['repeat'])!=set(range(200)):
                raise ValueError('registered CAL budgets and unique200repeats required')
        elif q['repeat'].iloc[0]!=-1:raise ValueError('source/default cannot be CAL replicates')
        if not set(q.status)<= {'ok','unavailable'}:raise ValueError('unknown calibration availability')
        for metric in METRICS:
            v=pd.to_numeric(q[metric],errors='raise').to_numpy(float) if metric in q else np.full(expected,np.nan)
            valid=np.isfinite(v)&q.status.eq('ok').to_numpy();complete=bool(valid.all())
            points.append(dict(**identity,comparison=kind,threshold_kind=kind,metric=metric,
                estimate=float(v.mean()) if complete else float('nan'),n_subsets=expected,
                n_valid_subsets=int(valid.sum()),subset_sample_sd=float(v.std(ddof=1)) if complete and expected>1 else float('nan'),
                subset_q025=float(np.quantile(v,.025)) if complete and expected>1 else float('nan'),
                subset_q975=float(np.quantile(v,.975)) if complete and expected>1 else float('nan'),
                status='ok' if complete else 'unavailable_subset_estimate'))
    point=pd.DataFrame(points);comparisons=[]
    for values,q in point.groupby(MODEL_KEYS+['metric'],sort=False,dropna=False):
        source=q[q.calibration_kind.eq('source_real')]
        if len(source)!=1:raise ValueError('one source threshold per model/flow/metric required')
        source_value=float(source.estimate.iloc[0])
        for row in q[q.calibration_kind.eq('target_real')].to_dict('records'):
            comparisons.append({**row,'comparison':'target_real_minus_source_real',
                'threshold_kind':'target_vs_source','estimate':row['estimate']-source_value,
                'primary_budget':row['budget']==20,'scope':'within-seed CAL mean minus same-seed source operating metric'})
    return point,pd.DataFrame(comparisons)


def _registry(config_path,backbone):
    if backbone not in BACKBONES:raise ValueError('unregistered D backbone')
    path=Path(config_path).resolve();config=json.loads(path.read_text());root=Path(config['artifact_root']).resolve()
    if path!=root/'configs/execution.json':raise ValueError('canonical frozen D execution config required')
    formal=config.get('formal_seeds',list(SEEDS))
    if len(formal)!=5 or set(formal)!=set(SEEDS):raise ValueError('exact five registered formal seeds required')
    tasks=[(run,t) for run,t in config.get('runs',{}).items() if t.get('phase')=='formal' and t.get('route','D')=='D' and t.get('backbone')==backbone]
    if len(tasks)!=5 or set(t.get('seed') for _,t in tasks)!=set(SEEDS):raise ValueError('exact five registered D formal seeds required')
    if any(t.get('recipe_frozen') is not True for _,t in tasks):raise ValueError('frozen formal D recipes required')
    runs={t['seed']:run for run,t in tasks}
    if len(runs)!=5:raise ValueError('duplicate D formal seed')
    return root,config,{s:runs[s] for s in SEEDS}


def summary_status(config_path,backbone):
    root,config,runs=_registry(config_path,backbone);rows=[]
    for seed,run in runs.items():
        folder=root/'analysis/D'/run;marker=folder/'complete.json'
        _safe_path(folder,root,directory=True,missing=True);_safe_path(marker,root,missing=True)
        present=marker.is_file() and json.loads(marker.read_text()).get('status') in ('ANALYSIS_COMPLETE','ANALYSIS_COMPLETE_WITH_NUMERIC_NA')
        rows.append(dict(seed=seed,run_id=run,analysis_marker_present=present))
    count=sum(x['analysis_marker_present'] for x in rows)
    return dict(status='D_ANALYSIS_MARKERS_PRESENT_VERIFICATION_REQUIRED' if count==5 else 'WAITING_D_FORMAL_ANALYSES',
        backbone=backbone,expected_training_seeds=5,analysis_markers_present=count,runs=rows,marker_only=True,
        GPU_launch_attempted=False,limitation='Inventory only; run verifies all current cache and analysis provenance.')


TABLES=('per_seed_metrics','five_seed_metrics','per_seed_effects','five_seed_effects',
    'per_seed_calibration','five_seed_calibration','per_seed_calibration_comparisons','five_seed_calibration_comparisons',
    'per_seed_oracles','five_seed_oracles','per_seed_numerical','five_seed_numerical')


def _input_inventory():
    names={'identity.json','training_reference.json','training_radii.npy','independent_rank_recomputation.json','limitations.json'}
    for role in ROLES:
        names.update(f'{role}/{n}' for n in ('frame_margins.parquet','video_scores.csv','metrics.csv','effects.csv','control_video_scores.csv',
            'video_diagnostics.csv','method_diagnostics.csv','numerical_audits.csv','pair_transition_ledgers.json','closure_bootstrap.json'))
        names.update(f'{role}/normal_bootstrap_{r}_{a}.json' for r in REPRESENTATIONS for a in AGGREGATIONS)
    names.update(f'rq4/{r}_frame_margins.parquet' for r in ('source_cal','target_cal','target_test'));names.add('rq4/scope.json')
    for r in REPRESENTATIONS:
        for a in AGGREGATIONS:
            names.update(f'rq4/{r}_{a}/{n}' for n in ('source_cal_scores.csv','target_cal_scores.csv','target_test_scores.csv',
                'calibration.csv','independent_calibration_recompute.json','oracle.json','primary_n20_vs_source_bootstrap.json'))
    return names



def _safe_path(path,root,*,directory=False,missing=False):
    path=Path(path)
    if not path.resolve().is_relative_to(root):raise ValueError('unsafe path outside registered root')
    for current in (path,*path.parents):
        if current==root:break
        if current.is_symlink():raise ValueError('unsafe symlink in summary input/output/lock path')
    if not path.exists():
        if missing:return
        raise ValueError('missing completion snapshot or artifact')
    if not (path.is_dir() if directory else path.is_file()):raise ValueError('summary path must be regular file or directory')


@contextmanager
def exclusive_file(path):
    """Private summary lock: nofollow open, regular single-link inode, then PID."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    try:fd=os.open(path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
    except OSError as error:
        if error.errno==errno.ELOOP:raise ValueError('unsafe summary lock symlink') from error
        raise
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('unsafe nonregular or aliased summary lock')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise RuntimeError('D summary lock is already held') from error
        os.ftruncate(fd,0);os.write(fd,str(os.getpid()).encode());yield
    finally:os.close(fd)


def _entries(folder,root):
    if not folder.exists():return set()
    out=set()
    for path in folder.rglob('*'):
        _safe_path(path,root,directory=path.is_dir())
        if path.is_file():out.add(str(path.relative_to(folder)))
    return out


def _atomic_write(path,root,writer):
    """Create private temp in-root, then replace; never follow a destination link.

    Interrupted private temps are retained outside the committed output inventory
    in .summary_scratch; source artifacts are never removed or overwritten.
    """
    path=Path(path);_safe_path(path,root,missing=True)
    scratch=root/'analysis/D/.summary_scratch';_safe_path(scratch,root,directory=True,missing=True)
    scratch.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=scratch,prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        writer(stream);stream.flush();os.fsync(stream.fileno());temporary=Path(stream.name)
    _safe_path(path,root,missing=True);temporary.replace(path)


def _json(path,root,value):
    def write(stream):
        json.dump(io._jsonable(value),stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    _atomic_write(path,root,write)


def _accept(config_path,root,run,backbone,seed):
    """Explicitly read-only verification; cannot invoke scoring or create files."""
    from . import cache_d,bootstrap_fast
    folder=root/'analysis/D'/run;marker=folder/'complete.json'
    if not marker.exists() and not marker.is_symlink():raise ValueError('incomplete D analyses; all five registered seeds required')
    _safe_path(folder,root,directory=True);_safe_path(marker,root)
    raw=marker.read_bytes();snapshot=hashlib.sha256(raw).hexdigest();receipt=json.loads(raw)
    if (receipt.get('status') not in ('ANALYSIS_COMPLETE','ANALYSIS_COMPLETE_WITH_NUMERIC_NA')
        or receipt.get('independent_recompute')!='PASSED_FOR_ALL_REPORTABLE_CELLS'
        or receipt.get('run_id')!=run or receipt.get('seed')!=seed or receipt.get('backbone')!=backbone):
        raise ValueError('D completed analysis status/model identity changed')
    inventory=_input_inventory();actual=_entries(folder,root)-{'complete.json'}
    if set(receipt.get('artifact_sha256',{}))!=inventory or actual!=inventory:raise ValueError('D analysis artifact inventory is incomplete or changed')
    # Mirror the existing producer identity without calling its dual-purpose run
    # function. Existing cache/context validators perform all live read-only checks.
    base,config,task=cache_d._context(config_path,run)
    if Path(base['artifact_root'])!=root or task['seed']!=seed or task['backbone']!=backbone:raise ValueError('D live context changed')
    cache=root/'features/D'/run;cache_marker=cache/'complete.json';_safe_path(cache_marker,root)
    cache_hash=digest(cache_marker)
    if json.loads(cache_marker.read_text()).get('status')!='CACHE_COMPLETE':raise ValueError('all seven D cache roles must be complete')
    freeze_path=root/'data/D/freeze.json';freeze=json.loads(freeze_path.read_text())
    if freeze.get('status')!='FROZEN' or freeze.get('method_count')!=21:raise ValueError('frozen D registry required')
    for role in cache_d.ROLES:
        manifest=root/f'data/D/{role}.parquet'
        if digest(manifest)!=freeze['manifest_sha256'].get(manifest.name):raise ValueError('D frozen manifest hash changed')
    expected=dict(cache_context=base,cache_completion_sha256=cache_hash,D_freeze_sha256=digest(freeze_path),
        analysis_code_sha256=digest(io.__file__),scoring_code_sha256=digest(score.__file__),bootstrap_code_sha256=digest(bootstrap_fast.__file__),
        calibration_repeats=200,test_bootstrap_repeats=2000,resampling_seed=20261023)
    if receipt.get('identity')!=expected or json.loads((folder/'identity.json').read_text())!=expected:raise ValueError('D analysis live identity/hash changed')
    for role in cache_d.ROLES:cache_d.verify_cache(cache/role,identity=cache_d._role_identity(base,role,root/f'data/D/{role}.parquet'))
    if any(digest(folder/name)!=value for name,value in receipt['artifact_sha256'].items()):raise ValueError('D committed analysis artifact hash changed')
    current,_,_=cache_d._context(config_path,run)
    if current!=base or digest(cache_marker)!=cache_hash:raise ValueError('D live cache/context changed during acceptance')
    _safe_path(marker,root)
    if digest(marker)!=snapshot:raise ValueError('D completion snapshot changed during acceptance')
    return folder,receipt,snapshot


def _code_ids():
    from . import cache_d,bootstrap_fast
    return {Path(m.__file__).name:digest(m.__file__) for m in (io,score,stats_h,cache_d,bootstrap_fast)}|{Path(__file__).name:digest(__file__)}


def _tag(table,seed,backbone,domain,**extra):
    if 'seed' in table and not table.seed.eq(seed).all():raise ValueError('saved per-seed table identity changed')
    return table.assign(seed=seed,backbone=backbone,domain=domain,**extra)


def _numeric_summary(rows):
    keys=['backbone','domain','grouping','representation','aggregation','scale'];out=[]
    if 'scale' not in rows:rows=rows.assign(scale=np.nan)
    for values,q in rows.groupby(keys,dropna=False,sort=False):
        if len(q)!=5 or set(q.seed)!=set(SEEDS) or q.seed.duplicated().any():raise ValueError('five exact seeds per numeric audit required')
        item=dict(zip(keys,values));item.update(n_seeds=5,n_ranking_usable=int(q.ranking_usable.eq(True).sum()),
            n_precision_triggered=int(q.triggered.eq(True).sum()),n_negative_control_evaluable=int(q.negative_control_order_preserved.notna().sum()),
            n_negative_control_preserved=int(q.negative_control_order_preserved.eq(True).sum()),scope='descriptive training-seed counts, not significance')
        out.append(item)
    return pd.DataFrame(out)


def _collect(root,runs,backbone,accepted):
    metrics=[];effects=[];calibration=[];oracles=[];numerical=[];support={};test_hashes=set();subset_signatures={}
    freeze=json.loads((root/'data/D/freeze.json').read_text());methods=sorted(freeze['method_registry']['FS9']+freeze['method_registry']['FR12'])
    expected_cells={(g,r,a) for g in ('correct','collision') for r in REPRESENTATIONS for a in AGGREGATIONS}
    for seed,run in runs.items():
        folder,receipt,snapshot=accepted[seed];support[str(seed)]={'run_id':run,'analysis_complete_sha256':snapshot,
            'artifacts':{name:{'path':str(folder/name),'sha256':value} for name,value in receipt['artifact_sha256'].items()}}
        for role in ROLES:
            base=folder/role;m=pd.read_csv(base/'metrics.csv',float_precision='round_trip')
            if set(map(tuple,m[io.CELLS].to_numpy()))!=expected_cells or len(m)!=8:raise ValueError('eight frozen D metric cells required')
            io.verify_saved_metrics(io.read_scores(base/'video_scores.csv'),m,methods)
            metrics.append(_tag(m,seed,backbone,role));stored=pd.read_csv(base/'effects.csv',float_precision='round_trip')
            recomputed=score.d_effects(m,seed=seed).set_index('aggregation');stored=stored.set_index('aggregation')
            if set(stored.index)!=set(AGGREGATIONS) or stored.index.duplicated().any():raise ValueError('D effect aggregation inventory changed')
            for agg in AGGREGATIONS:
                for name in ('delta_correct','delta_collision','interaction'):io._equal(stored.loc[agg,name],recomputed.loc[agg,name],'same-seed '+name)
                for name in ('valid','reversal'):
                    if stored.loc[agg,name]!=recomputed.loc[agg,name]:raise ValueError('same-seed effect flag changed')
            effects.append(_tag(stored.reset_index(),seed,backbone,role))
            numerical.append(_tag(pd.read_csv(base/'numerical_audits.csv',float_precision='round_trip'),seed,backbone,role))
        for rep in REPRESENTATIONS:
            for agg in AGGREGATIONS:
                base=folder/'rq4'/f'{rep}_{agg}';q=pd.read_csv(base/'calibration.csv',float_precision='round_trip')
                test=io.read_scores(base/'target_test_scores.csv');cal=io.read_scores(base/'target_cal_scores.csv')
                test_hash=hashlib.sha256('\n'.join(sorted(test.video_uid)).encode()).hexdigest()
                if set(q.test_id_hash)!={test_hash}:raise ValueError('RQ4 calibration metrics must use one fixed TEST')
                test_hashes.add(test_hash)
                expected=score.calibration_subsets(cal.video_uid);target=q[q.calibration_kind.eq('target_real')]
                if len(target)!=800 or target.duplicated(['repeat','budget']).any():raise ValueError('all200CALsubsets at4budgets required')
                indexed=target.set_index(['repeat','budget']);signature=[]
                for row in expected.to_dict('records'):
                    actual=indexed.loc[(row['repeat'],row['budget'])]
                    ids=json.loads(actual.calibration_ids) if isinstance(actual.get('calibration_ids'),str) else []
                    if actual.status!=row['status'] or tuple(ids)!=tuple(row['calibration_ids']):raise ValueError('CAL subsets differ from frozen common nested selection')
                    signature.append((row['repeat'],row['budget'],row['status'],tuple(ids)))
                flow=(rep,agg)
                if flow in subset_signatures and subset_signatures[flow]!=signature:raise ValueError('CAL subsets differ between training seeds')
                subset_signatures[flow]=signature
                checks=json.loads((base/'independent_calibration_recompute.json').read_text())
                if len(checks)!=int(q.status.eq('ok').sum()) or any(x['status']!='PASSED' for x in checks):raise ValueError('independent calibration audit incomplete')
                calibration.append(_tag(q,seed,backbone,'target_test',representation=rep,aggregation=agg))
                oracle=json.loads((base/'oracle.json').read_text());criteria=('max_ba','fpr_le_0.05','fpr_le_0.10')
                if isinstance(oracle,dict):
                    if oracle.get('status')!='UNRESOLVED_NUMERIC_ORDERING':raise ValueError('unregistered oracle NA')
                    oracle=[dict(criterion=c,status='unresolved_numeric_ordering',threshold=None,**{m:np.nan for m in METRICS}) for c in criteria]
                if len(oracle)!=3 or {x['criterion'] for x in oracle}!=set(criteria):raise ValueError('three actual oracle criteria required')
                oracles.append(_tag(pd.DataFrame(oracle),seed,backbone,'target_test',representation=rep,aggregation=agg))
    if len(test_hashes)!=1:raise ValueError('RQ4 TEST pool differs across seeds/flows')
    return dict(metrics=pd.concat(metrics,ignore_index=True),effects=pd.concat(effects,ignore_index=True),
        calibration=pd.concat(calibration,ignore_index=True),oracles=pd.concat(oracles,ignore_index=True),
        numerical=pd.concat(numerical,ignore_index=True)),support


def _tables(data):
    m=data['metrics'];keys=['seed','backbone','domain','grouping','representation','aggregation']
    long=m.melt(id_vars=keys,value_vars=['pooled_auc','macro_auc'],var_name='metric',value_name='estimate')
    long=long.assign(comparison='cell',threshold_kind='ranking')
    ms=seed_summary(long,[c for c in keys if c!='seed']+['comparison','threshold_kind','metric'])
    cal,comp=calibration_tables(data['calibration']);groups=['backbone','domain','representation','aggregation','budget','comparison','threshold_kind','metric']
    cs=seed_summary(cal,groups);ds=seed_summary(comp,groups)
    oracle=data['oracles'];okeys=['seed','backbone','domain','representation','aggregation','criterion']
    for name in METRICS:
        if name not in oracle:oracle=oracle.assign(**{name:np.nan})
    olong=oracle.melt(id_vars=okeys,value_vars=list(METRICS),var_name='metric',value_name='estimate')
    olong=olong.rename(columns={'criterion':'comparison'}).assign(threshold_kind='oracle_diagnostic')
    os=seed_summary(olong,['backbone','domain','representation','aggregation','comparison','threshold_kind','metric'])
    return dict(per_seed_metrics=m,five_seed_metrics=ms,per_seed_effects=data['effects'],five_seed_effects=effect_summary(data['effects']),
        per_seed_calibration=cal,five_seed_calibration=cs,per_seed_calibration_comparisons=comp,five_seed_calibration_comparisons=ds,
        per_seed_oracles=oracle,five_seed_oracles=os,per_seed_numerical=data['numerical'],five_seed_numerical=_numeric_summary(data['numerical']))



def _revalidate(config_path,root,runs,backbone,identity):
    for seed,run in runs.items():
        folder,current,snapshot=_accept(config_path,root,run,backbone,seed)
        if snapshot!=identity['analysis_complete_sha256'][str(seed)]:
            raise ValueError('D inputs changed during summary')
    if _code_ids()!=identity['code_sha256'] or digest(config_path)!=identity['config_sha256']:
        raise ValueError('D code/config changed during summary')


def summarize_backbone(config_path,backbone):
    """Assemble one backbone only after five accepted analysis/caches exist."""
    root,config,runs=_registry(config_path,backbone)
    output=root/f'analysis/D/{backbone}/five_seed_summary'
    guarded=[root/'analysis',root/'analysis/D',root/f'analysis/D/{backbone}',output,root/'analysis/D/.locks']
    if not output.resolve().is_relative_to(root) or any(p.is_symlink() for p in guarded):
        raise ValueError('unsafe summary output symlink or path outside registered root')
    accepted={seed:_accept(config_path,root,run,backbone,seed) for seed,run in runs.items()}
    markers={str(seed):snapshot for seed,(_,_,snapshot) in accepted.items()}
    identity=dict(schema='D_FIVE_SEED_SUMMARY_IDENTITY_V1',backbone=backbone,seeds=list(SEEDS),
        config_sha256=digest(config_path),analysis_complete_sha256=markers,code_sha256=_code_ids())
    output=root/f'analysis/D/{backbone}/five_seed_summary';inventory={f'{name}.csv' for name in TABLES}|{'identity.json','supporting_artifacts.json','limitations.json'}
    lock=root/f'analysis/D/.locks/{backbone}_five_seed_summary.lock';_safe_path(lock,root,missing=True)
    with exclusive_file(lock):
        complete=output/'complete.json';_safe_path(complete,root,missing=True)
        if complete.exists():
            record=json.loads(complete.read_text());actual=_entries(output,root)-{'complete.json'}
            if record.get('schema')!='D_FIVE_SEED_SUMMARY_COMPLETE_V1' or record.get('status') not in ('SUMMARY_COMPLETE','SUMMARY_COMPLETE_WITH_NUMERIC_NA') or record.get('identity')!=identity:
                raise ValueError('D summary identity/status changed')
            if record.get('training_seed_count')!=5 or record.get('backbone')!=backbone:raise ValueError('D summary model identity changed')
            if set(record.get('artifact_sha256',{}))!=inventory or actual!=inventory:raise ValueError('D summary artifact inventory changed')
            if any(digest(output/n)!=h for n,h in record['artifact_sha256'].items()):raise ValueError('D summary artifact hash changed')
            _revalidate(config_path,root,runs,backbone,identity)
            return record
        if shutil.disk_usage(root).free<=50*1024**3:raise RuntimeError('D summary requires50GiB reserve; preserve existing artifacts')
        prior=output/'identity.json'
        if prior.exists() and json.loads(prior.read_text())!=identity:raise ValueError('D partial summary identity changed; preserve existing artifacts')
        present=_entries(output,root)
        if not present<=inventory or present and not prior.exists():raise ValueError('unrecognized D partial summary inventory')
        output.mkdir(parents=True,exist_ok=True);_json(prior,root,identity)
        data,support=_collect(root,runs,backbone,accepted);tables=_tables(data)
        for name,table in tables.items():_atomic_write(output/f'{name}.csv',root,lambda stream,q=table:io._csv(stream,q))
        _json(output/'supporting_artifacts.json',root,dict(models=support,
            conditional_intervals='Retained per seed in original normal/closure/bootstrap files; never average CI endpoints.',
            mechanisms='Original per-video/method diagnostics, controls and3x3ledgers remain bound by path/hash.',calibration_repeats=200,test_bootstrap_repeats=2000))
        _json(output/'limitations.json',root,dict(backbone=backbone,training_seed_count=5,sample_sd_ddof=1,
            inference='descriptive across five registered training seeds; no15-model IID or significance claim',
            missing='five-model mean/SD NA when any seed unavailable; same-seed reversal denominators explicit',
            calibration='200subsets reduced within model first; subset spread is distinct from training-seed SD',
            intervals='Pointwise per-seed conditional TEST intervals retained by bound reference, no new cross-seed CI',
            cohorts='source_test/target_full andRQ4target_test never pooled; oracle diagnostic uses actual vertices',
            independence='Frozen source/content/name-proxy scope only; no strict population NP guarantee',
            numerical_NA={str(seed):r['status']=='ANALYSIS_COMPLETE_WITH_NUMERIC_NA' for seed,(_,r,_) in accepted.items()}))
        _revalidate(config_path,root,runs,backbone,identity)
        hashes={name:digest(output/name) for name in inventory}
        result=dict(schema='D_FIVE_SEED_SUMMARY_COMPLETE_V1',status='SUMMARY_COMPLETE_WITH_NUMERIC_NA' if any(r['status']=='ANALYSIS_COMPLETE_WITH_NUMERIC_NA' for _,r,_ in accepted.values()) else 'SUMMARY_COMPLETE',
            utc=datetime.now(timezone.utc).isoformat(),identity=identity,artifact_sha256=hashes,backbone=backbone,training_seed_count=5)
        _json(complete,root,result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('status','run'))
    parser.add_argument('--config',required=True);parser.add_argument('--backbone',required=True,choices=BACKBONES);args=parser.parse_args()
    result=summary_status(args.config,args.backbone) if args.action=='status' else summarize_backbone(args.config,args.backbone)
    print(json.dumps(io._jsonable(result),ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
