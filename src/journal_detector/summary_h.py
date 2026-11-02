"""Read-only acceptance and descriptive summary of three matched H seed pairs."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from . import analysis_h as analysis
from . import analysis_d as io
from . import cache_h,stats_h,trainer

SEEDS=(7,42,123)
GROUPS=('backbone','domain','cohort','aggregation','comparison','threshold_kind','metric')


def seed_summary(rows,group_columns=None):
    """Three training pairs are the replicates; a missing seed keeps mean/SD NA."""
    groups=list(GROUPS if group_columns is None else group_columns)
    if rows.empty or not set(groups+['seed','estimate']).issubset(rows):raise ValueError('nonempty paired H estimates and identities required')
    out=[]
    for values,q in rows.groupby(groups,sort=False,dropna=False):
        if len(q)!=3 or set(q.seed)!=set(SEEDS) or q.seed.duplicated().any():raise ValueError('exact three (3) registered H seed pairs required')
        values=(values,) if len(groups)==1 else values
        estimate=pd.to_numeric(q.estimate,errors='raise').to_numpy(float)
        if np.isinf(estimate).any():raise ValueError('H estimate cannot be infinite')
        valid=np.isfinite(estimate);complete=bool(valid.all());row=dict(zip(groups,values))
        row.update(n_seeds=3,n_valid_seeds=int(valid.sum()),complete_three_seeds=complete,
            estimate_mean=float(estimate.mean()) if complete else float('nan'),
            estimate_sample_sd=float(estimate.std(ddof=1)) if complete else float('nan'),
            n_positive=int((estimate[valid]>0).sum()),n_negative=int((estimate[valid]<0).sum()),
            n_zero=int((estimate[valid]==0).sum()),scope='descriptive matched training-pair variation; no significance claim')
        out.append(row)
    return pd.DataFrame(out)


def _registry(config_path,backbone):
    path=Path(config_path).resolve();config=json.loads(path.read_text());root=Path(config['artifact_root']).resolve()
    allowed={root/'configs/h_execution.json',root/f'configs/h_execution_{backbone}.json'}
    if path not in allowed or backbone not in analysis.BACKBONES:raise ValueError('canonical registered H config/backbone required')
    cache_h._registry(config,backbone)
    return root,config


def summary_status(config_path,backbone):
    root,_=_registry(config_path,backbone)
    rows=[]
    for seed in SEEDS:
        marker=root/f'analysis/H/{backbone}/seed{seed}/complete.json'
        present=marker.is_file() and json.loads(marker.read_text()).get('status') in ('ANALYSIS_COMPLETE','ANALYSIS_COMPLETE_WITH_NUMERIC_NA')
        rows.append(dict(seed=seed,analysis_marker_present=present))
    return dict(status='H_ANALYSIS_MARKERS_PRESENT_VERIFICATION_REQUIRED' if all(r['analysis_marker_present'] for r in rows) else 'WAITING_H_FORMAL_ANALYSES',
        expected_training_seed_pairs=3,backbone=backbone,runs=rows,marker_only=True)


def _accept(config_path,backbone,seed):
    root,_,_,_,identity=analysis._pair_context(config_path,backbone,seed)
    folder=root/f'analysis/H/{backbone}/seed{seed}'
    receipt=analysis._verify_completed(folder,identity)
    if receipt is None:raise ValueError('all three formal H pair analyses must be complete')
    return folder,receipt,trainer.digest(folder/'complete.json')


def _collect(accepted,backbone):
    metrics=[];comparisons=[];support={};coverage=None
    for seed,(folder,receipt,snapshot) in accepted.items():
        support[str(seed)]=dict(pair_analysis_path=str(folder),pair_complete_sha256=snapshot,
            artifacts={name:dict(path=str(folder/name),sha256=sha) for name,sha in receipt['artifact_sha256'].items()})
        candidate=json.loads((folder/'coverage.json').read_text())
        if coverage is None:coverage=candidate
        elif candidate!=coverage:raise ValueError('H candidate coverage differs between seed pairs')
        for aggregation in ('mean_logit','mean_probability'):
            cal={h:io.read_scores(folder/f'{aggregation}/source_cal_{h}.csv') for h in ('B','AB')}
            for role,methods in (('source_test',analysis.SOURCE_METHODS),('target_test',analysis.TARGET_METHODS)):
                for cohort,cells in (('all_B',('H00','H10')),('common',('H00','H01','H10','H11'))):
                    base=folder/aggregation/role/cohort
                    videos={cell:io.read_scores(base/f'{cell}_scores.csv') for cell in cells}
                    normalized=stats_h.validate_cohort(videos,fixed_methods=methods)
                    if any(not q.aggregation.eq(aggregation).all() for q in normalized.values()):raise ValueError('H saved aggregation identity changed')
                    thresholds=stats_h._source_thresholds(cal,normalized['H00'],aggregation)[0]
                    saved_thresholds=json.loads((base/'thresholds.json').read_text())
                    if thresholds!=saved_thresholds:raise ValueError('H source thresholds changed or recalibrated on AB input')
                    m=pd.read_csv(base/'metrics.csv',float_precision='round_trip')
                    analysis.verify_saved_metrics(normalized,m,thresholds,fixed_methods=methods)
                    c=pd.read_csv(base/'comparisons.csv',float_precision='round_trip')
                    values={cell:{kind:part.iloc[0].to_dict() for kind,part in q.groupby('threshold_kind')}
                            for cell,q in m.groupby('cell')}
                    expected=pd.DataFrame(stats_h._comparison_rows(values,stats_h._comparison_terms(cohort=='common')))
                    key=['comparison','threshold_kind','metric']
                    if len(c)!=len(expected) or c.duplicated(key).any():raise ValueError('H paired comparison inventory changed')
                    actual=c.set_index(key).estimate;reference=expected.set_index(key).estimate
                    if set(actual.index)!=set(reference.index):raise ValueError('H paired comparison identities changed')
                    for name,want in reference.items():io._equal(actual.loc[name],want,'H same-seed paired comparison')
                    tags=dict(seed=seed,backbone=backbone,domain=role,cohort=cohort,aggregation=aggregation)
                    metrics.append(m.assign(**tags));comparisons.append(c.assign(**tags))
    return pd.concat(metrics,ignore_index=True),pd.concat(comparisons,ignore_index=True),coverage,support


def summarize_backbone(config_path,backbone):
    root,_=_registry(config_path,backbone)
    accepted={seed:_accept(config_path,backbone,seed) for seed in SEEDS}
    identity=dict(schema='JOURNAL_H_THREE_SEED_SUMMARY_IDENTITY_V1',backbone=backbone,seeds=list(SEEDS),
        config_sha256=trainer.digest(config_path),pair_complete_sha256={str(s):item[2] for s,item in accepted.items()},
        analysis_code_sha256=analysis._code_identities(),summary_code_sha256=trainer.digest(__file__))
    output=root/f'analysis/H/{backbone}/three_seed_summary'
    if not output.resolve().is_relative_to(root):raise ValueError('unsafe H summary output path')
    inventory={'identity.json','per_seed_metrics.csv','per_seed_comparisons.csv','three_seed_metrics.csv',
        'three_seed_comparisons.csv','coverage.json','supporting_artifacts.json','limitations.json'}
    def revalidate():
        for seed in SEEDS:
            if _accept(config_path,backbone,seed)[2]!=identity['pair_complete_sha256'][str(seed)]:raise ValueError('H pair changed during summary')
        if trainer.digest(config_path)!=identity['config_sha256'] or analysis._code_identities()!=identity['analysis_code_sha256'] or trainer.digest(__file__)!=identity['summary_code_sha256']:
            raise ValueError('H summary config/code identity changed')
    with trainer.exclusive_file(root/f'analysis/H/.locks/{backbone}_three_seed_summary.lock'):
        marker=output/'complete.json'
        if marker.is_file():
            receipt=json.loads(marker.read_text());actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and p!=marker}
            if (receipt.get('schema')!='JOURNAL_H_THREE_SEED_SUMMARY_COMPLETE_V1' or receipt.get('identity')!=identity
                    or receipt.get('status') not in ('SUMMARY_COMPLETE','SUMMARY_COMPLETE_WITH_NUMERIC_NA')
                    or receipt.get('training_seed_pairs')!=3 or set(receipt.get('artifact_sha256',{}))!=inventory or actual!=inventory):raise ValueError('H completed three-seed summary identity/inventory mismatch')
            if any(trainer.digest(output/name)!=sha for name,sha in receipt['artifact_sha256'].items()):raise ValueError('H summary artifact hash changed')
            revalidate();return receipt
        if shutil.disk_usage(root).free<=50*1024**3:raise RuntimeError('H summary requires 50GiB reserve')
        output.mkdir(parents=True,exist_ok=True);header=output/'identity.json'
        if header.exists() and json.loads(header.read_text())!=identity:raise ValueError('H partial three-seed summary identity changed')
        actual={str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()}
        if not actual.issubset(inventory) or actual and not header.exists():raise ValueError('unknown H partial summary artifacts')
        io._save_json(header,identity)
        m,c,coverage,support=_collect(accepted,backbone)
        io._csv(output/'per_seed_metrics.csv',m);io._csv(output/'per_seed_comparisons.csv',c)
        melted=m.melt(id_vars=['seed','backbone','domain','cohort','aggregation','cell','threshold_kind'],
            value_vars=list(stats_h.METRICS),var_name='metric',value_name='estimate').rename(columns={'cell':'comparison'})
        io._csv(output/'three_seed_metrics.csv',seed_summary(melted).rename(columns={'comparison':'cell'}))
        io._csv(output/'three_seed_comparisons.csv',seed_summary(c));io._save_json(output/'coverage.json',coverage)
        io._save_json(output/'supporting_artifacts.json',dict(models=support,
            conditional_intervals='Per-seed shared-group conditional intervals retained by original artifact hash; interval endpoints are never averaged.'))
        io._save_json(output/'limitations.json',dict(training_seed_pairs=3,model_count=6,sample_sd_ddof=1,
            inference='descriptive matched three-seed training variation; 2000 conditional TEST draws are not extra training replicates',
            primary='DFD all-B Raw mean-logit H10-H00',cohorts='all-B and common four cells remain separate',
            source_calibration='fixed B SOURCE_CAL per model reused on both input histories; no target calibration',
            target_development_used=False,missing='mean and sample SD remain NA if any registered seed estimate is unavailable',
            group_support='Real/fake/per-method independent group counts and sparse-group interval NA retained per seed'))
        revalidate();hashes={name:trainer.digest(output/name) for name in inventory}
        receipt=dict(schema='JOURNAL_H_THREE_SEED_SUMMARY_COMPLETE_V1',
            status='SUMMARY_COMPLETE_WITH_NUMERIC_NA' if any(item[1]['status']=='ANALYSIS_COMPLETE_WITH_NUMERIC_NA' for item in accepted.values()) else 'SUMMARY_COMPLETE',
            utc=datetime.now(timezone.utc).isoformat(),identity=identity,backbone=backbone,training_seed_pairs=3,
            target_development_used=False,artifact_sha256=hashes)
        io._save_json(marker,receipt);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('status','run'))
    parser.add_argument('--config',required=True);parser.add_argument('--backbone',required=True,choices=analysis.BACKBONES);args=parser.parse_args()
    result=summary_status(args.config,args.backbone) if args.command=='status' else summarize_backbone(args.config,args.backbone)
    print(json.dumps(io._jsonable(result),indent=2,allow_nan=False))
