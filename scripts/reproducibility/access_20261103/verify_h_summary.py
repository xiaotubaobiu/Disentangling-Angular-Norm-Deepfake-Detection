"""Independently reduce all saved three-pair H metrics and comparisons."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT=None
CODE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(CODE/'scripts/figures'))
from formal_H_figure_data import BACKBONES,SEEDS,digest,load_verified


def check(original,official,keys,wide):
    source_keys=keys[:-1] if wide else keys
    expected=set()
    if wide:
        for key in original[source_keys].itertuples(index=False,name=None):
            for metric in official.metric.unique():expected.add(tuple(key)+(metric,))
    else:
        expected=set(original[keys].itertuples(index=False,name=None))
    actual=list(official[keys].itertuples(index=False,name=None))
    assert len(actual)==len(set(actual)) and set(actual)==expected
    for row in official.itertuples(index=False):
        selected=original
        for key in source_keys:
            selected=selected[selected[key].eq(getattr(row,key))]
        assert len(selected)==3 and set(selected.seed)==set(SEEDS)
        values=selected[row.metric if wide else 'estimate'].to_numpy(float)
        assert not np.isinf(values).any()
        valid=np.isfinite(values);complete=bool(valid.all())
        assert row.n_seeds==3 and row.n_valid_seeds==int(valid.sum())
        assert bool(row.complete_three_seeds)==complete
        assert row.n_positive==int((values[valid]>0).sum())
        assert row.n_negative==int((values[valid]<0).sum())
        assert row.n_zero==int((values[valid]==0).sum())
        want=[float(values.mean()),float(values.std(ddof=1))] if complete else [np.nan,np.nan]
        np.testing.assert_allclose([row.estimate_mean,row.estimate_sample_sd],want,
            rtol=0,atol=1e-12,equal_nan=True)
    return len(official)


def main():
    global ROOT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backbone',required=True,choices=BACKBONES)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--worker-exit',type=Path,required=True);args=p.parse_args()
    ROOT=args.root.resolve()
    folder=ROOT/'analysis/H'/args.backbone/'three_seed_summary'
    marker=folder/'complete.json';snapshot=digest(marker)
    exit_path=args.worker_exit
    worker=json.loads(exit_path.read_text())
    assert worker['status']=='SUCCEEDED' and worker['exit_code']==0
    transaction_path=ROOT/'operations'/f'{args.backbone}_summary_digest_transaction.json'
    transaction=json.loads(transaction_path.read_text())
    assert transaction['status']=='PASS_SEALED_LARGE_FILE_DIGEST_TRANSACTION' and transaction['sealed'] is True
    assert transaction['summary_complete_sha256']==snapshot
    assert transaction['statistical_code_modified'] is False and transaction['model_or_score_modified'] is False
    assert transaction['original_summary_code_sha256']==digest(CODE/'src/journal_detector/summary_h.py')
    assert transaction['wrapper_sha256']==digest(Path(__file__).with_name('verified_h_summary.py'))
    assert transaction['digest_helper_sha256']==digest(Path(__file__).with_name('verified_digest.py'))
    data=load_verified(ROOT,[args.backbone])
    base=['backbone','domain','cohort','aggregation']
    m=check(data['metrics'],pd.read_csv(folder/'three_seed_metrics.csv',float_precision='round_trip'),
        base+['cell','threshold_kind','metric'],True)
    c=check(data['comparisons'],data['official_comparisons'],base+['comparison','threshold_kind','metric'],False)
    assert m==c==336 and digest(marker)==snapshot
    result=dict(status='PASS_STRICT_H_WORKER_AND_INDEPENDENT_ALL_TABLE_REDUCTIONS',
        created=datetime.now(timezone.utc).isoformat(),backbone=args.backbone,training_seed_pairs=list(SEEDS),
        strict_summary_sha256=snapshot,strict_worker_exit_sha256=digest(exit_path),
        sealed_digest_transaction_sha256=digest(transaction_path),
        summary_rows_checked=m+c,mean_and_sample_SD_fields_checked=2*(m+c),
        counts_signs_completeness_and_NA_checked=True,primary_and_sensitivity_contrasts_reconstructed=len(data['paired']),
        input_sha256=data['sources'],producer_sha256=digest(__file__),
        scope='Original strict worker owns live raw/cache validation. This check independently reduces all saved original metric/comparison summaries and validates figure-reader identities.',
        full_goal_complete=False)
    out=ROOT/'operations'/f'H_{args.backbone}_strict_table_acceptance.json'
    assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','backbone','summary_rows_checked','mean_and_sample_SD_fields_checked']},indent=2))


if __name__=='__main__':main()
