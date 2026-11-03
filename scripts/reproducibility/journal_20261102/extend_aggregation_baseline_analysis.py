"""Extend saved-score aggregation identities without replacing prior reports."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT=Path('/home/z/workspace/deepfake_journal_20261102')
SEEDS={7,42,123,2026,3407}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    source=ROOT/'reports/D_partial_formal_metrics_current.csv'
    inputs={str(source.relative_to(ROOT)):sha(source)}
    registry=pd.read_csv(source,float_precision='round_trip')[['backbone','seed']].drop_duplicates().sort_values(['backbone','seed'])
    runs=[f'D_formal_{r.backbone}_seed{r.seed}' for r in registry.itertuples()]
    directory=ROOT/'reports'/f'aggregation_baselines_all_accepted_{len(runs)}'
    directory.mkdir(exist_ok=False)
    rows,frames,controls=[],[],[]
    ledger_count=0
    for run in runs:
        folder=ROOT/'analysis/D'/run;marker=folder/'complete.json'
        receipt=json.loads(marker.read_text())
        assert receipt['status']=='ANALYSIS_COMPLETE'
        assert receipt['independent_recompute']=='PASSED_FOR_ALL_REPORTABLE_CELLS'
        assert receipt['formal_trajectory_target_used'] is False
        inputs[str(marker.relative_to(ROOT))]=sha(marker)
        for domain in ['source_test','target_full']:
            base=folder/domain
            for name in ['metrics.csv','pair_transition_ledgers.json','numerical_audits.csv','frame_margins.parquet']:
                p=base/name;expected=receipt['artifact_sha256'][domain+'/'+name]
                assert sha(p)==expected;inputs[str(p.relative_to(ROOT))]=expected
            metrics=pd.read_csv(base/'metrics.csv',float_precision='round_trip').set_index(['grouping','representation','aggregation'])
            ledger=json.loads((base/'pair_transition_ledgers.json').read_text())
            for grouping in ['correct','collision']:
                a,b=[float(metrics.loc[(grouping,'raw',agg),'pooled_auc']) for agg in ['mean_logit','mean_probability']]
                c,d=[float(metrics.loc[(grouping,'l2',agg),'pooled_auc']) for agg in ['mean_logit','mean_probability']]
                raw_loss,l2_loss=a-b,c-d
                delta_logit,delta_prob=c-a,d-b
                assert abs((delta_prob-delta_logit)-(raw_loss-l2_loss))<1e-12
                for representation,before,after in [('raw',a,b),('l2',c,d)]:
                    selected=[x for x in ledger if x['grouping']==grouping and x['comparison']=='logit_to_probability'
                        and x.get('representation')==representation]
                    assert len(selected)==1
                    item=selected[0];counts=np.array(item['counts'],float);credit=np.array([0.,.5,1.])
                    assert counts.shape==(3,3) and np.isfinite(counts).all() and (counts>=0).all()
                    assert abs(counts.sum()-item['pair_mass'])<1e-8
                    assert abs(counts.sum(axis=1)@credit/item['pair_mass']-before)<1e-12
                    assert abs(counts.sum(axis=0)@credit/item['pair_mass']-after)<1e-12
                    ledger_count+=1
                rows.append(dict(run_id=run,backbone=receipt['backbone'],seed=receipt['seed'],domain=domain,grouping=grouping,
                    raw_logit_auc=a,raw_probability_auc=b,l2_logit_auc=c,l2_probability_auc=d,
                    raw_aggregation_loss_pp=100*raw_loss,l2_aggregation_loss_pp=100*l2_loss,
                    l2_gain_logit_pp=100*delta_logit,l2_gain_probability_pp=100*delta_prob,
                    gain_change_due_to_aggregation_pp=100*(delta_prob-delta_logit),
                    l2_probability_minus_raw_logit_pp=100*(d-a)))
            numeric=pd.read_csv(base/'numerical_audits.csv',float_precision='round_trip')
            negative=numeric[numeric.negative_control_order_preserved.notna()]
            assert negative.negative_control_order_preserved.eq(True).all()
            assert numeric.stable.eq(True).all() and numeric.ranking_usable.eq(True).all()
            controls.append(dict(run_id=run,domain=domain,rows=len(numeric),negative_controls_preserved=len(negative)))
            margins=pd.read_parquet(base/'frame_margins.parquet',columns=['raw_logit','l2_logit'])
            for representation in ['raw','l2']:
                z=margins[representation+'_logit'].to_numpy(float);assert np.isfinite(z).all();p=expit(z)
                frames.append(dict(run_id=run,backbone=receipt['backbone'],seed=receipt['seed'],domain=domain,
                    representation=representation,n_frames=len(z),abs_logit_p50=np.quantile(abs(z),.5),
                    abs_logit_p95=np.quantile(abs(z),.95),abs_logit_max=abs(z).max(),
                    fp64_probability_exact_zero=int((p==0).sum()),fp64_probability_exact_one=int((p==1).sum()),
                    first_order_sigmoid_rmse=float(np.sqrt(np.mean((p-(.5+.25*z))**2)))))
        print(json.dumps(dict(run=run,status='IDENTITY_AND_PAIR_LEDGERS_VERIFIED')),flush=True)
    losses=pd.DataFrame(rows);frame_stats=pd.DataFrame(frames)
    assert len(losses)==len(frame_stats)==4*len(runs) and ledger_count==8*len(runs)
    fields=['raw_aggregation_loss_pp','l2_aggregation_loss_pp','l2_gain_logit_pp','l2_gain_probability_pp',
        'gain_change_due_to_aggregation_pp','l2_probability_minus_raw_logit_pp']
    summaries=[]
    for bone,q in losses.groupby('backbone',sort=False):
        if set(q.seed)!=SEEDS:continue
        marker=ROOT/'analysis/D'/bone/'five_seed_summary/complete.json';x=json.loads(marker.read_text())
        assert x['status']=='SUMMARY_COMPLETE' and x['training_seed_count']==5
        inputs[str(marker.relative_to(ROOT))]=sha(marker)
        for (domain,grouping),part in q.groupby(['domain','grouping'],sort=False):
            assert len(part)==5 and set(part.seed)==SEEDS
            row=dict(backbone=bone,domain=domain,grouping=grouping,n_training_seeds=5)
            for field in fields:
                row[field+'_mean']=float(part[field].mean());row[field+'_sample_sd']=float(part[field].std(ddof=1))
            row['l2_probability_above_raw_logit_seed_count']=int(part.l2_probability_minus_raw_logit_pp.gt(0).sum())
            summaries.append(row)
    outputs={'aggregation_baseline_loss_per_model.csv':losses,'aggregation_frame_numerics_per_model.csv':frame_stats,
        'complete_five_seed_aggregation_baselines.csv':pd.DataFrame(summaries)}
    hashes={}
    for name,data in outputs.items():
        data.to_csv(directory/name,index=False,float_format='%.17g');hashes[name]=sha(directory/name)
    assert all(sha(ROOT/name)==h for name,h in inputs.items())
    result=dict(status='PASS_EXTENDED_EXACT_IDENTITY_SAVED_LEDGERS_AND_NUMERICAL_CONTROLS',
        created=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),models=runs,model_count=len(runs),
        per_model_rows=len(losses),frame_numeric_rows=len(frame_stats),pair_ledger_comparisons=ledger_count,
        negative_controls_preserved=sum(q['negative_controls_preserved'] for q in controls),
        complete_five_seed_backbones=sorted({q['backbone'] for q in summaries}),
        input_sha256=inputs,output_directory=str(directory.relative_to(ROOT)),output_sha256=hashes,
        producer_sha256=sha(__file__),original_high_precision_ranking_AUC_preserved=True,
        interpretation='Exact descriptive AUC identity; FP64 endpoint counts are numerical diagnostics, not a unique causal attribution to sigmoid saturation.',
        formula='delta_prob = delta_logit + (Raw_mean_logit_AUC - Raw_mean_prob_AUC) - (L2_mean_logit_AUC - L2_mean_prob_AUC)',
        full_goal_complete=False,complete_all15_models=len(runs)==15)
    (ROOT/'operations'/f'aggregation_baselines_{len(runs)}_models_acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','model_count','pair_ledger_comparisons','negative_controls_preserved','complete_five_seed_backbones']},indent=2),flush=True)


if __name__=='__main__':main()
