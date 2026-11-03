"""Apply the existing exact-rank grouping decomposition to accepted D models."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import runpy

import numpy as np
import pandas as pd
from journal_detector import trainer

ROOT = Path('/home/z/workspace/deepfake_journal_20261102')
SEEDS = {7,42,123,2026,3407}


def main():
    inputs = {}
    source = ROOT/'reports/D_partial_formal_metrics_current.csv'
    inputs[str(source.relative_to(ROOT))] = trainer.digest(source)
    frame = pd.read_csv(source, float_precision='round_trip')
    models = frame[['backbone','seed']].drop_duplicates().sort_values(['backbone','seed'])
    runs = [f'D_formal_{q.backbone}_seed{q.seed}' for q in models.itertuples()]
    original = ROOT/'operations/decompose_grouping_effect_rank_checked.py'
    inputs[str(original.relative_to(ROOT))] = trainer.digest(original)
    decompose = runpy.run_path(str(original))['decompose']
    out = ROOT/'reports'/f'grouping_decomposition_all_accepted_{len(runs)}'
    out.mkdir(exist_ok=False)
    rows, distributions = [], []
    for run in runs:
        marker = ROOT/'analysis/D'/run/'complete.json'
        receipt = json.loads(marker.read_text())
        assert receipt['status']=='ANALYSIS_COMPLETE'
        assert receipt['independent_recompute']=='PASSED_FOR_ALL_REPORTABLE_CELLS'
        assert receipt['formal_trajectory_target_used'] is False
        inputs[str(marker.relative_to(ROOT))] = trainer.digest(marker)
        for role in ['source_test','target_full']:
            part, counts = decompose(run,role,receipt)
            for row in part:
                row['backbone'] = receipt['backbone']
            rows.extend(part);distributions.extend(counts)
            for name in ['video_scores.csv','metrics.csv','effects.csv']:
                path = marker.parent/role/name
                inputs[str(path.relative_to(ROOT))] = trainer.digest(path)
        print(json.dumps(dict(run=run,status='EXACT_MEMBERSHIP_AND_RANK_CREDITS_VERIFIED',cell_rows=8)),flush=True)
    table = pd.DataFrame(rows)
    assert len(table)==8*len(runs)
    interactions=[]
    for (run,role,aggregation),q in table.groupby(['run_id','role','aggregation'],sort=False):
        q=q.set_index('representation')
        assert set(q.index)=={'raw','l2'}
        interaction=q.loc['l2','grouping_auc_change']-q.loc['raw','grouping_auc_change']
        rw=q.loc['l2','symmetric_reweight_contribution']-q.loc['raw','symmetric_reweight_contribution']
        av=q.loc['l2','symmetric_averaging_contribution']-q.loc['raw','symmetric_averaging_contribution']
        assert abs(rw+av-interaction)<1e-14
        interactions.append(dict(run_id=run,backbone=q.backbone.iloc[0],seed=int(q.seed.iloc[0]),role=role,
            aggregation=aggregation,grouping_by_representation_interaction=interaction,
            reweight_interaction=rw,averaging_interaction=av))
    interaction_table=pd.DataFrame(interactions)
    summaries=[]
    for bone,q in interaction_table.groupby('backbone',sort=False):
        if set(q.seed)!=SEEDS:
            continue
        marker=ROOT/'analysis/D'/bone/'five_seed_summary/complete.json'
        assert json.loads(marker.read_text())['training_seed_count']==5
        inputs[str(marker.relative_to(ROOT))]=trainer.digest(marker)
        for (role,agg),part in q.groupby(['role','aggregation'],sort=False):
            assert len(part)==5
            row=dict(backbone=bone,role=role,aggregation=agg,n_training_seeds=5)
            for key in ['grouping_by_representation_interaction','reweight_interaction','averaging_interaction']:
                row[key+'_mean_pp']=100*float(part[key].mean())
                row[key+'_sample_sd_pp']=100*float(part[key].std(ddof=1))
            np.testing.assert_allclose(row['grouping_by_representation_interaction_mean_pp'],
                row['reweight_interaction_mean_pp']+row['averaging_interaction_mean_pp'],atol=1e-12,rtol=0)
            summaries.append(row)
    outputs={'grouping_decomposition_cells.csv':table,'grouping_decomposition_interactions.csv':interaction_table,
        'collision_group_member_counts.csv':pd.DataFrame(distributions),
        'complete_five_seed_interaction_decomposition.csv':pd.DataFrame(summaries)}
    hashes={}
    for name,data in outputs.items():
        data.to_csv(out/name,index=False,float_format='%.17g');hashes[name]=trainer.digest(out/name)
    assert all(trainer.digest(ROOT/name)==h for name,h in inputs.items())
    record=dict(status='PASS_ALL_SELECTED_MODELS_EXACT_RANK_GROUPING_DECOMPOSITION',
        created=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),models=runs,model_count=len(runs),
        cell_rows=len(table),interaction_rows=len(interaction_table),
        complete_five_seed_backbones=sorted({x['backbone'] for x in summaries}),
        factorization='Member-count versus equal-group weighting; member scores versus group means; symmetric average of both intervention orders.',
        precision='400-decimal saved rank scores; exact rational AUC credits and contributions.',
        original_correct_and_collision_AUC_reconstructed=True,all_memberships_and_pair_credits_verified=True,
        scientific_inputs_unchanged=True,training_causal_attribution=False,input_sha256=inputs,
        output_directory=str(out.relative_to(ROOT)),output_sha256=hashes,producer_sha256=trainer.digest(__file__),
        complete_all15_models=len(runs)==15,full_goal_complete=False)
    trainer.save_json(ROOT/'operations'/f'grouping_decomposition_{len(runs)}_models_acceptance.json',record)
    print(json.dumps({k:record[k] for k in ['status','model_count','cell_rows','interaction_rows','complete_five_seed_backbones']},indent=2),flush=True)


if __name__=='__main__':
    main()
