"""Independent synthetic recomputation tests; no detector/model/target I/O."""
import unittest
import numpy as np
import pandas as pd
from reliability_v2 import scoring_d as s
from reliability_v2 import bootstrap_fast as f


def frames_fixture(groups=24, *, cross_collision=True, sparse_method=False):
    rows=[]
    for group in range(groups):
        for method in ('real','m1','m2'):
            if sparse_method and (method=='m2' and group!=groups-1): continue
            label=int(method!='real')
            video=('r'+str(group)) if not label else ('f'+str(group))
            if cross_collision and group==0 and method=='m2':video='distinct0'
            if cross_collision and group==1 and method=='m2':video='f0'
            uid=f'ff/{method}/{video}'
            for k in range(8):
                raw=(group%7-3)*.31+(0 if method=='real' else .37 if method=='m1' else .69)
                raw+=(-2.1 if k==0 else .3)+k*.05
                l2=raw/(1+(group%4)*.8)+(0 if method=='real' else -.2 if method=='m1' else .15)
                rows.append(dict(image_path=f'/{uid}/{k}',label=label,method_id=method,
                    video_id=video,video_uid=uid,group_id=f'g{group:04}',domain='ff',
                    role='source_test',rgb_sha256=f'{uid}/{k}',raw_logit=raw,l2_logit=l2))
    return pd.DataFrame(rows)


def materialized_video_reference(frames, *, repeats,seed,fixed_methods):
    """Independent explicit row repetition, baseline rank/pair scorer, same RNG."""
    closure=s.collision_closure(frames)
    groups=sorted(closure.closure_id.unique())
    mapping=closure.set_index('video_uid').closure_id.to_dict()
    videos,point=s.score_cached_frames(frames,fixed_methods=fixed_methods)
    videos['cid']=[mapping[uids[0]] for uids in videos.member_video_uids]
    rng=np.random.default_rng(seed);rows=[]
    for repeat in range(repeats):
        draw=rng.choice(groups,len(groups),replace=True)
        pieces=[videos[videos.cid.eq(group)] for group in draw]
        table=pd.concat(pieces,ignore_index=True)
        metrics=[]
        for (grouping,representation,aggregation),part in table.groupby(['grouping','representation','aggregation'],sort=False):
            row=s.metric_summary(part,fixed_methods=fixed_methods,allow_macro=grouping=='correct')
            row.update(grouping=grouping,representation=representation,aggregation=aggregation);metrics.append(row)
        metrics=pd.DataFrame(metrics)
        effect=s.d_effects(metrics).set_index('aggregation')
        row=dict(repeat=repeat,macro_valid=bool(metrics.query("grouping == 'correct'").macro_valid.all()))
        for aggregation in ('mean_logit','mean_probability'):
            for comparison in ('delta_correct','delta_collision','interaction'):
                row[f'{aggregation}:{comparison}']=effect.loc[aggregation,comparison]
            correct=metrics.query('grouping == "correct" and aggregation == @aggregation').set_index('representation')
            row[f'{aggregation}:macro_delta_correct']=correct.loc['l2','macro_auc']-correct.loc['raw','macro_auc']
        rows.append(row)
    return pd.DataFrame(rows)


class BootstrapFastTests(unittest.TestCase):
    def test_d_fast_matches_materialized_reaggregation_including_repeated_draws(self):
        frames=frames_fixture(4,cross_collision=True)
        out=f.bootstrap_d_fast(frames,repeats=12,seed=20261023,fixed_methods=['m1','m2'],min_groups=2,max_group_fraction=1.)
        closure=s.collision_closure(frames);groups=sorted(closure.closure_id.unique())
        rng=np.random.default_rng(20261023)
        repeats_found=False
        for repeat in range(12):
            chosen=rng.choice(groups,len(groups),replace=True)
            repeats_found |= len(set(chosen))<len(chosen)
            sample=s.resample_closure(frames,closure,chosen)
            _,metrics=s.score_cached_frames(sample,fixed_methods=['m1','m2'])
            expected=s.d_effects(metrics).set_index('aggregation')
            for aggregation in ('mean_logit','mean_probability'):
                for column in ('delta_correct','delta_collision','interaction'):
                    self.assertAlmostEqual(out['draws'].iloc[repeat][f'{aggregation}:{column}'],expected.loc[aggregation,column],places=10)
            self.assertEqual(sum(out['draws'].iloc[repeat].closure_draw_counts.values()),len(groups))
        self.assertTrue(repeats_found)

    def test_all_effects_and_pointwise_intervals_match_2000_explicit_row_draws(self):
        frames=frames_fixture(3,cross_collision=True)
        out=f.bootstrap_d_fast(frames,repeats=2000,seed=19,fixed_methods=['m1','m2'],min_groups=2,max_group_fraction=1.)
        expected=materialized_video_reference(frames,repeats=2000,seed=19,fixed_methods=['m1','m2'])
        for column in expected:
            if ':' not in column:continue
            np.testing.assert_allclose(out['draws'][column],expected[column],rtol=0,atol=1e-10,equal_nan=True)
            interval=out['intervals'].set_index('comparison').loc[column]
            self.assertAlmostEqual(interval.lower,expected[column].quantile(.025),places=10)
            self.assertAlmostEqual(interval.upper,expected[column].quantile(.975),places=10)

    def test_fixed_missing_method_and_conservative_closure_gates(self):
        frames=frames_fixture(4,cross_collision=False,sparse_method=True)
        out=f.bootstrap_d_fast(frames,repeats=30,seed=4,fixed_methods=['m1','m2'],min_groups=2,max_group_fraction=1.)
        self.assertGreater(out['macro_invalid_repeats'],0)
        self.assertTrue(out['draws'].loc[~out['draws'].macro_valid,'mean_logit:macro_delta_correct'].isna().all())
        blocked=f.bootstrap_d_fast(frames,repeats=2,fixed_methods=['m1','m2'])
        self.assertEqual(blocked['status'],'insufficient_closure_groups');self.assertIsNone(blocked['intervals'])
        enough=frames_fixture(24,cross_collision=False)
        concentrated=f.bootstrap_d_fast(enough,repeats=2,known_edges=[('g0000',f'g{i:04}') for i in range(1,14)])
        self.assertGreater(concentrated['largest_group_fraction'],.5);self.assertIsNone(concentrated['intervals'])

    def test_mpmath_order_survives_and_unresolved_cells_do_not_get_intervals(self):
        frames=frames_fixture(2,cross_collision=False)
        value={('real','g0000'):100.,('real','g0001'):102.,('m1','g0000'):101.,('m1','g0001'):103.,('m2','g0000'):101.,('m2','g0001'):103.}
        frames['raw_logit']=[value[(m,g)] for m,g in zip(frames.method_id,frames.group_id)]
        frames['l2_logit']=frames.raw_logit-100.
        out=f.bootstrap_d_fast(frames,repeats=3,min_groups=2,max_group_fraction=1.)
        row=out['point_cell_metrics'].query("grouping == 'correct' and representation == 'raw' and aggregation == 'mean_probability'").iloc[0]
        self.assertEqual(row.pooled_auc,.75)
        self.assertTrue(out['rank_metadata'].query("grouping == 'correct' and representation == 'raw' and aggregation == 'mean_probability'").high_precision_used.all())
        frames['raw_logit']+=1000
        unavailable=f.bootstrap_d_fast(frames,repeats=3,min_groups=2,max_group_fraction=1.)
        ci=unavailable['intervals'].set_index('comparison').loc['mean_probability:delta_correct']
        self.assertEqual(ci.valid_repeats,0);self.assertTrue(np.isnan(ci.lower))

    def test_normal_fast_matches_materialized_source_group_bootstrap(self):
        frames=frames_fixture(4,cross_collision=False,sparse_method=True)
        videos=s.aggregate_videos(frames)
        out=f.bootstrap_video_metrics_fast(videos,threshold=.2,repeats=25,seed=4,min_groups=2,max_group_fraction=1.)
        slow=s.bootstrap_video_metrics(videos,threshold=.2,repeats=25,seed=4,min_groups=2,max_group_fraction=1.)
        for name in ['pooled_auc','macro_auc','fpr','tpr','pooled_ba','macro_tpr','macro_ba','n_real','n_fake','false_positives']:
            np.testing.assert_allclose(out['draws'][name],slow['draws'][name],rtol=0,atol=1e-10,equal_nan=True)
        self.assertEqual(out['macro_invalid_repeats'],slow['macro_invalid_repeats'])

    def test_rq4_linear_acceptance_matches_200_sets_in_every_group_draw(self):
        q=pd.DataFrame(dict(video_uid=['r0','f0','r1','f1','r2','f2'],group_id=['g0','g0','g1','g1','g2','g2'],
            label=[0,1,0,1,0,1],method_id=['real','m','real','m','real','m'],score=[.1,.2,.4,.7,.9,.8]))
        thresholds=np.tile([.15,.5,.85,1.1],50)
        out=f.bootstrap_operating_comparison(q,.5,thresholds,repeats=20,seed=23,min_groups=2,max_group_fraction=1.)
        rng=np.random.default_rng(23)
        groups=sorted(q.group_id.unique())
        for repeat in range(20):
            draw=rng.choice(groups,len(groups),replace=True)
            table=pd.concat([q[q.group_id.eq(g)] for g in draw],ignore_index=True)
            source=s.metric_summary(table,.5)
            target=[s.metric_summary(table,t) for t in thresholds]
            for metric in ('fpr','tpr','pooled_ba'):
                expected=np.mean([r[metric] for r in target])-source[metric]
                self.assertAlmostEqual(out['draws'].iloc[repeat]['delta_'+metric],expected,places=10)
        self.assertEqual(len(out['calibration_spread']),200)
        self.assertEqual(out['calibration_repeats'],200)
        self.assertEqual(out['inference_scope'],'conditional_on_fixed_model_cal_pool_recorded_sets_and_known_test_groups')

    def test_rq4_tied_thresholds_probability_over_one_and_group_gate(self):
        q=pd.DataFrame(dict(video_uid=['r','f'],group_id=['g','g'],label=[0,1],method_id=['real','m'],score=[1.,1.]))
        thresholds=[1.,np.nextafter(1.,np.inf)]*100
        out=f.bootstrap_operating_comparison(q,1.,thresholds,repeats=3,min_groups=1,max_group_fraction=1.)
        self.assertEqual(out['point_estimates']['target_mean_fpr'],.5)
        self.assertEqual(out['point_estimates']['delta_fpr'],-.5)
        self.assertEqual(out['point_estimates']['delta_pooled_ba'],0.)
        gate=f.bootstrap_operating_comparison(q,1.,thresholds,repeats=3)
        self.assertEqual(gate['status'],'insufficient_source_groups');self.assertIsNone(gate['intervals'])

    def test_input_guards_and_real_only_bootstrap_draws(self):
        with self.assertRaises(ValueError):f.bootstrap_d_fast(frames_fixture(2),repeats=0)
        q=pd.DataFrame(dict(video_uid=['r','f'],group_id=['r','f'],label=[0,1],method_id=['real','m'],score=[0.,1.]))
        out=f.bootstrap_video_metrics_fast(q,repeats=20,seed=4,min_groups=2,max_group_fraction=1.)
        self.assertGreater(out['pooled_invalid_repeats'],0)
        with self.assertRaises(ValueError):f.bootstrap_operating_comparison(q,0.,[],repeats=2)
        with self.assertRaises(ValueError):f.bootstrap_operating_comparison(pd.concat([q,q]),0.,[.5],repeats=2)


if __name__=='__main__':unittest.main()
