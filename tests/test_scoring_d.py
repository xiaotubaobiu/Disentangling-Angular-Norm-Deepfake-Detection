"""Synthetic protocol regressions; these are not detector experiment results."""
import math
import unittest

import mpmath as mp
import numpy as np
import pandas as pd

from journal_detector import scoring_d as s


def fixture():
    rows, features = [], []
    # Hand-derived Raw means: real a=0, b=2, fake m1/x=1, m2/x=3.
    for label, method, video, value, group in [
        (0, 'real', 'a', 0., 'g1'), (0, 'real', 'b', 2., 'g2'),
        (1, 'm1', 'x', 1., 'g1'), (1, 'm2', 'x', 3., 'g2')]:
        for k in range(8):
            rows.append(dict(image_path=f'/{method}/{video}/{k}', label=label,
                method_id=method, video_id=video, video_uid=f'ff/{method}/{video}',
                group_id=group, domain='ff', role='source_test',
                rgb_sha256=f'{method}-{video}-{k}'))
            features.append([value, 1.])
    return pd.DataFrame(rows), np.asarray(features, dtype=np.float32)


class ScoringDTests(unittest.TestCase):
    def test_fp32_cache_fp64_head_difference_and_epsilon(self):
        f, h = fixture(); h[0] = 0
        out = s.frame_margins(f, h, np.array([[1., 2.], [3., 1.]], np.float32),
                              np.array([.25, .75], np.float32))
        self.assertEqual(out.raw_logit.dtype, np.dtype('float64'))
        self.assertEqual(out.iloc[0].l2_logit, .5)
        self.assertEqual(out.iloc[0].zero_norm, True)
        self.assertEqual(out.iloc[8].raw_logit, 3.5)
        with self.assertRaises(ValueError): s.frame_margins(f, h.astype('float64'), np.zeros((2, 2)), np.zeros(2))

    def test_eight_cells_collision_uses_all_16_frames_real_pool_unchanged(self):
        f, h = fixture()
        scores, metrics = s.score_eight_cells(f, h, np.array([[0., 0.], [1., 0.]], np.float32), np.zeros(2), fixed_methods=['m1', 'm2'])
        self.assertEqual(len(metrics), 8)
        raw = scores.query("representation == 'raw' and aggregation == 'mean_logit'")
        self.assertEqual(raw.query("grouping == 'correct' and label == 1").shape[0], 2)
        collision = raw.query("grouping == 'collision' and label == 1").iloc[0]
        self.assertEqual(collision.n_frames, 16); self.assertEqual(collision.score, 2.)
        self.assertEqual(raw.query("grouping == 'collision' and label == 0").shape[0], 2)
        self.assertEqual(metrics.query("grouping == 'correct' and representation == 'raw' and aggregation == 'mean_logit'").iloc[0].pooled_auc, .75)
        self.assertTrue(metrics.query("grouping == 'collision'").macro_auc.isna().all())

    def test_cross_class_id_collision_and_bad_frame_count_block(self):
        f,h=fixture(); f.loc[f.method_id.eq('m1'), 'video_id']='a'
        with self.assertRaisesRegex(ValueError, 'cross.class'): s.score_eight_cells(f,h,np.zeros((2,2)),np.zeros(2))
        f,h=fixture()
        with self.assertRaisesRegex(ValueError, '8'): s.score_eight_cells(f.iloc[:-1],h[:-1],np.zeros((2,2)),np.zeros(2))

    def test_pair_auc_ties_and_weighted_multiplicity(self):
        self.assertEqual(s.binary_auc([0,0,1,1], [0,2,1,3]), .75)
        self.assertEqual(s.binary_auc([0,1], [1,1]), .5)
        self.assertEqual(s.binary_auc([0,0,1], [0,2,1], [2,1,1]), 2/3)
        with self.assertRaises(ValueError): s.binary_auc([0,0],[0,1])

    def test_real_only_calibration_boundary_ties_and_probability_above_one(self):
        # At most one FP; nextafter(.8) excludes both tied .8 values.
        x=np.array([.1]*17+[.8,.8,1.])
        tau=s.empirical_fpr_threshold(x,.05)
        self.assertEqual(tau,np.nextafter(.8,np.inf));self.assertEqual(int((x>=tau).sum()),1)
        self.assertGreater(s.empirical_fpr_threshold([1.]*10,.05),1.)

    def test_nested_200_subsets_and_80_budget_single_unique(self):
        out=s.calibration_subsets([f'r{i}' for i in range(80)],seed=19)
        self.assertEqual(len(out),800)
        for repeat in range(200):
            records=out[out.repeat.eq(repeat)].set_index('budget')
            self.assertTrue(set(records.loc[10,'calibration_ids']) <= set(records.loc[20,'calibration_ids']))
            self.assertTrue(set(records.loc[40,'calibration_ids']) <= set(records.loc[80,'calibration_ids']))
        self.assertEqual(out.query('budget == 80').unique_set_count.iloc[0],1)
        short=s.calibration_subsets(['a']*2+['b'],budgets=(2,4),repeats=3)
        self.assertTrue(short.query('budget == 4').status.eq('unavailable').all())

    def test_metrics_pooled_ba_differs_macro_and_missing_fixed_method_invalid(self):
        q=pd.DataFrame(dict(label=[0,0,1,1,1,1],score=[0,2,3,3,3,0],method_id=['real','real','m1','m1','m1','m2']))
        r=s.metric_summary(q,threshold=1,fixed_methods=['m1','m2'])
        self.assertEqual(r['false_positives'],1);self.assertEqual(r['pooled_ba'],.625)
        self.assertEqual(r['macro_tpr'],.5);self.assertEqual(r['macro_ba'],.5)
        r=s.metric_summary(q[q.method_id.ne('m2')],fixed_methods=['m1','m2'])
        self.assertFalse(r['macro_valid']);self.assertTrue(math.isnan(r['macro_auc']))

    def test_oracle_uses_actual_vertices_and_all_unique_score_ties(self):
        q=pd.DataFrame(dict(label=[0,0,1,1],score=[.3,.3,.3,.8],method_id=['real','real','m','m']))
        out=s.oracle_operating_points(q)
        constrained=out.query("criterion == 'fpr_le_0.05'").iloc[0]
        self.assertEqual(constrained.fpr,0);self.assertEqual(constrained.tpr,.5)
        self.assertEqual(constrained.false_positives,0)
        self.assertEqual(out.query("criterion == 'max_ba'").iloc[0].pooled_ba,.75)

    def test_transition_three_by_three_reconstructs_exact_auc_change(self):
        out=s.pair_transition_ledger([0,0,1,1],[0,2,1,3],[2,0,1,1])
        self.assertEqual(out['counts'].shape,(3,3));self.assertEqual(int(out['counts'].sum()),4)
        self.assertEqual(out['auc_before'],.75);self.assertEqual(out['auc_after'],.5)
        self.assertEqual(out['delta_auc'],-.25);self.assertLessEqual(out['reconstruction_error'],1e-10)

    def test_high_precision_sigmoid_retains_extreme_order_and_escalates(self):
        groups=[np.array([1000.]*8),np.array([1001.]*8)]
        result=s.high_precision_aggregate(groups,aggregation='mean_probability',force=True)
        self.assertEqual(result['precision_history'],[80,160,320])
        # 320 decimal digits still cannot resolve 1-exp(-1000): no false success.
        self.assertFalse(result['stable']);self.assertFalse(result['ranking_usable'])
        stable=s.high_precision_aggregate([np.array([100.]*8),np.array([101.]*8)],force=True)
        self.assertTrue(stable['stable']);self.assertGreater(stable['scores'][1],stable['scores'][0])
        self.assertTrue(all(isinstance(x,mp.mpf) for x in stable['scores']))

    def test_controls_weighted_radius_and_complete_video_scaling(self):
        f,h=fixture(); frames=s.frame_margins(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2))
        report=s.controls_and_diagnostics(frames,training_radii=[1.,2.,10.],training_weights=[.1,.8,.1],fixed_methods=['m1','m2'])
        self.assertEqual(report['training_weighted_radius_median'],2.)
        scales=report['controls'].query("control == 'positive_scale'")
        self.assertEqual(set(scales.scale),{.1,1.})
        self.assertTrue((scales.n_frames>=8).all())
        self.assertIn('jensen_gap',report['video_diagnostics']);self.assertIn('local_affine_rmse',report['video_diagnostics'])

    def test_collision_closure_known_sources_and_repeat_draws(self):
        f,h=fixture(); closure=s.collision_closure(f)
        self.assertEqual(closure.closure_id.nunique(),1)
        sampled=s.resample_closure(f,closure,[closure.closure_id.iloc[0]]*2)
        self.assertEqual(len(sampled),64);self.assertEqual(sampled.draw_id.nunique(),2)
        self.assertEqual(sampled.groupby(['draw_id','video_uid']).size().tolist(),[8]*8)
        report=s.bootstrap_d_comparisons(s.frame_margins(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2)),repeats=5,fixed_methods=['m1','m2'])
        self.assertEqual(report['status'],'insufficient_closure_groups');self.assertIsNone(report['intervals'])

    def test_calibration_all_budgets_share_fixed_test_and_no_auc_on_source_real(self):
        source=pd.DataFrame(dict(video_uid=['s1','s2'],label=[0,0],score=[0.,1.],method_id=['real','real']))
        target=pd.DataFrame(dict(video_uid=['c1','c2'],label=[0,0],score=[0.,.5],method_id=['real','real']))
        test=pd.DataFrame(dict(video_uid=['t1','t2','f1','f2'],label=[0,0,1,1],score=[.1,.8,.4,.9],method_id=['real','real','m','m']))
        out=s.calibration_experiment(source,target,test,budgets=(1,2,4),repeats=3,fixed_methods=['m'])
        available=out[out.status.eq('ok')]
        self.assertEqual(available.test_id_hash.nunique(),1);self.assertTrue(available.n_real.eq(2).all())
        self.assertEqual(len(out.query("calibration_kind == 'source_real'")),1)
        self.assertTrue(out.query('budget == 4').status.eq('unavailable').all())

    def test_probability_cells_use_high_precision_auc_and_no_silent_float_rerank(self):
        f,h=fixture();h[:,0]=np.repeat([100.,102.,101.,103.],8)
        scores,metrics=s.score_eight_cells(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2))
        r=metrics.query("grouping == 'correct' and representation == 'raw' and aggregation == 'mean_probability'").iloc[0]
        self.assertEqual(r.pooled_auc,.75)
        v=scores.query("grouping == 'correct' and representation == 'raw' and aggregation == 'mean_probability'")
        self.assertTrue(v.high_precision_used.all());self.assertTrue(v.ranking_usable.all())
        h[:,0]+=1000
        _,metrics=s.score_eight_cells(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2))
        r=metrics.query("grouping == 'correct' and representation == 'raw' and aggregation == 'mean_probability'").iloc[0]
        self.assertTrue(math.isnan(r.pooled_auc));self.assertEqual(r.analysis_status,'unresolved_numeric_ordering')

    def test_delta_head_matches_original_two_output_head(self):
        f,h=fixture();w=np.array([[1.,2.],[3.,1.]],np.float32);b=np.array([.25,.75],np.float32)
        a=s.frame_margins(f,h,w,b)
        d=s.frame_margins_delta(f,h,w[1].astype('float64')-w[0].astype('float64'),float(b[1])-float(b[0]))
        np.testing.assert_array_equal(a.raw_logit,d.raw_logit);np.testing.assert_array_equal(a.l2_logit,d.l2_logit)

    def test_bootstrap_records_macro_invalid_draws_for_missing_fixed_method(self):
        f,h=fixture();frames=s.frame_margins(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2))
        frames.loc[frames.method_id.eq('m2'),'video_id']='y'
        report=s.bootstrap_d_comparisons(frames,repeats=20,seed=4,min_groups=2,max_group_fraction=1.,fixed_methods=['m1','m2'])
        self.assertEqual(report['status'],'ok');self.assertGreater(report['macro_invalid_repeats'],0)
        self.assertTrue(report['draws'].loc[~report['draws'].macro_valid,'mean_logit:macro_delta_correct'].isna().all())

    def test_cross_label_rgb_and_real_cal_test_overlap_block(self):
        f,h=fixture();f.loc[8,'rgb_sha256']=f.loc[0,'rgb_sha256']
        f.loc[8,'label']=1;f.loc[8,'method_id']='m'
        with self.assertRaises(ValueError): s.frame_margins(f,h,np.zeros((2,2)),np.zeros(2))
        real=pd.DataFrame(dict(video_uid=['r'],label=[0],score=[0.],method_id=['real']))
        test=pd.DataFrame(dict(video_uid=['r','f'],label=[0,1],score=[0.,1.],method_id=['real','m']))
        with self.assertRaises(ValueError):s.calibration_experiment(real,real,test,budgets=(1,),repeats=1)

    def test_source_real_only_aggregation_and_default_fixed_macro_bootstrap(self):
        f,h=fixture();mask=f.label.eq(0)
        frame=s.frame_margins(f[mask],h[mask],np.zeros((2,2)),np.zeros(2))
        out=s.aggregate_videos(frame,aggregation='mean_probability')
        self.assertEqual(len(out),2);self.assertTrue(out.label.eq(0).all())
        frames=s.frame_margins(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2))
        frames.loc[frames.method_id.eq('m2'),'video_id']='y'
        report=s.bootstrap_d_comparisons(frames,repeats=20,seed=4,min_groups=2,max_group_fraction=1.)
        self.assertGreater(report['macro_invalid_repeats'],0)

    def test_seed_effects_reversal_tolerance_and_mean_reversal_separate(self):
        metrics=pd.DataFrame([dict(grouping=g,representation=r,aggregation=a,pooled_auc=v)
            for a in ['mean_logit','mean_probability']
            for g,r,v in [('correct','raw',.7),('correct','l2',.6),('collision','raw',.4),('collision','l2',.5)]])
        effects=s.d_effects(metrics,seed=7)
        self.assertTrue(effects.reversal.all())
        self.assertAlmostEqual(effects.iloc[0].delta_correct,-.1)
        self.assertAlmostEqual(effects.iloc[0].interaction,.2)
        rows=[]
        for seed in [7,42,123,2026,3407]:rows.append(s.d_effects(metrics,seed=seed))
        summary=s.summarize_seed_effects(pd.concat(rows))
        self.assertTrue(summary.n_seed_reversals.eq(5).all());self.assertTrue(summary.mean_reversal.all())
        self.assertTrue(summary.complete_five_seeds.all());self.assertTrue(summary.interaction_sample_sd.eq(0).all())

    def test_positive_scale_controls_verify_order_and_training_weights_reject_target_radius(self):
        f,h=fixture();frames=s.frame_margins(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2))
        report=s.controls_and_diagnostics(frames,training_radii=[1,2],training_weights=[1,1])
        audit=report['numerical_audits'].query("aggregation == 'scaled_mean_logit'")
        self.assertEqual(len(audit),8);self.assertTrue(audit.negative_control_order_preserved.all())
        with self.assertRaises(ValueError):s.weighted_median([1,2],[-1,1])

    def test_normal_source_group_bootstrap_retains_multidraw_and_fixed_methods(self):
        q=pd.DataFrame(dict(video_uid=['r1','f1','r2','f2'],group_id=['a','a','b','b'],
            label=[0,1,0,1],score=[0.,1.,2.,3.],method_id=['real','m1','real','m2']))
        report=s.bootstrap_video_metrics(q,repeats=20,seed=4,min_groups=2,max_group_fraction=1.)
        self.assertEqual(report['status'],'ok');self.assertGreater(report['macro_invalid_repeats'],0)
        self.assertTrue(report['draws'].n_real.eq(2).all())
        self.assertTrue(report['draws'].n_fake.eq(2).all())

    def test_oracle_keeps_arbitrary_precision_roc_vertices(self):
        f,h=fixture();h[:,0]=np.repeat([100.,102.,101.,103.],8)
        scores,_=s.score_eight_cells(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2))
        q=scores.query("grouping == 'correct' and representation == 'raw' and aggregation == 'mean_probability'")
        oracle=s.oracle_operating_points(q)
        row=oracle.query("criterion == 'max_ba'").iloc[0]
        self.assertEqual(row.pooled_ba,.75)
        self.assertEqual(oracle.n_actual_vertices.iloc[0],5)
        self.assertEqual(row.operating_score_field,'rank_score')

    def test_bootstrap_refuses_unresolved_probability_intervals(self):
        f,h=fixture();h[:,0]=np.repeat([1100.,1102.,1101.,1103.],8)
        frames=s.frame_margins(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2))
        frames.loc[frames.method_id.eq('m2'),'video_id']='y'
        report=s.bootstrap_d_comparisons(frames,repeats=5,seed=4,min_groups=2,max_group_fraction=1.)
        affected=report['intervals'].query("comparison.str.startswith('mean_probability:')",engine='python')
        self.assertTrue(affected.valid_repeats.eq(0).all());self.assertTrue(affected.lower.isna().all())

    def test_precision_safe_audit_serialization_roundtrip_keeps_order(self):
        f,h=fixture();h[:,0]=np.repeat([100.,102.,101.,103.],8)
        scores,metrics=s.score_eight_cells(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2))
        q=scores.query("grouping == 'correct' and representation == 'raw' and aggregation == 'mean_probability'")
        exported=s.precision_safe_score_table(q)
        self.assertNotEqual(exported.rank_score_decimal.iloc[0],exported.rank_score_decimal.iloc[1])
        restored=s.restore_rank_scores(exported)
        self.assertEqual(s.metric_summary(restored)['pooled_auc'],.75)

    def test_weighted_auc_and_ledger_match_independent_pair_enumeration(self):
        rng=np.random.default_rng(23)
        for _ in range(10):
            y=np.array([0]*5+[1]*7);a=rng.integers(0,4,len(y));b=rng.integers(0,4,len(y));w=rng.integers(1,4,len(y))
            denom=sum(w[i]*w[j] for i in range(5) for j in range(5,12))
            expected=sum(w[i]*w[j]*(1 if a[j]>a[i] else .5 if a[j]==a[i] else 0)
                for i in range(5) for j in range(5,12))/denom
            self.assertAlmostEqual(s.binary_auc(y,a,w),expected)
            ledger=s.pair_transition_ledger(y,a,b,w)
            self.assertLessEqual(ledger['reconstruction_error'],1e-10)

    def test_closure_weighted_bootstrap_equals_materialized_draw_multiplicity(self):
        f,h=fixture();frames=s.frame_margins(f,h,np.array([[0.,0.],[1.,0.]],np.float32),np.zeros(2))
        frames.loc[frames.method_id.eq('m2'),'video_id']='y'
        closure=s.collision_closure(frames);groups=sorted(closure.closure_id.unique())
        drawn=np.random.default_rng(4).choice(groups,len(groups),replace=True)
        sample=s.resample_closure(frames,closure,drawn)
        _,metrics=s.score_cached_frames(sample,fixed_methods=['m1','m2'])
        expected=s.d_effects(metrics).set_index('aggregation')
        report=s.bootstrap_d_comparisons(frames,repeats=1,seed=4,min_groups=2,max_group_fraction=1.)
        row=report['draws'].iloc[0]
        self.assertEqual(sum(row.closure_draw_counts.values()),2)
        for aggregation in ['mean_logit','mean_probability']:
            self.assertAlmostEqual(row[f'{aggregation}:delta_correct'],expected.loc[aggregation,'delta_correct'])
            self.assertAlmostEqual(row[f'{aggregation}:interaction'],expected.loc[aggregation,'interaction'])

    def test_empty_target_real_pool_reports_all_budgets_unavailable(self):
        real=pd.DataFrame(dict(video_uid=['s'],label=[0],score=[0.],method_id=['real']))
        target=real.iloc[:0].copy()
        test=pd.DataFrame(dict(video_uid=['r','f'],label=[0,1],score=[0.,1.],method_id=['real','m']))
        rows=s.calibration_experiment(real,target,test,budgets=(10,20),repeats=2)
        missing=rows.query("calibration_kind == 'target_real'")
        self.assertEqual(len(missing),4);self.assertTrue(missing.status.eq('unavailable').all())


if __name__=='__main__': unittest.main()
