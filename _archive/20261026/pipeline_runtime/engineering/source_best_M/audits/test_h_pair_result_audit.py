import copy,csv,fcntl,hashlib,json,math,os,tempfile,unittest
from decimal import Decimal
from pathlib import Path
import numpy as np
import h_pair_result_audit as a


def rows(n=24,unusable=False):
    return [dict(video_uid=f'v{i:03}',group_id=f'g{i:03}',label=i%2,
                 method_id='real' if i%2==0 else 'fake',score=float(i%4),
                 rank=Decimal(i%4),ranking_usable=not unusable) for i in range(n)]

def brute(q,w,tau):
    r=[i for i,x in enumerate(q) if x['label']==0]; f=[i for i,x in enumerate(q) if x['label']==1]
    nr=sum(w[i] for i in r);nf=sum(w[i] for i in f)
    if not nr or not nf: raise ValueError('two classes')
    auc=sum(w[i]*w[j]*(2*(q[i]['rank']>q[j]['rank'])+(q[i]['rank']==q[j]['rank'])) for i in f for j in r)/(2*nr*nf)
    fp=sum(w[i]*(q[i]['score']>=tau) for i in r);tp=sum(w[i]*(q[i]['score']>=tau) for i in f)
    return dict(pooled_auc=auc,macro_auc=auc,n_real=nr,n_fake=nf,false_positives=fp,true_positives=tp,
        fpr=fp/nr,tpr=tp/nf,pooled_ba=(1-fp/nr+tp/nf)/2,macro_tpr=tp/nf,macro_ba=(1-fp/nr+tp/nf)/2)

class Numeric(unittest.TestCase):
    def test_integer_weighted_auc_matches_direct_pair_sum(self):
        q=rows(); c=a.IndependentCell(q,('fake',),np.arange(len(q)))
        rng=np.random.default_rng(17)
        for _ in range(30):
            w=rng.integers(0,4,len(q)); x=c.evaluate(w,2.)
            for k,v in brute(q,w,2.).items():a.close(x[k],v,k)
    def test_integer_count_comparison_is_exact(self):
        with self.assertRaises(AssertionError):a.close(2.00000000001,2,'integer count')
    def test_shared_group_expansion_matches_direct_pairs(self):
        q=rows();gi=np.arange(24)//3;d=np.arange(8,dtype=int)
        actual=a.IndependentCell(q,('fake',),gi).evaluate(d,2.)
        for k,v in brute(q,d[gi],2.).items():a.close(actual[k],v,k)
    def test_fixed_multimethod_macro_matches_direct_subsets(self):
        q=rows()
        for i,r in enumerate(q):
            if r['label']:r['method_id']='a' if i%4==1 else 'b'
        w=np.arange(24,dtype=int)%3+1;actual=a.IndependentCell(q,('a','b'),np.arange(24)).evaluate(w,2.)
        refs=[]
        for m in ('a','b'):
            inds=[i for i,r in enumerate(q) if r['label']==0 or r['method_id']==m]
            refs.append(brute([q[i] for i in inds],w[inds],2.))
        self.assertAlmostEqual(actual['macro_auc'],sum(x['pooled_auc'] for x in refs)/2)
        self.assertAlmostEqual(actual['macro_tpr'],sum(x['tpr'] for x in refs)/2)
    def test_threshold_tied_boundary_excludes_whole_tie(self):
        self.assertEqual(a.source_threshold([0.]*18+[1.,1.]),np.nextafter(1.,np.inf))
    def test_probability_saturation_boundary_exceeds_one(self):
        self.assertGreater(a.source_threshold([1.]*20),1.)
    def test_source_cal_empty_rejected(self):
        with self.assertRaises(ValueError):a.source_threshold([])
    def test_float_rank_uses_exact_binary_value(self):
        self.assertGreater(a.parse_rank('float64','0.1'),Decimal('0.1'))
    def test_mp_rank_does_not_collapse_small_difference(self):
        self.assertLess(a.parse_rank('mpmath','1.0000000000000000000000001'),a.parse_rank('mpmath','1.0000000000000000000000002'))
    def test_nonfinite_rank_rejected(self):
        with self.assertRaises(ValueError):a.parse_rank('mpmath','NaN')
    def test_unknown_rank_kind_rejected(self):
        with self.assertRaises(ValueError):a.parse_rank('bad','1')
    def test_missing_method_does_not_replace_fixed_macro(self):
        x=a.IndependentCell(rows(),('fake','absent'),np.arange(24)).evaluate(np.ones(24,dtype=int),0.)
        self.assertTrue(math.isnan(x['macro_auc']));self.assertFalse(x['macro_operating_valid']);self.assertTrue(math.isfinite(x['pooled_auc']))
    def test_unusable_rank_retains_operating_metrics(self):
        x=a.IndependentCell(rows(unusable=True),('fake',),np.arange(24)).evaluate(np.ones(24,dtype=int),2.)
        self.assertTrue(math.isnan(x['pooled_auc']));self.assertTrue(x['macro_operating_valid']);self.assertTrue(math.isfinite(x['macro_ba']))
    def test_zero_class_is_invalid(self):
        with self.assertRaises(ValueError):a.IndependentCell(rows(),('fake',),np.arange(24)).evaluate(np.array([1-i%2 for i in range(24)]),0.)
    def test_negative_effect_retained(self):
        x={k:{t:dict.fromkeys(a.METRICS,v) for t in a.KINDS} for k,v in [('H00',.8),('H10',.3)]}
        self.assertAlmostEqual(a.flatten(x)['default:comparison:H10-H00:pooled_auc'],-.5)
    def test_four_cell_effect_and_interaction_signs(self):
        x={k:{t:dict.fromkeys(a.METRICS,v) for t in a.KINDS} for k,v in [('H00',.4),('H01',.2),('H10',.3),('H11',.8)]}
        f=a.flatten(x)
        self.assertAlmostEqual(f['default:comparison:H01-H00:pooled_auc'],-.2)
        self.assertAlmostEqual(f['source_real:comparison:(H11-H10)-(H01-H00):pooled_auc'],.7)
    def test_na_effect_retained(self):
        x={k:{t:dict.fromkeys(a.METRICS,v) for t in a.KINDS} for k,v in [('H00',float('nan')),('H10',.3)]}
        self.assertTrue(math.isnan(a.flatten(x)['default:comparison:H10-H00:pooled_auc']))
    def test_na_mismatch_rejected(self):
        with self.assertRaises(AssertionError):a.close(None,0.,'NA')
    def test_nonfinite_not_accepted_as_finite(self):
        with self.assertRaises(AssertionError):a.close(float('inf'),1.,'inf')
    def test_linear_percentile(self):
        self.assertEqual(a.percentile([0.,10.],.025),.25)

class Bootstrap(unittest.TestCase):
    def fixture(self):
        q=rows();cells={'H00':q,'H10':[dict(x,score=-x['score'],rank=-x['rank']) for x in q]}
        cal={h:[dict(video_uid=f'cal{i:03}',group_id=f'calg{i:03}',label=0,method_id='real',score=float(i),rank=Decimal(i),ranking_usable=True) for i in range(20)] for h in ('B','AB')}
        doc=a.recompute_cohort(cells,cal,('fake',),'mean_logit',repeats=2000,seed=20261023)
        return cells,cal,doc
    def test_all_2000_exact_multiplicity_rng_and_intervals(self):
        cells,cal,doc=self.fixture(); summary=a.check_bootstrap(doc,cells,cal,('fake',),'mean_logit',2000,20261023)
        self.assertEqual(summary['draws_checked'],2000)
    def test_rng_saved_mapping_tamper_rejected(self):
        cells,cal,doc=self.fixture();doc['draws'][0]['group_draw_counts']={'g000':24}
        with self.assertRaises(AssertionError):a.check_bootstrap(doc,cells,cal,('fake',),'mean_logit',2000,20261023)
    def test_saved_draw_numeric_tamper_rejected(self):
        cells,cal,doc=self.fixture();doc['draws'][0]['default:cell:H00:pooled_auc']+=.01
        with self.assertRaises(AssertionError):a.check_bootstrap(doc,cells,cal,('fake',),'mean_logit',2000,20261023)
    def test_saved_interval_tamper_rejected(self):
        cells,cal,doc=self.fixture();doc['intervals'][0]['lower']=999
        with self.assertRaises(AssertionError):a.check_bootstrap(doc,cells,cal,('fake',),'mean_logit',2000,20261023)
    def test_missing_method_and_invalid_two_class_draws(self):
        q=rows(24)
        for i,r in enumerate(q):r.update(label=int(i in (0,1)),method_id=('fakeA' if i==0 else 'fakeB') if i in (0,1) else 'real')
        c={'H00':q,'H10':q};cal={h:[dict(x,video_uid='c'+x['video_uid'],label=0,method_id='real') for x in q] for h in ('B','AB')}
        d=a.recompute_cohort(c,cal,('fakeA','fakeB'),'mean_logit',repeats=2000)
        self.assertGreater(d['pooled_invalid_repeats'],0);self.assertGreater(d['macro_invalid_repeats'],d['pooled_invalid_repeats'])
        self.assertEqual(a.check_bootstrap(d,c,cal,('fakeA','fakeB'),'mean_logit')['draws_checked'],2000)
    def test_small_group_na_no_draws(self):
        q=rows(4);c={'H00':q,'H10':q};cal={h:[dict(x,video_uid='c'+x['video_uid'],label=0,method_id='real') for x in q] for h in ('B','AB')}
        d=a.recompute_cohort(c,cal,('fake',),'mean_logit',repeats=2000)
        self.assertEqual(d['status'],'insufficient_source_groups');self.assertEqual(d['draws'],[]);self.assertIsNone(d['intervals'])
    def test_cohort_identity_tamper_rejected(self):
        cells,cal,doc=self.fixture();cells['H10'][0]['group_id']='foreign'
        with self.assertRaises(ValueError):a.check_bootstrap(doc,cells,cal,('fake',),'mean_logit',2000,20261023)
    def test_cal_test_overlap_rejected(self):
        cells,cal,doc=self.fixture();cal['B'][0]['video_uid']=cells['H00'][0]['video_uid']
        with self.assertRaises(ValueError):a.check_bootstrap(doc,cells,cal,('fake',),'mean_logit',2000,20261023)

class Integrity(unittest.TestCase):
    def test_inventory_is_72_unique_artifacts(self):self.assertEqual(len(a.inventory()),72);self.assertEqual(len(set(a.inventory())),72)
    def test_lifetime_nonblocking_slot(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'slot1.lock';p.touch()
            with a.cpu_slot(Path(t),1):
                with self.assertRaises(RuntimeError):
                    with a.cpu_slot(Path(t),1):pass
            with a.cpu_slot(Path(t),1):pass
    def test_slot_missing_fail_closed(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError):
                with a.cpu_slot(Path(t),0):pass
    def test_slot_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'real').touch();(p/'slot0.lock').symlink_to(p/'real')
            with self.assertRaises(OSError):
                with a.cpu_slot(p,0):pass
    def test_default_entry_does_no_real_io(self):self.assertEqual(a.main([]),0)


class ArtifactAndCSV(unittest.TestCase):
    def fake_terminal(self,p):
        identity={'backbone':'toy','seed':42,'effective_precision':'AMP'}
        for name in a.inventory():
            f=p/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('{}')
        (p/'identity.json').write_text(json.dumps(identity))
        c=dict(schema='H_PAIR_ANALYSIS_COMPLETE_V1',status='ANALYSIS_COMPLETE',identity=identity,
            backbone='toy',seed=42,effective_precision='AMP',independent_recompute='PASSED_FOR_ALL_REPORTABLE_CELLS',
            artifact_sha256={n:a.sha(p/n) for n in a.inventory()})
        (p/'complete.json').write_text(json.dumps(c));return identity,a.sha(p/'complete.json')
    def test_terminal_exact_inventory_pass(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);i,h=self.fake_terminal(p);a.verify_terminal(p,i,h)
    def test_nested_extra_complete_must_not_hide(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);i,h=self.fake_terminal(p);(p/'extra').mkdir();(p/'extra/complete.json').write_text('{}')
            with self.assertRaises(AssertionError):a.verify_terminal(p,i,h)
    def test_artifact_drift_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);i,h=self.fake_terminal(p);(p/'coverage.json').write_text('changed')
            with self.assertRaises(AssertionError):a.verify_terminal(p,i,h)
    def test_terminal_status_not_complete_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);i,h=self.fake_terminal(p);c=a.read_json(p/'complete.json');c['status']='RUNNING';(p/'complete.json').write_text(json.dumps(c))
            with self.assertRaises(AssertionError):a.verify_terminal(p,i,a.sha(p/'complete.json'))
    def test_terminal_external_sha_drift_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);i,h=self.fake_terminal(p)
            with self.assertRaises(AssertionError):a.verify_terminal(p,i,'0'*64)
    def test_slot_released_on_error(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'slot0.lock').touch()
            with self.assertRaises(ValueError):
                with a.cpu_slot(p,0):raise ValueError('synthetic')
            with a.cpu_slot(p,0):pass
    def write_csv(self,p,**changes):
        row=dict(video_uid='v',label='0',method_id='real',group_id='g',score='0.10000000000000001',
            rank_score_kind='float64',rank_score_decimal='0.1',rank_score_export_digits=350,ranking_usable='True',
            representation='raw',grouping='correct',aggregation='mean_probability')
        row.update(changes)
        with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    def test_real_precision_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.csv';self.write_csv(p);r=a.read_scores(p,'mean_probability',True)[0]
            self.assertEqual(r['score'],.1);self.assertEqual(r['rank'],Decimal.from_float(.1))
    def test_csv_ambiguous_boolean_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.csv';self.write_csv(p,ranking_usable='1')
            with self.assertRaises(ValueError):a.read_scores(p,'mean_probability',True)
    def test_csv_cal_fake_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'s.csv';self.write_csv(p,label='1',method_id='fake')
            with self.assertRaises(AssertionError):a.read_scores(p,'mean_probability',True)
    def test_synthetic_saved_tables_end_to_end(self):
        cells,cal,doc=Bootstrap().fixture()
        _,groups,_,thresholds,evaluate=a.prepare(cells,cal,('fake',),'mean_logit');v=evaluate(np.ones(len(groups),dtype=np.int64))
        def safe(x):
            if isinstance(x,dict):return {k:safe(y) for k,y in x.items()}
            if isinstance(x,list):return [safe(y) for y in x]
            if isinstance(x,float) and not math.isfinite(x):return None
            return x
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'thresholds.json').write_text(json.dumps(thresholds));(p/'conditional_bootstrap.json').write_text(json.dumps(safe(doc)))
            metrics=[dict(cell=c,model_history=a.HISTORY[c],threshold_kind=k,**x) for c,kinds in v.items() for k,x in kinds.items()]
            with (p/'metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
            comp=[]
            for key,value in a.flatten(v).items():
                if ':comparison:' in key:
                    kind,_,name,metric=key.split(':',3);comp.append(dict(threshold_kind=kind,comparison=name,metric=metric,estimate=value))
            with (p/'comparisons.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(comp[0]));w.writeheader();w.writerows(comp)
            checks=[dict(cell=c,threshold_kind=k,status='PASSED',max_abs_error=0.) for c in cells for k in a.KINDS]
            (p/'independent_kernel_checks.json').write_text(json.dumps(checks))
            r=a.check_tables(p,cells,cal,('fake',),'mean_logit');self.assertEqual(r['draws_checked'],2000)
            self.assertIn('default:comparison:H10-H00:pooled_auc',r['independent_point_estimates'])
    def test_every_draw_has_direct_pair_reference(self):
        q=rows();c=a.IndependentCell(q,('fake',),np.arange(24));rng=np.random.default_rng(20261023)
        for _ in range(2000):
            w=np.bincount(rng.choice(24,size=24,replace=True),minlength=24)
            try:x=c.evaluate(w,2.)
            except ValueError:
                with self.assertRaises(ValueError):brute(q,w,2.)
            else:
                ref=brute(q,w,2.)
                for k,v in ref.items():a.close(x[k],v,k)

if __name__=='__main__':unittest.main()

