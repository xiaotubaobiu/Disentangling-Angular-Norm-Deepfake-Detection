"""Small synthetic independent D saved-result oracle tests; no real payload I/O."""
import copy
from collections.abc import Mapping
from decimal import Decimal
from fractions import Fraction
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

SPEC=importlib.util.spec_from_file_location('d_numeric',Path(__file__).parents[1]/'tools/m_source_best_d_numeric.py')
d=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(d)

def table():
    return pd.DataFrame(dict(video_uid=['r0','r1','a0','a1','b0'],group_id=['g0','g1','g0','g1','g2'],
        label=[0,0,1,1,1],method_id=['real','real','a','a','b'],score=[0.,1.,0.,2.,1.],
        rank_score_decimal=['0','1','0','2','1'],ranking_usable=True))

def test_closed_fraction_weighted_ties_and_macro():
    q=table();o=d._RankOracle(q,['a','b'])
    result=o.evaluate(np.array([2,1,3]))
    # Real weights: 0 -> 2, 1 -> 1. Fake: 0 -> 2, 2 -> 1, 1 -> 3.
    expected=Fraction(2*2+1*6+3*5,2*3*6)
    assert result['pooled_auc']==float(expected)
    assert result['macro_auc']==float((Fraction(10,18)+Fraction(5,6))/2)
    assert result['n_real']==3 and result['n_fake']==6
    assert result['macro_valid'] is True

def test_integer_multiplicity_missing_method_and_class_are_distinct():
    o=d._RankOracle(table(),['a','b'])
    result=o.evaluate(np.array([1,1,0]))
    assert result['macro_valid'] is False and result['macro_auc'] is None
    assert result['missing_methods']==['b']
    with pytest.raises(d._AbsentClass):o.evaluate(np.array([0,0,3]))
    with pytest.raises(ValueError):o.evaluate(np.array([1.5,1,1]))
    with pytest.raises(ValueError):o.evaluate(np.array([True,False,True]))

def test_decimal_order_is_not_float_collapsed():
    q=table().iloc[[0,2]].copy();q['rank_score_decimal']=['1.0000000000000000000001','1.0000000000000000000002']
    q['score']=1.;q['group_id']=['g0','g1']
    o=d._RankOracle(q,['a'])
    assert o.evaluate(np.ones(2,dtype=int))['pooled_auc']==1.
    q['ranking_usable']=False
    assert d._RankOracle(q,['a']).evaluate(np.ones(2,dtype=int))['pooled_auc'] is None

@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):
    # Producer supplies fixtures only. The oracle never calls its numeric helpers.
    from reliability_v2 import source_best_stats as producer,source_best_cache as cache
    from test_source_best_stats import Sink,select
    methods=d.FIXED_METHODS
    data={};w=np.array([[0.,0.],[1.,0.]],np.float32);b=np.zeros(2,np.float32)
    for role in ('train','source_cal','source_test','target_full','target_cal','target_test'):
        rows=[];features=[];real_only=role in ('source_cal','target_cal')
        domain='ff' if role in ('train','source_cal','source_test') else 'cdf'
        n=24 if role!='target_cal' else 20
        for group in range(n):
            for method in (['real'] if real_only else ['real',methods[group%len(methods)]]):
                label=int(method!='real');uid=f'{domain}/{role}/{method}/{group}'
                for k in range(8):
                    rows.append(dict(image_path=f'/synthetic/{uid}/{k}',label=label,method_id=method,
                        video_id=('r' if not label else 'f')+str(0 if label and group==1 else group),video_uid=uid,group_id=f'g{group:03}',
                        domain=domain,role=role,rgb_sha256=f'{uid}/{k}'))
                    features.append([(group%5-2)*.4+label*.32+(k%2)*.12,1.+(group%3)*.1])
        frame=pd.DataFrame(rows)
        if role=='train':frame['sampling_mass']=1./len(frame)
        data[role]=(frame,np.array(features,np.float32),w,b)
    selected=select('D',history=None,head=cache._head_hash(w,b,2))
    freeze=dict(status='FROZEN',method_count=21,method_registry={'FS9':list(methods[:9]),'FR12':list(methods[9:])},
        target_scores_used=False,cdf_calibration_scope={'scope':'SYNTHETIC_PROXY_ONLY'})
    sink=Sink(tmp_path_factory.mktemp('d_numeric_fixture'))
    producer.d_driver(data,selected,freeze,sink)
    return sink.values

def test_full_2000_draws_and_200_cal_rows_are_independently_recomputed(synthetic):
    result=d.audit_D_artifacts(synthetic)
    assert result['numeric_core_status']=='PASS'
    assert result['bootstrap_artifacts']==14
    assert result['draws_recomputed']==28000
    assert result['calibration_rows_recomputed']==3208
    assert result['scientific_final_acceptance'] is False
    assert result['unverified_numeric_artifacts']

@pytest.mark.parametrize('name,mutate',[
    ('source_test/normal_bootstrap_raw_mean_logit.json',lambda x:x['draws'][17].__setitem__('pooled_auc',.12345)),
    ('target_full/closure_bootstrap.json',lambda x:x['draws'][37].__setitem__('mean_logit:interaction',.23456)),
    ('rq4/raw_mean_logit/primary_n20_vs_source_bootstrap.json',lambda x:x['draws'][29].__setitem__('delta_fpr',.34567)),
    ('source_test/normal_bootstrap_l2_mean_probability.json',lambda x:x['intervals'][0].__setitem__('valid_repeats',1999)),
    ('source_test/normal_bootstrap_raw_mean_probability.json',lambda x:x['draws'][0]['group_draw_counts'].__setitem__('g000',True)),
    ('target_full/closure_bootstrap.json',lambda x:x.__setitem__('min_groups',1)),
])
def test_saved_payload_corruption_is_rejected(synthetic,name,mutate):
    changed=dict(synthetic);changed[name]=copy.deepcopy(changed[name]);mutate(changed[name])
    with pytest.raises(ValueError):d.audit_D_artifacts(changed)

def test_CAL_single_ulp_threshold_and_missing_draw_rejected(synthetic):
    changed=dict(synthetic);name='rq4/l2_mean_logit/calibration.csv';changed[name]=changed[name].copy()
    changed[name].loc[0,'threshold']=np.nextafter(changed[name].loc[0,'threshold'],np.inf)
    with pytest.raises(ValueError):d.audit_D_artifacts(changed)
    changed=dict(synthetic);name='target_full/normal_bootstrap_l2_mean_logit.json';changed[name]=copy.deepcopy(changed[name]);changed[name]['draws'].pop()
    with pytest.raises(ValueError):d.audit_D_artifacts(changed)

def test_inventory_is_exact_and_lazy_mapping_does_not_require_values(synthetic):
    class Lazy(Mapping):
        def __init__(self):self.reads=[]
        def __iter__(self):return iter(synthetic)
        def __len__(self):return len(synthetic)
        def __getitem__(self,key):self.reads.append(key);return synthetic[key]
    values=Lazy();d.audit_D_artifacts(values)
    assert max(values.reads.count(name) for name in set(values.reads))<=1
    changed=dict(synthetic);changed.pop('training_radii.npy')
    with pytest.raises(ValueError):d.audit_D_artifacts(changed)

def test_oracle_has_no_producer_kernel_dependency():
    source=Path(d.__file__).read_text()
    import ast
    imported=[]
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Import):imported.extend(n.name for n in node.names)
        if isinstance(node,ast.ImportFrom):imported.append(node.module or '')
    assert not any(any(name in module for name in ('scoring_d','bootstrap_fast','source_best_stats','analysis_d')) for module in imported)



def test_normal_all_2000_includes_missing_classes_methods_and_numeric_NA():
    from reliability_v2 import bootstrap_fast as producer,analysis_d as wire
    q=pd.DataFrame(dict(video_uid=[f'v{i}' for i in range(20)],group_id=[f'g{i:02}' for i in range(20)],
        label=[0]*18+[1,1],method_id=['real']*18+['a','b'],score=[i%4 for i in range(20)],ranking_usable=True))
    obj=wire._jsonable(producer.bootstrap_video_metrics_fast(q,repeats=2000,seed=20261023,fixed_methods=['a','b']))
    audit=d._Audit();oracle=d._RankOracle(q,['a','b']);groups=oracle.groups
    d._bootstrap(audit,obj,{None:oracle},groups,[1]*20,repeats=2000,seed=20261023,path='synthetic_invalid_classes')
    assert audit.draws==2000 and obj['pooled_invalid_repeats']>0 and obj['macro_invalid_repeats']>obj['pooled_invalid_repeats']
    q['ranking_usable']=False
    obj=wire._jsonable(producer.bootstrap_video_metrics_fast(q,repeats=2000,seed=20261023,fixed_methods=['a','b']))
    oracle=d._RankOracle(q,['a','b']);audit=d._Audit()
    d._bootstrap(audit,obj,{None:oracle},oracle.groups,[1]*20,repeats=2000,seed=20261023,path='synthetic_numeric_NA')
    assert all(row['valid_repeats']==0 and row['lower'] is None for row in obj['intervals'])


def test_frozen_insufficient_group_and_concentration_gates():
    from reliability_v2 import bootstrap_fast as producer,analysis_d as wire
    q=table();oracle=d._RankOracle(q,['a','b']);obj=wire._jsonable(producer.bootstrap_video_metrics_fast(q,fixed_methods=['a','b']))
    audit=d._Audit();d._bootstrap(audit,obj,{None:oracle},oracle.groups,[2,2,1],repeats=2000,seed=20261023,path='small_groups')
    assert audit.draws==0
    obj['status']='ok'
    with pytest.raises(ValueError):d._bootstrap(d._Audit(),obj,{None:oracle},oracle.groups,[2,2,1],repeats=2000,seed=20261023,path='bad_gate')


def test_closure_graph_merges_cross_source_ID_and_RGB_edges_independently():
    q=pd.DataFrame(dict(video_uid=['a','b','c','d'],group_id=['g1','g2','g3','g4'],domain='ff',label=1,
        video_id=['collision','collision','other','unique'],rgb_sha256=['x','y','y','z']))
    rows=d._closure(q)
    assert [row['closure_id'] for row in rows]==['closure:g1','closure:g1','closure:g1','closure:g4']


def test_RQ4_exact_tie_boundary_can_exceed_probability_one():
    threshold=d._threshold([1.]*20)
    assert threshold==np.nextafter(1.,np.inf) and threshold>1.



@pytest.mark.parametrize('field',['score','member_video_uids','n_frames'])
def test_saved_video_score_is_bound_to_frame_margins(synthetic,field):
    name='source_test/video_scores.csv';changed=dict(synthetic);changed[name]=changed[name].copy(deep=True)
    if field=='score':changed[name].loc[0,field]=np.nextafter(changed[name].loc[0,field],np.inf)
    elif field=='member_video_uids':changed[name].at[0,field]=('nonexistent_video',)
    else:changed[name].loc[0,field]+=8
    with pytest.raises(ValueError):d.audit_D_artifacts(changed)


def test_library_inventory_matches_exact_producer_declaration():
    from reliability_v2 import source_best_sink
    assert d._inventory()==set(source_best_sink.declared_inventory('D'))



def test_CAL80_replays_all_nested_200_subsets_and_full_pool_one_unique_set(synthetic):
    from io import StringIO
    from reliability_v2 import scoring_d as producer,analysis_d as wire
    source=synthetic['rq4/raw_mean_logit/source_cal_scores.csv']
    test=synthetic['rq4/raw_mean_logit/target_test_scores.csv']
    target=source.iloc[[0]*80].copy().reset_index(drop=True)
    target['video_uid']=[f'synthetic_CAL80/{i:03}' for i in range(80)]
    target['score']=[(i//2)*.01 for i in range(80)]
    rows=producer.calibration_experiment(source,target,test,fixed_methods=sorted(d.FIXED_METHODS))
    buf=StringIO();wire._csv(buf,rows);buf.seek(0);rows=pd.read_csv(buf,float_precision='round_trip')
    audit=d._Audit();threshold,targets,native=d._calibration(audit,source,target,test,rows,sorted(d.FIXED_METHODS),'mean_logit',
        cal_repeats=200,seed=20261023,path='synthetic_CAL80')
    assert audit.cal_rows==802 and len(targets)==200 and len(native)==802
    all80=rows[(rows.calibration_kind=='target_real') & (rows.budget==80)]
    assert len(all80)==200 and (all80.unique_set_count==1).all()
    assert len(set(all80.threshold))==1



def _group_binding_fixture(*,cross_collision=False):
    rows=[]
    for group in range(2):
        for label in (0,1):
            method=('b' if cross_collision and group else 'a') if label else 'real';uid=f'ff/source_test/{method}/{group}'
            video_id=('fake' if label else 'real')+'/'+str(0 if label and cross_collision else group)
            for k in range(8):
                rows.append(dict(video_uid=uid,group_id=f'g{group}',domain='ff',role='source_test',label=label,
                    method_id=method,video_id=video_id,rgb_sha256=f'{uid}/{k}',raw_logit=group+2*label,l2_logit=group+label))
    frame=pd.DataFrame(rows);expected=d._frame_expected(frame,role='source_test')
    grouping='collision' if cross_collision else 'correct';key=(grouping,'raw','mean_logit')
    q=pd.DataFrame(expected[key].values())
    # Reconstruct producer metadata independently of the oracle's expected row.
    q['group_id']=q.member_group_ids.map(lambda values:values[0] if len(values)==1 else None)
    q['rank_score']=pd.Series(q.score.tolist(),dtype=object);q['ranking_usable']=True
    return frame,expected,key,q


def test_DNUM_001_correct_video_group_is_bound_to_frame_group():
    _,expected,key,q=_group_binding_fixture()
    d._bind_frame_scores(d._Audit(),q,expected,key,'natural_original')
    assert d._RankOracle(q,['a']).groups==('g0','g1')
    for replacement in ('video_uid',None,'',float('nan')):
        bad=q.copy(deep=True)
        bad['group_id']=bad.video_uid if replacement=='video_uid' else replacement
        with pytest.raises(ValueError,match='group_id'):
            d._bind_frame_scores(d._Audit(),bad,expected,key,'natural_modified_group_id')


def test_DNUM_001_cross_source_collision_group_uses_native_None_and_score_CSV_empty(tmp_path):
    from reliability_v2 import analysis_d as wire
    _,expected,key,q=_group_binding_fixture(cross_collision=True)
    collision=q.index[q.member_group_ids.map(len).eq(2)].item()
    assert q.loc[collision,'group_id'] is None
    d._bind_frame_scores(d._Audit(),q,expected,key,'collision_native_None')
    path=tmp_path/'collision_scores.csv';wire.write_scores(path,q);saved=wire.read_scores(path)
    # Production read_scores uses keep_default_na=False: CSV null is empty str.
    assert saved.loc[collision,'group_id']==''
    d._bind_frame_scores(d._Audit(),saved,expected,key,'collision_score_CSV_empty')
    decoded_NA=saved.copy(deep=True);decoded_NA.loc[collision,'group_id']=float('nan')
    d._bind_frame_scores(d._Audit(),decoded_NA,expected,key,'collision_decoded_DataFrame_NaN')
    bad=saved.copy(deep=True);bad.loc[collision,'group_id']='g0'
    with pytest.raises(ValueError,match='group_id'):
        d._bind_frame_scores(d._Audit(),bad,expected,key,'collision_modified_group_id')
