import importlib.util, json, pathlib, sys, traceback, os, copy
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
V=pathlib.Path('/home/z/workspace/deepfake_detector_publish_20261023/studies/pipeline_reliability/v2')
R=pathlib.Path('/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
sys.path.insert(0,str(V))
from reliability_v2 import qualification_h as q, data_h as d, geometry_h as g, trainer
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
tq=module('independent_tq',V/'tests/test_qualification_h.py')
out=R/'reports/H_data_qualification_review_logs/independent_latest'
out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
results=[]
def check(name,fn):
    try: results.append(dict(name=name,status='PASS',result=fn()))
    except Exception as e: results.append(dict(name=name,status='FAIL',error=repr(e),traceback=traceback.format_exc()))
def domain_boundary():
    pixels=out/'pixels';pixels.mkdir(exist_ok=True)
    frame=tq.source_rows(pixels, 'TRAIN','B').assign(domain='source')
    assert len(q._source(frame,'TRAIN','B'))==128
    for bad in ['ff','dfd','target']:
        try:q._source(frame.assign(domain=bad),'TRAIN','B')
        except ValueError:continue
        raise AssertionError('noncanonical domain accepted: '+bad)
    return dict(actual_geometry_domain='source',canonical_source_accepted=True,noncanonical_domains_rejected=True)
check('real_geometry_source_domain_boundary_fixed',domain_boundary)
def paired_probe():
    pixels=out/'paired';pixels.mkdir(exist_ok=True)
    b,ab=q._paired_source_probes(*(tq.source_rows(pixels,'TRAIN',v).assign(domain='source') for v in ['B','AB']),'TRAIN')
    assert len(b)==len(ab)==128 and b.mother_id.tolist()==ab.mother_id.tolist()
    assert b.label.value_counts().to_dict()=={0:64,1:64}
    counts=b.groupby('method_id').video_uid.nunique().to_dict()
    assert counts=={'real':8,**{m:2 for m in q.METHODS}}
    masses=b.groupby('method_id').sampling_mass.sum().to_dict()
    assert np.isclose(masses['real'],.5) and all(np.isclose(masses[m],.125) for m in q.METHODS)
    for epoch in [1,2]:
        bd,bm=trainer.epoch_draws(b,1007,epoch,128);ad,am=trainer.epoch_draws(ab,1007,epoch,128)
        q.assert_byte_equal(bd,ad);q.assert_byte_equal(bm,am)
        ba=[trainer.augmentation_parameters(1007,epoch,b.iloc[int(i)].mother_id,j) for j,i in enumerate(bd)]
        aa=[trainer.augmentation_parameters(1007,epoch,ab.iloc[int(i)].mother_id,j) for j,i in enumerate(ad)]
        q.assert_byte_equal(ba,aa)
    return dict(rows=128,counts=counts,masses=masses,paired_mothers_samples_methods_keyed_augmentations=True)
check('actual_weighted_paired_probe_sampling_contract',paired_probe)
def real_cal_no_auc():
    dev=pd.DataFrame([dict(video_uid='r',method_id='real',label=0,score=0.),*[dict(video_uid=m,method_id=m,label=1,score=1.) for m in q.METHODS]])
    cal=pd.DataFrame([dict(video_uid='cal',method_id='real',label=0,score=0.)])
    old=trainer.auc_independent;calls=[]
    def guarded(y,s):
        assert set(y)=={0,1},'real CAL reached AUC'
        calls.append(len(y));return old(y,s)
    trainer.auc_independent=guarded
    try:
        result=q.recompute_source_metrics(dev,cal)
        assert result['source_cal']['auc_called'] is False and len(calls)==5
        return dict(two_class_auc_calls=len(calls),cal_false_positives=result['source_cal']['false_positives'])
    finally:trainer.auc_independent=old
check('real_cal_never_auc',real_cal_no_auc)
def source_unused_bridge():
    rec=[]
    for role,key in [('SOURCE_UNUSED','x'),('TRAIN','a'),('DEV','b')]:
        rec.append(dict(video=dict(role=role,group_id=key,video_uid=key,label=1),
            frames=[dict(mother_rgb_sha256='bridge',**{v:dict(ok=False) for v in g.VERSIONS})]))
    report,quarantine=g.h_content_audit(rec)
    assert report['quarantined_videos']==3 and all(r['video']['group_id'] in quarantine for r in rec)
    assert report['source_group_role_conflicts']
    return dict(quarantined_videos=3,SOURCE_UNUSED_bridge_closed=True)
check('source_unused_bridge_quarantines_entire_connected_component',source_unused_bridge)
def actual_candidate_metadata():
    rows=json.loads((R/'data/H/H_candidates_all.json').read_text())
    assert len(rows)==len({r['video_uid'] for r in rows})==8431
    counts=Counter(r['role'] for r in rows)
    assert counts==dict(TRAIN=3060,DEV=540,SOURCE_CAL=140,SOURCE_TEST=700,TARGET_TEST=3431,SOURCE_UNUSED=560),counts
    assert all(r['domain']==('target' if r['role']=='TARGET_TEST' else 'source') for r in rows)
    roles=defaultdict(set)
    for row in rows:
        if row['role']!='SOURCE_UNUSED':roles[row['group_id']].add(row['role'])
    assert all(len(v)==1 for v in roles.values())
    return dict(candidate_videos=len(rows),role_counts=dict(counts),known_group_roles_disjoint_before_RGB_closure=True,
        scope='metadata only; no production artifact audit or freeze')
check('production_candidate_count_domain_and_initial_source_roles',actual_candidate_metadata)
print(json.dumps(dict(tests=results,cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),cuda_initialized=torch.cuda.is_initialized(),
    code_hashes={n:trainer.digest(V/'reliability_v2'/n) for n in ['data_h.py','qualification_h.py','geometry_h.py']}),indent=2))

