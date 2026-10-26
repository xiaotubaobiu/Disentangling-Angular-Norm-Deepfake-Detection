import importlib.util, json, pathlib, sys, traceback
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
out=R/'reports/H_data_qualification_review_logs/independent'
out.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)
results=[]
def check(name,fn):
    try: results.append(dict(name=name,status='PASS',result=fn()))
    except Exception as e: results.append(dict(name=name,status='FAIL',error=repr(e),traceback=traceback.format_exc()))
def domain_boundary():
    frame=tq.source_rows(out/'pixels', 'TRAIN','B').assign(domain='source')
    try:q._source(frame,'TRAIN','B')
    except ValueError as e:return dict(accepted=False,actual_geometry_domain='source',reason=str(e),qualifier_expected_domain='ff')
    return dict(accepted=True,actual_geometry_domain='source')
(out/'pixels').mkdir(exist_ok=True)
check('real_geometry_source_domain_boundary',domain_boundary)
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
print(json.dumps(dict(tests=results,cuda_visible_devices=__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),cuda_initialized=torch.cuda.is_initialized()),indent=2))

