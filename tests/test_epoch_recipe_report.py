import importlib.util
import json
from pathlib import Path
import pytest


def load_module():
    path=Path(__file__).parents[1]/'scripts/epoch_development_report.py'
    assert path.is_file(),'epoch development reporting entry is not implemented'
    spec=importlib.util.spec_from_file_location('epoch_report',path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def make_run(root,seed,source_best=1):
    run=f'dev_{seed}';out=root/'runs'/run;out.mkdir(parents=True)
    task={'route':'D','phase':'development','backbone':'clip_vit_b16','seed':seed,'epochs':10}
    (out/'resolved_config.json').write_text(json.dumps(task))
    (out/'complete.json').write_text(json.dumps({'status':'TRAINING_COMPLETE','phase':'development',
        'best_epoch':source_best,'max_epochs':10,'epochs':2,'target_used':True,'early_stopped':True}))
    history=[{'epoch':1,'dev':{'pooled_auc':.9},'cdf_dev':{'pooled_auc':.7}},
             {'epoch':2,'dev':{'pooled_auc':.8},'cdf_dev':{'pooled_auc':.95}}]
    (out/'history.json').write_text(json.dumps(history))
    return run


def test_recipe_report_scores_deployable_source_selected_checkpoint_not_cdf_oracle(tmp_path):
    m=load_module();runs=[make_run(tmp_path,s) for s in [1007,1042]]
    report=m.cap_report(tmp_path,runs,'D','clip_vit_b16',10)
    assert report['mean_score']==pytest.approx(.7)
    assert report['epoch_candidate']['candidate_best_epoch']==2
    assert report['score_kind']=='cdf_dev_at_source_selected_ema'


def test_epoch_report_rejects_formal_seeds_or_missing_registered_repeat(tmp_path):
    m=load_module();run=make_run(tmp_path,7)
    with pytest.raises(ValueError,match='seed|development'):
        m.cap_report(tmp_path,[run],'D','clip_vit_b16',10)
    run=make_run(tmp_path,1007)
    with pytest.raises(ValueError,match='both|two|seed'):
        m.cap_report(tmp_path,[run],'D','clip_vit_b16',10)
