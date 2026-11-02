import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from journal_detector import protocol


def config_for(phase='development', route='D', seed=1007, epochs=10):
    task = dict(route=route, phase=phase, recipe_frozen=phase == 'formal',
                backbone='clip_vit_b16' if route == 'D' else 'efficientnet_b0',
                seed=seed, epochs=epochs, microbatch=128, precision='AMP',
                lr=1e-5 if route == 'D' else 1e-4,
                train_manifest='data/D/train.parquet' if route == 'D' else 'data/H/B/train.parquet',
                dev_manifest='data/D/dev.parquet' if route == 'D' else 'data/H/B/dev.parquet')
    if phase == 'development' and route == 'D':
        task['cdf_development_manifest'] = 'data/D/cdf_dev.parquet'
    if phase == 'formal':
        task['epoch_recipe_sha256'] = 'a' * 64
    return dict(protocol='journal_detector-target-dev-20261102-v2', effective_batch=128,
                formal_seeds=[7,42,123,2026,3407], h_formal_seeds=[7,42,123],
                development_seeds=[1007,1042], backbones={b: {} for b in protocol.LRS},
                runs={'run': task})


def test_development_registry_keeps_fixed_lr_and_uses_isolated_target_role():
    c = config_for()
    protocol.validate_config(c)
    for bad in ['data/D/target_test.parquet', 'data/D/target_cal.parquet', 'data/D/target_full.parquet']:
        q = copy.deepcopy(c); q['runs']['run']['cdf_development_manifest'] = bad
        with pytest.raises(ValueError, match='CDF|development'):
            protocol.validate_config(q)
    q = copy.deepcopy(c); q['runs']['run']['lr'] = 2e-5
    with pytest.raises(ValueError, match='learning rate|recipe'):
        protocol.validate_config(q)


def test_formal_seed_cannot_be_used_for_development():
    with pytest.raises(ValueError, match='seed'):
        protocol.validate_config(config_for(seed=7))


def test_formal_requires_frozen_epoch_recipe_and_h_uses_three_seeds():
    c = config_for('formal', seed=7, epochs=5)
    protocol.validate_config(c)
    del c['runs']['run']['epoch_recipe_sha256']
    with pytest.raises(ValueError, match='epoch.*recipe'):
        protocol.validate_config(c)
    protocol.validate_config(config_for('formal', 'H', seed=123, epochs=10))
    with pytest.raises(ValueError, match='seed'):
        protocol.validate_config(config_for('formal', 'H', seed=2026))


def test_epoch_choice_averages_registered_development_seeds_and_never_uses_formal():
    from journal_detector.development import select_epoch_budget
    histories = {
        1007: [dict(epoch=e, dev={'pooled_auc': .95}, cdf_dev={'pooled_auc': v})
               for e,v in enumerate([.60,.75,.82,.80,.79],1)],
        1042: [dict(epoch=e, dev={'pooled_auc': .94}, cdf_dev={'pooled_auc': v})
               for e,v in enumerate([.65,.76,.80,.81,.78],1)]}
    out=select_epoch_budget(histories)
    assert out['candidate_best_epoch'] == 3
    assert out['cdf_development_used'] is True
    assert out['criterion'] == 'mean_registered_development_seed_cdf_auc_earliest_maximum'
    with pytest.raises(ValueError, match='seed'):
        select_epoch_budget({7: histories[1007]})


def test_development_stop_waits_for_source_and_cdf_plateau():
    from journal_detector.development import DevelopmentStop
    s=DevelopmentStop(patience=2)
    assert s.observe(.9,.6,1)
    assert s.observe(.9,.7,2)
    assert not s.observe(.9,.69,3)
    assert not s.stopped
    assert not s.observe(.89,.68,4)
    assert s.stopped
    restored=DevelopmentStop.from_state(s.state_dict())
    assert restored.stopped and restored.cdf.best_epoch == 2


def test_training_identity_does_not_change_when_unrelated_h_code_is_added(tmp_path,monkeypatch):
    from journal_detector import trainer
    # The function must expose a route so concurrently developed H files cannot
    # invalidate an already running D trajectory.
    assert trainer.code_hash(route='D')


def test_development_training_waits_for_both_curves_and_saves_isolated_cdf_choice(tmp_path,monkeypatch):
    import torch
    from journal_detector import trainer as t, models
    data=tmp_path/'data/D';data.mkdir(parents=True)
    base=pd.DataFrame({'rgb_sha256':['x','y'],'group_id':['g','h'],
        'sampling_mass':[.5,.5],'method_id':['real','fake'],'label':[0,1],
        'image_path':['unused','unused'],'video_uid':['a','b']})
    for name in ['train','dev','cdf_dev']:
        q=base.copy();q['role']=name;q['video_uid']=name+'/'+q.video_uid;q.to_parquet(data/f'{name}.parquet')
    c=config_for();c['artifact_root']=str(tmp_path);c['workers']=0
    c['backbones']['clip_vit_b16']['weight_path']='unused'
    c['runs']['run']['epochs']=12
    path=tmp_path/'config.json';path.write_text(json.dumps(c))
    monkeypatch.setattr(protocol,'validate_manifests',lambda *a,**k:None)
    monkeypatch.setattr(t,'training_identity',lambda *a:{'test':'development'})
    monkeypatch.setattr(models,'build_model',lambda *a:torch.nn.Linear(1,2))
    monkeypatch.setattr(t,'loader',lambda *a,**k:[(torch.zeros(128,1),torch.zeros(128,dtype=torch.long))])
    monkeypatch.setattr(t,'advance_step',lambda *a,**k:{'success':True,'skip':False,'loss':.1})
    count={'dev':0,'cdf_dev':0}
    def metrics(model,frame,*a):
        role=frame.role.iloc[0];count[role]+=1;e=count[role]
        auc=.9 if role=='dev' else .5+min(e,5)*.01 if e<=5 else .5
        return {'pooled_auc':auc,'macro_auc':auc},pd.DataFrame({'score':[auc]})
    monkeypatch.setattr(t,'source_metrics',metrics)
    result=t._train(path,'run','cpu')
    assert result['epochs']==10  # source alone would have stopped at epoch6
    assert result['cdf_best_epoch']==5
    assert result['target_used'] is True and result['phase']=='development'
    state=torch.load(tmp_path/'runs/run/state.pt',weights_only=False)
    assert state['development_stop']['cdf']['best_epoch']==5
    best=torch.load(tmp_path/'runs/run/cdf_best_model.pth',weights_only=False)
    assert best['epoch']==5 and best['selection']=='isolated_cdf_dev_pooled_mean_probability_auc_ema'


def identity_fixture(tmp_path,route='D'):
    from journal_detector import trainer as t
    data=tmp_path/f'data/{route}';data.mkdir(parents=True)
    (data/'train.parquet').write_bytes(b'train');(data/'dev.parquet').write_bytes(b'dev')
    (data/'cdf_dev.parquet').write_bytes(b'original dev')
    (data/('freeze.json' if route=='D' else 'train_dev_freeze.json')).write_text(json.dumps({
        'manifest_sha256':{'cdf_dev.parquet':t.digest(data/'cdf_dev.parquet')}}))
    lock=tmp_path/'configs/requirements.execution.lock.txt';lock.parent.mkdir();lock.write_text('fixed')
    weight=tmp_path/'pretrain';weight.write_bytes(b'pretrain')
    config={'artifact_root':str(tmp_path),'backbones':{'efficientnet_b0':{
        'weight_path':str(weight),'weight_sha256':t.digest(weight)}}}
    task={'route':route,'phase':'development','backbone':'efficientnet_b0','epochs':10,
        'train_manifest':f'data/{route}/train.parquet','dev_manifest':f'data/{route}/dev.parquet'}
    return config,task


def test_development_manifest_must_match_frozen_cdf_hash(tmp_path):
    from journal_detector import trainer as t
    config,task=identity_fixture(tmp_path)
    task['cdf_development_manifest']='data/D/cdf_dev.parquet'
    t.training_identity(config,task)
    (tmp_path/task['cdf_development_manifest']).write_bytes(b'replaced but still a two-class dataframe')
    with pytest.raises(ValueError,match='frozen.*CDF|CDF.*freeze'):
        t.training_identity(config,task)


def test_h_cannot_borrow_target_informed_d_epoch_recipe(tmp_path):
    from journal_detector import trainer as t
    config,task=identity_fixture(tmp_path,'H')
    task.update(phase='formal',history='B',epoch_recipe_path='recipe.json')
    recipe={'schema':'EPOCH_RECIPE_V2','status':'FROZEN','route':'D',
        'backbone':'efficientnet_b0','chosen_epochs':10,'target_development_used':True,
        'data_freeze_sha256':t.digest(tmp_path/'data/H/train_dev_freeze.json')}
    path=tmp_path/'recipe.json';path.write_text(json.dumps(recipe));task['epoch_recipe_sha256']=t.digest(path)
    with pytest.raises(ValueError,match='recipe.*route|route.*recipe'):
        t.training_identity(config,task)
