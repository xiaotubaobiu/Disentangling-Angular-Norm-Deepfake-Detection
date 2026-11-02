import pandas as pd
import pytest
import torch


def paired_frames():
    a=pd.DataFrame({'mother_id':['m0','m1'],'sample_id':['m0','m1'],
        'video_uid':['v0','v1'],'label':[0,1],'method_id':['real','fake'],
        'sampling_mass':[.5,.5],'image_path':['b0','b1']})
    b=a.copy();b['image_path']=['ab0','ab1']
    return a,b


def test_h_pair_rejects_reordered_mothers_or_changed_sampling_mass():
    from journal_detector.trainer_h import validate_pair_frames
    a,b=paired_frames()
    validate_pair_frames(a,b)
    with pytest.raises(ValueError,match='mother|order'):
        validate_pair_frames(a,b.iloc[::-1])
    b.loc[0,'sampling_mass']=.4
    with pytest.raises(ValueError,match='sampling'):
        validate_pair_frames(a,b)


def test_h_common_epoch_uses_equal_weight_source_dev_and_recovers():
    from journal_detector.trainer_h import PairedSelection
    p=PairedSelection(patience=2)
    assert p.observe(.9,.7,1)
    assert p.best_score==pytest.approx(.8)
    assert p.observe(.8,.9,2)
    assert not p.observe(.9,.7,3)
    p=PairedSelection.from_state(p.state_dict())
    assert not p.observe(.8,.8,4)
    assert p.stopped and p.best_epoch==2


def test_h_paired_dropout_randomness_is_identical_for_each_update():
    from journal_detector.trainer_h import paired_step
    from journal_detector.trainer import set_random
    set_random(1007)
    a=torch.nn.Sequential(torch.nn.Dropout(.5),torch.nn.Linear(4,2))
    import copy
    b=copy.deepcopy(a)
    opts={k:torch.optim.AdamW(m.parameters(),lr=.001) for k,m in [('B',a),('AB',b)]}
    models={'B':a,'AB':b};scalers={'B':None,'AB':None}
    x=torch.ones(128,4);y=torch.arange(128)%2
    events=paired_step(models,opts,scalers,{'B':(x,y),'AB':(x,y)},0,1,10,.001,128,'cpu',1007)
    assert events['B']['loss']==events['AB']['loss']
    for p,q in zip(a.parameters(),b.parameters()):assert torch.equal(p,q)


def test_h_pair_atomic_resume_retains_common_epoch_and_both_checkpoints(tmp_path,monkeypatch):
    import copy,json
    from journal_detector import trainer as t, trainer_h as h, models, data_h
    a,b=paired_frames()
    for history,frame in [('B',a),('AB',b)]:
        folder=tmp_path/'data/H'/history;folder.mkdir(parents=True)
        frame.to_parquet(folder/'train.parquet');frame.to_parquet(folder/'dev.parquet')
    base=dict(route='H',phase='development',recipe_frozen=False,backbone='efficientnet_b0',
        seed=1007,epochs=10,microbatch=128,precision='AMP',lr=1e-4,dev_manifest='data/H/B/dev.parquet')
    tasks={name:{**base,'history':history,'train_manifest':f'data/H/{history}/train.parquet'} for name,history in [('b','B'),('ab','AB')]}
    config=dict(protocol='journal_detector-target-dev-20261102-v2',artifact_root=str(tmp_path),
        effective_batch=128,formal_seeds=[7,42,123,2026,3407],h_formal_seeds=[7,42,123],
        development_seeds=[1007,1042],workers=0,runs=tasks,
        backbones={name:{'weight_path':'unused'} for name in ['clip_vit_b16','dinov2_vit_b14','efficientnet_b0']})
    path=tmp_path/'config.json';path.write_text(json.dumps(config))
    monkeypatch.setattr(data_h,'validate_h_freeze',lambda *a,**k:{'status':'FROZEN'})
    monkeypatch.setattr(t,'training_identity',lambda cfg,task:{'history':task['history']})
    monkeypatch.setattr(models,'build_model',lambda *a:torch.nn.Linear(4,2))
    monkeypatch.setattr(t,'loader',lambda *a,**k:[(torch.zeros(128,4),torch.arange(128)%2)])
    monkeypatch.setattr(t,'source_metrics',lambda *a,**k:({'pooled_auc':.8,'macro_auc':.8},pd.DataFrame({'score':[.8]})))
    original=t.save_json
    def interrupted(p,obj):
        if p.name=='history.json' and len(obj)==3:raise RuntimeError('simulated H interruption')
        return original(p,obj)
    monkeypatch.setattr(t,'save_json',interrupted)
    with pytest.raises(RuntimeError,match='interruption'):h._train_pair(path,'b','ab','cpu')
    monkeypatch.setattr(t,'save_json',original)
    result=h._train_pair(path,'b','ab','cpu')
    assert {r['epochs'] for r in result.values()}=={6}
    assert {r['best_epoch'] for r in result.values()}=={1}
    assert all(r['selection']==h.H_SELECTION and not r['target_used'] for r in result.values())
    for run in ['b','ab']:
        ckpt=torch.load(tmp_path/'runs'/run/'best_model.pth',weights_only=False)
        assert ckpt['epoch']==1 and ckpt['selection']==h.H_SELECTION
    assert h._train_pair(path,'b','ab','cpu')==result
