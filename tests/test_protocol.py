import copy
import numpy as np
import pandas as pd
import pytest
import torch
from journal_detector.trainer import EMA, epoch_lr, Selection, source_metrics


def test_epoch_schedule_matches_conference():
    base=1e-5
    assert epoch_lr(0,15,base)==base
    assert epoch_lr(1,15,base)==base
    assert epoch_lr(14,15,base)==pytest.approx(base*.5*(1+np.cos(np.pi*13/14)))


def test_ema_parameters_live_bn_buffers_and_restore():
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2))
    ema=EMA(model,decay=.999)
    initial=copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for p in model.parameters(): p.add_(2)
        model[1].running_mean.fill_(3)
    ema.update(model)
    live=copy.deepcopy(model.state_dict())
    with ema.applied(model):
        assert torch.allclose(model[0].weight,initial['0.weight']+.002)
        assert torch.equal(model[1].running_mean,live['1.running_mean'])
    for key,value in model.state_dict().items(): assert torch.equal(value,live[key])


def test_selection_five_ties_stop_and_improvement_resets():
    s=Selection()
    assert s.observe(.8,1)
    for epoch in range(2,6):
        assert not s.observe(.8,epoch)
        assert not s.stopped
    recovered=Selection(**s.state_dict())
    assert not recovered.observe(.7,6)
    assert recovered.stopped and recovered.best_epoch==1
    assert recovered.observe(.9,7)
    assert recovered.wait==0 and not recovered.stopped


def test_validation_uses_natural_video_mean_probability(monkeypatch):
    import journal_detector.trainer as t
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=torch.nn.Identity();self.classifier=torch.nn.Linear(1,2)
            with torch.no_grad():self.classifier.weight.copy_(torch.tensor([[0.],[1.]]));self.classifier.bias.zero_()
    # mean-logit would rank real above fake; mean-prob ranks fake above real.
    margins=torch.tensor([[8.],[-2.],[2.],[2.],[1.],[1.]])
    f=pd.DataFrame({'video_uid':['real','real','a/001','a/001','b/001','b/001'],
      'method_id':['real','real','a','a','b','b'],'label':[0,0,1,1,1,1]})
    monkeypatch.setattr(t,'loader',lambda *a,**k:[(margins,torch.tensor(f.label.values))])
    metrics,videos=source_metrics(Model(),f,'clip_vit_b16',128,'cpu',0)
    assert len(videos)==3
    assert metrics['pooled_auc']==1.


def test_training_resume_preserves_ema_selection_and_stops_at_six(tmp_path,monkeypatch):
    import json
    import journal_detector.trainer as t
    import journal_detector.models as models
    import journal_detector.protocol as protocol
    root=tmp_path;data=root/'data/D';data.mkdir(parents=True)
    frame=pd.DataFrame({'rgb_sha256':['x'],'group_id':['g'],'sampling_mass':[1.],
      'method_id':['real'],'label':[0],'image_path':['unused']})
    for name in ['train','dev']:frame.to_parquet(data/f'{name}.parquet')
    config={'artifact_root':str(root),'backbones':{'toy':{'weight_path':'unused'}},'workers':0,
      'runs':{'run':{'backbone':'toy','seed':7,'epochs':15,'lr':.01,'microbatch':128,
      'train_manifest':'data/D/train.parquet','dev_manifest':'data/D/dev.parquet'}}}
    path=root/'config.json';path.write_text(json.dumps(config))
    monkeypatch.setattr(protocol,'validate_config',lambda *a:None)
    monkeypatch.setattr(protocol,'validate_manifests',lambda *a:None)
    monkeypatch.setattr(t,'training_identity',lambda *a:{'test':'identity'})
    monkeypatch.setattr(models,'build_model',lambda *a:torch.nn.Linear(1,2))
    monkeypatch.setattr(t,'loader',lambda *a,**k:[(torch.zeros(128,1),torch.zeros(128,dtype=torch.long))])
    def fake_step(model,*a):
        with torch.no_grad():
            for p in model.parameters():p.add_(1.)
        return {'success':True,'skip':False,'loss':.1}
    monkeypatch.setattr(t,'advance_step',fake_step)
    monkeypatch.setattr(t,'source_metrics',lambda *a:({'pooled_auc':.8,'macro_auc':.8},pd.DataFrame({'score':[.8]})))
    original=t.save_json
    def interrupt(path,obj):
        if Path(path).name=='history.json' and len(obj)==3:raise RuntimeError('simulated power loss')
        original(path,obj)
    from pathlib import Path
    monkeypatch.setattr(t,'save_json',interrupt)
    with pytest.raises(RuntimeError,match='power loss'):t._train(path,'run','cpu')
    state=torch.load(root/'runs/run/state.pt',weights_only=False)
    assert state['selection']['wait']==2 and state['epoch']==3
    monkeypatch.setattr(t,'save_json',original)
    result=t._train(path,'run','cpu')
    assert result['epochs']==6 and result['best_epoch']==1 and result['early_stopped']
    best=torch.load(root/'runs/run/best_model.pth',weights_only=False)
    final=torch.load(root/'runs/run/state.pt',weights_only=False)
    assert best['weights']=='ema' and best['epoch']==1
    assert final['selection']['wait']==5
    assert not torch.equal(best['model']['weight'],final['model']['weight'])
    assert not torch.equal(best['model']['weight'],final['ema']['weight'])
    assert t._train(path,'run','cpu')==result


def test_official_augmentation_matches_upstream_and_restores_rng():
    import ast, random
    from pathlib import Path
    from journal_detector.augmentation import augment,build_transform
    import albumentations as A
    import cv2
    source=Path(__file__).parents[1]/'third_party/df40/abstract_dataset.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DeepfakeAbstractBaseDataset')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='init_data_aug_method')
    scope={'A':A,'cv2':cv2,'IsotropicResize':type(build_transform().transforms[3].transforms[0])}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(source),'exec'),scope)
    from types import SimpleNamespace
    official=scope['init_data_aug_method'](SimpleNamespace(config={'resolution':224,'with_landmark':False,
      'data_aug':{'flip_prob':.5,'rotate_limit':[-10,10],'rotate_prob':.5,'blur_limit':[3,7],'blur_prob':.5,
      'brightness_limit':[-.1,.1],'contrast_limit':[-.1,.1],'quality_lower':40,'quality_upper':100}}))
    image=np.random.default_rng(4).integers(0,256,(224,224,3),dtype=np.uint8)
    for seed in range(12):
        random.seed(seed);np.random.seed(seed);expected=official(image=image)['image']
        random.seed(71);np.random.seed(71)
        actual=augment(image,seed)
        assert np.array_equal(actual,expected)
        assert random.random()==random.Random(71).random()
        assert np.random.random()==np.random.RandomState(71).random_sample()


def test_official_dino_cpu_interpolation_preserves_value_and_gradient():
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    sys.path.insert(0,str(Path(__file__).parents[1]/'third_party/dinov2'))
    from dinov2.models.vision_transformer import DinoVisionTransformer
    from journal_detector.models import official_position_cpu
    position=torch.randn(1,1370,8,requires_grad=True)
    proxy=SimpleNamespace(pos_embed=position,patch_size=14,interpolate_offset=.1,interpolate_antialias=False)
    x=torch.empty(1,257,8)
    expected=DinoVisionTransformer.interpolate_pos_encoding(proxy,x,224,224)
    actual=official_position_cpu(proxy,x,224,224)
    assert torch.equal(expected,actual)
    a=torch.autograd.grad(expected.square().sum(),position,retain_graph=True)[0]
    b=torch.autograd.grad(actual.square().sum(),position)[0]
    assert torch.equal(a,b) and b.abs().sum()>0
