import random
import numpy as np
import torch
from deepfake_detector import training as t


def test_clip_normalization_matches_public_weight_definition():
    assert hasattr(t,'normalization'), 'normalization not implemented'
    mean,std=t.normalization('clip_vit_b16')
    assert mean==(.48145466,.45762756,.40821073)
    assert std==(.26862954,.26130258,.27577711)


def test_checkpoint_restores_optimizer_rng_and_rejects_changed_protocol(tmp_path):
    assert hasattr(t,'save_checkpoint'), 'checkpoint persistence not implemented'
    random.seed(7);np.random.seed(7);torch.manual_seed(7)
    model=torch.nn.Linear(3,2);opt=torch.optim.AdamW(model.parameters(),lr=.01)
    x=torch.randn(4,3);y=torch.tensor([0,1,0,1])
    torch.nn.functional.cross_entropy(model(x),y).backward();opt.step();opt.zero_grad()
    path=tmp_path/'state.pt';identity={'config':'abc','manifest':'xyz'}
    t.save_checkpoint(path,model,opt,None,1,identity)
    expected=(random.random(),np.random.random(),torch.rand(1))
    torch.nn.functional.cross_entropy(model(x),y).backward();opt.step();target={k:v.clone() for k,v in model.state_dict().items()}
    restored=torch.nn.Linear(3,2);other=torch.optim.AdamW(restored.parameters(),lr=.01)
    assert t.restore_checkpoint(path,restored,other,None,identity)==1
    assert random.random()==expected[0] and np.random.random()==expected[1] and torch.equal(torch.rand(1),expected[2])
    torch.nn.functional.cross_entropy(restored(x),y).backward();other.step()
    assert all(torch.equal(v,restored.state_dict()[k]) for k,v in target.items())
    try:t.restore_checkpoint(path,restored,other,None,{'config':'CHANGED','manifest':'xyz'})
    except ValueError:pass
    else:raise AssertionError('Must reject changed protocol rather than silently resume')


def test_hierarchical_weights_equal_classes_methods_and_video_mass():
    assert hasattr(t,'sampling_weights'), 'balanced sampling not implemented'
    import pandas as pd
    df=pd.DataFrame({'label':[0,0,0,1,1,1,1], 'method_id':['real']*3+['a','a','b','b'], 'evaluation_unit':['video_frame']*7,'video_id':['r1','r1','r2','v1','v2','v1','v1']})
    w=t.sampling_weights(df)
    assert np.isclose(w[:3].sum(),.5)
    assert np.isclose(w[3:5].sum(),.25) and np.isclose(w[5:].sum(),.25)
    assert np.isclose(w[:2].sum(),w[2]) and np.isclose(w[5],w[6])


def test_epoch_history_commits_with_checkpoint_even_before_separate_json(tmp_path):
    import inspect
    assert 'history' in inspect.signature(t.save_checkpoint).parameters, 'history is not atomic with checkpoint'
    model=torch.nn.Linear(2,2);opt=torch.optim.AdamW(model.parameters())
    history=[{'epoch':1,'sample_slots':8,'sample_sequence_sha256':'frozen'}]
    path=tmp_path/'state.pt'
    t.save_checkpoint(path,model,opt,None,1,{'manifest':'same'},history=history)
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['history']==history
    assert state['epoch']==state['history'][-1]['epoch']
