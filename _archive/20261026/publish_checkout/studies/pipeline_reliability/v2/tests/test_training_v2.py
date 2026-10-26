import copy
import numpy as np
import pandas as pd
import pytest
import torch
from reliability_v2.trainer import (attempt_lr, epoch_draws, augmentation_parameters,
    empirical_threshold, auc_independent, save_state, load_state, advance_step, fp64_margin)
from reliability_v2.trainer import clip_gradients, repair_history, exclusive_file


def test_attempt_clock_warmup_cosine_end():
    assert attempt_lr(0, 10, 30, 1.) == pytest.approx(.1)
    assert attempt_lr(9, 10, 30, 1.) == pytest.approx(1.)
    assert attempt_lr(299, 10, 30, 1.) == pytest.approx(.05)
    assert all(attempt_lr(i, 10, 30, 1.) > 0 for i in range(300))


def test_draw_budget_rounded_and_actual_methods():
    df = pd.DataFrame({'sampling_mass':[.5,.25,.25],
        'method_contributions':['{"real":0.5}', '{"a":0.25}', '{"b":0.25}']})
    a, methods = epoch_draws(df, 1007, 1)
    b, methods_b = epoch_draws(df, 1007, 1)
    assert len(a) == 128 and np.array_equal(a,b)
    assert np.array_equal(methods, methods_b)
    assert set(methods) == {'real','a','b'}
    assert not np.array_equal(a, epoch_draws(df,1042,1)[0])


def test_augmentation_history_pair_key_is_path_independent():
    a=augmentation_parameters(7,1,'ff/source/001/frame0',13)
    assert a==augmentation_parameters(7,1,'ff/source/001/frame0',13)
    assert a!=augmentation_parameters(7,2,'ff/source/001/frame0',13)
    assert set(a)=={'flip','brightness','contrast'}


def test_threshold_ties_and_exact_small_fp_count():
    x=np.array([0.]*96+[1.])
    t=empirical_threshold(x,.05)
    assert t==np.nextafter(0.,np.inf)
    assert (x>=t).sum()==1
    assert empirical_threshold(np.array([1.,1.,1.]),.05)>1.


def test_independent_auc_counts_ties():
    assert auc_independent([0,0,1,1],[0.,1.,1.,2.])==pytest.approx(.875)


def test_margin_subtracts_head_parameters_after_fp64_cast():
    h=torch.tensor([[1.]],dtype=torch.float32)
    w=torch.tensor([[1.],[1.e10]],dtype=torch.float32)
    b=torch.zeros(2,dtype=torch.float32)
    assert fp64_margin(h,w,b).item()==9999999999.


def test_finite_extreme_gradients_clip_without_false_success_zero_update():
    p=torch.nn.Parameter(torch.zeros(2))
    p.grad=torch.tensor([5e19,-5e19])
    norm=clip_gradients([p],1.,False)
    assert torch.isfinite(norm) and norm.dtype==torch.float64
    assert p.grad.norm().item()==pytest.approx(1.,abs=1e-6)
    p.grad=torch.tensor([float('inf'),1.])
    with pytest.raises(FloatingPointError):clip_gradients([p],1.,False)


def test_restore_repairs_final_state_history_commit_gap(tmp_path):
    from reliability_v2.trainer import save_json
    save_json(tmp_path/'history.json',[{'epoch':1}])
    state={'epoch':2,'attempts':8,'history':[{'epoch':1},{'epoch':2}]}
    repair_history(tmp_path,state,4)
    import json
    assert len(json.loads((tmp_path/'history.json').read_text()))==2
    with pytest.raises(ValueError):repair_history(tmp_path,{**state,'attempts':7},4)


def test_duplicate_lifetime_run_lock_rejected(tmp_path):
    with exclusive_file(tmp_path/'run.lock'):
        with pytest.raises(BlockingIOError):
            with exclusive_file(tmp_path/'run.lock'):pass


def test_consecutive_skip_counter_survives_epoch_checkpoint(tmp_path):
    m=torch.nn.Linear(2,2);o=torch.optim.AdamW(m.parameters())
    save_state(tmp_path/'state.pt',m,o,None,1,{'x':1},[{'epoch':1}],1,consecutive_skips=7)
    state=load_state(tmp_path/'state.pt',m,o,None,{'x':1})
    assert state['consecutive_skips']==7


def test_resume_optimizer_rng_scheduler_scaler_state_exact(tmp_path):
    torch.manual_seed(17)
    m=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.3),torch.nn.Linear(8,2))
    o=torch.optim.AdamW(m.parameters(),lr=.01)
    x=torch.arange(32.).reshape(8,4)/32;y=torch.arange(8)%2
    advance_step(m,o,None,x,y,0,1,2,.01,8,'cpu')
    identity={'config':'a','data':'b','code':'c'}
    save_state(tmp_path/'state.pt',m,o,None,1,identity,[{'epoch':1}])
    advance_step(m,o,None,x,y,1,1,2,.01,8,'cpu')
    expected=copy.deepcopy(m.state_dict()); rng=torch.get_rng_state().clone()
    opt=copy.deepcopy(o.state_dict())
    other=torch.nn.Sequential(torch.nn.Linear(4,8),torch.nn.Dropout(.3),torch.nn.Linear(8,2))
    oo=torch.optim.AdamW(other.parameters(),lr=.01)
    state=load_state(tmp_path/'state.pt',other,oo,None,identity)
    assert state['epoch']==1 and state['attempts']==1
    advance_step(other,oo,None,x,y,1,1,2,.01,8,'cpu')
    assert all(torch.equal(v,other.state_dict()[k]) for k,v in expected.items())
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in opt['state'].items():
        for field,value in v.items():
            if torch.is_tensor(value): assert torch.equal(value,oo.state_dict()['state'][k][field])
    with pytest.raises(ValueError,match='identity'):
        load_state(tmp_path/'state.pt',other,oo,None,{'config':'changed'})
