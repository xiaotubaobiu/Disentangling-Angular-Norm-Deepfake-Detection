from pathlib import Path
import pytest
import torch


def fixture_batches():
    generator=torch.Generator().manual_seed(31)
    return [(torch.randn(128,3,generator=generator),torch.arange(128)%2) for _ in range(6)]


def factory():
    return torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.BatchNorm1d(8),
        torch.nn.Dropout(.35),torch.nn.Linear(8,2))


def test_optimizer_ema_bn_and_dropout_resume_match_after_rng_perturbation(tmp_path):
    from journal_detector.qualification_resume import qualify_model
    result=qualify_model(factory,fixture_batches(),tmp_path/'state.pt',{'synthetic_only':True},
        seed=7,lr=1e-3,device='cpu')
    assert result['status']=='QUALIFIED' and result['continuous_success']==6 and result['resumed_success']==4
    assert all(result['matching_state'].values())
    assert result['continuous_learning_rates']==pytest.approx([.001,.001,.001,.001,.0005,.0005])
    assert result['random_stream_perturbed_before_reload'] is True


def test_resume_qualification_detects_a_broken_rng_restore(tmp_path,monkeypatch):
    from journal_detector import trainer
    from journal_detector.qualification_resume import qualify_model
    original=trainer.load_state
    def broken(*args,**kwargs):
        result=original(*args,**kwargs);torch.manual_seed(9999);return result
    monkeypatch.setattr(trainer,'load_state',broken)
    with pytest.raises(ValueError,match='resume|matching'):
        qualify_model(factory,fixture_batches(),tmp_path/'state.pt',{'synthetic_only':True},
            seed=7,lr=1e-3,device='cpu')
