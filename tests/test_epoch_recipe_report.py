import importlib.util
import json
from pathlib import Path
import pytest


def load_module():
    path=Path(__file__).parents[1]/'scripts/epoch_development_report.py'
    assert path.is_file(),'epoch development reporting entry is not implemented'
    spec=importlib.util.spec_from_file_location('epoch_report',path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def make_run(root,seed,source_best=1,boundary=False,small_gain=False):
    import torch
    from journal_detector import protocol,trainer
    data=root/'data/D';data.mkdir(parents=True,exist_ok=True)
    for name in ['train','dev','cdf_dev']:
        (data/f'{name}.parquet').write_bytes(name.encode())
    trainer.save_json(data/'freeze.json',{'status':'FROZEN','manifest_sha256':{
        p.name:trainer.digest(p) for p in data.glob('*.parquet')}})
    environment=root/'configs/requirements.execution.lock.txt';environment.parent.mkdir(exist_ok=True)
    environment.write_text('SYNTHETIC_EVIDENCE_ONLY')
    weight=root/'initialization.bin';weight.write_bytes(b'fixed')
    run=f'dev_{seed}';out=root/'runs'/run;out.mkdir(parents=True)
    task=dict(route='D',phase='development',backbone='clip_vit_b16',seed=seed,epochs=10,
        recipe_frozen=False,microbatch=128,precision='AMP',lr=1e-5,
        train_manifest='data/D/train.parquet',dev_manifest='data/D/dev.parquet',
        cdf_development_manifest='data/D/cdf_dev.parquet')
    config_path=root/'configs/development.json'
    config=json.loads(config_path.read_text()) if config_path.exists() else dict(
        protocol=protocol.PROTOCOL,artifact_root=str(root),effective_batch=128,
        formal_seeds=protocol.SEEDS,h_formal_seeds=protocol.H_SEEDS,
        development_seeds=protocol.DEVELOPMENT_SEEDS,runs={},backbones={
            b:dict(weight_path=str(weight),weight_sha256=trainer.digest(weight)) for b in protocol.LRS})
    config['runs'][run]=task;trainer.save_json(config_path,config)
    identity=trainer.training_identity(config,task)
    selection=trainer.Selection();stop=__import__('journal_detector.development',fromlist=['DevelopmentStop']).DevelopmentStop()
    history=[]
    for epoch in range(1,11 if boundary else 8):
        source=.6+epoch*.02 if boundary else .9 if epoch==source_best else .8
        cdf=(.7+epoch*(.0001 if small_gain else .02)) if boundary else .7 if epoch==1 else .95
        selection.observe(source,epoch);stop.observe(source,cdf,epoch)
        history.append(dict(epoch=epoch,dev={'pooled_auc':source},cdf_dev={'pooled_auc':cdf},selection=selection.state_dict(),development_stop=stop.state_dict()))
    checkpoint={'identity':identity,'epoch':selection.best_epoch,'weights':'ema','auc':selection.best_score,'model':{}}
    torch.save(checkpoint,out/'best_model.pth')
    torch.save(dict(checkpoint,epoch=stop.cdf.best_epoch,auc=stop.cdf.best_score,
        selection='isolated_cdf_dev_pooled_mean_probability_auc_ema'),out/'cdf_best_model.pth')
    torch.save(dict(identity=identity,epoch=len(history),history=history,selection=selection.state_dict(),
        development_stop=stop.state_dict()),out/'state.pt')
    trainer.save_json(out/'resolved_config.json',task)
    trainer.save_json(out/'history.json',history)
    trainer.save_json(out/'complete.json',dict(status='TRAINING_COMPLETE',phase='development',
        identity=identity,best_epoch=selection.best_epoch,max_epochs=10,epochs=len(history),
        target_used=True,early_stopped=stop.stopped,selection='source_dev_pooled_mean_probability_auc_ema',
        best_sha256=trainer.digest(out/'best_model.pth'),cdf_best_epoch=stop.cdf.best_epoch,
        cdf_best_sha256=trainer.digest(out/'cdf_best_model.pth')))
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


@pytest.mark.parametrize('damage',['identity','registry','checkpoint','history'])
def test_epoch_report_rejects_unbound_development_evidence(tmp_path,damage):
    m=load_module();runs=[make_run(tmp_path,s) for s in [1007,1042]]
    out=tmp_path/'runs'/runs[0]
    if damage=='identity':
        path=out/'complete.json';q=json.loads(path.read_text());q['identity']['data_freeze']='f'*64;path.write_text(json.dumps(q))
    elif damage=='registry':
        path=out/'resolved_config.json';q=json.loads(path.read_text());q['lr']=.1;path.write_text(json.dumps(q))
    elif damage=='checkpoint':
        path=out/'best_model.pth';path.write_bytes(path.read_bytes()+b'changed')
    else:
        path=out/'history.json';q=json.loads(path.read_text());q[0]['cdf_dev']['pooled_auc']=.99;path.write_text(json.dumps(q))
    with pytest.raises(ValueError,match='identity|registered|checkpoint|history'):
        m.cap_report(tmp_path,runs,'D','clip_vit_b16',10)


def test_still_improving_observation_cap_cannot_be_frozen(tmp_path):
    m=load_module();runs=[make_run(tmp_path,s,boundary=True) for s in [1007,1042]]
    with pytest.raises(ValueError,match='improv|boundary|extend|unresolved'):
        m.freeze(tmp_path,runs,'D','clip_vit_b16',10)
    assert not (tmp_path/'configs/epoch_recipes/D_clip_vit_b16.json').exists()


def test_boundary_plateau_requires_explicit_evidence_backed_decision(tmp_path):
    m=load_module();runs=[make_run(tmp_path,s,boundary=True,small_gain=True) for s in [1007,1042]]
    with pytest.raises(ValueError,match='boundary|unresolved'):
        m.freeze(tmp_path,runs,'D','clip_vit_b16',10)
    result=m.freeze(tmp_path,runs,'D','clip_vit_b16',10,
        decision={'basis':'empirical_plateau','rationale':'Two complete fresh runs; last three CDF epochs change only0.02pp.'})
    recipe=json.loads(Path(result['recipe']).read_text())
    assert recipe['stopping_decision']['basis']=='empirical_plateau'
    assert recipe['development_report']['boundary_improving']
    assert recipe['test_used'] is False


def test_large_boundary_gain_cannot_be_overridden_by_a_freeform_plateau_note(tmp_path):
    m=load_module();runs=[make_run(tmp_path,s,boundary=True) for s in [1007,1042]]
    with pytest.raises(ValueError,match='improv|extend'):
        m.freeze(tmp_path,runs,'D','clip_vit_b16',10,
            decision={'basis':'empirical_plateau','rationale':'An unsupported claim that the cap is sufficient.'})
    assert not (tmp_path/'configs/epoch_recipes/D_clip_vit_b16.json').exists()


def test_fresh_cosine_confirmation_requires_a_verified_longer_reference(tmp_path):
    m=load_module();runs=[make_run(tmp_path,s,boundary=True) for s in [1007,1042]]
    with pytest.raises(ValueError,match='longer|confirm'):
        m.freeze(tmp_path,runs,'D','clip_vit_b16',10,
            decision={'basis':'fresh_cosine_comparison','rationale':'Reusing the same cap is not confirmation.',
                      'reference':{'runs':runs,'epochs':10}})
