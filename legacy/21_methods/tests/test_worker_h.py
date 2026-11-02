import copy
import json
from pathlib import Path
import pandas as pd
import pytest

from reliability_v2 import worker_h as w
from reliability_v2 import queue_h as q


def fixture(tmp_path, monkeypatch, formal=False):
    from reliability_v2 import data_h,qualification_h
    root=tmp_path; (root/'configs').mkdir();(root/'qualification').mkdir()
    weight=root/'weight.pt';weight.write_bytes(b'pretrained')
    (root/'configs/requirements.execution.lock.txt').write_text('locked')
    base=dict(artifact_root=str(root),allowed_gpus=[0,1,2,3],allowed_gpu_uuids=['u0','u1','u2','u3'],
        backbones={b:dict(weight_path=str(weight),weight_sha256=w.trainer.digest(weight),learning_rate=1e-5) for b in q.BACKBONES},workers=4,effective_batch=128)
    (root/'configs/base.json').write_text(json.dumps(base))
    tables={}
    for hist in ['B','AB']:
        for role in ['train','dev']:
            p=root/f'data/H/{hist}/{role}.parquet';p.parent.mkdir(parents=True,exist_ok=True)
            image=root/f'{hist}_{role}.png';image.write_bytes(b'known_source_image')
            pd.DataFrame([dict(image_path=str(image),content_sha256=w.trainer.digest(image))]).to_parquet(p)
            tables[f'{hist}/{role}']=dict(path=str(p),sha256=w.trainer.digest(p),rows=128)
    freeze=dict(status='FROZEN',tables=tables,training_draw_count=128)
    (root/'data/H/freeze.json').write_text(json.dumps(freeze))
    gate=dict(status='PASS',h_code_bundle_sha256='bundle',backbones={b:dict(microbatch=32,fp32_microbatch=16) for b in q.BACKBONES})
    (root/'qualification/H_gate.json').write_text(json.dumps(gate))
    c={**base,'schema':'H_EXECUTION_V2','development_seeds':[1007,1042],'formal_seeds':[7,42,123,2026,3407],
        'H_training_draw_count':128,'H_training_rows':128,'H_data_freeze_sha256':w.trainer.digest(root/'data/H/freeze.json'),
        'H_qualification_sha256':w.trainer.digest(root/'qualification/H_gate.json'),'h_code_bundle_sha256':'bundle',
        'H_microbatches':{b:{'AMP':32,'FP32':16} for b in q.BACKBONES},'runs':{}}
    recipe=root/'configs/H_frozen_recipes.json';recipe.write_text(json.dumps(dict(target_scores_used=False,recipes={b:dict(lr=1e-5,epochs=30) for b in q.BACKBONES})))
    phase='formal' if formal else 'development';seed=7 if formal else 1007
    run='H_formal_clip_vit_b16_B_seed7' if formal else 'H_dev_clip_vit_b16_B_lr1e-05_seed1007_e30'
    c['runs'][run]=q.registered_task(c,'clip_vit_b16','B',seed,1e-5,30,phase,recipe_sha256=w.trainer.digest(recipe))
    p=root/'configs/H_execution.json';p.write_text(json.dumps(c))
    monkeypatch.setattr(data_h,'validate_h_freeze',lambda root:freeze)
    monkeypatch.setattr(qualification_h,'validate_qualification',lambda root:gate)
    monkeypatch.setattr(w,'h_code_hash',lambda:'bundle')
    return p,run,c


def test_valid_source_only_task_and_actual_input_byte_guard(tmp_path,monkeypatch):
    p,run,c=fixture(tmp_path,monkeypatch)
    _,task,identity=w.validate_task(p,run,verify_inputs=True)
    assert task['draw_count']==128 and identity['train_manifest']==w.trainer.digest(tmp_path/task['train_manifest'])
    (tmp_path/'B_train.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source input byte'):w.validate_task(p,run,verify_inputs=True)


@pytest.mark.parametrize('key,value',[('workers',8),('allowed_gpu_uuids',['u4','u5','u6','u7']),('formal_seeds',[1007]),('development_seeds',[7])])
def test_mutated_global_execution_header_rejected(tmp_path,monkeypatch,key,value):
    p,run,c=fixture(tmp_path,monkeypatch);c[key]=value;p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='header|seed'):w.validate_task(p,run)


@pytest.mark.parametrize('field,value',[('microbatch',16),('h_code_bundle_sha256','old'),('train_manifest','data/H/AB/train.parquet'),('seed',7)])
def test_mutated_task_protocol_rejected(tmp_path,monkeypatch,field,value):
    p,run,c=fixture(tmp_path,monkeypatch);c['runs'][run][field]=value;p.write_text(json.dumps(c))
    with pytest.raises(ValueError):w.validate_task(p,run)


def test_formal_source_recipe_and_pretrained_binding(tmp_path,monkeypatch):
    p,run,c=fixture(tmp_path,monkeypatch,formal=True);w.validate_task(p,run)
    c['runs'][run]['lr']=3e-5;p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='same frozen source recipe'):w.validate_task(p,run)


def test_direct_worker_cannot_bypass_D_priority_or_physical_UUID_mapping(tmp_path,monkeypatch):
    from reliability_v2 import queue_d
    p,run,c=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(queue_d,'gpu_memory',lambda:{i:dict(uuid=f'u{i}',used_mib=0) for i in range(8)})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','u0')
    (tmp_path/'queue_status.json').write_text(json.dumps(dict(stage='P2_D_STANDARD_DEVELOPMENT',active=[])))
    monkeypatch.setattr(w.trainer,'train',lambda *a,**kw:pytest.fail('GPU trainer must not be called'))
    with pytest.raises(ValueError,match='retains priority'):w.run(p,run)
    (tmp_path/'queue_status.json').write_text(json.dumps(dict(stage='P3_D_CACHE_COMPLETE_ANALYSIS_PENDING',active=[])))
    assert w.verify_resource_boundary(c)['D_priority_checked']
    c['allowed_gpu_uuids']=['u4','u5','u6','u7'];monkeypatch.setenv('CUDA_VISIBLE_DEVICES','u4')
    with pytest.raises(ValueError,match='physical GPU0-3'):w.verify_resource_boundary(c)


def completion_fixture(tmp_path,monkeypatch):
    p,run,c=fixture(tmp_path,monkeypatch);_,task,identity=w.validate_task(p,run)
    out=tmp_path/'runs'/run;out.mkdir(parents=True);(out/'final.pt').write_bytes(b'final');(out/'source_best.pt').write_bytes(b'best')
    history=[dict(epoch=i+1,success=1,skip=0,dev={'macro_auc':.8}) for i in range(30)];(out/'history.json').write_text(json.dumps(history))
    receipt=dict(status='TRAINING_COMPLETE',identity=identity,epochs=30,attempts=30,success=30,skip=0,selection='frozen_last',target_used=False,
        final_sha256=w.trainer.digest(out/'final.pt'),source_best_sha256=w.trainer.digest(out/'source_best.pt'))
    (out/'complete.json').write_text(json.dumps(receipt))
    (out/'identity.json').write_text(json.dumps(identity));(out/'resolved_config.json').write_text(json.dumps(c))
    (out/'H_training_start.json').write_text(json.dumps(w._start_record(p,run,c,identity)))
    w.trainer.torch.save(dict(identity=identity,epoch=30,attempts=30,history=history),out/'state.pt')
    for name in w._completion_artifacts(30):
        if not(out/name).exists():(out/name).write_bytes(b'known committed epoch evidence')
    return p,run,c,out,receipt


def test_completed_run_validates_sealed_history_and_terminal_hashes(tmp_path,monkeypatch):
    p,run,c,out,receipt=completion_fixture(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='not been sealed'):w.verify_run_completion(p,run)
    assert w.finalize_completion_seal(p,run)==receipt
    assert w.verify_run_completion(p,run)==receipt
    original=(out/'history.json').read_bytes();history=json.loads(original);history[0]['dev']['macro_auc']=.99
    (out/'history.json').write_text(json.dumps(history))
    with pytest.raises(ValueError,match='artifact hash changed'):w.verify_run_completion(p,run)
    with pytest.raises(ValueError,match='history hash changed'):w.read_sealed_history(tmp_path,run)
    with pytest.raises(ValueError,match='artifact hash changed'):w.finalize_completion_seal(p,run)
    (out/'history.json').write_bytes(original)
    assert w.read_sealed_history(tmp_path,run)==json.loads(original)
    (out/'final.pt').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checkpoint hash'):w.verify_run_completion(p,run)


@pytest.mark.parametrize('mutation',['missing_start','state_history','missing_epoch','copied_identity'])
def test_unsealed_recovery_requires_start_checkpoint_and_fixed_inventory(tmp_path,monkeypatch,mutation):
    p,run,c,out,receipt=completion_fixture(tmp_path,monkeypatch)
    if mutation=='missing_start':(out/'H_training_start.json').unlink()
    elif mutation=='missing_epoch':(out/'epoch_30_source_dev.parquet').unlink()
    elif mutation=='state_history':
        state=w.trainer.torch.load(out/'state.pt',weights_only=False);state['history'][0]['dev']['macro_auc']=.99
        w.trainer.torch.save(state,out/'state.pt')
    else:
        start=json.loads((out/'H_training_start.json').read_text());start['run_id']='H_copied'
        (out/'H_training_start.json').write_text(json.dumps(start))
    with pytest.raises(ValueError,match='start evidence|checkpoint history|inventory'):
        w.finalize_completion_seal(p,run)
    assert not(out/'H_completion_seal.json').exists()


def test_completion_seal_cannot_be_relabelled_or_drop_artifacts(tmp_path,monkeypatch):
    p,run,c,out,receipt=completion_fixture(tmp_path,monkeypatch);w.finalize_completion_seal(p,run)
    path=out/'H_completion_seal.json';seal=json.loads(path.read_text());original=path.read_bytes()
    seal['run_id']='H_other';path.write_text(json.dumps(seal))
    with pytest.raises(ValueError,match='run/root/identity'):w.verify_run_completion(p,run)
    path.write_bytes(original);seal=json.loads(original);seal['artifact_sha256'].pop('epoch_01_updates.jsonl')
    path.write_text(json.dumps(seal))
    with pytest.raises(ValueError,match='inventory'):w.verify_run_completion(p,run)


def test_task_cannot_be_registered_under_another_backbone_name(tmp_path,monkeypatch):
    p,run,c=fixture(tmp_path,monkeypatch);c['runs'][run]['backbone']='dinov2_vit_b14';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='canonical'):w.validate_task(p,run)


def test_worker_validates_before_creating_output_locks(tmp_path,monkeypatch):
    p,run,c=fixture(tmp_path,monkeypatch);c['allowed_gpu_uuids']=['u4','u5','u6','u7'];p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='header'):w.run(p,run)
    assert not(tmp_path/'runs').exists()


@pytest.mark.parametrize('existing',['complete','partial'])
def test_worker_cannot_recreate_missing_start_over_existing_training(tmp_path,monkeypatch,existing):
    if existing=='complete':
        p,run,c,out,receipt=completion_fixture(tmp_path,monkeypatch)
        (out/'H_training_start.json').unlink()
    else:
        p,run,c=fixture(tmp_path,monkeypatch);out=tmp_path/'runs'/run;out.mkdir(parents=True)
        (out/'state.pt').write_bytes(b'existing partial state')
    monkeypatch.setattr(w,'verify_resource_boundary',lambda *a,**kw:None)
    monkeypatch.setattr(w.trainer,'train',lambda *a,**kw:pytest.fail('must not launder prior evidence'))
    with pytest.raises(ValueError,match='lack pre-training start'):w.run(p,run)
    assert not(out/'H_training_start.json').exists()
