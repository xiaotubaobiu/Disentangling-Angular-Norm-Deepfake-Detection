import json
from pathlib import Path
import numpy as np
import pytest

from reliability_v2 import queue_h as h


def config(tmp_path):
    return dict(artifact_root=str(tmp_path),schema='H_EXECUTION_V2',effective_batch=128,
        allowed_gpus=[0,1,2,3],allowed_gpu_uuids=['u0','u1','u2','u3'],
        development_seeds=[1007,1042],formal_seeds=[7,42,123,2026,3407],
        backbones={b:{} for b in h.BACKBONES},H_training_draw_count=128,
        H_training_rows=128,H_data_freeze_sha256='freeze',H_qualification_sha256='gate',
        h_code_bundle_sha256='code',H_microbatches={b:{'AMP':32,'FP32':16} for b in h.BACKBONES},runs={})


def history(epochs,value=0.8,late_gain=0):
    return [dict(epoch=i+1,dev={'macro_auc':value+(late_gain if i>=epochs-5 else 0)}) for i in range(epochs)]


def test_registered_counts_pair_order_and_shared_microbatch(tmp_path):
    c=config(tmp_path);tasks=h.standard_development_tasks(c)
    assert len(tasks)==20
    assert sum(t['backbone']=='dinov2_vit_b14' for t in tasks.values())==12
    ids=list(tasks)
    assert tasks[ids[0]]['history']=='B' and tasks[ids[1]]['history']=='AB'
    assert tasks[ids[0]]['seed']==tasks[ids[1]]['seed']==1007
    assert all(t['route']=='H' and t['microbatch']==32 and t['draw_count']==128 for t in tasks.values())
    recipes={b:{'lr':1e-5,'epochs':30} for b in h.BACKBONES}
    formal=h.formal_tasks(c,recipes,recipe_sha256='recipe')
    assert len(formal)==30 and all(t['phase']=='formal' and t['recipe_frozen'] for t in formal.values())
    fallback=h.formal_tasks(c,recipes,recipe_sha256='recipe',precision='FP32',backbone='clip_vit_b16')
    assert len(fallback)==10 and all(t['microbatch']==16 and t['precision']=='FP32' for t in fallback.values())
    assert not set(formal)&set(fallback)


def test_recipe_uses_all_four_histories_and_dino_any_history_rule():
    a=dict(lr=1e-5,epochs=30,histories=[history(30,.8),history(30,.8),history(30,.8),history(30,.8)])
    b=dict(lr=3e-5,epochs=30,histories=[history(30,.99),history(30,.99),history(30,.6),history(30,.6)])
    assert h.choose_shared_recipe([a,b])['lr']==1e-5
    b['histories']=[history(30,.8009)]*4
    assert h.choose_shared_recipe([a,b])['lr']==1e-5
    with pytest.raises(ValueError,match='four'):h.choose_shared_recipe([{**a,'histories':a['histories'][:2]}])
    hs=[history(30,.8)]*3+[history(30,.8,.003)]
    assert h.extend_budget(hs,30)==45
    assert h.extend_budget([history(45,.8,.003)]*4,45)==60
    assert h.extend_budget([history(60,.8,.1)]*4,60) is None


def test_D_priority_prevents_launch_even_on_idle_GPU_and_capacity_gate(tmp_path,monkeypatch):
    c=config(tmp_path)
    (tmp_path/'queue_status.json').write_text(json.dumps({'stage':'P2_D_STANDARD_DEVELOPMENT','active':[],'owner_pid':1}))
    assert not h.d_priority_ready(tmp_path)['ready']
    memory={i:{'uuid':f'u{i}','used_mib':0} for i in range(4)}
    assert h.dispatchable_devices(c,memory,set(),set(),d_ready=False)==[]
    assert h.dispatchable_devices(c,memory,{'u1'},{'u2'},d_ready=True)==[0,3]
    memory[3]['uuid']='wrong'
    with pytest.raises(ValueError,match='UUID'):h.dispatchable_devices(c,memory,set(),set(),d_ready=True)
    (tmp_path/'queue_status.json').write_text(json.dumps({'stage':'P3_D_CACHE_COMPLETE_ANALYSIS_PENDING','active':[]}))
    assert h.d_priority_ready(tmp_path)['ready']
    assert h.capacity_check(249*1024**3,initial=True)['allowed'] is False
    assert h.capacity_check(251*1024**3,initial=True)['allowed'] is True
    assert h.capacity_check(55*1024**3,additional_bytes=10*1024**3)['allowed'] is False


def fake_pairs(root,c,backbone='clip_vit_b16',asymmetric=False,precision='AMP',omit_seed=None):
    recipes={b:dict(lr=1e-5,epochs=30) for b in h.BACKBONES}
    tasks=h.formal_tasks(c,recipes,recipe_sha256='recipe',precision=precision,backbone=backbone)
    c['runs'].update(tasks)
    for run,t in tasks.items():
        if t['seed']==omit_seed:continue
        out=root/'runs'/run;out.mkdir(parents=True)
        rows=[]
        for epoch in range(1,31):
            sample=np.arange(128,dtype=np.int64);aug=np.c_[np.zeros(128),np.ones(128),np.ones(128)]
            methods=np.array(['real']*64+['Deepfakes']*16+['Face2Face']*16+['FaceSwap']*16+['NeuralTextures']*16,dtype='U64')
            for suffix,x in [('samples',sample),('augmentations',aug),('methods',methods)]:np.save(out/f'epoch_{epoch:02d}_{suffix}.npy',x,allow_pickle=False)
            skip=asymmetric and t['seed']==7 and ((t['history']=='B' and epoch==2) or(t['history']=='AB' and epoch==3))
            event=dict(epoch=epoch,attempt=epoch-1,skip=bool(skip),success=not skip,loss=.1,
                gradient_finite=True,gradient_norm=.2,scale_before=128.,scale_after=64. if skip else 128.)
            (out/f'epoch_{epoch:02d}_updates.jsonl').write_text(json.dumps(event)+'\n')
            rows.append(dict(epoch=epoch,attempts=1,success=int(not skip),skip=int(skip),draw_count=128,
                sample_sha256=h.digest(out/f'epoch_{epoch:02d}_samples.npy'),
                augmentation_sha256=h.digest(out/f'epoch_{epoch:02d}_augmentations.npy'),
                method_counts={str(k):int(v) for k,v in zip(*np.unique(methods,return_counts=True))}))
        (out/'history.json').write_text(json.dumps(rows));(out/'final.pt').write_bytes(run.encode())
    return tasks


def verifier(cpath,run):
    c=json.loads(Path(cpath).read_text());out=Path(c['artifact_root'])/'runs'/run
    if not(out/'history.json').exists():raise ValueError('incomplete formal pair')
    history=json.loads((out/'history.json').read_text())
    return dict(status='TRAINING_COMPLETE',final_sha256=h.digest(out/'final.pt'),epochs=30,
        attempts=30,success=sum(x['success'] for x in history),skip=sum(x['skip'] for x in history))


def test_all_five_pair_ledgers_required_and_matching_arrays(tmp_path):
    c=config(tmp_path);fake_pairs(tmp_path,c);p=tmp_path/'H.json';p.write_text(json.dumps(c))
    result=h.audit_paired_ledgers(p,'clip_vit_b16',verify=verifier)
    assert result['status']=='MATCHED_SUCCESSFUL_UPDATE_STREAMS' and len(result['pairs'])==5
    assert result['asymmetric_seed_count']==0
    path=tmp_path/'runs'/h.formal_id('clip_vit_b16','AB',7)/'epoch_01_augmentations.npy'
    x=np.load(path);x[0,1]=1.05;np.save(path,x)
    with pytest.raises(ValueError,match='augmentation|hash'):h.audit_paired_ledgers(p,'clip_vit_b16',verify=verifier)


def test_equal_skip_counts_at_different_attempts_trigger_uniform_fallback(tmp_path):
    c=config(tmp_path);fake_pairs(tmp_path,c,asymmetric=True);p=tmp_path/'H.json';p.write_text(json.dumps(c))
    result=h.audit_paired_ledgers(p,'clip_vit_b16',verify=verifier)
    assert result['status']=='REQUIRES_UNIFORM_FP32_FALLBACK'
    assert result['asymmetric_seed_count']==1 and result['fallback_pair_count']==5
    assert result['pairs'][0]['B_success']==result['pairs'][0]['AB_success']==29


def test_missing_formal_seed_cannot_open_target_export_gate(tmp_path):
    c=config(tmp_path);fake_pairs(tmp_path,c,omit_seed=3407);p=tmp_path/'H.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='incomplete'):h.audit_paired_ledgers(p,'clip_vit_b16',verify=verifier)


def test_misregistered_backbone_cannot_be_relabelled_by_pair_audit(tmp_path):
    c=config(tmp_path);fake_pairs(tmp_path,c);p=tmp_path/'H.json'
    c['runs'][h.formal_id('clip_vit_b16','B',7)]['backbone']='dinov2_vit_b14';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='pair identity'):h.audit_paired_ledgers(p,'clip_vit_b16',verify=verifier)


def test_FP32_skip_or_nonfinite_is_protocol_failure(tmp_path):
    c=config(tmp_path);fake_pairs(tmp_path,c,precision='FP32',asymmetric=True);p=tmp_path/'H.json';p.write_text(json.dumps(c))
    with pytest.raises(ValueError,match='FP32'):h.audit_paired_ledgers(p,'clip_vit_b16',precision='FP32',verify=verifier)


def test_completed_queue_reuses_receipts_and_duplicate_live_owner_rejected(tmp_path,monkeypatch):
    c=config(tmp_path);tasks=h.standard_development_tasks(c);c['runs'].update(tasks)
    p=tmp_path/'H.json';p.write_text(json.dumps(c))
    queue=h.HQueue(tmp_path,Path.cwd(),config_path=p)
    called=[]
    monkeypatch.setattr(queue,'_verified_completion',lambda run:called.append(run) or {'status':'TRAINING_COMPLETE'})
    monkeypatch.setattr(queue,'_launch',lambda *args:pytest.fail('completed task relaunched'))
    queue.batch(list(tasks),'TEST_REUSE')
    assert set(called)==set(tasks) and len(queue.completed)==20
    status=tmp_path/'H_queue_status.json';status.write_text(json.dumps({'active':[{'pid':123,'run_id':'owned'}]}))
    monkeypatch.setattr(h,'owned_worker_alive',lambda item:True)
    with pytest.raises(RuntimeError,match='existing'):h.HQueue(tmp_path,Path.cwd(),config_path=p)
