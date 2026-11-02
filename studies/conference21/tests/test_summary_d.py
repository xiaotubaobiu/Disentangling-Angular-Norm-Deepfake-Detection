"""Synthetic CPU-only tests for the frozen D five-seed assembly contract."""
import importlib.util
import numpy as np
import pandas as pd
import pytest

if importlib.util.find_spec('conference21.summary_d') is not None:
    from conference21 import summary_d as module
else:
    module = None


def test_seed_summary_uses_five_models_sample_sd_and_keeps_NA():
    # A wrong ddof, pooled model count or valid-subset mean must fail this test.
    assert module is not None, 'D final five-seed summary entry has not been implemented'
    rows=pd.DataFrame(dict(seed=[7,42,123,2026,3407],backbone='clip_vit_b16',
        domain='target_full',grouping='correct',representation='raw',aggregation='mean_logit',
        comparison='cell',threshold_kind='ranking',metric='pooled_auc',estimate=[.1,.2,.3,.4,.5]))
    result=module.seed_summary(rows,['backbone','domain','grouping','representation','aggregation','comparison','threshold_kind','metric'])
    assert result.iloc[0].estimate_mean==pytest.approx(.3)
    assert result.iloc[0].estimate_sample_sd==pytest.approx(.15811388300841897)
    assert result.iloc[0].n_seeds==5
    rows.loc[4,'estimate']=float('nan')
    result=module.seed_summary(rows,['backbone','domain','grouping','representation','aggregation','comparison','threshold_kind','metric'])
    assert result.iloc[0].n_valid_seeds==4 and np.isnan(result.iloc[0].estimate_mean)
    assert np.isnan(result.iloc[0].estimate_sample_sd)


def test_effect_summary_does_not_turn_four_valid_seeds_into_five():
    assert module is not None, 'D final five-seed summary entry has not been implemented'
    rows=pd.DataFrame([dict(seed=s,backbone='clip_vit_b16',domain='target_full',aggregation='mean_logit',
        delta_correct=.1,delta_collision=(-.1 if s==7 else .1),interaction=(-.2 if s==7 else 0.),valid=True,reversal=s==7)
        for s in [7,42,123,2026,3407]])
    result=module.effect_summary(rows).iloc[0]
    assert result.n_seed_reversals==1 and not result.mean_reversal
    assert result.delta_collision_mean==pytest.approx(.06)
    rows.loc[4,['delta_correct','delta_collision','interaction']]=float('nan');rows.loc[4,'valid']=False
    result=module.effect_summary(rows).iloc[0]
    assert result.n_valid_seeds==4 and np.isnan(result.delta_correct_mean)
    assert pd.isna(result.mean_reversal)


@pytest.mark.parametrize('change',['missing','duplicate','unexpected','development','other_backbone','unfrozen'])
def test_registry_rejects_substituted_or_incomplete_seeds_without_output(tmp_path,change):
    import json
    assert module is not None, 'D final five-seed summary entry has not been implemented'
    tasks={f'r{s}':dict(seed=s,phase='formal',route='D',recipe_frozen=True,backbone='clip_vit_b16') for s in [7,42,123,2026,3407]}
    if change=='missing':tasks.pop('r3407')
    elif change=='duplicate':tasks['duplicate']=dict(tasks['r7'])
    elif change=='unexpected':tasks['r3407']['seed']=999
    elif change=='development':tasks['r3407']['phase']='development'
    elif change=='other_backbone':tasks['r3407']['backbone']='dinov2_vit_b14'
    else:tasks['r3407']['recipe_frozen']=False
    p=tmp_path/'configs/execution.json';p.parent.mkdir();p.write_text(json.dumps({'artifact_root':str(tmp_path),'runs':tasks,'formal_seeds':[7,42,123,2026,3407]}))
    with pytest.raises(ValueError):module.summary_status(p,'clip_vit_b16')
    assert not(tmp_path/'analysis').exists()


def test_status_does_not_create_output_or_claim_marker_provenance(tmp_path):
    import json
    assert module is not None, 'D final five-seed summary entry has not been implemented'
    tasks={f'r{s}':dict(seed=s,phase='formal',route='D',recipe_frozen=True,backbone='clip_vit_b16') for s in [7,42,123,2026,3407]}
    p=tmp_path/'configs/execution.json';p.parent.mkdir();p.write_text(json.dumps({'artifact_root':str(tmp_path),'runs':tasks,'formal_seeds':[7,42,123,2026,3407]}))
    result=module.summary_status(p,'clip_vit_b16')
    assert result['status']=='WAITING_D_FORMAL_ANALYSES' and result['analysis_markers_present']==0
    assert result['expected_training_seeds']==5 and result['marker_only'] is True
    assert not(tmp_path/'analysis').exists()


def test_CAL_repeats_reduce_within_seed_and_unavailable_budget_stays_NA():
    assert module is not None, 'D final five-seed summary entry has not been implemented'
    rows=[]
    for seed in [7,42,123,2026,3407]:
        common=dict(seed=seed,backbone='clip_vit_b16',domain='target_test',representation='raw',aggregation='mean_logit')
        rows += [dict(**common,calibration_kind='default',budget=0,repeat=-1,status='ok',fpr=.3,tpr=.8,pooled_ba=.75),
                 dict(**common,calibration_kind='source_real',budget=140,repeat=-1,status='ok',fpr=.2,tpr=.8,pooled_ba=.8)]
        for budget in [10,20,40,80]:
            for repeat in range(200):rows.append(dict(**common,calibration_kind='target_real',budget=budget,repeat=repeat,
                status='unavailable' if budget==80 else 'ok',fpr=np.nan if budget==80 else .1,tpr=np.nan if budget==80 else .8,pooled_ba=np.nan if budget==80 else .85))
    points,comparisons=module.calibration_tables(pd.DataFrame(rows))
    part=points.query('budget==20 and metric=="fpr"');assert len(part)==5
    assert part.n_subsets.eq(200).all() and np.allclose(part.estimate,.1)
    assert points.query('budget==80 and metric=="fpr"').estimate.isna().all()
    difference=comparisons.query('budget==20 and metric=="fpr"')
    assert len(difference)==5 and np.allclose(difference.estimate,-.1)
    summary=module.seed_summary(difference,['backbone','domain','representation','aggregation','budget','comparison','threshold_kind','metric'])
    assert summary.iloc[0].n_seeds==5 and summary.iloc[0].estimate_mean==pytest.approx(-.1)


@pytest.fixture
def experiment(tmp_path,monkeypatch):
    """Actual CPU cache/analysis files; replace only pretrained model construction."""
    import hashlib,json,torch
    from PIL import Image
    from conference21 import trainer,cache_d,models,analysis_d
    class Encoder(torch.nn.Module):
        def forward(self,x):return x.mean(dim=(1,2,3))[:,None].expand(-1,512).contiguous()
    class Detector(torch.nn.Module):
        def __init__(self):
            super().__init__();self.encoder=Encoder();self.classifier=torch.nn.Linear(512,2)
    root=tmp_path/'synthetic';data=root/'data/D';data.mkdir(parents=True);init=root/'init.bin';init.write_bytes(b'SYNTHETIC_ONLY')
    lock=root/'configs/requirements.execution.lock.txt';lock.parent.mkdir();lock.write_text('syntheticCPUlocked\n')
    gate=root/'qualification/D_gate.json';trainer.save_json(gate,dict(status='PASS',training_code_sha256=trainer.code_hash()))
    for role in cache_d.ROLES:
        rows=[]
        for method,video,label,brightness,group in [('real','r0',0,20,'g0'),('real','r1',0,160,'g1'),('A','p',1,80,'g0'),('B','p',1,240,'g1')]:
            if role in ('source_cal','target_cal') and label:continue
            for n in range(8):
                image=root/'images'/role/method/video/f'{n}.png';image.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(8,8),(brightness+n,)*3).save(image)
                uid=f'{role}/{method}/{video}';rows.append(dict(sample_id=f'{uid}/{n}',image_path=str(image),label=label,
                    method_id=method,video_id=video,video_uid=uid,group_id=group,domain='ff' if role in ('train','dev','source_cal','source_test') else 'cdf',role=role,
                    rgb_sha256=trainer.digest(image),content_sha256=trainer.digest(image),bytes=image.stat().st_size,mtime_ns=image.stat().st_mtime_ns))
        table=pd.DataFrame(rows)
        if role=='train':table['sampling_mass']=1/len(table)
        table.to_parquet(data/f'{role}.parquet',index=False)
    trainer.save_json(data/'freeze.json',{'status':'FROZEN','method_count':21,'method_registry':{'FS9':['A','B']+[f'X{i}' for i in range(7)],'FR12':[f'Y{i}' for i in range(12)]},
        'manifest_sha256':{p.name:trainer.digest(p) for p in data.glob('*.parquet')},'cdf_calibration_scope':{'status':'SYNTHETIC_ONLY','ancestor_independence_certified':False}})
    tasks={f'D_formal_clip_vit_b16_seed{s}':dict(backbone='clip_vit_b16',seed=s,epochs=15,phase='formal',route='D',recipe_frozen=True,
        train_manifest='data/D/train.parquet',dev_manifest='data/D/dev.parquet',qualification_gate='qualification/D_gate.json',microbatch=128,lr=1e-5,precision="AMP") for s in [7,42,123,2026,3407]}
    from conference21.protocol import PROTOCOL,LRS
    all_tasks=dict(tasks)
    for b,lr in LRS.items():
        if b=='clip_vit_b16':continue
        for seed in [7,42,123,2026,3407]:
            all_tasks[f'D_formal_{b}_seed{seed}']=dict(next(iter(tasks.values())),backbone=b,lr=lr,seed=seed)
    config=dict(protocol=PROTOCOL,artifact_root=str(root),formal_seeds=[7,42,123,2026,3407],workers=0,cache_workers=0,cache_batch_size=32,cache_chunk_rows=4096,effective_batch=128,
        runs=all_tasks,backbones={b:dict(weight_path=str(init),weight_sha256=trainer.digest(init)) for b in LRS})
    path=root/'configs/execution.json';trainer.save_json(path,config)
    monkeypatch.setattr(models,'build_model',lambda *args:Detector())
    for run,task in tasks.items():
        out=root/'runs'/run;out.mkdir(parents=True);model=Detector()
        with torch.no_grad():model.classifier.weight[0].zero_();model.classifier.weight[1].fill_(.001);model.classifier.bias.zero_()
        identity=trainer.training_identity(config,task)
        torch.save(dict(model=model.state_dict(),identity=identity,epoch=1,weights='ema'),out/'best_model.pth')
        trainer.save_json(out/'complete.json',dict(status='TRAINING_COMPLETE',identity=identity,
            best_sha256=trainer.digest(out/'best_model.pth'),epochs=6,max_epochs=15,best_epoch=1,
            selection='source_dev_pooled_mean_probability_auc_ema',target_used=False))
        cache_d.extract(path,run,device='cpu');analysis_d.analyze_run(path,run)
    return root,path,tasks


def test_real_CPU_cache_analysis_assembly_and_verified_resume(experiment):
    root,path,tasks=experiment
    assert hasattr(module,'summarize_backbone'), 'D verified filesystem summary entry is missing'
    result=module.summarize_backbone(path,'clip_vit_b16');assert result['training_seed_count']==5 and result['backbone']=='clip_vit_b16'
    out=root/'analysis/D/clip_vit_b16/five_seed_summary';metrics=pd.read_csv(out/'per_seed_metrics.csv')
    assert len(metrics)==80 and set(metrics.domain)=={'source_test','target_full'}
    summary=pd.read_csv(out/'five_seed_metrics.csv');assert summary.n_seeds.eq(5).all()
    assert summary.query('metric=="macro_auc"').estimate_mean.isna().all()
    cal=pd.read_csv(out/'five_seed_calibration.csv');assert cal.query('budget==20 and metric=="fpr"').n_valid_seeds.eq(0).all()
    assert module.summarize_backbone(path,'clip_vit_b16')==result


def test_missing_analysis_does_not_start_scoring_or_write_summary(experiment,monkeypatch):
    root,path,tasks=experiment;last=list(tasks)[-1];(root/'analysis/D'/last/'complete.json').unlink()
    assert hasattr(module,'summarize_backbone'), 'D verified filesystem summary entry is missing'
    with pytest.raises(ValueError,match='incomplete|five|missing'):module.summarize_backbone(path,'clip_vit_b16')
    assert not(root/'analysis/D/clip_vit_b16/five_seed_summary').exists()


@pytest.mark.parametrize('change',['artifact','omitted_inventory','final_checkpoint','config','manifest'])
def test_rejects_stale_or_corrupt_input_provenance(experiment,change):
    import json
    root,path,tasks=experiment;run=list(tasks)[0];out=root/'analysis/D'/run
    if change=='artifact':(out/'source_test/metrics.csv').write_text('corrupt\n')
    elif change=='omitted_inventory':
        p=out/'complete.json';j=json.loads(p.read_text());j['artifact_sha256'].pop('source_test/metrics.csv');p.write_text(json.dumps(j))
    elif change=='final_checkpoint':
        with (root/'runs'/run/'best_model.pth').open('ab') as f:f.write(b'tamper')
    elif change=='config':
        j=json.loads(path.read_text());j['cache_batch_size']=17;path.write_text(json.dumps(j))
    else:
        p=root/'data/D/target_test.parquet';q=pd.read_parquet(p);q.loc[0,'sample_id']='changed';q.to_parquet(p,index=False)
    assert hasattr(module,'summarize_backbone'), 'D verified filesystem summary entry is missing'
    with pytest.raises(ValueError):module.summarize_backbone(path,'clip_vit_b16')
    assert not(root/'analysis/D/clip_vit_b16/five_seed_summary/complete.json').exists()


def test_summary_corruption_and_extra_file_rejected_on_reuse(experiment):
    root,path,tasks=experiment
    assert hasattr(module,'summarize_backbone'), 'D verified filesystem summary entry is missing'
    module.summarize_backbone(path,'clip_vit_b16');out=root/'analysis/D/clip_vit_b16/five_seed_summary';(out/'unexpected.txt').write_text('extra')
    with pytest.raises(ValueError,match='inventory'):module.summarize_backbone(path,'clip_vit_b16')
    (out/'unexpected.txt').unlink();(out/'five_seed_metrics.csv').write_text('tampered\n')
    with pytest.raises(ValueError,match='hash'):module.summarize_backbone(path,'clip_vit_b16')


def test_input_change_during_summary_cannot_commit_complete(experiment,monkeypatch):
    root,path,tasks=experiment;original=module.seed_summary;changed=[]
    def mutate(rows,columns):
        result=original(rows,columns)
        if not changed:
            p=root/'analysis/D'/list(tasks)[0]/'source_test/metrics.csv';p.write_text(p.read_text()+'\n');changed.append(True)
        return result
    monkeypatch.setattr(module,'seed_summary',mutate)
    with pytest.raises(ValueError,match='hash|changed'):module.summarize_backbone(path,'clip_vit_b16')
    assert not(root/'analysis/D/clip_vit_b16/five_seed_summary/complete.json').exists()


def test_duplicate_launcher_respects_existing_summary_lock(experiment):
    from conference21.trainer import exclusive_file
    root,path,tasks=experiment
    assert hasattr(module,'summarize_backbone'), 'D verified filesystem summary entry is missing'
    with exclusive_file(root/'analysis/D/.locks/clip_vit_b16_five_seed_summary.lock'):
        with pytest.raises((RuntimeError,BlockingIOError)):module.summarize_backbone(path,'clip_vit_b16')
    assert not(root/'analysis/D/clip_vit_b16/five_seed_summary/complete.json').exists()



def test_output_symlink_cannot_write_outside_registered_root(experiment,tmp_path):
    root,path,tasks=experiment;outside=tmp_path/'outside';outside.mkdir()
    (root/'analysis/D/clip_vit_b16').symlink_to(outside,target_is_directory=True)
    with pytest.raises(ValueError,match='outside|unsafe|symlink'):module.summarize_backbone(path,'clip_vit_b16')
    assert not list(outside.iterdir())


def test_cached_summary_rechecks_input_change_after_lock(experiment,monkeypatch):
    from contextlib import contextmanager
    root,path,tasks=experiment;module.summarize_backbone(path,'clip_vit_b16');original=module.exclusive_file
    @contextmanager
    def changing_lock(p):
        with original(p):
            target=root/'analysis/D'/list(tasks)[0]/'source_test/metrics.csv';target.write_text(target.read_text()+'\n')
            yield
    monkeypatch.setattr(module,'exclusive_file',changing_lock)
    with pytest.raises(ValueError,match='hash|changed'):module.summarize_backbone(path,'clip_vit_b16')


def test_partial_summary_resume_and_identity_mismatch_preserve_artifacts(experiment):
    import json
    root,path,tasks=experiment;module.summarize_backbone(path,'clip_vit_b16');out=root/'analysis/D/clip_vit_b16/five_seed_summary'
    (out/'complete.json').unlink();(out/'per_seed_metrics.csv').write_text('partial')
    assert module.summarize_backbone(path,'clip_vit_b16')['training_seed_count']==5
    assert len(pd.read_csv(out/'per_seed_metrics.csv'))==80
    (out/'complete.json').unlink();identity=out/'identity.json';j=json.loads(identity.read_text());j['seeds']=[7];identity.write_text(json.dumps(j))
    with pytest.raises(ValueError,match='identity'):module.summarize_backbone(path,'clip_vit_b16')
    assert not(out/'complete.json').exists()


def test_one_oracle_numeric_NA_does_not_erase_operating_counts(experiment):
    import json
    from conference21 import analysis_d
    root,path,tasks=experiment;run=list(tasks)[-1];out=root/'analysis/D'/run;oracle=out/'rq4/raw_mean_logit/oracle.json'
    oracle.write_text(json.dumps({'status':'UNRESOLVED_NUMERIC_ORDERING','reason':'synthetic unresolved ordering'}));p=out/'complete.json';j=json.loads(p.read_text());j['status']='ANALYSIS_COMPLETE_WITH_NUMERIC_NA';j['artifact_sha256']['rq4/raw_mean_logit/oracle.json']=analysis_d._digest(oracle);p.write_text(json.dumps(j))
    result=module.summarize_backbone(path,'clip_vit_b16');assert result['status']=='SUMMARY_COMPLETE_WITH_NUMERIC_NA'
    folder=root/'analysis/D/clip_vit_b16/five_seed_summary';table=pd.read_csv(folder/'five_seed_oracles.csv')
    q=table.query('representation=="raw" and aggregation=="mean_logit" and comparison=="fpr_le_0.05" and metric=="fpr"')
    assert q.n_valid_seeds.iloc[0]==4 and np.isnan(q.estimate_mean.iloc[0])
    operating=pd.read_csv(folder/'per_seed_calibration.csv');q=operating.query('seed==3407 and representation=="raw" and aggregation=="mean_logit" and comparison=="source_real" and metric=="fpr"')
    assert q.estimate.notna().all()



@pytest.mark.parametrize('kind',['partial_csv','dangling_csv','complete_csv','complete_marker','input_marker','dangling_lock'])
def test_all_file_level_links_are_rejected_without_outside_write(experiment,tmp_path,kind):
    import json
    root,path,tasks=experiment;outside=tmp_path/'external';original=b'PRESERVE_OUTSIDE'
    if kind in ('partial_csv','dangling_csv','complete_csv','complete_marker'):
        module.summarize_backbone(path,'clip_vit_b16');out=root/'analysis/D/clip_vit_b16/five_seed_summary'
        if kind in ('partial_csv','dangling_csv'):(out/'complete.json').unlink()
        name='complete.json' if kind=='complete_marker' else 'per_seed_metrics.csv';target=out/name
        if kind in ('complete_csv','complete_marker'):original=target.read_bytes()
        target.unlink()
        if kind!='dangling_csv':outside.write_bytes(original)
        target.symlink_to(outside)
    elif kind=='input_marker':
        target=root/'analysis/D'/list(tasks)[0]/'complete.json';original=target.read_bytes();outside.write_bytes(original);target.unlink();target.symlink_to(outside)
    else:
        target=root/'analysis/D/.locks/clip_vit_b16_five_seed_summary.lock';target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(outside)
    with pytest.raises(ValueError,match='unsafe|symlink|regular|outside'):module.summarize_backbone(path,'clip_vit_b16')
    if kind in ('dangling_csv','dangling_lock'):assert not outside.exists()
    else:assert outside.read_bytes()==original


def test_receipt_and_hash_use_one_stable_snapshot(experiment,monkeypatch):
    import json
    root,path,tasks=experiment;original=module._accept;changed=[]
    def update_previous_receipt(*args):
        result=original(*args)
        if args[-1]==42 and not changed:
            marker=root/'analysis/D'/list(tasks)[0]/'complete.json';j=json.loads(marker.read_text());j['status']='ANALYSIS_COMPLETE_WITH_NUMERIC_NA';marker.write_text(json.dumps(j));changed.append(True)
        return result
    monkeypatch.setattr(module,'_accept',update_previous_receipt)
    with pytest.raises(ValueError,match='changed|snapshot'):module.summarize_backbone(path,'clip_vit_b16')
    assert not(root/'analysis/D/clip_vit_b16/five_seed_summary/complete.json').exists()


def test_marker_disappearance_never_enters_per_model_scoring_or_writes(experiment,monkeypatch):
    from conference21 import cache_d
    root,path,tasks=experiment;run=list(tasks)[0];out=root/'analysis/D'/run;original=cache_d._context;changed=[];scoring=[]
    before={p.name:p.stat().st_mtime_ns for p in out.iterdir() if p.is_file() and p.name!='complete.json'}
    def remove_after_initial_check(config,key):
        result=original(config,key)
        if key==run and not changed:(out/'complete.json').unlink();changed.append(True)
        return result
    def forbidden_load(*args,**kwargs):scoring.append(True);raise AssertionError('summary attempted per-model scoring')
    monkeypatch.setattr(cache_d,'_context',remove_after_initial_check);monkeypatch.setattr(cache_d,'load_features',forbidden_load)
    with pytest.raises(ValueError,match='changed|missing|snapshot|disappeared'):module.summarize_backbone(path,'clip_vit_b16')
    assert not scoring
    assert before=={p.name:p.stat().st_mtime_ns for p in out.iterdir() if p.is_file() and p.name!='complete.json'}
