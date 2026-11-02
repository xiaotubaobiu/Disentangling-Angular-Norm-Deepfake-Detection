"""Synthetic-only checks for the new paired H cache and three-seed analysis."""
import copy
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from journal_detector import data_h, trainer, protocol

ROLES=('B/source_cal','B/source_test','B/target_test','common/B/source_test',
       'common/B/target_test','common/AB/source_test','common/AB/target_test')


def module(name):
    assert importlib.util.find_spec('journal_detector.'+name) is not None, 'New journal H implementation is required'
    return importlib.import_module('journal_detector.'+name)


def pools_fixture():
    data={history:{} for history in ('B','AB')}
    for role in ROLES:
        leaf=role.split('/')[-1]
        version='AB' if role.startswith('common/AB/') else 'B'
        domain='target' if leaf=='target_test' else 'source'
        videos=[('real','r0',0,'g0',-2.),('real','r1',0,'g1',-1.),
                ('Deepfakes','f0',1,'g0',1.),('Face2Face','f1',1,'g1',2.),
                ('FaceSwap','f2',1,'g2',3.),('NeuralTextures','f3',1,'g3',4.)]
        if leaf=='source_cal':videos=[('real','cal0',0,'cal0',-3.),('real','cal1',0,'cal1',-2.)]
        elif role.startswith('B/'):
            videos += [('real','r2',0,'g4',-3.),('Deepfakes','f4',1,'g4',5.)]
        rows=[];features=[]
        for method,video,label,group,z in videos:
            uid=f'H/{domain}/{method}/{video}'
            for i in range(8):
                mid=f'{uid}/mother/{i}'
                rows.append(dict(image_path=f'/{version}/{mid}.png',label=label,method_id=method,
                    video_id=video,video_uid=uid,group_id=group,domain=domain,role=leaf,
                    mother_id=mid,mother_rgb_sha256=f'mother/{mid}',
                    rgb_sha256=f'{version}/{mid}',content_sha256=f'bytes/{version}/{mid}',
                    processing_version=version))
                features.append([z if version=='B' else -z,1.])
        frame=pd.DataFrame(rows);h=np.asarray(features,dtype=np.float32)
        for history in data:
            W=np.asarray([[0.,0.],[1. if history=='B' else -1.,0.]],dtype=np.float32)
            data[history][role]=(frame.copy(),h.copy(),W,np.zeros(2,dtype=np.float32))
    return data


def test_four_cells_and_allB_deployment_keep_same_cohort_and_source_threshold():
    analysis=module('analysis_h')
    pools=analysis.score_pools(pools_fixture())
    assert set(pools['source_test']['all_B'])=={'H00','H10'}
    all_b=pools['target_test']['all_B'];common=pools['target_test']['common']
    assert len(all_b['H00'])==8 and len(common['H00'])==6
    assert analysis.io.independent_auc(all_b['H00'].label,all_b['H00'].rank_score)==1.
    assert analysis.io.independent_auc(all_b['H10'].label,all_b['H10'].rank_score)==0.
    result=analysis.analyze_cohort(common,pools['source_cal'],
        fixed_methods=('Deepfakes','Face2Face','FaceSwap','NeuralTextures'),repeats=5)
    row=result['comparisons'].query("comparison=='(H11-H10)-(H01-H00)' and metric=='pooled_auc' and threshold_kind=='default'").iloc[0]
    assert row.estimate==2.
    assert result['thresholds']['B']<0 and result['thresholds']['AB']>0
    assert result['bootstrap']['intervals'] is None
    assert result['bootstrap']['n_fake_groups']==4


def test_H_scores_reject_different_logical_mothers_across_histories():
    data=pools_fixture()
    data['AB']['common/AB/target_test'][0].loc[0,'mother_id']='WRONG_MOTHER'
    with pytest.raises(ValueError,match='mother'):
        module('analysis_h').score_pools(data)


def test_three_seed_summary_uses_matched_pairs_sample_sd_and_preserves_NA():
    summary=module('summary_h')
    rows=pd.DataFrame(dict(seed=[7,42,123],backbone='efficientnet_b0',domain='target_test',
        cohort='all_B',aggregation='mean_logit',comparison='H10-H00',threshold_kind='source_real',
        metric='pooled_auc',estimate=[.1,.2,.3]))
    result=summary.seed_summary(rows).iloc[0]
    assert result.n_seeds==3 and result.estimate_mean==pytest.approx(.2)
    assert result.estimate_sample_sd==pytest.approx(.1)
    rows.loc[2,'estimate']=np.nan
    result=summary.seed_summary(rows).iloc[0]
    assert result.n_valid_seeds==2 and np.isnan(result.estimate_mean)
    with pytest.raises(ValueError,match='three|3'):
        summary.seed_summary(rows.iloc[:2])


def test_coverage_uses_full_unique_candidate_denominator_and_quarantine(tmp_path):
    analysis=module('analysis_h')
    rows=pd.DataFrame(dict(video_uid=['r','f0','f1','f2'],label=[0,1,1,1],
        method_id=['real','m','m','m'],dataset='DFD',role='target_test',processed=True,
        decode_ok=[True,True,False,True],A=[True,True,False,True],B=[True,True,False,True],
        AB=[True,True,False,True],common=[True,True,False,True],quarantined=[False,False,False,True]))
    coverage=tmp_path/'coverage.csv';rows.to_csv(coverage,index=False)
    inventory=tmp_path/'inventory.json';trainer.save_json(inventory,{'records':rows[['video_uid','label','method_id','role','dataset']].to_dict('records')})
    freeze={'metadata':{'coverage':{'path':str(coverage),'sha256':trainer.digest(coverage),
        'candidate_videos':4,'schema':'JOURNAL_H_VIDEO_COVERAGE_V1','denominator_scope':'all candidate videos'},
        'inventory':{'path':str(inventory),'sha256':trainer.digest(inventory),'candidate_videos':4}}}
    report=analysis.coverage_report(tmp_path,freeze)
    assert report['candidate_videos']==4 and report['real_candidate_videos']==1
    target=next(r for r in report['by_role_method'] if r['method_id']=='m')
    assert target['candidate_videos']==3 and target['geometric_B_videos']==2
    assert target['active_B_videos']==1 and target['quarantined_videos']==1
    assert target['active_B_fraction']==pytest.approx(1/3)
    rows.loc[len(rows)]=rows.iloc[0];rows.to_csv(coverage,index=False)
    freeze['metadata']['coverage']['sha256']=trainer.digest(coverage)
    with pytest.raises(ValueError,match='unique|duplicate|candidate'):
        analysis.coverage_report(tmp_path,freeze)


@pytest.fixture
def formal_H(tmp_path,monkeypatch):
    from PIL import Image
    cache=module('cache_h')
    source=tmp_path/'data/H';source.mkdir(parents=True)
    lock=tmp_path/'configs/requirements.execution.lock.txt';lock.parent.mkdir();lock.write_text('SYNTHETIC_ONLY\n')
    init=tmp_path/'initialization.bin';init.write_bytes(b'SYNTHETIC_ONLY')
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/journal.json').read_text())
    config.update(protocol=protocol.PROTOCOL,artifact_root=str(tmp_path),h_formal_seeds=[7,42,123],
        development_seeds=[1007,1042],runs={},workers=0,cache_workers=0,cache_batch_size=128)
    for spec in config['backbones'].values():spec.update(weight_path=str(init),weight_sha256=trainer.digest(init))
    data=pools_fixture();tables={}
    for role in ROLES:
        frame=data['B'][role][0].copy()
        for i,row in frame.iterrows():
            image=tmp_path/'images'/role/(str(i)+'.png');image.parent.mkdir(parents=True,exist_ok=True)
            Image.new('RGB',(4,4),(25+int(row.label)*150+i%8,)*3).save(image)
            frame.loc[i,'image_path']=str(image);frame.loc[i,'content_sha256']=trainer.digest(image)
        tables[role]=frame
    # The common-B rows must reference the exact all-B input files.
    for leaf in ('source_test','target_test'):
        all_b=tables['B/'+leaf].set_index('mother_id')
        q=tables['common/B/'+leaf]
        for key in ('image_path','content_sha256'):
            q[key]=q.mother_id.map(all_b[key])
    for history in ('B','AB'):
        for role in ('train','dev'):
            q=tables['common/'+history+'/source_test'].copy();q['role']=role
            q['sampling_mass']=1/len(q);tables[history+'/'+role]=q
    for key,frame in tables.items():
        path=source/(key+'.parquet');path.parent.mkdir(parents=True,exist_ok=True);frame.to_parquet(path,index=False)
    freeze={'status':'FROZEN','schema':'JOURNAL_H_DATA_FREEZE_V1',
        'tables':{key:{'sha256':trainer.digest(source/(key+'.parquet'))} for key in tables},
        'metadata':{},'scope':'SOURCE_TRAIN_DEV_CAL'}
    trainer.save_json(source/'train_dev_freeze.json',freeze)
    trainer.save_json(source/'freeze.json',dict(freeze,scope='FULL_SOURCE_TARGET'))
    def validate(root,source_only=False,**kwargs):
        return json.loads((Path(root)/'data/H'/('train_dev_freeze.json' if source_only else 'freeze.json')).read_text())
    monkeypatch.setattr(data_h,'validate_h_freeze',validate)
    recipe=tmp_path/'recipes/H_efficientnet_b0.json'
    trainer.save_json(recipe,dict(schema='EPOCH_RECIPE_V2',status='FROZEN',route='H',backbone='efficientnet_b0',
        chosen_epochs=9,target_development_used=False,paired_histories=['B','AB'],
        data_freeze_sha256=trainer.digest(source/'train_dev_freeze.json')))
    for seed in (7,42,123):
        pair=tmp_path/'pairs'/str(seed)/'state.pt';pair.parent.mkdir(parents=True);pair.write_bytes(b'PAIRED_SYNTHETIC_STATE')
        for history in ('B','AB'):
            run=f'H_formal_efficientnet_b0_seed{seed}_{history}'
            task=dict(route='H',phase='formal',recipe_frozen=True,history=history,backbone='efficientnet_b0',
                seed=seed,epochs=9,lr=.0001,microbatch=128,precision='AMP',
                train_manifest=f'data/H/{history}/train.parquet',dev_manifest='data/H/B/dev.parquet',
                epoch_recipe_path=str(recipe.relative_to(tmp_path)),epoch_recipe_sha256=trainer.digest(recipe))
            config['runs'][run]=task
    config_path=tmp_path/'configs/h_execution.json';trainer.save_json(config_path,config)
    for run,task in config['runs'].items():
        identity=trainer.training_identity(config,task);out=tmp_path/'runs'/run;out.mkdir(parents=True)
        torch.save(dict(model={},identity=identity,epoch=2,weights='ema',selection='paired_source_dev_B_mean_probability_auc_ema'),out/'best_model.pth')
        pair=tmp_path/'pairs'/str(task['seed'])/'state.pt'
        trainer.save_json(out/'complete.json',dict(status='TRAINING_COMPLETE',route='H',phase='formal',
            history=task['history'],identity=identity,epochs=6,max_epochs=9,best_epoch=2,
            selection='paired_source_dev_B_mean_probability_auc_ema',target_used=False,target_development_used=False,
            epoch_recipe_sha256=task['epoch_recipe_sha256'],best_sha256=trainer.digest(out/'best_model.pth'),
            pair_state_path=str(pair),pair_state_sha256=trainer.digest(pair)))
    return config_path,config


def test_H_context_accepts_only_new_frozen_source_selected_pair(formal_H):
    path,config=formal_H
    run=next(iter(config['runs']))
    base,_,task=module('cache_h')._context(path,run)
    assert base['route']=='H' and base['history']=='B' and base['best_epoch']==2
    assert base['target_development_used'] is False
    assert base['training_identity']['data_freeze']==trainer.digest(path.parent.parent/'data/H/train_dev_freeze.json')


def test_H_per_backbone_config_can_extend_without_changing_initial_config(formal_H):
    path,config=formal_H;run=next(iter(config['runs']))
    extension=path.with_name('h_execution_efficientnet_b0.json');trainer.save_json(extension,config)
    base,_,_=module('cache_h')._context(extension,run)
    assert base['config_path']==str(extension)
    wrong=path.with_name('h_execution_clip_vit_b16.json');trainer.save_json(wrong,config)
    with pytest.raises(ValueError):module('cache_h')._context(wrong,run)


@pytest.mark.parametrize('change',['development','D_route','partner_epoch','epoch_recipe'])
def test_H_cache_rejects_nonformal_or_unpaired_identity(formal_H,change):
    path,config=formal_H;run=next(iter(config['runs']));root=path.parent.parent
    if change=='development':config['runs'][run].update(phase='development',recipe_frozen=False,seed=1007);trainer.save_json(path,config)
    elif change=='D_route':config['runs'][run]['route']='D';trainer.save_json(path,config)
    elif change=='partner_epoch':
        partner=next(key for key,t in config['runs'].items() if t['seed']==7 and t['history']=='AB')
        receipt=root/'runs'/partner/'complete.json';data=json.loads(receipt.read_text());data['best_epoch']=3;trainer.save_json(receipt,data)
    else:
        recipe=root/config['runs'][run]['epoch_recipe_path'];recipe.write_text(recipe.read_text()+'\n')
    with pytest.raises(ValueError):module('cache_h')._context(path,run)


def test_synthetic_CPU_cache_analysis_three_seed_summary_and_corruption(formal_H,monkeypatch):
    cache=module('cache_h');analysis=module('analysis_h');summary=module('summary_h')
    from journal_detector import models
    path,config=formal_H;root=path.parent.parent
    class Encoder(torch.nn.Module):
        def forward(self,x):return x.mean((1,2,3))[:,None].expand(-1,1280).contiguous()
    class Detector(torch.nn.Module):
        def __init__(self):super().__init__();self.encoder=Encoder();self.classifier=torch.nn.Linear(1280,2)
    monkeypatch.setattr(models,'build_model',lambda *args:Detector())
    full=root/'data/H/freeze.json';freeze=json.loads(full.read_text());candidates=[]
    for role in ROLES:
        table=root/f'data/H/{role}.parquet';frame=pd.read_parquet(table)
        if role.endswith('target_test'):
            frame.loc[frame.label.eq(1),'method_id']='DeepFakeDetection';frame.to_parquet(table,index=False)
            freeze['tables'][role]['sha256']=trainer.digest(table)
    for leaf in ('source_cal','source_test','target_test'):
        frame=pd.read_parquet(root/f'data/H/B/{leaf}.parquet')
        common=set(pd.read_parquet(root/f'data/H/common/B/{leaf}.parquet').video_uid) if leaf!='source_cal' else set(frame.video_uid)
        for row in frame.drop_duplicates('video_uid').to_dict('records'):
            candidates.append({key:row[key] for key in ('video_uid','label','method_id','role')})
            candidates[-1].update(dataset='DFD' if leaf=='target_test' else 'FF++',processed=True,decode_ok=True,
                A=True,B=True,AB=row['video_uid'] in common,common=row['video_uid'] in common,quarantined=False)
    coverage=root/'data/H/coverage.csv';pd.DataFrame(candidates).to_csv(coverage,index=False)
    inventory=root/'data/H/inventory.json';trainer.save_json(inventory,candidates)
    freeze['metadata'].update(coverage=dict(path=str(coverage),sha256=trainer.digest(coverage),candidate_videos=len(candidates),
        schema='JOURNAL_H_VIDEO_COVERAGE_V1',denominator_scope='SYNTHETIC_ONLY_COMPLETE_CANDIDATES'),
        inventory=dict(path=str(inventory),sha256=trainer.digest(inventory),candidate_videos=len(candidates)))
    trainer.save_json(full,freeze)
    for run,task in config['runs'].items():
        out=root/'runs'/run;checkpoint=torch.load(out/'best_model.pth',weights_only=False);model=Detector()
        with torch.no_grad():
            model.classifier.weight.zero_();model.classifier.bias.zero_();model.classifier.weight[1,0]=1 if task['history']=='B' else -1
        checkpoint['model']=model.state_dict();torch.save(checkpoint,out/'best_model.pth')
        receipt=json.loads((out/'complete.json').read_text());receipt['best_sha256']=trainer.digest(out/'best_model.pth');trainer.save_json(out/'complete.json',receipt)
    for run in config['runs']:
        result=cache.extract(path,run,device='cpu')
        assert result['status']=='CACHE_COMPLETE' and set(result['roles'])==set(ROLES)
    for seed in (7,42,123):assert analysis.analyze_pair(path,'efficientnet_b0',seed)['status']=='ANALYSIS_COMPLETE'
    report=summary.summarize_backbone(path,'efficientnet_b0')
    assert report['training_seed_pairs']==3 and report['target_development_used'] is False
    folder=root/'analysis/H/efficientnet_b0/three_seed_summary'
    q=pd.read_csv(folder/'three_seed_comparisons.csv')
    primary=q.query("domain=='target_test' and cohort=='all_B' and aggregation=='mean_logit' and comparison=='H10-H00' and metric=='pooled_auc'")
    assert primary.n_seeds.eq(3).all() and primary.estimate_mean.eq(-1.).all() and primary.estimate_sample_sd.eq(0.).all()
    assert json.loads((folder/'coverage.json').read_text())['candidate_videos']==len(candidates)
    monkeypatch.setattr(models,'build_model',lambda *args:(_ for _ in ()).throw(AssertionError('resume must not rebuild a model')))
    assert summary.summarize_backbone(path,'efficientnet_b0')==report
    assert analysis.analyze_pair(path,'efficientnet_b0',7)['status']=='ANALYSIS_COMPLETE'
    first=next(iter(config['runs']));assert cache.extract(path,first,device='cpu')['status']=='CACHE_COMPLETE'
    chunk=root/f'features/H/{first}/B/source_test/chunk_000000.npy'
    with chunk.open('ab') as stream:stream.write(b'CORRUPTION')
    with pytest.raises(ValueError,match='hash'):cache.verify_cache(chunk.parent)
