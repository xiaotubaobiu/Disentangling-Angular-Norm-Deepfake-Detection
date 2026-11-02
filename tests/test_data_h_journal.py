"""H uses logical mother identity and the same balanced draws for B and A→B."""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def data():
    return importlib.import_module('journal_detector.data_h')


def pool_rows():
    rows=[]
    for method in ['real','Deepfakes','Face2Face','FaceSwap','NeuralTextures']:
        for video in range(2):
            for slot in range(8):
                rows.append({'method_id':method,'label':int(method!='real'),
                    'video_uid':f'FFPP/source/{method}/{video}',
                    'mother_id':f'{method}/{video}/{slot}','sample_id':f'{method}/{video}/{slot}',
                    'rgb_sha256':'deliberately_shared','image_path':f'{method}/{video}/{slot}.png'})
    return pd.DataFrame(rows)


def test_training_retains_all_mothers_even_when_output_rgb_coincides():
    h = data()
    weighted = h.weight_training_pool(pool_rows())
    assert len(weighted) == 80
    assert not weighted.mother_id.duplicated().any()
    assert np.isclose(weighted.sampling_mass.sum(),1)
    assert weighted.groupby('method_id').sampling_mass.sum().to_dict() == {
        'Deepfakes':.125,'Face2Face':.125,'FaceSwap':.125,'NeuralTextures':.125,'real':.5}
    assert json.loads(weighted.iloc[0].method_contributions) == {'real':.03125}


def test_weighted_b_ab_draws_are_paired_by_mother():
    h = data()
    b = h.weight_training_pool(pool_rows())
    ab = pool_rows().sample(frac=1,random_state=33)
    ab['image_path'] = 'AB/' + ab.image_path
    paired_b,paired_ab = h.pair_training_pools(b,ab)
    assert paired_b.mother_id.tolist() == paired_ab.mother_id.tolist()
    assert paired_b.sampling_mass.tolist() == paired_ab.sampling_mass.tolist()
    from journal_detector.trainer import epoch_draws
    draws_b,methods_b = epoch_draws(paired_b,1007,0)
    draws_ab,methods_ab = epoch_draws(paired_ab,1007,0)
    assert paired_b.iloc[draws_b].mother_id.tolist() == paired_ab.iloc[draws_ab].mother_id.tolist()
    assert methods_b.tolist() == methods_ab.tolist()
    with pytest.raises(ValueError,match='mother'):
        h.pair_training_pools(b,ab.iloc[:-1])


def test_official_pairs_keep_train_dev_cal_test_source_ids_disjoint():
    h = data()
    partition = h.source_partition()
    assert len(partition)==1000
    counts = pd.Series({sid:info['role'] for sid,info in partition.items()}).value_counts().to_dict()
    assert counts == {'train':612,'dev':108,'source_cal':140,'source_test':140}
    groups={}
    for sid,info in partition.items(): groups.setdefault(info['group_id'],set()).add(info['role'])
    assert all(len(roles)==1 for roles in groups.values())


def test_build_tables_uses_common_train_dev_and_all_b_test():
    h = data()
    records=[]; manifests={}
    for role in ['train','dev','source_test']:
        for i,method in enumerate(['real','Deepfakes','Face2Face','FaceSwap','NeuralTextures']):
            uid=f'FFPP/source/{method}/{role}'
            row={'video_uid':uid,'video_id':role,'method_id':method,'label':int(method!='real'),
                 'dataset':'FFPP','domain':'source','role':role,'group_id':'pair/'+role,'source_ids':[]}
            records.append(row)
            frames=[]
            for slot in range(8):
                frame={'mother_frame_index':slot,'mother_timestamp_seconds':float(slot),
                       'mother_rgb_sha256':f'{uid}-{slot}', 'mother_id':f'{uid}/mother/{slot}'}
                for version in ['B','AB']:
                    frame[version]={'ok':True,'path':f'{uid}/{slot}/{version}.png',
                        'rgb_sha256':f'{uid}-{slot}-{version}','byte_sha256':'filehash'}
                frames.append(frame)
            common=not(role=='source_test' and method=='FaceSwap')
            manifests[uid]={'frames':frames,'eligible':{'B':True,'AB':common},'common_B_AB':common}
    tables=h.build_tables(records,manifests)
    assert len(tables['B/train'])==len(tables['AB/train'])==40
    assert tables['B/dev'].mother_id.tolist()==tables['AB/dev'].mother_id.tolist()
    assert len(tables['B/source_test'])==40
    assert len(tables['common/B/source_test'])==32
    assert len(tables['AB/source_test'])==32
    assert tables['common/B/source_test'].mother_id.tolist()==tables['common/AB/source_test'].mother_id.tolist()


def test_source_audit_includes_unused_video_rgb_bridges():
    h=data()
    records=[{'video_uid':'train','role':'train','group_id':'pair/A'},
             {'video_uid':'unused','role':'source_unused','group_id':'pair/C'},
             {'video_uid':'cal','role':'source_cal','group_id':'pair/C'},
             {'video_uid':'dev','role':'dev','group_id':'pair/B'}]
    audits={'train':{'selected_frames':[{'rgb_sha256':'bridge'}]},
            'unused':{'selected_frames':[{'rgb_sha256':'bridge'}]},
            'cal':{'selected_frames':[{'rgb_sha256':'calonly'}]},
            'dev':{'selected_frames':[{'rgb_sha256':'unrelated'}]}}
    result=h.source_graph_audit(records,audits)
    assert result['quarantined_video_uids']==['cal','train','unused']
    assert result['candidate_videos']==4
    assert result['components']['train']==result['components']['cal']
    assert 'dev' not in result['quarantined_video_uids']


def test_source_freeze_binds_ordered_tables_and_rejects_table_mutation(tmp_path):
    h=data()
    rows=pool_rows(); rows['rgb_sha256']=rows.mother_id
    b,ab=h.pair_training_pools(rows,rows.copy())
    tables={'B/train':b,'AB/train':ab,'B/dev':b.copy(),'AB/dev':ab.copy(),
            'B/source_cal':b[b.label.eq(0)].copy(),'AB/source_cal':ab[ab.label.eq(0)].copy()}
    receipt=h.write_table_freeze(tmp_path,tables,scope='SOURCE_TRAIN_DEV_CAL',metadata={'test':True})
    assert receipt['draw_count']==128
    assert h.validate_h_freeze(tmp_path,source_only=True)['status']=='FROZEN'
    target=tmp_path/'data/H/B/train.parquet'
    target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='frozen H table changed'):
        h.validate_h_freeze(tmp_path,source_only=True)


def test_coverage_keeps_pending_and_rejected_candidates_in_denominator():
    h=data()
    records=[{'video_uid':uid,'role':'target_test','dataset':'DFD','method_id':'real','label':0}
             for uid in ['ok','failed','pending']]
    manifests={'ok':{'status':'COMPLETE','frames':[{}],'eligible':{'A':True,'B':True,'AB':True},'common_B_AB':True},
               'failed':{'status':'COMPLETE','frames':[{}],'eligible':{'A':True,'B':True,'AB':False},
                         'common_B_AB':False,'version_failures':{'AB':['0:B_redetection:no_qualified_face']}}}
    rows,counts=h.coverage(records,manifests)
    assert len(rows)==3 and counts.candidate_videos.sum()==3
    assert rows.loc[rows.video_uid.eq('pending'),'failure'].iloc[0]=='PENDING_PREPROCESSING'
    assert 'B_redetection' in rows.loc[rows.video_uid.eq('failed'),'failure'].iloc[0]


def test_full_freeze_preserves_source_tables_and_evidence(tmp_path):
    h=data()
    from journal_detector.geometry_h import sha256_file
    rows=pool_rows();rows['rgb_sha256']=rows.mother_id
    b,ab=h.pair_training_pools(rows,rows.copy())
    tables={'B/train':b,'AB/train':ab,'B/dev':b.copy(),'AB/dev':ab.copy(),
            'B/source_cal':b[b.label.eq(0)].copy(),'AB/source_cal':ab[ab.label.eq(0)].copy()}
    evidence=tmp_path/'source_coverage.csv';evidence.write_text('video_uid,processed\nv0,true\n')
    h.write_table_freeze(tmp_path,tables,scope='SOURCE_TRAIN_DEV_CAL',
        metadata={'evidence_files':{str(evidence):sha256_file(evidence)}})
    source=tmp_path/'data/H/train_dev_freeze.json';before=source.read_bytes()
    table_hashes={key:sha256_file(tmp_path/'data/H'/(key+'.parquet')) for key in tables}
    full=dict(tables);full['B/target_test']=b.copy();full['AB/target_test']=ab.copy()
    h.write_table_freeze(tmp_path,full,scope='ALL_H_ROLES')
    assert source.read_bytes()==before
    assert {key:sha256_file(tmp_path/'data/H'/(key+'.parquet')) for key in tables}==table_hashes
    assert h.validate_h_freeze(tmp_path,source_only=True)['status']=='FROZEN'
    assert h.validate_h_freeze(tmp_path)['train_dev_freeze_sha256']==sha256_file(source)
