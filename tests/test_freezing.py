import json
import numpy as np
import pandas as pd
from deepfake_detector import freezing as f,training as t


def test_shared_fake_content_is_one_input_with_two_method_contributions():
    assert hasattr(f,'canonical_training_pool'), 'content-canonical pool missing'
    df=pd.DataFrame([
      dict(domain='ff',experiment_role='train',label=0,method_id='real',method_type='real',video_id='r',evaluation_unit='video_frame',rgb_sha256='real',image_path='/real'),
      dict(domain='ff',experiment_role='train',label=1,method_id='SG2',method_type='EFS',video_id='',evaluation_unit='image',rgb_sha256='fake',image_path='/SG2'),
      dict(domain='ff',experiment_role='train',label=1,method_id='SG3',method_type='EFS',video_id='',evaluation_unit='image',rgb_sha256='fake',image_path='/SG3')])
    pool,associations=f.canonical_training_pool(df)
    assert len(pool)==2 and len(associations)==3
    shared=pool[pool.label.eq(1)].iloc[0]
    assert json.loads(shared.method_memberships)==['SG2','SG3']
    assert np.isclose(t.sampling_weights(pool)[pool.label.eq(1)].sum(),.5)
    assert np.isclose(associations.groupby('method_id').sampling_contribution.sum()['SG2'],.25)
    assert np.isclose(associations.groupby('method_id').sampling_contribution.sum()['SG3'],.25)


def test_source_roles_filter_known_video_ids_but_never_efs_bucket_ids():
    assert hasattr(f,'source_role_filter'), 'known source filter missing'
    df=pd.DataFrame([
      dict(domain='ff',released_role='train',label=1,method_type='FS',source_ids='["001","002"]'),
      dict(domain='ff',released_role='train',label=1,method_type='FS',source_ids='["003","004"]'),
      dict(domain='ff',released_role='test',label=1,method_type='FS',source_ids='["001","002"]'),
      dict(domain='ff',released_role='train',label=1,method_type='EFS',source_ids='[]'),
      dict(domain='cdf',released_role='test',label=1,method_type='FS',source_ids='[]')])
    result=f.source_role_filter(df,{'001':'train','002':'train','003':'test','004':'test'})
    assert list(result.experiment_role)==['train','EXCLUDED','EXCLUDED','train','cdf_test']
    assert result.loc[3,'upstream_source_role']=='unknown'
