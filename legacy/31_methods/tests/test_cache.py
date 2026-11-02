import json
import numpy as np
import pandas as pd
import pytest
from deepfake_detector import cache as c
from deepfake_detector.manifest import digest


def make_cache(root):
    root.mkdir(exist_ok=True)
    pd.DataFrame({'sample_id':['a','b'],'label':[0,1]}).to_parquet(root/'manifest.parquet',index=False)
    np.save(root/'000000000.npy',np.array([[1.,2.],[-1.,3.]],dtype=np.float32))
    metadata={'identity':{'checkpoint':'fixed','manifest':'input'},'rows':2,
      'cached_manifest_sha256':digest(root/'manifest.parquet'),
      'chunks':[{'start':0,'rows':2,'file':'000000000.npy','sha256':digest(root/'000000000.npy')}],
      'head':{'weight':[[0.,0.],[2.,-1.]],'bias':[1.,3.]}}
    (root/'complete.json').write_text(json.dumps(metadata))
    return metadata


def test_cache_checks_hashes_alignment_and_signed_two_class_head(tmp_path):
    assert hasattr(c,'load_features'), 'verified cache loader missing'
    meta=make_cache(tmp_path)
    df,h,w,b=c.load_features(tmp_path,meta['identity'])
    assert list(df.sample_id)==['a','b'] and h.dtype==np.float32
    assert np.array_equal(w,[2.,-1.]) and b==2.
    assert np.array_equal(h@w+b,[2.,-3.])
    with (tmp_path/'000000000.npy').open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='hash'):c.load_features(tmp_path,meta['identity'])


def test_cache_rejects_gapped_chunks_and_changed_manifest(tmp_path):
    assert hasattr(c,'load_features'), 'verified cache loader missing'
    meta=make_cache(tmp_path)
    meta['chunks'][0]['start']=1
    (tmp_path/'complete.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='contiguous'):c.load_features(tmp_path,meta['identity'])
    meta['chunks'][0]['start']=0
    (tmp_path/'complete.json').write_text(json.dumps(meta))
    pd.DataFrame({'sample_id':['b','a'],'label':[1,0]}).to_parquet(tmp_path/'manifest.parquet',index=False)
    with pytest.raises(ValueError,match='manifest'):c.load_features(tmp_path,meta['identity'])
