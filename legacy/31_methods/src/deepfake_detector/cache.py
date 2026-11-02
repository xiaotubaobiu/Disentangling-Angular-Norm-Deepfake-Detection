"""Hash-verified aligned raw-feature caches."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .manifest import digest


def verify_cache(root,identity=None):
    root=Path(root);meta=json.loads((root/'complete.json').read_text())
    if identity is not None and meta['identity']!=identity:raise ValueError('Feature cache identity mismatch')
    if digest(root/'manifest.parquet')!=meta.get('cached_manifest_sha256'):raise ValueError('Feature cache manifest hash mismatch')
    position=0;dimension=None
    for item in meta['chunks']:
        if item['start']!=position:raise ValueError('Feature cache chunks are not contiguous')
        name=Path(item['file'])
        if name.name!=str(name) or name.is_absolute():raise ValueError('Unsafe cache chunk path')
        path=root/name
        if digest(path)!=item['sha256']:raise ValueError('Feature chunk hash mismatch')
        x=np.load(path,mmap_mode='r',allow_pickle=False)
        if x.ndim!=2 or x.shape[0]!=item['rows'] or x.dtype!=np.float32 or not np.isfinite(x).all():raise ValueError('Feature chunk shape/dtype/finiteness mismatch')
        if dimension is None:dimension=x.shape[1]
        elif x.shape[1]!=dimension:raise ValueError('Feature dimension mismatch')
        position+=len(x)
    if position!=meta['rows']:raise ValueError('Feature cache row count mismatch')
    df=pd.read_parquet(root/'manifest.parquet')
    if len(df)!=position or ('sample_id' in df and df.sample_id.duplicated().any()):raise ValueError('Feature manifest row alignment mismatch')
    return meta,df


def load_features(root,identity=None):
    root=Path(root);meta,df=verify_cache(root,identity)
    weight=np.asarray(meta['head']['weight'],dtype=np.float64);bias=np.asarray(meta['head']['bias'],dtype=np.float64)
    if weight.ndim!=2 or weight.shape[0]!=2 or bias.shape!=(2,) or not np.isfinite(weight).all() or not np.isfinite(bias).all():raise ValueError('Invalid two-class raw head')
    x=np.concatenate([np.load(root/q['file'],allow_pickle=False) for q in meta['chunks']]) if meta['chunks'] else np.empty((0,weight.shape[1]),np.float32)
    if x.shape[1]!=weight.shape[1]:raise ValueError('Head/feature dimension mismatch')
    return df,x,weight[1]-weight[0],float(bias[1]-bias[0])
