"""Prepare a separate experiment registry; never launches a model or training."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pandas as pd
from .trainer import digest,save_json,exclusive_file
from .protocol import PROTOCOL,LRS,SEEDS,validate_config,validate_manifests


def prepare(source_root,output_root):
    source=Path(source_root).resolve();root=Path(output_root).resolve()
    if root==source or source in root.parents:raise ValueError('new artifact root must be separate')
    with exclusive_file(root/'prepare.lock'):
        data=source/'data/D';freeze=json.loads((data/'freeze.json').read_text())
        if freeze.get('status')!='FROZEN' or freeze.get('method_count')!=21:raise ValueError('invalid source data freeze')
        for name,expected in freeze['manifest_sha256'].items():
            if Path(name).name!=name:raise ValueError('unsafe frozen filename')
            if digest(data/name)!=expected:raise ValueError('source frozen file changed: '+name)
        out=root/'data/D';out.mkdir(parents=True,exist_ok=True)
        for name in [*freeze['manifest_sha256'],'freeze.json']:
            destination=out/name
            if destination.exists():
                if digest(destination)!=digest(data/name):raise ValueError('existing data differs: '+name)
            else:shutil.copy2(data/name,destination)
        validate_manifests(root,pd.read_parquet(out/'train.parquet'),pd.read_parquet(out/'dev.parquet'))
        baseline=json.loads((source/'configs/base.json').read_text())
        config={k:baseline[k] for k in ['backbones','allowed_gpus','allowed_gpu_uuids']}
        config.update(protocol=PROTOCOL,artifact_root=str(root),effective_batch=128,workers=4,formal_seeds=SEEDS,
          checkpoint_selection='source_dev_pooled_mean_probability_auc_ema',runs={})
        for backbone,lr in LRS.items():
            config['backbones'][backbone]['learning_rate']=lr
            if digest(config['backbones'][backbone]['weight_path'])!=config['backbones'][backbone]['weight_sha256']:
                raise ValueError('initialization changed')
            for seed in SEEDS:
                config['runs'][f'D_formal_{backbone}_seed{seed}']={
                  'route':'D','phase':'formal','recipe_frozen':True,'backbone':backbone,'seed':seed,
                  'epochs':15,'microbatch':128,'precision':'AMP','lr':lr,
                  'train_manifest':'data/D/train.parquet','dev_manifest':'data/D/dev.parquet'}
        validate_config(config)
        configs=root/'configs';configs.mkdir(exist_ok=True)
        path=configs/'execution.json'
        if path.exists() and json.loads(path.read_text())!=config:raise ValueError('existing registry differs')
        save_json(path,config)
        lock=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
        lockpath=configs/'requirements.execution.lock.txt'
        if lockpath.exists() and lockpath.read_text()!=lock:raise ValueError('execution environment changed')
        lockpath.write_text(lock)
        receipt={'status':'PREPARED_NOT_TRAINED','config':str(path),'runs':len(config['runs']),
          'source_root':str(source),'source_freeze_sha256':digest(data/'freeze.json'),
          'copied_freeze_sha256':digest(out/'freeze.json'),'protocol':PROTOCOL}
        save_json(root/'prepared.json',receipt)
        return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',default='/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
    p.add_argument('--output-root',default='/home/z/workspace/deepfake_conference21_20261102')
    a=p.parse_args();print(json.dumps(prepare(a.source_root,a.output_root),indent=2))
