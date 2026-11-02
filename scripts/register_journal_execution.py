"""Register immutable development trajectories; no models are launched here."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPOSITORY/'src'))
from journal_detector import protocol
from journal_detector.trainer import save_json


def development_registry(root):
    root=Path(root).resolve()
    config=json.loads((REPOSITORY/'configs/journal.json').read_text())
    config.update(protocol=protocol.PROTOCOL,artifact_root=str(root),workers=6,
                  development_seeds=protocol.DEVELOPMENT_SEEDS,h_formal_seeds=protocol.H_SEEDS,
                  epoch_development_caps={'clip_vit_b16':10,'efficientnet_b0':15,'dinov2_vit_b14':30},
                  target_development_used=True,runs={})
    for backbone,spec in config['backbones'].items():
        spec['weight_path']=str(root/'weights'/Path(spec['weight_path']).name)
        epochs=config['epoch_development_caps'][backbone]
        for seed in protocol.DEVELOPMENT_SEEDS:
            run=f'D_development_{backbone}_seed{seed}_cap{epochs}'
            config['runs'][run]=dict(route='D',phase='development',recipe_frozen=False,
                backbone=backbone,seed=seed,epochs=epochs,microbatch=128,precision='AMP',lr=protocol.LRS[backbone],
                train_manifest='data/D/train.parquet',dev_manifest='data/D/dev.parquet',
                cdf_development_manifest='data/D/cdf_dev.parquet')
    protocol.validate_config(config)
    path=root/'configs/development.json'
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('existing development registry differs; do not overwrite trajectories')
    save_json(path,config)
    environment=root/'configs/requirements.execution.lock.txt'
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    if environment.exists() and environment.read_text()!=lock:
        raise ValueError('execution dependencies changed after environment freeze')
    environment.write_text(lock)
    return path


def h_development_registry(root):
    root=Path(root).resolve()
    config=json.loads((root/'configs/development.json').read_text())
    config['target_development_used']=False;config['runs']={}
    for seed in protocol.DEVELOPMENT_SEEDS:
        for history in ['B','AB']:
            run=f'H_development_efficientnet_b0_{history}_seed{seed}_cap15'
            config['runs'][run]=dict(route='H',phase='development',history=history,
                recipe_frozen=False,backbone='efficientnet_b0',seed=seed,epochs=15,
                microbatch=128,precision='AMP',lr=protocol.LRS['efficientnet_b0'],
                train_manifest=f'data/H/{history}/train.parquet',dev_manifest='data/H/B/dev.parquet')
    protocol.validate_config(config)
    path=root/'configs/h_development.json'
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('existing H development registry differs')
    save_json(path,config)
    return path


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    print(development_registry(args.root))
    print(h_development_registry(args.root))
