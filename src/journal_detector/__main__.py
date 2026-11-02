"""Unified CLI: prepare, train, extract, analyze, summarize, status."""
import argparse
import json
from pathlib import Path
from . import data_d
from .prepare import DEFAULT_CONFIG, execution_config, prepare, status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','train','extract','analyze','summarize','status'])
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--run-id');p.add_argument('--backbone')
    p.add_argument('--data-root',type=Path,default=data_d.DATA_ROOT)
    p.add_argument('--definition-root',type=Path,default=data_d.DEFINITION_ROOT)
    p.add_argument('--output-root',type=Path)
    p.add_argument('--workers',type=int,default=4,help='fresh byte/RGB audit workers for prepare')
    a=p.parse_args()
    if a.action in ['train','extract','analyze'] and not a.run_id:p.error('--run-id required')
    if a.action=='summarize' and not a.backbone:p.error('--backbone required')
    if a.action=='prepare':result=prepare(a.config,a.output_root,a.data_root,a.definition_root,a.workers)
    elif a.action=='status':result=status(a.config,a.output_root)
    else:
        config=execution_config(a.config,a.output_root)
        path=Path(config['artifact_root'])/'configs/execution.json'
        if not path.is_file():p.error('experiment is not prepared; run prepare first')
        if json.loads(path.read_text()) != config:p.error('prepared registry differs from requested configuration')
        if a.action=='train':
            from . import trainer
            result=trainer.train(path,a.run_id)
        elif a.action=='extract':
            from . import cache_d
            result=cache_d.extract(path,a.run_id)
        elif a.action=='analyze':
            from . import analysis_d
            result=analysis_d.analyze_run(path,a.run_id)
        else:
            from . import summary_d
            result=summary_d.summarize_backbone(path,a.backbone)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=='__main__':main()
