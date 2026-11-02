"""Unified CLI: prepare, train, extract, analyze, summarize, status."""
import argparse
import json
from . import trainer,cache_d,analysis_d,summary_d
from .prepare import prepare


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','train','extract','analyze','summarize','status'])
    p.add_argument('--config',default='/home/z/workspace/deepfake_conference21_20261102/configs/execution.json')
    p.add_argument('--run-id');p.add_argument('--backbone')
    p.add_argument('--source-root',default='/home/z/workspace/deepfake_pipeline_reliability_v2_20261023')
    p.add_argument('--output-root',default='/home/z/workspace/deepfake_conference21_20261102')
    a=p.parse_args()
    if a.action in ['train','extract','analyze'] and not a.run_id:p.error('--run-id required')
    if a.action=='summarize' and not a.backbone:p.error('--backbone required')
    if a.action=='prepare':result=prepare(a.source_root,a.output_root)
    elif a.action=='train':result=trainer.train(a.config,a.run_id)
    elif a.action=='extract':result=cache_d.extract(a.config,a.run_id)
    elif a.action=='analyze':result=analysis_d.analyze_run(a.config,a.run_id)
    elif a.action=='summarize':result=summary_d.summarize_backbone(a.config,a.backbone)
    else:result=analysis_d.formal_cache_status(a.config)
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))


if __name__=='__main__':main()
