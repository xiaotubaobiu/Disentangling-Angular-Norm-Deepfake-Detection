"""Compare only registered development runs and freeze a verified epoch cap."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

REPOSITORY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPOSITORY/'src'))
from journal_detector import protocol, trainer
from journal_detector.development import select_epoch_budget


def cap_report(root,runs,route,backbone,epochs):
    root=Path(root);records=[];seen=set();curves={}
    for run in runs:
        folder=root/'runs'/run;task=json.loads((folder/'resolved_config.json').read_text())
        complete=json.loads((folder/'complete.json').read_text());history=json.loads((folder/'history.json').read_text())
        if (task.get('route')!=route or task.get('backbone')!=backbone or task.get('epochs')!=epochs
            or task.get('phase')!='development' or task.get('seed') not in protocol.DEVELOPMENT_SEEDS
            or complete.get('status')!='TRAINING_COMPLETE' or complete.get('phase')!='development'):
            raise ValueError('epoch cap comparison requires matching registered development route/backbone/seed')
        key=(task['seed'],task.get('history','B'))
        if key in seen:raise ValueError('duplicate development trajectory')
        seen.add(key)
        selected=[row for row in history if row['epoch']==complete['best_epoch']]
        if len(selected)!=1:raise ValueError('source-selected epoch absent from development history')
        row=selected[0]
        if route=='D':
            if complete.get('target_used') is not True:raise ValueError('CDF development was not evaluated')
            score=float(row['cdf_dev']['pooled_auc']);curves[task['seed']]=history
        else:
            if complete.get('target_used') is not False:raise ValueError('H used target information')
            score=float(row['dev']['pooled_auc'])
        records.append({'run_id':run,'seed':task['seed'],'history':task.get('history','B'),
            'source_selected_epoch':complete['best_epoch'],'actual_stop_epoch':complete['epochs'],
            'early_stopped':complete['early_stopped'],'score':score,
            'complete_sha256':trainer.digest(folder/'complete.json'),
            'history_sha256':trainer.digest(folder/'history.json'),
            'task_sha256':trainer.digest(folder/'resolved_config.json')})
    expected={(seed,h) for seed in protocol.DEVELOPMENT_SEEDS for h in (['B'] if route=='D' else ['B','AB'])}
    if seen!=expected:raise ValueError('requires both registered development seeds and all paired histories')
    if not np.isfinite([x['score'] for x in records]).all():raise ValueError('nonfinite development score')
    if route=='H':
        for seed in protocol.DEVELOPMENT_SEEDS:
            if len({r['source_selected_epoch'] for r in records if r['seed']==seed})!=1:
                raise ValueError('H models did not select a common epoch')
    return {'route':route,'backbone':backbone,'epochs':epochs,'development_runs':records,
        'mean_score':float(np.mean([r['score'] for r in records])),
        'score_kind':'cdf_dev_at_source_selected_ema' if route=='D' else 'paired_source_dev_B_auc',
        'epoch_candidate':select_epoch_budget(curves) if route=='D' else None,
        'test_used':False}


def freeze(root,runs,route,backbone,epochs,enqueue=False):
    root=Path(root).resolve();report=cap_report(root,runs,route,backbone,epochs)
    data_freeze=root/('data/D/freeze.json' if route=='D' else 'data/H/train_dev_freeze.json')
    recipe={'schema':'EPOCH_RECIPE_V2','status':'FROZEN','route':route,'backbone':backbone,
        'chosen_epochs':epochs,'target_development_used':route=='D','test_used':False,
        'development_seeds':protocol.DEVELOPMENT_SEEDS,'paired_histories':['B','AB'] if route=='H' else None,
        'data_freeze_sha256':trainer.digest(data_freeze),'development_report':report,
        'fixed_learning_rate':protocol.LRS[backbone],'actual_batch':128,
        'criterion':report['score_kind'],'decision_time':time.time()}
    recipe_path=root/f'configs/epoch_recipes/{route}_{backbone}.json'
    if recipe_path.exists():
        existing=json.loads(recipe_path.read_text())
        expected={k:v for k,v in recipe.items() if k!='decision_time'}
        if {k:v for k,v in existing.items() if k!='decision_time'}!=expected:
            raise ValueError('existing epoch decision differs; cannot silently refreeze')
        recipe=existing
    else:trainer.save_json(recipe_path,recipe)
    config=json.loads((root/'configs/development.json').read_text());config['runs']={}
    config['target_development_used']=route=='D'
    for seed in protocol.SEEDS if route=='D' else protocol.H_SEEDS:
        for history in ['B'] if route=='D' else ['B','AB']:
            run=f'D_formal_{backbone}_seed{seed}' if route=='D' else f'H_formal_{backbone}_{history}_seed{seed}'
            task=dict(route=route,phase='formal',recipe_frozen=True,backbone=backbone,seed=seed,
                epochs=epochs,microbatch=128,precision='AMP',lr=protocol.LRS[backbone],
                train_manifest='data/D/train.parquet' if route=='D' else f'data/H/{history}/train.parquet',
                dev_manifest='data/D/dev.parquet' if route=='D' else 'data/H/B/dev.parquet',
                epoch_recipe_path=str(recipe_path.relative_to(root)),epoch_recipe_sha256=trainer.digest(recipe_path))
            if route=='H':task['history']=history
            config['runs'][run]=task
    protocol.validate_config(config)
    name=f'formal_training_{backbone}.json' if route=='D' else ('h_execution.json' if backbone=='efficientnet_b0' else f'h_execution_{backbone}.json')
    path=root/'configs'/name
    if path.exists() and json.loads(path.read_text())!=config:raise ValueError('frozen formal registry differs')
    trainer.save_json(path,config)
    if route=='D':
        files=[root/f'configs/formal_training_{b}.json' for b in protocol.LRS]
        if all(p.is_file() for p in files):
            combined={**config,'runs':{}}
            for p in files:combined['runs'].update(json.loads(p.read_text())['runs'])
            protocol.validate_config(combined)
            if len(combined['runs'])!=15:raise ValueError('complete D canonical registry requires15models')
            canonical=root/'configs/execution.json'
            if canonical.exists() and json.loads(canonical.read_text())!=combined:raise ValueError('canonical D config changed')
            trainer.save_json(canonical,combined)
    if enqueue:
        queue=root/'operations/jobs.json'
        with trainer.exclusive_file(root/'operations/jobs_update.lock'):
            registry=json.loads(queue.read_text());known={j['id'] for j in registry['jobs']}
            for seed in protocol.SEEDS if route=='D' else protocol.H_SEEDS:
                if route=='D':
                    run=f'D_formal_{backbone}_seed{seed}';job_id=run
                    argv=[sys.executable,'-m','journal_detector.trainer','--config',str(path),'--run-id',run]
                else:
                    b=f'H_formal_{backbone}_B_seed{seed}';ab=f'H_formal_{backbone}_AB_seed{seed}';job_id=f'H_formal_{backbone}_pair_seed{seed}'
                    argv=[sys.executable,'-m','journal_detector.trainer_h','--config',str(path),'--b-run',b,'--ab-run',ab]
                if job_id in known:continue
                registry['jobs'].append({'id':job_id,'route':route,'priority':4,'argv':argv,
                    'depends_on_files':[str(data_freeze),str(root/f'operations/code_ready_{route}.json')],
                    'code_gate':str(root/f'operations/code_ready_{route}.json')})
            trainer.save_json(queue,registry)
    return {'recipe':str(recipe_path),'formal_config':str(path),'report':report}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--route',choices=['D','H'],required=True);p.add_argument('--backbone',choices=list(protocol.LRS),required=True)
    p.add_argument('--epochs',type=int,required=True);p.add_argument('--runs',nargs='+',required=True)
    p.add_argument('--freeze',action='store_true');p.add_argument('--enqueue',action='store_true');a=p.parse_args()
    value=freeze(a.root,a.runs,a.route,a.backbone,a.epochs,a.enqueue) if a.freeze else cap_report(a.root,a.runs,a.route,a.backbone,a.epochs)
    print(json.dumps(value,ensure_ascii=False,indent=2))
