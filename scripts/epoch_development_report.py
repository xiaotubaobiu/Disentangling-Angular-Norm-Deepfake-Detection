"""Compare only registered development runs and freeze a verified epoch cap."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

REPOSITORY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPOSITORY/'src'))
from journal_detector import protocol, trainer
from journal_detector.development import select_epoch_budget


def _registered_config(root,route,path=None):
    path=(root/('configs/development.json' if route=='D' else 'configs/h_development.json')
          if path is None else Path(path)).resolve()
    if not path.is_relative_to(root/'configs'):
        raise ValueError('registered development config must belong to artifact configs')
    config=json.loads(path.read_text());protocol.validate_config(config)
    if Path(config['artifact_root']).resolve()!=root or config.get('protocol')!=protocol.PROTOCOL:
        raise ValueError('registered development root/protocol differs')
    return path,config


def _verified_run(root,run,config,state_cache):
    folder=root/'runs'/run
    task=json.loads((folder/'resolved_config.json').read_text())
    if config['runs'].get(run)!=task or task.get('phase')!='development':
        raise ValueError('run differs from registered development task')
    complete=json.loads((folder/'complete.json').read_text())
    history=json.loads((folder/'history.json').read_text())
    identity=trainer.training_identity(config,task)
    if complete.get('identity')!=identity:
        raise ValueError('development training identity differs from live frozen evidence')
    route=task['route'];kind=task.get('history','B')
    selection_name=('source_dev_pooled_mean_probability_auc_ema' if route=='D'
                    else 'paired_source_dev_B_mean_probability_auc_ema')
    if (complete.get('status')!='TRAINING_COMPLETE' or complete.get('phase')!='development'
        or complete.get('selection')!=selection_name or complete.get('target_used') is not (route=='D')
        or complete.get('max_epochs')!=task['epochs'] or complete.get('epochs')!=len(history)
        or not history or [r['epoch'] for r in history]!=list(range(1,len(history)+1))
        or len(history)>task['epochs']):
        raise ValueError('incoherent development completion/history/selection')
    checkpoint=folder/'best_model.pth'
    if trainer.digest(checkpoint)!=complete.get('best_sha256'):
        raise ValueError('development best checkpoint hash mismatch')
    selected=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if (selected.get('identity')!=identity or selected.get('epoch')!=complete.get('best_epoch')
        or selected.get('weights')!='ema' or route=='H' and selected.get('selection')!=selection_name):
        raise ValueError('development best checkpoint identity/epoch/EMA mismatch')
    state_path=folder/'state.pt' if route=='D' else Path(complete.get('pair_state_path','')).resolve()
    if route=='H' and not state_path.is_relative_to(root/'pairs'):
        raise ValueError('H paired state path is outside registered pair artifacts')
    if state_path not in state_cache:
        state_cache[state_path]=(torch.load(state_path,map_location='cpu',weights_only=False),trainer.digest(state_path))
    state,state_sha=state_cache[state_path]
    if route=='D':
        if state.get('identity')!=identity or state.get('history')!=history:
            raise ValueError('development atomic state identity/history mismatch')
    else:
        partners={x.get('history','B'):trainer.training_identity(config,x) for x in config['runs'].values()
                  if all(x.get(k)==task.get(k) for k in ['route','phase','backbone','seed','epochs'])}
        if (set(partners)!={'B','AB'} or state.get('identity')!=partners
            or state.get('histories',{}).get(kind)!=history or complete.get('pair_state_sha256')!=state_sha):
            raise ValueError('H paired atomic state identity/history mismatch')
    if state.get('epoch')!=len(history):raise ValueError('development state epoch differs from history')
    selection=trainer.Selection();stop=__import__('journal_detector.development',fromlist=['DevelopmentStop']).DevelopmentStop()
    criterion=[]
    for row in history:
        source=float(row['dev']['pooled_auc'])
        if route=='D':
            value=float(row['cdf_dev']['pooled_auc']);selection.observe(source,row['epoch']);stop.observe(source,value,row['epoch'])
        else:
            if row['dev']!=row['paired_dev'][kind]:raise ValueError('H paired history own-source metrics mismatch')
            value=sum(float(row['paired_dev'][h]['pooled_auc']) for h in ['B','AB'])/2
            selection.observe(value,row['epoch'])
        if not np.isfinite([source,value]).all() or row.get('selection')!=selection.state_dict():
            raise ValueError('development history selection/finite score mismatch')
        criterion.append(value)
    stopped=stop.stopped if route=='D' else selection.stopped
    if (complete.get('best_epoch')!=selection.best_epoch or complete.get('early_stopped') is not stopped
        or selected.get('auc')!=selection.best_score or state.get('selection')!=selection.state_dict()
        or not stopped and len(history)!=task['epochs']):
        raise ValueError('development stop/best checkpoint history is incoherent')
    if route=='D':
        cdf_path=folder/'cdf_best_model.pth'
        if (complete.get('cdf_best_epoch')!=stop.cdf.best_epoch or state.get('development_stop')!=stop.state_dict()
            or trainer.digest(cdf_path)!=complete.get('cdf_best_sha256')):
            raise ValueError('isolated CDF development checkpoint/history mismatch')
        cdf=torch.load(cdf_path,map_location='cpu',weights_only=False)
        if (cdf.get('identity')!=identity or cdf.get('epoch')!=stop.cdf.best_epoch
            or cdf.get('weights')!='ema' or cdf.get('selection')!='isolated_cdf_dev_pooled_mean_probability_auc_ema'
            or cdf.get('auc')!=stop.cdf.best_score):
            raise ValueError('isolated CDF development checkpoint identity/epoch mismatch')
    return task,complete,history,dict(state_path=str(state_path),state_sha256=state_sha,
        checkpoint_sha256=complete['best_sha256'],training_identity=identity),criterion


def cap_report(root,runs,route,backbone,epochs,development_config=None):
    root=Path(root).resolve();records=[];seen=set();curves={};state_cache={};boundary=[]
    config_path,config=_registered_config(root,route,development_config)
    for run in runs:
        task,complete,history,evidence,criterion=_verified_run(root,run,config,state_cache)
        if (task.get('route')!=route or task.get('backbone')!=backbone or task.get('epochs')!=epochs
            or task.get('seed') not in protocol.DEVELOPMENT_SEEDS):
            raise ValueError('epoch cap comparison requires matching registered development route/backbone/seed')
        key=(task['seed'],task.get('history','B'))
        if key in seen:raise ValueError('duplicate development trajectory')
        seen.add(key)
        row=history[complete['best_epoch']-1]
        if route=='D':score=float(row['cdf_dev']['pooled_auc']);curves[task['seed']]=history
        else:score=float(row['dev']['pooled_auc'])
        records.append({'run_id':run,'seed':task['seed'],'history':task.get('history','B'),
            'source_selected_epoch':complete['best_epoch'],'actual_stop_epoch':complete['epochs'],
            'early_stopped':complete['early_stopped'],'score':score,**evidence,
            'complete_sha256':trainer.digest(root/'runs'/run/'complete.json'),
            'history_sha256':trainer.digest(root/'runs'/run/'history.json'),
            'task_sha256':trainer.digest(root/'runs'/run/'resolved_config.json')})
        if route=='D' or task.get('history')=='B':
            if len(criterion)==epochs and (len(criterion)==1 or criterion[-1]>max(criterion[:-1])):
                recent=criterion[-3:]
                boundary.append(dict(seed=task['seed'],observations=len(recent),
                    recent_gain=max(recent)-recent[0],criterion='CDF_DEV' if route=='D' else 'paired_source_DEV_B'))
    expected={(seed,h) for seed in protocol.DEVELOPMENT_SEEDS for h in (['B'] if route=='D' else ['B','AB'])}
    if seen!=expected:raise ValueError('requires both registered development seeds and all paired histories')
    if not np.isfinite([x['score'] for x in records]).all():raise ValueError('nonfinite development score')
    if route=='H':
        for seed in protocol.DEVELOPMENT_SEEDS:
            paired=[r for r in records if r['seed']==seed]
            if (len({r['source_selected_epoch'] for r in paired})!=1
                or len({r['state_sha256'] for r in paired})!=1):raise ValueError('H models did not select a common atomic epoch/state')
    return {'route':route,'backbone':backbone,'epochs':epochs,'development_runs':records,
        'development_config_path':str(config_path),'development_config_sha256':trainer.digest(config_path),
        'mean_score':float(np.mean([r['score'] for r in records])),
        'score_kind':'cdf_dev_at_source_selected_ema' if route=='D' else 'paired_source_dev_B_auc',
        'epoch_candidate':select_epoch_budget(curves) if route=='D' else None,
        'boundary_improving':boundary,'test_used':False}


def stopping_decision(root,report,decision=None):
    # This gate documents an epoch-budget decision, not a significance test and
    # not a change to the conference best-EMA/patience computation.
    limit=.005 if report['route']=='D' else .0025
    if decision is None:
        if report['boundary_improving']:
            raise ValueError('unresolved improving observation boundary: extend or supply a verified stopping decision')
        return dict(basis='observed_nonboundary_or_early_stop',test_used=False)
    if not isinstance(decision.get('rationale'),str) or not decision['rationale'].strip():
        raise ValueError('explicit stopping decision rationale required')
    basis=decision.get('basis')
    if basis=='empirical_plateau':
        if any(x['observations']<3 or x['recent_gain']>limit for x in report['boundary_improving']):
            raise ValueError('development criterion still improving at boundary; extend the observation cap')
        return dict(decision,recent_three_epoch_gain_limit=limit,
                    interpretation='Descriptive epoch-budget plateau gate; not statistical convergence.',test_used=False)
    if basis=='fresh_cosine_comparison':
        ref=decision.get('reference',{})
        reference=cap_report(root,ref['runs'],report['route'],report['backbone'],ref['epochs'],ref.get('development_config'))
        if (reference['epochs']<=report['epochs'] or report['mean_score']<reference['mean_score']
            or any(x['observations']<3 or x['recent_gain']>limit for x in reference['boundary_improving'])):
            raise ValueError('fresh cosine confirmation does not resolve the longer improving cap')
        return dict(decision,verified_reference=reference,test_used=False)
    raise ValueError('unregistered stopping decision basis')


def freeze(root,runs,route,backbone,epochs,enqueue=False,development_config=None,decision=None):
    root=Path(root).resolve();report=cap_report(root,runs,route,backbone,epochs,development_config)
    stop_decision=stopping_decision(root,report,decision)
    data_freeze=root/('data/D/freeze.json' if route=='D' else 'data/H/train_dev_freeze.json')
    recipe={'schema':'EPOCH_RECIPE_V2','status':'FROZEN','route':route,'backbone':backbone,
        'chosen_epochs':epochs,'target_development_used':route=='D','test_used':False,
        'development_seeds':protocol.DEVELOPMENT_SEEDS,'paired_histories':['B','AB'] if route=='H' else None,
        'data_freeze_sha256':trainer.digest(data_freeze),'development_report':report,
        'fixed_learning_rate':protocol.LRS[backbone],'actual_batch':128,
        'criterion':report['score_kind'],'stopping_decision':stop_decision,'decision_time':time.time()}
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
    p.add_argument('--development-config',type=Path);p.add_argument('--decision-file',type=Path)
    p.add_argument('--freeze',action='store_true');p.add_argument('--enqueue',action='store_true');a=p.parse_args()
    decision=json.loads(a.decision_file.read_text()) if a.decision_file else None
    value=freeze(a.root,a.runs,a.route,a.backbone,a.epochs,a.enqueue,a.development_config,decision) if a.freeze else cap_report(a.root,a.runs,a.route,a.backbone,a.epochs,a.development_config)
    print(json.dumps(value,ensure_ascii=False,indent=2))
