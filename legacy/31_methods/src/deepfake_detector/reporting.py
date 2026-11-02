"""Verified five-seed summaries: no winning-flow or target selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .manifest import save_json,digest,METHODS
from .stages import stage_identity,completed_stage,finish_stage

def summarize_runs(frame,seeds):
    keys=['backbone','domain','unit','flow'];rows=[]
    numeric=[c for c in frame.select_dtypes(include='number').columns if c!='seed']
    for key,g in frame.groupby(keys,sort=True):
        if len(g)!=len(seeds) or set(g.seed)!=set(seeds):raise ValueError('Summary requires exactly five distinct prescribed seeds')
        row=dict(zip(keys,key));row['n']=len(g)
        for col in numeric:
            v=g[col].to_numpy(dtype=float)
            row.update({col+'_mean':float(np.mean(v)),col+'_SD':float(np.std(v,ddof=1)),col+'_min':float(np.min(v)),col+'_max':float(np.max(v))})
        rows.append(row)
    return pd.DataFrame(rows)

def analyze_run(config,project,backbone,seed):
    from .training import run_identity
    from .queueing import verified_run
    from .cache import load_features
    from .experiments import analyze_payload
    project=Path(project);root=Path(config['artifact_root']);rid=f'{backbone}_seed{seed}'
    job={'run_id':rid,'backbone':backbone,'seed':seed}
    if not verified_run(config,job,root):raise ValueError('Analysis requires verified final checkpoint and all aligned caches')
    frozen=json.loads((root/'data/freeze_complete.json').read_text());roles=['train','ff_test','cdf_test']+(['calibration'] if frozen['calibration_available'] else [])
    inputs={'freeze':root/'data/freeze_complete.json','checkpoint':root/'runs'/rid/'final.pt'}
    inputs.update({r:root/'features'/rid/r/'complete.json' for r in roles})
    identity=stage_identity('analysis_'+rid,config,project,inputs);out=root/'analysis'/rid
    if completed_stage(out,identity):return json.loads((out/'analysis.json').read_text())
    payload={};w=None;b=None
    for role in roles:
        df,h,rw,rb=load_features(root/'features'/rid/role)
        if w is not None and (not np.array_equal(w,rw) or b!=rb):raise ValueError('Cached heads do not match across roles')
        w,b=rw,rb;payload[role]=(df,h)
    result=analyze_payload(payload,w,b,config,out,METHODS['FS']+METHODS['FR'],sum(METHODS.values(),[]),seed=seed)
    save_json(out/'run.json',{'backbone':backbone,'seed':seed,'identity':run_identity(config,backbone,seed,root)})
    finish_stage(out,identity,[str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p.name!='complete.json'])
    return result

def report(config,project):
    project=Path(project);root=Path(config['artifact_root']);rows=[];method_rows=[];effects=[];details=[]
    for backbone in config['backbones']:
        for seed in config['seeds']:
            result=analyze_run(config,project,backbone,seed)
            rid=f'{backbone}_seed{seed}'
            details.append({'backbone':backbone,'seed':seed,'result':result})
            for domain,values in result['A'].items():
                for unit in ['video','image']:
                    for flow,metrics in values[unit].items():
                        rows.append({'backbone':backbone,'seed':seed,'domain':domain,'unit':unit,'flow':flow,**{k:v for k,v in metrics.items() if isinstance(v,(int,float))}})
                effects.append({'backbone':backbone,'seed':seed,'domain':domain,**{k:v for k,v in values.items() if isinstance(v,(int,float))}})
            table=pd.read_csv(root/'analysis'/rid/'metrics_by_method.csv');table['backbone']=backbone;table['seed']=seed;method_rows.append(table)
    out=root/'report';out.mkdir(exist_ok=True);frame=pd.DataFrame(rows)
    frame.to_csv(out/'metrics_by_seed.csv',index=False);summarize_runs(frame,config['seeds']).to_csv(out/'metrics_five_seed_summary.csv',index=False)
    pd.concat(method_rows).to_csv(out/'metrics_by_seed_method.csv',index=False);pd.DataFrame(effects).to_csv(out/'paired_flow_differences.csv',index=False)
    pairs=[]
    for key,g in frame.groupby(['backbone','domain','unit']):
        pivot=g.pivot(index='seed',columns='flow',values='method_macro_AUC')
        for before in pivot.columns:
            for after in pivot.columns:
                if before>=after:continue
                d=pivot[after]-pivot[before];pairs.append(dict(zip(['backbone','domain','unit'],key))|{'before':before,'after':after,'mean_delta':d.mean(),'SD_delta':d.std(),'wins':int((d>0).sum()),'losses':int((d<0).sum()),'ties':int((d==0).sum())})
    pd.DataFrame(pairs).to_csv(out/'paired_five_seed_summary.csv',index=False)
    # Saved sample tables alone reconstruct all primary AUC/threshold metrics.
    from .analysis import metrics
    for row in rows:
        path=root/'analysis'/f'{row["backbone"]}_seed{row["seed"]}'/'scores'/row['domain']/f'{row["unit"]}_{row["flow"]}.parquet'
        sample=pd.read_parquet(path);threshold=float(sample.threshold.iloc[0]) if 'threshold' in sample else None
        recomputed=metrics(sample,threshold)
        for name in ['method_macro_AUC','pooled_AUC','FPR','macro_TPR','macro_BA']:
            if name in row and not np.isclose(row[name],recomputed[name],atol=1e-12,rtol=0,equal_nan=True):raise ValueError('Saved sample metrics fail reconstruction')
    for name,table in consolidate_results(details).items():
        if table.empty:raise ValueError('Missing consolidated report evidence: '+name)
        if set(table.backbone)!=set(config['backbones']) or any(set(g.seed)!=set(config['seeds']) for _,g in table.groupby('backbone')):raise ValueError('Incomplete five-seed detail evidence')
        table.to_csv(out/(name+'.csv'),index=False)
    methods=pd.concat(method_rows)
    lookup={m:kind for kind,ms in METHODS.items() for m in ms};methods['method_type']=methods.method.map(lookup)
    grouped=methods[methods.flow.str.startswith('image_')].groupby(['backbone','seed','domain','flow','method_type'])[['pooled_AUC','fake_Brier','pooled_TPR']].mean().reset_index()
    grouped.to_csv(out/'image_FS_FR_EFS_by_seed.csv',index=False)
    figures=stability_visuals(frame,root,out,config)
    if len(figures)!=6 or not all((out/p).is_file() for p in figures):raise ValueError('Missing prescribed stability/threshold-relative figures')
    record={'status':'ANALYSIS_REPORT_COMPLETE','consolidated_A_B_C_D_evidence':True,'stability_and_source_threshold_relative_figures':figures,'runs':15,'seeds':config['seeds'],'all_primary_metrics_reconstructed_from_saved_scores':True,'TPR95':'UNAVAILABLE','target_selection_performed':False}
    save_json(out/'report.json',record)
    (project/'reports/results.md').write_text('DF40 v1.1 complete: 15 final-epoch models, 30 frozen matched heads. All five seeds are reported separately per backbone. Saved sample scores reconstruct primary metrics. TPR95 is unavailable because independent fake calibration lacks full method coverage.\n\nTables: '+str(out)+'\n')
    return record

def consolidate_results(records):
    collections={k:[] for k in ['scale_controls','length_controls','matched_heads','pair_transitions','test_bootstrap','calibration_bootstrap']}
    def boot_rows(value,path,base,destination):
        if isinstance(value,dict):
            if any(k in value for k in ['replicates','conditional_95_interval','median']):
                destination.append({**base,'scope':'/'.join(path),**value})
            else:
                for key,sub in value.items():boot_rows(sub,path+[str(key)],base,destination)
    for record in records:
        base={k:record[k] for k in ['backbone','seed']};result=record['result']
        for key in ['scale_controls','length_controls','pair_transitions']:
            for row in result['B'][key]:collections[key].append({**base,**row})
        for kind,head in result['matched_heads'].items():
            fit={k:v for k,v in head.items() if k not in ['weight','bias','ff','cdf']}
            for domain in ['ff','cdf']:
                for aggregation,value in head.get(domain,{}).items():collections['matched_heads'].append({**base,'kind':kind,'domain':domain,'aggregation':aggregation,**fit,**value})
        for scope,value in result['D']['test_resampling'].items():
            boot_rows(value,[scope],{**base,'uncertainty':'test sampling conditional on fixed trained model/thresholds','conditioning':value.get('conditioning'),'iterations':value.get('iterations')},collections['test_bootstrap'])
        for scope,value in result['D']['calibration_resampling'].items():
            if isinstance(value,dict):boot_rows(value,[scope],{**base,'uncertainty':'calibration sampling conditional on fixed model/test outcomes','conditioning':value.get('conditioning'),'iterations':value.get('iterations')},collections['calibration_bootstrap'])
    return {k:pd.DataFrame(v) for k,v in collections.items()}

def stability_visuals(frame,root,out,config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=[]
    for backbone in config['backbones']:
        fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
        for ui,unit in enumerate(['video','image']):
            for mi,metric in enumerate(['method_macro_AUC','FPR']):
                ax=axes[ui,mi]
                for (domain,flow),g in frame[(frame.backbone==backbone)&(frame.unit==unit)].groupby(['domain','flow']):
                    g=g.set_index('seed').loc[config['seeds']]
                    ax.plot(range(5),g[metric],marker='o',label=domain+'/'+flow)
                ax.set_xticks(range(5),config['seeds']);ax.set_title(unit+' '+metric);ax.set_xlabel('training seed');ax.set_ylim(0,1);ax.legend(fontsize=6)
        name=backbone+'_five_seed_stability.png';fig.savefig(out/name,dpi=150);plt.close(fig);files.append(name)
        flows=['video_RL','video_RP','video_DL','video_DP','image_raw','image_direction']
        fig,axes=plt.subplots(6,5,figsize=(17,19),constrained_layout=True)
        for si,seed in enumerate(config['seeds']):
            folder=root/'analysis'/f'{backbone}_seed{seed}'/'scores'
            for fi,flow in enumerate(flows):
                cal=pd.read_parquet(folder/'calibration'/f'{flow}.parquet');cal=cal[cal.label.eq(0)]
                threshold=float(cal.threshold.iloc[0]);mean=np.average(cal.score,weights=cal.weight)
                sd=float(np.sqrt(np.average((cal.score-mean)**2,weights=cal.weight)))
                if sd<=0:sd=1.
                ax=axes[fi,si]
                for domain,label in [('calibration','source cal'),('ff','FF test'),('cdf','CDF test')]:
                    d=cal if domain=='calibration' else pd.read_parquet(folder/domain/f'{flow}.parquet')
                    d=d[d.label.eq(0)];value=(d.score-threshold)/sd
                    ax.hist(value,bins=np.linspace(-4,4,81),weights=d.weight/d.weight.sum(),histtype='step',label=label)
                ax.axvline(0,color='black',linewidth=.7);ax.set_title(flow+' seed'+str(seed),fontsize=8);ax.set_xlabel('(score-source threshold)/source real SD',fontsize=6);ax.legend(fontsize=5)
        name=backbone+'_real_scores_relative_to_source_threshold.png';fig.savefig(out/name,dpi=130);plt.close(fig);files.append(name)
    return files
