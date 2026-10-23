"""A–D analysis of verified frozen features; never select on target outcomes."""
from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd
from .analysis import metrics,unit_weights,fit_matched_head,test_bootstrap,calibration_bootstrap
from .training import sampling_weights
from .scoring import sigmoid,auc,fpr_threshold,pair_transition
from .manifest import save_json


def training_median(df,h):
    if 'experiment_role' not in df or not df.experiment_role.eq('train').all():raise ValueError('Length scale must use source training only')
    r=np.linalg.norm(np.asarray(h,dtype=np.float64),axis=1);w=sampling_weights(df)
    if len(r)!=len(df) or not np.isfinite(r).all():raise ValueError('Invalid training features')
    order=np.argsort(r,kind='stable');c=float(r[order][np.searchsorted(np.cumsum(w[order]),.5,side='left')])
    if c<=0:raise ValueError('Positive source-training median required')
    return c


def length_logits(h,w,b,c):
    h=np.asarray(h,dtype=np.float64);u=h/np.maximum(np.linalg.norm(h,axis=1),1e-12)[:,None]
    if not np.isfinite(c) or c<=0:raise ValueError('Invalid source-only scale')
    return {'h':h@w+b,'h_over_c':(h/c)@w+b,'u':u@w+b,'c_u':(c*u)@w+b}


def tables_from_logits(df,z):
    z=np.asarray(z,dtype=np.float64)
    if z.shape!=(len(df),) or not np.isfinite(z).all():raise ValueError('Invalid aligned logits')
    frame=df.copy();frame['score']=z;frame['probability']=sigmoid(z)
    v=frame[frame.evaluation_unit.eq('video_frame')];keys=['domain','method_id','video_id']
    if len(v):
        if v.groupby(keys).label.nunique().max()!=1 or v.groupby(keys).group_id.nunique().max()!=1:raise ValueError('Mixed video label/resampling group')
        agg={'label':'first','method_type':'first','score':'mean','probability':'mean','group_id':'first','experiment_role':'first'}
        base=v.groupby(keys,sort=True).agg(agg).reset_index();base['sample_id']=base[keys].astype(str).agg('/'.join,axis=1);base['evaluation_unit']='video'
    else:base=pd.DataFrame(columns=[*keys,'label','method_type','score','probability','group_id','experiment_role','sample_id','evaluation_unit'])
    l=base.copy();l['probability']=sigmoid(l.score.to_numpy(dtype=float))
    p=base.copy();p['score']=p.probability
    return {'image':frame,'L':l,'P':p}


def primary_tables(df,h,w,b):
    z=length_logits(h,w,b,1.);raw=tables_from_logits(df,z['h']);direction=tables_from_logits(df,z['u'])
    return {'image_raw':raw['image'],'image_direction':direction['image'],
            'video_RL':raw['L'],'video_RP':raw['P'],'video_DL':direction['L'],'video_DP':direction['P']}


def _save_table(path,df,threshold=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    cols=[k for k in ['sample_id','domain','experiment_role','method_id','method_type','evaluation_unit','video_id','group_id','label','score','probability'] if k in df]
    out=df[cols].copy();out['weight']=unit_weights(df)
    if threshold is not None:out['threshold']=threshold;out['prediction']=(out.score>=threshold).astype(int)
    out.to_parquet(path,index=False)


def _summary(values):
    x=np.asarray(values,dtype=float);valid=x[np.isfinite(x)]
    return {'replicates':len(x),'valid':len(valid),'invalid':len(x)-len(valid),
            'median':float(np.median(valid)) if len(valid) else None,
            'conditional_95_interval':np.quantile(valid,[.025,.975]).tolist() if len(valid) else None}


def contrasts(values):
    rl,rp,dl,dp=(values[k] for k in ['RL','RP','DL','DP'])
    return {'delta_raw':rp-rl,'delta_dir':dp-dl,'delta_norm_L':dl-rl,'delta_norm_P':dp-rp,'interaction':dp-dl-rp+rl}


def _macro_pair_weights(df):
    w=unit_weights(df);y=df.label.to_numpy();real=y==0;w[real]/=w[real].sum()
    methods=sorted(set(df.loc[y==1,'method_id']))
    for m in methods:
        mask=(y==1)&df.method_id.eq(m).to_numpy();w[mask]/=w[mask].sum()*len(methods)
    return w


def _method_metrics(df,threshold):
    result={};real=df.label.eq(0).to_numpy()
    for method in sorted(set(df.loc[df.label.eq(1),'method_id'])):
        mask=real|df.method_id.eq(method).to_numpy();v=metrics(df[mask].reset_index(drop=True),threshold)
        result[method]={k:v[k] for k in ['pooled_AUC','fake_Brier']+(['pooled_TPR'] if threshold is not None else [])}
    return result


def _validate_payload(payload,video_methods,image_methods):
    for role in ['train','ff_test','cdf_test']:
        if role not in payload:raise ValueError('Missing mandatory feature role')
    for role,(df,h) in payload.items():
        if not len(df) or len(h)!=len(df) or not df.experiment_role.eq(role).all() or df.sample_id.duplicated().any():raise ValueError('Feature role/alignment mismatch')
        if not set(df.label)<=set([0,1]) or not df.loc[df.label.eq(0),'method_id'].eq('real').all():raise ValueError('Invalid frozen labels')
        if role in ['train','calibration'] and not df.domain.eq('ff').all():raise ValueError('No target input in training or source calibration')
        if role in ['ff_test','cdf_test']:
            if set(df.loc[df.label.eq(1),'method_id'])!=set(image_methods):raise ValueError('Image method coverage mismatch')
            vids=df[df.evaluation_unit.eq('video_frame')]
            if set(vids.loc[vids.label.eq(1),'method_id'])!=set(video_methods):raise ValueError('Video method coverage mismatch')
            if not vids.groupby(['domain','method_id','video_id']).size().eq(8).all():raise ValueError('Frozen evaluation videos require exactly8frames')


def analyze_payload(payload,w,b,config,out,expected_video_methods,expected_image_methods,seed=2026):
    _validate_payload(payload,expected_video_methods,expected_image_methods)
    out=Path(out);out.mkdir(parents=True,exist_ok=True);train,ht=payload['train'];c=training_median(train,ht)
    tables={role:primary_tables(df,h,w,b) for role,(df,h) in payload.items() if role!='train'}
    calibration=tables.get('calibration');thresholds={};source_rates={}
    if calibration:
        for flow,df in calibration.items():
            real=df[df.label.eq(0)]
            if not len(real):raise ValueError('FPR5 needs real source calibration, never a target substitute')
            t=fpr_threshold(real.score.to_numpy(),unit_weights(real),config['source_FPR_budget'])
            thresholds[flow]=t;source_rates[flow]=float(np.average(real.score.ge(t),weights=unit_weights(real)))
            _save_table(out/'scores/calibration'/f'{flow}.parquet',df,t)
    result={'A':{},'B':{'source_training_median':c,'scales':config['scales'],'pair_transitions':[],'scale_controls':[],'length_controls':[]},
            'FPR5':{'available':calibration is not None,'requires':'independent source real only','thresholds':{k:{'threshold':v,'source_FPR':source_rates[k]} for k,v in thresholds.items()}},
            'TPR95':{'available':False,'reason':'no certified independent fake calibration with complete21video/31image method coverage; real-only calibration does not provide TPR95'},
            'matched_heads':{},'D':{'test_resampling':{},'calibration_resampling':{'available':calibration is not None}},
            'target_selection_performed':False,'independence_claim':'fixed models/manifests; known relationships and declared surrogate groups only; unknown ancestry is not certified'}
    pd.DataFrame({'sample_id':train.sample_id,'norm':np.linalg.norm(ht.astype(float),axis=1),'sampling_mass':sampling_weights(train)}).to_parquet(out/'training_norms.parquet',index=False)
    method_rows=[]
    for role,domain in [('ff_test','ff'),('cdf_test','cdf')]:
        df,h=payload[role];ts=tables[role];summary={'video':{},'image':{},'methods':{}}
        for flow,t in ts.items():
            threshold=thresholds.get(flow);unit,name=flow.split('_',1);v=metrics(t,threshold)
            summary[unit][name]=v;summary['methods'][flow]=_method_metrics(t,threshold)
            for method,m in summary['methods'][flow].items():method_rows.append({'domain':domain,'flow':flow,'method':method,**m})
            _save_table(out/'scores'/domain/f'{flow}.parquet',t,threshold)
        summary.update(contrasts({k:v['method_macro_AUC'] for k,v in summary['video'].items()}));result['A'][domain]=summary
        for before,after in [('RL','RP'),('DL','DP'),('RL','DL'),('RP','DP')]:
            a=ts['video_'+before];z=ts['video_'+after]
            if not np.array_equal(a.sample_id,z.sample_id):raise ValueError('Pair ledger alignment mismatch')
            weights=_macro_pair_weights(a);matrix,delta=pair_transition(a.score,z.score,a.label,weights)
            direct=auc(z.score,z.label,weights)-auc(a.score,a.label,weights)
            if abs(delta-direct)>1e-10:raise ValueError('Pair ledger fails AUC reconstruction')
            credit=np.array([0.,.5,1.]);change=credit[None,:]-credit[:,None]
            result['B']['pair_transitions'].append({'domain':domain,'before':before,'after':after,'states':['wrong','tie','correct'],'table':matrix.tolist(),
              'ledger_delta':delta,'direct_delta':direct,'beneficial_fraction':float(matrix[change>0].sum()/matrix.sum()),
              'harmful_fraction':float(matrix[change<0].sum()/matrix.sum()),'total_changed_fraction':float((matrix.sum()-np.trace(matrix))/matrix.sum())})
        zz=length_logits(h,w,b,c)
        for condition,z in zz.items():
            for agg,t in tables_from_logits(df,z).items():
                v=metrics(t);result['B']['length_controls'].append({'domain':domain,'condition':condition,'aggregation':agg,**v})
                _save_table(out/'controls'/domain/'length'/condition/f'{agg}.parquet',t)
        norms=np.linalg.norm(h.astype(float),axis=1)
        for kind,z in [('raw',zz['h']),('direction',zz['u'])]:
            frame=df.copy();frame['z']=z;frame['r']=norms;frame['positive']=(z>0).astype(float)
            diag=frame[frame.evaluation_unit.eq('video_frame')].groupby(['domain','method_id','video_id']).agg(logit_mean=('z','mean'),logit_std=('z',lambda v:float(np.std(v))),logit_min=('z','min'),logit_max=('z','max'),positive_fraction=('positive','mean'),length_mean=('r','mean')).reset_index()
            for quantile in [0.,.25,.5,.75,1.]:diag[f'logit_q{quantile}']=frame[frame.evaluation_unit.eq('video_frame')].groupby(['domain','method_id','video_id']).z.quantile(quantile).to_numpy()
            (out/'numerics'/domain).mkdir(parents=True,exist_ok=True);diag.to_parquet(out/'numerics'/domain/f'{kind}_video.parquet',index=False)
            for scale in config['scales']:
                controlled=tables_from_logits(df,float(scale)*z);p=sigmoid(float(scale)*z)
                for agg,t in controlled.items():
                    row={'domain':domain,'kind':kind,'scale':scale,'aggregation':agg,**metrics(t),
                      'frame_near_endpoint_fraction':float(((p<=1e-6)|(p>=1-1e-6)).mean()),'frame_exact_endpoint_fraction':float(((p==0)|(p==1)).mean())}
                    result['B']['scale_controls'].append(row)
                    _save_table(out/'controls'/domain/'scale'/kind/str(scale)/f'{agg}.parquet',t)
                post=controlled['L'].copy();post['score']=sigmoid(post.score);post['probability']=post.score
                v=metrics(post);base=metrics(controlled['L'])
                result['B']['scale_controls'].append({'domain':domain,'kind':kind,'scale':scale,'aggregation':'post_mean_sigmoid',**v,
                  'theoretical_order_AUC':base['method_macro_AUC'],'finite_precision_AUC_delta':v['method_macro_AUC']-base['method_macro_AUC'],
                  'exact_tied_video_pairs':int(sum(n*(n-1)//2 for n in post.score.value_counts()))})
                _save_table(out/'controls'/domain/'scale'/kind/str(scale)/'post_mean_sigmoid.parquet',post)
        # Paired target resampling is separate at each evaluation unit.
        for unit in ['video','image']:
            selected={k.split('_',1)[1]:v for k,v in ts.items() if k.startswith(unit+'_')};th={k:thresholds.get(unit+'_'+k) for k in selected}
            reps,metadata=test_bootstrap(selected,th,config['test_bootstrap'],seed)
            arrays={f'{flow}__{metric}':v for flow,values in reps.items() for metric,v in values.items()}
            effects={}
            for metric in sorted(set.intersection(*(set(values) for values in reps.values()))):
                values={flow:values[metric] for flow,values in reps.items()}
                if unit=='video':effect=contrasts(values)
                else:effect={'delta_norm':values['direction']-values['raw']}
                for name,v in effect.items():effects[metric+'__'+name]=_summary(v);arrays['effect__'+metric+'__'+name]=v
            (out/'bootstrap').mkdir(exist_ok=True);np.savez_compressed(out/'bootstrap'/f'test_{domain}_{unit}.npz',**arrays)
            result['D']['test_resampling'][domain if unit=='video' else domain+'_image']={**metadata,'metrics':{f:_summary_dict(v) for f,v in reps.items()},'paired_effects':effects}
    # Same frozen training rows, zero initialization and identical L2/budget for both heads.
    qt=sampling_weights(train);u=ht.astype(float)/np.maximum(np.linalg.norm(ht.astype(float),axis=1),1e-12)[:,None]
    for kind,x in [('raw',ht),('direction',u)]:
        hw,hb,fit=fit_matched_head(x,train.label,qt,config['matched_head_L2'],config['matched_head_max_iter'])
        result['matched_heads'][kind]={'weight':hw.tolist(),'bias':hb,**fit,'encoder_updated':False,'normalization_running_state_updated':False}
        for role,domain in [('ff_test','ff'),('cdf_test','cdf')]:
            df,h=payload[role];features=h.astype(float) if kind=='raw' else h.astype(float)/np.maximum(np.linalg.norm(h.astype(float),axis=1),1e-12)[:,None]
            values=tables_from_logits(df,features@hw+hb)
            result['matched_heads'][kind][domain]={agg:metrics(t) for agg,t in values.items()}
            for agg,t in values.items():_save_table(out/'matched'/kind/domain/f'{agg}.parquet',t)
    if calibration:
        for unit in ['video','image']:
            ct={k.split('_',1)[1]:v for k,v in calibration.items() if k.startswith(unit+'_')}
            targets={f:{d:tables[r][unit+'_'+f] for r,d in [('ff_test','ff'),('cdf_test','cdf')]} for f in ct}
            reps=calibration_bootstrap(ct,targets,config['calibration_bootstrap'],seed,config['source_FPR_budget'])
            rows=[];summaries={}
            for flow,values in reps.items():
                keys=sorted({k for v in values for k in v if k!='valid'})
                summaries[flow]={k:_summary([v.get(k,np.nan) for v in values]) for k in keys}
                for i,v in enumerate(values):rows.append({'replicate':i,'flow':flow,**v})
            pd.DataFrame(rows).to_parquet(out/'bootstrap'/f'calibration_{unit}.parquet',index=False)
            result['D']['calibration_resampling'][unit]={'iterations':config['calibration_bootstrap'],'conditioning':'fixed targets/models; resample declared source groups only','summaries':summaries}
        # Positive affine transform with synchronized thresholds must preserve decisions.
        affine=[]
        for role,domain in [('ff_test','ff'),('cdf_test','cdf')]:
            for flow,t in tables[role].items():
                threshold=thresholds[flow];changed=2.*t.score.to_numpy()+.37;transformed=2.*threshold+.37
                mismatch=int(np.count_nonzero((changed>=transformed)!=(t.score.to_numpy()>=threshold)))
                if mismatch:raise ValueError('Affine synchronized-threshold negative control failed')
                affine.append({'domain':domain,'flow':flow,'scale':2.,'offset':.37,'prediction_mismatches':mismatch})
        result['B']['affine_controls']=affine
    pd.DataFrame(method_rows).to_csv(out/'metrics_by_method.csv',index=False)
    save_json(out/'thresholds.json',{'FPR5':result['FPR5'],'TPR95':result['TPR95']})
    save_json(out/'analysis.json',result);return result


def _summary_dict(values):return {k:_summary(v) for k,v in values.items()}
