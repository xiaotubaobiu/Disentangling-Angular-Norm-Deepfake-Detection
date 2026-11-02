"""Paired, unit-aware metrics and frozen-feature diagnostics; never fit targets."""
from pathlib import Path
import json,hashlib,warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from .scoring import sigmoid,auc,fpr_threshold,pair_transition
from .manifest import digest,save_json


def unit_weights(df):
    weights=np.ones(len(df),dtype=np.float64)
    if 'evaluation_unit' not in df:return weights
    mask=df.evaluation_unit.to_numpy()=='video_frame'
    if mask.any():
        keys=['domain','method_id','video_id']
        counts=df.loc[mask].groupby(keys)['video_id'].transform('size').to_numpy()
        weights[mask]=1/counts
    return weights


def _mean(values,weights):
    return float(np.average(values,weights=weights)) if np.sum(weights)>0 else float('nan')


def metrics(df,threshold=None,weights=None):
    y=df.label.to_numpy();score=df.score.to_numpy();p=df.probability.to_numpy();w=unit_weights(df) if weights is None else np.asarray(weights)
    real=y==0;fake=y==1;methods=sorted(set(df.loc[fake,'method_id']))
    out={'real_videos':int(real.sum()),'fake_units':int(fake.sum()),'fake_methods':len(methods),
      'pooled_AUC':auc(score,y,w) if w[real].sum()>0 and w[fake].sum()>0 else float('nan')}
    method_auc=[]
    for method in methods:
        mask=real|(df.method_id.to_numpy()==method)
        if w[mask&real].sum()>0 and w[mask&fake].sum()>0:method_auc.append(auc(score[mask],y[mask],w[mask]))
    out['method_macro_AUC']=float(np.mean(method_auc)) if method_auc else float('nan')
    out['real_Brier']=_mean(p[real]**2,w[real]);out['fake_Brier']=_mean((1-p[fake])**2,w[fake]);out['class_balanced_Brier']=.5*(out['real_Brier']+out['fake_Brier'])
    out['native_Brier']=_mean((p-y)**2,w)
    if threshold is not None:
        pred=score>=threshold;out['threshold']=float(threshold);out['FPR']=_mean(pred[real],w[real]);out['pooled_TPR']=_mean(pred[fake],w[fake])
        tprs={method:_mean(pred[df.method_id.to_numpy()==method],w[df.method_id.to_numpy()==method]) for method in methods}
        out['macro_TPR']=float(np.mean(list(tprs.values()))) if tprs else float('nan');out['macro_BA']=.5*(out['macro_TPR']+1-out['FPR']);out['TPR_by_method']=tprs
    return out


def score_features(df,h,w,b):
    h=np.asarray(h,dtype=np.float64);w=np.asarray(w,dtype=np.float64)
    if h.shape[0]!=len(df) or not np.isfinite(h).all():raise ValueError('Feature/input alignment or finiteness failure')
    norm=np.maximum(np.linalg.norm(h,axis=1),1e-12)
    logits={'raw':h@w+b,'direction':(h/norm[:,None])@w+b};result={}
    for kind,z in logits.items():
        frame=df.copy();frame['score']=z;frame['probability']=sigmoid(z);result[f'image_{kind}']=frame
        video=frame[frame.evaluation_unit=='video_frame'].copy();keys=['domain','method_id','video_id']
        if len(video):
            if video.groupby(keys).label.nunique().max()!=1:raise ValueError('Mixed video labels')
            agg={'label':'first','score':'mean','probability':'mean','group_id':'first'}
            if 'method_type' in video:agg['method_type']='first'
            base=video.groupby(keys,sort=True).agg(agg).reset_index();base['sample_id']=base[keys].astype(str).agg('/'.join,axis=1);base['evaluation_unit']='video'
        else:base=pd.DataFrame(columns=[*keys,'label','score','probability','group_id','sample_id','evaluation_unit'])
        letter='R' if kind=='raw' else 'D'
        l=base.copy();l['probability']=sigmoid(l.score.to_numpy(dtype=float));result[f'video_{letter}L']=l
        p=base.copy();p['score']=p.probability;result[f'video_{letter}P']=p
    return result


def calibrate(calibration,budget=.05):
    if calibration is None or not len(calibration):raise ValueError('Independent source calibration unavailable; no target substitute')
    if 'experiment_role' in calibration and not calibration.experiment_role.eq('calibration').all():raise ValueError('Non-calibration input rejected')
    real=calibration[calibration.label==0]
    if not len(real):raise ValueError('Source-real calibration unavailable')
    return fpr_threshold(real.score.to_numpy(),unit_weights(real),budget)


def fit_matched_head(features,labels,weights,penalty=1e-4,max_iter=2000):
    x=np.asarray(features,dtype=np.float64);y=np.asarray(labels,dtype=np.float64);q=np.asarray(weights,dtype=np.float64);q=q/q.sum()
    def loss(v):
        z=x@v[:-1]+v[-1];diff=q*(sigmoid(z)-y)
        value=np.sum(q*(np.logaddexp(0,z)-y*z))+.5*penalty*np.dot(v[:-1],v[:-1])
        grad=np.r_[x.T@diff+penalty*v[:-1],diff.sum()]
        return value,grad
    fit=minimize(loss,np.zeros(x.shape[1]+1),jac=True,method='L-BFGS-B',options={'maxiter':max_iter,'gtol':1e-8,'ftol':1e-12})
    return fit.x[:-1],float(fit.x[-1]),{'converged':bool(fit.success),'iterations':int(fit.nit),'message':str(fit.message),'objective':float(fit.fun),'L2':penalty,'bias_penalized':False,'initialization':'zero','target_labels_used':False}


def weighted_auc_replicates(score,label,weights,codes,draws,mask):
    ids=np.flatnonzero(mask);order=ids[np.argsort(np.asarray(score)[ids],kind='stable')];sc=np.asarray(score)[order];lab=np.asarray(label)[order]
    starts=np.r_[0,np.flatnonzero(np.diff(sc)!=0)+1];rep=[]
    for j in range(0,len(draws),64):
        mass=draws[j:j+64,codes[order]]*weights[order]
        r=np.add.reduceat(mass*(lab==0),starts,axis=1);f=np.add.reduceat(mass*(lab==1),starts,axis=1)
        denom=r.sum(1)*f.sum(1);num=(f*(np.cumsum(r,axis=1)-.5*r)).sum(1)
        rep.append(np.divide(num,denom,out=np.full(len(denom),np.nan),where=denom>0))
    return np.concatenate(rep)


def test_bootstrap(tables,thresholds,iterations,seed):
    """Shared group draws across flows; returns replicates, not population guarantees."""
    first=next(iter(tables.values()));keys=first.sample_id.to_numpy();groups,codes=np.unique(first.group_id.to_numpy().astype(str),return_inverse=True)
    draws=np.random.default_rng(seed).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=iterations).astype(float)
    reps={};y=first.label.to_numpy();real=y==0;methods=sorted(set(first.loc[y==1,'method_id']));mv=first.method_id.to_numpy();w=unit_weights(first)
    for flow,df in tables.items():
        if not np.array_equal(df.sample_id.to_numpy(),keys):raise ValueError('Paired bootstrap row alignment failure')
        score=df.score.to_numpy();all_auc=weighted_auc_replicates(score,y,w,codes,draws,np.ones(len(y),bool))
        macro=np.stack([weighted_auc_replicates(score,y,w,codes,draws,real|(mv==m)) for m in methods])
        value={'pooled_AUC':all_auc,'method_macro_AUC':np.mean(macro,axis=0)}
        if thresholds.get(flow) is not None:
            pred=score>=thresholds[flow]
            def rate(mask):
                den=draws@np.bincount(codes[mask],weights=w[mask],minlength=len(groups));num=draws@np.bincount(codes[mask],weights=w[mask]*pred[mask],minlength=len(groups))
                return np.divide(num,den,out=np.full(iterations,np.nan),where=den>0)
            fpr=rate(real);mtpr=np.mean(np.stack([rate(mv==m) for m in methods]),axis=0)
            value.update(FPR=fpr,macro_TPR=mtpr,macro_BA=(mtpr+1-fpr)/2)
        reps[flow]=value
    invalid={flow:{k:int((~np.isfinite(v)).sum()) for k,v in values.items()} for flow,values in reps.items()}
    return reps,{'groups':len(groups),'iterations':iterations,'unit':'frozen manifest group_id','conditioning':'fixed trained models, fixed approved manifests/source policies/thresholds; uncertified group relationships remain outside coverage','multiplicity_adjusted':False,'invalid_replicates':invalid,'macro_missing_method_policy':'replicate invalid; never average a reduced method subset'}


def threshold_rate_lookup(df):
    """Prepare weighted inclusive tail counts once, with no AUC recomputation."""
    y=df.label.to_numpy();mv=df.method_id.to_numpy();score=df.score.to_numpy();w=unit_weights(df)
    methods=sorted(set(mv[y==1]));tails={}
    for name,mask in [('real',y==0)]+[(m,(y==1)&(mv==m)) for m in methods]:
        order=np.argsort(score[mask],kind='stable');s=score[mask][order];q=w[mask][order]
        suffix=np.r_[np.cumsum(q[::-1])[::-1],0.]
        tails[name]=(s,suffix,float(q.sum()))
    def evaluate(threshold):
        def rate(name):
            s,suffix,total=tails[name]
            return float(suffix[np.searchsorted(s,threshold,side='left')]/total) if total>0 else float('nan')
        fpr=rate('real');tpr=float(np.mean([rate(m) for m in methods])) if methods else float('nan')
        return {'FPR':fpr,'macro_TPR':tpr,'macro_BA':.5*(tpr+1-fpr)}
    return evaluate


def calibration_bootstrap(cal_tables,test_tables,iterations,seed,budget=.05):
    first=next(iter(cal_tables.values()));groups,codes=np.unique(first.group_id.astype(str),return_inverse=True)
    draws=np.random.default_rng(seed).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=iterations).astype(float)
    result={}
    for flow,cal in cal_tables.items():
        if not np.array_equal(cal.sample_id.to_numpy(),first.sample_id.to_numpy()):raise ValueError('Calibration resample alignment failure')
        real=cal.label.to_numpy()==0;w=unit_weights(cal);values=[]
        evaluators={domain:threshold_rate_lookup(t) for domain,t in test_tables[flow].items()}
        for d in draws:
            keep=real&(d[codes]>0)
            if not keep.any():values.append({'valid':False});continue
            threshold=fpr_threshold(cal.score.to_numpy()[keep],w[keep]*d[codes[keep]],budget)
            row={'valid':True,'threshold':threshold}
            for domain,evaluate in evaluators.items():
                row.update({f'{domain}_{k}':v for k,v in evaluate(threshold).items()})
            values.append(row)
        result[flow]=values
    return result
