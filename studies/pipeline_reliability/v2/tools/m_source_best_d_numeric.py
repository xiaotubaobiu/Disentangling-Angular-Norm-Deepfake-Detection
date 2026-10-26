"""Independent read-only numeric oracle for the frozen selected-M D core.

No producer imports, file/model/network I/O, transport or resource admission.
Inputs are caller-authenticated saved ranks. Decimal order, real-rank bisect
boundaries, integer weighted Mann-Whitney and Fraction ratios independently
recompute every draw. Unverified diagnostics remain explicit blockers.
"""
from collections.abc import Mapping
from bisect import bisect_left, bisect_right
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy.special import expit

FIXED_METHODS=('fsgan','faceswap','simswap','inswap','blendface','uniface','mobileswap','e4s','facedancer',
    'fomm','facevid2vid','wav2lip','MRAA','one_shot_free','pirender','tpsm','lia','danet','sadtalker','mcnet','hyperreenact')
AGGREGATIONS=('mean_logit','mean_probability')
CELLS=tuple((g,r,a) for g in ('correct','collision') for r in ('raw','l2') for a in AGGREGATIONS)


def _require(ok,why):
    if not ok:raise ValueError(why)


def _inventory():
    out={'training_radii.npy','training_reference.json','independent_rank_recomputation.json','limitations.json','rq4/scope.json'}
    for role in ('source_test','target_full'):
        out.update(f'{role}/{name}' for name in ('frame_margins.parquet','video_scores.csv','metrics.csv','effects.csv',
            'control_video_scores.csv','video_diagnostics.csv','method_diagnostics.csv','numerical_audits.csv','pair_transition_ledgers.json','closure_bootstrap.json'))
        out.update(f'{role}/normal_bootstrap_{r}_{a}.json' for r in ('raw','l2') for a in AGGREGATIONS)
    out.update(f'rq4/{role}_frame_margins.parquet' for role in ('source_cal','target_cal','target_test'))
    for r in ('raw','l2'):
        for a in AGGREGATIONS:
            out.update(f'rq4/{r}_{a}/{name}' for name in ('source_cal_scores.csv','target_cal_scores.csv','target_test_scores.csv',
                'calibration.csv','independent_calibration_recompute.json','primary_n20_vs_source_bootstrap.json'))
    _require(len(out)==60,'internal D60 inventory drift')
    return out


def _sequence(value):
    if isinstance(value,str):
        try:value=json.loads(value)
        except (ValueError,TypeError):raise ValueError('JSON sequence required')
    _require(isinstance(value,(list,tuple,np.ndarray)), 'sequence required')
    return list(value)


def _na(value):
    return value is None or (isinstance(value,(float,np.floating)) and math.isnan(float(value)))


class _Audit:
    def __init__(self):self.checks=0;self.max_error=0.;self.draws=0;self.boots=0;self.cal_rows=0;self.names=set();self.bootstrap_states=[];self.numeric_NA_cells=[]
    def equal(self,actual,saved,path,*,exact=False):
        self.checks+=1
        if _na(actual):
            _require(_na(saved),path+': expected NA');return
        if isinstance(actual,(bool,np.bool_)):
            _require(isinstance(saved,(bool,np.bool_)) and bool(actual)==bool(saved),path+': boolean differs');return
        if isinstance(actual,(int,np.integer)):
            _require(not isinstance(saved,(bool,np.bool_)) and isinstance(saved,(int,np.integer,float,np.floating))
                and math.isfinite(float(saved)) and int(saved)==saved and int(saved)==int(actual),path+': integer differs');return
        if isinstance(actual,(float,np.floating,Fraction)):
            _require(not isinstance(saved,(bool,np.bool_)) and isinstance(saved,(int,float,np.integer,np.floating)),path+': numeric required')
            x,y=float(actual),float(saved)
            if math.isinf(x):_require(x==y,path+': infinity differs');return
            error=abs(x-y);_require(math.isfinite(error) and (x==y if exact else error<=1e-10),path+': numeric differs')
            self.max_error=max(self.max_error,error);return
        if isinstance(actual,(list,tuple,np.ndarray)):
            saved=_sequence(saved);_require(len(actual)==len(saved),path+': sequence length differs')
            for i,(x,y) in enumerate(zip(actual,saved)):self.equal(x,y,f'{path}[{i}]',exact=exact)
            return
        if isinstance(actual,Mapping):
            _require(isinstance(saved,Mapping),path+': mapping required')
            for key,value in actual.items():
                _require(key in saved,path+': missing '+str(key));self.equal(value,saved[key],path+'/'+str(key),exact=exact)
            return
        _require(actual==saved,path+': value differs')


class _AbsentClass(ValueError):pass


class _RankOracle:
    """Integer weighted concordance by real-rank bisect; no producer kernels."""
    def __init__(self,videos,fixed_methods,*,allow_macro=True,group_column='group_id',groups=None):
        _require(isinstance(videos,pd.DataFrame),'score DataFrame required')
        required={'video_uid','label','score','method_id',group_column}
        _require(required<=set(videos) and len(videos)>0,'nonempty complete score columns required')
        q=videos.reset_index(drop=True)
        _require(not q[list(required)].isna().any().any() and not q.video_uid.duplicated().any(),'unique complete score identity required')
        _require(set(q.label.unique())=={0,1},'fixed TEST requires both classes')
        self.labels=q.label.to_numpy(dtype=np.int8);self.real=self.labels==0;self.fake=~self.real
        self.methods=tuple(fixed_methods);_require(len(set(self.methods))==len(self.methods),'distinct fixed methods required')
        self.method=q.method_id.to_numpy();self.allow_macro=allow_macro
        _require(q.loc[self.real,'method_id'].eq('real').all() and not q.loc[self.fake,'method_id'].eq('real').any(),'method label inconsistency')
        self.score=q.score.to_numpy(dtype=np.float64);_require(np.isfinite(self.score).all(),'finite deployment score required')
        self.usable=True
        if 'ranking_usable' in q:
            _require(all(isinstance(x,(bool,np.bool_)) for x in q.ranking_usable),'typed ranking usable flags required')
            self.usable=all(q.ranking_usable)
        if 'rank_score_decimal' in q:raw=q.rank_score_decimal.tolist()
        elif 'rank_score' in q:raw=q.rank_score.tolist()
        else:raw=q.score.tolist()
        try:self.ranks=[Decimal(str(x)) for x in raw]
        except Exception as e:raise ValueError('finite exact decimal rank values required') from e
        _require(all(x.is_finite() for x in self.ranks),'finite exact decimal rank values required')
        self.groups=tuple(sorted(q[group_column].unique())) if groups is None else tuple(groups)
        lookup={x:i for i,x in enumerate(self.groups)};_require(len(lookup)==len(self.groups),'distinct groups required')
        _require(set(q[group_column])<=set(lookup),'group assignment outside group inventory')
        self.group_indices=np.array([lookup[x] for x in q[group_column]],dtype=np.int64)
        self.real_order=sorted(np.flatnonzero(self.real),key=self.ranks.__getitem__)
        real_ranks=[self.ranks[i] for i in self.real_order]
        self.fake_rows=np.flatnonzero(self.fake)
        self.left=np.array([bisect_left(real_ranks,self.ranks[i]) for i in self.fake_rows],dtype=np.int64)
        self.right=np.array([bisect_right(real_ranks,self.ranks[i]) for i in self.fake_rows],dtype=np.int64)
        self.method_positions={m:np.flatnonzero(self.method[self.fake_rows]==m) for m in self.methods}
    def evaluate(self,dense,threshold=None):
        dense=np.asarray(dense)
        _require(dense.shape==(len(self.groups),) and dense.dtype.kind in 'iu' and (dense>=0).all(),'nonnegative integer multiplicities required')
        _require(not len(dense) or int(dense.max())<=np.iinfo(np.int64).max//len(self.labels),'integer multiplicity overflow bound exceeded')
        w=dense[self.group_indices].astype(np.int64);nr=int(w[self.real].sum());nf=int(w[self.fake].sum())
        wf=w[self.fake_rows]
        if not nr or not nf:raise _AbsentClass('draw is missing a class')
        _require(2*nr*nf<np.iinfo(np.int64).max,'integer concordance overflow bound exceeded')
        method_mass={m:int(wf[positions].sum()) for m,positions in self.method_positions.items()}
        missing=[m for m in self.methods if not method_mass[m]]
        valid=bool(self.allow_macro and self.usable and self.methods and not missing)
        out=dict(n_real=nr,n_fake=nf,real_mass=float(nr),fake_mass=float(nf),pooled_auc=None,macro_auc=None,
            macro_valid=valid,missing_methods=missing,ranking_usable=self.usable,
            analysis_status='ok' if self.usable else 'unresolved_numeric_ordering',macro_tpr=None,macro_ba=None)
        if self.usable:
            prefix=np.r_[np.int64(0),np.cumsum(w[self.real_order],dtype=np.int64)]
            twice=prefix[self.left]+prefix[self.right]
            contribution=w[self.fake_rows]*twice
            out['pooled_auc']=float(Fraction(int(contribution.sum()),2*nr*nf))
            if valid:
                out['macro_auc']=float(np.mean([float(Fraction(int(contribution[self.method_positions[m]].sum()),2*nr*method_mass[m])) for m in self.methods]))
        if threshold is not None:
            predicted=self.score>=threshold;fp=sum(int(x) for x in w[self.real & predicted]);tp=sum(int(x) for x in w[self.fake & predicted])
            fpr,tpr=fp/nr,tp/nf
            out.update(threshold=float(threshold),operating_score_field='score',false_positives=fp,false_positive_mass=float(fp),
                true_positives=tp,fpr=fpr,tpr=tpr,pooled_ba=(1-fpr+tpr)/2)
            if valid:
                mt=float(np.mean([sum(int(x) for x in w[self.fake & (self.method==m) & predicted])/method_mass[m] for m in self.methods]))
                out.update(macro_tpr=mt,macro_ba=(1-fpr+mt)/2)
        return out


def _table(value,name):
    _require(isinstance(value,pd.DataFrame),name+': DataFrame required');return value


def _effects(cells):
    out={}
    for a in AGGREGATIONS:
        delta={}
        for g in ('correct','collision'):
            raw,l2=cells[(g,'raw',a)],cells[(g,'l2',a)]
            delta[g]=None if raw['pooled_auc'] is None or l2['pooled_auc'] is None else l2['pooled_auc']-raw['pooled_auc']
            out[f'{a}:delta_{g}']=delta[g]
            if g=='correct':out[f'{a}:macro_delta_correct']=None if raw['macro_auc'] is None or l2['macro_auc'] is None else l2['macro_auc']-raw['macro_auc']
        out[f'{a}:interaction']=None if any(x is None for x in delta.values()) else delta['collision']-delta['correct']
    return out


def _closure(frames):
    _require(isinstance(frames,pd.DataFrame) and {'video_uid','group_id','domain','label','video_id','rgb_sha256'}<=set(frames),'closure frame identities required')
    q=frames;_require(not q[['video_uid','group_id','domain','label','video_id','rgb_sha256']].isna().any().any(),'complete closure identity required')
    _require(not (q.groupby('video_uid').group_id.nunique()>1).any(),'natural video spans source groups')
    graph={g:set() for g in q.group_id.unique()}
    for columns in (['domain','label','video_id'],['rgb_sha256']):
        for _,part in q.groupby(columns):
            names=list(part.group_id.unique())
            for name in names[1:]:graph[names[0]].add(name);graph[name].add(names[0])
    component={}
    for start in sorted(graph):
        if start in component:continue
        todo=[start];members=set()
        while todo:
            current=todo.pop()
            if current in members:continue
            members.add(current);todo.extend(graph[current]-members)
        for name in members:component[name]='closure:'+min(members)
    records=q[['video_uid','group_id']].drop_duplicates().to_dict('records')
    for row in records:row['closure_id']=component[row['group_id']]
    return records


def _frame_expected(frame,*,groupings=('correct','collision'),role=None):
    """Independently bind saved video scores/membership to supplied frame margins.

    expit/NumPy mean are the registered machine score primitives, never an AUC
    or bootstrap oracle. Arbitrary precision ranking values remain saved input.
    """
    required={'video_uid','group_id','domain','role','label','method_id','video_id','rgb_sha256','raw_logit','l2_logit'}
    _require(required<=set(frame) and len(frame)>0,'complete frame margins required')
    _require(not frame[list(required)].isna().any().any(),'complete frame margin identities required')
    if role is not None:_require(frame.role.eq(role).all() and frame.domain.eq('ff' if role in ('source_cal','source_test') else 'cdf').all(),'logical D role/domain binding differs')
    _require(not (frame.groupby(['domain','method_id','video_id']).video_uid.nunique()>1).any(),'duplicate canonical video identity')
    _require(np.isfinite(frame[['raw_logit','l2_logit']].to_numpy(float)).all(),'finite frame margins required')
    _require(set(frame.label.unique())<={0,1} and frame.domain.nunique()==1 and frame.role.nunique()==1,'one fixed frame role/domain required')
    _require(not (frame.groupby(['domain','video_id']).label.nunique()>1).any(),'cross-class ID collision')
    _require(not (frame.groupby('rgb_sha256').label.nunique()>1).any(),'cross-class RGB collision')
    for _,part in frame.groupby('video_uid',sort=False):
        _require(len(part)==8 and part.rgb_sha256.nunique()==8,'eight distinct frame RGBs per natural video required')
        _require(len(part[['group_id','domain','role','label','method_id','video_id']].drop_duplicates())==1,'natural video identity differs across frames')
    out={}
    for grouping in groupings:
        keys=['video_uid'] if grouping=='correct' else ['domain','label','video_id']
        for _,part in frame.groupby(keys,sort=False):
            first=part.iloc[0];fake_collision=grouping=='collision' and first.label==1
            uid=f'{first.domain}/collision/{first.video_id}' if fake_collision else first.video_uid
            base=dict(video_uid=uid,domain=first.domain,role=first.role,label=int(first.label),video_id=first.video_id,
                group_id=first.group_id if part.group_id.nunique()==1 else None,
                method_id='collision' if fake_collision else first.method_id,
                member_video_uids=sorted(part.video_uid.unique()),member_group_ids=sorted(part.group_id.unique()),
                n_frames=len(part),n_methods=part.method_id.nunique())
            for representation in ('raw','l2'):
                z=part[representation+'_logit'].to_numpy(float)
                for aggregation in AGGREGATIONS:
                    key=(grouping,representation,aggregation)
                    score=float(np.mean(z)) if aggregation=='mean_logit' else float(np.mean(expit(z)))
                    out.setdefault(key,{})[uid]=dict(base,score=score)
    return out


def _bind_frame_scores(audit,scores,expected,key,path):
    _require(key in expected and not scores.video_uid.duplicated().any() and set(scores.video_uid)==set(expected[key]),path+': frame/video inventory differs')
    for row in scores.to_dict('records'):
        actual=expected[key][row['video_uid']]
        _require('group_id' in row,path+': group_id column required')
        if actual['group_id'] is None:
            # Only a cross-source collision has no single source group. Native
            # producer None serializes to an empty CSV field; registered
            # read_scores(keep_default_na=False) restores that field as ''.
            # A decoded DataFrame may expose that null as NaN. Normalize
            # this field only for the source-derived cross-group collision;
            # absent/empty/NaN groups for natural videos still fail.
            _require(key[0]=='collision' and (_na(row['group_id']) or
                isinstance(row['group_id'],str) and row['group_id']==''),
                path+'/'+row['video_uid']+': group_id must be a source-derived collision null')
            row=dict(row,group_id=None)
        audit.equal(actual,row,path+'/'+row['video_uid'])
        audit.equal(actual['score'],row.get('score'),path+'/'+row['video_uid']+'/machine_score',exact=True)


def _identity_signature(table):
    columns=[name for name in ('video_uid','group_id','domain','role','label','method_id','video_id','member_video_uids','member_group_ids','n_frames','n_methods') if name in table]
    def norm(value):
        if isinstance(value,(tuple,list,np.ndarray)):return tuple(value)
        if _na(value):return None
        return value
    return {row['video_uid']:tuple(norm(row[c]) for c in columns) for row in table[columns].to_dict('records')}


def _counts(saved,groups,rng,path):
    _require(isinstance(saved,Mapping),path+': exact count mapping required')
    expected=np.bincount(rng.choice(len(groups),size=len(groups),replace=True),minlength=len(groups))
    observed=np.zeros(len(groups),dtype=np.int64);lookup={g:i for i,g in enumerate(groups)}
    for group,count in saved.items():
        _require(group in lookup and type(count) is int and count>0,path+': positive strict integer draw count required')
        observed[lookup[group]]=count
    _require(np.array_equal(expected,observed),path+': seeded multiplicities differ')
    return expected


def _intervals(audit,saved,point,draws,name_column,path,*,none_if_no_valid=False):
    if none_if_no_valid and not draws:_require(saved is None,path+': expected absent intervals');return
    _require(isinstance(saved,list) and len(saved)==len(point),path+': exact interval inventory required')
    indexed={r.get(name_column):r for r in saved};_require(len(indexed)==len(point) and set(indexed)==set(point),path+': interval names differ')
    for name,value in point.items():
        values=[row[name] for row in draws if row.get(name) is not None]
        expected=dict(estimate=value,lower=float(np.quantile(values,.025)) if values else None,
            upper=float(np.quantile(values,.975)) if values else None,valid_repeats=len(values),interval_type='pointwise_conditional_percentile')
        audit.equal(expected,indexed[name],path+'/'+name)



def _bootstrap(audit,obj,oracles,groups,sizes,*,repeats,seed,path,closure=None):
    _require(isinstance(obj,dict),path+': bootstrap mapping required')
    is_closure=closure is not None
    group_key='closure_groups' if is_closure else 'source_groups'
    nkey='n_closure_groups' if is_closure else 'n_source_groups'
    fraction=max(sizes)/sum(sizes)
    _require(type(obj.get('repeats')) is int and type(obj.get('seed')) is int and type(obj.get('min_groups')) is int,'strict bootstrap header integer types required')
    audit.equal(dict(repeats=repeats,seed=seed,min_groups=20,max_group_fraction=.5,
        **{group_key:list(groups),nkey:len(groups)},largest_group_fraction=fraction),obj,path,exact=True)
    point_cells={k:o.evaluate(np.ones(len(groups),dtype=np.int64)) for k,o in oracles.items()}
    if is_closure:
        point=_effects(point_cells);audit.equal(point,obj.get('point_estimates'),path+'/point_estimates')
        audit.equal(closure,obj.get('closure'),path+'/closure')
        metrics=obj.get('point_cell_metrics');_require(isinstance(metrics,list) and len(metrics)==8,path+': exact8 point cells required')
        indexed={tuple(row.get(k) for k in ('grouping','representation','aggregation')):row for row in metrics}
        _require(set(indexed)==set(CELLS),path+': point cell inventory differs')
        for key,value in point_cells.items():audit.equal(value,indexed[key],path+'/point_cell/'+str(key))
        metadata=obj.get('rank_metadata');_require(isinstance(metadata,list) and len(metadata)==8,path+': rank metadata inventory required')
        meta={tuple(row.get(k) for k in ('grouping','representation','aggregation')):row for row in metadata}
        _require(set(meta)==set(CELLS),path+': rank metadata cells differ')
        for key,oracle in oracles.items():audit.equal(dict(n_videos=len(oracle.labels),n_tie_bins=len(set(oracle.ranks)),ranking_usable=oracle.usable),meta[key],path+'/rank_metadata/'+str(key))
        scope='conditional_on_known_source_and_collision_closure'
    else:
        value=point_cells[None];audit.equal(value,obj.get('point_estimate'),path+'/point_estimate');point={k:value[k] for k in ('pooled_auc','macro_auc')}
        scope='conditional_on_caller_known_source_groups'
    audit.equal(scope,obj.get('inference_scope'),path+'/inference_scope')
    audit.boots+=1;audit.names.add(path)
    audit.bootstrap_states.append(dict(artifact=path,status=obj.get('status'),draws=len(obj.get('draws') or []),
        pooled_invalid_repeats=obj.get('pooled_invalid_repeats'),macro_invalid_repeats=obj.get('macro_invalid_repeats')))
    if len(groups)<20 or fraction>.5:
        audit.equal('insufficient_closure_groups' if is_closure else 'insufficient_source_groups',obj.get('status'),path+'/status')
        _require(obj.get('draws')==[] and obj.get('intervals') is None,path+': gated draws/CI must be absent')
        audit.equal(dict(macro_invalid_repeats=0,pooled_invalid_repeats=0),obj,path);return
    draws=obj.get('draws');_require(isinstance(draws,list) and len(draws)==repeats,path+': complete draw inventory required')
    rng=np.random.default_rng(seed);valid=[];pooled_invalid=macro_invalid=0
    for i,row in enumerate(draws):
        label=f'{path}/draw/{i}';audit.equal(i,row.get('repeat'),label+'/repeat')
        dense=_counts(row.get('closure_draw_counts' if is_closure else 'group_draw_counts'),groups,rng,label)
        try:cells={k:o.evaluate(dense) for k,o in oracles.items()}
        except _AbsentClass:
            pooled_invalid+=1;macro_invalid+=1;audit.equal(dict(pooled_valid=False,macro_valid=False),row,label)
            for key in point:audit.equal(None,row.get(key),label+'/'+key)
            continue
        macro=all(c['macro_valid'] for k,c in cells.items() if not is_closure or k[0]=='correct')
        macro_invalid+=int(not macro);expected=dict(pooled_valid=True,macro_valid=macro)
        expected.update(_effects(cells) if is_closure else cells[None]);audit.equal(expected,row,label)
        valid.append(expected)
    audit.draws+=len(draws)
    audit.equal(dict(macro_invalid_repeats=macro_invalid,pooled_invalid_repeats=pooled_invalid,
        status='ok' if valid else 'no_valid_draws'),obj,path)
    _intervals(audit,obj.get('intervals'),point,valid,'comparison' if is_closure else 'metric',path+'/intervals',none_if_no_valid=not is_closure)


def _threshold(values):
    values=sorted(float(x) for x in values);_require(values and all(math.isfinite(x) for x in values),'finite real-only CAL scores required')
    return float(np.nextafter(values[len(values)-math.floor(.05*len(values)+1e-12)-1],np.inf))


def _calibration(audit,source,target,test,rows,methods,aggregation,*,cal_repeats,seed,path):
    for name,q in [('source',source),('target',target)]:
        _require(isinstance(q,pd.DataFrame) and {'video_uid','label','score'}<=set(q),path+': CAL table columns required')
        _require(not q.video_uid.duplicated().any() and q.label.eq(0).all(),path+': unique real CAL required')
        _require(not set(q.video_uid)&set(test.video_uid),path+': CAL/TEST overlap')
    _require(len(source)>0,path+': source CAL required')
    _require(isinstance(rows,pd.DataFrame) and len(rows)==2+4*cal_repeats,path+': exact full CAL row inventory required')
    ids=sorted(target.video_uid);_require(len(ids)==len(set(ids)),path+': CAL IDs must be unique')
    score=dict(zip(target.video_uid,target.score));rng=np.random.default_rng(seed);subsets={}
    for repeat in range(cal_repeats):
        permutation=rng.permutation(len(ids))
        for budget in (10,20,40,80):subsets[(repeat,budget)]=[ids[i] for i in permutation[:budget]] if len(ids)>=budget else []
    unique={budget:len({tuple(sorted(v)) for (r,b),v in subsets.items() if b==budget and v}) for budget in (10,20,40,80)}
    oracle=_RankOracle(test,methods);ones=np.ones(len(oracle.groups),dtype=np.int64)
    hash_ids=hashlib.sha256('\n'.join(sorted(test.video_uid)).encode()).hexdigest()
    source_threshold=_threshold(source.score);target_thresholds=[];native=[]
    expected_keys=[('default',-1,0),('source_real',-1,len(source))]+[('target_real',r,b) for r in range(cal_repeats) for b in (10,20,40,80)]
    for i,(row,key) in enumerate(zip(rows.to_dict('records'),expected_keys)):
        label=f'{path}/row/{i}';kind,repeat,budget=key
        audit.equal(dict(calibration_kind=kind,repeat=repeat,budget=budget,test_id_hash=hash_ids),row,label)
        if kind=='target_real':
            selected=subsets[(repeat,budget)];audit.equal(selected,_sequence(row.get('calibration_ids')),label+'/ids')
            audit.equal(unique[budget],row.get('unique_set_count'),label+'/unique_sets')
            if not selected:
                audit.equal('unavailable',row.get('status'),label+'/status')
                for k in ('threshold','n_real','n_fake','real_mass','fake_mass','pooled_auc','macro_auc','ranking_usable','analysis_status',
                    'macro_valid','macro_tpr','macro_ba','missing_methods','operating_score_field','false_positives','false_positive_mass',
                    'true_positives','fpr','tpr','pooled_ba','ancestor_independence'):audit.equal(None,row.get(k),label+'/'+k)
                continue
            threshold=_threshold([score[uid] for uid in selected])
        else:threshold=source_threshold if kind=='source_real' else 0. if aggregation=='mean_logit' else .5
        audit.equal(threshold,row.get('threshold'),label+'/threshold',exact=True)
        expected=oracle.evaluate(ones,threshold);expected.update(status='ok',ancestor_independence='caller_manifest_required')
        audit.equal(expected,row,label)
        native.append(dict(calibration_kind=kind,repeat=repeat,budget=budget,status='PASSED',false_positives=expected['false_positives']))
        if kind=='target_real' and budget==20:target_thresholds.append(threshold)
    audit.cal_rows+=len(rows);audit.names.add(path)
    return source_threshold,target_thresholds,native


def _rq4(audit,obj,test,source_threshold,target_thresholds,*,repeats,seed,path):
    audit.boots+=1;audit.names.add(path)
    audit.bootstrap_states.append(dict(artifact=path,status=obj.get('status'),draws=len(obj.get('draws') or []),
        pooled_invalid_repeats=obj.get('pooled_invalid_repeats'),macro_invalid_repeats=obj.get('macro_invalid_repeats')))
    if not target_thresholds:
        _require(set(obj)=={'status','calibration_budget','available_target_real_videos','intervals'},path+': exact unavailable RQ4 fields required')
        audit.equal(dict(status='UNAVAILABLE_CALIBRATION_BUDGET',calibration_budget=20,intervals=None),obj,path);return
    q=test.reset_index(drop=True);groups=tuple(sorted(q.group_id.unique()));lookup={g:i for i,g in enumerate(groups)}
    labels=q.label.to_numpy();x=q.score.to_numpy(float);indices=np.array([lookup[g] for g in q.group_id]);real=labels==0;fake=labels==1
    _require(real.any() and fake.any() and np.isfinite(x).all(),path+': finite two-class TEST required')
    sizes=[int(q.group_id.eq(g).sum()) for g in groups];fraction=max(sizes)/sum(sizes)
    thresholds=np.array(target_thresholds,dtype=np.float64)
    # Explicit threshold acceptance counts, independent of producer searchsorted.
    accepted=np.array([sum(value>=t for t in thresholds) for value in x],dtype=np.int64)
    source=x>=source_threshold
    def evaluate(dense):
        weights=dense[indices];nr=sum(int(w) for w in weights[real]);nf=sum(int(w) for w in weights[fake])
        if not nr or not nf:raise _AbsentClass('operating draw missing class')
        sfp=sum(int(w) for w in weights[real & source]);stp=sum(int(w) for w in weights[fake & source])
        tfp=sum(int(w)*int(a) for w,a in zip(weights[real],accepted[real]));ttp=sum(int(w)*int(a) for w,a in zip(weights[fake],accepted[fake]))
        sf,st=sfp/nr,stp/nf;tf,tt=tfp/(len(thresholds)*nr),ttp/(len(thresholds)*nf)
        sb,tb=(1-sf+st)/2,(1-tf+tt)/2
        return dict(source_fpr=sf,source_tpr=st,source_pooled_ba=sb,target_mean_fpr=tf,target_mean_tpr=tt,target_mean_pooled_ba=tb,
            delta_fpr=tf-sf,delta_tpr=tt-st,delta_pooled_ba=tb-sb,source_false_positives=sfp,target_mean_false_positives=tfp/len(thresholds),n_real=nr,n_fake=nf)
    point=evaluate(np.ones(len(groups),dtype=np.int64))
    audit.equal(source_threshold,obj.get('source_threshold'),path+'/source_threshold',exact=True)
    audit.equal(dict(repeats=repeats,seed=seed,min_groups=20,max_group_fraction=.5,n_source_groups=len(groups),source_groups=list(groups),
        largest_group_fraction=fraction,source_threshold=source_threshold,calibration_budget=20,calibration_repeats=len(thresholds),point_estimates=point,
        recorded_thresholds_sha256=hashlib.sha256(thresholds.tobytes()).hexdigest(),fixed_test_id_hash=hashlib.sha256('\n'.join(sorted(q.video_uid)).encode()).hexdigest(),
        model_count=1,calibration_sets_are_models=False,
        inference_scope='conditional_on_fixed_model_cal_pool_recorded_sets_and_known_test_groups',
        score_space='frozen_FP64_empirical_operating_score; decision_score_ge_threshold',
        ancestry_scope='caller_manifest_known_or_proxy_groups_only; no_ancestry_certification'),obj,path)
    spread=obj.get('calibration_spread');_require(isinstance(spread,list) and len(spread)==len(thresholds),path+': complete CAL spread required')
    for repeat,(threshold,row) in enumerate(zip(thresholds,spread)):
        pred=x>=threshold;fp=int(pred[real].sum());tp=int(pred[fake].sum());fpr,tpr=fp/real.sum(),tp/fake.sum();ba=(1-fpr+tpr)/2
        audit.equal(dict(calibration_repeat=repeat,threshold=float(threshold),fpr=fpr,tpr=tpr,pooled_ba=ba,false_positives=fp,true_positives=tp,
            delta_fpr=fpr-point['source_fpr'],delta_tpr=tpr-point['source_tpr'],delta_pooled_ba=ba-point['source_pooled_ba'],
            scope='CAL_subset_spread_on_fixed_TEST; not_a_TEST_bootstrap_interval'),row,path+'/spread/'+str(repeat))
    if len(groups)<20 or fraction>.5:
        audit.equal(dict(status='insufficient_source_groups',pooled_invalid_repeats=0),obj,path)
        _require(obj.get('draws')==[] and obj.get('intervals') is None,path+': gated draws/CI must be absent');return
    draws=obj.get('draws');_require(isinstance(draws,list) and len(draws)==repeats,path+': complete RQ4 draws required')
    rng=np.random.default_rng(seed);valid=[];invalid=0
    for repeat,row in enumerate(draws):
        label=f'{path}/draw/{repeat}';audit.equal(repeat,row.get('repeat'),label+'/repeat');dense=_counts(row.get('group_draw_counts'),groups,rng,label)
        try:expected=dict(pooled_valid=True,**evaluate(dense))
        except _AbsentClass:
            expected=dict(pooled_valid=False);invalid+=1
            for key in point:audit.equal(None,row.get(key),label+'/'+key)
        audit.equal(expected,row,label)
        if expected['pooled_valid']:valid.append(expected)
    audit.draws+=len(draws)
    audit.equal(dict(status='ok' if valid else 'no_valid_draws',pooled_invalid_repeats=invalid),obj,path)
    keys=('source_fpr','source_tpr','source_pooled_ba','target_mean_fpr','target_mean_tpr','target_mean_pooled_ba','delta_fpr','delta_tpr','delta_pooled_ba')
    _intervals(audit,obj.get('intervals'),{key:point[key] for key in keys},valid,'comparison',path+'/intervals',none_if_no_valid=True)



def audit_D_artifacts(artifacts: Mapping, *, repeats=2000,cal_repeats=200,resampling_seed=20261023,context=None)->dict:
    """Raise on any core mismatch; consume one cohort/setting at a time.

    Context optionally supplies fixed_methods/job.seed; it never opens paths.
    Exact frozen2000/200/20261023 options are mandatory. Diagnostic exclusions
    are explicit and never silently treated as fully recomputed D60 output.
    """
    _require(isinstance(artifacts,Mapping) and set(artifacts)==_inventory(),'exact D60 logical inventory required')
    _require(type(repeats) is int and repeats==2000 and type(cal_repeats) is int and cal_repeats==200
        and type(resampling_seed) is int and resampling_seed==20261023,'frozen TEST2000/CAL200/seed20261023 required')
    context={} if context is None else context;_require(isinstance(context,Mapping),'context mapping required')
    methods=list(context.get('fixed_methods',sorted(FIXED_METHODS)))
    _require(len(methods)==21 and set(methods)==set(FIXED_METHODS),'exact frozen FS9+FR12 methods required')
    methods=sorted(methods);audit=_Audit()
    limitations=artifacts['limitations.json'];_require(isinstance(limitations,dict),'limitations object required')
    audit.equal(dict(checkpoint_selection='source_best',update_matched_primary_claim=False,calibration_repeats=200,test_bootstrap_repeats=2000,
        resampling_seed=20261023,production_admission_provided_here=False),limitations,'limitations.json')
    seed=limitations.get('source_only_selected_prefix',{}).get('seed')
    _require(type(seed) is int and seed in (7,42,123,2026,3407),'registered disclosed model seed required')
    if 'job' in context:audit.equal(seed,context['job'].get('seed'),'context/job/seed')
    for role in ('source_test','target_full'):
        scores=_table(artifacts[f'{role}/video_scores.csv'],role+'/scores');metrics=_table(artifacts[f'{role}/metrics.csv'],role+'/metrics')
        _require({'grouping','representation','aggregation'}<=set(scores) and len(metrics)==8,'exact D8 metrics and score cell columns required')
        parts={key:part.reset_index(drop=True) for key,part in scores.groupby(['grouping','representation','aggregation'],sort=False)}
        _require(set(parts)==set(CELLS),'exact D8 score cells required')
        indexed={tuple(row[k] for k in ('grouping','representation','aggregation')):row for row in metrics.to_dict('records')}
        _require(set(indexed)==set(CELLS),'exact distinct D8 metrics required')
        frames=_table(artifacts[f'{role}/frame_margins.parquet'],role+'/frames');expected_frames=_frame_expected(frames,role=role)
        point={}
        for key,part in parts.items():
            _bind_frame_scores(audit,part,expected_frames,key,role+'/scores/'+str(key))
            oracle=_RankOracle(part,methods,allow_macro=key[0]=='correct',group_column='video_uid');point[key]=oracle.evaluate(np.ones(len(oracle.groups),dtype=np.int64))
            audit.equal(point[key],indexed[key],role+'/metrics/'+str(key))
        audit.numeric_NA_cells.extend(dict(role=role,cell=list(key)) for key,value in point.items() if value['pooled_auc'] is None)
        audit.names.add(f'{role}/metrics.csv')
        effects=_table(artifacts[f'{role}/effects.csv'],role+'/effects');_require(len(effects)==2,'both aggregation effects required')
        effect=_effects(point)
        for row,aggregation in zip(effects.to_dict('records'),AGGREGATIONS):
            dc,dcol=effect[f'{aggregation}:delta_correct'],effect[f'{aggregation}:delta_collision']
            def sign(x):return None if x is None else 1 if x>1e-10 else -1 if x< -1e-10 else 0
            a,b=sign(dc),sign(dcol);valid=a is not None and b is not None
            audit.equal(dict(seed=seed,aggregation=aggregation,delta_correct=dc,delta_collision=dcol,interaction=effect[f'{aggregation}:interaction'],
                sign_correct=a,sign_collision=b,reversal=bool(valid and a*b==-1),valid=valid,tolerance=1e-10),row,role+'/effects/'+aggregation)
        audit.names.add(f'{role}/effects.csv')
        for representation in ('raw','l2'):
            for aggregation in AGGREGATIONS:
                part=parts[('correct',representation,aggregation)];oracle=_RankOracle(part,methods);groups=oracle.groups
                name=f'{role}/normal_bootstrap_{representation}_{aggregation}.json';obj=artifacts[name]
                _bootstrap(audit,obj,{None:oracle},groups,[int(part.group_id.eq(g).sum()) for g in groups],repeats=repeats,seed=resampling_seed,path=name)
                del obj
        closure=_closure(frames)
        uidmap={row['video_uid']:row['closure_id'] for row in closure};uidset=set(uidmap);groups=tuple(sorted(set(uidmap.values())))
        prepared={}
        for key,part in parts.items():
            part=part.copy();assign=[]
            _require('member_video_uids' in part,'closure membership vectors required')
            for members in part.member_video_uids:
                member_list=_sequence(members);_require(member_list and set(member_list)<=uidset,'known closure members required')
                mapped={uidmap[uid] for uid in member_list};_require(len(mapped)==1,'video spans closure groups');assign.append(next(iter(mapped)))
            part['_closure']=assign;prepared[key]=_RankOracle(part,methods,allow_macro=key[0]=='correct',group_column='_closure',groups=groups)
        name=f'{role}/closure_bootstrap.json';obj=artifacts[name]
        _bootstrap(audit,obj,prepared,groups,[sum(row['closure_id']==g for row in closure) for g in groups],repeats=repeats,seed=resampling_seed,path=name,closure=closure)
        del obj,frames,scores,metrics,parts,prepared,expected_frames
    rq4_expected={}
    for role in ('source_cal','target_cal','target_test'):
        frame=_table(artifacts[f'rq4/{role}_frame_margins.parquet'],role+'/frames')
        rq4_expected[role]=_frame_expected(frame,groupings=('correct',),role=role)
        del frame
    common_pool_identity={}
    for representation in ('raw','l2'):
        for aggregation in AGGREGATIONS:
            folder=f'rq4/{representation}_{aggregation}'
            pools={role:_table(artifacts[f'{folder}/{role}_scores.csv'],folder+'/'+role) for role in ('source_cal','target_cal','target_test')}
            for role,q in pools.items():
                _bind_frame_scores(audit,q,rq4_expected[role],('correct',representation,aggregation),folder+'/'+role)
                signature=_identity_signature(q)
                if role in common_pool_identity:_require(signature==common_pool_identity[role],folder+': CAL/TEST identities must be common across settings')
                else:common_pool_identity[role]=signature
            name=f'{folder}/calibration.csv';rows=_table(artifacts[name],name)
            source,targets,native=_calibration(audit,pools['source_cal'],pools['target_cal'],pools['target_test'],rows,methods,aggregation,cal_repeats=cal_repeats,seed=resampling_seed,path=name)
            name=f'{folder}/independent_calibration_recompute.json';audit.equal(native,artifacts[name],name);audit.names.add(name)
            name=f'{folder}/primary_n20_vs_source_bootstrap.json';obj=artifacts[name]
            if not targets:audit.equal(len(pools['target_cal']),obj.get('available_target_real_videos'),name+'/available_target_real_videos')
            _rq4(audit,obj,pools['target_test'],source,targets,repeats=repeats,seed=resampling_seed,path=name)
            del obj,rows,pools
    inputs=sorted(name for name in _inventory() if name.endswith('_scores.csv') and 'control_' not in name or name.endswith('frame_margins.parquet'))
    unverified=sorted(_inventory()-audit.names-set(inputs))
    return dict(schema='M_SOURCE_BEST_D_INDEPENDENT_NUMERIC_CORE_V1',status='PASS_SELECTED_D_CORE',numeric_core_status='PASS',
        audited_scope='SELECTED_FOUR_CORE_ONLY',accepted_for_selected_core=True,blockers_for_selected_core=[],
        bootstrap_artifacts=audit.boots,draws_recomputed=audit.draws,calibration_rows_recomputed=audit.cal_rows,
        scalar_checks=audit.checks,max_absolute_error=audit.max_error,bootstrap_states=audit.bootstrap_states,
        preserved_numeric_NA_cells=audit.numeric_NA_cells,numerical_artifacts_recomputed=sorted(audit.names),
        saved_score_and_frame_identity_inputs=inputs,unverified_numeric_artifacts=unverified,
        coverage=dict(point_AUC_cells=16,normal_bootstraps=8,closure_bootstraps=2,RQ4_operating_bootstraps=4,
            CAL_rows=4*(2+4*cal_repeats),bootstrap_repeats=repeats,CAL_repeats=cal_repeats,
            paired_effect_CI_basis='recorded_common_source_and_collision_closure',p_values='absent_in_frozen_protocol'),blockers=[],
        all_2000_draws_checked_where_group_gate_allows=True,all_200_calibration_subsets_checked=True,
        all_output_metrics_numerically_recomputed=False,frame_to_video_machine_scores_and_memberships_recomputed=True,
        high_precision_rank_values_are_caller_authenticated_inputs=True,
        rank_algorithm='Decimal exact ordering; real-rank bisect boundaries; integer weighted Mann-Whitney; Fraction ratio',
        conditional_scope='fixed model and caller-authenticated saved score ranks, source/proxy groups and fixed CAL pool',
        source_feature_or_head_derivation_recomputed=False,controls_diagnostics_and_high_precision_aggregation_recomputed=False,
        provenance_transport_or_resource_acceptance=False,scientific_final_acceptance=False)
