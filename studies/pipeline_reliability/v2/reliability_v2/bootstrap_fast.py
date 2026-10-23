"""Bounded-memory conditional bootstrap for the frozen v2 analysis protocol.

Pure CPU numerical APIs: no file/model loading, training, target selection,
GPU work, or mutation of scoring_d and its frozen training/cache provenance.
Rank/tie groups are constructed ONCE with the original FP64/mpmath objects.
Exact integer group multiplicities then reproduce explicit repeated draws.

RQ4 intervals condition on the fixed model, full frozen CAL pool, the recorded
calibration subsets/thresholds, and caller's known/proxy TEST groups. They do
not treat 200 subsets as 200 models or establish unknown ancestor independence.
The pre-target gates min_groups=20/max_group_fraction=.5 are inherited exactly;
they are conservative operational gates, not mathematical sufficiency claims.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import math
import time

import mpmath as mp
import numpy as np
import pandas as pd

from . import scoring_d as scoring


def _options(repeats, min_groups, max_group_fraction):
    if isinstance(repeats, bool) or int(repeats) != repeats or repeats <= 0:
        raise ValueError('positive integer repeats required')
    if isinstance(min_groups, bool) or int(min_groups) != min_groups or min_groups < 1:
        raise ValueError('positive integer min_groups required')
    if not 0 < max_group_fraction <= 1:
        raise ValueError('max_group_fraction must lie in (0,1]')


@dataclass(frozen=True)
class _GroupIndex:
    names: tuple
    lookup: dict

    @classmethod
    def make(cls, names):
        ordered=tuple(sorted(names))
        return cls(ordered,{name:i for i,name in enumerate(ordered)})


class _DrawCounts(Mapping):
    """Read-only exact multiplicities in sparse integer arrays sharing names.

    Supports dict(mapping), mapping.items()/values(), and lookup like the
    baseline dictionary, without storing millions of per-entry Python objects.
    """
    def __init__(self, groups, dense):
        self._groups=groups
        self._indices=np.flatnonzero(dense).astype(np.int32)
        self._counts=np.asarray(dense[self._indices],dtype=np.int32)

    def __getitem__(self,key):
        index=self._groups.lookup[key]
        position=int(np.searchsorted(self._indices,index))
        if position>=len(self._indices) or self._indices[position]!=index:
            raise KeyError(key)
        return int(self._counts[position])

    def __iter__(self):
        return (self._groups.names[int(i)] for i in self._indices)

    def __len__(self):
        return len(self._indices)

    @property
    def storage_bytes(self):
        return self._indices.nbytes+self._counts.nbytes

    def __repr__(self):
        return f'ExactMultiplicity(groups={len(self)},draws={int(self._counts.sum())})'


def _draw_multiplicities(groups,rng):
    # choice(int G) uses the same integer draws as choice(sorted names),
    # independently regression-tested against materialized closure draws.
    sampled=rng.choice(len(groups.names),size=len(groups.names),replace=True)
    return np.bincount(sampled,minlength=len(groups.names))


def _video_input(videos, *, require_methods=True):
    required={'video_uid','group_id','label','score'} | ({'method_id'} if require_methods else set())
    if not required <= set(videos):
        raise ValueError(f'missing video columns: {sorted(required-set(videos))}')
    q=videos.copy().reset_index(drop=True)
    if q.empty or q[list(required)].isna().any().any() or q.video_uid.duplicated().any():
        raise ValueError('nonempty unique natural videos with complete identity required')
    if set(q.label.unique()) != {0,1}:
        raise ValueError('fixed TEST must include real=0 and fake=1')
    if not all(mp.isfinite(x) for x in q.score):
        raise ValueError('finite score values required')
    return q


class _RankCell:
    """Precision-preserving tie bins plus vectorized weighted concordance."""
    def __init__(self, videos, group_indices, fixed_methods, *, allow_macro=True, threshold=None):
        self.labels=videos.label.to_numpy(dtype=np.int8)
        self.real=self.labels==0
        self.fake=self.labels==1
        self.group_indices=np.asarray(group_indices,dtype=np.int64)
        self.fixed_methods=tuple(fixed_methods)
        self.allow_macro=bool(allow_macro)
        self.usable=bool(videos.ranking_usable.all()) if 'ranking_usable' in videos else True
        self.high_precision=bool(videos.high_precision_used.any()) if 'high_precision_used' in videos else False
        values=list(videos.rank_score if 'rank_score' in videos else videos.score)
        if not all(mp.isfinite(value) for value in values):
            raise ValueError('nonfinite ranking score')
        order=sorted(range(len(values)),key=values.__getitem__)
        self.tie_bin=np.empty(len(values),dtype=np.int64)
        current=-1;previous=None
        for position,index in enumerate(order):
            if position==0 or values[index]!=previous:current+=1
            self.tie_bin[index]=current
            previous=values[index]
        self.n_tie_bins=current+1
        self.real_tie_bin=self.tie_bin[self.real]
        self.fake_tie_bin=self.tie_bin[self.fake]
        method_lookup={method:i for i,method in enumerate(self.fixed_methods)}
        methods=videos.loc[self.fake,'method_id'] if 'method_id' in videos else ['']*int(self.fake.sum())
        self.fake_method=np.asarray([method_lookup.get(method,-1) for method in methods],dtype=np.int64)
        self.registered=self.fake_method>=0
        self.predicted=None if threshold is None else np.asarray([score>=threshold for score in videos.score],dtype=bool)
        self.threshold=threshold

    def evaluate(self, multiplicities):
        w=np.asarray(multiplicities[self.group_indices],dtype=np.float64)
        wr,wf=w[self.real],w[self.fake]
        nr,nf=float(wr.sum()),float(wf.sum())
        if nr<=0 or nf<=0:
            raise ValueError('two-class AUC requires both labels with positive mass')
        nm=len(self.fixed_methods)
        method_mass=np.bincount(self.fake_method[self.registered],weights=wf[self.registered],minlength=nm) if nm else np.empty(0)
        missing=tuple(method for method,mass in zip(self.fixed_methods,method_mass) if mass<=0)
        macro_valid=bool(self.allow_macro and self.usable and nm and not missing)
        auc=macro=float('nan')
        if self.usable:
            real_mass=np.bincount(self.real_tie_bin,weights=wr,minlength=self.n_tie_bins)
            # Prefix+half-tie credit is algebraically the fake/real pair ledger.
            concordance=np.cumsum(real_mass)-.5*real_mass
            pair_contribution=wf*concordance[self.fake_tie_bin]
            auc=float(pair_contribution.sum()/(nr*nf))
            if macro_valid:
                method_numerator=np.bincount(self.fake_method[self.registered],
                    weights=pair_contribution[self.registered],minlength=nm)
                macro=float(np.mean(method_numerator/(nr*method_mass)))
        result=dict(n_real=int(nr),n_fake=int(nf),real_mass=nr,fake_mass=nf,
            pooled_auc=auc,macro_auc=macro,macro_valid=macro_valid,missing_methods=missing,
            ranking_usable=self.usable,analysis_status='ok' if self.usable else 'unresolved_numeric_ordering',
            macro_tpr=float('nan'),macro_ba=float('nan'))
        if self.predicted is not None:
            fp=float(w[self.real & self.predicted].sum())
            tp=float(w[self.fake & self.predicted].sum())
            fpr,tpr=fp/nr,tp/nf
            result.update(threshold=self.threshold,operating_score_field='score',false_positives=int(fp),
                false_positive_mass=fp,true_positives=int(tp),fpr=fpr,tpr=tpr,pooled_ba=(1-fpr+tpr)/2)
            if macro_valid:
                method_tp=np.bincount(self.fake_method[self.registered],
                    weights=(wf*self.predicted[self.fake])[self.registered],minlength=nm)
                mt=float(np.mean(method_tp/method_mass))
                result.update(macro_tpr=mt,macro_ba=(1-fpr+mt)/2)
        return result


def _effect_values(cells):
    result={}
    for aggregation in ('mean_logit','mean_probability'):
        deltas={}
        for grouping in ('correct','collision'):
            raw=cells[(grouping,'raw',aggregation)]
            l2=cells[(grouping,'l2',aggregation)]
            delta=l2['pooled_auc']-raw['pooled_auc']
            result[f'{aggregation}:delta_{grouping}']=delta
            if grouping=='correct':result[f'{aggregation}:macro_delta_correct']=l2['macro_auc']-raw['macro_auc']
            deltas[grouping]=delta
        result[f'{aggregation}:interaction']=deltas['collision']-deltas['correct']
    return result


def _intervals(draws, point, *, name_column='comparison'):
    rows=[]
    for name,value in point.items():
        observed=draws[name].dropna() if name in draws else pd.Series(dtype=float)
        rows.append({name_column:name,'estimate':value,
            'lower':float(observed.quantile(.025)) if len(observed) else float('nan'),
            'upper':float(observed.quantile(.975)) if len(observed) else float('nan'),
            'valid_repeats':len(observed),'interval_type':'pointwise_conditional_percentile'})
    return pd.DataFrame(rows)


def bootstrap_d_fast(frame_scores, *, repeats=2000, seed=20261023, fixed_methods=None,
                     known_edges=(), min_groups=20, max_group_fraction=.5):
    """Common closure bootstrap matching scoring_d's exact draw multiplicities.

    Construct all eight video cells once. Source+same-ID+RGB closure must map
    every correct/collision video to ONE component. No per-draw frame grouping,
    pandas metrics, sorting, or float64 conversion of mpmath rankings occurs.
    Return point estimates, pointwise conditional intervals, draw table, closure
    metadata, rank metadata, and actual preparation/bootstrap timing.
    """
    _options(repeats,min_groups,max_group_fraction)
    if 'draw_id' in frame_scores:raise ValueError('supply original frame cache, not prior bootstrap draws')
    started=time.perf_counter()
    methods=list(fixed_methods) if fixed_methods is not None else sorted(frame_scores.loc[frame_scores.label.eq(1),'method_id'].unique())
    if len(set(methods))!=len(methods):raise ValueError('fixed methods must be unique')
    closure=scoring.collision_closure(frame_scores,known_edges=known_edges)
    sizes=closure.groupby('closure_id').size()
    groups=_GroupIndex.make(sizes.index)
    videos,point_metrics=scoring.score_cached_frames(frame_scores,fixed_methods=methods)
    uid_to_closure=closure.set_index('video_uid').closure_id.to_dict()
    assignments=[]
    for members in videos.member_video_uids:
        mapped={uid_to_closure[uid] for uid in members}
        if len(mapped)!=1:raise ValueError('a video score spans multiple closure components')
        assignments.append(groups.lookup[next(iter(mapped))])
    videos['_closure_index']=assignments
    prepared={};metadata=[]
    for key,part in videos.groupby(['grouping','representation','aggregation'],sort=False):
        cell=_RankCell(part,part._closure_index.to_numpy(),methods,allow_macro=key[0]=='correct')
        prepared[key]=cell
        metadata.append(dict(zip(('grouping','representation','aggregation'),key),n_videos=len(part),
            n_tie_bins=cell.n_tie_bins,high_precision_used=cell.high_precision,ranking_usable=cell.usable))
    point_cells={key:cell.evaluate(np.ones(len(groups.names),dtype=np.int64)) for key,cell in prepared.items()}
    preparation=time.perf_counter()-started
    base=dict(point_estimates=_effect_values(point_cells),point_cell_metrics=point_metrics,
        closure=closure,closure_groups=groups.names,rank_metadata=pd.DataFrame(metadata),
        n_closure_groups=len(groups.names),largest_group_fraction=float(sizes.max()/sizes.sum()),
        repeats=int(repeats),seed=seed,min_groups=min_groups,max_group_fraction=max_group_fraction,
        inference_scope='conditional_on_known_source_and_collision_closure',
        macro_invalid_repeats=0,pooled_invalid_repeats=0,
        preparation_seconds=preparation,bootstrap_seconds=0.,
        multiplicity_storage='sparse_exact_integer_mapping; dict(mapping) exports to JSON')
    if len(groups.names)<min_groups or base['largest_group_fraction']>max_group_fraction:
        return dict(**base,status='insufficient_closure_groups',intervals=None,draws=pd.DataFrame(),
            limitation='closure too few/concentrated; report per-seed descriptions only')
    started=time.perf_counter();rng=np.random.default_rng(seed);rows=[]
    for repeat in range(int(repeats)):
        dense=_draw_multiplicities(groups,rng)
        record=dict(repeat=repeat,closure_draw_counts=_DrawCounts(groups,dense))
        try:
            cells={key:cell.evaluate(dense) for key,cell in prepared.items()}
        except ValueError as error:
            if 'two-class AUC' not in str(error):raise
            base['pooled_invalid_repeats']+=1;base['macro_invalid_repeats']+=1
            rows.append(dict(**record,pooled_valid=False,macro_valid=False));continue
        macro_valid=all(value['macro_valid'] for key,value in cells.items() if key[0]=='correct')
        base['macro_invalid_repeats']+=int(not macro_valid)
        rows.append(dict(**record,pooled_valid=True,macro_valid=macro_valid,**_effect_values(cells)))
    draws=pd.DataFrame(rows)
    base['bootstrap_seconds']=time.perf_counter()-started
    base['saved_multiplicity_array_bytes']=sum(row.closure_draw_counts.storage_bytes for row in draws.itertuples())
    valid=draws[draws.pooled_valid]
    return dict(**base,status='ok' if len(valid) else 'no_valid_draws',draws=draws,
        intervals=_intervals(valid,base['point_estimates']))


def bootstrap_video_metrics_fast(videos, *, threshold=None, repeats=2000, seed=20261023,
                                 fixed_methods=None,min_groups=20,max_group_fraction=.5):
    """Known-source-group bootstrap of correct video metrics, without row copies.

    Integer multiplicities reproduce bootstrap_video_metrics' materialized rows
    including repeated FP counts and fixed-method invalidation. Rank/tie groups
    are prepared once using original FP64/mpmath score objects.
    """
    _options(repeats,min_groups,max_group_fraction)
    started=time.perf_counter();q=_video_input(videos)
    methods=list(fixed_methods) if fixed_methods is not None else sorted(q.loc[q.label.eq(1),'method_id'].unique())
    if len(set(methods))!=len(methods):raise ValueError('fixed methods must be unique')
    sizes=q.groupby('group_id').size();groups=_GroupIndex.make(sizes.index)
    indices=np.asarray([groups.lookup[g] for g in q.group_id],dtype=np.int64)
    prepared=_RankCell(q,indices,methods,threshold=threshold)
    point=prepared.evaluate(np.ones(len(groups.names),dtype=np.int64))
    base=dict(point_estimate=point,n_source_groups=len(groups.names),source_groups=groups.names,
        largest_group_fraction=float(sizes.max()/sizes.sum()),repeats=int(repeats),seed=seed,
        macro_invalid_repeats=0,pooled_invalid_repeats=0,min_groups=min_groups,max_group_fraction=max_group_fraction,
        inference_scope='conditional_on_caller_known_source_groups',
        preparation_seconds=time.perf_counter()-started,bootstrap_seconds=0.)
    if len(groups.names)<min_groups or base['largest_group_fraction']>max_group_fraction:
        return dict(**base,status='insufficient_source_groups',draws=pd.DataFrame(),intervals=None)
    started=time.perf_counter();rng=np.random.default_rng(seed);rows=[]
    for repeat in range(int(repeats)):
        dense=_draw_multiplicities(groups,rng)
        try:
            row=prepared.evaluate(dense)
            base['macro_invalid_repeats']+=int(not row['macro_valid'])
            row.update(repeat=repeat,pooled_valid=True)
        except ValueError as error:
            if 'two-class AUC' not in str(error):raise
            row=dict(repeat=repeat,pooled_valid=False,macro_valid=False)
            base['pooled_invalid_repeats']+=1;base['macro_invalid_repeats']+=1
        row['group_draw_counts']=_DrawCounts(groups,dense);rows.append(row)
    draws=pd.DataFrame(rows);base['bootstrap_seconds']=time.perf_counter()-started
    columns=['pooled_auc','macro_auc']+(['fpr','tpr','pooled_ba','macro_tpr','macro_ba'] if threshold is not None else [])
    valid=draws[draws.pooled_valid]
    return dict(**base,status='ok' if len(valid) else 'no_valid_draws',draws=draws,
        intervals=_intervals(valid,{name:point[name] for name in columns},name_column='metric') if len(valid) else None)


def bootstrap_operating_comparison(fixed_test, source_threshold, target_thresholds, *,
        repeats=2000, seed=20261023, min_groups=20, max_group_fraction=.5,calibration_budget=20):
    """RQ4 target-real mean operating effect versus source on one fixed TEST.

    target_thresholds is the recorded common set of 200 n20 calibration
    thresholds (other recorded counts/budgets are allowed and labeled). For
    video i, acceptance_i = #(threshold_j<=score_i)/J. This exact linear mean
    makes each TEST-group bootstrap equivalent to averaging all J explicitly
    evaluated thresholds on that SAME draw. Decision ties use >= throughout.

    Operates deliberately in the frozen FP64 deployment 'score' space, as
    scoring_d.calibration_experiment does. No AUC/ranking calculation occurs.
    CAL subset spread is returned separately; percentile intervals concern TEST
    resampling conditional on the fixed model/CAL pool/recorded thresholds.
    """
    _options(repeats,min_groups,max_group_fraction)
    if int(calibration_budget)!=calibration_budget or calibration_budget<1:
        raise ValueError('positive calibration_budget required')
    started=time.perf_counter();q=_video_input(fixed_test,require_methods=False)
    if any(isinstance(x,mp.mpf) for x in q.score):
        raise ValueError('RQ4 accepts frozen FP64 operating scores, not arbitrary-precision rank_score')
    thresholds=np.asarray(target_thresholds,dtype=np.float64)
    source=float(source_threshold)
    if thresholds.ndim!=1 or not len(thresholds) or np.isnan(thresholds).any() or math.isnan(source):
        raise ValueError('recorded nonempty thresholds without NaN required')
    x=q.score.to_numpy(dtype=np.float64);real=q.label.to_numpy()==0;fake=~real
    sizes=q.groupby('group_id').size();groups=_GroupIndex.make(sizes.index)
    indices=np.asarray([groups.lookup[group] for group in q.group_id],dtype=np.int64)
    acceptance=np.searchsorted(np.sort(thresholds),x,side='right')/len(thresholds)
    source_acceptance=x>=source
    def sums(mask,weights=None):
        return np.bincount(indices[mask],weights=None if weights is None else weights[mask],minlength=len(groups.names)).astype(np.float64)
    nr,nf=sums(real),sums(fake)
    source_fp,source_tp=sums(real,source_acceptance),sums(fake,source_acceptance)
    target_fp,target_tp=sums(real,acceptance),sums(fake,acceptance)
    def evaluate(dense):
        real_count,fake_count=float(dense@nr),float(dense@nf)
        if real_count<=0 or fake_count<=0:raise ValueError('two-class operating metrics require both labels')
        source_fpr,source_tpr=float(dense@source_fp)/real_count,float(dense@source_tp)/fake_count
        target_fpr,target_tpr=float(dense@target_fp)/real_count,float(dense@target_tp)/fake_count
        source_ba,target_ba=(1-source_fpr+source_tpr)/2,(1-target_fpr+target_tpr)/2
        return dict(source_fpr=source_fpr,source_tpr=source_tpr,source_pooled_ba=source_ba,
            target_mean_fpr=target_fpr,target_mean_tpr=target_tpr,target_mean_pooled_ba=target_ba,
            delta_fpr=target_fpr-source_fpr,delta_tpr=target_tpr-source_tpr,delta_pooled_ba=target_ba-source_ba,
            source_false_positives=int(dense@source_fp),target_mean_false_positives=float(dense@target_fp),
            n_real=int(real_count),n_fake=int(fake_count))
    point=evaluate(np.ones(len(groups.names),dtype=np.int64))
    fixed_real=np.sort(x[real]);fixed_fake=np.sort(x[fake])
    fp=len(fixed_real)-np.searchsorted(fixed_real,thresholds,side='left')
    tp=len(fixed_fake)-np.searchsorted(fixed_fake,thresholds,side='left')
    cal_fpr,cal_tpr=fp/len(fixed_real),tp/len(fixed_fake)
    spread=pd.DataFrame(dict(calibration_repeat=np.arange(len(thresholds)),threshold=thresholds,
        fpr=cal_fpr,tpr=cal_tpr,pooled_ba=(1-cal_fpr+cal_tpr)/2,false_positives=fp,true_positives=tp))
    for name in ('fpr','tpr','pooled_ba'):spread['delta_'+name]=spread[name]-point['source_'+name]
    spread['scope']='CAL_subset_spread_on_fixed_TEST; not_a_TEST_bootstrap_interval'
    base=dict(point_estimates=point,n_source_groups=len(groups.names),source_groups=groups.names,
        largest_group_fraction=float(sizes.max()/sizes.sum()),repeats=int(repeats),seed=seed,
        min_groups=min_groups,max_group_fraction=max_group_fraction,source_threshold=source,
        calibration_budget=int(calibration_budget),calibration_repeats=len(thresholds),
        recorded_thresholds_sha256=hashlib.sha256(thresholds.tobytes()).hexdigest(),
        fixed_test_id_hash=hashlib.sha256('\n'.join(sorted(q.video_uid)).encode()).hexdigest(),
        calibration_spread=spread,pooled_invalid_repeats=0,
        inference_scope='conditional_on_fixed_model_cal_pool_recorded_sets_and_known_test_groups',
        score_space='frozen_FP64_empirical_operating_score; decision_score_ge_threshold',
        ancestry_scope='caller_manifest_known_or_proxy_groups_only; no_ancestry_certification',
        model_count=1,calibration_sets_are_models=False,
        preparation_seconds=time.perf_counter()-started,bootstrap_seconds=0.)
    if len(groups.names)<min_groups or base['largest_group_fraction']>max_group_fraction:
        return dict(**base,status='insufficient_source_groups',draws=pd.DataFrame(),intervals=None)
    started=time.perf_counter();rng=np.random.default_rng(seed);rows=[]
    for repeat in range(int(repeats)):
        dense=_draw_multiplicities(groups,rng)
        try:row=dict(repeat=repeat,pooled_valid=True,**evaluate(dense))
        except ValueError:
            row=dict(repeat=repeat,pooled_valid=False);base['pooled_invalid_repeats']+=1
        row['group_draw_counts']=_DrawCounts(groups,dense);rows.append(row)
    draws=pd.DataFrame(rows);base['bootstrap_seconds']=time.perf_counter()-started
    names=['source_fpr','source_tpr','source_pooled_ba','target_mean_fpr','target_mean_tpr','target_mean_pooled_ba',
        'delta_fpr','delta_tpr','delta_pooled_ba']
    valid=draws[draws.pooled_valid]
    return dict(**base,status='ok' if len(valid) else 'no_valid_draws',draws=draws,
        intervals=_intervals(valid,{name:point[name] for name in names}) if len(valid) else None)
