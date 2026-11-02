"""D-route inference interventions and conditional statistical audits.

Only consumes NEW frozen FP32 feature caches and their original two-output head.
No model loading, training, role assignment, target selection, GPU use, or I/O.
Caller owns the frozen role/ancestor manifests; this module cannot certify them.
Labels: real=0, fake=1. Decisions use score >= threshold everywhere.
"""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

import mpmath as mp
import numpy as np
import pandas as pd
from scipy.special import expit

EPSILON = 1e-12
REQUIRED = ('image_path', 'label', 'method_id', 'video_id', 'video_uid',
            'group_id', 'domain', 'role', 'rgb_sha256')
STATE_NAMES = ('incorrect', 'tie', 'correct')


def _finite(values):
    return all(bool(mp.isfinite(x)) for x in values)


def _validated_frames(frames, expected_frames=8):
    q = frames.copy().reset_index(drop=True)
    missing = set(REQUIRED) - set(q)
    if missing:
        raise ValueError(f'missing frame columns: {sorted(missing)}')
    if q.empty or q[list(REQUIRED)].isna().any().any():
        raise ValueError('empty cache or missing frame identity')
    if not set(q.label.unique()) <= {0, 1}:
        raise ValueError('labels must be real=0, fake=1')
    if q.role.nunique() != 1 or q.domain.nunique() != 1:
        raise ValueError('score each frozen role/domain cache separately')
    if not q.loc[q.label.eq(0), 'method_id'].eq('real').all():
        raise ValueError('real method identity must be real')
    if q.loc[q.label.eq(1), 'method_id'].eq('real').any():
        raise ValueError('fake method cannot be real')
    draw = ['draw_id'] if 'draw_id' in q else []
    for _, part in q.groupby(draw + ['video_uid'], sort=False):
        if part[['domain', 'method_id', 'video_id', 'label', 'group_id']].drop_duplicates().shape[0] != 1:
            raise ValueError('video_uid maps to multiple identities or source groups')
        if expected_frames is not None and (len(part) != expected_frames or part.rgb_sha256.nunique() != expected_frames):
            raise ValueError(f'each natural video must contain {expected_frames} distinct RGB frames')
    # A method-dropped ID crossing real/fake is undefined, even when UID differs.
    if (q.groupby(['domain', 'video_id']).label.nunique() > 1).any():
        raise ValueError('cross-class video ID collision')
    if (q.groupby('rgb_sha256').label.nunique() > 1).any():
        raise ValueError('cross-class RGB collision')
    if (q.groupby(['domain', 'method_id', 'video_id']).video_uid.nunique() > 1).any():
        raise ValueError('duplicate canonical video identity')
    return q


def frame_margins(frames, features_fp32, W, b, *, eps=EPSILON):
    """Return frame identities plus FP64 Raw/L2 margins from the original head.

    W shape (2,d), b shape (2,), real output 0, fake output 1. Radius is
    computed in FP64 from the saved FP32 h; u=h/max(radius,1e-12).
    """
    q = _validated_frames(frames)
    h = np.asarray(features_fp32)
    if h.dtype != np.float32:
        raise ValueError('features must be the saved raw FP32 cache')
    w, bias = np.asarray(W, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] != len(q) or w.shape != (2, h.shape[1]) or bias.shape != (2,):
        raise ValueError('feature/head dimensions do not agree')
    if eps != EPSILON or not np.isfinite(h).all() or not np.isfinite(w).all() or not np.isfinite(bias).all():
        raise ValueError('nonfinite features/head or nonprotocol epsilon')
    delta_w, delta_b = w[1] - w[0], float(bias[1] - bias[0])
    return frame_margins_delta(q, h, delta_w, delta_b, eps=eps)


def frame_margins_delta(frames, features_fp32, delta_w, delta_b, *, eps=EPSILON):
    """Equivalent API when a cache loader already subtracted head rows in FP64."""
    q = _validated_frames(frames)
    h = np.asarray(features_fp32)
    dw = np.asarray(delta_w, dtype=np.float64)
    if h.dtype != np.float32 or h.ndim != 2 or h.shape[0] != len(q) or dw.shape != (h.shape[1],):
        raise ValueError('raw FP32 features and aligned FP64 head difference required')
    if eps != EPSILON or not np.isfinite(h).all() or not np.isfinite(dw).all() or not np.isfinite(delta_b):
        raise ValueError('nonfinite input or nonprotocol epsilon')
    h64 = h.astype(np.float64)
    radius = np.linalg.norm(h64, axis=1)
    q['radius'] = radius
    q['zero_norm'] = radius < eps
    q['epsilon'] = eps
    q['direction_projection'] = (h64 / np.maximum(radius[:, None], eps)) @ dw
    q['bias_difference'] = delta_b
    q['raw_logit'] = h64 @ dw + delta_b
    q['l2_logit'] = q.direction_projection + delta_b
    return q


def _video_scores(frame_scores, grouping, representation, aggregation, logit_column=None):
    q = frame_scores
    if grouping not in ('correct', 'collision'):
        raise ValueError('unknown grouping')
    draw = ['draw_id'] if 'draw_id' in q else []
    keys = draw + (['video_uid'] if grouping == 'correct' else ['domain', 'label', 'video_id'])
    column = logit_column or f'{representation}_logit'
    rows, logit_groups = [], []
    for _, part in q.groupby(keys, sort=False):
        z = part[column].to_numpy(dtype=np.float64)
        if aggregation == 'mean_logit':
            score = float(np.mean(z))
        elif aggregation == 'mean_probability':
            score = float(np.mean(expit(z)))
        else:
            raise ValueError('aggregation must be mean_logit or mean_probability')
        r = part.iloc[0]
        fake_collision = grouping == 'collision' and r.label == 1
        row = dict(domain=r.domain, role=r.role, label=int(r.label),
                   method_id='collision' if fake_collision else r.method_id,
                   video_id=r.video_id,
                   video_uid=f'{r.domain}/collision/{r.video_id}' if fake_collision else r.video_uid,
                   group_id=r.group_id if part.group_id.nunique() == 1 else None,
                   member_video_uids=tuple(sorted(part.video_uid.unique())),
                   member_group_ids=tuple(sorted(part.group_id.unique())),
                   grouping=grouping, representation=representation, aggregation=aggregation,
                   score=score, n_frames=len(part), n_methods=part.method_id.nunique())
        if draw:
            row['draw_id'] = r.draw_id
        rows.append(row)
        logit_groups.append(z)
    out = pd.DataFrame(rows)
    hp = high_precision_aggregate(logit_groups, aggregation=aggregation)
    # Keep machine score for empirical thresholds, distinct precision-safe ranking.
    out['rank_score'] = pd.Series(hp['scores'], dtype=object)
    out['high_precision_used'] = hp['triggered']
    out['ranking_usable'] = hp['ranking_usable']
    out['precision_history'] = [tuple(hp['precision_history'])] * len(out)
    return out


def aggregate_videos(frame_scores, *, grouping='correct', representation='raw', aggregation='mean_logit'):
    """Aggregate a single frozen role, including real-only SOURCE_CAL, without AUC."""
    return _video_scores(_validated_frames(frame_scores), grouping, representation, aggregation)


def precision_safe_score_table(videos):
    """Exportable score rows without mpmath's short default string conversion.

    350 significant digits preserve the protocol's <=320-digit scores. CSV
    writers must serialize this table, not the live mp.mpf rank_score objects.
    """
    out=videos.copy()
    values=out.pop('rank_score') if 'rank_score' in out else out.score
    out['rank_score_kind']=['mpmath' if isinstance(v,mp.mpf) else 'float64' for v in values]
    out['rank_score_decimal']=[mp.nstr(v,n=350,strip_zeros=False) if isinstance(v,mp.mpf) else repr(float(v)) for v in values]
    out['rank_score_export_digits']=350
    return out


def restore_rank_scores(serialized):
    """Restore precision-safe export while keeping comparisons above FP64."""
    out=serialized.copy();values=[]
    with mp.workdps(370):
        for row in out.itertuples():
            if row.rank_score_kind=='mpmath': values.append(mp.mpf(row.rank_score_decimal))
            elif row.rank_score_kind=='float64': values.append(float(row.rank_score_decimal))
            else: raise ValueError('unrecognized rank-score serialization kind')
    out['rank_score']=pd.Series(values,index=out.index,dtype=object)
    return out


def score_eight_cells(frames, features_fp32, W, b, *, fixed_methods=None):
    """Return (video_scores, metric_rows) for the prespecified 2x2x2 cells."""
    return score_cached_frames(frame_margins(frames, features_fp32, W, b), fixed_methods=fixed_methods)


def score_cached_frames(frame_scores, *, fixed_methods=None):
    q = _validated_frames(frame_scores)
    if not {'raw_logit', 'l2_logit'} <= set(q) or not np.isfinite(q[['raw_logit', 'l2_logit']]).all().all():
        raise ValueError('finite FP64 Raw/L2 frame margins required')
    methods = list(fixed_methods) if fixed_methods is not None else sorted(q.loc[q.label.eq(1), 'method_id'].unique())
    scores, metrics = [], []
    for grouping in ('correct', 'collision'):
        for representation in ('raw', 'l2'):
            for aggregation in ('mean_logit', 'mean_probability'):
                v = _video_scores(q, grouping, representation, aggregation)
                scores.append(v)
                m = metric_summary(v, fixed_methods=methods, allow_macro=grouping == 'correct')
                m.update(grouping=grouping, representation=representation, aggregation=aggregation)
                metrics.append(m)
    return pd.concat(scores, ignore_index=True), pd.DataFrame(metrics)


def binary_auc(labels, scores, weights=None):
    """Exact empirical weighted concordance with half credit for ties.

    Scores may be mpmath.mpf; sorting/comparisons preserve their precision.
    Only the final empirical AUC ratio is converted to float.
    """
    y, x = np.asarray(labels), list(scores)
    w = np.ones(len(y), dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    if y.ndim != 1 or len(y) != len(x) or w.shape != y.shape or not set(y) <= {0, 1}:
        raise ValueError('invalid labels/scores/weights')
    if not _finite(x) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('nonfinite scores or invalid weights')
    nr, nf = float(w[y == 0].sum()), float(w[y == 1].sum())
    if nr <= 0 or nf <= 0:
        raise ValueError('two-class AUC requires both labels with positive mass')
    indices = sorted(range(len(x)), key=x.__getitem__)
    concordance, real_below, start = 0., 0., 0
    while start < len(indices):
        stop = start + 1
        while stop < len(indices) and x[indices[stop]] == x[indices[start]]:
            stop += 1
        group = indices[start:stop]
        real = sum(float(w[i]) for i in group if y[i] == 0)
        fake = sum(float(w[i]) for i in group if y[i] == 1)
        concordance += fake * (real_below + .5 * real)
        real_below += real
        start = stop
    return concordance / (nr * nf)


def metric_summary(videos, threshold=None, fixed_methods=None, *, allow_macro=True, threshold_score_field='score'):
    """Same-pool AUC and operating metrics; absent fixed methods invalidate macro."""
    q = videos
    y, scores = q.label.to_numpy(), q.score.to_numpy()
    rank_scores = q.rank_score.to_numpy() if 'rank_score' in q else scores
    ranking_usable = bool(q.ranking_usable.all()) if 'ranking_usable' in q else True
    w = q['weight'].to_numpy(dtype=float) if 'weight' in q else np.ones(len(q))
    nr, nf = float(w[y == 0].sum()), float(w[y == 1].sum())
    out = dict(n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()),
               real_mass=nr, fake_mass=nf,
               pooled_auc=binary_auc(y, rank_scores, w) if ranking_usable else float('nan'),
               ranking_usable=ranking_usable,
               analysis_status='ok' if ranking_usable else 'unresolved_numeric_ordering',
               macro_auc=float('nan'), macro_valid=False, macro_tpr=float('nan'),
               macro_ba=float('nan'), missing_methods=tuple())
    methods = list(fixed_methods) if fixed_methods is not None else sorted(q.loc[q.label.eq(1), 'method_id'].unique())
    missing = [m for m in methods if not ((q.method_id == m) & (q.label == 1)).any()]
    out['missing_methods'] = tuple(missing)
    macro_valid = bool(allow_macro and methods and not missing and ranking_usable)
    out['macro_valid'] = macro_valid
    if macro_valid:
        out['macro_auc'] = float(np.mean([binary_auc(y[(y == 0) | (q.method_id == m)],
            rank_scores[(y == 0) | (q.method_id == m)], w[(y == 0) | (q.method_id == m)]) for m in methods]))
    if threshold is not None:
        operating_scores = q[threshold_score_field].to_numpy()
        predicted = np.asarray([x >= threshold for x in operating_scores])
        fp = int(((y == 0) & predicted).sum())
        fpr = float(w[(y == 0) & predicted].sum()) / nr
        tpr = float(w[(y == 1) & predicted].sum()) / nf
        out.update(threshold=threshold, operating_score_field=threshold_score_field, false_positives=fp,
                   false_positive_mass=float(w[(y == 0) & predicted].sum()),
                   true_positives=int(((y == 1) & predicted).sum()), fpr=fpr, tpr=tpr,
                   pooled_ba=(1 - fpr + tpr) / 2)
        if macro_valid:
            mt = [float(w[(y == 1) & (q.method_id == m) & predicted].sum()) /
                  float(w[(y == 1) & (q.method_id == m)].sum()) for m in methods]
            out.update(macro_tpr=float(np.mean(mt)), macro_ba=(1 - fpr + float(np.mean(mt))) / 2)
    return out


def empirical_fpr_threshold(real_scores, alpha=.05):
    """Smallest feasible FP64 score>=tau boundary under empirical FPR<=alpha.

    nextafter can exceed 1 for saturated probability scores. This real-only
    routine never calls binary_auc or metric_summary.
    """
    x = np.asarray(real_scores, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all() or not 0 <= alpha < 1:
        raise ValueError('finite nonempty real scores and 0<=alpha<1 required')
    allowed = int(math.floor(alpha * len(x) + 1e-12))
    boundary = np.sort(x)[len(x) - allowed - 1]
    return float(np.nextafter(boundary, np.inf))


def calibration_subsets(real_video_ids, *, budgets=(10, 20, 40, 80), repeats=200, seed=20261023):
    """Common random permutations, nested prefixes, unavailable budgets as NA."""
    ids = sorted(set(real_video_ids))
    if any(int(n) != n or n <= 0 for n in budgets) or repeats <= 0:
        raise ValueError('positive integer budgets and repeats required')
    rng, rows = np.random.default_rng(seed), []
    for repeat in range(repeats):
        permutation = rng.permutation(len(ids))
        for n in budgets:
            available = len(ids) >= n
            selected = tuple(ids[i] for i in permutation[:n]) if available else tuple()
            rows.append(dict(repeat=repeat, budget=n, calibration_ids=selected,
                             status='ok' if available else 'unavailable'))
    out = pd.DataFrame(rows)
    for n in budgets:
        mask = out.budget.eq(n)
        count = len({tuple(sorted(x)) for x in out.loc[mask & out.status.eq('ok'), 'calibration_ids']})
        out.loc[mask, 'unique_set_count'] = count
    out['unique_set_count'] = out.unique_set_count.astype(int)
    return out


def _unique_video_table(q, real_only=False, allow_empty=False):
    if (q.empty and not allow_empty) or q.video_uid.isna().any() or q.video_uid.duplicated().any():
        raise ValueError('nonempty unique video_uid rows required')
    if real_only and not q.label.eq(0).all():
        raise ValueError('calibration accepts real videos only')


def calibration_experiment(source_real, target_real, fixed_test, *, aggregation='mean_logit',
                           budgets=(10,20,40,80), repeats=200, seed=20261023, fixed_methods=None):
    """Thresholds on real CAL, all evaluation metrics on one fixed TEST table.

    Reuse the same seed/ID pool across every model/flow. Full candidate CAL pool
    relation exclusions must already have been frozen by manifest preparation.
    Direct ID overlap is blocked here; ancestor independence is not inferred.
    """
    _unique_video_table(source_real,real_only=True)
    _unique_video_table(target_real,real_only=True,allow_empty=True)
    _unique_video_table(fixed_test)
    if set(target_real.video_uid) & set(fixed_test.video_uid):
        raise ValueError('entire target CAL pool must be disjoint from TEST')
    if set(source_real.video_uid) & set(fixed_test.video_uid):
        raise ValueError('source CAL and TEST overlap')
    if aggregation not in ('mean_logit', 'mean_probability'):
        raise ValueError('unknown aggregation')
    test_hash = hashlib.sha256('\n'.join(sorted(fixed_test.video_uid)).encode()).hexdigest()
    rows = []
    def evaluate(kind, threshold, **extras):
        r = metric_summary(fixed_test, threshold, fixed_methods)
        r.update(calibration_kind=kind, status='ok', test_id_hash=test_hash,
                 ancestor_independence='caller_manifest_required', **extras)
        rows.append(r)
    evaluate('default', 0. if aggregation == 'mean_logit' else .5, repeat=-1, budget=0)
    evaluate('source_real', empirical_fpr_threshold(source_real.score), repeat=-1, budget=len(source_real))
    index = target_real.set_index('video_uid')
    subsets = calibration_subsets(target_real.video_uid, budgets=budgets, repeats=repeats, seed=seed)
    for r in subsets.to_dict('records'):
        if r.pop('status') != 'ok':
            rows.append(dict(calibration_kind='target_real', status='unavailable', test_id_hash=test_hash, **r))
            continue
        evaluate('target_real', empirical_fpr_threshold(index.loc[list(r['calibration_ids']), 'score']), **r)
    return pd.DataFrame(rows)


def oracle_operating_points(videos, *, fixed_methods=None):
    """Post-hoc diagnostic maxima at actual attainable ROC vertices only."""
    field = 'rank_score' if 'rank_score' in videos else 'score'
    scores = list(videos[field])
    if 'ranking_usable' in videos and not videos.ranking_usable.all():
        raise ArithmeticError('oracle unavailable until high-precision ordering resolves')
    if not len(scores) or not _finite(scores):
        raise ValueError('finite scores required')
    # Every distinct-score decision set plus all-real is an actual vertex.
    maximum=max(scores)
    all_real_boundary = mp.inf if isinstance(maximum,mp.mpf) else float(np.nextafter(maximum,np.inf))
    thresholds = [all_real_boundary] + sorted(set(scores), reverse=True)
    vertices = pd.DataFrame([metric_summary(videos, t, fixed_methods,threshold_score_field=field) for t in thresholds])
    rows = []
    for name, limit in [('max_ba', None), ('fpr_le_0.05', .05), ('fpr_le_0.10', .10)]:
        subset = vertices if limit is None else vertices[vertices.fpr <= limit + 1e-15]
        # At a constrained vertex choose highest TPR, breaking ties by BA/FPR.
        order = ['pooled_ba', 'tpr', 'threshold'] if limit is None else ['tpr', 'pooled_ba', 'threshold']
        row = subset.sort_values(order, ascending=False).iloc[0].to_dict()
        row.update(criterion=name, n_actual_vertices=len(vertices), diagnostic_only=True)
        rows.append(row)
    return pd.DataFrame(rows)


def pair_transition_ledger(labels, before, after, weights=None, *, chunk_size=128):
    """3x3 incorrect/tie/correct fake-real pair transitions, exact DeltaAUC.

    Supports arbitrary precision score objects and repeated-bootstrap weights.
    Chunking bounds temporary memory rather than building the full pair table.
    """
    y, a, b = np.asarray(labels), np.asarray(before), np.asarray(after)
    if len(a) != len(y) or len(b) != len(y):
        raise ValueError('aligned score rows required')
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    auc_a, auc_b = binary_auc(y, a, w), binary_auc(y, b, w)
    real, fake = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    counts = np.zeros((3, 3), dtype=np.float64)
    for start in range(0, len(fake), chunk_size):
        fi = fake[start:start + chunk_size]
        aa, ar = a[fi, None], a[real][None, :]
        bb, br = b[fi, None], b[real][None, :]
        state_a = (aa > ar).astype(int) * 2 + (aa == ar).astype(int)
        state_b = (bb > br).astype(int) * 2 + (bb == br).astype(int)
        mass = w[fi, None] * w[real][None, :]
        for i in range(3):
            for j in range(3):
                counts[i, j] += float(mass[(state_a == i) & (state_b == j)].sum())
    delta = float(sum(counts[i,j] * (j-i)/2 for i in range(3) for j in range(3)) / counts.sum())
    error = abs(delta - (auc_b - auc_a))
    if error > 1e-10:
        raise ArithmeticError(f'pair ledger failed DeltaAUC reconstruction: {error}')
    return dict(state_names=STATE_NAMES, counts=counts, pair_mass=float(counts.sum()),
                auc_before=auc_a, auc_after=auc_b, delta_auc=delta, reconstruction_error=error)


def _ordering_signature(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    return tuple((i, bool(k and values[i] == values[order[k-1]])) for k, i in enumerate(order))


def high_precision_aggregate(logit_groups: Sequence, *, aggregation='mean_probability',
                             force=False, precisions=(80,160,320), reference_scores=None, logit_scale=1.):
    """Recompute identical FP64 logits with mpmath; never float-rerank.

    Auto-trigger for float probability endpoints/ties or a negative-control
    order mismatch. Stability requires unchanged ordering at two precisions
    AND no unresolved endpoint saturation. Even 320 digits may be insufficient;
    that condition is returned as unusable rather than silently accepted.
    """
    groups = [np.asarray(z, dtype=np.float64) for z in logit_groups]
    if not groups or any(not len(z) or not np.isfinite(z).all() for z in groups):
        raise ValueError('finite nonempty per-video logit groups required')
    if aggregation not in ('mean_logit', 'mean_probability', 'sigmoid_mean_logit'):
        raise ValueError('unknown aggregation')
    if not np.isfinite(logit_scale) or logit_scale <= 0:
        raise ValueError('positive finite whole-margin scale required')
    numeric = [float((logit_scale*z).mean()) if aggregation == 'mean_logit' else
               float(expit(logit_scale*z).mean()) if aggregation == 'mean_probability' else
               float(expit((logit_scale*z).mean())) for z in groups]
    ties = len(set(numeric)) != len(numeric)
    endpoints = aggregation != 'mean_logit' and any(x in (0.,1.) for x in numeric)
    mismatch = reference_scores is not None and _ordering_signature(numeric) != _ordering_signature(list(reference_scores))
    if not (force or ties or endpoints or mismatch):
        return dict(scores=numeric, precision_history=[], stable=True, ranking_usable=True,
                    triggered=False, reason='no_numeric_trigger')
    history, previous, stable, result = [], None, False, None
    for dps in precisions:
        with mp.workdps(dps):
            result = []
            for z in groups:
                values = [mp.mpf(float(x))*mp.mpf(str(logit_scale)) for x in z]
                if aggregation == 'mean_logit':
                    value = mp.fsum(values) / len(values)
                elif aggregation == 'mean_probability':
                    value = mp.fsum([1/(1+mp.exp(-x)) for x in values]) / len(values)
                else:
                    value = 1/(1+mp.exp(-mp.fsum(values)/len(values)))
                result.append(value)
            signature = _ordering_signature(result)
            saturated = aggregation != 'mean_logit' and any(x == 0 or x == 1 for x in result)
            control_ok = reference_scores is None or signature == _ordering_signature(list(reference_scores))
            stable = previous == signature and not saturated and control_ok
            history.append(dps)
            if stable:
                break
            previous = signature
    return dict(scores=result, precision_history=history, stable=stable, ranking_usable=stable,
                triggered=True, reason='negative_control_failure' if mismatch else 'ties_or_endpoints_or_forced',
                scores_are_arbitrary_precision=True)


def weighted_median(values, weights):
    x, w = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.shape != x.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('finite radii with nonnegative nonzero training mass required')
    order = np.argsort(x, kind='stable')
    return float(x[order[np.searchsorted(np.cumsum(w[order]), w.sum()/2, side='left')]])


def controls_and_diagnostics(frame_scores, *, training_radii, training_weights, fixed_methods=None):
    """Complete-video controls and per-video mechanism audit tables.

    c is the weighted TRAIN radius median, never a target statistic. Positive
    scaling applies a*z per frame to all frames of each video before pooling.
    Local-affine residual fits sigmoid(z) on z within each video; Jensen gap
    is mean(sigmoid(z))-sigmoid(mean(z)).
    """
    q = _validated_frames(frame_scores)
    c = weighted_median(training_radii, training_weights)
    controls, diagnostics, method_rows, numerical, transitions = [], [], [], [], []
    cached, _ = score_cached_frames(q, fixed_methods=fixed_methods)
    for grouping in ('correct', 'collision'):
        keys = (['draw_id'] if 'draw_id' in q else []) + (['video_uid'] if grouping == 'correct' else ['domain','label','video_id'])
        parts = list(q.groupby(keys, sort=False))
        for representation in ('raw','l2'):
            base = _video_scores(q,grouping,representation,'mean_logit')
            zgroups = [part[f'{representation}_logit'].to_numpy(float) for _,part in parts]
            for scale in (.1,1.):
                sq=q.copy();sq['_control_logit']=scale*sq[f'{representation}_logit']
                for aggregation in ('mean_logit','mean_probability'):
                    v=_video_scores(sq,grouping,representation,aggregation,'_control_logit')
                    v['control']='positive_scale';v['scale']=scale;controls.append(v)
                hp_scale=high_precision_aggregate(zgroups,aggregation='mean_logit',
                    reference_scores=base.rank_score.tolist(),logit_scale=scale)
                numerical.append(dict(grouping=grouping,representation=representation,
                    aggregation='scaled_mean_logit',scale=scale,
                    precision_history=hp_scale['precision_history'],stable=hp_scale['stable'],
                    ranking_usable=hp_scale['ranking_usable'],triggered=hp_scale['triggered'],
                    reason=hp_scale['reason'],negative_control_order_preserved=bool(hp_scale['ranking_usable'] and
                        _ordering_signature(hp_scale['scores'])==_ordering_signature(base.rank_score.tolist()))))
            v=base.copy();v['score']=expit(v.score);v['control']='sigmoid_mean_logit';v['scale']=1.;controls.append(v)
            for aggregation, reference in [('mean_probability',None),('sigmoid_mean_logit',base.score.tolist())]:
                hp=high_precision_aggregate(zgroups,aggregation=aggregation,reference_scores=reference)
                numerical.append(dict(grouping=grouping,representation=representation,aggregation=aggregation,
                    precision_history=hp['precision_history'],stable=hp['stable'],ranking_usable=hp['ranking_usable'],
                    triggered=hp['triggered'],reason=hp['reason'],
                    auc_high_precision=binary_auc(base.label,hp['scores']) if hp['ranking_usable'] else float('nan'),
                    negative_control_order_preserved=bool(hp['ranking_usable']) if reference is not None else None))
            for (_,part),z in zip(parts,zgroups):
                p=expit(z);mean=float(z.mean())
                variance=float(np.mean((z-mean)**2))
                slope=float(np.mean((z-mean)*(p-p.mean()))/variance) if variance>0 else 0.
                residual=p-(float(p.mean())+slope*(z-mean))
                diagnostics.append(dict(grouping=grouping,representation=representation,
                    video_uid=part.video_uid.iloc[0],label=int(part.label.iloc[0]),n_frames=len(z),
                    n_methods=part.method_id.nunique(),mean_logit=mean,frame_variance=variance,
                    jensen_gap=float(p.mean()-expit(mean)),local_affine_rmse=float(np.sqrt(np.mean(residual**2))),
                    probability_exact_zeros=int((p==0).sum()),probability_exact_ones=int((p==1).sum()),
                    probability_unique_count=len(set(p)),
                    between_method_variance=float(part.groupby('method_id')[f'{representation}_logit'].mean().var(ddof=0))))
        for aggregation in ('mean_logit','mean_probability'):
            a=cached.query('grouping == @grouping and representation == "raw" and aggregation == @aggregation')
            b=cached.query('grouping == @grouping and representation == "l2" and aggregation == @aggregation')
            ledger=pair_transition_ledger(a.label,a.rank_score,b.rank_score) if a.ranking_usable.all() and b.ranking_usable.all() else dict(status='unresolved_numeric_ordering')
            transitions.append(dict(grouping=grouping,comparison='raw_to_l2',aggregation=aggregation,**ledger))
        for representation in ('raw','l2'):
            a=cached.query('grouping == @grouping and representation == @representation and aggregation == "mean_logit"')
            b=cached.query('grouping == @grouping and representation == @representation and aggregation == "mean_probability"')
            ledger=pair_transition_ledger(a.label,a.rank_score,b.rank_score) if a.ranking_usable.all() and b.ranking_usable.all() else dict(status='unresolved_numeric_ordering')
            transitions.append(dict(grouping=grouping,comparison='logit_to_probability',representation=representation,**ledger))
        cq=q.copy();cq['_control_logit']=c*cq.direction_projection+cq.bias_difference
        for aggregation in ('mean_logit','mean_probability'):
            v=_video_scores(cq,grouping,'c_direction',aggregation,'_control_logit');v['control']='weighted_train_radius_direction';v['scale']=c;controls.append(v)
    for (method,label),part in q.groupby(['method_id','label'],sort=False):
        method_rows.append(dict(method_id=method,label=int(label),n_frames=len(part),
            raw_frame_variance=float(part.raw_logit.var(ddof=0)),l2_frame_variance=float(part.l2_logit.var(ddof=0)),
            radius_mean=float(part.radius.mean()),radius_median=float(part.radius.median())))
    return dict(training_weighted_radius_median=c,controls=pd.concat(controls,ignore_index=True),
                video_diagnostics=pd.DataFrame(diagnostics),method_diagnostics=pd.DataFrame(method_rows),
                numerical_audits=pd.DataFrame(numerical),transition_ledgers=transitions)


def collision_closure(frames, *, known_edges=()):
    """Union known source group edges AND all method-dropped collision edges.

    known_edges contains pairs of caller source group IDs. Existing group_id
    must already encode known ancestry/RGB relationships. Unknown relations
    remain unknown and must be disclosed by the caller.
    """
    q=_validated_frames(frames)
    groups=sorted(set(q.group_id));parent={g:g for g in groups}
    def find(x):
        if x not in parent: parent[x]=x
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[max(ra,rb)]=min(ra,rb)
    for a,b in known_edges: union(a,b)
    for _,part in q.groupby(['domain','label','video_id']):
        members=sorted(part.group_id.unique())
        for other in members[1:]: union(members[0],other)
    # RGB equality is also a known connection even if caller supplied coarse groups incorrectly.
    for _,part in q.groupby('rgb_sha256'):
        members=sorted(part.group_id.unique())
        for other in members[1:]: union(members[0],other)
    records=q[['video_uid','group_id']].drop_duplicates().copy()
    records['closure_id']=[f'closure:{find(g)}' for g in records.group_id]
    return records.reset_index(drop=True)


def resample_closure(frames, closure, sampled_closure_ids):
    """Materialize repeated draws with draw_id; later groupby must retain it."""
    mapping=closure.set_index('video_uid').closure_id.to_dict()
    if not set(frames.video_uid)<=set(mapping): raise ValueError('closure misses videos')
    assignment=frames.video_uid.map(mapping)
    chunks=[]
    for draw_id,g in enumerate(sampled_closure_ids):
        part=frames[assignment.eq(g)].copy()
        if part.empty: raise ValueError('unknown sampled closure ID')
        part['draw_id']=draw_id;chunks.append(part)
    return pd.concat(chunks,ignore_index=True) if chunks else frames.iloc[:0].assign(draw_id=pd.Series(dtype=int))


def _comparisons(metrics):
    out={}
    for aggregation in ('mean_logit','mean_probability'):
        d={}
        for grouping in ('correct','collision'):
            part=metrics[(metrics.grouping==grouping)&(metrics.aggregation==aggregation)].set_index('representation')
            d[grouping]=float(part.loc['l2','pooled_auc']-part.loc['raw','pooled_auc'])
            out[f'{aggregation}:delta_{grouping}']=d[grouping]
            if grouping == 'correct':
                out[f'{aggregation}:macro_delta_correct']=float(part.loc['l2','macro_auc']-part.loc['raw','macro_auc'])
        out[f'{aggregation}:interaction']=d['collision']-d['correct']
    return out


def d_effects(metrics, *, seed=None, tolerance=1e-10):
    """Per-seed signed L2-Raw deltas and collision-minus-correct interaction."""
    rows=[]
    def sign(x):
        return None if not np.isfinite(x) else (1 if x>tolerance else -1 if x< -tolerance else 0)
    for aggregation in ('mean_logit','mean_probability'):
        part=metrics[metrics.aggregation.eq(aggregation)]
        if len(part)!=4 or part.duplicated(['grouping','representation']).any():
            raise ValueError('exactly four grouping/representation cells per aggregation required')
        table=part.set_index(['grouping','representation']).pooled_auc
        correct=float(table.loc[('correct','l2')]-table.loc[('correct','raw')])
        collision=float(table.loc[('collision','l2')]-table.loc[('collision','raw')])
        sc,si=sign(correct),sign(collision)
        valid=sc is not None and si is not None
        rows.append(dict(seed=seed,aggregation=aggregation,delta_correct=correct,
            delta_collision=collision,interaction=collision-correct,sign_correct=sc,
            sign_collision=si,reversal=bool(valid and sc*si == -1),valid=valid,tolerance=tolerance))
    return pd.DataFrame(rows)


def summarize_seed_effects(effects, *, expected_seeds=(7,42,123,2026,3407), tolerance=1e-10):
    """Sample SD across training seeds; mean reversal is distinct from seed count."""
    rows=[]
    for aggregation,part in effects.groupby('aggregation',sort=False):
        if part.seed.duplicated().any(): raise ValueError('duplicate training seeds in effect summary')
        complete=set(part.seed)==set(expected_seeds) and bool(part.valid.all())
        row=dict(aggregation=aggregation,n_seeds=len(part),complete_five_seeds=complete,
                 n_valid_seeds=int(part.valid.sum()),n_seed_reversals=int(part.reversal.sum()))
        for column in ('delta_correct','delta_collision','interaction'):
            values=part.loc[part.valid,column]
            row[f'{column}_mean']=float(values.mean())
            row[f'{column}_sample_sd']=float(values.std(ddof=1)) if len(values)>1 else float('nan')
        a,b=row['delta_correct_mean'],row['delta_collision_mean']
        row['mean_reversal']=bool((a>tolerance and b< -tolerance) or (a< -tolerance and b>tolerance))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_d_comparisons(frame_scores, *, repeats=2000, seed=20261023, fixed_methods=None,
                            known_edges=(), min_groups=20, max_group_fraction=.5):
    """Common closure-group bootstrap with pointwise conditional intervals.

    Conservative reporting gate defaults: >=20 connected groups and largest
    group <=50% of natural videos. This is an explicit operational limitation,
    not a mathematical sufficiency guarantee. Otherwise describe across seeds.
    Missing fixed methods invalidate macro for that draw; two-class failures
    invalidate its pooled comparisons. Training seeds are never pooled as IID.
    """
    q=_validated_frames(frame_scores)
    if fixed_methods is None: fixed_methods=sorted(q.loc[q.label.eq(1),'method_id'].unique())
    closure=collision_closure(q,known_edges=known_edges)
    sizes=closure.groupby('closure_id').size();groups=sorted(sizes.index)
    score_table,point=score_cached_frames(q,fixed_methods=fixed_methods)
    closure_map=closure.set_index('video_uid').closure_id.to_dict()
    score_closures=[]
    for members in score_table.member_video_uids:
        ids={closure_map[uid] for uid in members}
        if len(ids)!=1:raise ValueError('collision closure failed to contain every score member')
        score_closures.append(next(iter(ids)))
    score_table['closure_id']=score_closures
    base=dict(point_estimates=_comparisons(point),n_closure_groups=len(groups),
              largest_group_fraction=float(sizes.max()/sizes.sum()),repeats=repeats,
              min_groups=min_groups,max_group_fraction=max_group_fraction,
              inference_scope='conditional_on_known_source_and_collision_closure',
              macro_invalid_repeats=0,pooled_invalid_repeats=0)
    if len(groups)<min_groups or base['largest_group_fraction']>max_group_fraction:
        return dict(**base,status='insufficient_closure_groups',intervals=None,draws=pd.DataFrame(),
                    limitation='closure too few/concentrated; report per-seed descriptions only')
    rng=np.random.default_rng(seed);rows=[]
    for repeat in range(repeats):
        sampled=rng.choice(groups,size=len(groups),replace=True)
        unique,counts=np.unique(sampled,return_counts=True)
        draw_counts={str(g):int(n) for g,n in zip(unique,counts)}
        sample=score_table.copy()
        sample['weight']=sample.closure_id.map(draw_counts).fillna(0.)
        sample=sample[sample.weight>0]
        try:
            metric_rows=[]
            for (grouping,representation,aggregation),part in sample.groupby(['grouping','representation','aggregation'],sort=False):
                r=metric_summary(part,fixed_methods=fixed_methods,allow_macro=grouping=='correct')
                r.update(grouping=grouping,representation=representation,aggregation=aggregation)
                metric_rows.append(r)
            metrics=pd.DataFrame(metric_rows)
        except ValueError as error:
            if 'two-class AUC' not in str(error): raise
            base['pooled_invalid_repeats']+=1
            base['macro_invalid_repeats']+=1
            rows.append(dict(repeat=repeat,pooled_valid=False,macro_valid=False,closure_draw_counts=draw_counts));continue
        macro_valid=bool(metrics.loc[metrics.grouping.eq('correct'),'macro_valid'].all())
        base['macro_invalid_repeats']+=int(not macro_valid)
        rows.append(dict(repeat=repeat,pooled_valid=True,macro_valid=macro_valid,
                         closure_draw_counts=draw_counts,**_comparisons(metrics)))
    draws=pd.DataFrame(rows);valid=draws[draws.pooled_valid]
    intervals=pd.DataFrame([dict(comparison=name,estimate=estimate,
        lower=float(valid[name].dropna().quantile(.025)) if valid[name].notna().any() else float('nan'),
        upper=float(valid[name].dropna().quantile(.975)) if valid[name].notna().any() else float('nan'),
        valid_repeats=int(valid[name].notna().sum()),interval_type='pointwise_conditional_percentile')
        for name,estimate in base['point_estimates'].items()])
    return dict(**base,status='ok' if len(valid) else 'no_valid_draws',intervals=intervals,draws=draws)


def bootstrap_video_metrics(videos, *, threshold=None, repeats=2000, seed=20261023,
                            fixed_methods=None, min_groups=20, max_group_fraction=.5):
    """Normal-condition bootstrap over caller's known source/relationship groups.

    Operates on a correct-grouped video table, independent of the cross-grouping
    closure interaction procedure. Duplicate draws remain duplicate rows.
    """
    q=videos.copy()
    _unique_video_table(q)
    if 'group_id' not in q or q.group_id.isna().any(): raise ValueError('known source/proxy group_id required')
    if fixed_methods is None: fixed_methods=sorted(q.loc[q.label.eq(1),'method_id'].unique())
    sizes=q.groupby('group_id').size();groups=sorted(sizes.index)
    base=dict(point_estimate=metric_summary(q,threshold,fixed_methods),n_source_groups=len(groups),
        largest_group_fraction=float(sizes.max()/sizes.sum()),repeats=repeats,
        macro_invalid_repeats=0,pooled_invalid_repeats=0,
        inference_scope='conditional_on_caller_known_source_groups',min_groups=min_groups,
        max_group_fraction=max_group_fraction)
    if len(groups)<min_groups or base['largest_group_fraction']>max_group_fraction:
        return dict(**base,status='insufficient_source_groups',draws=pd.DataFrame(),intervals=None)
    rng=np.random.default_rng(seed);rows=[]
    for repeat in range(repeats):
        pieces=[]
        for draw_id,group in enumerate(rng.choice(groups,len(groups),replace=True)):
            part=q[q.group_id.eq(group)].copy();part['draw_id']=draw_id;pieces.append(part)
        sample=pd.concat(pieces,ignore_index=True)
        try:
            row=metric_summary(sample,threshold,fixed_methods)
            row.update(repeat=repeat,pooled_valid=True)
            base['macro_invalid_repeats']+=int(not row['macro_valid'])
        except ValueError as error:
            if 'two-class AUC' not in str(error):raise
            row=dict(repeat=repeat,pooled_valid=False,macro_valid=False)
            base['pooled_invalid_repeats']+=1;base['macro_invalid_repeats']+=1
        rows.append(row)
    draws=pd.DataFrame(rows);valid=draws[draws.pooled_valid]
    columns=['pooled_auc','macro_auc']+(['fpr','tpr','pooled_ba','macro_tpr','macro_ba'] if threshold is not None else [])
    intervals=pd.DataFrame([dict(metric=name,estimate=base['point_estimate'][name],
        lower=float(valid[name].quantile(.025)),upper=float(valid[name].quantile(.975)),
        valid_repeats=int(valid[name].notna().sum()),interval_type='pointwise_conditional_percentile') for name in columns]) if len(valid) else None
    return dict(**base,status='ok' if len(valid) else 'no_valid_draws',draws=draws,intervals=intervals)
