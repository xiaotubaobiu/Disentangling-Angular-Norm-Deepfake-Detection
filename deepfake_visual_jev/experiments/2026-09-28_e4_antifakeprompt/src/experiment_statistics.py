"""Single-seed paired physical-video statistics; no synthetic seed replication."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss, log_loss

from evaluation.e2_gates import _ClusterMetric, PRIMARY
from evaluation.semantic_eval import _task_values, _ece, GROUP_KEYS


def detection_metrics(rows):
    labels = rows.label.to_numpy(dtype=int)
    scores = rows.prob_fake.to_numpy(dtype=float)
    if set(labels) != {0, 1} or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('binary metrics need finite probabilities and both classes')
    pred = scores >= .5
    return {'auroc': float(roc_auc_score(labels, scores)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, pred)),
        'accuracy': float((pred == labels).mean()),
        'real_recall': float((~pred[labels == 0]).mean()),
        'fake_recall': float(pred[labels == 1].mean()),
        'brier': float(brier_score_loss(labels, scores)), 'ece': float(_ece(labels, scores)),
        'nll': float(log_loss(labels, np.clip(scores, 1e-7, 1 - 1e-7))),
        'physical_videos': int(rows.video_key.nunique()), 'video_variant_rows': len(rows)}


def task_metric(rows, task, *, ba=False):
    if ba and task in ('T1', 'T3'):
        return float(balanced_accuracy_score(rows.label, rows.prob_fake >= .5))
    return float(_task_values(task, rows)[PRIMARY[task]])


def paired_bootstrap(baseline, candidate, metric, *, n_resamples=10000, seed=20260928, lower_quantile=.05):
    if n_resamples < 1 or lower_quantile not in (.025, .05):
        raise ValueError('invalid bootstrap settings')
    baseline, candidate = baseline.copy(), candidate.copy()
    keys = sorted(set(baseline.video_key))
    if keys != sorted(set(candidate.video_key)):
        raise ValueError('paired physical video sets differ')
    for col in ('seed', 'domain'):
        if set(baseline[col]) != set(candidate[col]) or len(set(baseline[col])) != 1:
            raise ValueError(f'paired {col} mismatch; this analysis is single-seed only')
    if set(baseline.seed) != {42}:
        raise ValueError('this protocol requires seed42')
    pair_keys = (['video_key', 'label', 'method_id', 'video_id'] if metric == 'auroc'
                 else [key for key in GROUP_KEYS if key not in ('seed', 'domain', 'condition')])
    for frame in (baseline, candidate):
        if any(key not in frame for key in pair_keys) or frame[pair_keys].isna().any().any():
            raise ValueError('missing paired video/variant identity')
        if frame.duplicated(pair_keys).any():
            raise ValueError('duplicate paired video/variant rows')
    baseline = baseline.sort_values(pair_keys).reset_index(drop=True)
    candidate = candidate.sort_values(pair_keys).reset_index(drop=True)
    if not baseline[pair_keys].astype(str).equals(candidate[pair_keys].astype(str)):
        raise ValueError('paired video/variant identity mismatch')
    if 'num_frames' not in baseline or 'num_frames' not in candidate:
        raise ValueError('missing paired frame counts')
    counts = baseline.num_frames.to_numpy(dtype=float)
    if (not np.array_equal(counts, candidate.num_frames.to_numpy(dtype=float))
            or not np.isfinite(counts).all() or (counts <= 0).any() or (counts != np.floor(counts)).any()):
        raise ValueError('paired frame counts differ or are invalid')
    for frame in (baseline, candidate):
        if frame.groupby('video_key').label.nunique().gt(1).any():
            raise ValueError('inconsistent physical labels')
    left_truth = baseline.drop_duplicates('video_key').set_index('video_key').loc[keys]
    right_truth = candidate.drop_duplicates('video_key').set_index('video_key').loc[keys]
    if not left_truth[['label', 'method_id']].equals(right_truth[['label', 'method_id']]):
        raise ValueError('paired physical labels/methods differ')
    index = {key: i for i, key in enumerate(keys)}
    left = _ClusterMetric(baseline, metric, baseline.video_key.map(index).to_numpy(), len(keys))
    right = _ClusterMetric(candidate, metric, candidate.video_key.map(index).to_numpy(), len(keys))
    one = np.ones((1, len(keys)))
    base_point, candidate_point = float(left(one)[0]), float(right(one)[0])
    strata = [np.asarray(group, dtype=int) for group in
              left_truth.reset_index().groupby(['label', 'method_id']).indices.values()]
    rng = np.random.default_rng(seed)
    values = np.empty(n_resamples)
    chunk = max(1, min(64, 1_000_000 // max(len(baseline), len(candidate))))
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        weights = np.zeros((size, len(keys)))
        for group in strata:
            weights[:, group] = rng.multinomial(len(group), np.full(len(group), 1 / len(group)), size=size)
        values[start:start + size] = right(weights) - left(weights)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite bootstrap replicates')
    return {'point': candidate_point - base_point,
            'lower': float(np.quantile(values, lower_quantile)), 'upper': float(np.quantile(values, .975)),
            'lower_quantile': lower_quantile, 'n_resamples': n_resamples, 'n_videos': len(keys),
            'baseline': base_point, 'candidate': candidate_point,
            'baseline_variant_rows': len(baseline), 'candidate_variant_rows': len(candidate),
            'scope': 'conditional_on_seed42; label/method-stratified physical-video resampling'}
