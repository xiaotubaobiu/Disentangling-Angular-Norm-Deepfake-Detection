"""Read complete formal D summaries without inference or changing score rules."""
from pathlib import Path
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

SEEDS = (7, 42, 123, 2026, 3407)
BACKBONES = ('clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0')
DOMAINS = ('source_test', 'target_full')
AGGREGATIONS = ('mean_logit', 'mean_probability')
CELL_KEYS = ['backbone', 'seed', 'domain', 'grouping', 'representation', 'aggregation']
EFFECT_KEYS = ['backbone', 'seed', 'domain', 'aggregation']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _matched(frame, columns, expected):
    if not set(columns) <= set(frame):
        raise ValueError('complete matched five-seed cells required')
    keys = list(frame[columns].itertuples(index=False, name=None))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('complete matched five-seed cells required; missing or duplicate setting')


def _validate_tables(metrics, effects, backbone):
    expected = set(itertools.product([backbone], SEEDS, DOMAINS, ['correct', 'collision'],
        ['raw', 'l2'], AGGREGATIONS))
    _matched(metrics, CELL_KEYS, expected)
    _matched(effects, EFFECT_KEYS, set(itertools.product([backbone], SEEDS, DOMAINS, AGGREGATIONS)))
    values = metrics.pooled_auc.to_numpy(float)
    if (not np.isfinite(values).all() or ((values < 0) | (values > 1)).any()
            or not metrics.ranking_usable.eq(True).all()):
        raise ValueError('all five matched AUC estimates must be reportable; no silent seed exclusion')
    for domain, q in metrics.groupby('domain', sort=False):
        nr, nf, nc = (140, 2884, 280) if domain == 'source_test' else (157, 12037, 1830)
        if (not q.n_real.eq(nr).all() or not q.loc[q.grouping.eq('correct'), 'n_fake'].eq(nf).all()
                or not q.loc[q.grouping.eq('collision'), 'n_fake'].eq(nc).all()):
            raise ValueError('fixed natural-video or collision-group counts changed')
    saved = effects.set_index(EFFECT_KEYS)
    for tag, q in metrics.groupby(EFFECT_KEYS, sort=False):
        cells = q.set_index(['grouping', 'representation']).pooled_auc
        correct = cells['correct', 'l2'] - cells['correct', 'raw']
        collision = cells['collision', 'l2'] - cells['collision', 'raw']
        expected_values = [correct, collision, collision-correct]
        actual = saved.loc[tag, ['delta_correct', 'delta_collision', 'interaction']].to_numpy(float)
        if not np.isfinite(actual).all() or not np.allclose(actual, expected_values, rtol=0, atol=1e-12):
            raise ValueError('saved effects do not reconstruct from their eight AUC cells')


def load_verified(root, backbones=BACKBONES):
    """Bind plots to strict summaries and the five original analysis endpoints.

    The strict summary owns live raw-image/cache validation. This reader verifies
    its committed bytes and source analysis receipts; it does not replace that
    original acceptance with a marker count or a partial-seed mean.
    """
    root = Path(root).resolve()
    backbones = tuple(backbones)
    if not backbones or len(set(backbones)) != len(backbones) or not set(backbones) <= set(BACKBONES):
        raise ValueError('registered nonduplicate backbones required')
    sources, tables = {}, []
    for backbone in backbones:
        folder = root/'analysis/D'/backbone/'five_seed_summary'
        marker = folder/'complete.json'
        if not marker.is_file():
            raise ValueError('completed strict five-seed summary required: ' + backbone)
        snapshot = digest(marker)
        record = json.loads(marker.read_text())
        if (record.get('schema') != 'D_FIVE_SEED_SUMMARY_COMPLETE_V1'
                or record.get('status') not in ['SUMMARY_COMPLETE', 'SUMMARY_COMPLETE_WITH_NUMERIC_NA']
                or record.get('backbone') != backbone):
            raise ValueError('completed strict five-seed summary required: ' + backbone)
        identity = record.get('identity', {})
        seeds = identity.get('seeds', [])
        if record.get('training_seed_count') != 5 or len(seeds) != 5 or set(seeds) != set(SEEDS):
            raise ValueError('all five registered training seeds required')
        hashes = record.get('artifact_sha256', {})
        if not {'per_seed_metrics.csv', 'per_seed_effects.csv'} <= set(hashes):
            raise ValueError('strict summary tables missing')
        for name, expected in hashes.items():
            path = folder/name
            if not path.resolve().is_relative_to(folder.resolve()) or not path.is_file() or digest(path) != expected:
                raise ValueError('strict summary artifact hash changed')
            sources[str(path.relative_to(root))] = expected
        for seed in SEEDS:
            path = root/'analysis/D'/f'D_formal_{backbone}_seed{seed}'/'complete.json'
            if not path.is_file() or digest(path) != identity.get('analysis_complete_sha256', {}).get(str(seed)):
                raise ValueError('source model analysis changed after strict summary')
            sources[str(path.relative_to(root))] = digest(path)
        metrics = pd.read_csv(folder/'per_seed_metrics.csv', float_precision='round_trip')
        effects = pd.read_csv(folder/'per_seed_effects.csv', float_precision='round_trip')
        _validate_tables(metrics, effects, backbone)
        if digest(marker) != snapshot or any(digest(root/name) != expected for name, expected in sources.items()):
            raise ValueError('summary inputs changed during figure read')
        sources[str(marker.relative_to(root))] = snapshot
        tables.append((metrics, effects))
    return dict(metrics=pd.concat([pair[0] for pair in tables], ignore_index=True),
        effects=pd.concat([pair[1] for pair in tables], ignore_index=True), sources=sources,
        backbones=list(backbones), training_seeds=list(SEEDS))


def figure_tables(metrics, effects, domain):
    if domain not in DOMAINS:
        raise ValueError('registered TEST domain required')
    absolute = metrics.loc[metrics.domain.eq(domain)].copy()
    absolute['auc_percent'] = absolute.pooled_auc*100
    paired = effects.loc[effects.domain.eq(domain)].melt(id_vars=EFFECT_KEYS,
        value_vars=['delta_correct', 'interaction'], var_name='effect', value_name='estimate')
    paired['estimate_pp'] = paired.estimate*100
    return absolute, paired


def paired_seed_summary(frame):
    groups = [name for name in ['backbone', 'domain', 'aggregation', 'effect'] if name in frame]
    result = []
    for values, q in frame.groupby(groups, sort=False):
        if len(q) != 5 or set(q.seed) != set(SEEDS) or q.seed.duplicated().any():
            raise ValueError('all five registered paired training seeds required')
        x = q.estimate_pp.to_numpy(float)
        if not np.isfinite(x).all():
            raise ValueError('all paired estimates required; no silent seed exclusion')
        result.append(dict(zip(groups, values), n_training_seeds=5,
            estimate_mean_pp=float(x.mean()), estimate_sample_sd_pp=float(x.std(ddof=1))))
    return pd.DataFrame(result)
