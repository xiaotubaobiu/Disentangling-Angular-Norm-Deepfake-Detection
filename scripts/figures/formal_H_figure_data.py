"""Read strict H summaries and retain each contrast's own video cohort."""
from pathlib import Path
import hashlib
import itertools
import json

import numpy as np
import pandas as pd

BACKBONES = ('clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0')
SEEDS = (7, 42, 123)
DOMAINS = ('source_test', 'target_test')
AGGREGATIONS = ('mean_logit', 'mean_probability')
KEYS = ['backbone', 'seed', 'domain', 'cohort', 'aggregation', 'cell', 'threshold_kind']
CONTRASTS = {'fixed_B': ('all_B', 'H10-H00'),
             'interaction': ('common', '(H11-H10)-(H01-H00)')}
COUNTS = {('source_test', 'all_B'): (139, 558), ('source_test', 'common'): (136, 543),
          ('target_test', 'all_B'): (162, 1977), ('target_test', 'common'): (90, 1271)}
INVENTORY = {'identity.json', 'supporting_artifacts.json', 'limitations.json',
    'three_seed_comparisons.csv', 'per_seed_comparisons.csv', 'per_seed_metrics.csv',
    'three_seed_metrics.csv', 'coverage.json'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_tables(metrics, comparisons, official, backbone):
    expected = {(backbone, seed, domain, cohort, aggregation, cell, threshold)
        for seed, domain, aggregation, threshold in itertools.product(SEEDS, DOMAINS,
            AGGREGATIONS, ['default', 'source_real'])
        for cohort, cells in [('all_B', ['H00', 'H10']), ('common', ['H00', 'H01', 'H10', 'H11'])]
        for cell in cells}
    keys = list(metrics[KEYS].itertuples(index=False, name=None))
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError('complete matched three-seed H cells required; no missing or duplicate setting')
    auc = metrics.pooled_auc.to_numpy(float)
    if (not np.isfinite(auc).all() or ((auc < 0) | (auc > 1)).any()
            or not metrics.ranking_usable.eq(True).all()):
        raise ValueError('all three model estimates must be reportable; no silent seed exclusion')
    history = metrics.cell.map({'H00': 'B', 'H01': 'B', 'H10': 'AB', 'H11': 'AB'})
    if not metrics.model_history.eq(history).all():
        raise ValueError('H model history does not match its own cell')
    for (domain, cohort), q in metrics.groupby(['domain', 'cohort'], sort=False):
        nr, nf = COUNTS[domain, cohort]
        if not q.n_real.eq(nr).all() or not q.n_fake.eq(nf).all():
            raise ValueError('fixed all-B/common cohort counts changed')
    if metrics.groupby(KEYS[:-1]).pooled_auc.nunique().ne(1).any():
        raise ValueError('H AUC changed with the threshold kind')
    rows = []
    for seed, domain, aggregation in itertools.product(SEEDS, DOMAINS, AGGREGATIONS):
        part = metrics[metrics.seed.eq(seed) & metrics.domain.eq(domain)
            & metrics.aggregation.eq(aggregation) & metrics.threshold_kind.eq('default')]
        for effect, (cohort, comparison) in CONTRASTS.items():
            cells = part[part.cohort.eq(cohort)].set_index('cell').pooled_auc
            estimate = (cells.H10 - cells.H00 if effect == 'fixed_B'
                else (cells.H11 - cells.H10) - (cells.H01 - cells.H00))
            saved = comparisons[comparisons.backbone.eq(backbone) & comparisons.seed.eq(seed)
                & comparisons.domain.eq(domain) & comparisons.cohort.eq(cohort)
                & comparisons.aggregation.eq(aggregation) & comparisons.comparison.eq(comparison)
                & comparisons.threshold_kind.eq('default') & comparisons.metric.eq('pooled_auc')]
            if (len(saved) != 1 or not np.isfinite(saved.estimate.iloc[0])
                    or not np.isclose(saved.estimate.iloc[0], estimate, rtol=0, atol=1e-12)):
                raise ValueError('saved H comparison does not reconstruct from its same-cohort cells')
            rows.append(dict(backbone=backbone, seed=seed, domain=domain, aggregation=aggregation,
                cohort=cohort, comparison=comparison, effect=effect, estimate=float(estimate)))
    paired = pd.DataFrame(rows)
    for (domain, aggregation, effect), q in paired.groupby(['domain', 'aggregation', 'effect'], sort=False):
        cohort, comparison = CONTRASTS[effect]
        saved = official[official.backbone.eq(backbone) & official.domain.eq(domain)
            & official.cohort.eq(cohort) & official.aggregation.eq(aggregation)
            & official.comparison.eq(comparison) & official.threshold_kind.eq('default')
            & official.metric.eq('pooled_auc')]
        if len(saved) != 1:
            raise ValueError('one official three-seed comparison summary required')
        row = saved.iloc[0]
        if (not row.complete_three_seeds or row.n_valid_seeds != 3
                or not np.isclose(row.estimate_mean, q.estimate.mean(), rtol=0, atol=1e-12)
                or not np.isclose(row.estimate_sample_sd, q.estimate.std(ddof=1), rtol=0, atol=1e-12)):
            raise ValueError('official H mean/SD does not match three paired differences')
    return paired


def load_verified(root, backbones=BACKBONES):
    """Verify committed summaries; their strict workers own live input validation."""
    root = Path(root).resolve()
    backbones = tuple(backbones)
    if not backbones or len(set(backbones)) != len(backbones) or not set(backbones) <= set(BACKBONES):
        raise ValueError('registered nonduplicate H backbones required')
    for backbone in backbones:
        if not (root / 'analysis/H' / backbone / 'three_seed_summary/complete.json').is_file():
            raise ValueError('completed strict three-seed summary required: ' + backbone)
    sources = {}
    for name in ['data/H/freeze.json', 'data/H/train_dev_freeze.json']:
        sources[name] = digest(root / name)
    metrics, comparisons, summaries, paired = [], [], [], []
    for backbone in backbones:
        folder = root / 'analysis/H' / backbone / 'three_seed_summary'
        marker = folder / 'complete.json'
        snapshot = digest(marker)
        record = json.loads(marker.read_text())
        identity = record.get('identity', {})
        if (record.get('schema') != 'JOURNAL_H_THREE_SEED_SUMMARY_COMPLETE_V1'
                or record.get('status') not in ['SUMMARY_COMPLETE', 'SUMMARY_COMPLETE_WITH_NUMERIC_NA']
                or record.get('backbone') != backbone or record.get('training_seed_pairs') != 3
                or record.get('target_development_used') is not False
                or len(identity.get('seeds', [])) != 3 or set(identity.get('seeds', [])) != set(SEEDS)
                or set(record.get('artifact_sha256', {})) != INVENTORY):
            raise ValueError('complete strict three-seed summary identity/inventory required')
        for name, expected in record['artifact_sha256'].items():
            path = folder / name
            if digest(path) != expected:
                raise ValueError('strict H summary artifact hash changed')
            sources[str(path.relative_to(root))] = expected
        if json.loads((folder / 'identity.json').read_text()) != identity:
            raise ValueError('strict H summary identity changed')
        for seed in SEEDS:
            path = root / 'analysis/H' / backbone / f'seed{seed}/complete.json'
            expected = identity.get('pair_complete_sha256', {}).get(str(seed))
            if digest(path) != expected:
                raise ValueError('H source pair analysis changed after strict summary')
            source = json.loads(path.read_text())
            if (source.get('backbone') != backbone or source.get('seed') != seed
                    or source.get('formal_trajectory_target_used') is not False
                    or source.get('target_development_used') is not False
                    or source.get('independent_recompute') != 'PASSED_FOR_ALL_REPORTABLE_CELLS'):
                raise ValueError('H source pair formal identity changed')
            contexts = source['identity']['cache_contexts']
            for history in ['B', 'AB']:
                context = contexts[history]
                if (context['run_id'] != f'H_formal_{backbone}_{history}_seed{seed}'
                        or context['full_data_freeze_sha256'] != sources['data/H/freeze.json']
                        or context['source_data_freeze_sha256'] != sources['data/H/train_dev_freeze.json']
                        or context['best_epoch'] != source['paired_epoch']
                        or context['formal_trajectory_target_used'] is not False
                        or context['target_development_used'] is not False):
                    raise ValueError('H paired cache/model/data identity changed')
            sources[str(path.relative_to(root))] = expected
        m = pd.read_csv(folder / 'per_seed_metrics.csv', float_precision='round_trip')
        c = pd.read_csv(folder / 'per_seed_comparisons.csv', float_precision='round_trip')
        s = pd.read_csv(folder / 'three_seed_comparisons.csv', float_precision='round_trip')
        paired.append(validate_tables(m, c, s, backbone))
        metrics.append(m); comparisons.append(c); summaries.append(s)
        if digest(marker) != snapshot:
            raise ValueError('strict H summary changed during figure read')
        sources[str(marker.relative_to(root))] = snapshot
    if any(digest(root / name) != expected for name, expected in sources.items()):
        raise ValueError('H figure input bytes changed during read')
    return dict(metrics=pd.concat(metrics, ignore_index=True), comparisons=pd.concat(comparisons, ignore_index=True),
        official_comparisons=pd.concat(summaries, ignore_index=True), paired=pd.concat(paired, ignore_index=True),
        sources=sources, backbones=list(backbones), training_seeds=list(SEEDS))
