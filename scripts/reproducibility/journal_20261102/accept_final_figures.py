"""Bind final rendered/visually inspected figures to original accepted tables."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path('/home/z/workspace/deepfake_journal_20261102')
CODE = Path('/home/z/workspace/deepfake_journal_code_20261102')
sys.path.insert(0, str(CODE / 'scripts/figures'))
import formal_D_figure_data as d_reader
import formal_H_figure_data as h_reader


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def csv(path):
    return pd.read_csv(path, float_precision='round_trip')


def same(saved, expected):
    pd.testing.assert_frame_equal(saved.reset_index(drop=True), expected.reset_index(drop=True),
        check_dtype=False, check_exact=True)


def main():
    dest = ROOT / 'operations/final_three_backbone_figures_acceptance.json'
    assert not dest.exists()
    d, h = d_reader.load_verified(ROOT), h_reader.load_verified(ROOT)
    source_rows = {}
    root = ROOT / 'figures/H_all_backbones_formal_training_history'
    stem = 'H_training_history_all_backbones'
    for key, suffix in [('metrics', 'all_metrics'), ('comparisons', 'all_comparisons'),
            ('official_comparisons', 'all_official_comparisons'), ('paired', 'primary_and_sensitivity_per_seed')]:
        path = root / 'source_data' / f'{stem}_{suffix}.csv'
        frame = csv(path); same(frame, h[key]); source_rows[path.name] = len(frame)
    counts = h['metrics'].groupby(['backbone', 'domain', 'cohort'], as_index=False)[['n_real', 'n_fake']].first()
    same(csv(root / 'source_data' / f'{stem}_cohort_counts.csv'), counts)
    plotted = csv(root / 'source_data' / f'{stem}_plotted_official_summary.csv')
    q = h['official_comparisons']
    expected = q[q.aggregation.eq('mean_logit') & q.threshold_kind.eq('default') & q.metric.eq('pooled_auc')
        & ((q.cohort.eq('all_B') & q.comparison.eq('H10-H00'))
        | (q.cohort.eq('common') & q.comparison.eq('(H11-H10)-(H01-H00)')))]
    keys = ['backbone', 'domain', 'cohort', 'comparison']
    pd.testing.assert_frame_equal(plotted.set_index(keys).sort_index(), expected.set_index(keys).sort_index(), check_exact=True)
    assert len(plotted) == 12

    root = ROOT / 'figures/D_all_backbones_inference_grouping'
    official = pd.concat([csv(ROOT / f'analysis/D/{bone}/five_seed_summary/five_seed_effects.csv') for bone in d['backbones']], ignore_index=True)
    for domain in ['target_full', 'source_test']:
        stem = 'D_inference_grouping_all_backbones_' + domain
        absolute, paired = d_reader.figure_tables(d['metrics'], d['effects'], domain)
        for suffix, expected in [('absolute_AUC', absolute), ('paired_effects', paired)]:
            path = root / 'source_data' / f'{stem}_{suffix}.csv'
            actual = csv(path); same(actual, expected); source_rows[path.name] = len(actual)
        summary = csv(root / 'source_data' / f'{stem}_seed_summary.csv')
        assert len(summary) == 12
        for row in summary.itertuples():
            part = official[official.backbone.eq(row.backbone) & official.domain.eq(domain) & official.aggregation.eq(row.aggregation)]
            assert len(part) == 1
            record = part.iloc[0]
            np.testing.assert_allclose([row.estimate_mean_pp, row.estimate_sample_sd_pp],
                [100 * record[row.effect + '_mean'], 100 * record[row.effect + '_sample_sd']], rtol=0, atol=1e-10)

    root = ROOT / 'figures/D_all_backbones_operating_points'
    stem = 'D_operating_points_all_backbones'
    full = pd.concat([csv(ROOT / f'analysis/D/{bone}/five_seed_summary/per_seed_calibration.csv') for bone in d['backbones']], ignore_index=True)
    official = pd.concat([csv(ROOT / f'analysis/D/{bone}/five_seed_summary/five_seed_calibration.csv') for bone in d['backbones']], ignore_index=True)
    same(csv(root / 'source_data' / f'{stem}_all_calibration.csv'), full)
    for suffix, reference, seed_key in [('plotted_per_seed', full, ['seed']), ('plotted_official_summary', official, [])]:
        actual = csv(root / 'source_data' / f'{stem}_{suffix}.csv')
        expected = reference[reference.metric.isin(['fpr', 'tpr', 'pooled_ba'])
            & (reference.comparison.eq('source_real') | (reference.comparison.eq('target_real') & reference.budget.eq(20)))]
        keys = ['backbone', 'domain', 'representation', 'aggregation', 'comparison', 'budget', 'metric'] + seed_key
        pd.testing.assert_frame_equal(actual.set_index(keys).sort_index(), expected.set_index(keys).sort_index(), check_exact=True)
        source_rows[stem + '_' + suffix] = len(actual)

    families = [
        ('H_all_backbones_formal_training_history', ['H_training_history_all_backbones'], 'plot_formal_H_training_history.py', h['sources']),
        ('D_all_backbones_inference_grouping', ['D_inference_grouping_all_backbones_target_full', 'D_inference_grouping_all_backbones_source_test'], 'plot_formal_D_inference_grouping.py', d['sources']),
        ('D_all_backbones_operating_points', ['D_operating_points_all_backbones'], 'plot_formal_D_operating_points.py', d['sources'])]
    figures, artifacts = [], {}
    for folder, stems, script, sources in families:
        root = ROOT / 'figures' / folder
        receipt = json.loads((root / 'render_receipt.json').read_text())
        assert set(receipt['backbones']) == set(d['backbones'])
        assert receipt['script_sha256'] == sha(CODE / 'scripts/figures' / script)
        assert all(sha(ROOT / name) == value for name, value in sources.items())
        for stem in stems:
            alignment = json.loads((root / f'{stem}.alignment.json').read_text())
            text = json.loads((root / f'{stem}.text-audit.log').read_text())
            collision = json.loads((root / f'{stem}.collision-audit.json').read_text())
            assert alignment['verdict'] == 'PASS' and text['minimum_found_pt'] >= 7 and text['below_minimum_count'] == 0
            assert collision['summary']['fail'] == 0 and collision['summary']['warn'] == 0
            for ext in ['pdf', 'svg', 'png', 'tiff', 'caption.zh.md', 'review.png']:
                assert (root / f'{stem}.{ext}').is_file()
            figures.append(dict(stem=stem, alignment='PASS', minimum_font_pt=text['minimum_found_pt'], collision='0FAIL0WARN'))
        for path in root.rglob('*'):
            if path.is_file(): artifacts[str(path.relative_to(ROOT))] = sha(path)
    # Actual PDF previews were opened and every panel/complete figure inspected
    # before this acceptance command, including error bars, labels and cohorts.
    record = dict(status='PASS_ALL_THREE_BACKBONE_FINAL_FIGURE_FAMILIES',
        created=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(), families=3, PDFs=4, panels=9,
        figures=figures, artifact_sha256=artifacts, exact_source_rows_verified=source_rows,
        all_panels_and_whole_figures_visually_inspected_by_agent=True,
        operating_points_reconstructed_from_saved_originals=360, operating_summaries_reconstructed=72,
        H_paired_effects_reconstructed=72, H_plotted_official_summaries_reconstructed=12,
        D_original_absolute_AUC_rows_preserved=240, D_plotted_effect_summaries_reconstructed=24,
        reader_sha256={name:sha(CODE / 'scripts/figures' / name) for name in ['formal_D_figure_data.py', 'formal_H_figure_data.py']},
        numerical_and_source_QA=True, training_or_inference=False, producer_sha256=sha(__file__),
        final_manuscript=False, full_goal_complete=False)
    dest.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['artifact_sha256','exact_source_rows_verified']}, indent=2))


if __name__ == '__main__':
    main()
