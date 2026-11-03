"""Check the manuscript against independent reductions of accepted seed rows."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--old-root', type=Path, required=True)
    args = p.parse_args()
    root, old = args.root.resolve(), args.old_root.resolve()
    code = Path(__file__).resolve().parents[3]
    paper = code / 'paper/ieee-access'
    folder = root / 'reports/final_formal_33'
    docs = root / 'reports/manuscript_revision'
    zh = (docs / '中文稿.md').read_text()
    en = (docs / 'manuscript_en.qmd').read_text()
    assert sha(docs / 'manuscript_en.qmd') == sha(paper / 'manuscript_en.qmd')
    # Obsidian and Git use different image roots; all prose and numeric values
    # must remain identical after removing only the Markdown image targets.
    normalize_images = lambda text: re.sub(r'(!\[[^\]]*\])\([^)]+\)', r'\1(IMAGE)', text)
    assert normalize_images(zh) == normalize_images((paper / 'manuscript_zh.md').read_text())
    checked = []

    def reduce_and_find(values, label, in_zh=True, in_en=True, signed=False):
        values = np.asarray(values, float) * 100
        assert np.isfinite(values).all()
        mean, sd = float(np.mean(values)), float(np.std(values, ddof=1))
        first = f'{mean:+.4f}' if signed else f'{mean:.4f}'
        if in_zh:
            assert first + f'±{sd:.4f}' in zh, (label, 'Chinese', first, sd)
        if in_en:
            # A positive effect may be written as a gain without an explicit
            # plus sign; negative signs are always required.
            starts = [first, f'{mean:.4f}'] if signed and mean >= 0 else [first]
            assert any('$' + x + f'\\pm{sd:.4f}$' in en for x in starts), (label, 'English', first, sd)
        checked.append(dict(claim=label, n_seeds=len(values), mean_percent_or_pp=mean,
                            sample_sd_percent_or_pp=sd, chinese=in_zh, english=in_en))

    h = pd.read_csv(folder / 'H_per_seed_metrics.csv', float_precision='round_trip')
    c = pd.read_csv(folder / 'H_per_seed_comparisons.csv', float_precision='round_trip')
    assert len(h) == 432 and len(c) == 3024
    for (bone, domain, cell), q in h[h.cohort.eq('all_B') & h.threshold_kind.eq('default') & h.aggregation.eq('mean_logit')].groupby(['backbone','domain','cell']):
        assert len(q) == 3 and set(q.seed) == {7,42,123}
        reduce_and_find(q.pooled_auc, f'H all-B {bone}/{domain}/{cell}')
    for (bone, domain), q in c[c.cohort.eq('all_B') & c.comparison.eq('H10-H00')
            & c.threshold_kind.eq('default') & c.aggregation.eq('mean_logit') & c.metric.eq('pooled_auc')].groupby(['backbone','domain']):
        assert len(q) == 3 and set(q.seed) == {7,42,123}
        reduce_and_find(q.estimate, f'H all-B paired {bone}/{domain}', signed=True)
    for (bone, cell), q in h[h.domain.eq('target_test') & h.cohort.eq('common') & h.threshold_kind.eq('default')
            & h.aggregation.eq('mean_logit')].groupby(['backbone','cell']):
        reduce_and_find(q.pooled_auc, f'H common four-cell {bone}/{cell}', in_en=False)
    for (bone, agg, term), q in c[c.domain.eq('target_test') & c.cohort.eq('common') & c.threshold_kind.eq('default')
            & c.metric.eq('pooled_auc') & c.comparison.isin(['H01-H00','H11-H10','(H11-H10)-(H01-H00)'])].groupby(['backbone','aggregation','comparison']):
        if agg == 'mean_probability' and term != '(H11-H10)-(H01-H00)':
            continue
        # The English main text keeps the interaction; the Chinese main text
        # also preserves all three input-switch contrasts.
        in_en = (term == '(H11-H10)-(H01-H00)' and agg == 'mean_logit') or (bone != 'dinov2_vit_b14' and agg == 'mean_logit')
        reduce_and_find(q.estimate, f'H input effect {bone}/{agg}/{term}', in_en=in_en, signed=True)
    d = pd.read_csv(folder / 'D_per_seed_metrics.csv', float_precision='round_trip')
    for key, q in d[d.domain.eq('target_full')].groupby(['backbone','grouping','representation','aggregation']):
        assert len(q) == 5 and set(q.seed) == {7,42,123,2026,3407}
        reduce_and_find(q.pooled_auc, 'D target AUC ' + '/'.join(key))
    cal = pd.read_csv(folder / 'D_per_seed_calibration.csv', float_precision='round_trip')
    cal = cal[cal.domain.eq('target_test') & cal.representation.eq('raw') & cal.aggregation.eq('mean_logit')
              & (cal.comparison.eq('source_real') | (cal.comparison.eq('target_real') & cal.budget.eq(20)))
              & cal.metric.isin(['pooled_auc','fpr','tpr','pooled_ba'])]
    for key, q in cal.groupby(['backbone','comparison','metric']):
        assert len(q) == 5 and set(q.seed) == {7,42,123,2026,3407}
        reduce_and_find(q.estimate, 'D operating point ' + '/'.join(key))
    # Every D table remains the exact original accepted byte sequence.
    table_receipt = json.loads((folder / 'complete.json').read_text())
    for name, expected in table_receipt['output_sha256'].items():
        assert sha(folder / name) == expected
        if name.startswith('D_'):
            assert sha(folder / name) == sha(old / 'reports/final_formal_33' / name)
    previous = (root / 'operations/obsidian_before_H10/中文稿.md').read_text()
    original = previous[previous.index('### 3.3'):previous.index('## 4. 讨论')]
    current = zh[zh.index('### 3.3'):zh.index('## 4. 讨论')]
    assert current.replace('**图4｜','**图3｜').replace('**图5｜','**图4｜') == original
    for bad in ['63.0385','65.3210','79.4973','79.4814','2.2825±2.8863','@@','(author?)','LAYOUT PREFLIGHT']:
        assert bad not in zh and bad not in en, bad
    selection = pd.read_csv(folder / 'formal_model_selection.csv', float_precision='round_trip')
    revised = selection[selection.route.eq('H') & selection.backbone.isin(['dinov2_vit_b14','efficientnet_b0'])]
    assert len(revised) == 12 and revised.max_epochs.eq(10).all() and revised.actual_epochs.eq(10).all()
    for (bone, seed), q in revised.groupby(['backbone','seed']):
        assert set(q.history) == {'B','AB'} and q.best_epoch.nunique() == 1 and 1 <= q.best_epoch.iloc[0] <= 10
    # Recompute the retained, specified cross-backbone example from actual
    # revised model scores; no old fifteen-epoch witness may be carried over.
    w = h[h.domain.eq('target_test') & h.cohort.eq('common') & h.aggregation.eq('mean_logit')
          & h.threshold_kind.eq('default') & h.seed.eq(123) & h.cell.isin(['H10','H11'])
          & h.backbone.isin(['clip_vit_b16','efficientnet_b0'])]
    assert len(w) == 4
    for value in w.pooled_auc:
        assert f'{100*value:.4f}' in zh and f'{100*value:.4f}' in en
    pivot = w.pivot(index='backbone', columns='cell', values='pooled_auc')
    for cell in ['H10','H11']:
        gap = 100*(pivot.loc['clip_vit_b16',cell]-pivot.loc['efficientnet_b0',cell])
        assert f'{gap:+.4f}' in zh and f'{gap:+.4f}' in en
    family = h[h.domain.eq('target_test') & h.cohort.eq('common') & h.aggregation.eq('mean_logit')
          & h.threshold_kind.eq('default') & h.cell.isin(['H10','H11'])
          & h.backbone.isin(['clip_vit_b16','efficientnet_b0'])]
    family = family.pivot(index=['seed','cell'], columns='backbone', values='pooled_auc')
    gaps = (family.clip_vit_b16 - family.efficientnet_b0).unstack('cell')
    assert len(gaps) == 3 and set(gaps.index) == {7,42,123}
    flips = int((gaps.H10 * gaps.H11 < 0).sum())
    assert f'{flips} strict point-estimate reversals among three seeds' in en
    assert f'三个种子中有{flips}个严格点估计反转' in zh
    cited = {key for block in re.findall(r'\[([^\[\]]*@[A-Za-z][^\[\]]*)\]', en)
             for key in re.findall(r'(?<!@)@([A-Za-z][A-Za-z0-9_]*)', block)}
    bibliographic = set(re.findall(r'^@\w+\{([^,]+),', (paper / 'references.bib').read_text(), re.M))
    assert cited == bibliographic and len(cited) == 12, (cited,bibliographic)
    assert 'name: Shiyu Chen' in en and 'name: Kangbin Yim' in en
    assert en.count('name: ') == 2
    assert 'This is a revision after initial target results were available' in en
    assert 'originally target-blind budget choice' in en
    assert '基金' in (docs / '实验计划.md').read_text()
    for figure in re.findall(r'\]\((figures/[^)]+)\)', en):
        assert (paper / figure).is_file(), figure
    result = dict(status='PASS_INDEPENDENT_SEED_REDUCTIONS_AND_MANUSCRIPT_SCOPE',
                  claims_checked=len(checked), claims=checked, D_tables_byte_identical=True,
                  Chinese_D_results_unchanged_except_caption_numbers=True,
                  revised_models=12, fixed_budget=10, reference_keys_verified=sorted(cited),
                  inputs={name:sha(folder / name) for name in table_receipt['output_sha256']},
                  documents={p.name:sha(p) for p in docs.glob('*') if p.suffix in ['.md','.qmd']},
                  producer_sha256=sha(__file__))
    out = root / 'operations/manuscript_numeric_acceptance.json'
    assert not out.exists()
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['claims','inputs','documents']},indent=2))


if __name__ == '__main__':
    main()
