"""Plot saved paired H effects with complete three-seed strict summaries."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from formal_H_figure_data import BACKBONES, CONTRASTS, SEEDS, digest, load_verified, load_verified_roots

DISPLAY = {'clip_vit_b16': 'CLIP', 'dinov2_vit_b14': 'DINOv2', 'efficientnet_b0': 'EfficientNet-B0'}
DOMAINS = [('source_test', '#4E79A7', 'FF++'), ('target_test', '#CC8F5B', 'DFD')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--roots-map', type=Path, help='JSON with roots and expected_epoch_budgets for an explicit revision')
    parser.add_argument('--qa-dir', type=Path, required=True)
    parser.add_argument('--backbone', choices=BACKBONES)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    backbones = [args.backbone] if args.backbone else BACKBONES
    if args.roots_map:
        mapping = json.loads(args.roots_map.read_text())
        data = load_verified_roots({b: mapping['roots'][b] for b in backbones}, backbones,
            {b: mapping['expected_epoch_budgets'][b] for b in backbones})
    else:
        data = load_verified(args.root, backbones)
    sys.path.insert(0, str(args.qa_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.linewidth': .7, 'legend.frameon': False, 'savefig.facecolor': 'white'})
    output = args.output_dir or args.root / 'figures/formal_H_training_history'
    source = output / 'source_data'
    source.mkdir(parents=True, exist_ok=True)
    tag = 'all_backbones' if len(data['backbones']) == 3 else data['backbones'][0]
    stem = f'H_training_history_{tag}'
    # Preserve every original estimate and threshold kind, including sensitivity
    # analyses; the displayed primary contrasts use Raw mean-logit only.
    for key, suffix in [('metrics', 'all_metrics'), ('comparisons', 'all_comparisons'),
            ('official_comparisons', 'all_official_comparisons'), ('paired', 'primary_and_sensitivity_per_seed')]:
        data[key].to_csv(source / f'{stem}_{suffix}.csv', index=False, float_format='%.17g')
    counts = data['metrics'].groupby(['backbone', 'domain', 'cohort'], as_index=False)[['n_real', 'n_fake']].first()
    counts.to_csv(source / f'{stem}_cohort_counts.csv', index=False)
    paired = data['paired'][data['paired'].aggregation.eq('mean_logit')]
    official = data['official_comparisons']
    height = 3.2 if len(data['backbones']) == 1 else 4.2
    fig, axes = plt.subplots(1, 2, figsize=(7.204724409448819, height), sharey=True)
    fig.subplots_adjust(left=.205, right=.985, bottom=.85 / height, top=1 - .8 / height, wspace=.30)
    selected = []
    for panel, (effect, title) in enumerate([('fixed_B', 'Fixed-B deployment'),
            ('interaction', 'Training–test history interaction')]):
        ax = axes[panel]
        bounds = [0.]
        cohort, comparison = CONTRASTS[effect]
        for bi, backbone in enumerate(data['backbones']):
            for di, (domain, color, _) in enumerate(DOMAINS):
                q = paired[paired.backbone.eq(backbone) & paired.domain.eq(domain) & paired.effect.eq(effect)]
                q = q.set_index('seed').loc[list(SEEDS)].reset_index()
                row = official[official.backbone.eq(backbone) & official.domain.eq(domain)
                    & official.cohort.eq(cohort) & official.aggregation.eq('mean_logit')
                    & official.comparison.eq(comparison) & official.threshold_kind.eq('default')
                    & official.metric.eq('pooled_auc')].iloc[0]
                values = 100 * q.estimate.to_numpy(float)
                mean, sd = 100 * row.estimate_mean, 100 * row.estimate_sample_sd
                bounds.extend([*values, mean - sd, mean + sd])
                center = bi + (di - .5) * .34
                for si, (value, marker) in enumerate(zip(values, ['o', 's', '^'])):
                    ax.plot(value, center - .085 + si * .045, marker=marker, markersize=2.5,
                        markerfacecolor='white', markeredgecolor=color, markeredgewidth=.65,
                        linestyle='none', zorder=3)
                ax.errorbar(mean, center + .09, xerr=sd, fmt='D', markersize=3,
                    color=color, linewidth=.8, capsize=1.8, markeredgewidth=.6, zorder=4)
                selected.append(row.to_dict())
        span = max(bounds) - min(bounds)
        pad = max(.15, .12 * span)
        ax.set_xlim(min(bounds) - pad, max(bounds) + pad)
        ax.set_ylim(len(data['backbones']) - .45, -.55)
        ax.set_yticks(np.arange(len(data['backbones'])), [DISPLAY[x] for x in data['backbones']])
        ax.tick_params(axis='y', length=0, pad=6)
        ax.set_title(title, loc='left', pad=19)
        ax.set_xlabel('Paired AUC difference (pp)')
        ax.axvline(0, color='#B6BBC0', linewidth=.7, linestyle=(0, (3, 3)), zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.annotate('ab'[panel], xy=(0, 1), xycoords='axes fraction', xytext=(-23, 19),
            textcoords='offset points', fontsize=9, fontweight='bold', ha='left', va='bottom', annotation_clip=False)
    handles = [Line2D([0], [0], marker='D', color=color, markersize=3, linewidth=.8) for _, color, _ in DOMAINS]
    fig.legend(handles, [label for _, _, label in DOMAINS], loc='upper center', bbox_to_anchor=(.59, .98),
        ncol=2, handlelength=1.2, columnspacing=1.5)
    fig.text(.205, .43 / height, 'Raw · Mean-logit · Three matched training seeds per backbone (7, 42, 123).', fontsize=7)
    fig.text(.205, .23 / height, 'Open circle / square / triangle: seeds 7 / 42 / 123; diamond and whiskers: mean ± sample SD.', fontsize=7)
    fig.text(.205, .06 / height, 'a: all B-eligible videos. b: common B/A→B-eligible videos; interaction = (H11−H10)−(H01−H00).', fontsize=7)
    fig.canvas.draw()
    base = output / stem
    alignment = require_matplotlib_panel_alignment(fig, json_out=str(base) + '.alignment.json',
        overlay_svg=str(base) + '.alignment.svg', require_panel_labels=True, strict=True,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    fig.savefig(str(base) + '.pdf')
    fig.savefig(str(base) + '.svg')
    fig.savefig(str(base) + '.png', dpi=600)
    fig.savefig(str(base) + '.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    import pandas as pd
    pd.DataFrame(selected).to_csv(source / f'{stem}_plotted_official_summary.csv', index=False, float_format='%.17g')
    base.with_suffix('.caption.zh.md').write_text(
        '图｜训练预处理历史对固定部署输入及测试输入变化的影响。a，在全部B成功队列上比较M2与M1的AUC差H10−H00；'
        'b，在B/A→B共同成功队列上计算四格交互(H11−H10)−(H01−H00)。M1为B训练，M2为A→B训练。'
        '所有模型使用Raw推理与mean-logit聚合；两种测试域分别为FF++与DFD。空心圆、方形和三角形分别为7、42、123三个配对训练种子，'
        '实心菱形及误差线为官方三种子配对差的均值±样本标准差，不表示测试视频置信区间。'
        '两面板各自保留对应的视频队列，不跨队列相减；全部四格AUC、mean-prob敏感性及阈值结果保留于Source Data。\n')
    if any(digest(args.root / name) != expected for name, expected in data['sources'].items()):
        raise ValueError('H strict-summary inputs changed during render')
    contract = Path(__file__).parent / 'contracts/H_formal_training_history_contract.json'
    receipt = dict(status='RENDERED_AWAITING_PDF_AND_VISUAL_QA', backbones=data['backbones'],
        training_seeds=list(SEEDS), source_sha256=data['sources'], script_sha256=digest(__file__),
        contract_sha256=digest(contract), alignment=alignment['verdict'],
        full_original_metrics_rows=len(data['metrics']), full_original_comparison_rows=len(data['comparisons']),
        primary_and_sensitivity_paired_rows=len(data['paired']), plotted_official_summary_rows=len(selected),
        displayed_statistics='Three matched training-pair contrasts and official mean ± sample SD',
        models_inferred=False, thresholds_recomputed=False, preprocessing_recomputed=False)
    (output / 'render_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
