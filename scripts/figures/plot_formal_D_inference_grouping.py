"""Plot formal inference and grouping effects from complete five-seed summaries."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from formal_D_figure_data import (BACKBONES, SEEDS, digest, figure_tables,
    load_verified, paired_seed_summary)

LABELS = dict(clip_vit_b16='CLIP', dinov2_vit_b14='DINOv2', efficientnet_b0='EfficientNet-B0')
COLORS = dict(mean_logit='#4E79A7', mean_probability='#CC8F5B')
AGG_LABELS = dict(mean_logit='Mean-logit', mean_probability='Mean-prob')


def render(data, domain, output, require_matplotlib_panel_alignment):
    absolute, paired = figure_tables(data['metrics'], data['effects'], domain)
    summary = paired_seed_summary(paired)
    tag = 'all_backbones' if len(data['backbones']) == 3 else data['backbones'][0]
    stem = f'D_inference_grouping_{tag}_{domain}'
    source = output/'source_data'
    source.mkdir(parents=True, exist_ok=True)
    absolute.to_csv(source/(stem+'_absolute_AUC.csv'), index=False, float_format='%.17g')
    paired.to_csv(source/(stem+'_paired_effects.csv'), index=False, float_format='%.17g')
    summary.to_csv(source/(stem+'_seed_summary.csv'), index=False, float_format='%.17g')

    fig, axes = plt.subplots(1, 2, figsize=(7.204724409448819, 3.4645669291338583))
    fig.subplots_adjust(left=.087, right=.985, bottom=.27, top=.78, wspace=.33)
    effects = ['delta_correct', 'interaction']
    titles = ['Correct video grouping', 'Effect of merging videos']
    ylabels = ['L2 − Raw (AUC, pp)', 'Change in L2 gain (AUC, pp)']
    for panel, (ax, effect, title, ylabel) in enumerate(zip(axes, effects, titles, ylabels)):
        bounds = [0.]
        for i, backbone in enumerate(data['backbones']):
            for j, aggregation in enumerate(COLORS):
                center = i+(j-.5)*.36
                q = paired[(paired.backbone == backbone) & (paired.aggregation == aggregation)
                    & (paired.effect == effect)].set_index('seed').loc[list(SEEDS)]
                values = q.estimate_pp.to_numpy(float)
                row = summary[(summary.backbone == backbone) & (summary.aggregation == aggregation)
                    & (summary.effect == effect)].iloc[0]
                mean, sd = row.estimate_mean_pp, row.estimate_sample_sd_pp
                for k, value in enumerate(values):
                    ax.plot(center-.045+(k-2)*.018, value, marker='o', markersize=3.,
                        markerfacecolor='white', markeredgecolor=COLORS[aggregation],
                        markeredgewidth=.7, linestyle='none', zorder=3)
                ax.errorbar(center+.075, mean, yerr=sd, fmt='D', markersize=3.5,
                    color=COLORS[aggregation], linewidth=.9, capsize=2, markeredgewidth=.7, zorder=4)
                bounds.extend(values)
                bounds.extend([mean-sd, mean+sd])
        lower, upper = min(bounds), max(bounds)
        pad = max(.05, (upper-lower)*.16)
        ax.set_ylim(lower-pad, upper+pad)
        ax.set_xlim(-.55, len(data['backbones'])-.45)
        ax.set_xticks(np.arange(len(data['backbones'])), [LABELS[b] for b in data['backbones']])
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc='left', pad=14)
        ax.axhline(0, color='#AEB5BB', linewidth=.7, linestyle=(0, (3, 3)), zorder=0)
        ax.tick_params(axis='x', length=0, pad=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.annotate('ab'[panel], xy=(0, 1), xycoords='axes fraction', xytext=(-28, 14),
            textcoords='offset points', fontsize=9, fontweight='bold', va='bottom',
            ha='left', annotation_clip=False)
    handles = [Line2D([0], [0], marker='D', color=color, linewidth=.8, markersize=3.5)
        for color in COLORS.values()]
    fig.legend(handles, list(AGG_LABELS.values()), loc='upper center',
        bbox_to_anchor=(.53, .975), ncol=2, handlelength=1.2, columnspacing=2.0)
    domain_label, counts = ('DF40-CDF', '157 real / 12,037 fake; 157 real / 1,830 collision groups') if domain == 'target_full' else (
        'DF40-FF', '140 real / 2,884 fake; 140 real / 280 collision groups')
    fig.text(.087, .15, f'{domain_label} · Natural videos and merged units: {counts}.', fontsize=7)
    fig.text(.087, .095, 'Five matched training seeds per backbone. Open circles: seeds; diamonds: mean ± sample SD.', fontsize=7)
    fig.text(.087, .04, 'Right: collision gain − correct gain. A larger L2 gain does not imply a higher absolute AUC.', fontsize=7)
    fig.canvas.draw()
    base = output/stem
    checked = require_matplotlib_panel_alignment(fig, json_out=str(base)+'.alignment.json',
        overlay_svg=str(base)+'.alignment.svg', require_panel_labels=True, strict=True,
        tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    fig.savefig(str(base)+'.pdf')
    fig.savefig(str(base)+'.svg')
    fig.savefig(str(base)+'.png', dpi=600)
    fig.savefig(str(base)+'.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    caption = (f'图｜{domain_label}上推理L2的配对效应及错误视频合并的影响。'
        'a，正确自然视频分组下，L2 AUC减Raw AUC。b，错误合并下的L2增益减正确分组下的L2增益。'
        '两种聚合使用相同冻结模型、相同逐帧预测和相应分组定义。空心圆为五个正式训练种子，实心菱形及误差线为配对差的均值±样本SD（ddof=1），'
        '不代表测试视频置信区间。正交互不自动意味着正确负、错误正的结论反转。'
        '全部八格绝对AUC、所有种子及图中差值保存在CSV Source Data；不同分组的评价单位数量不同。\n')
    base.with_suffix('.caption.zh.md').write_text(caption)
    return dict(domain=domain, filename=stem, alignment=checked['verdict'],
        absolute_AUC_rows=len(absolute), paired_effect_rows=len(paired), seed_summary_rows=len(summary))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--qa-dir', type=Path, required=True)
    parser.add_argument('--backbone', choices=BACKBONES)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    data = load_verified(args.root, [args.backbone] if args.backbone else BACKBONES)
    sys.path.insert(0, str(args.qa_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.linewidth': .7, 'legend.frameon': False, 'savefig.facecolor': 'white'})
    output = args.output_dir or args.root/'figures/formal_D_inference_grouping'
    output.mkdir(parents=True, exist_ok=True)
    panels = [render(data, domain, output, require_matplotlib_panel_alignment)
        for domain in ['target_full', 'source_test']]
    if any(digest(args.root/name) != expected for name, expected in data['sources'].items()):
        raise ValueError('formal figure input changed during render')
    contract = Path(__file__).parent/'contracts/D_formal_inference_grouping_contract.json'
    record = dict(status='RENDERED_AWAITING_PDF_AND_VISUAL_QA', backend='python',
        backbones=data['backbones'], training_seeds=data['training_seeds'], panels=panels,
        input_sha256=data['sources'], script_sha256=digest(__file__),
        data_loader_sha256=digest(Path(__file__).with_name('formal_D_figure_data.py')),
        contract_sha256=digest(contract), models_inferred=False,
        all_selected_seeds_shown=True, uncertainty='within-seed differences; mean and sample SD across five training seeds')
    (output/'render_receipt.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
