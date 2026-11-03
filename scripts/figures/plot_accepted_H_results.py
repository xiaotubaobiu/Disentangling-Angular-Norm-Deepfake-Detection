"""Render accepted H coverage and paired effects, without model inference."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_panel(ax, label):
    ax.annotate(label, xy=(0, 1), xycoords='axes fraction', xytext=(-27, 17),
                textcoords='offset points', fontsize=9, fontweight='bold',
                ha='left', va='bottom', annotation_clip=False)


def export(fig, path, require_matplotlib_panel_alignment):
    fig.canvas.draw()
    report = require_matplotlib_panel_alignment(fig, json_out=str(path) + '.alignment.json',
                       overlay_svg=str(path) + '.alignment.svg',
                       require_panel_labels=True, strict=True,
                       tolerance_pt=1.5, gutter_tolerance_pt=1.5)
    fig.savefig(str(path) + '.pdf')
    fig.savefig(str(path) + '.svg')
    fig.savefig(str(path) + '.png', dpi=600)
    fig.savefig(str(path) + '.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    return report['verdict']


def coverage(root, out, alignment):
    source = root / 'reports/H_preprocessing/b_vs_ab_membership.csv'
    frame = pd.read_csv(source)
    assert len(frame) == 4 and not frame[['dataset', 'label']].duplicated().any()
    assert (frame[['common', 'B_only', 'AB_only', 'neither']].sum(axis=1)
            == frame.candidate_videos).all()
    order = [('FFPP', 0), ('FFPP', 1), ('DFD', 0), ('DFD', 1)]
    frame = frame.set_index(['dataset', 'label']).loc[order].reset_index()
    assert frame.candidate_videos.tolist() == [140, 560, 363, 3068]
    frame['B'] = frame.common + frame.B_only
    frame['AB'] = frame.common + frame.AB_only
    frame.to_csv(out / 'source_data/H_preprocessing_coverage.csv', index=False)
    labels = ['FF++\nReal', 'FF++\nFake', 'DFD\nReal', 'DFD\nFake']
    x = np.arange(4)
    fig, axes = plt.subplots(1, 2, figsize=(7.204724409448819, 3.2283464566929134))
    fig.subplots_adjust(left=.087, right=.985, bottom=.22, top=.78, wspace=.27)
    ax = axes[0]
    for idx, (key, name, color, hatch) in enumerate([
            ('B', 'B', '#4E79A7', ''), ('AB', 'A→B', '#CC8F5B', ''),
            ('common', 'Common', '#89928D', '')]):
        values = frame[key].to_numpy() / frame.candidate_videos.to_numpy() * 100
        ax.bar(x + (idx - 1) * .23, values, width=.21, label=name,
               color=color, hatch=hatch, linewidth=.35, edgecolor='white')
    ax.set_title('Eligible videos by preprocessing history', pad=28, loc='left')
    ax.set_ylabel('Eligible videos (%)')
    ax.legend(loc='lower left', bbox_to_anchor=(-.015, 1.02), ncol=3,
              borderaxespad=0, handlelength=1.1, columnspacing=1.0)
    ax = axes[1]
    bottom = np.zeros(4)
    colors = ['#89928D', '#4E79A7', '#CC8F5B', '#E3E5E7']
    hatches = ['', '', '', '']
    names = ['Common', 'B only', 'A→B only', 'Neither']
    for key, name, color, hatch in zip(['common', 'B_only', 'AB_only', 'neither'], names, colors, hatches):
        values = frame[key].to_numpy() / frame.candidate_videos.to_numpy() * 100
        ax.bar(x, values, bottom=bottom, width=.58, color=color, label=name,
               hatch=hatch, linewidth=.4, edgecolor='white')
        bottom += values
    assert np.allclose(bottom, 100)
    ax.set_title('Video-set overlap', pad=28, loc='left')
    ax.set_ylabel('Candidate videos (%)')
    ax.legend(loc='lower left', bbox_to_anchor=(-.015, 1.02), ncol=2,
              borderaxespad=0, handlelength=1.1, columnspacing=1.0)
    for idx, ax in enumerate(axes):
        label_panel(ax, 'ab'[idx])
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(-.6, 3.6)
        ax.tick_params(axis='x', length=0, pad=5)
        ax.spines[['top', 'right']].set_visible(False)
    # All candidate videos in both displayed TEST roles are included. Counts
    # are denominators, not independent sampling replicates; no CI is drawn.
    fig.text(.087, .045, 'Candidate videos: FF++ 140 real / 560 fake; DFD 363 real / 3,068 fake.', fontsize=7)
    return source, export(fig, out / 'H_preprocessing_coverage', alignment)


def paired(root, out, alignment):
    folder = root / 'analysis/H/efficientnet_b0/three_seed_summary'
    complete = json.loads((folder / 'complete.json').read_text())
    for name, sha in complete['artifact_sha256'].items():
        assert digest(folder / name) == sha, name
    source = folder / 'per_seed_comparisons.csv'
    full = pd.read_csv(source)
    frame = full[(full.aggregation == 'mean_logit') & (full.metric == 'pooled_auc')
                 & (full.threshold_kind == 'default')].copy()
    selected = []
    fig, axes = plt.subplots(1, 2, figsize=(7.204724409448819, 3.2283464566929134))
    fig.subplots_adjust(left=.087, right=.985, bottom=.27, top=.78, wspace=.32)
    colors = {'source_test': '#4E79A7', 'target_test': '#CC8F5B'}
    domains = ['source_test', 'target_test']
    domain_labels = ['FF++', 'DFD']
    seeds = [7, 42, 123]
    markers = ['o', 's', '^']

    def add(ax, subset, center, color):
        subset = subset.set_index('seed').loc[seeds].reset_index()
        assert len(subset) == 3 and subset.estimate.notna().all()
        values = subset.estimate.to_numpy() * 100
        selected.append(subset)
        for seed_idx, (value, marker) in enumerate(zip(values, markers)):
            ax.plot(center + (seed_idx - 1) * .045, value, marker=marker,
                    markersize=3.3, color=color, markerfacecolor='white',
                    markeredgewidth=.7, linestyle='none', zorder=3)
        ax.errorbar(center + .15, values.mean(), yerr=values.std(ddof=1),
                    fmt='D', markersize=3.4, color=color, capsize=2,
                    linewidth=.9, markeredgewidth=.7, zorder=4)

    for idx, domain in enumerate(domains):
        q = frame[(frame.domain == domain) & (frame.cohort == 'all_B')
                  & (frame.comparison == 'H10-H00')]
        add(axes[0], q, idx, colors[domain])
    axes[0].set_xticks([0, 1], domain_labels)
    axes[0].set_xlim(-.4, 1.6)
    axes[0].set_ylim(-1.1, 1.0)
    axes[0].set_ylabel('M2 − M1 on B inputs (AUC, pp)')
    axes[0].set_title('Fixed-B deployment comparison', pad=28, loc='left')
    axes[0].text(.01, 1.06, 'All B-eligible videos', transform=axes[0].transAxes, fontsize=7)
    comparisons = ['H01-H00', 'H11-H10', '(H11-H10)-(H01-H00)']
    for idx, comparison in enumerate(comparisons):
        for domain_idx, domain in enumerate(domains):
            q = frame[(frame.domain == domain) & (frame.cohort == 'common')
                      & (frame.comparison == comparison)]
            add(axes[1], q, idx + (domain_idx - .5) * .36, colors[domain])
    axes[1].set_xticks(range(3), ['M1 input\nswitch', 'M2 input\nswitch', 'Interaction'])
    axes[1].set_xlim(-.6, 2.6)
    axes[1].set_ylim(-2.0, 2.2)
    axes[1].set_ylabel('Paired AUC difference (pp)')
    axes[1].set_title('Train–test history interaction', pad=28, loc='left')
    handles = [Line2D([0], [0], marker='D', color=color, markersize=3.5, linewidth=.8)
               for color in colors.values()]
    axes[1].legend(handles, domain_labels, loc='lower left', bbox_to_anchor=(0, 1.02),
                   ncol=2, borderaxespad=0, handlelength=1, columnspacing=1.2)
    for idx, ax in enumerate(axes):
        label_panel(ax, 'ab'[idx])
        ax.axhline(0, color='#B6BBC0', linewidth=.7, linestyle=(0, (3, 3)), zorder=0)
        ax.tick_params(axis='x', length=0, pad=6)
        ax.spines[['top', 'right']].set_visible(False)
    # All three registered matched training seeds are shown. Mean ± sample SD
    # describes seed variability; these are not video-bootstrap confidence CIs.
    fig.text(.087, .10, 'EfficientNet-B0 · Raw · Mean-logit · Three matched training seeds (7, 42, 123).', fontsize=7)
    fig.text(.087, .035, 'Open markers: individual seeds; filled diamonds and whiskers: mean ± sample SD. Input switch: B to A→B.', fontsize=7)
    pd.concat(selected, ignore_index=True).to_csv(out / 'source_data/H_efficientnet_paired_effects.csv', index=False)
    return source, export(fig, out / 'H_efficientnet_paired_effects', alignment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--qa-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.qa_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 7, 'pdf.fonttype': 42,
        'svg.fonttype': 'none', 'axes.linewidth': .7, 'legend.frameon': False,
        'savefig.facecolor': 'white'})
    out = args.root / 'figures'
    (out / 'source_data').mkdir(parents=True, exist_ok=True)
    assert (out / 'figure_contract.json').is_file()
    source_records = {}
    for renderer in [coverage, paired]:
        source, alignment = renderer(args.root, out, require_matplotlib_panel_alignment)
        source_records[str(source.relative_to(args.root))] = dict(sha256=digest(source), alignment=alignment)
    receipt = dict(status='RENDERED_AWAITING_PDF_AND_VISUAL_QA', source=source_records,
                   script_sha256=digest(__file__), backend='python', models_inferred=False,
                   matplotlib_version=matplotlib.__version__, pandas_version=pd.__version__,
                   numpy_version=np.__version__)
    (out / 'H_figures_render_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
