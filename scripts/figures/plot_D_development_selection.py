"""Plot completed developer histories, without evaluating or selecting TEST."""
from pathlib import Path
import argparse
import json
import sys
import hashlib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--qa-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(args.qa_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    receipt_path = root / 'operations/source_vs_cdf_development_epochs.json'
    accepted = json.loads(receipt_path.read_text())
    assert accepted['status'] == 'COMPLETED_DEVELOPMENT_HISTORY_COMPARISON'
    assert accepted['completed_runs'] == 12 and accepted['test_used'] is False
    for name, expected in accepted['input_sha256'].items():
        assert digest(name) == expected, name
    table_path = root / 'reports/development_model_selection/source_vs_cdf_development_epochs.csv'
    assert digest(table_path) == accepted['csv_sha256']
    table = pd.read_csv(table_path)
    assert len(table) == 12 and table.run_id.nunique() == 12
    bone_order = {'clip_vit_b16': 0, 'efficientnet_b0': 1, 'dinov2_vit_b14': 2}
    names = {'clip_vit_b16': 'CLIP', 'efficientnet_b0': 'EfficientNet', 'dinov2_vit_b14': 'DINOv2'}
    table['display_order'] = table.backbone.map(bone_order)
    table = table.sort_values(['display_order', 'seed', 'cap'], ascending=[True, True, False]).reset_index(drop=True)
    curves = []
    for row in table.itertuples():
        history = json.loads((root / 'runs' / row.run_id / 'history.json').read_text())
        assert len(history) == row.actual_stop_epoch
        assert max(history, key=lambda x: x['dev']['pooled_auc'])['epoch'] == row.source_best_epoch
        assert max(history, key=lambda x: x['cdf_dev']['pooled_auc'])['epoch'] == row.cdf_dev_best_epoch
        curves.extend(dict(run_id=row.run_id, backbone=row.backbone, seed=row.seed, cap=row.cap,
                           epoch=x['epoch'], source_auc_percent=100*x['dev']['pooled_auc'],
                           cdf_dev_auc_percent=100*x['cdf_dev']['pooled_auc']) for x in history)
    curves = pd.DataFrame(curves)
    assert len(curves) == 102 and np.isfinite(curves[['source_auc_percent', 'cdf_dev_auc_percent']]).all().all()
    assert ((curves[['source_auc_percent', 'cdf_dev_auc_percent']] >= 0) &
            (curves[['source_auc_percent', 'cdf_dev_auc_percent']] <= 100)).all().all()
    out = root / 'figures'
    source = out / 'source_data'
    source.mkdir(exist_ok=True)
    table.to_csv(source / 'D_development_selected_epochs.csv', index=False)
    curves.to_csv(source / 'D_development_all_epoch_curves.csv', index=False)

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.linewidth': .7, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(7.204724409448819, 4.330708661417323),
                             gridspec_kw={'width_ratios': [1, 1]})
    fig.subplots_adjust(left=.20, right=.98, bottom=.19, top=.80, wspace=.48)
    blue, ochre, neutral = '#4E79A7', '#B67A48', '#B8BFC4'
    labels = []
    y = np.arange(12)[::-1]
    for index, row in enumerate(table.itertuples()):
        labels.append(f"{names[row.backbone]} {row.seed}/{row.cap}{'*' if row.chosen_budget else ''}")
        axes[0].plot([row.source_best_epoch, row.cdf_dev_best_epoch], [y[index]]*2,
                     color=neutral, linewidth=1.1, zorder=1)
        axes[0].plot(row.source_best_epoch, y[index], marker='o', markersize=5,
                     markerfacecolor='white', markeredgecolor=blue, markeredgewidth=1, zorder=2)
        axes[0].plot(row.cdf_dev_best_epoch, y[index], marker='^', markersize=3.3,
                     color=ochre, zorder=3)
    axes[0].set_yticks(y, labels)
    axes[0].set_ylim(-.7, 11.7)
    axes[0].set_xlim(.2, 30.8)
    axes[0].set_xticks([1, 5, 10, 15, 20, 25, 30])
    axes[0].set_xlabel('Selected epoch')
    axes[0].set_title('Best epoch by selection domain', loc='left', pad=18)
    axes[0].tick_params(axis='y', length=0, pad=5)
    for seed, linestyle in [(1007, '-'), (1042, '--')]:
        q = curves[(curves.backbone == 'dinov2_vit_b14') & (curves.seed == seed)]
        assert len(q) in [25, 27]
        assert q[['source_auc_percent', 'cdf_dev_auc_percent']].min().min() >= 65
        for column, color, marker in [('source_auc_percent', blue, 'o'),
                                      ('cdf_dev_auc_percent', ochre, '^')]:
            axes[1].plot(q.epoch, q[column], color=color, linestyle=linestyle, linewidth=.95)
            best = q.loc[q[column].idxmax()]
            axes[1].plot(best.epoch, best[column], marker=marker, markersize=4.5,
                         markerfacecolor='white' if marker == 'o' else color,
                         markeredgecolor=color, markeredgewidth=1, zorder=5)
    axes[1].set_xlim(.2, 30.8)
    axes[1].set_xticks([1, 10, 20, 30])
    axes[1].set_ylim(65, 95)
    axes[1].set_yticks([65, 70, 75, 80, 85, 90, 95])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Video AUC (%)')
    axes[1].set_title('DINOv2 development trajectories', loc='left', pad=18)
    seed_handles = [Line2D([], [], color='#5B686D', linewidth=1, linestyle=s, label=f'Seed {n}')
                    for n, s in [(1007, '-'), (1042, '--')]]
    axes[1].legend(handles=seed_handles, loc='lower left', bbox_to_anchor=(-.02, 1.005),
                   borderaxespad=0, ncol=2, columnspacing=.8, handlelength=1.8)
    for index, ax in enumerate(axes):
        ax.spines[['top', 'right']].set_visible(False)
        ax.text(-.13, 1.115, 'ab'[index], transform=ax.transAxes,
                fontsize=8, fontweight='bold', ha='left', va='top')
    domain_handles = [Line2D([], [], color=blue, marker='o', markerfacecolor='white',
                            linewidth=1, markersize=4.5, label='FF++ source DEV'),
                      Line2D([], [], color=ochre, marker='^', linewidth=1,
                            markersize=4, label='Isolated CDF DEV')]
    fig.legend(handles=domain_handles, loc='upper center', bbox_to_anchor=(.58, .985),
               ncol=2, columnspacing=2, handlelength=2)
    fig.text(.06, .085, '12 completed development runs; * frozen budget. Labels show backbone, seed / maximum epochs.', fontsize=7)
    fig.text(.06, .035, 'Separate observed histories; no seed average, confidence interval, or formal TEST selection.', fontsize=7)
    path = out / 'D_development_source_CDF_selection'
    fig.canvas.draw()
    alignment = require_matplotlib_panel_alignment(fig, json_out=str(path)+'.alignment.json',
        overlay_svg=str(path)+'.alignment.svg', tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        require_panel_labels=True, strict=True)
    fig.savefig(str(path)+'.pdf')
    fig.savefig(str(path)+'.svg')
    fig.savefig(str(path)+'.png', dpi=600)
    fig.savefig(str(path)+'.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    for name, expected in accepted['input_sha256'].items():
        assert digest(name) == expected
    rendered = dict(status='RENDERED_AWAITING_PDF_VISUAL_QA', backend='python',
        script_sha256=digest(__file__), accepted_comparison_sha256=digest(receipt_path),
        completed_developer_trajectories=12, all_epoch_source_rows=102,
        DINO_curve_rows=52, alignment=alignment['verdict'],
        source_sha256={p.name: digest(p) for p in source.glob('D_development_*.csv')},
        training_performed=False, model_inference_performed=False, formal_test_used=False)
    (out / 'D_development_selection_render_receipt.json').write_text(json.dumps(rendered, indent=2)+'\n')
    print(json.dumps(rendered, indent=2))


if __name__ == '__main__':
    main()
