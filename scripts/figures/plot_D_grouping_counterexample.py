"""Explain one verified grouping counterexample; not a seed-robustness plot."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_accepted_H_results import digest, label_panel


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--qa-dir', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    sys.path.insert(0, str(args.qa_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    folder = root / 'reports/grouping_decomposition'
    weights_path = folder / 'CDF_effective_method_weights_CLIP2026.csv'
    cells_path = folder / 'grouping_decomposition_cells.csv'
    weights_receipt = json.loads((root / 'operations/CDF_method_reweighting_CLIP2026.json').read_text())
    cells_receipt = json.loads((root / 'operations/grouping_decomposition_two_CLIP_seeds.json').read_text())
    assert weights_receipt['status'] == cells_receipt['status'] == 'PASS'
    assert digest(weights_path) == weights_receipt['output_csv_sha256']
    assert digest(cells_path) == cells_receipt['output_sha256'][cells_path.name]
    weights = pd.read_csv(weights_path)
    cells = pd.read_csv(cells_path)
    cells = cells[(cells.seed == 2026) & (cells.role == 'target_full')
                  & (cells.aggregation == 'mean_logit')].set_index('representation')
    assert len(weights) == 21 and set(cells.index) == {'raw', 'l2'}
    assert cells.n_real.eq(157).all() and cells.natural_fake.eq(12037).all()
    assert cells.collision_fake.eq(1830).all()
    assert weights.videos.sum() == 12037
    assert np.allclose(weights[['natural_mass', 'ID_equal_member_mass']].sum(), 1)
    categories = [('wav2lip', weights.method_id.eq('wav2lip')),
                  ('sadtalker', weights.method_id.eq('sadtalker')),
                  ('Other 19', ~weights.method_id.isin(['wav2lip', 'sadtalker']))]
    masses = pd.DataFrame([dict(method=name,
        natural_mass=weights.loc[mask, 'natural_mass'].sum(),
        ID_equal_member_mass=weights.loc[mask, 'ID_equal_member_mass'].sum())
        for name, mask in categories])
    assert np.allclose(masses[['natural_mass', 'ID_equal_member_mass']].sum(), 1)
    columns = ['A00_natural_equal', 'A10_group_equal_member_scores', 'A11_group_equal_group_mean']
    delta = (cells.loc['l2', columns].astype(float) - cells.loc['raw', columns].astype(float)) * 100
    assert delta.iloc[0] < 0 < delta.iloc[1] and 0 < delta.iloc[2] < delta.iloc[1]
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
        'legend.frameon': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
        'axes.linewidth': .7, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(7.204724409448819, 3.2283464566929134))
    fig.subplots_adjust(left=.087, right=.985, bottom=.28, top=.78, wspace=.32)
    x = np.arange(3)
    for idx, (key, name, color) in enumerate([
            ('natural_mass', 'Natural-video weights', '#4E79A7'),
            ('ID_equal_member_mass', 'Equal-ID weights', '#CC8F5B')]):
        axes[0].bar(x + (idx - .5) * .32, masses[key] * 100, width=.28,
                    color=color, label=name)
    axes[0].set_xticks(x, masses.method)
    axes[0].set_ylim(0, 100)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].set_ylabel('Effective fake-video weight (%)')
    axes[0].set_title('Unequal method reweighting', pad=28, loc='left')
    axes[0].legend(loc='lower left', bbox_to_anchor=(-.015, 1.02), ncol=1,
                   borderaxespad=0, handlelength=1.1, labelspacing=.3)
    axes[1].plot(x, delta, color='#5B686D', linewidth=1.1,
                 marker='o', markersize=4.2)
    axes[1].axhline(0, color='#A9AFB5', linewidth=.7, linestyle=(0, (3, 3)))
    axes[1].set_xticks(x, ['Natural-video\nweights', 'Equal-ID\nweights', 'Equal ID\n+ Mean'])
    axes[1].set_xlim(-.4, 2.4)
    axes[1].set_ylim(-.4, 1.05)
    axes[1].set_yticks([-.25, 0, .25, .5, .75, 1])
    axes[1].set_ylabel('L2 − Raw AUC (pp)')
    axes[1].set_title('Relative conclusion flips before averaging', pad=28, loc='left')
    axes[1].text(0, 1.07, 'Same model and fixed frame scores', transform=axes[1].transAxes, fontsize=7)
    for xi, value in enumerate(delta):
        axes[1].annotate(f'{value:+.4f}', (xi, value),
                         xytext=(0, 9 if value >= 0 else -9),
                         textcoords='offset points', ha='center',
                         va='bottom' if value >= 0 else 'top', fontsize=7)
    for idx, ax in enumerate(axes):
        label_panel(ax, 'ab'[idx])
        ax.tick_params(axis='x', length=0, pad=6)
        ax.spines[['top', 'right']].set_visible(False)
    # These points are deterministic metrics of one fixed model/cohort, not
    # stochastic seed aggregates. No uncertainty band or confidence CI is drawn.
    fig.text(.087, .105, 'One counterexample: CLIP seed 2026 · DF40-CDF · Mean-logit (no Sigmoid).', fontsize=7)
    fig.text(.087, .04, '157 real / 12,037 fake natural videos; 1,830 fake ID groups. Not a five-seed robustness estimate.', fontsize=7)
    out = root / 'figures'
    masses.to_csv(out / 'source_data/D_CLIP2026_method_weights.csv', index=False)
    cells.loc[['raw', 'l2'], columns].to_csv(out / 'source_data/D_CLIP2026_grouping_path.csv')
    path = out / 'D_CLIP2026_grouping_counterexample'
    fig.canvas.draw()
    alignment_report = require_matplotlib_panel_alignment(fig,
        json_out=str(path) + '.alignment.json', overlay_svg=str(path) + '.alignment.svg',
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, require_panel_labels=True, strict=True)
    fig.savefig(str(path) + '.pdf')
    fig.savefig(str(path) + '.svg')
    fig.savefig(str(path) + '.png', dpi=600)
    fig.savefig(str(path) + '.tiff', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)
    alignment = alignment_report['verdict']
    receipt = dict(status='RENDERED_AWAITING_PDF_AND_VISUAL_QA', backend='python',
        script_sha256=digest(__file__), alignment=alignment,
        source_sha256={str(p.relative_to(root)): digest(p) for p in [weights_path, cells_path]},
        derived_delta_pp=delta.tolist(), trained_models=1, final_five_seed_summary=False,
        models_inferred=False, training_performed=False)
    (out / 'D_CLIP2026_figure_render_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
