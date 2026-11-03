"""Display saved source/n20 operating points without recomputing thresholds."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from formal_D_figure_data import BACKBONES, SEEDS, digest, load_verified

DISPLAY = dict(clip_vit_b16='CLIP', dinov2_vit_b14='DINOv2', efficientnet_b0='EfficientNet-B0')
FLOWS = [('raw','mean_logit','Raw logit'),('raw','mean_probability','Raw prob'),
         ('l2','mean_logit','L2 logit'),('l2','mean_probability','L2 prob')]
KINDS = [('source_real','#78858F','Source real CAL'),('target_real','#8C73A8','Target real CAL (n = 20)')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--qa-dir',type=Path,required=True)
    parser.add_argument('--backbone',choices=BACKBONES)
    parser.add_argument('--output-dir',type=Path)
    args = parser.parse_args()
    data = load_verified(args.root,[args.backbone] if args.backbone else BACKBONES)
    sys.path.insert(0,str(args.qa_dir))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':7,'axes.titlesize':8,'axes.labelsize':7,'xtick.labelsize':7,
        'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none',
        'axes.linewidth':.7,'legend.frameon':False,'savefig.facecolor':'white'})
    output = args.output_dir or args.root/'figures/formal_D_operating_points'
    source = output/'source_data'
    source.mkdir(parents=True,exist_ok=True)
    all_models,all_summary = [],[]
    for backbone in data['backbones']:
        folder = args.root/'analysis/D'/backbone/'five_seed_summary'
        models = pd.read_csv(folder/'per_seed_calibration.csv',float_precision='round_trip')
        means = pd.read_csv(folder/'five_seed_calibration.csv',float_precision='round_trip')
        assert set(models.backbone)==set(means.backbone)=={backbone}
        assert models.domain.eq('target_test').all() and means.domain.eq('target_test').all()
        all_models.append(models);all_summary.append(means)
    full = pd.concat(all_models,ignore_index=True)
    aggregate = pd.concat(all_summary,ignore_index=True)
    tag = 'all_backbones' if len(data['backbones'])==3 else data['backbones'][0]
    stem = f'D_operating_points_{tag}'
    # Retain every original CAL budget and metric, including unavailable n80.
    full.to_csv(source/(stem+'_all_calibration.csv'),index=False,float_format='%.17g')
    selected,saved_summary = [],[]
    nrows = len(data['backbones'])*5-1
    height = 3.6 if len(data['backbones'])==1 else 5.5
    fig,axes = plt.subplots(1,3,figsize=(7.204724409448819,height),sharey=True)
    # Reserve physical space for axis labels, three footnotes and the legend.
    fig.subplots_adjust(left=.205,right=.985,bottom=1./height,top=1.-.75/height,wspace=.20)
    axis_bounds = {metric:[0.,100.] for metric in ['fpr','tpr','pooled_ba']}
    ticks,labels = [],[]
    for bi,backbone in enumerate(data['backbones']):
        for fi,(representation,aggregation,flow_label) in enumerate(FLOWS):
            y = bi*5+fi
            ticks.append(y);labels.append(f'{DISPLAY[backbone]} · {flow_label}')
            for ki,(kind,color,_) in enumerate(KINDS):
                q = full[full.backbone.eq(backbone)&full.representation.eq(representation)
                    &full.aggregation.eq(aggregation)&full.comparison.eq(kind)]
                s = aggregate[aggregate.backbone.eq(backbone)&aggregate.representation.eq(representation)
                    &aggregate.aggregation.eq(aggregation)&aggregate.comparison.eq(kind)]
                if kind=='target_real':q=q[q.budget.eq(20)];s=s[s.budget.eq(20)]
                for ax,metric in zip(axes,['fpr','tpr','pooled_ba']):
                    points = q[q.metric.eq(metric)].copy()
                    if len(points)!=5 or set(points.seed)!=set(SEEDS) or points.seed.duplicated().any():
                        raise ValueError('all five matched model operating points required')
                    points = points.set_index('seed').loc[list(SEEDS)].reset_index()
                    row = s[s.metric.eq(metric)]
                    if len(row)!=1 or not row.complete_five_seeds.eq(True).all():
                        raise ValueError('complete official five-seed operating summary required')
                    row = row.iloc[0];values=points.estimate.to_numpy(float)
                    if (not np.isfinite(values).all() or not np.isclose(row.estimate_mean,values.mean(),rtol=0,atol=1e-12)
                            or not np.isclose(row.estimate_sample_sd,values.std(ddof=1),rtol=0,atol=1e-12)):
                        raise ValueError('official model mean/SD does not reconstruct')
                    axis_bounds[metric].extend([100*(row.estimate_mean-row.estimate_sample_sd),
                        100*(row.estimate_mean+row.estimate_sample_sd)])
                    center = y+(ki-.5)*.44
                    for si,value in enumerate(values):
                        ax.plot(100*value,center-.14+si*.045,marker='o',markersize=1.6,
                            markerfacecolor='white',markeredgecolor=color,markeredgewidth=.55,
                            linestyle='none',zorder=3)
                    ax.errorbar(100*row.estimate_mean,center+.17,
                        xerr=100*row.estimate_sample_sd,fmt='D',markersize=2.8,
                        color=color,linewidth=.75,capsize=1.6,markeredgewidth=.6,zorder=4)
                    selected.append(points)
                    saved_summary.append(row.to_dict())
    for panel,(ax,title,label,metric) in enumerate(zip(axes,
            ['False-positive rate','Recall','Balanced accuracy'],['FPR (%)','TPR (%)','BA (%)'],
            ['fpr','tpr','pooled_ba'])):
        ax.set_title(title,loc='left',pad=14)
        ax.set_xlabel(label)
        ax.set_xticks([0,25,50,75,100])
        ax.set_xlim(min(axis_bounds[metric])-4,max(axis_bounds[metric])+4)
        ax.set_ylim(nrows-.4,-.6)
        ax.set_yticks(ticks,labels)
        ax.tick_params(axis='y',length=0,pad=5)
        ax.spines[['top','right']].set_visible(False)
        for line in range(1,len(data['backbones'])):
            ax.axhline(line*5-.7,color='#D9DDE0',linewidth=.6,zorder=0)
        ax.annotate('abc'[panel],xy=(0,1),xycoords='axes fraction',xytext=(-20,14),
            textcoords='offset points',fontsize=9,fontweight='bold',ha='left',va='bottom',annotation_clip=False)
    handles=[Line2D([0],[0],marker='D',color=color,markersize=3.,linewidth=.8) for _,color,_ in KINDS]
    fig.legend(handles,[label for _,_,label in KINDS],loc='upper center',bbox_to_anchor=(.58,.975),
        ncol=2,handlelength=1.1,columnspacing=1.5)
    fig.text(.205,.375/height,'DF40-CDF · Correct grouping · Fixed TEST: 93 real / 12,037 fake videos.',fontsize=7)
    fig.text(.205,(16./72)/height,'Open circles: five training seeds; diamonds: official mean ± sample SD.',fontsize=7)
    fig.text(.205,(5./72)/height,'Target n20: 200 CAL subsets averaged within each model. AUC is unchanged by thresholds.',fontsize=7)
    fig.canvas.draw()
    base=output/stem
    alignment=require_matplotlib_panel_alignment(fig,json_out=str(base)+'.alignment.json',
        overlay_svg=str(base)+'.alignment.svg',require_panel_labels=True,strict=True,
        tolerance_pt=1.5,gutter_tolerance_pt=1.5)
    fig.savefig(str(base)+'.pdf')
    fig.savefig(str(base)+'.svg')
    fig.savefig(str(base)+'.png',dpi=600)
    fig.savefig(str(base)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)
    pd.concat(selected,ignore_index=True).to_csv(source/(stem+'_plotted_per_seed.csv'),index=False,float_format='%.17g')
    pd.DataFrame(saved_summary).to_csv(source/(stem+'_plotted_official_summary.csv'),index=False,float_format='%.17g')
    caption=('图｜源真实校准与20个目标真实视频校准的操作点。a，误报率FPR；b，召回率TPR；c，平衡准确率BA。'
        '每行对应一个骨干及Raw/L2、mean-logit/mean-prob分数组合。空心圆为五个正式训练种子，实心菱形和误差线为官方五种子均值±样本SD。'
        '目标校准的200个子集先在每个模型内平均；误差线不表示目标测试置信区间。源与目标校准使用完全相同的正确视频分组TEST和分数，阈值变化不改变AUC。'
        '图中为预先指定的源CAL与目标n20主比较；默认阈值和10/40/80预算保存在完整Source Data，80预算不可用处保留NA。\n')
    base.with_suffix('.caption.zh.md').write_text(caption)
    if any(digest(args.root/name)!=expected for name,expected in data['sources'].items()):
        raise ValueError('strict summary inputs changed during render')
    record=dict(status='RENDERED_AWAITING_PDF_AND_VISUAL_QA',backbones=data['backbones'],training_seeds=list(SEEDS),
        source_sha256=data['sources'],script_sha256=digest(__file__),alignment=alignment['verdict'],
        full_calibration_rows=len(full),plotted_per_seed_rows=sum(len(q) for q in selected),
        plotted_official_summary_rows=len(saved_summary),models_inferred=False,thresholds_recomputed=False,
        displayed_statistics='Official mean and sample SD across five model estimates; no CAL-subset pooling')
    contract = Path(__file__).parent/'contracts/D_formal_operating_points_contract.json'
    record['contract_sha256'] = digest(contract)
    (output/'render_receipt.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
