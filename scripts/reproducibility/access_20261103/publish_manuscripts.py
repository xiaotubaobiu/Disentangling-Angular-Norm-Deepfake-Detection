"""Replace revised H primary prose only after mixed-root tables are accepted."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

import pandas as pd

BONES = ['clip_vit_b16', 'dinov2_vit_b14', 'efficientnet_b0']
NAMES = dict(zip(BONES, ['CLIP', 'DINOv2', 'EfficientNet-B0']))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def one(frame, **keys):
    for key, value in keys.items():
        frame = frame[frame[key].eq(value)]
    if len(frame) != 1:
        raise ValueError(f'expected one accepted table row: {keys}, found {len(frame)}')
    return frame.iloc[0]


def pm(row, signed=False, latex=False):
    mean, sd = row['estimate_mean'] * 100, row['estimate_sample_sd'] * 100
    first = f'{mean:+.4f}' if signed else f'{mean:.4f}'
    return f'${first}\\pm{sd:.4f}$' if latex else f'{first}±{sd:.4f}'


def mdtable(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|',
                      *['|' + '|'.join(map(str, row)) + '|' for row in rows]])


def textable(headers, rows, label, caption, environment='table*'):
    return ('```{=latex}\n\\begin{' + environment + '}[t]\n\\centering\n\\caption{' + caption + '}\n\\label{' + label + '}\n'
            '\\setlength{\\tabcolsep}{5pt}\n\\footnotesize\n\\begin{tabular}{' + 'l' * len(headers) + '}\n\\toprule\n'
            + ' & '.join(headers) + ' \\\\\n\\midrule\n'
            + '\n'.join(' & '.join(map(str, row)) + ' \\\\' for row in rows)
            + '\n\\bottomrule\n\\end{tabular}\n\\end{' + environment + '}\n```')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--old-root', type=Path, required=True)
    p.add_argument('--obsidian', type=Path, required=True)
    args = p.parse_args()
    root, old, obs = args.root.resolve(), args.old_root.resolve(), args.obsidian.resolve()
    code = Path(__file__).resolve().parents[3]
    paper = code / 'paper/ieee-access'
    folder = root / 'reports/final_formal_33'
    receipt = json.loads((folder / 'complete.json').read_text())
    assert receipt['status'] == 'ACCEPTED_FULL_33_MODEL_TABLE_ASSEMBLY'
    assert receipt['expected_H_epoch_budgets'] == dict.fromkeys(BONES, 10)
    for name, expected in receipt['output_sha256'].items():
        assert sha(folder / name) == expected
        if name.startswith('D_'):
            assert sha(folder / name) == sha(old / 'reports/final_formal_33' / name)
    frames = {name: pd.read_csv(folder / name, float_precision='round_trip') for name in [
        'H_primary_AUC_summary.csv', 'H_primary_paired_effects.csv', 'H_cross_backbone_gap_changes_per_seed.csv',
        'D_absolute_AUC_summary.csv', 'D_operating_points_all_budgets_wide.csv', 'formal_model_selection.csv']}
    hm, hc = frames['H_primary_AUC_summary.csv'], frames['H_primary_paired_effects.csv']
    sel = frames['formal_model_selection.csv']
    fresh = sel[sel.route.eq('H') & sel.backbone.isin(['dinov2_vit_b14', 'efficientnet_b0'])]
    assert len(fresh) == 12 and fresh.max_epochs.eq(10).all() and fresh.actual_epochs.eq(10).all()
    assert set(fresh.seed) == {7, 42, 123}
    selected = {b: [int(one(fresh, backbone=b, seed=s, history='B').best_epoch) for s in [7,42,123]]
                for b in ['dinov2_vit_b14', 'efficientnet_b0']}
    for b, epochs in selected.items():
        assert epochs == [int(one(fresh, backbone=b, seed=s, history='AB').best_epoch) for s in [7,42,123]]
    selected_zh = '；'.join(NAMES[b] + '：' + '/'.join(map(str,epochs)) + '轮' for b,epochs in selected.items())
    selected_en = ' '.join(NAMES[b] + ': paired epochs ' + ', '.join(map(str,epochs)) +
        ' for seeds 7, 42, and 123, respectively.' + (' All paired optima lie at the cap; a source-validation plateau is not established within this window.' if epochs == [10,10,10] else '')
        for b,epochs in selected.items())
    auc = lambda b, d, cohort, cell, agg='mean_logit': one(hm, backbone=b, domain=d, cohort=cohort, cell=cell, aggregation=agg)
    effect = lambda b, d, cohort, term, agg='mean_logit': one(hc, backbone=b, domain=d, cohort=cohort, comparison=term, aggregation=agg)
    fixed_md, fixed_tex, common_md, interactions_md = [], [], [], []
    for b in BONES:
        for d, dn in [('source_test', 'FF++'), ('target_test', 'DFD')]:
            a, z = auc(b, d, 'all_B', 'H00'), auc(b, d, 'all_B', 'H10')
            e = effect(b, d, 'all_B', 'H10-H00')
            fixed_md.append([NAMES[b], dn, pm(a), pm(z), pm(e, True)])
            fixed_tex.append([NAMES[b], dn, pm(a, latex=True), pm(z, latex=True), pm(e, True, True)])
        common_md.append([NAMES[b], *[pm(auc(b, 'target_test', 'common', cell)) for cell in ['H00', 'H01', 'H10', 'H11']]])
        es = [effect(b, 'target_test', 'common', term) for term in ['H01-H00', 'H11-H10', '(H11-H10)-(H01-H00)']]
        interactions_md.append([NAMES[b], *[pm(e, True) for e in es], f"{int(es[-1].n_positive)}/{int(es[-1].n_negative)}"])
    dino_fixed = effect('dinov2_vit_b14', 'target_test', 'all_B', 'H10-H00')
    dino_inter = effect('dinov2_vit_b14', 'target_test', 'common', '(H11-H10)-(H01-H00)')
    eff_fixed = effect('efficientnet_b0', 'target_test', 'all_B', 'H10-H00')
    eff_input = [effect('efficientnet_b0', 'target_test', 'common', term)
                 for term in ['H01-H00', 'H11-H10', '(H11-H10)-(H01-H00)']]
    mp = [effect(b, 'target_test', 'common', '(H11-H10)-(H01-H00)', 'mean_probability') for b in BONES]
    ranks = frames['H_cross_backbone_gap_changes_per_seed.csv']
    assert len(ranks) == 108
    rank_count = int(ranks.strict_rank_flip.sum())
    rank_family = ranks[ranks.backbone_a.eq('clip_vit_b16') & ranks.backbone_b.eq('efficientnet_b0')
        & ranks.domain.eq('target_test') & ranks.aggregation.eq('mean_logit') & ranks.effect.eq('test_input_M2')]
    assert len(rank_family) == 3 and set(rank_family.seed) == {7,42,123}
    witness = one(rank_family, seed=123)
    cells = pd.read_csv(folder / 'H_per_seed_metrics.csv', float_precision='round_trip')
    witness_values = [float(one(cells, backbone=b, domain='target_test', cohort='common', cell=cell,
        aggregation='mean_logit', threshold_kind='default', seed=123).pooled_auc) * 100
        for b in ['clip_vit_b16', 'efficientnet_b0'] for cell in ['H10','H11']]
    flips = int(rank_family.strict_rank_flip.sum())
    witness_zh = (f'固定保留原指定的跨骨干对照：同为A→B训练的seed123模型，在同一90真/1,271假DFD共同视频上，'
        f'CLIP在B/A→B输入的AUC为{witness_values[0]:.4f}%/{witness_values[1]:.4f}%，EfficientNet-B0为'
        f'{witness_values[2]:.4f}%/{witness_values[3]:.4f}%。CLIP减EfficientNet的差从{100*witness.gap_before:+.4f}变为'
        f'{100*witness.gap_after:+.4f}个百分点；该指定M2比较三个种子中有{flips}个严格点估计反转。')
    witness_en = (f'The originally specified cross-backbone comparison is retained with the revised weights. '
        f'For AB-trained seed123 models on the same 90 real/1,271 fake DFD videos, CLIP scores '
        f'{witness_values[0]:.4f} percent under B and {witness_values[1]:.4f} percent under AB; EfficientNet-B0 scores '
        f'{witness_values[2]:.4f} and {witness_values[3]:.4f} percent. The CLIP-minus-EfficientNet gap changes from '
        f'${100*witness.gap_before:+.4f}$ to ${100*witness.gap_after:+.4f}$ points. '
        f'This specified M2 comparison has {flips} strict point-estimate reversals among three seeds.')
    target_rank_flips = int(ranks[ranks.domain.eq('target_test') & ranks.effect.isin(['test_input_M1','test_input_M2'])].strict_rank_flip.sum())
    source_dino = auc('dinov2_vit_b14', 'source_test', 'all_B', 'H00')
    h3 = f'''### 3.2 训练历史与测试处理共同影响检测性能和模型比较

相同B部署输入下，训练处理历史的影响随骨干变化。CLIP在DFD的H10−H00为-1.5801±0.3008个百分点，三个种子均下降；本轮十轮重跑的EfficientNet-B0为{pm(eff_fixed, True)}，{int(eff_fixed.n_positive)}正{int(eff_fixed.n_negative)}负；DINOv2为{pm(dino_fixed, True)}，{int(dino_fixed.n_positive)}正{int(dino_fixed.n_negative)}负。下表使用Raw推理和mean-logit聚合；±为三个配对训练种子的样本标准差。

{mdtable(['骨干','测试域','M1→B AUC，%','M2→B AUC，%','配对差H10−H00，百分点'], fixed_md)}

在相同共同成功的DFD队列上进行四格测试，得到以下结果。四格AUC单位为%，差值单位为百分点；每项差值均先在同一种子内计算。

{mdtable(['骨干','H00：M1→B','H01：M1→A→B','H10：M2→B','H11：M2→A→B'], common_md)}

{mdtable(['骨干','M1输入切换，百分点','M2输入切换，百分点','四格交互，百分点','交互正/负种子'], interactions_md)}

mean-prob的平均交互分别为CLIP {pm(mp[0], True)}、DINOv2 {pm(mp[1], True)}、EfficientNet-B0 {pm(mp[2], True)}个百分点，相应正/负种子为{int(mp[0].n_positive)}/{int(mp[0].n_negative)}、{int(mp[1].n_positive)}/{int(mp[1].n_negative)}、{int(mp[2].n_positive)}/{int(mp[2].n_negative)}。输入历史效应与交互须保留骨干、种子和聚合条件。

{witness_zh}全部108项逐种子对照共有{rank_count}项严格点估计反转，完整家族全部保留。比较始终使用同一队列和注册配方，不作为架构单因子效应。

![三骨干预处理历史效应](../../../attachments/deepfake-detector/期刊实验_20261102/H10_H_training_history_all_backbones.png)

**图3｜完整训练历史改变固定部署性能及对测试输入变化的响应。** a，全部B成功队列上的H10−H00。b，共同成功队列上的(H11−H10)−(H01−H00)。M1为B训练、M2为A→B训练；采用Raw推理和mean-logit。种子7、42、123分别显示；菱形与误差条为配对差均值±样本标准差。三个骨干训练上限均为十轮；DINOv2和EfficientNet为本轮重训结果，两面板分别保留自己的固定队列。

'''
    backup = root / 'operations/obsidian_before_H10'
    zh = (backup / '中文稿.md').read_text()
    original_d = zh[zh.index('### 3.3'):zh.index('## 4. 讨论')]
    for n in [4, 3, 2, 1]:
        zh = zh.replace(f'**图{n}｜', f'**图{n + 1}｜')
    zh = zh.replace('训练轮数由独立开发种子1007、1042确定', '初始训练预算由独立开发种子1007、1042开发，H本轮十轮修订单独记录')
    zh = zh.replace('正式训练使用连续5轮不改善的早停规则。', '除本轮H-DINOv2和H-EfficientNet固定训满10轮外，正式训练使用连续5轮不改善的早停规则。')
    begin = zh.index('预处理路线单独以源验证确定预算')
    end = zh.index('\n\nD路线每epoch', begin)
    zh = zh[:begin] + f'''预处理路线三个骨干最大预算统一为10轮。CLIP保留原十轮模型；本轮将DINOv2及EfficientNet-B0分别从公开预训练重新训满10轮，学习率5×10⁻⁵和10⁻⁴、batch128及其余训练设置不变，余弦调度按10轮总预算重新计算。预算内仍用共同源DEV-B选最佳EMA，两历史共享选点；seed7/42/123选点为{selected_zh}，十二个新模型实际均完成10轮。本修订发生在观察初版H目标结果之后，明确记录这一时间顺序；十轮轨迹不逐轮评价DFD，不依据DFD重新选checkpoint。旧长预算产物保留于历史归档，当前主结果全部使用上述新模型。

![开发阶段源域与CDF开发AUC曲线](../../../attachments/deepfake-detector/期刊实验_20261102/D_development_AUC_common_budgets.png)

**图1｜源验证与隔离CDF开发集的逐轮AUC。** 三列为CLIP、EfficientNet-B0、DINOv2，两行为开发种子1007、1042。使用EMA、Raw、正确视频身份和mean-prob；源数据为DF40-FF，目标为隔离的DF40-CDF开发集。每条轨迹保留实际完成轮数。DINOv2的源最佳为22/20轮，CDF最佳为5/1轮；其余预算确认曲线另附。图中没有正式TEST逐轮选点。
''' + zh[end:]
    zh = zh.replace('初始EfficientNet结果用于触发预先约定的骨干扩展，DFD不用于选择H训练轮数或校准阈值。', '初始EfficientNet结果用于触发预先约定的骨干扩展；H-DINOv2/EfficientNet十轮预算修订的时间顺序见2.2节。预算内源选点及H阈值校准不使用DFD。')
    begin, end = zh.index('### 3.2'), zh.index('### 3.3')
    zh = zh[:begin] + h3 + zh[end:]
    zh = zh.replace('H-DINOv2在该配方下的源测试AUC约79.5%', f'H-DINOv2在本轮十轮配方下B训练的源测试AUC为{source_dino.estimate_mean * 100:.4f}%')
    zh = zh.replace('固定版本](https://github.com/xiaotubaobiu/Disentangling-Angular-Norm-Deepfake-Detection/tree/7233d3b204abb447ab7964839312da05b8b3c9b9)', '原执行版本](https://github.com/xiaotubaobiu/Disentangling-Angular-Norm-Deepfake-Detection/tree/7233d3b204abb447ab7964839312da05b8b3c9b9)；H-DINOv2/EfficientNet十轮固定预算实现见[修订版本](https://github.com/xiaotubaobiu/Disentangling-Angular-Norm-Deepfake-Detection/tree/9b6286cd0cacfd96bcdb43f917f77ec25ae2079c)')
    # Only caption numbers change inside the unchanged D Results block.
    new_d = zh[zh.index('### 3.3'):zh.index('## 4. 讨论')]
    assert new_d.replace('**图4｜', '**图3｜').replace('**图5｜', '**图4｜') == original_d
    zh = zh.replace('不据目标结果重新调参', '不按跨骨干比较的优劣另调各模型；十轮预算修订的时间顺序见2.2节')
    if not target_rank_flips:
        zh = zh.replace('跨骨干配对进一步显示，同一测试输入切换可以使骨干间差距变化，乃至出现种子级AUC点估计排名反转。', '跨骨干配对量化了同一测试输入切换引起的性能差距变化；本轮目标域输入切换比较未出现严格排名反转。')
        zh = zh.replace('并观察到个别跨骨干AUC点估计排名随测试输入历史反转', '跨骨干性能差距也随测试输入历史改变')
    zh = zh.replace('final_manuscript: true', 'final_manuscript: false')
    zh = zh.replace('status: complete_chinese_manuscript', 'status: H_fixed10_results_verified_author_metadata_pending')
    for obsolete in ['63.0385', '65.3210', '79.4973', '79.4814', '2.2825±2.8863', '0.6393±1.0870', '0.9232±0.2897']:
        assert obsolete not in zh, f'obsolete primary H-DINO number remains: {obsolete}'
    dsummary = frames['D_absolute_AUC_summary.csv']
    dc = dsummary[dsummary.domain.eq('target_full') & dsummary.metric.eq('pooled_auc')]
    drows = []
    for b in BONES:
        for grouping in ['correct', 'collision']:
            for agg, an in [('mean_logit', 'mean-logit'), ('mean_probability', 'mean-probability')]:
                a = one(dc, backbone=b, grouping=grouping, aggregation=agg, representation='raw')
                z = one(dc, backbone=b, grouping=grouping, aggregation=agg, representation='l2')
                drows.append([NAMES[b], grouping, an, pm(a, latex=True), pm(z, latex=True), f'${100 * (z.estimate_mean - a.estimate_mean):+.4f}$'])
    cal = frames['D_operating_points_all_budgets_wide.csv']
    crows = []
    for b in BONES:
        for threshold, label in [('source_real', 'Source real'), ('target_real', 'Target real, $n=20$')]:
            q = cal[cal.backbone.eq(b) & cal.domain.eq('target_test') & cal.representation.eq('raw') & cal.aggregation.eq('mean_logit') & cal.threshold_kind.eq(threshold)]
            if threshold == 'target_real':
                q = q[q.budget.eq(20)]
            assert len(q) == 1, (b, threshold, len(q))
            q = q.iloc[0]
            crows.append([NAMES[b], label, *[f'${100*q[k + "_estimate_mean"]:.4f}\\pm{100*q[k + "_estimate_sample_sd"]:.4f}$' for k in ['pooled_auc', 'fpr', 'tpr', 'pooled_ba']]])
    en = (root / 'operations/english_draft_template.qmd').read_text()
    counts = [['DF40-FF training', '611', '12,232'], ['DF40-FF development', '108', '2,175'],
              ['Source real calibration', '140', '0'], ['DF40-FF test', '140', '2,884'],
              ['DF40-CDF development', '20', '420'], ['DF40-CDF ranking', '157', '12,037'],
              ['Target real calibration', '64', '0'], ['DF40-CDF threshold test', '93', '12,037']]
    inserts = {
        'H_SELECTION_SENTENCE': selected_en,
        'H_EFF_FIXED_SENTENCE': f'EfficientNet-B0 changes by {pm(eff_fixed, True, True)} points, with {int(eff_fixed.n_positive)} positive and {int(eff_fixed.n_negative)} negative paired differences.',
        'H_EFF_INPUT_SENTENCE': f'EfficientNet-B0 changes by {pm(eff_input[0], True, True)} and {pm(eff_input[1], True, True)} points for $M_1$ and $M_2$, respectively, giving interaction {pm(eff_input[2], True, True)} points.',
        'H_RANK_WITNESS': witness_en,
        'H_RANK_ABSTRACT': ('paired comparisons can change the ordering of backbones when test preprocessing changes' if target_rank_flips else 'paired comparisons quantify changes in the gaps between backbones when test preprocessing changes'),
        'H_DINO_FIXED_SENTENCE': f'DINOv2 changes by {pm(dino_fixed, True, True)} points, with {int(dino_fixed.n_positive)} positive and {int(dino_fixed.n_negative)} negative paired differences.',
        'H_DINO_INTERACTION_SENTENCE': f'The revised DINOv2 interaction is {pm(dino_inter, True, True)} points, with {int(dino_inter.n_positive)} positive and {int(dino_inter.n_negative)} negative seeds.',
        'H_ALL_RANK_SENTENCE': f'The complete family retains all 108 seed-level comparisons, including {rank_count} strict point-estimate reversals.',
        'H_DINO_DISCUSSION_SENTENCE': f'Under the revised ten-epoch recipe, the B-trained DINOv2 source-test AUC is {source_dino.estimate_mean*100:.4f} percent; the comparison therefore concerns these fitted detectors, including their achieved source learning.',
        'H_FIXED_TABLE': textable(['Backbone','Test domain','$M_1\\to B$ AUC (\\%)','$M_2\\to B$ AUC (\\%)','Paired difference (pp)'], fixed_tex, 'tab:h-fixed', 'Fixed-B deployment AUC and paired training-history differences. Mean $\\pm$ sample SD over three paired seeds; raw features and mean-logit aggregation.'),
        'D_COUNTS_TABLE': textable(['Role','Real videos','Fake videos'], counts, 'tab:d-cohorts', 'Frozen scoring cohorts after eligibility and relationship checks. The ranking cohort includes the 64 calibration real videos; threshold TEST excludes them.', environment='table'),
        'D_EIGHT_CELLS_TABLE': textable(['Backbone','Grouping','Aggregation','Raw AUC (\\%)','L2 AUC (\\%)','Mean difference (pp)'], drows, 'tab:scoring', 'Complete target scoring design. AUC values show five-model mean $\\pm$ sample SD. Differences are L2 minus Raw under the same grouping and aggregation.'),
        'D_OPERATING_TABLE': textable(['Backbone','Calibration','Same-TEST AUC (\\%)','FPR (\\%)','TPR (\\%)','BA (\\%)'], crows, 'tab:operating', 'Raw mean-logit operating points on the same 93 real and 12,037 fake target test videos. Target subset results are averaged within each model before computing five-model mean and sample SD.'),
    }
    for key, value in inserts.items():
        assert en.count('@@' + key + '@@') == 1, key
        en = en.replace('@@' + key + '@@', value)
    assert '@@' not in en
    output = root / 'reports/manuscript_revision'
    output.mkdir(exist_ok=False)
    (output / '中文稿.md').write_text(zh)
    (output / 'manuscript_en.qmd').write_text(en)
    (paper / 'manuscript_en.qmd').write_text(en)
    repository_zh = zh
    for original, local in [
        ('D_development_AUC_common_budgets.png','development_auc.png'),
        ('H_preprocessing_coverage.png','preprocessing_coverage.png'),
        ('H10_H_training_history_all_backbones.png','preprocessing_effects.png'),
        ('FINAL_D_inference_grouping_all_backbones_target_full.png','normalization_grouping.png'),
        ('FINAL_D_operating_points_all_backbones.png','operating_points.png')]:
        repository_zh = repository_zh.replace('../../../attachments/deepfake-detector/期刊实验_20261102/' + original, 'figures/' + local)
    (paper / 'manuscript_zh.md').write_text(repository_zh)
    # Results retain the already audited D sections, replacing H and selection.
    results = (backup / '实验结果.md').read_text()
    a, b = results.index('## 3. RQ1'), results.index('## 4. RQ2')
    results = results[:a] + '## 3. RQ1：DINO与EfficientNet十轮修订后的完整预处理历史结果\n\n' + h3[h3.index('\n\n') + 2:] + '\n' + results[b:]
    results = results.replace('完整正式实验', '完整正式实验的H十轮修订')
    start = results.index('| 路线 | 骨干 | seed |')
    end = results.index('\n\nD每轮', start)
    selection_rows = [[q.route, NAMES[q.backbone], int(q.seed), q.history, int(q.max_epochs), int(q.actual_epochs),
                       int(q.best_epoch), f'{100*q.source_selection_best_score:.4f}', int(q.amp_skips)]
                      for q in sel.itertuples(index=False)]
    results = results[:start] + mdtable(['路线','骨干','seed','训练历史','最大轮数','实际轮数','选中轮数','源选分数%','AMP跳步'], selection_rows) + results[end:]
    results = results.replace('patience5、EMA.999、会议学习率及其余会议训练设置固定。', 'EMA.999、会议学习率及其余会议设置固定；H-DINO与EfficientNet本次训满10轮并在预算内源选最优，其余保持patience5。此次预算修订发生在观察初版结果之后，不能表述为最初未见目标结果的预算决定。')
    results = results.replace('FINAL_formal_model_selection.csv', 'H10_formal_model_selection.csv')
    results = results.replace('开发seed1007/1042决定最大预算；', '初始预算由开发seed1007/1042确定，本次H-DINO与EfficientNet改为用户指定固定十轮；')
    results = results.replace('FINAL_H_', 'H10_H_')
    results = results.replace('下列JSON回执绑定原工件hash，最终稿数值按这些表核查。', '下列回执区分当前修订与保留的D证据。D及H-CLIP来自原产物根，H-DINO与EfficientNet十二模型来自十轮修订根；映射见publication_sources.json。')
    results = re.sub(r'- `final_formal_table_assembly_acceptance.json`：`[^`]+`',
        '- `H10_final_formal_table_assembly_acceptance.json`：`' + sha(root / 'operations/final_formal_table_assembly_acceptance.json') + '`', results)
    figures_receipt = root / 'operations/publication_figure_acceptance.json'
    assert figures_receipt.is_file(), 'accept revised publication figures before generating primary notes'
    results = re.sub(r'- `final_three_backbone_figures_acceptance.json`：`[^`]+`',
        '- `H10_publication_figure_acceptance.json`：`' + sha(figures_receipt) + '`', results)
    # Replace the compact answer with current data instead of stale sign claims.
    a, b = results.index('## 1. 四个问题'), results.index('## 2. 数据')
    results = results[:a] + f'''## 1. 四个问题的答案

1. **预处理历史会改变固定部署表现。** CLIP在DFD下降1.5801±0.3008点，EfficientNet变化{pm(eff_fixed, True)}点，十轮DINO为{pm(dino_fixed, True)}点。四格交互与跨骨干比较见第3节；不能概括为二次处理一律有害。
2. **推理L2效应依赖聚合基线。** DINO五种子在正确mean-logit均下降、mean-prob均提高；以Raw mean-logit共同参照时三个骨干的L2 mean-prob平均差均不为正。
3. **错误分组可造成部分种子反转。** CDF正确负/错误正为CLIP mean-logit 1/5、EfficientNet mean-prob 3/5，DINO两聚合0/5。六个骨干×聚合的五种子均值均无该方向反转；绝对AUC仍下降。
4. **高AUC不能保证阈值可迁移。** 同一CDF阈值TEST上，CLIP源校准误报100%、EfficientNet87.0968%、DINO8.3871%；目标20真实校准提高前两者BA，但DINO因召回损失平均BA下降。

''' + results[b:]
    for obsolete in ['63.0385', '65.3210', '79.4973', '79.4814', '2.2825±2.8863']:
        assert obsolete not in results, f'obsolete result remains outside replaced H section: {obsolete}'
    results = results.replace('H跨骨干反转是指定共同队列、注册配方和seed的AUC点估计反例，不代表总体显著优劣', f'H跨骨干比较保留指定共同队列、注册配方和全部seed；当前108项比较出现{rank_count}项严格点估计反转，不推断总体显著优劣')
    (output / '实验结果.md').write_text(results)
    (output / '实验计划.md').write_text((root / 'operations/实验计划_修订模板.md').read_text())
    old_record = (backup / '实验记录.md').read_text()
    dev_start = old_record.index('|骨干|seed|cap|')
    dev_end = old_record.index('\n\nD-DINO两个seed', dev_start)
    development_table = old_record[dev_start:dev_end]
    summary_rows = []
    for route in ['D', 'H']:
        for b in BONES:
            q = sel[sel.route.eq(route) & sel.backbone.eq(b) & sel.history.eq('B')].sort_values('seed')
            expected = [7, 42, 123, 2026, 3407] if route == 'D' else [7, 42, 123]
            q = q.set_index('seed').loc[expected].reset_index()
            summary_rows.append([route, NAMES[b], '/'.join(map(str, expected)), str(int(q.max_epochs.iloc[0])),
                '/'.join(str(int(x)) for x in q.actual_epochs), '/'.join(str(int(x)) for x in q.best_epoch)])
    fresh_rows = [[NAMES[q.backbone], int(q.seed), q.history, int(q.actual_epochs), int(q.best_epoch),
                   f'{100*q.source_own_mean_probability_video_auc:.4f}', f'{100*q.source_selection_best_score:.4f}',
                   int(q.success), int(q.amp_skips)] for q in fresh.sort_values(['backbone','seed','history']).itertuples(index=False)]
    record = f'''---
title: 期刊实验｜执行记录与H十轮替换
updated: 2026-11-03
status: results_verified_author_metadata_pending
---

# 实验记录

## 1. 当前结果身份

D15模型及H-CLIP六模型保留原预测，原产物根为`{old}`，原执行代码提交7233d3b204abb447ab7964839312da05b8b3c9b9。H-DINO与EfficientNet十二模型在`{root}`重新训练，固定十轮实现提交9b6286cd0cacfd96bcdb43f917f77ec25ae2079c。`publication_sources.json`按骨干绑定原根或新根；所有H模型共享同一全量冻结数据和源冻结数据hash。

本次四篇旧笔记在`operations/obsidian_before_H10`按原字节和SHA备份。原长预算H-DINO与EfficientNet训练、结果和权重仍在原根；它们不进入当前主表、图或稿件。/Dataset原始视频未改动，处理图像沿用原冻结PNG路径，仅复制元数据到新根，不重新做人脸处理。

## 2. 训练资格与固定参数

三骨干均为公开预训练视觉编码器全参数＋二输出线性头。会议学习率CLIP1e-5、DINO5e-5、EfficientNet1e-4，AdamW/CE/WD5e-4、实际batch128、AMP、clip1.0、EMA.999及DF40增强保持。输入均224；A的336是中间预处理分辨率。

此前已核验实际增强输入的batch128前后向，以及连续6步与保存2步后恢复4步的参数、BN、AdamW、scaler、EMA和事件一致性。DINO位置插值保持官方公式并在CPU保留梯度以满足确定性反向；无前向作用的mask_token不用于否定全参数微调。训练资格不等于高AUC保证。

## 3. D轮数开发记录

开发种子1007、1042独立于正式五种子，使用隔离CDF-DEV。候选预算改变cosine总长度，因此均从公开初始化重训，不截取长轨迹冒充短预算。CLIP/EfficientNet根据实际2/5轮确认冻结最大2；DINO保留30轮预算与正式源patience5。

{development_table}

D-DINO源最佳与CDF最佳相差17/19轮，不能概括为两者接近。正式权重仍由各正式种子的源DEV最佳EMA选择，DINO不强制使用第30轮。12条轨迹共102个epoch已归档；主图展示共同预算的6条轨迹，其余确认预算另图保留。开发曲线已加入中英文稿。

## 4. H预算修订与真实选点

CLIP保留原源验证开发确定的10轮预算和模型。H-DINO与EfficientNet分别由原60/15轮预算修订为固定十轮；修订发生在查看初版H结果之后。本轮十二模型均从公开初始化重新开始，余弦总预算改为10，其余设置不变；不从旧权重续训，也不直接使用原轨迹第10轮。

十轮轨迹不提前停止，每轮按两模型在同一B源DEV上的mean-prob AUC均值保留预算内最佳共同EMA。seed7/42/123的选点为{selected_zh}。十轮内部不逐轮评价DFD，不用DFD重新选epoch。

{mdtable(['骨干','seed','历史','实际轮数','共同选中轮数','本模型源DEV-B AUC%','共同源选分数%','成功更新','AMP跳步'], fresh_rows)}

## 5. 33个正式模型的预算与选点

{mdtable(['路线','骨干','seed顺序','最大预算','实际轮数','最佳EMA轮数'], summary_rows)}

H两历史共享选点。完整33行模型身份、源选分数、成功/跳步数、配方hash和最佳权重hash见`H10_formal_model_selection.csv`。保存最佳EMA、最终恢复state、每轮抽样/更新事件和DEV分数；不声称逐轮独立保存全部权重。

## 6. 数据和评价操作

D每自然视频固定8帧，训练102,272次有放回抽样，真实质量0.5、21假方法各0.5/21。CDF开发20真/420假，890个关联候选不进入正式角色；目标CAL64真与阈值TEST93真/12,037假隔离。碰撞组保留全部8k帧，不重新随机取8帧；CDF单位157真/12,037假变为157真/1,830假组。

H固定原视频和8母帧，A=RetinaFace5点336，B=dlib HOG+68点224，A→B读回A的PNG后重新完整运行B。失败不补帧，8帧全部成功且最终RGB不同才合格。共同TRAIN2,951和DEV525视频；每轮23,680次抽样，两模型共享抽样、增强和模型随机操作。固定部署在全部B成功队列，交互在共同成功队列；最终输入都224。处理图像、mask、变换和失败分母保留。

D八格共享特征与原头。默认/源真实/目标真实阈值使用同一TEST；目标20为主、10/40敏感性、80不可用，200嵌套CAL子集先在模型内平均。逐模型计数和方法级结果全部保留。目标关系组不满足门槛的bootstrap区间为NA，训练种子SD不代替独立来源不确定性。

## 7. 验收和论文交付

十二个十轮训练、十二个完整特征导出、六份种子分析、原严格三种子汇总及独立均值/SD/NA核查完成后才生成当前主表。所有D输出表与原版字节一致，H-CLIP原估计未改变；全108项跨骨干对照重新汇总。

H图以新DINO及EfficientNet替换原数据，并核对Source Data、PDF字体、面板和碰撞；D正式图保持原字节，另加入已有开发曲线。中英文稿通过同一当前主表生成。英文采用IEEE Access官方2026-05-13模板，Quarto输出LaTeX后还须独立pdfLaTeX/BibTeX编译并逐页检查；构建和视觉核查回执另附。

作者暂为Shiyu Chen、Kangbin Yim两位，后者通讯。基金、会议前作的题名/会议/年份/引用及两位正式简介待作者补充，不编造条目。完整LaTeX项目放在本笔记目录的`latex/`子目录。已连接[Overleaf项目](https://www.overleaf.com/project/6ae9d9c5fe829b1a7b43ec90)；源文件使用独立Git文档目录同步，认证令牌不进入任何文件，最终上传回执另附。论文讨论保留低源学习、处理成功子集、聚合基线及种子级反转的实际边界。
'''
    (output / '实验记录.md').write_text(record)
    provenance = dict(status='VERIFIED_RESULT_MANUSCRIPTS_GENERATED', table_receipt_sha256=sha(folder / 'complete.json'),
                      selected_revised_H_epochs=selected, revised_H_models=12, fixed_revised_H_epochs=10, strict_rank_flips=rank_count,
                      unchanged_D_result_text=True, unchanged_D_tables_byte_identical=True,
                      files={str(p): sha(p) for p in output.iterdir() if p.is_file()},
                      author_metadata_pending=['funding','conference_predecessor_citation','two_author_biographies'],
                      producer_sha256=sha(__file__), published_to_obsidian=False)
    (output / 'generation_receipt.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in provenance.items() if k != 'files'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
