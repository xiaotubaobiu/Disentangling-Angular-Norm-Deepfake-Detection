"""Report the user-requested FF++ T1–T5 diagnostic after Stage B failure."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

PHASE = Path(__file__).resolve().parents[1]


def class_diagnostics(split, expected):
    root = PHASE / 'results/t1_t5_exploratory'
    frames = pd.concat([pd.read_csv(root / f'semantic_{split}_shard{i}.csv')
                        for i in range(4)], ignore_index=True)
    keys = ['video_key', 'task_id', 'variant_id', 'question_id']
    videos = frames.groupby(keys, sort=False, dropna=False).agg({
        'method_family': 'first', 'target_semantic': 'first',
        'p_yes': 'mean', 'p_camera': 'mean', 'p_swap': 'mean', 'p_reenact': 'mean',
    }).reset_index()
    t4 = videos[videos.task_id.eq('T4')]
    t4_prediction = t4.p_yes.ge(0.5).map({True: 'yes', False: 'no'})
    t4_detail = {name: {'count': int(t4.target_semantic.eq(name).sum()),
                        'recall': float(t4_prediction[t4.target_semantic.eq(name)].eq(name).mean())}
                 for name in ('yes', 'no')}
    assert abs(sum(item['recall'] for item in t4_detail.values()) / 2 -
               expected['T4']['value']) < 1e-12
    t5 = videos[videos.task_id.eq('T5')]
    t5_prediction = t5[['p_camera', 'p_swap', 'p_reenact']].idxmax(axis=1).str.removeprefix('p_')
    t5_detail = {name: {'count': int(t5.method_family.eq(name).sum()),
                        'predicted_count': int(t5_prediction.eq(name).sum()),
                        'recall': float(t5_prediction[t5.method_family.eq(name)].eq(name).mean())}
                 for name in ('camera', 'swap', 'reenact')}
    assert abs(sum(item['recall'] for item in t5_detail.values()) / 3 -
               expected['T5']['value']) < 1e-12
    return {'t4': t4_detail, 't5': t5_detail}


def main():
    stage_b = json.loads((PHASE / 'results/stage_b/stage_b_result.json').read_text())
    diagnostic = json.loads((PHASE / 'results/t1_t5_exploratory/stage_c_result.json').read_text())
    training = json.loads((PHASE / 'results/training/complete.json').read_text())
    gate = json.loads((PHASE / 'results/summary/e5_a_gate_status.json').read_text())
    if stage_b['stage_b_status'] != gate['status'] or gate['status'] != 'FAIL':
        raise ValueError('this report requires the retained Stage B FAIL qualification')
    if diagnostic['mode'] != 'post_stage_b_failure_diagnostic':
        raise ValueError('unexpected diagnostic mode')
    if stage_b['checkpoint_sha256'] != diagnostic['checkpoint_sha256'] or (
            stage_b['checkpoint_sha256'] != training['best_checkpoint_sha256'] or
            stage_b['source_sha256'] != diagnostic['source_sha256']):
        raise ValueError('diagnostic used a different checkpoint or FF++ source')

    seen = diagnostic['task_metrics']['train']
    heldout = diagnostic['task_metrics']['heldout']
    class_detail = {'seen': class_diagnostics('train', seen),
                    'heldout': class_diagnostics('heldout', heldout)}
    (PHASE / 'results/t1_t5_exploratory/class_diagnostics.json').write_text(
        json.dumps(class_detail, ensure_ascii=False, indent=2) + '\n')
    output = PHASE / 'results/t1_t5_exploratory/t1_t5_metrics.csv'
    with output.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['task', 'metric', 'seen', 'heldout'])
        writer.writerow(['T2', 'balanced_accuracy', stage_b['conditions']['train']['t2_ba'],
                         stage_b['conditions']['heldout']['t2_ba']])
        for task in ('T1', 'T3', 'T4', 'T5'):
            writer.writerow([task, seen[task]['metric'], seen[task]['value'], heldout[task]['value']])

    table = '\n'.join(
        f"| {task} | {metric} | {seen_value:.2%} | {heldout_value:.2%} |"
        for task, metric, seen_value, heldout_value in (
            ('T1', 'AUROC', seen['T1']['value'], heldout['T1']['value']),
            ('T2', '平衡准确率', stage_b['conditions']['train']['t2_ba'],
             stage_b['conditions']['heldout']['t2_ba']),
            ('T3', 'AUROC', seen['T3']['value'], heldout['T3']['value']),
            ('T4', '平衡准确率', seen['T4']['value'], heldout['T4']['value']),
            ('T5', '三类宏召回', seen['T5']['value'], heldout['T5']['value']),
        ))
    report = f'''# E5-A：FF++ T1–T5 完整诊断结果

**原 E5-A qualification 结论仍为 FAIL**：Stage B 未见问法的极性一致率 {stage_b['conditions']['heldout']['polarity_consistency']:.2%}，未达到预设的 90% 门槛。用户随后要求完成 T1–T5，因此本报告是门槛失败后的诊断性扩展，不回溯修改原资格判定。

## 方法与范围

- 使用同一个 seed 42 BridgeTower 检查点，SHA256 `{diagnostic['checkpoint_sha256']}`；训练和选模未改，最佳 FF++ 验证候选 CE 为 {training['best_validation_ce']:.6f}。
- 沿用冻结 FF++ 测试物理划分、原 T1–T5 seen/held-out 模板和视频级帧概率聚合。每个语言集合及任务覆盖 3,049 个物理视频；只使用 GPU 0–3。
- T2 来自已完成的 Stage B；T1/T3/T4/T5 来自本次诊断。没有运行 CDF、重新训练、另选检查点或 E5-B/C/D。

## FF++ 结果

| 任务 | 指标 | seen | held-out |
|---|---|---:|---:|
{table}

T1/T3 是视频 AUROC；T2/T4 是平衡准确率；T5 是 camera/swap/reenact 三类宏召回。这些指标不能直接互相比较为同一种“准确率”。

T5 held-out 的逐类召回为 camera {class_detail['heldout']['t5']['camera']['recall']:.2%}、swap {class_detail['heldout']['t5']['swap']['recall']:.2%}、reenact {class_detail['heldout']['t5']['reenact']['recall']:.2%}。因此尽管宏召回为 {heldout['T5']['value']:.2%}，swap 类仅有 {class_detail['heldout']['t5']['swap']['recall']:.2%} 被正确识别，存在明显类别失衡。T4 held-out 的 Yes/No 召回分别为 {class_detail['heldout']['t4']['yes']['recall']:.2%}/{class_detail['heldout']['t4']['no']['recall']:.2%}。逐类计数见 `results/t1_t5_exploratory/class_diagnostics.json`。

## 语义关系检查

- T2 未见问法极性一致率：{stage_b['conditions']['heldout']['polarity_consistency']:.2%}；预设门槛 90%，失败。
- T4 未见问法中，匹配伪造家族的问题得分高于反事实家族问题：{diagnostic['t4_matched_over_counterfactual']:.2%}（{diagnostic['t4_family_pairs']:,} 个物理视频候选配对）。
- 该 T4 家族排序低于原 E5-A 设定的 75% 参考门槛；这是失败后补做的诊断结果，不改变已确定的资格判定。
- 本次 T1/T3/T4/T5 候选换序语义预测一致率：{diagnostic['permutation_consistency']:.2%}（{diagnostic['permutation_comparisons']:,} 次帧级请求），最大概率差 {diagnostic['permutation_max_probability_error']:.8g}。T2 held-out 换序一致率 {stage_b['conditions']['heldout']['permutation_consistency']:.2%}。

逐帧预测、四分片校验文件、汇总 JSON 与 CSV 位于 `results/t1_t5_exploratory/`。原 E5-A FAIL 报告和 gate 文件位于 `reports/e5_a_final_report_zh.md` 与 `results/summary/`。
'''
    path = PHASE / 'reports/e5_a_t1_t5_exploratory_zh.md'
    path.write_text(report)
    print(json.dumps({'report': str(path), 'csv': str(output), 'status': 'exploratory_complete'}))


if __name__ == '__main__':
    main()
