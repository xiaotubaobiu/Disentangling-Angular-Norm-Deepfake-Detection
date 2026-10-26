"""Finalize E5-A PASS/FAIL after Stage B and Stage C complete."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE / 'scripts'))
from train import atomic_json, digest


def main():
    b = json.loads((PHASE / 'results/stage_b/stage_b_result.json').read_text())
    c = json.loads((PHASE / 'results/stage_c/stage_c_result.json').read_text())
    if b['stage_b_status'] != 'PASS' or b['checkpoint_sha256'] != c['checkpoint_sha256'] or (
            b['source_sha256'] != c['source_sha256']):
        raise ValueError('Stage C finalization requires passed Stage B and identical sources')
    training = json.loads((PHASE / 'results/training/complete.json').read_text())
    history = json.loads((PHASE / 'results/training/history.json').read_text())
    physical = json.loads((PHASE / 'results/protocol/physical_split_audit.json').read_text())
    loading = json.loads((PHASE / 'results/protocol/model_loading_audit.json').read_text())
    if training['best_checkpoint_sha256'] != b['checkpoint_sha256']:
        raise ValueError('best selected checkpoint differs from evaluation')
    seen = c['task_metrics']['train']
    heldout = c['task_metrics']['heldout']
    drop = {task: seen[task]['value'] - heldout[task]['value'] for task in ('T1', 'T3')}
    gates = {
        'G1_t2_heldout': b['gates']['G1_t2_heldout'],
        'G2_polarity': b['gates']['G2_polarity'],
        'G3_t1_t3': all((seen[task]['value'] >= 0.80 and heldout[task]['value'] >= 0.75 and
                        drop[task] <= 0.10) for task in ('T1', 'T3')),
        'G4_t4': heldout['T4']['value'] >= 0.60 and c['t4_matched_over_counterfactual'] >= 0.75,
        'G5_t5': heldout['T5']['value'] >= 0.55,
        'G6_permutation': (b['gates']['G6_permutation'] and c['permutation_consistency'] == 1.0 and
                           c['permutation_max_probability_error'] <= b['permutation_probability_tolerance']),
    }
    failed = [name for name, passed in gates.items() if not passed]
    status = 'FAIL' if failed else 'PASS'
    gate = {'status': status, 'stage_reached': 'C', 'gates': gates, 'failed_gates': failed,
        't2_heldout_ba': b['conditions']['heldout']['t2_ba'],
        'polarity_consistency': b['conditions']['heldout']['polarity_consistency'],
        't1_seen_auc': seen['T1']['value'], 't1_heldout_auc': heldout['T1']['value'],
        't3_seen_auc': seen['T3']['value'], 't3_heldout_auc': heldout['T3']['value'],
        't4_heldout_ba': heldout['T4']['value'],
        't4_matched_over_counterfactual': c['t4_matched_over_counterfactual'],
        't5_heldout_macro_recall': heldout['T5']['value'],
        'permutation_consistency': min(b['conditions']['heldout']['permutation_consistency'],
                                      c['permutation_consistency']),
        'seen_to_heldout_drop': drop}
    summary = PHASE / 'results/summary'
    summary.mkdir(parents=True, exist_ok=True)
    atomic_json(summary / 'e5_a_gate_status.json', gate)
    atomic_json(summary / 'e5_a_metrics.json', {
        'status': status, 'checkpoint_revision': loading['revision'],
        'checkpoint_sha256': training['best_checkpoint_sha256'],
        'total_parameters': loading['total_parameters'],
        'trainable_parameters': loading['trainable_parameters'],
        'best_validation_candidate_ce': training['best_validation_ce'],
        'training_history': history, 'physical_split_audit': physical,
        'stage_b': b, 'stage_c': c, 'gates': gates})
    with (summary / 'e5_a_metrics.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['stage', 'condition', 'task', 'metric', 'value'])
        for split in ('train', 'heldout'):
            writer.writerow(['B', split, 'T2', 'balanced_accuracy', b['conditions'][split]['t2_ba']])
            writer.writerow(['B', split, 'T2', 'polarity_consistency', b['conditions'][split]['polarity_consistency']])
            for task in ('T1', 'T3', 'T4', 'T5'):
                item = c['task_metrics'][split][task]
                writer.writerow(['C', split, task, item['metric'], item['value']])
        writer.writerow(['C', 'heldout', 'T4', 'matched_over_counterfactual', c['t4_matched_over_counterfactual']])
        writer.writerow(['C', 'all', 'all', 'permutation_consistency', c['permutation_consistency']])
    table = '\n'.join(
        f"| {task} | {seen[task]['metric']} | {seen[task]['value']:.2%} | {heldout[task]['value']:.2%} |"
        for task in ('T1', 'T3', 'T4', 'T5'))
    gate_table = '\n'.join(f"| {name} | {'PASS' if passed else 'FAIL'} |" for name, passed in gates.items())
    conclusion = (
        'BridgeTower 在当前 FF++ qualification protocol 下能够支持 Jev-style shared candidate decision，并同时表现出基础视觉判别能力、held-out language transfer、polarity understanding 和 manipulation-family semantics。因此可以进入下一阶段研究 H1 forensic vision 与该 semantic backbone 的结合。'
        if status == 'PASS' else
        '当前 BridgeTower 方案在完整 FF++ qualification 中未通过全部门槛；根据预先规定的退出条件停止，不进行调参或后续架构实验。')
    report = f'''# E5-A BridgeTower 最终报告

**E5-A STATUS: {status}。** Stage B 全部通过后执行 Stage C；最终失败门槛：{', '.join(failed) if failed else '无'}。

## 1. Experiment objective 与模型

检验已进行判别式视觉语言预训练的 BridgeTower backbone，能否以原生 ITM 共享候选评分学习 FF++ 图像真假与伪造类别，并保留问题极性和未见语言表达的语义理解。

- 检查点：`{loading['repository']}`，revision `{loading['revision']}`；RoBERTa slow BPE tokenizer revision `{loading['tokenizer_revision']}`，词表与 BridgeTower 附带词表相同。文件哈希见 `results/protocol/model_loading_audit.json`。
- 架构：每个候选与问题组成完整文本，分别经过同一 BridgeTower，取原生 ITM match logit；所有候选共享权重，softmax 后用候选交叉熵训练。无候选专属分类器、生成任务或新增大 MLP。
- 总参数 {loading['total_parameters']:,}；可训练 {loading['trainable_parameters']:,}。冻结 `vision_model`、`text_model`；仅训练跨模态/bridge 模块和原生 ITM head。

## 2. 数据、泄漏审计与训练

只使用 DF40 派生 FF++ 来源域。物理视频：训练 {physical['splits']['train']['physical_videos']:,}、验证 {physical['splits']['val']['physical_videos']:,}、测试 {physical['splits']['ffpp_full']['physical_videos']:,}；测试中真实 {physical['splits']['ffpp_full']['real_videos']:,}、伪造 {physical['splits']['ffpp_full']['fake_videos']:,}。训练/验证/测试物理视频交集均为 0；测试每视频最多 4 帧，先按视频平均概率再计分。

仅 seed 42。T1–T5 每轮各约 20%；T1–T3 真假平衡，T4 Yes/No 平衡，T5 三家族平衡。详细逐轮计数见 `results/protocol/sampling_audit.json`。最多 5 轮，每轮 4096 个输入，FP32；训练和验证历史见 `results/training/history.json`，各轮无跳过优化。最佳检查点只按 FF++ validation candidate CE 选择，第 {min(history, key=lambda x: x['val_candidate_ce'])['epoch']} 轮，CE={training['best_validation_ce']:.6f}。错误分词和 T4 采样的早期未完成运行在测试前被隔离，没有用于本结论。

## 3. Stage B killer test

| 条件 | T2 视频变体平衡准确率 | 极性一致率 | 候选换序语义一致率 |
|---|---:|---:|---:|
| seen | {b['conditions']['train']['t2_ba']:.2%} | {b['conditions']['train']['polarity_consistency']:.2%} | 未测试 |
| held-out | {b['conditions']['heldout']['t2_ba']:.2%} | {b['conditions']['heldout']['polarity_consistency']:.2%} | {b['conditions']['heldout']['permutation_consistency']:.2%} |

Stage B 门槛：held-out T2 BA ≥70%，极性一致率 ≥90%，换序一致率 100% 且最大语义概率误差 ≤{b['permutation_probability_tolerance']:.0e}；全部通过后才执行 Stage C。

## 4. Stage C FF++ T1–T5

| 任务 | 指标 | seen | held-out |
|---|---|---:|---:|
| T2 | balanced accuracy | {b['conditions']['train']['t2_ba']:.2%} | {b['conditions']['heldout']['t2_ba']:.2%} |
{table}

T1/T3 seen→held-out 降幅分别为 {drop['T1']:.2%}/{drop['T3']:.2%}。T4 匹配家族问题分数高于反事实家族的比例为 {c['t4_matched_over_counterfactual']:.2%}（{c['t4_family_pairs']:,} 个视频候选配对）。Stage C 候选换序语义一致率 {c['permutation_consistency']:.2%}（{c['permutation_comparisons']:,} 次），最大概率误差 {c['permutation_max_probability_error']:.8g}。T1/T3 为视频 AUROC，T2/T4 为平衡准确率，T5 为三类宏召回；这些指标不能互相替代。

## 5. Gates 与最终判定

| Gate | 状态 |
|---|---|
{gate_table}

**E5-A STATUS: {status}。** {conclusion} 现在可以退出，因为 Stage B、Stage C 的全部预定指标和 G1–G6 已在同一检查点、同一 FF++ 物理测试划分上完成并按门槛裁决。没有运行 CDF、压缩、RL 或 E5-B/C/D。

单种子结果不覆盖重新训练种子的随机性；FF++ 与训练共享伪造方法，不能解释为未见方法泛化。最初 checkpoint 文件的 Fast tokenizer 在本环境会逐字符分词；正式运行使用官方 slow BPE，所有问题候选输入的最大长度为 31 token。视觉与语言关系能力的因果来源仍需后续受控研究，当前只判定资格是否通过。
'''
    path = PHASE / 'reports/e5_a_final_report_zh.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    print(json.dumps({'status': status, 'failed_gates': failed, 'report': str(path)}))


if __name__ == '__main__':
    main()
