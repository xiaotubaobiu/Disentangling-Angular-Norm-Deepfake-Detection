"""Audit all 72 evaluations before producing the stage-one scientific report."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import subprocess

import numpy as np
import pandas as pd
import torch
import yaml

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(ROOT / 'src')]
from experiment_data import atomic_json, sha256
from experiment_statistics import paired_bootstrap, detection_metrics, task_metric
from evaluation.e2_gates import PRIMARY

ARMS = ('A0', 'A1', 'A2')
DOMAINS = ('ffpp', 'cdf')


def read_videos(arm, condition, domain, paired=False):
    suffix = 'paired_videos' if paired else 'videos'
    return pd.read_csv(PHASE / f'results/evaluation/{arm}/{condition}/{domain}_{suffix}.csv',
        keep_default_na=False, dtype={'video_id': str, 'video_key': str, 'method_id': str})


def table(frame, columns, formatters=None):
    formatters = formatters or {}
    lines = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join(['---'] * len(columns)) + ' |']
    for _, row in frame.iterrows():
        lines.append('| ' + ' | '.join(formatters.get(c, str)(row[c]) for c in columns) + ' |')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    cfg = yaml.safe_load((PHASE / 'configs/stage1.yaml').read_text())
    conditions = cfg['evaluation']['conditions']
    protocol = json.loads((PHASE / 'results/protocol/manifest.json').read_text())
    for path, expected in protocol['files'].items():
        if sha256(path) != expected:
            raise ValueError(f'frozen protocol input changed: {path}')
    summary = PHASE / 'results/summary'
    summary.mkdir(parents=True, exist_ok=True)
    complete, missing, raw_metrics = [], [], []
    for arm in ARMS:
        for domain in DOMAINS:
            for condition in conditions:
                path = PHASE / f'results/evaluation/{arm}/{condition}/{domain}_completion.json'
                if not path.exists():
                    missing.append(f'{arm}/{domain}/{condition}')
                    continue
                info = json.loads(path.read_text())
                fp = info['fingerprint']
                if info.get('completed') is not True or fp.get('smoke') is not False or fp.get('reportable') is not True:
                    raise ValueError(f'nonreportable evaluation: {path}')
                if fp['arm'] != arm or fp['domain'] != domain or fp['protocol_sha256'] != sha256(PHASE / 'results/protocol/manifest.json'):
                    raise ValueError('wrong evaluation identity')
                checkpoint = PHASE / ('results/assets/COCO_150k_SD3_SD2IP.pth' if arm == 'A0'
                                      else f'results/training/{arm}/best.pth')
                if (fp['checkpoint_sha256'] != sha256(checkpoint)
                        or fp['assets_sha256'] != sha256(PHASE / 'results/assets/manifest.json')):
                    raise ValueError('evaluation used a different checkpoint/assets')
                for name, expected in fp['sources'].items():
                    if sha256(name) != expected:
                        raise ValueError(f'evaluation source changed: {name}')
                base = PHASE / f'results/evaluation/{arm}'
                for rel, expected in info['files'].items():
                    if sha256(base / rel) != expected:
                        raise ValueError(f'changed evaluation artifact: {base / rel}')
                split = domain + ('_full' if condition in ('official_binary', 'fixed_h1') else '_semantic')
                if info['physical_videos'] != protocol['splits'][split]['videos'] or info['physical_frames'] != protocol['splits'][split]['frames']:
                    raise ValueError(f'incomplete physical coverage: {arm}/{domain}/{condition}')
                complete.append(f'{arm}/{domain}/{condition}')
                metrics = json.loads((base / condition / f'{domain}_metrics.json').read_text())
                raw_metrics.extend(dict(row, arm=arm) for row in metrics)
    audit = {'complete': len(complete), 'expected': len(ARMS) * len(DOMAINS) * len(conditions),
             'completed_conditions': complete, 'missing_conditions': missing, 'final': not missing}
    atomic_json(summary / 'artifact_audit.json', audit)
    if raw_metrics:
        pd.DataFrame(raw_metrics).to_csv(summary / 'all_recorded_metrics.csv', index=False)
    if args.partial:
        print(json.dumps(audit), flush=True)
        return
    if missing:
        print(json.dumps(audit), flush=True)
        raise RuntimeError('full report requires all 72 evaluation groups')
    subprocess.run([sys.executable, str(PHASE / 'scripts/audit_physical.py')], cwd=PHASE, check=True)
    for arm in ('A1', 'A2'):
        root = PHASE / f'results/training/{arm}'
        info = json.loads((root / 'complete.json').read_text())
        if info.get('completed') is not True or info.get('reportable') is not True or info.get('smoke') is not False:
            raise ValueError('training is not complete/reportable')
        if info['best_checkpoint_sha256'] != sha256(root / 'best.pth') or info['epochs'] != cfg['train']['epochs']:
            raise ValueError('wrong selected checkpoint or budget')
        for path, expected in info['source_sha256'].items():
            if sha256(PHASE / path) != expected:
                raise ValueError('training implementation changed')
    initial1 = torch.load(PHASE / 'results/training/A1/initial.pth', map_location='cpu', weights_only=True)['state']
    initial2 = torch.load(PHASE / 'results/training/A2/initial.pth', map_location='cpu', weights_only=True)['state']
    if not all(torch.equal(value, initial2[key]) for key, value in initial1.items()):
        raise ValueError('arms did not start from identical soft prompts')
    detection, task_rows, negation, intervals, relationships = [], [], [], [], []
    video_sets = {}
    for arm in ARMS:
        for domain in DOMAINS:
            for condition in conditions:
                if condition == 'permutation':
                    perm = json.loads((PHASE / f'results/evaluation/{arm}/{condition}/{domain}_metrics.json').read_text())
                    relationships.extend(dict(row, arm=arm) for row in perm)
                    continue
                videos = read_videos(arm, condition, domain)
                video_sets[arm, domain, condition] = videos
                for task, frame in videos.groupby('task_id', sort=False):
                    primary = task_metric(frame, task)
                    stored = [row for row in raw_metrics if row['arm'] == arm and row['domain'] == domain
                              and row['condition'] == condition and row['task_id'] == task
                              and row.get('method_id') == '__all__' and row['metric'] == PRIMARY[task]]
                    if (len(stored) != 1 or stored[0]['count'] != len(frame)
                            or not np.isclose(stored[0]['value'], primary, atol=1e-10, rtol=0)):
                        raise ValueError(f'primary metric disagrees with video rows: {arm}/{domain}/{condition}/{task}')
                    task_rows.append({'arm': arm, 'domain': domain, 'condition': condition, 'task': task,
                        'primary_metric': PRIMARY[task], 'primary': primary,
                        'ba': task_metric(frame, task, ba=True), 'physical_videos': frame.video_key.nunique(),
                        'video_variant_rows': len(frame)})
                if condition == 'counterfactual':
                    relationships.extend(row for row in raw_metrics if row['arm'] == arm and row['domain'] == domain
                                         and row['condition'] == condition and row['metric'] in
                                         ('polarity_consistency', 'true_family_ordering'))
                if condition in ('official_binary', 'fixed_h1'):
                    if videos.video_key.duplicated().any():
                        raise ValueError('fixed detection duplicated physical videos')
                    detection.append({'arm': arm, 'domain': domain, 'condition': condition, **detection_metrics(videos)})
                if condition == 'negation':
                    for question, frame in videos.groupby('question_id'):
                        negation.append({'arm': arm, 'domain': domain, 'question': question,
                            'ba': task_metric(frame, 'T2'), 'physical_videos': frame.video_key.nunique()})
    detection = pd.DataFrame(detection)
    tasks = pd.DataFrame(task_rows)
    detection.to_csv(summary / 'detection.csv', index=False)
    tasks.to_csv(summary / 'all_conditions_tasks.csv', index=False)
    tasks[tasks.condition.isin(['seen', 'heldout'])].pivot(
        index=['domain', 'task', 'primary_metric'], columns=['arm', 'condition'],
        values=['primary', 'ba']).to_csv(summary / 't1_t5_comparison.csv')
    pd.DataFrame(relationships).to_csv(summary / 'relationship_diagnostics.csv', index=False)
    pd.DataFrame(negation).to_csv(summary / 'negation_by_question.csv', index=False)

    def interval(left, right, metric, label, *, quantile=.05):
        value = paired_bootstrap(left, right, metric, n_resamples=cfg['evaluation']['bootstrap_samples'],
            seed=cfg['evaluation']['bootstrap_seed'], lower_quantile=quantile)
        row = {'comparison': label, 'metric': metric, **value}
        intervals.append(row)
        print(json.dumps(row), flush=True)
        return value

    gates = {}
    for domain in DOMAINS:
        ni = interval(video_sets['A1', domain, 'fixed_h1'], video_sets['A2', domain, 'fixed_h1'], 'auroc', f'{domain}/A2-A1/fixed_h1')
        interval(video_sets['A1', domain, 'official_binary'], video_sets['A2', domain, 'official_binary'], 'auroc', f'{domain}/A2-A1/official_binary')
        gate = {'G1': {'pass': ni['point'] >= -.01 and ni['lower'] >= -.01, 'interval': ni}}
        seen, heldout = [], []
        for task in PRIMARY:
            aa = video_sets['A1', domain, 'heldout'].query('task_id == @task')
            bb = video_sets['A2', domain, 'heldout'].query('task_id == @task')
            interval(aa, bb, task, f'{domain}/A2-A1/heldout/{task}', quantile=.025)
            seen.append(task_metric(video_sets['A2', domain, 'seen'].query('task_id == @task'), task))
            heldout.append(task_metric(bb, task))
        drops = np.array(seen) - np.array(heldout)
        gate['G2'] = {'pass': bool(drops.mean() <= .02 and (drops <= .05).all()),
                      'task_drops': dict(zip(PRIMARY, drops.tolist())), 'macro_drop': float(drops.mean())}
        perm = json.loads((PHASE / f'results/evaluation/A2/permutation/{domain}_metrics.json').read_text())
        perm = {r['metric']: r['value'] for r in perm}
        gate['G3'] = {'pass': perm['max_probability_error'] <= 1e-6 and perm['prediction_consistency'] == 1., **perm}
        controls = pd.read_csv(PHASE / f'results/evaluation/A2/counterfactual/{domain}_controls.csv')
        polarity = float(controls.loc[controls.metric == 'polarity_consistency', 'value'].mean())
        ordering = float(controls.loc[controls.metric == 'true_family_ordering', 'value'].mean())
        gate['G4'] = {'pass': heldout[1] >= .80 and polarity >= .95, 't2_ba': heldout[1], 'polarity_consistency': polarity}
        gate['G5'] = {'pass': heldout[3] >= .65 and heldout[4] >= .60 and ordering >= .80,
                      't4_ba': heldout[3], 't5_macro_ba': heldout[4], 'family_ordering': ordering}
        for matched in (False, True):
            gains = []
            for kind in ('candidates', 'questions'):
                condition = 'meaningless_' + ('matched_' if matched else '') + kind
                paired = read_videos('A2', condition, domain, paired=True)
                for task in PRIMARY if kind == 'candidates' else ('T2', 'T4'):
                    baseline = paired.query('task_id == @task').copy()
                    meaningful = baseline.copy()
                    for column in baseline:
                        if column.startswith('meaningful_') and column.removeprefix('meaningful_') in baseline:
                            meaningful[column.removeprefix('meaningful_')] = baseline[column]
                    metric = 'T2' if matched and task in ('T1', 'T3') else task
                    gain = interval(baseline, meaningful, metric, f'{domain}/meaningful-{condition}/{task}', quantile=.05 if matched else .025)
                    gains.append({'task': task, 'control': kind,
                                  'metric_name': 'balanced_accuracy' if matched and task in ('T1', 'T3') else PRIMARY[task],
                                  'pass': gain['point'] >= .10 and gain['lower'] > 0, **gain})
            gate['length_matched' if matched else 'G6'] = {'pass': all(g['pass'] for g in gains), 'comparisons': gains}
        gates[domain] = gate
    passed = all(gates[d][f'G{i}']['pass'] for d in DOMAINS for i in range(1, 7))
    verdict = '初步支持该单种子协议下的候选决策替换；仍需更多种子确认' if passed else '当前单种子、固定预算下的替换方案未同时通过检测与语义要求'
    atomic_json(summary / 'bootstrap_intervals.json', intervals)
    atomic_json(summary / 'gate_status.json', {'verdict': verdict, 'passed_all_G1_G6': passed, 'domains': gates,
        'scope': 'single seed42; diagnostic subset; bootstrap conditional on this fitted seed',
        'limitations': ['original G3 includes batch regrouping and FP16 numerical sensitivity',
                        'backbone family/vision encoder/capacity cannot be isolated by comparing with E3']})
    benchmark_path = summary / 'benchmark.json'
    if not benchmark_path.exists():
        with (PHASE / 'results/logs/benchmark.log').open('a') as log:
            subprocess.run([sys.executable, '-u', str(PHASE / 'scripts/benchmark.py')], cwd=PHASE,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='4'),
                stdout=log, stderr=subprocess.STDOUT, check=True)
    benchmark = json.loads(benchmark_path.read_text())
    resource_rows = []
    for arm, values in benchmark['arms'].items():
        if values['checkpoint_sha256'] != sha256(PHASE / f'results/training/{arm}/best.pth'):
            raise ValueError('benchmark used different checkpoint')
        resource_rows.append({'arm': arm, 'parameters': values['parameters'],
            'trainable_parameters': values['trainable_parameters'],
            'latency_ms': values['full']['median_latency_ms'],
            'cached_latency_ms': values['cached']['median_latency_ms'],
            'images_per_second': values['full']['serial_images_per_second'],
            'peak_GiB': values['full']['peak_allocated_bytes'] / 1024 ** 3})
    lines = ['# E4 AntifakePrompt 第一阶段结果', '', '**' + verdict + '。**', '',
        'A0=官方prompt checkpoint；A1=词表评分同任务微调；A2=共享标量评分同任务微调。三个模型共用预训练骨干、预处理和测试清单，A1/A2共享训练序列与初始软提示。', '',
        '只做FF++/CDF（DF40派生来源域、相同16方法），仅seed42。固定真假覆盖3049/9901视频、每视频最多4帧；语义诊断覆盖652/690视频、穷举问法组合。区间为10000次配对物理视频bootstrap，不能代替跨训练种子不确定性。', '',
        '微调预算：每组3轮、每轮4096例，共12288次训练输入，并非完整遍历训练库三次。两组均按FF++验证候选CE选模，CDF不参与选择；A1与A2使用不同训练损失，不能直接比较其loss数值。', '',
        '物理路径复核：训练索引18,752个视频键实际对应9,962个视频，其中586个真实视频各带16种方法别名；验证160条视频索引对应156个物理视频（28真，四个真视频在冻结选模CE中权重为2）。这些是重复采样条目，不能作为独立样本计数。两组使用相同索引与权重，未事后改动选模；所有训练/验证/两域测试物理目录交集为0，测试与语义子集无物理重复。详见physical_audit.json。', '',
        '## 固定真假检测', '', 'AUROC、BA与召回以百分数显示；Brier为原始误差。AUROC不是答对率。FF++与CDF的真实视频仅140/3049、178/9901；全部判假也会有95.41%/98.20%的普通准确率，因此必须结合AUROC、BA和两类召回判断。', '',
        table(detection, ['arm', 'domain', 'condition', 'auroc', 'balanced_accuracy', 'real_recall', 'fake_recall', 'brier'],
            {k: lambda x: f'{x * 100:.2f}' for k in ['auroc', 'balanced_accuracy', 'real_recall', 'fake_recall']} | {'brier': lambda x: f'{x:.4f}'}), '',
        '## T1–T5 seen/held-out', '',
        'T1=真假描述二选一；T2=真假问句Yes/No（含极性）；T3=多组真假措辞二选一；T4=换脸/表情驱动家族Yes/No；T5=原片/换脸/表情驱动三分类。', '',
        'T1/T3主指标AUROC，T2/T4为BA，T5为三类macro BA；seen仍是测试视频，只表示表达集合供A1/A2微调使用。A0未做本地微调，沿用同一语言分组标签方便比较。held-out表达未供A1/A2微调使用；部分任务的表达组合数不同，差距不全等于单一措辞变化。下方BA提供统一决策口径，二分类随机基线50%、T5三分类33.33%。', '']
    main_tasks = tasks[tasks.condition.isin(['seen', 'heldout'])]
    lines += [table(main_tasks, ['arm', 'domain', 'condition', 'task', 'primary_metric', 'primary', 'ba'],
                    {'primary': lambda x: f'{x * 100:.2f}', 'ba': lambda x: f'{x * 100:.2f}'}), '',
        '## 全条件与否定诊断', '',
        '无意义条件列出控制本身的表现，不是语义增益；未评估的任务没有行，不能当成零分。逐方法指标、精确计数与其余分类指标保存在all_recorded_metrics.csv。', '',
        '普通无意义文本控制的G6按各任务原主指标计算增益；另列的length_matched补充判定对T1/T3使用BA，其余任务沿用主指标。下表同时保留主指标与BA，避免将长度匹配的BA增益误读成AUROC增益。', '',
        table(tasks, ['arm', 'domain', 'condition', 'task', 'primary', 'ba'],
              {'primary': lambda x: f'{x * 100:.2f}', 'ba': lambda x: f'{x * 100:.2f}'}), '',
        table(pd.DataFrame(negation), ['arm', 'domain', 'question', 'ba'], {'ba': lambda x: f'{x * 100:.2f}'}), '',
        '## 关系语义与候选换序', '',
        table(pd.DataFrame(relationships), ['arm', 'domain', 'metric', 'value', 'count']), '',
        '## 假设判定与限制', '',
        table(pd.DataFrame([{'domain': d, 'gate': g, 'pass': value['pass']} for d, evidence in gates.items() for g, value in evidence.items()]), ['domain', 'gate', 'pass']), '',
        '官方AntifakePrompt检测已使用候选NLL排序，本实验测试的是词表评分/目标换成共享标量评分/候选CE的完整替换。两者同时变化，不能进一步唯一归因到一个线性头。', '',
        'G1检验相对A1的排序非劣性，不是绝对检测能力门槛。如果A1本身较弱，即便G1通过，也不能独立推出模型已能可靠检测或整套语义假设成立。其余门槛、BA与两类召回仍需分别检查。', '',
        '输入审计显示固定真假描述在Vicuna下为6/10个答案token，而官方yes/no为1/1。求和语言似然可能产生候选长度偏好；长度统计本身不能证明失效原因。两种问法都要报告，不能事后换评分规则挑选更好结果。详见candidate_length_audit.json。', '',
        '词表头删除后仍保留7B语言Transformer。原版与改造版都在相同协议评估，优于/劣于某一组仅支持这个组合下的结论；与SmolVLM的差异无法单独证明模型大小、CLIP结构或判断函数是唯一原因。', '',
        'G3原始审计会按候选数量重组batch，因此其严格概率误差还包含FP16数值敏感性；原始数值和门槛保留，额外固定batch诊断不能覆盖或抹去失败。', '',
        '下一步应由A0起点、A2−A1检测区间和held-out语义缺口共同决定：若A0本身检测弱，先核实该骨干对当前人脸伪造域的适用性；若A1可用而A2下降，定位读出与目标替换；若两者检测可用但语言迁移弱，优先检验任务训练与关系语义。第一阶段到此停止，不自动开展语言压缩或RL。', '',
        '## 资源与延迟', '', '同一V100 GPU0、batch=1、两个固定候选、预热20次后计时50次。缓存只免去视觉塔计算，语言模块仍完整执行。吞吐为串行batch=1，不能代表最大批处理吞吐。', '',
        table(pd.DataFrame(resource_rows), ['arm', 'parameters', 'trainable_parameters', 'latency_ms', 'cached_latency_ms', 'images_per_second', 'peak_GiB'],
              {k: lambda x: f'{x:.3f}' for k in ('latency_ms', 'cached_latency_ms', 'images_per_second', 'peak_GiB')}), '',
        '机器可读证据：results/summary/detection.csv、t1_t5_comparison.csv、all_conditions_tasks.csv、all_recorded_metrics.csv、relationship_diagnostics.csv、negation_by_question.csv、bootstrap_intervals.json、gate_status.json、artifact_audit.json、benchmark.json。']
    path = PHASE / 'reports/stage1_report_zh.md'
    path.write_text('\n'.join(lines) + '\n')
    atomic_json(summary / 'report_sources.json', {
        'protocol_sha256': sha256(PHASE / 'results/protocol/manifest.json'),
        'analysis_sources': {str(p.relative_to(PHASE)): sha256(p) for p in
            [Path(__file__), PHASE / 'src/experiment_statistics.py', PHASE / 'scripts/audit_physical.py',
             PHASE / 'scripts/benchmark.py']},
        'shared_metric_sources': {str(p): sha256(p) for p in
            [ROOT / 'src/evaluation/e2_gates.py', ROOT / 'src/evaluation/semantic_eval.py']},
        'outputs': {str(p.relative_to(PHASE)): sha256(p) for p in
            [path, *sorted(summary.glob('*.csv')), summary / 'bootstrap_intervals.json',
             summary / 'gate_status.json', summary / 'benchmark.json']}})
    print('REPORT COMPLETE', path, flush=True)


if __name__ == '__main__':
    main()
