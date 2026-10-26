"""Refresh a non-final snapshot from complete, hash-validated evaluations only."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(PHASE.parents[1] / 'src')]
from experiment_statistics import detection_metrics, task_metric
from evaluation.e2_gates import PRIMARY


def main():
    # Reuse the authoritative checkpoint/source/artifact/physical-count checks.
    subprocess.run([sys.executable, str(PHASE / 'scripts/report.py'), '--partial'],
                   cwd=PHASE, check=True, stdout=subprocess.DEVNULL,
                   env=dict(os.environ, OMP_NUM_THREADS='2'))
    summary = PHASE / 'results/summary'
    audit = json.loads((summary / 'artifact_audit.json').read_text())
    if audit['final']:
        print('All evaluation artifacts are present and hash-checked. '
              'Final statistics/report remain owned by the stage1 pipeline; no interim snapshot regenerated.')
        return
    detection, semantic = [], []
    for identity in audit['completed_conditions']:
        arm, domain, condition = identity.split('/')
        if condition not in ('official_binary', 'fixed_h1', 'seen', 'heldout'):
            continue
        root = PHASE / f'results/evaluation/{arm}/{condition}'
        videos = pd.read_csv(root / f'{domain}_videos.csv', keep_default_na=False,
                             dtype={'video_id': str, 'video_key': str, 'method_id': str})
        context = dict(arm=arm, domain=domain, condition=condition)
        if condition in ('official_binary', 'fixed_h1'):
            if videos.video_key.duplicated().any():
                raise ValueError('fixed detection must contain one row per video')
            detection.append(dict(context, **detection_metrics(videos)))
            continue
        if set(videos.task_id) != set(PRIMARY):
            raise ValueError('incomplete T1-T5 condition')
        stored = json.loads((root / f'{domain}_metrics.json').read_text())
        for task, frame in videos.groupby('task_id'):
            value = task_metric(frame, task)
            metric = [r for r in stored if r['task_id'] == task and r.get('method_id') == '__all__'
                      and r['metric'] == PRIMARY[task]]
            if (len(metric) != 1 or metric[0]['count'] != len(frame)
                    or not np.isclose(metric[0]['value'], value, rtol=0, atol=1e-10)):
                raise ValueError('stored task metric differs from canonical video rows')
            semantic.append(dict(context, task=task, metric=PRIMARY[task], value=value,
                                 ba=task_metric(frame, task, ba=True),
                                 physical_videos=frame.video_key.nunique(), video_variant_rows=len(frame)))
    pd.DataFrame(detection).to_csv(summary / 'interim_detection.csv', index=False)
    pd.DataFrame(semantic).to_csv(summary / 'interim_semantic.csv', index=False)
    lines = ['# E4 当前结果快照（第一阶段未完成）', '',
             f'更新时间：{datetime.now(timezone.utc).isoformat()}；已核验 {audit["complete"]}/{audit["expected"]} 组。', '',
             'A0=官方prompt checkpoint；A1=保留词表评分的软提示微调；A2=共享标量评分的软提示及读出头微调。仅seed42、每组3轮×4096训练输入。', '',
             '只评估DF40派生FF++/CDF来源域、相同16种方法。固定真假覆盖3049/9901视频；语义诊断覆盖652/690视频。当前表不是官方原始benchmark复现，也不是完整第一阶段结论。', '',
             '## 固定真假检测', '',
             'official_binary=官方问题Is this photo real [*]?与yes/no；fixed_h1=固定真假描述二选一。仅列出已完成组合。', '',
             '| 模型 | 域 | 问法 | AUROC % | BA % | real recall % | fake recall % |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for row in detection:
        lines.append('| ' + ' | '.join([row['arm'], row['domain'], row['condition']] +
                     [f'{row[k] * 100:.2f}' for k in ('auroc', 'balanced_accuracy', 'real_recall', 'fake_recall')]) + ' |')
    lines += ['', 'FF++/CDF全部判假也会得到95.41%/98.20%的普通准确率，因此必须结合AUROC、BA和两类召回判断。', '',
              '## T1–T5', '',
              'T1=真假描述选择；T2=真假Yes/No及极性；T3=不同真假措辞；T4=换脸/表情驱动家族Yes/No；T5=原片/换脸/表情驱动三分类。', '',
              'T1/T3主指标为AUROC，T2/T4为BA，T5为macro BA。AUROC不是答对率。seen指供A1/A2微调使用的表达集合；全部表格仍来自测试视频。A0未做本地微调，仅沿用语言分组标签。—表示尚未完成，不能当零分。']
    for domain in ('ffpp', 'cdf'):
        for condition in ('seen', 'heldout'):
            group = [r for r in semantic if r['domain'] == domain and r['condition'] == condition]
            if not group:
                continue
            lines += ['', f'### {domain} / {condition}', '',
                      '| 任务 | 主指标 | A0 % | A1 % | A2 % | A0 BA % | A1 BA % | A2 BA % |',
                      '| --- | --- | --- | --- | --- | --- | --- | --- |']
            for task, metric in PRIMARY.items():
                by_arm = {r['arm']: r for r in group if r['task'] == task}
                numbers = [f'{by_arm[a][value] * 100:.2f}' if a in by_arm else '—'
                           for value in ('value', 'ba') for a in ('A0', 'A1', 'A2')]
                lines.append('| ' + ' | '.join([task, metric, *numbers]) + ' |')
    ci_path = summary / 'interim_fixed_h1_ffpp_bootstrap.json'
    if ci_path.exists():
        ci = json.loads(ci_path.read_text())['interval']
        lines += ['', '## 已完成的固定真假配对统计', '',
                  f'FF++ A2−A1 AUROC：{ci["point"] * 100:+.2f}个百分点；10000次配对物理视频bootstrap单侧95%下界{ci["lower"] * 100:+.2f}个百分点。条件于当前seed42；不能代替跨种子不确定性。', '',
                  '相对弱基线的非劣性不等于绝对检测有效，必须结合语义门槛与两类召回。']
    lines += ['', '## 状态与解释边界', '',
              '当前产物只支持已完成条件下的观察。固定全量与语义子集的人群及问法不同；seen/held-out部分任务的表达组合数不同。不能把差异全归因于一个因素。', '',
              'A1/A2保留同一7B语言Transformer，同时改变读出与训练目标；本阶段不能独立证明模型大小、语言推理容量或单一评分头是唯一原因。第一阶段全部评估、控制、配对统计与资源测量完成后再判定，并停止后续实验。', '',
              '证据：results/summary/interim_detection.csv、interim_semantic.csv、artifact_audit.json、all_recorded_metrics.csv；正式最终报告为reports/stage1_report_zh.md。']
    (PHASE / 'reports/current_detection_snapshot_zh.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'validated_conditions': audit['complete'], 'expected_conditions': audit['expected'],
                      'detection_rows': len(detection), 'semantic_task_rows': len(semantic), 'final': False}))


if __name__ == '__main__':
    main()
