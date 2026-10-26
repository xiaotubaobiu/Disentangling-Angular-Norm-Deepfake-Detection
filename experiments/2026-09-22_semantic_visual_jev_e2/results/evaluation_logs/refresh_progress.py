#!/usr/bin/env python3
"""Refresh the human-readable progress page from existing execution evidence."""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import time

EXP = Path('/home/z/project/deepfake_detection/experiments/2026-09-22_semantic_visual_jev_e2')
RESULTS = EXP / 'results'
DOMAINS = ('ffpp', 'cdf', 'dfd')
CONDITIONS = ('fixed_h1', 'seen', 'heldout', 'permutation', 'counterfactual', 'meaningless_candidates', 'meaningless_questions')
SEEDS = (42, 7, 2048)

def records(seed):
    logs = list((RESULTS / f'training_logs/seed_{seed}').glob('*/attempt_0/0/stdout.log'))
    if logs:
        result = []
        for line in logs[0].read_text().splitlines():
            if line.startswith('{'):
                row = json.loads(line)
                if 'epoch' in row:
                    result.append(row)
        return result
    with (RESULTS / f'progress/seed_{seed}_history.csv').open() as stream:
        return [{**row, 'epoch': int(row['epoch']), 'val_seen_ce': float(row['val_seen_ce']), 'val_seen_accuracy': float(row['val_seen_accuracy'])} for row in csv.DictReader(stream)]

def collect():
    histories = {seed: records(seed) for seed in SEEDS}
    completion = {}
    for seed in SEEDS:
        path = RESULTS / f'progress/seed_{seed}_completion.json'
        completion[seed] = json.loads(path.read_text()) if path.exists() else None
    metrics = sorted(str(path.relative_to(RESULTS / 'evaluation')) for path in (RESULTS / 'evaluation').glob('seed_*/*/*_metrics.json'))
    status_path = RESULTS / 'evaluation_logs/current_run.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    hold_path = RESULTS / 'evaluation_logs/dispatch_hold.json'
    if hold_path.exists():
        status['dispatch_hold'] = json.loads(hold_path.read_text())
    return histories, completion, metrics, status

def render(histories, completion, metrics, status):
    count = len(metrics)
    primary_count = sum(path.endswith(('/ffpp_metrics.json','/cdf_metrics.json')) for path in metrics)
    trained = sum(value is not None for value in completion.values())
    training_lines = []
    evidence_lines = []
    for seed in SEEDS:
        rows = histories[seed]
        best = min(rows, key=lambda row: row['val_seen_ce']) if rows else None
        done = completion[seed]
        state = f'已完成；第 {done["completed_epochs"]} 轮结束' if done else '进行中；最佳值为当前已观察值'
        training_lines.append(f'| {seed} | {state} | {rows[-1]["epoch"] if rows else 0} | {best["epoch"] if best else "—"} | {best["val_seen_ce"]:.6f} | {best["val_seen_accuracy"]:.2%} |' if best else f'| {seed} | {state} | 0 | — | — | — |')
        evidence_lines.append(f'- seed {seed}：[逐轮数值](results/progress/seed_{seed}_history.csv)' + (f'；[完成核验](results/progress/seed_{seed}_completion.json)，checkpoint SHA256 `{done["checkpoint_sha256"]}`。' if done else '；最新轮次以[持久训练日志](results/training_logs/seed_2048/)为准，训练结束后保存最终核验。'))
    matrix = []
    for seed in SEEDS:
        cells = [str(sum(f'seed_{seed}/{condition}/{domain}_metrics.json' in metrics for condition in CONDITIONS)) + '/7' for domain in DOMAINS]
        matrix.append(f'| {seed} | ' + ' | '.join(cells) + ' |')
    completed_fixed = sum('/fixed_h1/' in path for path in metrics)
    current = status.get('event', 'waiting')
    if status.get('dispatch_hold', {}).get('active'):
        runtime = 'seed42 语义评估在 meaningless_candidates/FF++ 的重复行数值比较处退出 1，原退出码与四 rank traceback 已保留；seed7 既有作业的真实完成/退出继续跟踪。按协调指令，不重启 seed42、不启动 seed2048，等待根因修复及来源指纹方案；这不是计划内调度交接。'
    elif current in {'coordinator_failed', 'parallel_supervisor_failed', 'parallel_evaluation_worker_failed'}:
        runtime = '协调器已因异常停止，待定位后接续。错误：`' + status.get('error', '').replace('`','') + '`。'
    elif count == 63:
        runtime = '63 组正式评估产物已齐，正在核对各 launcher 退出码；随后进行完整性审计与 gates。'
    elif current == 'waiting_for_existing_training' or current == 'training_epoch_completed':
        runtime = '已完成种子的 fixed_h1 评估已执行；协调器等待 seed 2048 自然完成并核验后接续。'
    else:
        runtime = '三种子 fixed_h1 已全部完成；语义/控制条件按最多两个并发的四卡评估继续执行，每个 run 仍固定使用 GPU 0–3 与 batch128。'
    stage3 = f'已完成，{trained}/3 种子' if trained == 3 else f'进行中，{trained}/3 种子已完成核验'
    stage4 = f'主域 {primary_count}/42；实际全域 {count}/63'
    latest = dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
    run_dir = status.get('run_directory', '')
    relative_run = str(Path(run_dir).relative_to(EXP)) if run_dir else 'results/evaluation_logs'
    return f'''# E2 实验进度

最近核对：{latest}。本页由持久训练日志、完成核验 JSON、正式 metrics 与协调器事件更新；完整逐轮数值保留 CSV。

当前目标：验证 E2-Semantic Visual-Jev 是否通过预注册 G1–G6。按用户最新要求，报告只需 FF++/CDF 两个主域；三种子 × 两域 × 七条件的 42 组及配套产物齐备后即进入报告阶段，不等待最后 DFD 性能组。DFD permutation 保留用于冻结 G3，既有评估进程继续自然完成。

## 任务清单

| 阶段 | 状态 | 完成依据 / 下一步 |
|---|---|---|
| 1. 现有 H1 对照、E2 实现与预检 | 已完成 | H1 三种子 × FF++/CDF 共 6 份视频预测；CPU 测试 315 passed / 33 warnings，退出码 0；四卡训练/评估 smoke 通过 |
| 2. 三域视觉特征缓存 | 已完成，3/3 域 | FF++ 97,036 / CDF 311,984 / DFD 103,339 个唯一物理帧特征；缓存生成与校验退出码 0 |
| 3. 正式三种子训练 | {stage3} | 最多 15 轮，seen 验证 CE 连续 5 轮未改善提前停止；仅依此选择 checkpoint |
| 4. 同口径真假检测及完整语义评估 | {stage4} | 主域已完成 {primary_count}/42 组；既有作业仍按实际 63 组矩阵跟踪，fixed_h1 已完成 {completed_fixed}/9 组 |
| 5. 配对与结果完整性审计 | 部分已核对，完整审计待执行 | fixed_h1 三种子主域配对与 G1 输入已核验；完整主域语义/控制集合、配对及指纹审计待 42 组齐备 |
| 6. Bootstrap、G1–G6 判定与报告 | G1 已完成；G2–G6 待全主域 | G1 三种子 10,000 次配对视频 bootstrap 已完成，CDF 未通过；42 个主域组及必要审计产物齐备后执行其余 gates 和主域报告，launcher exit 另行追踪 |

## 正式训练

| 种子 | 状态 | 最新完整轮次 | 最佳轮次 | 最佳 seen 验证 CE | 最佳轮次混合验证准确率 |
|---|---|---:|---:|---:|---:|
{chr(10).join(training_lines)}

{chr(10).join(evidence_lines)}

已完成的正式 checkpoint 均通过 `evaluate_semantic.validate_evaluation_checkpoint` 核验。原训练 launcher 观察会话丢失时，退出码如实记录为 null，并以最终 rank0 汇总、所有轮次四 rank 诊断、worker 退出和实际 checkpoint 作为完成证据。

训练验证准确率混合 T1–T5，不直接与 H1 二分类结果相减。三种子同口径 fixed_h1 视频 AUROC 均值：DF40 的 FF++ 来源域 H1 为 0.9985308321、E2 为 0.9984874527；CDF 来源域 H1 为 0.9112673875、E2 为 0.8221806782。10,000 次共享种子的配对、标签/方法分层视频 bootstrap 已完成：E2−H1 的单侧 95% 下界分别为 -0.0002742310 与 -0.1026723143。FF++ 来源域满足 −0.01 非劣界，CDF 来源域未通过 G1；这项结论已覆盖三个种子。详见 [G1 完整 bootstrap 证据](results/progress/g1_fixed_task_bootstrap.json)。G2–G6 等待完整主域语义/控制结果。

## 正式评估矩阵

每格为已经写出正式 metrics 的条件数 / 7。七条件为 fixed_h1、seen、heldout、permutation、counterfactual、meaningless_candidates、meaningless_questions。

| 种子 | FF++ | CDF | DFD |
|---|---|---|---|
{chr(10).join(matrix)}

本次报告限于 FF++/CDF；DFD 的既有性能评估自然完成但不进入用户所需结果，其 permutation 审计保留用于 G3。metrics 写出表示该组产物完成；整个调用成功以独立的 launcher completion JSON 为准。

## GPU 与运行证据

- GPU 0–3：保持 batch128 的四卡训练/评估；GPU 4–7 未使用。
- seed42/7 的 checkpoint 冻结后，获准在 seed2048 继续原训练时提前评估它们的 fixed_h1；启动前每卡剩约 28.3 GiB、系统可用约 314 GiB。未改训练、科学代码、阈值或 checkpoint 选择，测试结果不影响训练。
- {runtime}
- 首个完整 seed42/seen/CDF 资源测量用时 572.24 秒；采样峰值进程树 PSS 46.01 GiB、单卡显存最高 5.78 GiB、主机最低可用 309.81 GiB。基于该余量启用保守并发 2；[资源测量](results/evaluation_logs/resource_profile_seed_42_seen_cdf/summary.json)。
- 并发交接只短暂挂起纯调度父进程，不挂起 torchrun 或评估 rank；seed7 首组真实 metrics 后恢复父进程并保留原 wait 句柄。旧调度器之后拒绝重复 seed7 时如实记录为调度交接，最终依据六个实际 launcher 的退出码与 63 组库存汇总。
- [主域就绪证据](results/evaluation_logs/primary_report_ready.json)在 42 组主域及三个 DFD permutation 审计落盘后生成；这不代表所有 launcher 已退出。
- [当前协调器状态](results/evaluation_logs/current_run.json)、[持久事件与逐调用日志]({relative_run}/)、[总控制台日志](results/evaluation_logs/coordinator_console.log)。每次调用保存精确 argv、环境、逐 rank 日志及 launcher 退出码；不覆盖已完成组。
- [协议审计与报告限制](reports/protocol_audit_zh.md)；完整实验命令见 [README](README.md)。

运行代码提交：`e1d9c32387e11b0bc5a9515e6c33a6d85b5d4539`。
'''

last = None
while True:
    state = collect()
    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    if digest != last:
        page = render(*state)
        temp = EXP / 'PROGRESS.md.tmp'
        temp.write_text(page)
        temp.replace(EXP / 'PROGRESS.md')
        print(json.dumps({'updated_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'evaluation_groups': len(state[2]), 'latest_training_epochs': {seed: rows[-1]['epoch'] for seed, rows in state[0].items() if rows}}, sort_keys=True), flush=True)
        last = digest
    if state[3].get('event') == 'formal_evaluation_completed':
        break
    time.sleep(10)
