#!/usr/bin/env python3
"""Sole PROGRESS writer for the fresh, primary-only v2 cohort."""
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
EXP=Path('/home/z/project/deepfake_detection/experiments/2026-09-22_semantic_visual_jev_e2')
RESULTS=EXP/'results'
ROOT=RESULTS/'evaluation_primary_v2'
SEEDS=(42,7,2048)
CONDITIONS=('fixed_h1','seen','heldout','permutation','counterfactual','meaningless_candidates','meaningless_questions')
SOURCE='0529ab9127e142d18afef5f0ffd52c2781ff506e449cded116510afd8e3d0af7'

def collect():
    metrics=sorted(str(p.relative_to(ROOT)) for p in ROOT.glob('seed_*/*/*_metrics.json'))
    status=json.loads((RESULTS/'evaluation_logs/current_run_v2.json').read_text())
    done={s:json.loads((RESULTS/f'progress/seed_{s}_completion.json').read_text()) for s in SEEDS}
    histories={}
    for s in SEEDS:
        with (RESULTS/f'progress/seed_{s}_history.csv').open() as f: histories[s]=list(csv.DictReader(f))
    return metrics,status,done,histories

def render(metrics,status,done,histories):
    count=len(metrics)
    training=[]
    matrix=[]
    provenance=[]
    for s in SEEDS:
        best=min(histories[s],key=lambda x:float(x['val_seen_ce']))
        training.append(f'| {s} | 已完成 | {done[s]["completed_epochs"]} | {best["epoch"]} | {float(best["val_seen_ce"]):.6f} | {float(best["val_seen_accuracy"]):.2%} |')
        matrix.append(f'| {s} | '+ ' | '.join(f'{sum(f"seed_{s}/{c}/{d}_metrics.json" in metrics for c in CONDITIONS)}/7' for d in ('ffpp','cdf'))+' |')
        provenance.append(f'- seed {s}：[逐轮数值](results/progress/seed_{s}_history.csv)、[训练完成核验](results/progress/seed_{s}_completion.json)；checkpoint SHA256 `{done[s]["checkpoint_sha256"]}`。')
    active=', '.join(f'seed {seed}（launcher PID {pid}）' for seed,pid in status.get('active',{}).items()) or '无'
    pending=', '.join(map(str,status.get('pending_seeds',[]))) or '无'
    exits=', '.join(f'seed {seed}: {code}' for seed,code in status.get('launcher_exits',{}).items()) or '尚未退出'
    event=status.get('event','')
    if any(code != 0 for code in status.get('launcher_exits', {}).values()): runtime='seed7 的实际 launcher 已退出 1；rank1/2 在等待 rank0 汇总结果的 NCCL broadcast 时超过默认 600 秒 watchdog 超时。后续 seed2048 已停止派发，seed42 保持原运行并继续收集真实退出。22/42 组不代表完整成功；正在定位并保留全部证据。'
    elif event=='primary_v2_completed': runtime='42 组主域及配套产物齐备，三个实际 launcher 均已退出 0；等待正式完整性审计、gates 和报告。'
    elif event=='primary_v2_failed': runtime='至少一个实际 launcher 非零退出，协调器已保存退出码并停止派发后续种子；已启动的其他调用继续收集真实退出。'
    elif (RESULTS/'evaluation_logs/primary_v2_report_ready.json').exists(): runtime='42 组主域及配套产物齐备，可以开始报告审计；launcher 最终退出仍独立跟踪。'
    else: runtime='全新 v2 主域评估进行中；最多两个四卡 launcher 并发，空出一槽后接续 seed2048。'
    relative=Path(status['run_directory']).relative_to(EXP)
    latest=dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
    return f'''# E2 实验进度

最近核对：{latest}。本页由三种子训练完成证据、v2 正式 metrics 和协调器事件自动更新。

当前目标：验证 E2-Semantic Visual-Jev 的预注册指标。按用户最新范围，只评估 DF40 的 FF++ 来源域与 CDF 来源域；全新 `results/evaluation_primary_v2` 共三种子 × 两域 × 七条件 = 42 组。本次不运行 DFD，包括 DFD permutation。旧 `results/evaluation` 保留用于历史追溯，不复制、不混入 v2。

## 任务清单

| 阶段 | 状态 | 完成依据 / 下一步 |
|---|---|---|
| 1. 实现、预检与数值重复行修复 | 已完成 | 316 CPU tests passed / 33 warnings，退出码 0；独立 review 无 P0/P1/P2；修复提交 `103ab5c8fd7eedf5d2ba5c1a0f9fd24ad14c2107` |
| 2. 冻结视觉特征缓存 | 已完成 | FF++ 97,036 / CDF 311,984 / DFD 103,339 个唯一物理帧；v2 仅读取两个主域样本 |
| 3. 正式三种子训练 | 已完成，3/3 种子 | 全部训练 9 轮、最佳第 4 轮；seen 验证 CE 连续 5 轮未改善提前停止 |
| 4. v2 主域七条件评估 | {count}/42 组 | 全新来源指纹与独立输出根；metrics 最后原子落盘，配套文件同步核对 |
| 5. v2 配对、指纹与完整性审计 | 待 42 组齐备 | 正式审计只接收 v2；旧组不充当新产物 |
| 6. Bootstrap、gates 与报告 | 待 v2 完整审计 | 两主域范围与冻结三域 gate 的差异将在报告中明确记录 |

## 正式训练

| 种子 | 状态 | 完整轮次 | 最佳轮次 | 最佳 seen 验证 CE | 最佳轮次混合验证准确率 |
|---|---|---:|---:|---:|---:|
{chr(10).join(training)}

{chr(10).join(provenance)}

正式 checkpoint 均通过 `evaluate_semantic.validate_evaluation_checkpoint` 核验。原训练 launcher 会话无法追溯时，真实退出码仍记为 null，以最终 rank0 汇总、逐轮四 rank 诊断和实际 checkpoint 佐证完成，不补写退出 0。训练验证准确率混合 T1–T5，不能直接与 H1 二分类指标相减。

历史 cohort 的 G1 已完成三种子 10,000 次配对、标签/方法分层视频 bootstrap：FF++ 来源域 H1/E2 平均 AUROC 为 0.9985308321 / 0.9984874527；CDF 来源域为 0.9112673875 / 0.8221806782；E2−H1 单侧 95% 下界分别为 −0.0002742310 / −0.1026723143，CDF 历史结果未通过 G1。该项是完整三种子历史证据，详见 [原 G1 证据](results/progress/g1_fixed_task_bootstrap.json)；它不替代此次 v2 正式结果。

## v2 主域评估矩阵

每格为已写出正式 metrics 的条件数 / 7。条件为 fixed_h1、seen、heldout、permutation、counterfactual、meaningless_candidates、meaningless_questions。

| 种子 | FF++ 来源域 | CDF 来源域 |
|---|---|---|
{chr(10).join(matrix)}

{runtime}

- 当前运行：{active}；排队种子：{pending}。
- 已观察的实际 launcher 退出码：{exits}。
- [42 组主域及配套产物就绪证据](results/evaluation_logs/primary_v2_report_ready.json)独立于 launcher 退出记录。
- [当前协调器状态](results/evaluation_logs/current_run_v2.json)、[本次命令与逐 rank 日志]({relative}/)、[总控制台](results/evaluation_logs/primary_v2_supervisor_console.log)。每个 seed 保留精确 argv/环境、启动 PID、真实 `Popen.wait()` 退出码。

## 运行与来源证据

- GPU 0–3、每卡 batch128、每个 launcher 四 rank，最多两个同时运行。GPU 4–7 未用于本实验。
- 新 cohort 在 2026-09-23 05:19:34 UTC 启动 seed42 和 seed7；seed2048 自动等候空闲槽。
- v2 evaluation source bundle SHA256：`{SOURCE}`。每个 `run_provenance.json` 绑定冻结 checkpoint、config、语言清单、缓存清单和该来源指纹；缓存来源协议包含三域 manifest，不表示运行了 DFD。
- 旧 seed42 因浮点舍入重复行比较失败，实际 exit 1；修复仅为已有 1e-6 容差字段补上 `value` 别名。旧 seed7 收到受控 SIGTERM 后实际 exit 1，所有旧 rank 已退出。[停止请求](results/evaluation_logs/parallel_20260923T043530Z/seed7_controlled_stop_request.json)、[真实完成记录](results/evaluation_logs/parallel_20260923T043530Z/semantic_controls_seed_7/completion.json)、[GPU 清空前置核验](results/evaluation_logs/parallel_20260923T043530Z/seed7_controlled_stop_completion.json)。
- 旧资源测量显示 seen/CDF 单 run 峰值进程树 PSS 46.01 GiB、单卡显存 5.78 GiB；双 run 观测单卡约 14.5 GiB，按授权维持最多两个。[资源证据](results/evaluation_logs/resource_profile_seed_42_seen_cdf/summary.json)。
- [协议审计与报告限制](reports/protocol_audit_zh.md)；[实验说明](README.md)。
'''

last=None
while True:
    state=collect()
    digest=hashlib.sha256(json.dumps(state,sort_keys=True).encode()).hexdigest()
    if digest!=last:
        temp=EXP/f'PROGRESS.md.{os.getpid()}.tmp'
        temp.write_text(render(*state))
        temp.replace(EXP/'PROGRESS.md')
        print(json.dumps({'updated_at_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'evaluation_groups':len(state[0]),'event':state[1]['event']}),flush=True)
        last=digest
    if state[1].get('event') in ('primary_v2_completed','primary_v2_failed'): break
    time.sleep(10)
