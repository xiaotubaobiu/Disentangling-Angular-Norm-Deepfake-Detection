# E4: AntifakePrompt 候选评分对照

**状态：已归档（2026-09-29）。** 第一阶段因已观察到的分类和语义判断表现较弱而提前停止；正式评估完成并校验 15/72 组。两组训练已完成，但缺失的评估不应计为失败或零分。归档结论与产物入口见 [E4 归档说明](ARCHIVED.md)，已完成指标见 [暂停时结果汇报](reports/paused_results_report_zh.md)。以下执行命令仅供复现记录，不表示实验仍在运行。

第一阶段：同一 InstructBLIP/Vicuna7B 骨干的 A0 官方 prompt、A1 词表评分微调、A2 共享标量评分微调。设计及判定见项目 `docs/superpowers/specs/2026-09-28-antifakeprompt-design.md`；状态见 `PROGRESS.md`。当前实验不可引用为已成功。

## 可复现依赖

- 上游 commit: `e261aa6ee4c398f63a36c08b7ebad3455dc1428e`，完整源码与license在 `upstream/`，未改写上游网络实现。
- Python3.11、PyTorch2.6.0+cu124、transformers4.28.1，独立 `.venv`；此环境共享系统torch，不改变E2/E3环境。
- `configs/assets.json` 列出每个公开模型URL、revision、尺寸及SHA256。运行 `.venv/bin/python scripts/prepare_assets.py --verify-only` 校验，省略参数可下载缺失文件。
- 上游BERT只读tokenizer/config；Q-Former全部预训练参数由InstructBLIP权重载入并检查。EVA官方39层特征提取器有意忽略原checkpoint的第40层、分类头和最终norm。
- 两域与所有语言池来自已保留项目，物理数据根为 `/Dataset/deepfake_detection/DF40_all`。不是官方原始FF++/Celeb-DF benchmark。

## 执行命令

均在本目录执行。每个进程显式绑定一张物理GPU（仅0–3）。

```bash
OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests
.venv/bin/python scripts/prepare_protocol.py
.venv/bin/python scripts/prepare_controls.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/smoke_model.py
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --arm A1 --smoke
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train.py --arm A2 --smoke
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/evaluate.py --arm A0 --domain ffpp --smoke
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/train.py --arm A1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python scripts/train.py --arm A2
CUDA_VISIBLE_DEVICES=2 .venv/bin/python scripts/evaluate.py --arm A0 --domain ffpp
CUDA_VISIBLE_DEVICES=3 .venv/bin/python scripts/evaluate.py --arm A0 --domain cdf
```

已冻结的数据清单不会被prepare_protocol覆盖。训练中断按相同代码/协议用 `--resume` 从上个完整epoch恢复；完整训练/评估不会静默覆写。已完成评估条件会验证产物hash后跳过，未完成cache须先审计并保留失败产物再恢复。

A1/A2评估将最后命令的arm改为对应名称；必须存在完整训练记录和其选中checkpoint。可用 `--conditions` 指定冻结条件中的子集，完整报告必须审计所有条件是否齐全。

## 解释边界

官方检测已按候选token NLL排序，不执行开放式生成。本改造测试移除词表头后的共享标量决策，仍保留7B语言Transformer。删除词表头不能等同于显著模型压缩或降低推理能力。

LM概率采用候选答案log likelihood求和后softmax；排序与官方接口等价，去掉官方随batch padding变化的公共分母。预处理采用公开checkpoint内部实际数据统计，所有组一致。精确长度匹配同时约束BERT和Llama；无意义词库仅根据分词性质扩展，不使用测试标签。

固定真假两域全视频各至多4帧；T1–T5诊断子集每个fake方法最多32视频且保留全部real视频，穷举18个seen和16个held-out问句/候选组组合。单seed42/小预算结果仅属于第一阶段，不能直接与旧三种子、32帧的E3指标相减归因。
