# 2026-11-02期刊实验：已冻结的实际配置

本目录保存这轮实际使用的六份预算决定和33个正式任务定义。它补充根目录的15轮基础模板；模型训练是否完成由独立产物中的运行回执记录。

| 路线 | CLIP最大轮数 | DINOv2最大轮数 | EfficientNet最大轮数 | 正式模型 |
|---|---:|---:|---:|---:|
| D：DF40-FF来源21方法 | 2 | 30 | 2 | 三骨干×五seed，共15 |
| H：FF++原视频B/A→B | 10 | 60 | 15 | 三骨干×三seed×两历史，共18 |

学习率分别为1e-5、5e-5、1e-4。三个骨干均全参数微调视觉编码器和二分类头，实际batch128，AdamW/CE/WD5e-4、AMP、梯度裁剪1、EMA.999，训练输出不额外L2。D正式权重按源DEV视频mean-prob pooled AUC选最佳EMA、patience5；H在共同DEV-B上按B/AB模型AUC均值选共同轮数和早停。

D使用开发seed1007/1042的源DEV和隔离CDF-DEV确定总预算；H开发只使用源DEV。正式D五seed为7/42/123/2026/3407，H三seed为7/42/123，正式训练轨迹只按源验证选点。

## 文件含义

- `frozen_decisions.json`：六份预算、固定参数、预训练权重来源/哈希和各路线选点规则。
- `formal_tasks.json`：33个实际任务对象，以及对应产物根内的训练、缓存和分析配置路径。
- `epoch_recipes/`：六份原始决定JSON的逐字节副本。其SHA256与正式任务中的`epoch_recipe_sha256`相同；其中开发状态绝对路径是原运行的溯源字段。
- `cohort_definitions.json`：数据冻结哈希、逐角色/方法规模、CDF开发隔离和H共同/全部B队列定义。
- `geometry_protocol.json`、`geometry_frozen.json`：实际A/B/AB几何规则、实现与权重来源的原始副本。
- `requirements.execution.lock.txt`：这轮实际Python环境的版本清单。
- `snapshot_sha256.json`：本目录文件和原始输入的哈希，供独立核对。

几何协议的`protocol_sha256`使用规范JSON计算；`snapshot_sha256.json`记录文件字节哈希。两种标识的计算对象不同，复核时分别核对。

## 与运行目录的关系

该目录是版本化的实验定义。实际模块读取产物根的`configs/execution.json`、`h_execution.json`和各骨干配置；调用方法见[代码说明](../../../docs/期刊实验代码说明.md#本轮运行配置与直接接口)。`formal_tasks.json`可用于逐项核对这些配置中的任务。

原始视频、冻结图像、清单、环境身份、模型状态和缓存保留在独立产物根。`training_identity()`将数据freeze、初始化、环境、代码和epoch配方绑定到权重；根目录或图像路径改变后，应重新构建相应清单/身份，作为新的实验登记。预算、数据划分和选点规则可按这里的定义重建，已有权重的恢复仍需匹配其原身份。

D每轮799个batch，H每模型每轮185个。CLIP的D2/H10最大预算分别最多1598/1850次更新尝试，因此epoch数量不能直接表示跨路线的训练工作量。实际成功更新、最佳轮和停止轮由运行记录给出。

## 只读核对

在仓库根目录执行：

```bash
python - <<'PY'
from pathlib import Path
import hashlib, json
root = Path('configs/experiments/journal_20261102')
snapshot = json.loads((root/'snapshot_sha256.json').read_text())
for name, expected in snapshot['files_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, name
tasks = json.loads((root/'formal_tasks.json').read_text())['runs']
assert len(tasks) == 33
assert sum(t['route'] == 'D' for t in tasks.values()) == 15
assert sum(t['route'] == 'H' for t in tasks.values()) == 18
print('Snapshot hashes and 33 task definitions verified')
PY
```
