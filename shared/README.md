# 重合部分怎么维护

两个项目分别运行，各自持有基础代码的本地快照。此目录只做登记，不是运行时依赖，也没有第三套可执行源码。

| 部分 | 管理方式 |
|---|---|
| 数据索引、裁剪、采样、变换 | 两边本地副本；共同文件及 SHA256 见 [common_code.json](common_code.json) |
| 基础模型、训练器、评估工具 | 同上，修改后在另一边明确选择是否同步并跑相应测试 |
| `data/builders.py`、`models/factory.py` | 有意分开：旧论文移除语义模块接线；新论文保留冻结 E2 源码字节 |
| FF++、CDF 原始数据 | 外部数据只存一份，默认 `/Dataset/deepfake_detection/DF40_all`；两项目各有 `provenance/datasets.json` |
| H1 所用的旧 CLIP 检查点 | 新项目持有独立本地副本，来源和哈希在导入清单中登记 |
| 实验结果与模型 | 本次已实际复制，重跑不写回原工作区；Git 默认不跟踪大文件 |

后续修改共同模块时，不自动覆盖另一项目。先比对 `provenance/common_code.json`，同步后分别验证并新增版本记录。新实验使用新的配置与输出目录；冻结 E2 的源文件哈希是证据的一部分，修改它意味着新版本实验。

整体搬走任一 `projects/*` 目录时，同时带走其本地 `experiments/*/results/`；无需复制此目录或另一个论文项目。Visual-Jev 搬家后的绝对路径由 `scripts/relocate_artifacts.py` 重绑定并登记。
