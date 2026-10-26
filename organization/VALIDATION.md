# 两篇论文拆分验收

完成日期：2026-09-23。执行与复核均由当前助手完成，未使用子智能体。整理以原工作区实际文件为准，保留已有未提交内容和既有暂存项。

## 交付位置

| 项目 | 内容 | 使用入口 |
|---|---|---|
| `projects/feature_geometry` | 旧论文五阶段、训练与分析代码、30 个历史检查点及结果 | [项目说明](../projects/feature_geometry/README.md) |
| `projects/visual_jev` | H1/E2、正式及历史实验产物、独立训练入口和必要基线资产 | [项目说明](../projects/visual_jev/README.md) |

两个目录均为真实文件副本，不依赖兄弟项目或父工作区的源码。公共基础代码共 26 个文件，版本及哈希在各项目 `provenance/common_code.json` 和 [共同代码登记](../shared/common_code.json) 中保存。原始图像数据仍由两项目读取外部目录，各自登记数据位置。

原始工作区代码与实验保留原路径，用于追溯历史日志和来源指纹；日常研究进入 `projects/`。根 README 和架构说明改为导航，原版保存在 `legacy_navigation/`。每个项目都有独立环境说明、安装元数据、测试、配置与运行脚本。

## 实测结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 旧项目完整 CPU 测试 | 102 passed，1 warning，退出码 0 | [日志](../projects/feature_geometry/provenance/pytest.log) |
| 新项目完整 CPU 测试 | 230 passed，22 warnings，退出码 0 | [日志](../projects/visual_jev/provenance/pytest.log) |
| 两项目分别安装、构建 wheel | 均通过；发行名分别为 `deepfake-feature-geometry` / `visual-jev` | 各项目 `provenance/install_check.log`、`wheel_check.log` |
| Python 隔离导入检查 | 旧项目 43 个、新项目 29 个已加载本地模块全部来自自身 `src/` | 各项目 `provenance/isolated_import_check.json` |
| 训练入口 | 旧训练、H1、E2 的 `--help` 均正常退出 | 各项目 `provenance/train*_help.log` |
| 旧论文 CPU 回放 | 七阶段 dry-run 全部通过，约 11 秒 | [回放日志](../projects/feature_geometry/provenance/threshold_replay.log) |
| 新论文正式门槛回放 | FF++/CDF，10,000 次 bootstrap；完整数值、区间和裁决与冻结结果完全一致 | [回放验收](../projects/visual_jev/runs/split_validation_v2/verification.json) |
| 新论文独立产物审计 | 42/42 组，108 项通过，0 错误 | [审计](../projects/visual_jev/runs/split_validation_v2/artifact_audit.json) |
| 原始来源与复制完整性 | 1,409 个源文件按原路径或导航归档核验；767 + 695 个导入文件无未说明变更 | [完整性核验](integrity_verification.json) |
| 当前文档导航 | 链接核查通过；保留原字节的历史导航不作现行入口 | [链接核验](documentation_verification.json) |

测试警告为已有依赖弃用/运行警告，没有失败。此次验证使用两套独立 `.venv` 安装各自源码，第三方科学计算包通过 `--system-site-packages` 使用主机已有版本；没有验证全新机器从网络安装全部依赖，也没有重新训练模型或重新执行 GPU 图像推理。安装要求、已观察环境和搬家说明均已记录。

## 科学证据保持情况

E2 总裁决仍为 `not_supported`：G3 通过，G1/G2/G4/G5/G6 未通过。FF++ H1/E2 AUROC 为 0.9985308320646924 / 0.9984874527328979；CDF 为 0.911267387533556 / 0.8221806781942197。

E2 评估源码集合的 SHA256 保持为：

```text
0529ab9127e142d18afef5f0ffd52c2781ff506e449cded116510afd8e3d0af7
```

锁定配置、语言表、模型、预测及缓存均保持原字节。只有副本中 8 份运行路径元数据被重定位，原件与前后哈希保存在新项目 `provenance/path_metadata_original/`、`path_relocations.json`。正式回放仅覆盖 FF++、CDF；保留 DFD 历史缓存登记是为了维持原来源指纹。

旧项目只裁剪了新论文接入：数据加载器的 4 个语义函数和模型工厂的 2 个 Jev 分支；其余原函数通过 AST 比对保持一致。新项目 H1 训练器来自原通用训练器，只适配根路径；审计工具移入正式 `scripts/` 以便源码单独传输。

## 使用与维护

在各自项目根目录运行：

```bash
# 旧论文：CPU 小规模流程回放，输出到新 runs/ 目录
python scripts/replay_threshold.py
```

```bash
# 新论文：重定位检查、冻结门槛回放和 FF++/CDF 审计
python scripts/replay_e2.py
```

两个脚本都为新输出创建目录，已有目录会被拒绝，避免覆盖结果。新项目重定位支持幂等重跑及再次搬家；对应回归测试同时验证损坏检查点会在元数据改写前被拒绝。

两个项目应使用不同 Python 环境；共同模块的后续修改显式同步并记录新版本。搬家时重新创建 `.venv`，同时传输本地大型实验资产；这些大文件默认不由 Git 跟踪。

本次为保留原始证据而选择完整复制实验历史，额外占用约 21 GiB。根目录旧代码和结果没有清理；不使用它们作为新工作的运行入口。
