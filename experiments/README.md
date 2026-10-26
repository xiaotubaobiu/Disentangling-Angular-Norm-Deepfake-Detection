# 实验目录（按时间分期）

每个子目录是一个**自包含的实验阶段**：跑这一阶段用到的脚本、配置、结果、图、报告、测试、设计文档都在里面。可复用的库代码不放这里，统一放在仓库根的 `src/`。

目录名格式：`<阶段起始日期>_<主题>`。

## 阶段总览

| 阶段目录 | 时间跨度 | 主题 | 一句话结论 |
|---|---|---|---|
| `2026-05-05_paper_feature_geometry` | 2026-05-05 ~ 06-19 | 30 个 checkpoint 的训练 + raw / direction / length 三种推理模式的论文级评估 | FF++ 上 raw≈direction 近满分；CDF 上 direction>raw（**仅在按 `video_id` 聚合时成立**）；DFD 上 raw>direction≫length |
| `2026-07-21_frame_midterm_diagnosis` | 2026-07-21 ~ 07-23 | 30 个 checkpoint × 3 个域的帧级全量重推理与诊断 | 产出 `per_frame_scores.parquet`（39.8M 帧，后两个阶段的数据底座）；5 个假设中 H1/H5 部分支持，H2/H3/H4 被否；CDF 上 direction 并不优于 raw（native pooled Δ=−0.0044，4/15） |
| `2026-07-23_four_questions` | 2026-07-23 ~ 07-24 | 四个问题的实证检验（Q1 归一化增益分化 / Q2 模长跨域稳定性 / Q3 length-only 可靠性 / Q4 多方法公共方向） | Q1 支持、Q2 部分支持、Q3 支持、**Q4 不支持**；决策取 A（径向—角向自适应），放弃 B |
| `2026-07-27_pooling_saturation` | 2026-07-27 ~ 07-28 | direction>raw 到底是特征几何还是 sigmoid 饱和 + 概率池化伪影 | 主因是 A（池化伪影，温度校准后优势归零）+ D（强方法依赖，FSGAN 与 SadTalker 符号相反） |
| `2026-07-30_threshold_transfer` | 2026-07-30 | 在 FF++ validation 上校准+选阈值后冻结迁到 CDF，阈值还可靠吗 | 不可靠：`R_transfer` 0.198 BA（0/30 个 cell 达标），1% FPR 预算实测 64%，温度校准改不了决策边界；主因是 real 类平移 +4.15 vs fake +0.36 的类条件漂移 |

阅读顺序建议：阶段 1 给出现象 → 阶段 2 给出帧级证据底座 → 阶段 3 拆解四个机制问题 → 阶段 4 推翻「方向更好」的朴素解释 → 阶段 5 把问题从「排序」推进到「决策」（阈值能不能迁移）。

## 阶段内部结构

```text
experiments/<date>_<topic>/
  README.md      # 这一阶段做了什么、结论、怎么重跑
  scripts/       # 这一阶段的全部脚本（唯一的脚本存放位置）
  configs/       # 这一阶段的配置快照
  results/       # 表格、parquet 缓存、日志、checkpoint 等重产物（git 忽略）
  figures/       # 图（git 忽略）
  reports/       # markdown 报告
  docs/          # 该阶段的设计/计划文档
  tests/         # 该阶段专属的单元测试（库级测试在仓库根 tests/）
```

不是每个阶段都有全部子目录（例如阶段 2 的表和图都在 `results/parallel_20260721_213914/` 这个 run 目录里，因为流水线就是按 run 目录组织输出的）。

阶段 2 另有 `reports/provenance.md`：逐产物的生成脚本/命令/时间线、已知偏差（cartesian-join 缺陷、被撤回的 figure 01、已删除的 per_ckpt 分片）与复现验证记录。

## 路径约定

阶段脚本统一用两个锚点，不依赖当前工作目录：

```python
PHASE = Path(__file__).resolve().parents[1]   # experiments/<date>_<topic>/
ROOT  = PHASE.parents[1]                      # 仓库根
sys.path.insert(0, str(ROOT / "src"))
```

- **阶段自己的**输入输出（`configs/`、`results/`、`figures/`、`reports/`）→ 相对 `PHASE`，在 yaml 里也写成相对阶段目录的路径（如 `tables: results/tables`）。
- **跨阶段共享的**产物（帧级 parquet、checkpoint 根目录）→ 相对 `ROOT`，在 yaml 里写完整的仓库相对路径。

跨阶段依赖只有两条：

```text
2026-05-05_.../results/checkpoints                                  → 阶段 2/3/4/5 都要读
2026-07-21_.../results/parallel_20260721_213914/per_frame_scores.parquet → 阶段 3/4/5 都要读
2026-07-27_.../results/val_cache/*__val.parquet                     → 阶段 5 的源域（FF++ validation）
```

## 关于历史文档里的旧路径

`reports/` 和 `docs/` 里的文件是**当时的记录**，正文没有改写，因此仍会出现整理前的旧路径。对应关系：

| 文档里的旧路径 | 现在的位置 |
|---|---|
| `scripts/<name>.py` | `experiments/<对应阶段>/scripts/<name>.py` |
| `configs/study_config.yaml` | `experiments/2026-07-23_four_questions/configs/study_config.yaml` |
| `configs/pooling_saturation_config.yaml` | `experiments/2026-07-27_pooling_saturation/configs/pooling_saturation_config.yaml` |
| `outputs/tables/`、`outputs/figures/`、`outputs/logs/` | 对应阶段的 `results/tables/`、`figures/`、`results/logs/` |
| `outputs/z_cache/`、`outputs/val_cache/` | `2026-07-23_.../results/z_cache/`、`2026-07-27_.../results/val_cache/` |
| `outputs/frame_midt_diag/parallel_20260721_213914/` | `2026-07-21_.../results/parallel_20260721_213914/` |
| `experiments/final_results/checkpoints/` | `2026-05-05_.../results/checkpoints/` |
| `src/deepfake_detection/<pkg>/` | `src/<pkg>/` |

报告里的图片链接已改成阶段内相对路径（`../figures/...`），可以直接在阶段目录里预览。

## 新开一个阶段

```bash
D=experiments/$(date +%Y-%m-%d)_<topic>
mkdir -p $D/{scripts,configs,results,figures,reports,docs,tests}
```

然后：脚本放 `scripts/`（用上面的 `PHASE`/`ROOT` 锚点）、配置放 `configs/`、写 `README.md`，并在本文件的阶段总览里加一行。可复用的逻辑抽到 `src/`，不要在 `scripts/` 之间复制粘贴。
