# 本轮论文图表的重绘代码

前三个Python脚本逐字节保留此前已经实际绘图和验收的代码，第四个用于完成正式五seed后的推理/分组对照图。它们读取保存的评价结果，不加载模型，也不重新训练、推理或选择权重。输入为本轮产物根，输出为该根下的`figures/`；重新执行会更新对应图表及渲染回执。

| 脚本 | 输出与证据范围 | 输入与统计单位 |
|---|---|---|
| `plot_accepted_H_results.py` | H预处理覆盖率；初始EfficientNet配对差值 | FF++/DFD全部候选TEST视频的处理成员表；三个完整配对训练seed的严格汇总 |
| `plot_D_grouping_counterexample.py` | CLIP seed2026分组反转的权重/均值分解 | 已独立重算的固定模型和固定逐帧分数；单例，不是五seed频率 |
| `plot_D_development_selection.py` | 源DEV/CDF-DEV最佳轮数；DINO开发曲线 | 全部12条已完成开发轨迹、102条epoch记录；开发数据，不是正式TEST选点 |
| `plot_formal_D_inference_grouping.py` | 正确分组的L2−Raw；错误合并对该增益的影响 | 原严格五seed汇总；逐seed配对差及均值±样本SD；附全部八格绝对AUC |

H初始配对差值图显示全部三个训练seed，以及均值±样本SD。D正式图显示五个训练seed的配对差及均值±样本SD，不能用200个校准子集代替五个训练seed。覆盖率为冻结候选视频的普查；单模型分解和独立开发曲线不画推断性置信区间。跨骨干完整D图和H其余骨干的图表须等待相应严格汇总。

正式D脚本通过`formal_D_figure_data.py`核验原严格汇总、全部已提交工件哈希和五个模型的分析回执，并从八格AUC还原差值。默认要求三个骨干全部完成；`--backbone`可显式绘制一个已完成五seed的骨干。该选项仍要求完整五seed，不会以部分种子生成均值。图中主对照为正确分组的L2增益和分组交互；全部八格绝对AUC保存在Source Data，避免把增益放大解读为绝对AUC提高。

## 环境和命令

实际绘图环境版本保存在`requirements-figures.lock.txt`，与训练环境分别记录。还需提供`nature-figure`工具的`scripts`目录，其中包含`audit_panel_alignment.py`、`audit_pdf_text.py`、`audit_figure_collisions.py`和`validate_figure.py`。这些QA工具不属于本仓库，须自行提供；`--qa-dir`显式指定目录。

```bash
export JOURNAL_ARTIFACT_ROOT=/home/z/workspace/deepfake_journal_20261102
export FIGURE_QA_DIR=/path/to/nature-figure/scripts

python scripts/figures/plot_accepted_H_results.py \
  --root "$JOURNAL_ARTIFACT_ROOT" --qa-dir "$FIGURE_QA_DIR"
python scripts/figures/plot_D_grouping_counterexample.py \
  --root "$JOURNAL_ARTIFACT_ROOT" --qa-dir "$FIGURE_QA_DIR"
python scripts/figures/plot_D_development_selection.py \
  --root "$JOURNAL_ARTIFACT_ROOT" --qa-dir "$FIGURE_QA_DIR"
python scripts/figures/plot_formal_D_inference_grouping.py \
  --root "$JOURNAL_ARTIFACT_ROOT" --qa-dir "$FIGURE_QA_DIR"
```

若只有DINOv2的原严格五seed汇总完成，可在最后一个命令上添加`--backbone dinov2_vit_b14`。正式脚本同时输出CDF和源TEST两图，默认保存到`figures/formal_D_inference_grouping/`；可用`--output-dir`指定独立重绘目录。首次DINOv2实测输出为`figures/D_dinov2_five_seed/`，两图的四面板、实际PDF最小7pt文字、对齐和碰撞检查均已通过；这不表示另外两个骨干已完成。

渲染时使用1.5pt容差检查两个面板的最终绘图区，并导出PDF、可编辑SVG、600dpi PNG和TIFF，以及CSV Source Data。`contracts/`保存科学问题、面板含义、统计和导出定义；H脚本还要求产物`figures/figure_contract.json`存在。

原验收回执和原输入必须完整。开发曲线回执保留实际运行的绝对输入路径作为哈希溯源；移动产物后应重新生成、核验相应比较回执，不将修改路径后的清单冒充原身份。

## 每次重绘后的检查

```bash
python "$FIGURE_QA_DIR/validate_figure.py" scripts/figures/plot_D_grouping_counterexample.py
python "$FIGURE_QA_DIR/audit_pdf_text.py" \
  "$JOURNAL_ARTIFACT_ROOT/figures/D_CLIP2026_grouping_counterexample.pdf" --min-pt 5
python "$FIGURE_QA_DIR/audit_figure_collisions.py" \
  "$JOURNAL_ARTIFACT_ROOT/figures/D_CLIP2026_grouping_counterexample.pdf" \
  --json-out "$JOURNAL_ARTIFACT_ROOT/figures/D_CLIP2026_grouping_counterexample.collision-audit.json"
```

对其余脚本和每个输出PDF执行同样检查，并逐面板查看最终版。渲染回执的`AWAITING_PDF_AND_VISUAL_QA`表示刚生成文件，不能替代PDF文字/碰撞检查、数值溯源和视觉复核。静态检查对“不适用的误差条”可能给出警告，应按上述统计单位逐项审阅；不据此添加虚构误差范围。

本目录的复制不改变训练、缓存、评分、统计或已冻结输入。重绘复核应使用独立输出目录，保留先前验收图表及其哈希。
