# 2026-05-05 论文级特征几何实验（paper feature geometry）

时间跨度：2026-05-05 ~ 2026-06-19（checkpoint 训练 05-05 ~ 05-15，三模式评估 05-31，黑白版图 06-19）

## 做了什么

训练 30 个 checkpoint，并在同一批 checkpoint 上比较三种推理模式，考察「跨数据集的可迁移性来自特征方向还是特征模长」：

- **模型集合**：3 个 backbone（CLIP / DINOv2 / EfficientNet）× 5 个 seed（7/42/123/999/2048）× 2 种训练模式（`raw`、`normtrain`）= 30 个 checkpoint。
- **推理模式**：`raw`（方向 + 模长）、`direction`（L2 归一化，只用方向）、`length`（只用 ‖z‖ 当分数）。
- **评估域**：FF++、CDF、DFD。

## 主要结论

- FF++：raw 与 direction 都接近满分。
- CDF：direction > raw > length。
- DFD：raw > direction ≫ length；DFD 正负样本约 1:10，必须看 AUC 与 balanced accuracy，普通 ACC 会被极端阈值骗过去。
- **重要限定**：`reports/final_results_notes.md` 记录了复现审计——CDF 上「direction > raw，15/15 胜」只在按 `video_id` 单独分组（把同一身份的不同伪造方法合并成一个"视频"）时成立；换成语义正确的 `(method_id, video_id)` 分组后优势消失并轻微反转（−0.0044，4/15）。这条限定是后续三个阶段的出发点。

## 目录内容

```text
scripts/
  train.py                      # 训练入口（torchrun 多卡）
  evaluate.py                   # 标准 FF++ / CDF / DFD 评估
  launch_training.py            # 批量生成多 backbone 训练配置并串行启动（仅 dinov2/efficientnet；xception 已归档会明确报错）
  evaluate_norm_correction.py   # raw / feature_norm / feature_length 三模式评估
  summarize_paper_results.py    # 打印论文 AUC 汇总表
  generate_bw_figures.py        # 生成黑白（印刷版）帧级图
  prepare_dfd_df40.py           # 把 DFD 整理成 DF40 风格的 JSON/帧目录
  download_faceforensics.py     # FaceForensics++ 官方下载脚本
configs/                        # 训练配置快照（hpval 学习率验证 / multibb 20 个 run 的配置）
                                # 注：xception 与重复的 *_base.yaml 于 2026-07-29 删除，
                                # 备份在仓库根 .deleted_configs_backup_20260729.tar.gz；
                                # base 模板统一以仓库根 configs/ 为准
results/
  checkpoints/<backbone>/<mode>/seed_<seed>__<ts>/   # 30 个 checkpoint（含 config、日志、best_model.pth）
  auc_video_level_60rows.csv                          # 视频级 AUC：3 backbone × 5 seed × 4 模式
  dfd_balanced_thresholds_45rows.csv                  # DFD 视频级平衡阈值结果
  dfd_balanced_thresholds_summary.csv                 # 上表的分组汇总
figures/frame_level/, figures/frame_level_bw/         # 帧级可视化（t-SNE、校准曲线、阈值、模长直方图、cosine 热图）
reports/final_results_notes.md                        # 结果说明 + `direction > raw` 复现审计
docs/                                                 # 论文/中期设计与实现计划
```

`results/checkpoints/` 是后续三个阶段共用的 checkpoint 根目录，路径写在各阶段配置的 `ckpt_root` 里。

图是可视化产物，不是论文数值来源；论文级数值一律以 `results/` 里的视频级 CSV 为准。

## 怎么重跑

```bash
cd <仓库根>
P=experiments/2026-05-05_paper_feature_geometry

# 训练（配置模板在仓库根 configs/，也可用本目录 configs/ 里的快照）
torchrun --nproc_per_node=8 $P/scripts/train.py --config configs/clip_base.yaml

# 标准评估
python $P/scripts/evaluate.py --config configs/clip_base.yaml \
  --checkpoint $P/results/checkpoints/clip/raw/seed_42__20260505_130308/output/best_model.pth \
  --domain all

# 三模式（raw / direction / length）评估
python $P/scripts/evaluate_norm_correction.py --config configs/clip_base.yaml \
  --checkpoint $P/results/checkpoints/clip/raw/seed_42__20260505_130308/output/best_model.pth \
  --splits ffpp,cdf,dfd --mode raw,feature_norm,feature_length

# 打印 AUC 汇总
python $P/scripts/summarize_paper_results.py
```

脚本默认输出到本阶段目录（`results/`、`figures/`），不再写仓库根的 `outputs/`。新的训练 run 默认落在 `experiments/_runs/<时间戳>/`（git 忽略）；定稿后再挪进某个阶段目录。

## 已删除的中间缓存

原 `features/frame_level/` 是 45 个帧级 `.npz`（约 32GB）纯中间缓存，已删除；需要重画帧级图时可以从 `results/checkpoints/` 重新抽取。
