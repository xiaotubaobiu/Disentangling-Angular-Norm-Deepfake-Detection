# Deepfake Pipeline Reliability — v1.0

本包把本次确定的**两数据路线、四个研究问题**落实为实验协议、中英文论文工作稿、配置注册表，以及可运行的分析/参考训练代码。

**重要状态**：没有在用户服务器上执行新模型训练，没有处理真实FF++/DFD原片，没有生成新21方法实验结果，没有修改或推送任何远程仓库。这里的CPU测试和演示只验证实现逻辑，不能代替真实数据与GPU资格检查。

## 先阅读这三个文件

- `docs/实验设计与执行协议_中文.md`：完整研究设计、变量、对照、训练预算、统计与停止条件。
- `paper/论文初稿_中文.md`：中文正文工作稿，结果位置保留明确待填标记。
- `paper/manuscript_en.md`：对应英文工作稿，不是只有章节标题。

## 实验范围

| 路线 | 数据 | 训练模型数 | 用途 |
|---|---|---:|---|
| H | 原始FF++四方法训练；FF++/DFD测试 | 2种训练历史×3骨干×5种子=30 | B(X)与B(A(X))训练、共同B部署输入 |
| D | DF40衍生FF的9FS+12FR训练；FF/CDF测试 | 3骨干×5种子=15 | 推理L2、八条件评分、独立目标真实校准 |

正式训练共45个Raw模型，源域开发预实验另计。八种评分不需要八倍训练。EFS/FE、目标域微调、完整训练L2、对比学习因果归因不纳入本轮。

## 目录

```
configs/       45个训练配置草案、run_registry、study与claim_evidence
src/pipeline_study/
  core.py       AUC/阈值/正确与碰撞聚合/特征分解/配对账本
  analysis.py   八格审计、来源组bootstrap、嵌套目标真实校准、高精度计分
  cli.py        audit / score / calibrate / precision
  preprocess.py 按绑定图像的关键点/框执行A或B，保留几何与血缘
  training.py   全参数Raw参考训练、source-only开发、配对特征导出
examples/      明确标记的合成自检数据与演示脚本
results/       本次代码测试报告与合成smoke结果；不是论文实验结果
paper/         中英文稿、引用核验表、结果槽位与成稿工具
manifests/     输入字段规范
```

## 环境与安装

建议在服务器的独立conda环境中使用Python 3.10+。依赖下限不是已验证的唯一组合；正式运行前生成并冻结实际`pip freeze`、CUDA、GPU与驱动信息。不能用一台机器的环境声明跨平台bit一致。

```bash
cd deepfake_pipeline_study_v1
python -m pip install -e '.[test,train,preprocess]'
pytest -q
```

分析核心仅需NumPy、pandas、SciPy与PyYAML。高精度诊断需mpmath。真实训练需PyTorch、timm和open_clip_torch及本地预训练权重。权重不自动下载。只加载自己可信来源的PyTorch checkpoint。

## 1. 数据准备与资格检查

复用当前新版仓库已审计的来源解析器和八帧抽样，不复用旧方法ID遗漏的代码。新代码不能仅把旧CSV的EFS行删掉后就宣称获得完整的新训练协议。

最少字段见`manifests/SCHEMA.md`。训练、DEV与校准/测试在原片和派生关系上先隔离，再产生图像变体。`source_group`必须来自来源元数据；随意填不同字符串不能证明独立性。

本包没有实现新的脸检测模型，也没有访问实际原视频。需要服务器既有人脸检测/关键点模块导出注释，并把注释绑定到输入文件SHA256。A→B中的第二次检测读取A图，不能读取原片。预处理代码在处理后删除旧坐标字段，防止直接沿用原图坐标。

```bash
python -m pipeline_study.preprocess --manifest original_annotated.csv --pipeline A \
  --annotation-model-sha256 VERIFIED_DETECTOR_SHA --output generated_A
# 在 generated_A 输出图上重新检测并产生 A_annotated.csv 后：
python -m pipeline_study.preprocess --manifest A_annotated.csv --pipeline B \
  --annotation-model-sha256 VERIFIED_DETECTOR_SHA --output generated_AB
```

单次B版本从相同母帧的`original_annotated.csv`调用B。不要把已处理DF40图片当作未经处理原片来验证RQ1。

## 2. 源域开发与正式配方冻结

45个YAML中的`/REQUIRED_MANIFESTS/`与`/REQUIRED_LOCAL_WEIGHTS/`是明确待接入路径，不是存在的文件。替换为服务器路径，并核对各骨干的权重格式、输出维数、位置编码、输入标准化和全参数可训练范围。

初始epoch30、lr为建议开发起点，不是已经证明收敛的设置。未冻结的配置默认禁止正式启动；源域预实验用`--pilot`：

```bash
CUDA_VISIBLE_DEVICES=0 python -m pipeline_study.training train \
  --config configs/D_released_dinov2_vit_b14_s42.yaml --pilot --out runs/pilot_dino42
```

先按协议用源DEV解决DINO训练稳定性，再写下仅源域的配方选择记录，把正式配置`recipe_status`置为`FROZEN`。配置/数据/预训练权重hash会写入训练identity；不同identity禁止恢复同一运行。正式五种子不按CDF挑选。

```bash
CUDA_VISIBLE_DEVICES=0 python -m pipeline_study.training train \
  --config configs/D_released_clip_vit_b16_s7.yaml --out runs/D_clip_s7
```

参考runner保存`last.pt`与`best_source.pt`、优化器/AMP/RNG状态、每轮采样数组与源DEV日志。正式主分析用冻结预算的`last.pt`；不把训练中最高目标分数当作选点。恢复只承诺epoch边界，不承诺batch中途或跨平台完全一致。相同轮数不等于相同成功AMP更新数，两者均记录。

GPU建议0–3四个独立作业并行，不抢占现有任务、不使用4–7。正式启动前必须实际完成一个小规模GPU资格运行：前向维数、反向梯度、两个连续恢复epoch、checkpoint导出、真实manifest计数和源DEV分数独立复算。当前包的CPU通过不能替代这些步骤。

## 3. 导出同一模型的Raw/L2分数

```bash
CUDA_VISIBLE_DEVICES=0 python -m pipeline_study.training export \
  --checkpoint runs/D_clip_s7/last.pt --manifest frozen/D/target_test.csv \
  --out caches/D_clip_s7_target_test
```

导出以FP32编码器前向得到raw特征，保存分块NPZ及线性头；FP64重建两输出margin差、L2 margin、r、q、b。当前参考runner导出不使用FP16，这一点需要与旧协议明确区分并冻结。推理L2不重新训练分类头。

源CAL、目标CAL和目标TEST分别导出。每个缓存要有模型hash、清单hash、角色和实验ID。源域pilot模型默认不得导出校准或测试角色。

## 4. 八条件与精度审计

```bash
pipeline-study audit --frames caches/D_clip_s7_target_test/frames.csv \
  --out analysis/D_clip_s7_target_test
```

输出：`eight_cell_auc.csv`、logit诊断、3×3排序转移、正确分组视频分数与oracle诊断。`collision`条件明确标为合成错误单位，方法宏AUC为不适用。默认8个不同RGB帧；不同domain/checkpoint/seed禁止混在同次分析。

```bash
pipeline-study precision --frames caches/D_clip_s7_target_test/frames.csv \
  --feature raw --grouping correct --digits 80 160 --out precision/D_clip_s7
```

精度诊断比较实际高精度排序，不将mp概率先转换回float64再计算AUC。若80/160位仍不稳定，再增加320位。不能从已经饱和的概率倒推精确logits。

## 5. 校准输入与独立测试

真实CAL仅包含一个类别，不能调用需要两类的AUC审计；使用`score`生成视频分数：

```bash
pipeline-study score --frames caches/source_cal/frames.csv --feature raw --aggregation logit --out scored/source_raw
pipeline-study score --frames caches/target_cal/frames.csv --feature raw --aggregation logit --out scored/target_raw
pipeline-study score --frames caches/target_test/frames.csv --feature raw --aggregation logit --out scored/test_raw
pipeline-study calibrate --source-cal scored/source_raw/videos.csv \
  --target-cal scored/target_raw/videos.csv --test scored/test_raw/videos.csv \
  --budgets 10 20 40 80 --repeats 200 --out calibration/raw
```

L2与概率聚合重复此流程，保持同一校准ID和seed。校准/测试来源重叠、计分方式或checkpoint不一致会报错。源threshold、默认threshold与目标各预算都在同一测试集评价。`--rule np`仅是已有统计参照，不是本研究新方法；样本数或独立单位条件不满足时输出不可用。

`calibration_repeats.csv`保存每次结果，`nested_calibration_ids.csv`保存精确子集，`subsample_summary.csv`是子抽样描述统计，不是总体置信区间。另用`paired_bootstrap`作固定模型的来源组测试区间。

## 6. 成稿

`paper/result_slots.json`列出需要填入的新结果。运行结果应先通过审计，再写入槽位或生成T1–T8表；不要自动把某次最好的结果填进摘要。

```bash
python paper/fill_manuscript.py --template paper/manuscript_en.md \
  --values paper/result_slots.json --out paper/manuscript_en_filled.md
```

默认保留未填标记；`--release-check`会拒绝含未填槽位或待生成表格的文件。参考文献R1–R12在`references.md`，原会议论文引用必须另行核实，不能拿学位格式PDF冒充出版版本。

## 当前验证边界

CPU单元测试覆盖AUC与sklearn一致性、ties、分组碰撞、校准边界、NP最小样本条件、校准/测试隔离、模型身份保留、共同嵌套抽样、确定性几何变换和合成小模型的训练→导出。CLI合成演示另外保存。

**未验证**：实际人脸检测、真实数据来源链、三种真实预训练权重接口、V100训练/吞吐/确定性、所有正式实验结论。请勿把本包标成“45个模型已经训练完成”。

本包没有自动上传数据、论文或代码，没有删改旧结果，也没有为公开代码擅自指定知识产权许可。作者可在确认第三方依赖、数据许可和会议出版协议后设置仓库许可。
