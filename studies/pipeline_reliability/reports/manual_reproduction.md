# DF40 v1.1 手动复现与验收

本指南复现本轮 31 方法、3 backbone × 5 seed 的固定实验。复现源码基线为 `664fd62e22e3822d64e95db31a8fc56336c58f27`；计算代码 SHA256 为 `7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5`。本次交付的最终状态、科学结果和独立验收另见 `final_results.md`、`final_independent_acceptance.json`。这些文档不会改变计算代码 SHA。

原项目是 `/home/z/project/deepfake_detector`，原工件目录是 `/home/z/workspace/deepfake_detector_v11_20261007`。原项目的 `artifacts` 和 `.venv` 是快捷链接。原始数据只读，原始 Obsidian v1.1 设计保持不变，失败与旧快照全部保留。

## 1. 先检查现有实验

```bash
python /home/z/project/deepfake_detector/run.py status
cat /home/z/project/deepfake_detector/artifacts/execution_status.json
cat /home/z/project/deepfake_detector/artifacts/queue_status.json
```

同时检查记录中的 supervisor/worker PID、进程 cwd/comm 和 execution/queue 锁。原任务仍运行时继续读状态；本轮没有重复启动、补跑或人为中断正常训练。最终验收需要 15 个 epoch15 终态、15 份四角色完整缓存、30 个匹配分类头、A–D 分析、五种子汇总以及原 LLM 的同一容器正常运行和实际 `/health` HTTP200。训练或 pilot 完成不等于整体实验完成。

## 2. 只在新的项目和输出目录进行新实验

以下是将来重新运行的命令，不是本次正在执行的任务。新项目和新工件路径必须原先不存在。保留足够 SSD 空间；本轮实际空间见 `storage.json` 和最终进度凭据。

```bash
OLD_PROJECT=/home/z/project/deepfake_detector
OLD_ARTIFACT=/home/z/workspace/deepfake_detector_v11_20261007
NEW_PROJECT=/home/z/project/deepfake_detector_repro_YYYYMMDD
NEW_ARTIFACT=/home/z/workspace/deepfake_detector_repro_YYYYMMDD

test ! -e "$NEW_PROJECT" && test ! -e "$NEW_ARTIFACT" || exit 1
git -C "$OLD_PROJECT" worktree add --detach "$NEW_PROJECT" 664fd62e22e3822d64e95db31a8fc56336c58f27
mkdir "$NEW_ARTIFACT"
ln -s "$NEW_ARTIFACT" "$NEW_PROJECT/artifacts"
ln -s "$OLD_ARTIFACT/environment_clean" "$NEW_PROJECT/.venv"
```

同机复现可只读复用原固定 Python 环境。独立机器使用 Python 3.11.5，按 `requirements.full.lock.txt` 安装；Torch 2.6.0+cu124、torchvision 0.21.0+cu124 及全部依赖版本都固定。硬件、CUDA/驱动和 Python 构建差异需要记录，不能据此承诺跨环境逐 bit 一致。`run.py` 会自动切换到该项目 `.venv/bin/python`。

在新项目中仅修改它自己的 `configs/v1.1.json`：`artifact_root` 指向 `NEW_PROJECT/artifacts`，数据根指向同一只读数据；初始化路径指向新权重目录。核心设置保持 224、有效 batch128、AdamW WD5e-4、15 epoch final、三个 LR=1e-5/5e-5/1e-4、seeds=[7,42,123,2026,3407]，不增加早停、EMA或调度器。`prepare` 会写入**该项目的主配置**，所以仅在原项目添加一个备用 config 文件不是独立复现方式。


新 checkout 的具体配置路径写法：

```bash
NEW_PROJECT="$NEW_PROJECT" NEW_ARTIFACT="$NEW_ARTIFACT" "$OLD_PROJECT/.venv/bin/python" - <<'PY'
import os,json,pathlib
p=pathlib.Path(os.environ['NEW_PROJECT'])/'configs/v1.1.json'
c=json.loads(p.read_text())
c['artifact_root']=str(pathlib.Path(os.environ['NEW_PROJECT'])/'artifacts')
for name,file in [('clip_vit_b16','ViT-B-16.pt'),('dinov2_vit_b14','dinov2_vitb14_pretrain.pth'),('efficientnet_b0','efficientnet_b0.pth')]:
    c['backbones'][name]['weight_path']=str(pathlib.Path(os.environ['NEW_ARTIFACT'])/'weights'/file)
p.write_text(json.dumps(c,indent=2)+'\n')
PY
```
## 3. 固定原始证据和初始化输入

冻结操作依赖两项已有验证输入：作者修订后的官方 62 JSON 副本，以及原 FF-val 的独立 real 候选清单。`audit` 不会自动重新下载官方修订目录，`prepare` 也不会自动生成 real 候选清单。复现时从本轮只读证据复制它们：

```bash
mkdir -p "$NEW_ARTIFACT/data/official_verification" "$NEW_ARTIFACT/weights"
cp -a "$OLD_ARTIFACT/data/official_verification/official_metadata" "$NEW_ARTIFACT/data/official_verification/"
cp -a "$OLD_ARTIFACT/data/source_splits" "$NEW_ARTIFACT/data/"
cp "$OLD_ARTIFACT/data/source_real_candidates.parquet" "$NEW_ARTIFACT/data/source_real_candidates.parquet"
cp "$OLD_ARTIFACT/weights/dinov2_vitb14_pretrain.pth" "$NEW_ARTIFACT/weights/"
cp "$OLD_ARTIFACT/weights/efficientnet_b0.pth" "$NEW_ARTIFACT/weights/"
cp /home/z/.cache/clip/ViT-B-16.pt "$NEW_ARTIFACT/weights/ViT-B-16.pt"
```

在新 config 中将 CLIP 的 `weight_path` 指向最后复制的 `ViT-B-16.pt`。另两个权重由 prepare 固定到新 artifact 的 weights 目录。`reproduction_input_assets.json` 列出全部官方副本与源划分的 SHA。候选清单 SHA=`72efb17f42ad319ce6bef58ceb1ffd50a09ec46294560416faa1db29a2eb379b`。初始化 SHA：CLIP=`5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`；DINO=`0b8b82f85de91b424aded121c7e1dcc2b7bc6d0adeea651bf73a13307fad8c73`；EfficientNet=`7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`。

这些复制是复用本轮已审计的输入证据，不是对未知生成祖先的新认证。公共 master URL 日后可能变化；仅成功下载不能证明是本轮版本。复制前后核对 SHA，保留 provenance 和官方/本机核对报告。需要重新建立校准候选或验证新的官方版本时，应另立数据版本、重做来源和完整 byte/RGB 证据。

## 4. CPU 数据冻结与协议核对

```bash
cd "$NEW_PROJECT"
python run.py prepare
python run.py audit
python run.py evidence
python run.py freeze --approval configs/source_isolation_authorization.json
python run.py preflight
python run.py dry-run
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src .venv/bin/python -B -m pytest tests -q
```

CPU审计逐路径解码、byte/RGB哈希；freeze 只使用已授权的来源/内容规则，保留发布角色和逐条 disposition。已有 freeze 标记禁止覆盖。检查新旧 frozen manifest 的**完整逻辑行、顺序、dtype、标签、RGB、sample/group、sampling_mass**；校准 `index_path` 随新目录变化应单独记录。新目录/时间戳使配置、路径凭据和 freeze marker SHA 变化，不能把这些当作图像或划分变化。若科学输入内容变化，先解释差异，不能冒用本轮身份。

本轮目标清单为 train250000、FFtest59935、CDFtest368922、real-cal1120。训练保留265527方法关联贡献并合并重复 RGB 输入，共享 SG2/SG3 内容不删除方法。21 FS/FR 视频每个保留8个不同 RGB；10 EFS 以图像评价，数字存储桶不作为自然视频。四角色选中 RGB 无交集。Real=0/Negative，Fake=1/Positive，logit margin=Fake−Real，预测条件 score>=冻结阈值。图像流程中视频帧每行权重1/8，独立 EFS 图像权重1，视频流程每视频权重1。


计算流程使用 Fake−Real 的二类 head margin `z=w·h+b`，direction 将 raw embedding 替换为 `u=h/max(||h||,1e-12)`、保持原 head。视频 RL/DL 是8帧 logit 的平均及其 sigmoid 概率；RP/DP 是8帧 sigmoid 概率的平均，以该平均概率排名。图像 raw/direction 各使用对应 margin。四视频流程与两图像流程各自在 source-real calibration 上冻结 FPR5 阈值；不跨流程套用数值阈值。
## 5. 一次完整执行

核对 GPU4–7 空闲且现有相关作业均已结束，再在新项目执行一次：

```bash
python run.py execute
```

execute 依次预检、三骨干实际 GPU microbatch128 资格测试、隔离的1epoch pilot、15个正式训练及四角色特征缓存，然后恢复原 onecat-vllm，再完成 CPU 配对头/分析/报告，并检查实际服务健康。它记录非秘密服务状态，只正常 stop/start 原容器，保留 restart policy。GPU0–3/Jev 和 onecat-gateway 不参与。

手动分阶段可使用 `probe`、`pilot`、`run`、`report`，但应先读各阶段身份和锁。自动完整链推荐 `execute`。`run`/`resume` 只处理正式队列，不会自动完成后续 report。`report` 是 CPU 阶段，会复用身份和输出哈希完全匹配的既有分析。

训练不补 AMP 跳步：每模型3750000 draws/29310 optimizer attempts，预算以 draws/15epoch 计；successful updates 与跳步数单独披露。AMP初始scale128，TF32关闭、严格确定性、CuDNN benchmark关闭、worker0。DINO保持原37×37可学习位置网格，FP32 CPU antialias bicubic resize 带 autograd；不丢弃或固定这些位置参数。

## 6. 恢复、失败和结果复核

```bash
python run.py resume
# 队列完成后独立 CPU 报告：
python run.py report
```

每个完成 epoch 原子保存 model/optimizer/AMP/Python/NumPy/CPU/CUDA RNG、history 与真实 sampler 数组/哈希。失败从上个完整 epoch 续接；未承诺 batch 中点恢复。任一输入、初始化、代码或配置身份变化会拒绝冒充原 run。完整缓存块、行对齐、终态 head 和阶段输出哈希均校验，失败工件保留。小型随机 CUDA 试验已验证完整 Opt/AMP/RNG 续接一致；本轮正常正式作业没有人为中断重放。

原 owner 崩溃且所有原 worker 已退出、目标 GPU 无冲突时，才执行 `python run.py restore`；该命令拒绝活 owner。最终验收必须核对原容器完整 ID、Running 状态、restart policy 及真实 HTTP200，不能用历史恢复记录代替当前健康。

结果在 `artifacts/analysis/<backbone>_seed<seed>/` 和 `artifacts/report/`。逐样本 score/probability/label/weight/threshold/prediction 可独立重算 AUC、Brier、FPR、TPR；GT行×预测列 [Real,Fake] 的混淆矩阵为 [[TN,FP],[FN,TP]]。宏 TPR 是方法等权平均，不能用 pooled TP/(TP+FN) 替代。legacy `real_videos` 字段在图像流程中实际计 real 行数，须按单位解释。

最终表覆盖所有五种子、六主流程、尺度/长度控制、30匹配头、3×3转移、2000测试bootstrap和1000独立校准bootstrap、FS/FR/EFS分组，并包含6幅稳定性/源阈值相对分数图。匹配头固定source train、L2=1e-4、最多2000迭代，未收敛须披露，不能借CDF选择超参数或重训。

没有完整独立fake校准，因此TPR95不可用。FPR5只从原FF-val real冻结阈值；不能借FFtest/CDF调阈值。已知FF来源与精确内容的隔离不认证未知EFS/音频驱动祖先。缺任一方法的宏bootstrap重复整条无效，条件95%区间只使用有效重复，并披露RDDM图像分组支持导致的无效次数。五训练种子的SD/范围与条件于固定模型的测试/校准bootstrap是不同不确定性，分别报告。

本轮完整终态保留全部Opt/AMP/RNG；冻结代码没有逐step AMP skip事件日志，具体跳步位置/当时scale不可恢复。这是已披露的日志缺口，不能推算、补造或重放正式训练。
