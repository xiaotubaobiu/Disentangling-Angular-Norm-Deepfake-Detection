# 深伪检测四问题实证检验 — 设计文档 (v2, 评审修订版)

- 日期:2026-07-23(v2 修订:2026-07-23)
- 范围:`/home/z/project/deepfake_detection`
- 训练域 FF++,测试域 CDF,**DFD 完全排除**(不读、不汇总、不讨论)
- 3 backbone(CLIP / DINOv2 / EfficientNet)× 5 seed(`[7, 42, 123, 999, 2048]`)
- raw-trained 支持三种推理(raw / direction / length),norm-trained 支持 norm 推理
- norm-trained 仅作补充对照,**不进入 Q1/Q2/Q3 的 raw-trained 内部干预分析**
- 计算资源:仅允许 GPU 0–5(`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5`,并行上限 6)

> v2 关键立场修正:
> 1. **Q1 的 `AUC_dir−AUC_raw = H−B` 是定义恒等式,只做正确性自检,不作机制证明**。机制由「翻转样本解剖 + 部分归一化曲线」提供。
> 2. **Q4 的公共方向 AUC 只作「训练监督可行性诊断」,不是可部署单图检测性能**。

---

## 1. 已确认的数据资产

| 资产 | 路径 | 说明 |
|---|---|---|
| 帧级分数 | `outputs/frame_midt_diag/parallel_20260721_213914/per_frame_scores.parquet` (39.8M 行) | `checkpoint_id, backbone, seed, train_mode, infer_mode∈{raw,direction,length,normtrain_norm}, domain∈{ffpp,cdf,dfd}, method_id, label, video_id, image_path, feature_norm(=r), real_logit, fake_logit, signed_logit(=s), prob_fake, prediction_correct` |
| 视频级分数 | `.../per_video_scores.parquet` (982K 行) | `video_prob_fake, mean_feature_norm, std_feature_norm, mean_signed_logit, std_signed_logit, num_frames` |
| 方法级指标 | `.../per_method_metrics.csv` | 每方法 auc/ba/threshold_regret/score 均值方差 |
| delta-AUC / bootstrap | `cdf_delta_auc_per_method.csv`、`bootstrap_direction_vs_raw.csv` | 已有 direction vs raw、video-level paired bootstrap |
| checkpoints | `experiments/final_results/checkpoints/<backbone>/<raw|normtrain>/seed_<s>__<ts>/output/best_model.pth` | 30 个;线性头 `nn.Linear(D,2)` |
| 原始数据 | image_path 指向 `/Dataset/.../FaceForensics++/...` | 同帧对齐、split 恢复、canonical key 的来源 |

**关键事实(已验证):** 分类头为单层线性 `nn.Linear(D,2)` → `s_raw=wᵀz+b`、`s_dir=wᵀu+b`(`u=z/‖z‖`)乘法分解严格成立;三种推理来自同一 raw-trained checkpoint 的同一帧(按 `checkpoint_id+image_path+infer_mode` 对齐);direction>raw 在 15/15 CDF 上确认;同一真实帧按方法被复制多份(采样前需去重);**原始向量 z 未落盘**,Q4 需用 `diagnosis/feature_extractor.py` 重抽。

---

## 2. 全局口径与约定

1. **score**:`s_raw = signed_logit|infer_mode=raw`;`s_dir = signed_logit|infer_mode=direction`;`q = log(feature_norm+1e-12)`;`r = feature_norm`;`u = z/r`(Q4 抽取后)。
2. **DFD 排除**:`domain != 'dfd'`,丢弃 `dfd_native`/`DFD_*`。
3. **raw-trained 为主**;normtrain 仅补充对照列。
4. **real 去重 + cluster bootstrap**:去重键含 `domain`(见 §3 canonical_frame_id);CI 以 `video_id`/`source_video_id` 为 cluster。**8.4M 帧不视为独立样本。**
5. **固定随机种子**(`configs/study_config.yaml`)。
6. **seed 分开算再汇总** mean±std。
7. **frame-level + video-level 双报**。

---

## 3. 第 0 步:`00_inspect_project.py` — schema + canonical key + split 审计

产出 `reports/schema_report.md`,含:文件/列/行数/含义;各 `backbone/seed/method/domain/label` 样本数与**视频数**(去重后);重复行/泄漏迹象;real 复制情况。

**【修正3-A】canonical key 构造与验证(不得假设字段存在):**
`source_video_id` 与 `frame_id` **不是已确认字段**。必须先构造并验证 `canonical_frame_id`,优先级:
1. 原始 manifest 的 `domain + video_id + frame_index`(首选);
2. 标准化后的绝对 `image_path`;
3. 路径解析 + **人工抽查 ≥100 条**。
- real 去重键 = `(domain, source_video_id, canonical_frame_id)`,**必须含 domain**(防跨域同名合并)。
- 同时构造 `source_video_id`(real 的为其自身;fake 的为对应真实源)。

**【修正3-B】FF++ split 可恢复性(必查项 A):** 判定能否为 `domain=ffpp` 行恢复 train/val/test。
- 能 → Q3 用 FF++ **validation** 定向(严格首选)。
- 不能 → 回退:Q3 用 FF++ **train** 定向并标注(仍无测试泄漏)。**绝不用 test 或 CDF 定向。**

**【修正3-C】Q4 配对审计(必查项 B,见 §7):** 审计通过后才允许 Q4 抽特征。

---

## 4. Q1 — rank-flip + 翻转解剖 + 部分归一化(为什么 FSGAN 升、SadTalker 降)

### 4.1 恒等式(仅作正确性自检)
- CC=dir对∧raw对;WW=dir错∧raw错;**H=dir对∧raw错**;**B=dir错∧raw对**。
- 无 ties 时 `AUC_dir−AUC_raw = H−B`。`identity_error=|(auc_dir−auc_raw)−(H−B)|`。
- **【修正2】identity_error 的合理条件**:AUC、H、B 必须**基于同一批 pair**(同全量或同采样)并用精确二维计数(Mann–Whitney);此时误差应接近机器误差。**禁止拿完整 AUC 与 20 万随机 pair 的 H−B 比较后强求 <1e-4**。
- **ties 处理**:报告 `tie_rate`(s(fake)==s(real));严格 CC/WW/H/B 用 `>`;恒等校验用与 AUC 一致的 0.5 权重版。

### 4.2 两套采样口径【修正2】
| 名称 | 抽样方式 | 用途 |
|---|---|---|
| **Native frame AUC** | fake 帧等权、real 帧等权 | 对齐现有热图/论文主指标 |
| **Video-balanced AUC** | 先等权抽视频,再每视频等帧数 | 检查长视频是否主导 |
- 两套分别计算 AUC、CC/WW/H/B。**不得用 video-balanced 的 H−B 解释 native frame AUC。**
- 每组合 ≥20 万 pair(real/fake 视频等权、每视频等帧数),固定种子;小方法全量、大方法分层。

### 4.3 视频级口径【修正1】
- 先把每视频帧分数聚合为 video score(`mean`,以 `median`/`top-k mean` 作稳健对照);
- 再对该方法 fake videos vs 去重 real videos 做相同 AUC/CC/WW/H/B;real/fake **视频等权**。

### 4.4 全量自检先行
CLIP / seed 42 / FSGAN、SadTalker / FF++、CDF **全量**计算,确认(同批 pair、精确计数下)`identity_error` 接近机器误差,再扩展采样版。

### 4.5 翻转样本解剖【修正1 — 机制证据】
对 H 类与 B 类 pair 分别统计:
- `m_dir = s_dir(f)−s_dir(r)`、`m_raw = s_raw(f)−s_raw(r)`、`ρ = log(‖z_r‖/‖z_f‖)`。
- 至少回答:H 类中 fake 模长偏大还是 real?direction margin 是否本就很小(易被模长翻转)?FSGAN vs SadTalker 的 ρ 分布差异?该差异 FF++→CDF 如何变?

### 4.6 部分归一化曲线【修正1 — 自适应设计依据】
- `z_γ = ‖z‖^γ · u`(γ=1→raw;γ=0→direction);`s_γ = wᵀz_γ+b`(两 logit 头用 `w=w_fake−w_real, b=b_fake−b_real`)。
- 扫描 γ∈[0,1](如 11 点),对每方法求最优 γ(FF++ 选、CDF 报)。
- **判定**:FSGAN 最优 γ→0、SadTalker→1,才为「自适应使用模长」提供设计依据。

### 4.7 输出
`outputs/tables/Q1_rank_flip.csv`(backbone,seed,method,domain,view∈{native,video},auc_raw,auc_dir,delta_auc,CC,WW,H,B,H_minus_B,identity_error,tie_rate);`Q1_flip_dissection.csv`、`Q1_partial_norm_curve.csv`。
**图**:`Q1_H_B_FSGAN_SadTalker.png`、`Q1_H_minus_B_vs_delta_auc.png`、`Q1_FFPP_vs_CDF_rank_flip.png`、`Q1_partial_norm_gamma.png`、`Q1_rho_distribution.png`。

### 4.8 判定
恒等式仅证明「ΔAUC 由多少排序翻转构成」;**机制结论依赖 §4.5 解剖 + §4.6 γ 曲线**。FSGAN 多数 H>B 且 γ→0、SadTalker B>H 且 γ→1 → 支持;仅个别 backbone → 部分支持;γ 无差异或解剖不支持 → 不支持,回查头部线性性/score 定义/checkpoint 一致性。

---

## 5. Q2 — 径向稳定性(模长真假关系是否跨域稳定)

### 5.1 指标
- `delta(m,d) = [mean(q|fake,m,d)−mean(q|real,d)] / sqrt((var(q|fake,m,d)+var(q|real,d))/2)`,d∈{ffpp,cdf}。
- `RTG = |delta_cdf − delta_ffpp|`(**绝对**,丢方向)。
- **【修正4】补充带方向指标**:`STG = delta_cdf − delta_ffpp`;`R_retain = sign(delta_ffpp)·delta_cdf`。
  - 借此区分:① 同号强度变;② 符号翻转;③ 训练域有信号、目标域消失(R_retain≈0);④ 目标域反向(R_retain<0)。
- `length_auc_{ffpp,cdf}`、`dir_minus_raw_cdf`(取自 Q1)。

### 5.2 统计单位(防伪重复)【修正4】
- **不得把 16 method×5 seed=80 当 80 个独立样本。**
- **主分析**:每 backbone、method 先对 5 seed 取均 → ~16 方法级点 → 在方法级点上算 Pearson、Spearman、permutation test;**LOMO 检查是否由 FSGAN/SadTalker 单点驱动**。
- **次要分析**:seed 作重复测量 → 混合效应模型或 cluster-robust regression(method 为固定效应,seed 为重复)。
- **按 backbone 分别算,三 backbone 不混合。** video-cluster bootstrap 95% CI。

### 5.3 输出与判定
`outputs/tables/Q2_radial_stability.csv`(backbone,seed,method,delta_ffpp,delta_cdf,RTG,STG,R_retain,sign_flip,length_auc_ffpp,length_auc_cdf,dir_minus_raw_cdf)。
**图**:`Q2_delta_FFPP_vs_CDF.png`、`Q2_RTG_vs_normalization_gain.png`、`Q2_method_radial_effect_forest.png`。
**判定**:RTG 越大 → direction 提升 越大,支持「归一化删除跨域不稳定径向作用」;**Q2 只提供关联证据,不证因果**(raw 效果还依赖模长×angular 联合分布)。无相关 → 转 angle-length 联合分布/图像质量(写 next_step)。

---

## 6. Q3 — length-only 稳健性(CLIP 高 length-only AUC 是否真实)

### 6.1 定向粒度【修正5】
- `score_length = q`。
- **主结果:checkpoint 级全局定向** `o_{backbone,seed}` — 在 FF++ **validation 全方法**上定一个统一方向:`o=+1 if AUC_FF,val(q)≥0.5 else −1`;`s_length=o·q`;固定应用到 FF++ test 与 CDF;**禁止按 CDF 翻转**。(最符合部署:模型不知测试样本属何方法。)
- **辅助:逐方法定向** `o_{backbone,seed,method}`,仅称**「诊断上界」**(假定推理时已知方法),不作主结论。
- split 依赖见 §3 必查项 A;val 不可得 → 回退 FF++ **train** 定向并标注。

### 6.2 指标(每 backbone×seed)
帧 micro / 方法 macro / video-weighted macro / video-level(mean,median,top-k)/ equalized-sampling(≥100 次) / 各方法 / 三 backbone 对比。
**【修正5】三点补充:**
- pooled micro AUC 中 **real 只保留一份**(不因 16 方法复制 16 次)。
- top-k 的 k **预固定或只在 FF++ val 选**(不按 CDF 选)。
- equalized sampling 100 次得到的是**「重采样分布区间」,非统计 CI**;真正 CI 仍需 video-cluster bootstrap。

### 6.3 输出与判定
`outputs/tables/Q3_length_robustness.csv`;图 `Q3_micro_macro_video_auc.png`、`Q3_method_length_auc_forest.png`、`Q3_equalized_sampling_distribution.png`。
**判定**:CLIP 在 macro/video/equalized/多数方法/5 seed 仍明显高于 0.5 → 支持;仅 pooled micro 高 → 「主要由数据组成/样本权重造成」。**不分析 DFD。**

---

## 7. Q4 — 多方法对齐(对比学习是否有依据)

### 7.1 严格执行顺序【修正3/6】
1. **配对审计(硬门,抽 z 之前)**:仅用路径+metadata 构造候选 `group_id`(同 domain/target_video/frame_id;尽可能同 source identity、driver、人脸框、resize、编码);随机抽查 **100–500 组**确认真实帧与各伪造方法**确实对应同一 target frame**;**FS 与 FR 分开**;检查 source identity/driver/帧号/裁剪框/缺失方法。**审计未通过 → Q4 判「数据不足」,不抽特征。**
2. **特征重抽一致性门控**【修正6】:正式抽 15 ckpt 前,先用 1 ckpt、约 1000 帧验证:
   - `|‖z_new‖ − stored feature_norm| ≈ 0`;
   - `head(z_new)` 匹配已有 raw score;
   - `head(z_new/‖z_new‖)` 匹配已有 direction score。
   - 否则(抽错层/preprocessing 不一致/ckpt 加载错/crop·resize 不一致)停止,先修。
3. 审计+门控通过后才抽 15 raw-trained ckpt(FF++/CDF),缓存 `outputs/z_cache/<ckpt>__<domain>.parquet`。
4. FS source identity 或 FR driver 无法保证相同 → 降级 **weak pairing**,显式标注;**FS 与 FR 位移不得混入同一公共方向**。

### 7.2 位移与几何【修正6】
- 对每个 real–fake 配对:`Δr=q(fake)−q(real)`。
- **同帧组内**(同一 `u_r`,同切空间):用 `d_tangent = normalize(u_f − (u_fᵀu_r)u_r)` 或严格 log-map `v=(sinθ/θ)(u_f−cosθ·u_r)`,`θ=arccos(u_fᵀu_r)`;组内两方法 m,n 算 `cos(d(i,m),d(i,n))`。
- **跨帧全局 / PCA / LOFO**:**不得直接平均不同切空间的 tangent**;改用 **ambient chord** `d_simple = u_f − u_r`(或平行移动;第一版用 chord)。
- `p_global = normalize(mean_{i,m} d_simple(i,m))`(仅在 FF++ 估)。

### 7.3 负对照(增强)【修正6】
除随机错配 real(太易区分)外,加:① 同真实帧两次良性增强;② 同帧统一 JPEG 重编码;③ 同帧 resize/blur;④ 同身份相邻帧。
检验:same-frame alignment vs 各负对照:source-video cluster bootstrap + permutation + Wilcoxon signed-rank + Benjamini–Hochberg。

### 7.4 公共方向是否只是分类器方向 w【修正6】
- `ŵ = w/‖w‖`;`d_⊥ = d − (dᵀŵ)ŵ`。
- 比较:原始 displacement alignment vs 投影掉 ŵ 后的 **residual alignment**。投影后公共性消失 → 说明「公共方向」只是 binary classifier 已学的 w,新对比损失价值有限。

### 7.5 泛化(严格命名)【修正6】
- **LOMO 严格称「leave-one-method-out direction estimation」**(留出方法不参与 `p_global` 估计,但 encoder 训练时已见该方法)。
- **不称「unseen forgery generalization」**:真未见需对某 backbone **重训排除方法 m** 再测 m(见附录,范围外)。
- PCA/SVD:第 1 主成分解释方差、前 k 主成分、各方法在公共子空间投影比例。
- CDF 仅用于最终验证。
- **定位**:`p_globalᵀd` 只作「训练监督可行性诊断」,**非可部署单图检测器**;最终方法仍需训练单图分类器。

### 7.6 输出与判定
`Q4_multiforgery_alignment.csv`、`Q4_leave_one_method_out.csv`;图 `Q4_same_frame_vs_random_alignment.png`、`Q4_method_cosine_matrix.png`、`Q4_PCA_explained_variance.png`、`Q4_leave_one_method_out.png`、`Q4_residual_after_classifier_proj.png`。
**判定(六条全满足才支持)**:① same-frame 多方法位移 cosine 明显高于全部负对照;② 多 seed 稳定;③ ≥2 backbone 稳定;④ CDF 中仍存在;⑤ LOMO 方向估计能迁移到留出方法;⑥ residual(去 ŵ)后仍有公共性、且非裁剪/分辨率/编码差异造成。仅训练域/单方法 → 部分支持;与随机错配无差异 → 不支持,建议多 prototype / 方法私有分支 / 相对排序学习。

---

## 8. 统计与实现要求(硬性)
seed 分开算再汇总;8.4M 帧不视为独立样本;bootstrap 以 `video_id`/`source_video_id` 为单位;主要结果 frame+video 双报;**CDF 不用于定向/选参/选公共方向**;DFD 排除;中间结果落 parquet/csv;随机过程固定种子;不只靠 t-SNE;不因结果不符而删方法/seed。

---

## 9. 执行顺序(采纳评审建议)
1. `00_inspect`:canonical frame key + schema + split 审计。
2. Q4 配对审计(硬门)+ **特征重抽一致性门控**(1 ckpt/1000 帧)。
3. Q1:native frame 与 video-balanced 两套 + H/B 翻转解剖 + γ 部分归一化曲线。
4. Q2:方法级统计单位 + STG/R_retain + LOMO。
5. Q3:checkpoint 级全局定向(逐方法作诊断上界)。
6. **生成 `reports/midterm_report.md`**(Q1/Q2/Q3 结果 + 是否值得开发径向—角向方法的初步判断)。**不等 Q4。**
7. Q4:**先单 backbone、单 seed 小规模**确认信号 → 再扩展 15 ckpt。
8. Q4 全局分析用 chord displacement,tangent 仅同帧组内。
9. LOFO 严格称「方向估计留一法」;最终报告 + next_step + A/B/C/D 决策。

---

## 10. 风险与回退
| 风险 | 触发 | 回退 |
|---|---|---|
| source/frame_id 不可构造 | §3 canonical 失败 | 用标准化 image_path + 人工抽查;仍失败则 Q4 数据不足 |
| FF++ split 不可恢复 | 必查项 A 否 | Q3 用 FF++ **train** 定向并标注 |
| 同帧配对不可靠 | Q4 配对审计否 | Q4 判「数据不足」,不抽特征 |
| 特征重抽不一致 | 一致性门控超阈 | 修 layer/preprocessing/ckpt/crop 后再抽 |
| identity_error 不收敛 | Q1 自检超阈 | 回查头部线性性/score 定义/同批 pair/精确计数 |
| RTG 无相关 | Q2 | 改查 angle-length 联合分布/图像质量 |
| 公共方向即分类器 w | Q4 residual 消失 | 报告对比损失增益有限,建议多 prototype |

---

## 11. 附录:论文级扩展(范围外,需额外训练资源,不在当前四问内)
当前四问为**观测性/诊断性**实证。以下主张若要写入论文,需额外受控实验,单独立项:
- **A. 预训练为何有效**:相同结构下 随机初始化 vs ImageNet/CLIP/DINO 预训练;frozen vs full fine-tuning;比较 FF++ 域内与 CDF 跨域差异。(需重训多个变体)
- **B. 模长漂移的图像因素**:Q1/Q2 得结果后,做同图控制干预:JPEG / resize / blur / 超分辨率 / 人脸框大小 / 高频能量,定位模长漂移来源。(需重新抽特征)

---

## 12. 最终决策口径
报告末尾在 A(径向—角向自适应)/ B(同帧多方法对比学习)/ C(两者结合)/ D(放弃其一或两者)中给明确决策,**完全由四问结果驱动,不预设**。中期报告先就 A 给初步判断。注意:决策依据是**机制证据(Q1 解剖+γ、Q2 关联、Q4 公共方向+residual)**,而非 Q1 恒等式或 Q4 配对 AUC。
