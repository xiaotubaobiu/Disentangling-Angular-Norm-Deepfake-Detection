"""Human scientific handoff from accepted immutable artifacts; no fitting or training."""
import pathlib,json,hashlib,datetime,sys,copy
import numpy as np,pandas as pd
P=pathlib.Path('/home/z/project/deepfake_detector');R=(P/'artifacts').resolve();D=P/'reports/final_report'
A=json.loads((P/'reports/final_independent_acceptance.json').read_text());assert A['status']=='INDEPENDENT_FINAL_ACCEPTANCE_PASS'
S=json.loads((P/'reports/final_selected_source_state_check.json').read_text());assert S['status']=='FINAL_SELECTED_SOURCE_STAT_CHECK_PASS'
X=json.loads((D/'supplemental_export_receipt.json').read_text());assert X['status']=='SUPPLEMENTAL_EXPORTS_COMPLETE'
C=json.loads((P/'configs/v1.1.json').read_text());F=json.loads((R/'data/freeze_complete.json').read_text());M=json.loads((P/'reports/completed_model_state_receipts.json').read_text())['completed_models'];K=json.loads((P/'reports/completed_run_cache_receipts.json').read_text())['completed_runs'];probe=json.loads((R/'probe_complete.json').read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
code='7d56c6e5803450369852b5865130a3ed815d87e3e6a91226d276f0fd092611b5'
sys.path.insert(0,str(P/'src'))
from deepfake_detector.orchestration import code_hash
assert code_hash(P)==code
config_identity=hashlib.sha256(json.dumps(C,sort_keys=True).encode()).hexdigest();assert probe['config_identity']==config_identity
resolved=[]
for b in C['backbones']:
 for s in C['seeds']:
  rid=f'{b}_seed{s}';identity=json.loads((R/'runs'/rid/'identity.json').read_text());assert identity['config']==config_identity and identity['code_sha256']==code and identity['backbone']==b and identity['seed']==s and identity['microbatch']==probe['backbones'][b]['microbatch']==128
  assert identity['initialization']==C['backbones'][b]['weight_sha256'] and identity['manifest']==F['manifest_sha256']['train.parquet']
  resolved.append({'run_id':rid,'resolved_after_execution_from_saved_canonical_configuration_and_native_identity':True,'saved_native_identity':identity,'full_canonical_configuration':copy.deepcopy(C),'active_backbone_initialization_and_LR':C['backbones'][b],'optimizer':'AdamW','effective_batch':128,'actual_microbatch':128,'epochs':15,'selection':'final_epoch15','complete_state_sha256':M[rid]['state_sha256'],'final_model_sha256':M[rid]['final_model_sha256'],'original_identity_file_sha256':sha(R/'runs'/rid/'identity.json')})
(P/'reports/resolved_formal_run_configs.json').write_text(json.dumps({'created_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'note':'Explicit handoff resolution of saved immutable canonical config plus actual per-run identities; original execution files and checkpoints remain unchanged.','canonical_config_file_sha256':sha(P/'configs/v1.1.json'),'canonical_sorted_config_identity':config_identity,'runs':resolved},indent=2)+'\n')
t=pd.read_csv(D/'metrics_five_seed_summary.csv');assert t.n.eq(5).all();heads=pd.read_csv(D/'matched_heads.csv');pair=pd.read_csv(D/'paired_metric_differences_all_seeds.csv');coverage=pd.read_csv(D/'relative_score_plot_coverage.csv')
labels={'clip_vit_b16':'CLIP ViT-B/16','dinov2_vit_b14':'DINOv2 ViT-B/14','efficientnet_b0':'EfficientNet-B0'}
def table(cols,rows):return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def value(row,name,percent=False):
 multiplier=100 if percent else 1
 return f'{multiplier*row[name+"_mean"]:.2f} ± {multiplier*row[name+"_SD"]:.2f}' if percent else f'{row[name+"_mean"]:.4f} ± {row[name+"_SD"]:.4f}'
main=[];thresholds=[];images=[]
for b in C['backbones']:
 for domain in ['ff','cdf']:
  q=t[t.backbone.eq(b)&t.domain.eq(domain)&t.unit.eq('video')].set_index('flow')
  main.append([labels[b],domain.upper(),*[value(q.loc[f],'method_macro_AUC') for f in ['RL','RP','DL','DP']]])
  for flow in ['RL','RP','DL','DP']:
   row=q.loc[flow];thresholds.append([labels[b],domain.upper(),flow,value(row,'FPR',True),value(row,'macro_TPR',True),value(row,'macro_BA',True),value(row,'class_balanced_Brier')])
  for flow in ['raw','direction']:
   row=t[t.backbone.eq(b)&t.domain.eq(domain)&t.unit.eq('image')&t.flow.eq(flow)].iloc[0];images.append([labels[b],domain.upper(),flow,value(row,'method_macro_AUC'),value(row,'FPR',True),value(row,'macro_TPR',True),value(row,'macro_BA',True),value(row,'class_balanced_Brier')])
matched=[]
for b in C['backbones']:
 for domain in ['ff','cdf']:
  row=[labels[b],domain.upper()]
  for aggregation in ['L','P','image']:
   for kind in ['raw','direction']:
    g=heads[heads.backbone.eq(b)&heads.domain.eq(domain)&heads.aggregation.eq(aggregation)&heads.kind.eq(kind)];assert len(g)==5;row.append(f'{g.method_macro_AUC.mean():.4f} ± {g.method_macro_AUC.std(ddof=1):.4f}')
  matched.append(row)
contrasts=[]
for b in C['backbones']:
 for domain in ['ff','cdf']:
  row=[labels[b],domain.upper()]
  for name in ['delta_raw','delta_dir','delta_norm_L','delta_norm_P','interaction']:
   g=pair[pair.backbone.eq(b)&pair.domain.eq(domain)&pair.metric.eq('method_macro_AUC')&pair.contrast.eq(name)];v=g.delta_percentage_points;assert len(v)==5;row.append(f'{v.mean():+.3f} ± {v.std(ddof=1):.3f}; {int((v>0).sum())}/{int((v<0).sum())}/{int((v==0).sum())}')
  contrasts.append(row)
controls=A['mechanism_negative_controls'];scale_max=max(x['scale_image_and_mean_logit_AUC_max_abs_difference'] for x in controls);length_max=max(x['global_length_rescale_image_and_mean_logit_AUC_max_abs_difference'] for x in controls);post_max=max(x['post_mean_sigmoid_max_abs_finite_precision_AUC_delta'] for x in controls)
trunc=(coverage.weighted_mass_below_minus4+coverage.weighted_mass_above_plus4).max();budget={k:sum(x[k] for x in M.values()) for k in ['draws','attempts','successes','AMP_skips']};head_unique=heads.drop_duplicates(['backbone','seed','kind']);assert len(head_unique)==30 and head_unique.converged.all()
text=f'''# DF40 v1.1 正式实验结论与交付

15 个正式模型、四角色完整缓存、30 个匹配头与 A–D 分析全部完成，并通过独立终态验收。**本轮主要发现是排序质量与源阈值迁移表现分离，且计分效应依赖骨干、种子和读出。** CDF 视频 RL 的 CLIP / DINO / EfficientNet 宏 AUC 均值分别为 0.8847 / 0.7435 / 0.8611，而 FPR 为 90.17% / 10.28% / 75.14%，对应宏 TPR 为 99.70% / 41.65% / 97.30%。这些阈值各自仅用源真实校准池冻结，源 FPR 为5%，没有用 CDF 调阈值。较高 AUC 不等于该源阈值可控制目标误报，较低 FPR 也须与检出率一起解释。

按原 head 做方向归一化后，CDF 视频 mean-logit 宏 AUC 的五种子均值在三个骨干均下降：CLIP 约24.61pp、DINO 1.40pp、EfficientNet 1.67pp；这不是对其他训练算法或任意归一化方法的结论。DINO 的 RL→DL 降低 FPR 约1.13pp，同时降低宏 TPR 约11.51pp，属于误报与检出的权衡。raw 下概率平均相对 logit 平均的效应在 CLIP/EfficientNet 为负、DINO 均值为正，没有在本轮中获得跨骨干通用的聚合赢家。

本轮是固定协议的**回顾性验证**：研究历史已看过 CDF，不称为前瞻盲测。所有31方法均参与源训练，不声称未见方法泛化。三个骨干各自报告5种子的点、均值、样本SD、范围，不合并为15个同分布种子；以下 ± 均表示跨训练种子的样本SD。

## 视频主表：21 FS/FR 方法

{table(['骨干','测试域','RL 宏AUC','RP 宏AUC','DL 宏AUC','DP 宏AUC'],main)}

RL=raw帧logit平均，RP=raw帧sigmoid概率平均；DL/DP使用单位方向特征并保持原 head。视频固定8个不同RGB帧。完整五种子与min/max见 [metrics_by_seed.csv](final_report/metrics_by_seed.csv)、[metrics_five_seed_summary.csv](final_report/metrics_five_seed_summary.csv)。

## 各流程的源 FPR5 冻结阈值

FPR、宏TPR、宏BA单位为百分比；± 的单位为百分点。每个骨干×种子×流程独立使用同一源real校准池确定阈值，原样迁移。宏BA=(宏TPR+1−FPR)/2。Real=0/Negative，Fake=1/Positive，score越大越像Fake，预测Fake条件为score≥阈值。

{table(['骨干','测试域','流程','FPR %','宏TPR %','宏BA %','类平衡Brier'],thresholds)}

FPR的真实集合在同域/同模型/同流程下共享，不按伪造方法重复计为独立测量。未做概率校准；这里是源real的阈值校准，Brier分解不能证明概率已校准。TPR95缺少完整独立fake校准覆盖，已明确标记不可用。

## 图像/帧表：31 方法

每个自然视频的8帧总权重1，EFS每独立图像权重1；EFS数字存储桶不是自然视频。图像/帧和视频指标分开。

{table(['骨干','测试域','流程','宏AUC','FPR %','宏TPR %','宏BA %','类平衡Brier'],images)}

逐方法及FS/FR/EFS结果见 [metrics_by_seed_method.csv](final_report/metrics_by_seed_method.csv)、[image_FS_FR_EFS_by_seed.csv](final_report/image_FS_FR_EFS_by_seed.csv)。native/real/fake/class-balanced Brier完整值保留于主CSV；legacy `real_videos` 字段在图像流程里实际计real行数，不能当作独立视频数。

## 配对计分效应与机制控制

下表为宏AUC差（pp）的均值±SD；末尾为正/负/零的种子次数。Δraw=RP−RL，Δdir=DP−DL，Δnorm,L=DL−RL，Δnorm,P=DP−RP，交互=Δdir−Δraw。5/5同号是本轮观察，不能推广为普遍规律或未经多重校正的显著性结论。

{table(['骨干','测试域','Δraw','Δdir','Δnorm,L','Δnorm,P','交互'],contrasts)}

固定尺度a=[0.25,0.5,1,2,4]全部报告；c仅由源训练sampling_mass加权长度中位数估计。h、h/c、u、c·u和不同聚合均保留，未按CDF选择尺度。正比例缩放图像/mean-logit排序控制最大AUC偏差={scale_max:.3g}，统一长度缩放的相应最大偏差={length_max:.3g}；同步正仿射变换阈值的全部预测差异为0。3×3 pair转移账本与直接宏AUC差一致，完整表在 [pair_transitions.csv](final_report/pair_transitions.csv)。

聚合后sigmoid理论上保序，但FP64精确端点饱和可产生新ties；本轮记录的post-mean sigmoid最大绝对宏AUC数值差为{post_max:.6g}。这项实际数值效应单独记录，不能把它与帧级sigmoid和多帧平均不交换混为一谈。近端比例、精确端点比例、ties和所有尺度结果见 [scale_controls.csv](final_report/scale_controls.csv)，长度控制见 [length_controls.csv](final_report/length_controls.csv)。

## 30 个冻结特征匹配头

所有30头收敛，实际迭代范围{int(head_unique.iterations.min())}–{int(head_unique.iterations.max())}，上限2000；同一源训练行与sampling_mass，零初始化，全批L-BFGS-B，固定L2=1e-4、不惩罚bias，骨干和normalization状态均冻结。raw头使用h，direction头使用u；未运行可选λ敏感性网格，未用目标标签选择参数。以下为各匹配头的五种子宏AUC；这些重拟合头的结果不复用原head的数值阈值。

{table(['骨干','测试域','raw L','direction L','raw P','direction P','raw image','direction image'],matched)}

CDF上raw与direction重拟合后的mean-logit均值差仍存在；CLIP的差距缩小也伴随raw与direction两边均值下降和较大的种子SD，不能称为方向头救回了目标表现。这里只比较当前表示、线性读出与固定正则化条件，不能推出信息论损失或任何方法使用长度均无用，也不是归一化端到端训练算法对比。所有头的完整五种子、目标指标与收敛状态见 [matched_heads.csv](final_report/matched_heads.csv)。

## 不确定性与图表

2000次测试来源组配对bootstrap与1000次源校准重采样分开，均条件于固定模型、清单、阈值和已声明分组；不合并成总体95%保证。缺任一规定方法时整条宏指标重复无效，不平均缩水方法集合。FF图像宏重复无效次数按seed[7,42,123,2026,3407]为[39,39,39,34,37]，CDF图像为[274,264,247,288,284]，原因是RDDM代理组支持稀少；视频各2000次均有效。未知EFS/音频驱动祖先及CDF/EFS代理组不被认证为人口独立性。详见 [bootstrap_support_structure.json](bootstrap_support_structure.json)、[test_bootstrap.csv](final_report/test_bootstrap.csv)、[calibration_bootstrap.csv](final_report/calibration_bootstrap.csv)。

原6幅图逐byte保留。补充3幅逐种子AUC/FPR配对差图和3幅全观测范围的real相对源阈值加权ECDF（PNG/PDF），全部五种子点保留。原直方图固定bins为[-4,4]，实际被该窗口遗漏的最大加权质量为{100*trunc:.2f}%，逐panel覆盖见 [relative_score_plot_coverage.csv](final_report/relative_score_plot_coverage.csv)；完整ECDF补充了范围外的分布，不改动任何计分或阈值。

'''
for b in C['backbones']:
 text+=f"- {labels[b]}：[主指标稳定性](final_report/{b}_five_seed_stability.png)、[逐种子配对差](final_report/{b}_paired_flow_differences.png)、[real全范围ECDF](final_report/{b}_real_scores_relative_full_range_ECDF.png)。\n"
text+=f'''
## 数据协议、执行证据与复现

作者修订62份官方JSON与本机索引逐byte/语义一致；独立解析实际路径、realpath/inode和像素证明本地存在内容复用，不能仅由JSON结构同构或同seed basename下结论，也不把尚未认证原图压缩包来源链的本地重复归咎作者原档案。原始数据只读，发布角色与逐条来源/去重处置保留；原v1.1设计SHA保持不变，执行记录另写。已知FF来源隔离、选中RGB角色间零交集；未知祖先范围明确披露。官方说明的跨方法训练/测试规则不构成像素唯一性声明，EFS数字桶属于兼容目录，见 [官方/本机核对](official_local_verification.md)、[作者issue14](https://github.com/YZY-stack/DF40/issues/14#issuecomment-2432880783)、[作者issue16](https://github.com/YZY-stack/DF40/issues/16#issuecomment-2428452229)。

全量1514777路径Byte/RGB/解码审计0失败。冻结train250000、FFtest59935、CDFtest368922、real-cal1120；cal来自140个原FF-val视频/70已知FF pair组，fake-cal=0。训练保留265527关联贡献并合并重复RGB输入，共享SG2/3内容保留两方法贡献，不删除方法。所有31方法包括e4s，e4e/FE排除；CDF从未用于训练、校准或选模型/阈值。CDF real id10_0001只有3个不同RGB，不足8帧而未进入视频主表，不是解码失败。

15模型均为epoch15 final，224/实际与有效batch128、AdamW WD5e-4、三个预定LR、五个固定seed，无EMA/早停/调度。总预算{budget['draws']} draws、{budget['attempts']} attempts、{budget['successes']} successful updates、{budget['AMP_skips']} AMP skips。225份真实epoch sampler数组逐一精确重建，同seed的三骨干共享预定抽样序列。Opt/AMP/Python/NumPy/CPU/CUDA RNG完整终态保留；冻结代码未记录逐step AMP skip事件，具体步号和当时scale不可恢复，未补造、补步或重放。小型随机CUDA Opt/AMP/RNG续接一致性已验证；恢复粒度为完成epoch，正式健康作业未人为中断。

CPU回归60项曾通过，三骨干实际GPU资格、隔离pilot及15正式run均通过。两次前置execution失败（3974097预检、3977426 DINO资格）与旧工件保留，旧freeze/错误来源注释/路径resolver修订尝试也保留在归档和原账本；DINO严格确定性CUDA AA插值反传限制在正式队列前修复为CPU FP32可微插值，保留原37×37学习位置网格，随后真实GPU资格与全部DINO正式训练通过。未把pilot或旧实验结果填入主表。

独立验收UTC={A['checked_UTC']}：从raw缓存重算六主流程score/probability、行/标签/权重、源阈值和全部主指标；校准重采样阈值独立重建，测试bootstrap数组/区间/无效次数/配对代数核对；匹配head分数、终态模型/完整状态SHA、阶段输出、负对照和原6PNG核对通过。终态{S['selected_paths_stat_checked']}个选中路径size/mtime与原审计基线一致，属于变化筛查，不是新的逐Byte/RGB证明。

原supervisor3984143自然结束，正式workers全部退出0，原队列与执行锁已释放，无重复启动或新observer/schedule。原onecat-vllm相同完整ID及unless-stopped policy恢复，实际健康HTTP200；GPU0–3/Jev和onecat-gateway未干预。最初stop超时退出137、OOM=false已按实记录，不称为无异常退出。

可复现源码和固定输入/初始化/配置/环境SHA全部保留。计算代码SHA=`{code}`。完整模型、Opt/AMP/RNG、raw缓存、逐样本score和bootstrap数组在`/home/z/workspace/deepfake_detector_v11_20261007`；项目`artifacts`提供快捷链接。小型报告/图表镜像在`reports/final_report`，供本地阅读和源码包携带。15份完整配置由原canonical config与实际保存identity解析，见 [resolved_formal_run_configs.json](resolved_formal_run_configs.json)，明确标记为终态解析，不冒充训练前日志。

[独立验收凭据](final_independent_acceptance.json) · [源文件终态stat核对](final_selected_source_state_check.json) · [15缓存凭据](completed_run_cache_receipts.json) · [训练预算](formal_training_budget_counts.csv) · [手动复现指南](manual_reproduction.md)

复现使用独立checkout/新artifact_root，固定官方修订副本、源候选和初始权重；`prepare`会写该项目主config。单一CLI仍为`run.py`，见手动指南。原始设计与旧项目/旧论文第三条均未改写，本轮没有更新回放。无执行阻碍；TPR95、未知祖先认证和逐step AMP日志缺口按实披露。
'''
f=P/'reports/final_results.md';assert not f.exists();f.write_text(text)
g=P/'reports/manual_reproduction.md';g.write_text(g.read_text().replace('另见 `results.md`','另见 `final_results.md`'))
assert code_hash(P)==code
print(json.dumps({'status':'HUMAN_RESULTS_AND_15_RESOLVED_CONFIGS_WRITTEN','human_report':str(f),'resolved_runs':len(resolved),'compute_code_unchanged':True,'budget':budget,'original_histogram_max_unshown_mass':float(trunc)}))
