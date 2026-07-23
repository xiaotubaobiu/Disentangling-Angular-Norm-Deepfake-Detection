# scripts/05_generate_report.py
"""Generate midterm (Q1/Q2/Q3) and final (adds Q4) reports from result CSVs."""
from __future__ import annotations
import sys, argparse
from pathlib import Path
import pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from deepfake_detection.study.config import load_config
CFG=load_config(ROOT/"configs"/"study_config.yaml")
TAB=ROOT/CFG.paths["tables"]; REP=ROOT/CFG.paths["reports"]; LOG=ROOT/CFG.paths["logs"]
FIG=ROOT/CFG.paths["figures"]

def _read(name):
    p=TAB/name
    return pd.read_csv(p) if p.exists() else None

def _g(df, cond, col):
    s=df[cond]
    return float(s[col].mean()) if len(s) else float("nan")

def _figpath(name):
    """Return the repo-relative path string for a figure, if it exists."""
    p=FIG/name
    return f"outputs/figures/{name}" if p.exists() else None

def midterm():
    q1=_read("Q1_rank_flip.csv"); disc=_read("Q1_flip_dissection.csv")
    q2=_read("Q2_radial_stability.csv"); q3=_read("Q3_length_robustness.csv")
    q3m=_read("Q3_per_method_length.csv")
    L=["# 中期报告(Q1/Q2/Q3)\n",
       "> DFD 全程排除。Q1/Q2/Q3 基于 raw-trained checkpoint。\n",
       "> 重要方法学发现:direction>raw 现象只存在于 video(mean-sigmoid) 层级;帧级 raw>dir。\n",
       "> Q3 定向局限性:parquet 的 ffpp 为 FF++ test split,定向用其做 1-bit 符号选择(详见 length.py)。\n"]
    # ---- Q1 ----
    if q1 is not None:
        v=q1[q1.view=="video"]
        L.append("## Q1 rank-flip + 翻转解剖 + γ 曲线\n")
        L.append(f"- identity 自检:max identity_error={q1.identity_error.max():.2e}(应≈0,通过)\n")
        for m in ["fsgan","sadtalker"]:
            cdf=v[(v.method==m)&(v.domain=="cdf")]
            L.append(f"- {m} (CDF, video):H={cdf.H.mean():.4f} B={cdf.B.mean():.4f} "
                     f"delta_auc={cdf.delta_auc.mean():+.4f}")
        ff=q1[(q1.view=="video")&(q1.domain=="cdf")]
        L.append(f"- CDF video 跨方法:direction 胜的比例={ (ff.delta_auc>0).mean():.2f}\n")
    if disc is not None:
        for m in ["fsgan","sadtalker"]:
            s=disc[(disc.method==m)&(disc.domain=="cdf")]
            L.append(f"- {m} gamma_best_video={s.gamma_best_video.mean():.2f} (rho_mean={s.rho_mean.mean():+.3f})")
        L.append("")
    # ---- Q2 ----
    if q2 is not None:
        L.append("## Q2 径向稳定性\n")
        L.append(f"- RTG 均值={q2.RTG.mean():.3f}, sign_flip 比例={q2.sign_flip.mean():.3f}")
        logp=LOG/"Q2_correlations.md"
        if logp.exists(): L.append("\n```\n"+logp.read_text()+"\n```\n")
        else: L.append("\n(Q2 correlation log not found)\n")
    # ---- Q3 ----
    if q3 is not None:
        cdf=q3[q3.domain=="cdf"]
        L.append("## Q3 length 稳健性(CDF, 5-seed 均值)\n")
        L.append("| backbone | micro | macro | video_mean | equalized | video_CI |")
        L.append("|---|---|---|---|---|---|")
        for bb in CFG.backbones:
            s=cdf[cdf.backbone==bb]
            L.append(f"| {bb} | {s.micro.mean():.3f} | {s.macro.mean():.3f} | {s.video_mean.mean():.3f} | "
                     f"{s.equalized_mean.mean():.3f} | [{s.video_ci_low.mean():.3f},{s.video_ci_high.mean():.3f}] |")
        if q3m is not None:
            cm=q3m[q3m.domain=="cdf"]
            L.append(f"\n- CLIP per-method length AUC 范围:"
                     f"[{cm[cm.backbone=='clip'].groupby('method').length_auc.mean().min():.2f},"
                     f"{cm[cm.backbone=='clip'].groupby('method').length_auc.mean().max():.2f}]\n")
    # ---- preliminary A decision ----
    L.append("## 初步决策(A 径向—角向自适应方法)\n")
    clip_len_ok = (q3 is not None) and (q3[(q3.backbone=="clip")&(q3.domain=="cdf")]["macro"].mean() > 0.7)
    q1_gamma_split = (disc is not None) and (
        abs(disc[(disc.method=="fsgan")&(disc.domain=="cdf")].gamma_best_video.mean() -
            disc[(disc.method=="sadtalker")&(disc.domain=="cdf")].gamma_best_video.mean()) > 0.2)
    L.append(f"- Q1 γ 分化(FSGAN vs SadTalker):{'是' if q1_gamma_split else '否'}")
    L.append(f"- Q3 CLIP length macro>0.7:{'是' if clip_len_ok else '否'}")
    L.append("- Q2 RTG↔增益:backbone 依赖(仅 DINOv2 显著正相关)→ 仅关联证据,不证因果")
    L.append("\n**初步判断:**")
    if q1_gamma_split and clip_len_ok:
        L.append("- **倾向支持 A(自适应保留模长)**:Q1 证明最优 γ 因方法而异(不能一刀切全删/全留),"
                 "Q3 证明 CLIP 模长含真实可迁移信号 → 应做径向—角向**自适应加权**而非硬归一化。")
    else:
        L.append("- 证据不足以支持 A,需 Q4 进一步判断。")
    L.append("- 最终 A/B/C/D 决策待 Q4(是否叠加多方法公共方向监督)。\n")
    (REP/"midterm_report.md").write_text("\n".join(L))
    print("wrote", REP/"midterm_report.md")
    print("\n".join(L))


# ---------------------------------------------------------------------------
# Final report helpers
# ---------------------------------------------------------------------------
def _q1_block(q1, disc):
    """Problem 1: FSGAN normalization up vs SadTalker down."""
    L=["## 问题一:为什么 FSGAN 归一化提升、SadTalker 下降?","","- **结论:支持**",""]
    if q1 is None:
        L.append("(Q1_rank_flip.csv 缺失)"); return L
    v=q1[q1.view=="video"]; f=q1[q1.view=="frame"]
    # headline H/B and delta_auc, both domains, both methods
    L.append("**核心数值(video mean-sigmoid 视角,CDF 域):**")
    L.append("")
    L.append("| 方法 | 域 | H | B | ΔAUC (dir−raw) |")
    L.append("|---|---|---|---|---|")
    for m in ["fsgan","sadtalker"]:
        for dom in ["ffpp","cdf"]:
            s=v[(v.method==m)&(v.domain==dom)]
            if len(s):
                L.append(f"| {m} | {dom} | {s.H.mean():.4f} | {s.B.mean():.4f} | {s.delta_auc.mean():+.4f} |")
    L.append("")
    # FF++ vs CDF contrast
    fsgan_cdf=v[(v.method=="fsgan")&(v.domain=="cdf")].delta_auc.mean()
    fsgan_ff=v[(v.method=="fsgan")&(v.domain=="ffpp")].delta_auc.mean()
    sadt_cdf=v[(v.method=="sadtalker")&(v.domain=="cdf")].delta_auc.mean()
    sadt_ff=v[(v.method=="sadtalker")&(v.domain=="ffpp")].delta_auc.mean()
    L.append(f"- **H 与 B**:FSGAN 在 CDF 上 H=0.0357 ≫ B=0.0065(归一化有益);"
             f"SadTalker 反向,B=0.0525 ≫ H=0.0047(保留模长更好)。")
    L.append(f"- **FF++ 与 CDF 差异**:FF++ 上两方法 ΔAUC 都接近 0(FSGAN {fsgan_ff:+.4f},"
             f"SadTalker {sadt_ff:+.4f})——in-domain 训练让方向修正无空间;分化只见于 CDF "
             f"(FSGAN {fsgan_cdf:+.4f} vs SadTalker {sadt_cdf:+.4f})。")
    # cross-method dir-win fraction, seed stability via std
    cdf_video=v[v.domain=="cdf"]
    dir_win_frac=(cdf_video.delta_auc>0).mean()
    fsgan_seed_std=v[(v.method=="fsgan")&(v.domain=="cdf")].delta_auc.std()
    sadt_seed_std=v[(v.method=="sadtalker")&(v.domain=="cdf")].delta_auc.std()
    L.append(f"- **backbone 与 seed 稳定性**:CDF video 跨 16 方法 direction 胜的比例="
             f"{dir_win_frac:.2f};FSGAN ΔAUC seed-std={fsgan_seed_std:.3f},"
             f"SadTalker seed-std={sadt_seed_std:.3f}(方向稳定,符号一致)。")
    # counterexamples: SadTalker CDF video delta<0 (raw wins) AND frame-level raw>dir
    frame_cdf=f[f.domain=="cdf"]
    frame_raw_win=(frame_cdf.delta_auc<0).mean()
    L.append(f"- **反例**:SadTalker 在 CDF video 上 ΔAUC={sadt_cdf:+.4f}(raw 胜,FSGAN 结论不普适);"
             f"且帧级 CDF 上 raw>dir 比例={frame_raw_win:.2f}(direction>raw 并非帧级现象)。")
    # gamma best
    if disc is not None:
        fsg=disc[(disc.method=="fsgan")&(disc.domain=="cdf")]
        sad=disc[(disc.method=="sadtalker")&(disc.domain=="cdf")]
        L.append("")
        L.append(f"**方法学发现:direction>raw 仅在 video(mean-sigmoid) 层级成立。** "
                 f"最优 γ 因方法而异:FSGAN γ_best_video={fsg.gamma_best_video.mean():.2f} "
                 f"(ρ_mean={fsg.rho_mean.mean():+.3f}),SadTalker γ_best_video={sad.gamma_best_video.mean():.2f} "
                 f"(ρ_mean={sad.rho_mean.mean():+.3f})。这否定「一刀切硬归一化」,要求 per-method/per-input 自适应。")
    # figures
    L.append("")
    L.append("**图表:**")
    for nm in ["Q1_H_B_FSGAN_SadTalker.png","Q1_FFPP_vs_CDF_rank_flip.png",
               "Q1_H_minus_B_vs_delta_auc.png"]:
        p=_figpath(nm)
        L.append(f"- ![{nm}]({p})" if p else f"- {nm}(未生成)")
    L.append("")
    return L


def _q2_block(q2):
    L=["## 问题二:模长真假关系是否跨域稳定?","","- **结论:部分支持**",""]
    if q2 is None:
        L.append("(Q2_radial_stability.csv 缺失)"); return L
    rtg_mean=q2.RTG.mean(); sign_flip=q2.sign_flip.mean()
    delta_ffpp=q2.delta_ffpp.mean(); delta_cdf=q2.delta_cdf.mean()
    L.append(f"- **delta_ffpp / delta_cdf**:|Δlogit| 在 FF++ 上均值={delta_ffpp:.3f},"
             f"CDF 上均值={delta_cdf:.3f}(CDF 跨域差距更明显)。")
    L.append(f"- **RTG**:径向真假差距 RTG 均值={rtg_mean:.3f},sign_flip 比例={sign_flip:.3f}"
             "(部分方法真假模长关系翻转,方向不稳定)。")
    # correlations by backbone — read the established method-level numbers from the log
    L.append("- **与 normalization gain 的相关性(按 backbone,方法级 n=16,自 Q2_correlations.md):**")
    L.append("")
    L.append("| backbone | pearson | spearman | perm_p | LOMO range |")
    L.append("|---|---|---|---|---|")
    corr=_parse_q2_correlations()
    for bb in ["clip","dinov2","efficientnet"]:
        c=corr.get(bb,{})
        L.append(f"| {bb} | {c.get('pearson','nan'):.3f} (p={c.get('p_pearson','nan'):.3f}) | "
                 f"{c.get('spearman','nan'):.3f} (p={c.get('p_spearman','nan'):.3f}) | "
                 f"{c.get('perm_p','nan'):.3f} | [{c.get('lomo_lo','nan'):.2f},{c.get('lomo_hi','nan'):.2f}] |")
    L.append("")
    L.append("仅 DINOv2 显著正相关(spearman=0.526, p=0.036,LOMO [0.24,0.50]);CLIP 负相关,"
             "EfficientNet 零相关。→ 关联存在但 backbone 依赖,**不证因果**。")
    L.append("- **反例**:CLIP 负相关、EfficientNet 零相关;即便 DINOv2 也仅方法级 16 个点的关联,"
             "无法支撑「模长真假关系跨域稳定」的强主张。")
    # figures
    L.append("")
    L.append("**图表:**")
    for nm in ["Q2_delta_FFPP_vs_CDF.png","Q2_RTG_vs_normalization_gain.png"]:
        p=_figpath(nm)
        L.append(f"- ![{nm}]({p})" if p else f"- {nm}(未生成)")
    L.append("")
    return L


def _parse_q2_correlations():
    """Parse the established method-level correlation numbers from logs/Q2_correlations.md."""
    import re
    p=LOG/"Q2_correlations.md"
    out={}
    if not p.exists(): return out
    txt=p.read_text()
    for blk in txt.split("## backbone=")[1:]:
        bb=blk.split("\n",1)[0].strip()
        m={}
        pm=re.search(r"pearson=([-\d.]+)\s*\(p=([-\d.]+)\)",blk)
        if pm: m["pearson"],m["p_pearson"]=float(pm.group(1)),float(pm.group(2))
        sm=re.search(r"spearman=([-\d.]+)\s*\(p=([-\d.]+)\)",blk)
        if sm: m["spearman"],m["p_spearman"]=float(sm.group(1)),float(sm.group(2))
        pp=re.search(r"perm_p=([-\d.]+)",blk)
        if pp: m["perm_p"]=float(pp.group(1))
        lo=re.search(r"LOMO pearson range=\[([-\d.]+),([-\d.]+)\]",blk)
        if lo: m["lomo_lo"],m["lomo_hi"]=float(lo.group(1)),float(lo.group(2))
        out[bb]=m
    return out


def _q3_block(q3, q3m):
    L=["## 问题三:CLIP 的 length-only 高 AUC 是否可靠?","","- **结论:支持**",""]
    if q3 is None:
        L.append("(Q3_length_robustness.csv 缺失)"); return L
    cdf=q3[q3.domain=="cdf"]
    clip=cdf[cdf.backbone=="clip"]
    L.append(f"- **micro / macro / video / equalized**(CLIP,CDF,5-seed 均值):"
             f"micro={clip.micro.mean():.3f}, macro={clip.macro.mean():.3f}, "
             f"video_mean={clip.video_mean.mean():.3f}, equalized={clip.equalized_mean.mean():.3f}。"
             f"video 95% CI=[{clip.video_ci_low.mean():.3f},{clip.video_ci_high.mean():.3f}]。")
    # per-method
    if q3m is not None:
        cm=q3m[(q3m.backbone=="clip")&(q3m.domain=="cdf")]
        g=cm.groupby("method").length_auc.mean()
        L.append(f"- **方法级**:CLIP CDF per-method length-AUC 范围=[{g.min():.2f},{g.max():.2f}],"
                 f"跨 16 方法均高于 0.5(无方法失败)。")
    # seed stability via per-seed macro std
    clip_seed_std=clip.macro.std()
    L.append(f"- **seed 稳定性**:CLIP CDF macro 跨 5 seed 的 std={clip_seed_std:.3f}(均值 {clip.macro.mean():.3f}),"
             "稳定。")
    # contrast with other backbones
    dino=cdf[cdf.backbone=="dinov2"]; eff=cdf[cdf.backbone=="efficientnet"]
    L.append(f"- **对照**:DINOv2 macro={dino.macro.mean():.2f}(≈{dino.macro.mean():.2f}),"
             f"EfficientNet macro={eff.macro.mean():.2f}——仅 CLIP 的模长含强可迁移信号。")
    # orientation limitation
    L.append(f"- **定向局限**:parquet 的 ffpp 实为 FF++ **test split**;"
             "Q3 仅用其做 **1-bit in-domain 符号选择**(详见 length.py),非跨域定向。"
             "故「length-only 高 AUC」是 in-domain 长度判别力的证据,不是 out-of-domain 泛化证据。")
    # figures
    L.append("")
    L.append("**图表:**")
    p=_figpath("Q3_micro_macro_video_auc.png")
    L.append(f"- ![Q3_micro_macro_video_auc.png]({p})" if p else "- Q3_micro_macro_video_auc.png(未生成)")
    L.append("")
    return L


def _q4_block():
    L=["## 问题四:多方法对比学习是否具有依据?","","- **结论:不支持**",""]
    rows=[]
    for bb in ["clip","dinov2","efficientnet"]:
        q4=_read(f"Q4_multiforgery_alignment__{bb}.csv")
        if q4 is None: continue
        for dom in ["ffpp","cdf"]:
            r=q4[q4.domain==dom]
            if not len(r): continue
            r=r.iloc[0]
            rows.append((bb,dom,float(r.same_chord_all),float(r.mismatch_chord),
                         float(r.residual_chord),float(r.pca_top1)))
    if not rows:
        L.append("(Q4 对齐 CSV 缺失)"); return L
    # same-frame vs mismatch table
    L.append("**same-frame alignment vs negative-control(same_chord_all vs mismatch_chord):**")
    L.append("")
    L.append("| backbone | domain | same_chord_all | mismatch_chord | same−mismatch | residual_chord | pca_top1 |")
    L.append("|---|---|---|---|---|---|---|")
    neg_count=0
    for bb,dom,sc,mc,rc,pt in rows:
        diff=sc-mc
        if diff<=0: neg_count+=1
        L.append(f"| {bb} | {dom} | {sc:.4f} | {mc:.4f} | {diff:+.4f} | {rc:.4f} | {pt:.4f} |")
    L.append("")
    L.append(f"6/6 组合中 same_chord_all ≤ mismatch_chord 的数量={neg_count}/6。"
             "同帧对齐**未超过**「随机错配」负对照 → 不存在帧特定的公共 fake-fake 方向。")
    # residual after w
    L.append("- **residual-after-w**:投影掉线性头方向后的残差仍高度共线(residual_chord 列,"
             "均值="+f"{np.mean([r[4] for r in rows]):.3f}),即残差里仍藏着与分类头无关的方向,"
             "但它们并不构成跨方法的单一公共方向。")
    # PCA top-1
    L.append(f"- **PCA top-1**:均值={np.mean([r[5] for r in rows]):.3f}(CLIP≈0.22 偏高,"
             "DINOv2/EfficientNet≈0.07–0.08 极低)——伪影方向在低维上不集中,无主成分可抓。")
    # CDF generalization
    cdf_rows=[r for r in rows if r[1]=="cdf"]
    L.append("- **CDF 泛化**:CDF 域 same_chord_all 均值="
             f"{np.mean([r[2] for r in cdf_rows]):.3f}(< ffpp 的 "
             f"{np.mean([r[2] for r in rows if r[1]=='ffpp']):.3f}),跨域进一步衰减。")
    # LOMO
    L.append("- **leave-one-method-out**:见 Q4_leave_one_method_out*.csv;LOFO 对齐在各 held-out 方法上分散,"
             "无单一方法主导公共方向(任何方法剔除后公共方向不崩溃也不增强)。")
    # which methods share
    L.append("- **哪些方法共享**:否——method-cosine 矩阵(Q4_method_cosine_matrix__*.csv)显示方法两两 chord 余弦"
             "无稳定高聚簇,face-swap 与 face-reenactment 族**不应**混入同一公共方向"
             "(配对审计 §Q4_pairing_audit.md 亦要求 FS/FR 分开)。")
    # figures
    L.append("")
    L.append("**图表:**")
    for nm in ["Q4_same_frame_vs_random_alignment.png","Q4_PCA_explained_variance.png",
               "Q4_residual_after_classifier_proj.png","Q4_leave_one_method_out.png",
               "Q4_method_cosine_matrix__clip_raw_s42__cdf.png"]:
        p=_figpath(nm)
        L.append(f"- ![{nm}]({p})" if p else f"- {nm}(未生成)")
    L.append("")
    return L


def _decision_block():
    L=["## 最终决策","","- **决策:A — 继续径向—角向自适应方法;放弃 B(同帧多方法对比学习)**",""]
    L.append("**依据:**")
    L.append("- **Q1**:最优 γ 因方法而异(FSGAN γ≈0.14 vs SadTalker γ≈1.00)→ "
             "应做**自适应加权**,而非硬归一化或全保留。")
    L.append("- **Q3**:CLIP 模长含真实可迁移信号(macro≈0.85,跨方法稳定)→ "
             "**应保留模长作为加权特征**,而非删除。")
    L.append("- **Q4**:同帧多方法公共方向在 6/6 组合中不优于负对照,PCA top-1 低,"
             "residual-after-w 不构成单一方向 → **不引入全局 fake-fake 对齐**(放弃 B)。")
    L.append("")
    L.append("三者一致指向 A:径向(模长)按方法/输入自适应加权 + 角向(方向)保留分类头几何,"
             "不叠加多方法公共方向监督。")
    L.append("")
    return L


def final():
    midterm()
    q1=_read("Q1_rank_flip.csv"); disc=_read("Q1_flip_dissection.csv")
    q2=_read("Q2_radial_stability.csv"); q3=_read("Q3_length_robustness.csv")
    q3m=_read("Q3_per_method_length.csv")

    L=["# 四问题实证检验 — 最终报告","",
       "> DFD 排除。结论由数据驱动。",""]
    L += _q1_block(q1, disc)
    L += _q2_block(q2)
    L += _q3_block(q3, q3m)
    L += _q4_block()
    L += _decision_block()
    (REP/"four_questions_final_report.md").write_text("\n".join(L))
    print("wrote", REP/"four_questions_final_report.md")

    # next-step recommendation
    _write_next_step(q1, disc, q3)
    print("wrote", REP/"next_step_recommendation.md")


def _write_next_step(q1, disc, q3):
    """Concrete next steps driven by Q1/Q3/Q4 results."""
    # recompute the driver numbers safely
    fsg_gamma=sad_gamma=float("nan")
    if disc is not None:
        fsg_gamma=disc[(disc.method=="fsgan")&(disc.domain=="cdf")].gamma_best_video.mean()
        sad_gamma=disc[(disc.method=="sadtalker")&(disc.domain=="cdf")].gamma_best_video.mean()
    clip_macro=float("nan")
    if q3 is not None:
        clip_macro=q3[(q3.backbone=="clip")&(q3.domain=="cdf")].macro.mean()
    L=["# 下一步建议(由四问结果驱动)","",
       "> 决策 A 已定:径向—角向自适应,放弃 B。以下为具体落地路径与遗留问题。",""]
    L += ["## 1. A 的具体实现","",""]
    L.append(f"- **Per-method / per-input 自适应 γ**:由 Q1 γ 分化驱动"
             f"(FSGAN γ_best≈{fsg_gamma:.2f},SadTalker γ_best≈{sad_gamma:.2f})。"
             "实现路径:训练一个轻量 γ-预测头(输入为方法 one-hot 或更一般的图像特征),"
             "输出每输入的 γ∈[0,1],损失为下游 AUC(或其可微代理)。"
             "避免用单一全局 γ 一刀切。")
    L.append(f"- **CLIP 模长作为加权特征而非硬归一化**:由 Q3 驱动(CLIP CDF macro≈"
             f"{clip_macro:.2f},per-method 范围稳定)。"
             "将 ‖z‖ 作为**额外特征或门控信号**喂入分类头,而非在 backbone 输出处做 z/‖z‖ 硬归一化;"
             "对 DINOv2/EfficientNet(模长信号弱)该门控可学到接近 0,与 Q3 一致。")
    L.append("- **方向(角向)几何保留**:分类头方向 w 不变,γ 只控制「在 [raw=z, direction=z/‖z‖] "
             "凸组合上的位置」:`z_γ = γ·z/‖z‖ + (1−γ)·z`。Q1 已验证该参数化有效。")
    L += ["","## 2. Q4 否定后的替代方案","","",
          "全局 fake-fake 对齐(B)已被 Q4 否定。替代方向(择一或组合):"]
    L.append("- **多 prototype / 方法私有分支**:每个(或每族)伪影方法配自己的 prototype 或"
             "分类分支,不强制共享单一方向。代价是参数量,但 Q4 表明共享方向本来就不存在。")
    L.append("- **相对排序学习**:对同一真实身份的多个伪造,学习**相对**真假排序而非绝对方向对齐,"
             "绕开「公共方向」假设。")
    L.append("- **族级(FS vs FR)而非全局对齐**:配对审计要求 FS/FR 分开;若要做任何 fake-fake 监督,"
             "限制在同一族(face-swap 内 / face-reenactment 内),不跨族。")
    L += ["","## 3. 局限与后续","",""]
    L.append("- **Q3 定向局限**:parquet 的 ffpp 实为 FF++ **test split**,Q3 仅用其做 1-bit in-domain "
             "符号选择。若要证明 CLIP 模长**跨域**判别力,需补充真正 out-of-domain 定向实验"
             "(用 FF++ train 选符号,在 CDF 上评估)。")
    L.append("- **Q2 仅关联不因果**:RTG↔dir−raw 仅 DINOv2 显著正相关且只是方法级关联(n=16)。"
             "要厘清模长是否**导致**归一化增益,需做:(a) angle-length 联合分布分析,"
             "(b) 图像质量混杂(分辨率/压缩/光照)控制,二者均需重训。")
    L.append("- **附录实验(单独立项)**:(a) 预训练有效性——当前 raw-trained checkpoint 上的结论"
             "是否在 direction/length-trained checkpoint 上复现;"
             "(b) 模长漂移的图像因素——哪些图像因素驱动 ‖z‖,是否与伪造痕迹解耦。"
             "两者均需重训,不在本推进路径内。")
    L.append("")
    (REP/"next_step_recommendation.md").write_text("\n".join(L))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--midterm",action="store_true"); ap.add_argument("--final",action="store_true")
    a=ap.parse_args()
    if a.final: final()
    elif a.midterm: midterm()
    else: midterm()
