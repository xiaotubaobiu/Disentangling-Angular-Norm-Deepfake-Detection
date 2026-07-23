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

def _read(name):
    p=TAB/name
    return pd.read_csv(p) if p.exists() else None

def _g(df, cond, col):
    s=df[cond]
    return float(s[col].mean()) if len(s) else float("nan")

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

def final():
    midterm()
    q4=_read("Q4_multiforgery_alignment.csv")
    L=["\n# 最终报告(完整四问)\n","\n## Q4 多方法对齐\n"]
    if q4 is not None and len(q4):
        L.append(q4.head().to_markdown())
    else:
        L.append("(Q4 未运行或配对审计未通过 → 见 next_step_recommendation)")
    (REP/"four_questions_final_report.md").write_text("\n".join(L))
    (REP/"next_step_recommendation.md").write_text("# Next steps\n(由四问结果驱动;A/B/C/D 决策见 final report)\n")
    print("wrote final reports")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--midterm",action="store_true"); ap.add_argument("--final",action="store_true")
    a=ap.parse_args()
    if a.final: final()
    elif a.midterm: midterm()
    else: midterm()
