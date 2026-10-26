# scripts/09_generate_pooling_report.py
"""Exp7 (within-video temporal consistency) + final A/B/C/D report.

Temporal: per video, raw vs direction, temporal_stats (std/mad/iqr/q90-q10/p_pos/
adj_diff/auto_corr/outlier_ratio). Does direction reduce within-video variance/outliers?

Report: reads all Q_pooling_*.csv, applies the data-driven A/B/C/D decision logic
(spec sec 5), writes reports/pooling_saturation_final_report.md (spec sec 16 structure).
DFD excluded; all params fit only on FF++ val.
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

PHASE = Path(__file__).resolve().parents[1]   # experiments/<date>_<phase>/
ROOT = PHASE.parents[1]                              # repository root
sys.path.insert(0, str(ROOT / "src"))

from study_pooling.config import load_config
from study.io import load_frame_subset, dedup_reals
from study_pooling.temporal import temporal_stats

CFG = load_config(PHASE / "configs" / "pooling_saturation_config.yaml")
TAB = PHASE / CFG.paths["tables"]
FIG = PHASE / CFG.paths["figures"]
REP = PHASE / CFG.paths["reports"]
for d in (TAB, FIG, REP):
    d.mkdir(parents=True, exist_ok=True)

METRICS = ["std", "mad", "iqr", "q90_q10", "p_pos", "adj_diff", "auto_corr", "outlier_ratio"]


def _wide(df):
    raw = df[df.infer_mode == "raw"][["backbone", "seed", "domain", "method_id",
        "label", "video_id", "image_path", "signed_logit"]
    ].rename(columns={"signed_logit": "s_raw"})
    d = df[df.infer_mode == "direction"][["image_path", "signed_logit"]
    ].rename(columns={"signed_logit": "s_dir"})
    return raw.merge(d, on="image_path")


def temporal_run():
    fpath = str(ROOT / CFG.parquet_frame)
    if _args.smoke:
        targets = [(CFG.smoke["backbone"], CFG.smoke["seed"], CFG.smoke["methods"])]
    else:
        targets = [(bb, sd, CFG.methods) for bb in CFG.backbones for sd in CFG.seeds]
    rows = []
    for bb, sd, methods in targets:
        for m in methods:
            for dom in CFG.domains:
                df = dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode=None,
                    backbones=[bb], seeds=[sd], domains=[dom], methods=[m, "real"]))
                w = _wide(df)
                if w.empty:
                    continue
                for vid, g in w.groupby("video_id", sort=False):
                    lbl = int(g.label.iloc[0])
                    for mode, col in [("raw", "s_raw"), ("direction", "s_dir")]:
                        st = temporal_stats(g, col)
                        rows.append({"backbone": bb, "seed": sd, "method": m, "domain": dom,
                                     "video_id": vid, "label": lbl, "infer_mode": mode, **st})
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "Q_pooling_temporal_consistency.csv", index=False)
    _temporal_plots(out)
    return out


def _temporal_plots(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if out.empty:
        return
    # Fig 1: within-video std raw vs direction (mean per method)
    pvt = out.groupby(["method", "infer_mode"])["std"].mean().unstack("infer_mode")
    if "raw" in pvt.columns and "direction" in pvt.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(pvt))
        ax.bar(x - 0.2, pvt["raw"], 0.4, label="raw")
        ax.bar(x + 0.2, pvt["direction"], 0.4, label="direction")
        ax.set_xticks(x); ax.set_xticklabels(pvt.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("mean within-video std (logit)")
        ax.set_title("Within-video variance: raw vs direction (by method)")
        ax.legend()
        fig.tight_layout(); fig.savefig(FIG / "Q_pooling_within_video_variance.png", dpi=120); plt.close(fig)

    # Fig 2: temporal consistency gain = (std_raw - std_dir) by method
    gain = out.groupby(["method", "infer_mode"])["std"].mean().unstack("infer_mode")
    if "raw" in gain.columns and "direction" in gain.columns:
        gain["gain"] = gain["raw"] - gain["direction"]
        gain = gain.sort_values("gain")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(np.arange(len(gain)), gain["gain"])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(np.arange(len(gain))); ax.set_xticklabels(gain.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("std(raw) - std(direction)")
        ax.set_title("Temporal consistency gain: positive => direction reduces within-video variance")
        fig.tight_layout(); fig.savefig(FIG / "Q_pooling_temporal_consistency_gain.png", dpi=120); plt.close(fig)
    plt.close("all")


# ---------------------------------------------------------------------------
# Final report (A/B/C/D decision)
# ---------------------------------------------------------------------------
def _read(name):
    p = TAB / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def _mean(df, val_col, by):
    if df.empty:
        return pd.Series(dtype=float)
    return df.groupby(by)[val_col].mean()


def build_report():
    agg = _read("Q_pooling_aggregation_auc.csv")
    sat = _read("Q_pooling_saturation_stats.csv")
    jen = _read("Q_pooling_jensen_gap.csv")
    gam = _read("Q_pooling_gamma_scan.csv")
    temp = _read("Q_pooling_temperature.csv")
    calib = _read("Q_pooling_calibrated_auc.csv")
    scale = _read("Q_pooling_scale_matched_auc.csv")
    rf = _read("Q_pooling_rank_flip_by_aggregator.csv")
    tcon = _read("Q_pooling_temporal_consistency.csv")

    # Delta_prob / Delta_logit / I_pool per method (mean over seed/backbone/domain)
    summ = pd.DataFrame()
    if not agg.empty:
        dp = agg[agg.aggregation == "mean_prob"].groupby(["method", "domain"])["delta_auc"].mean()
        dl = agg[agg.aggregation == "mean_logit"].groupby(["method", "domain"])["delta_auc"].mean()
        summ = pd.DataFrame({"Delta_prob": dp, "Delta_logit": dl})
        summ["I_pool"] = summ["Delta_prob"] - summ["Delta_logit"]

    # calibrated delta (mean_prob after temperature) per method
    if not calib.empty:
        summ["Delta_temp"] = calib.groupby(["method", "domain"])["delta_calib"].mean()
    if not scale.empty:
        summ["Delta_scale"] = scale.groupby(["method", "domain"])["delta_scale"].mean()

    # saturation: mean sat_010 raw vs direction (fake)
    sat_cmp = pd.DataFrame()
    if not sat.empty and "sat_010" in sat.columns:
        s10 = sat[sat.label == "fake"].groupby(["method", "domain", "infer_mode"])["sat_010"].mean().unstack("infer_mode")
        if "raw" in s10.columns and "direction" in s10.columns:
            sat_cmp = s10.rename(columns={"raw": "sat_raw", "direction": "sat_dir"})
            sat_cmp["sat_excess_raw"] = sat_cmp["sat_raw"] - sat_cmp["sat_dir"]

    # Jensen: mean |J| raw vs direction
    jen_cmp = pd.DataFrame()
    if not jen.empty:
        jj = jen.groupby(["method", "domain", "infer_mode"])["mean_abs_J"].mean().unstack("infer_mode")
        if "raw" in jj.columns and "direction" in jj.columns:
            jen_cmp = jj.rename(columns={"raw": "Jabs_raw", "direction": "Jabs_dir"})

    # gamma optimum per method/aggregator (cdf = oracle diagnostic)
    gamma_cmp = pd.DataFrame()
    if not gam.empty:
        gc = gam[gam.domain == "cdf"].copy()
        # best gamma (max video_auc) per (method, aggregation)
        idx = gc.groupby(["method", "aggregation"])["video_auc"].idxmax()
        best = gc.loc[idx, ["method", "aggregation", "gamma"]]
        gamma_cmp = best.pivot_table(index="method", columns="aggregation",
                                     values="gamma", aggfunc="first")

    # temporal: variance reduction per method
    tcon_cmp = pd.DataFrame()
    if not tcon.empty:
        ts = tcon.groupby(["method", "domain", "infer_mode"])["std"].mean().unstack("infer_mode")
        if "raw" in ts.columns and "direction" in ts.columns:
            tcon_cmp = ts.rename(columns={"raw": "std_raw", "direction": "std_dir"})
            tcon_cmp["var_reduction"] = tcon_cmp["std_raw"] - tcon_cmp["std_dir"]
        to = tcon.groupby(["method", "domain", "infer_mode"])["outlier_ratio"].mean().unstack("infer_mode")
        if "raw" in to.columns and "direction" in to.columns:
            tcon_cmp["out_red"] = to["raw"] - to["direction"]

    # ---- A/B/C/D decision (aggregate over methods) ----
    verdict = _decide(summ, sat_cmp, tcon_cmp)

    md = []
    md.append("# Pooling / Saturation 机制研究 — 最终报告\n")
    md.append(f"- 日期:2026-07-27\n- 分支:`study/pooling-saturation`\n- 数据:FF++(test)+ CDF(DFD 排除);raw-trained checkpoint × {{clip,dinov2,efficientnet}} × 5 seeds × 16 methods\n")
    md.append("- 所有校准参数(T / robust-scale / Platt)仅在 **FF++ val** 拟合;CDF γ_best 仅作 oracle 诊断。\n\n")

    md.append("## 0. 结论摘要\n\n")
    md.append(f"**判定:{verdict['label']}**\n\n")
    for line in verdict["evidence"]:
        md.append(f"- {line}\n")
    md.append("\n")
    md.append("**下一步方法建议:** " + verdict["next"] + "\n\n")

    md.append("## 1. 实验1 聚合器对照(prob vs logit 池化)\n\n")
    md.append("`delta_auc = AUC_direction − AUC_raw`(均值 over seed/backbone)。`I_pool = Delta_prob − Delta_logit`。\n\n")
    if not summ.empty:
        md.append(summ.round(4).to_markdown() + "\n\n")
    md.append("- **关键判读**:若 `Delta_prob>0` 但 `Delta_logit≤0`,则 direction 在视频级的「领先」来自概率聚合而非特征本身(指向 A)。\n\n")

    md.append("## 2. 实验2 饱和统计\n\n")
    if not sat_cmp.empty:
        md.append("fake 帧饱和率 @0.10(raw vs direction,均值):\n\n")
        md.append(sat_cmp.round(4).to_markdown() + "\n\n")
        md.append("- `sat_excess_raw>0` ⇒ raw 比 direction 更饱和(支持 A:raw 受 sigmoid 压缩更严重)。\n\n")

    md.append("## 3. 实验3 Jensen gap\n\n")
    md.append("`J = mean σ(s) − σ(mean s)`,per-video `|J|` 均值(raw vs direction):\n\n")
    if not jen_cmp.empty:
        md.append(jen_cmp.round(4).to_markdown() + "\n\n")
    md.append("- direction 是否系统性降低 |J| ⇒ 饱和造成的概率聚合偏移是否被方向归一化削弱。\n\n")

    md.append("## 4. 实验4 温度校准(FF++ val 拟合)\n\n")
    if not temp.empty:
        tt = temp.groupby("backbone")[["T_raw", "T_dir",
            "val_ece_before_raw", "val_ece_after_raw",
            "val_ece_before_dir", "val_ece_after_dir"]].mean().round(3)
        md.append("温度 T 与 ECE(校准前后):\n\n")
        md.append(tt.to_markdown() + "\n\n")
    if "Delta_temp" in summ.columns:
        md.append("温度后 `Delta_temp`(mean_prob)均值 per method:\n\n")
        md.append(summ[["Delta_temp"]].round(4).to_markdown() + "\n\n")
        md.append("- **关键判读**:`Delta_prob>0` 经温度后是否 → `Delta_temp≈0` 或转负?若是 ⇒ 强支持 A(池化伪影被温度消除)。\n\n")

    md.append("## 5. 实验5 尺度匹配(val 拟合 robust scale)\n\n")
    if "Delta_scale" in summ.columns:
        md.append("robust 标准化后 `Delta_scale`(mean_prob):\n\n")
        md.append(summ[["Delta_scale"]].round(4).to_markdown() + "\n\n")
        md.append("- 尺度匹配是否同样抹平方向优势?与温度结论是否一致。\n\n")

    md.append("## 6. 实验6 各聚合器 rank-flip (H−B)\n\n")
    if not rf.empty:
        pvt = rf.groupby(["method", "aggregation"])["H_minus_B"].mean().unstack("aggregation")
        cols = [c for c in ["mean_prob", "mean_logit", "median_logit", "trimmed_logit",
                            "temperature_mean_prob", "standardized_mean_prob"] if c in pvt.columns]
        md.append("mean H−B by method × aggregator:\n\n")
        md.append(pvt[cols].round(2).to_markdown() + "\n\n")
        maxid = rf.identity_error.abs().max()
        md.append(f"- **identity 自检**:max|identity_error| = {maxid:.2e}(应≈0)。\n")
        md.append("- **关键判读**:FSGAN 是否只在 mean_prob 下 H≫B,mean_logit 下 H≈B 或 B>H?SadTalker 是否多数聚合器 B>H?\n\n")

    md.append("## 7. 实验7 视频内时间一致性\n\n")
    if not tcon_cmp.empty:
        md.append("within-video std(raw vs direction)与离群比例下降:\n\n")
        md.append(tcon_cmp.round(4).to_markdown() + "\n\n")
        md.append("- `var_reduction>0` ⇒ direction 降低视频内方差(支持 C);若该改善与 mean_logit AUC 提升相关而饱和/Jensen 不能解释,则判 C。\n\n")

    md.append("## 8. γ 扫描(真实 b,CDF=oracle)\n\n")
    if not gamma_cmp.empty:
        md.append("CDF(oracle)上每个 method × aggregator 的最优 γ(0=direction,1=raw):\n\n")
        md.append(gamma_cmp.round(2).to_markdown() + "\n\n")
        md.append("- **关键判读**:FSGAN 与 SadTalker 最优 γ 是否相反?最优 γ 是否随聚合器剧变?γ 跨聚合器稳定且偏向 0 ⇒ 支持 B。\n\n")

    md.append("## 9. 机制判定(A/B/C/D)\n\n")
    md.append("| 机制 | 证据条件 | 本数据 |\n|---|---|---|\n")
    for row in verdict["table"]:
        md.append(row + "\n")
    md.append("\n")

    md.append("## 10. 下一步方法建议\n\n")
    md.append(verdict["next"] + "\n\n")

    md.append("## 11. 偏差记录\n\n")
    md.append("- FF++ val 用项目 `build_val_loader` 切片(FF++ train videos [586:686]),非官方 140 清单;leak-free,与训练 val 一致(已与用户确认)。\n")
    md.append("- DFD 全程排除。\n")
    md.append("- CDF 上 γ_best / T 评估仅作诊断,从未用于挑选聚合器或正式方法。\n")
    md.append("- bootstrap:video-cluster paired ≥2000;BH 多重比较(在跨方法汇总时应用);帧不作独立样本。\n")

    out = "".join(md)
    (REP / "pooling_saturation_final_report.md").write_text(out, encoding="utf-8")
    return out, verdict


def _decide(summ, sat_cmp, tcon_cmp):
    """Apply spec sec-5 A/B/C/D logic on method-aggregated evidence."""
    evidence = []
    table = []
    if summ.empty:
        return dict(label="数据不足(实验1 表为空)",
                    evidence=["实验1 聚合器表未生成,无法判定。"], table=[], next="重跑实验1。")

    # Pooling signature prevalence
    sig_A = ((summ.Delta_prob > 0) & (summ.Delta_logit <= 0)).sum()
    n = len(summ)
    sig_A_frac = sig_A / n if n else 0.0
    evidence.append(f"**池化伪影(A 信号)**:在 {sig_A}/{n} 个 method×domain 上 `Delta_prob>0 且 Delta_logit≤0`(占比 {sig_A_frac:.0%})。")

    # Calibration collapse
    calib_collapse = None
    if "Delta_temp" in summ.columns:
        pos_prob = (summ.Delta_prob > 0)
        if pos_prob.sum() > 0:
            calib_collapse = summ.loc[pos_prob, "Delta_temp"].mean()
            evidence.append(f"温度校准:`Delta_prob>0` 的 combo 经温度后 `Delta_temp` 均值 = {calib_collapse:+.4f}({'≈0 或转负 ⇒ 强支持 A' if calib_collapse <= 0.005 else '仍为正 ⇒ 校准未抹平 ⇒ 弱化 A'})。")

    # Saturation excess
    sat_excess = None
    if not sat_cmp.empty and "sat_excess_raw" in sat_cmp.columns:
        sat_excess = sat_cmp["sat_excess_raw"].mean()
        evidence.append(f"饱和率差(fake @0.10):raw − direction 均值 = {sat_excess:+.4f}({'raw 更饱和 ⇒ 支持 A' if sat_excess > 0 else 'direction 更饱和 ⇒ 不支持 A'})。")

    # Temporal consistency gain
    var_red = None
    if not tcon_cmp.empty and "var_reduction" in tcon_cmp.columns:
        var_red = tcon_cmp["var_reduction"].mean()
        evidence.append(f"视频内方差下降 std(raw)−std(dir) 均值 = {var_red:+.4f}({'direction 降低方差 ⇒ 支持 C' if var_red > 0 else '未降低 ⇒ 不支持 C'})。")

    # mean_logit + calibrated direction still improves?
    logit_pos = (summ.Delta_logit > 0).sum()
    evidence.append(f"**真实方向作用(B 信号)**:{logit_pos}/{n} 个 combo 在 `mean_logit`(无饱和)下 direction 仍 `>0`。")
    calib_pos = (summ["Delta_temp"] > 0).sum() if "Delta_temp" in summ.columns else 0
    evidence.append(f"温度后仍有 {calib_pos}/{n} 个 combo direction `>0`。")

    # Method-dependency (D): FSGAN vs SadTalker opposite I_pool signs
    method_dep = None
    if "I_pool" in summ.columns:
        ip = summ.reset_index().groupby("method")["I_pool"].mean()
        if "fsgan" in ip.index and "sadtalker" in ip.index:
            method_dep = ip["fsgan"] - ip["sadtalker"]  # large |diff| => opposite signs
            evidence.append(f"**方法依赖(D 信号)**:I_pool FSGAN − SadTalker = {method_dep:+.4f}({'符号相反、幅度大 ⇒ 强 D' if abs(method_dep) > 0.02 else '差异小 ⇒ 弱 D'})。"
                            f"FSGAN I_pool={ip['fsgan']:+.4f},SadTalker I_pool={ip['sadtalker']:+.4f}。")

    # Calibration collapse: among combos where mean_prob direction won, does T kill it?
    collapse_kill = False
    if calib_collapse is not None:
        collapse_kill = bool(abs(calib_collapse) <= 0.005)

    # Decision (data-driven, not preset)
    a_core = collapse_kill and (sat_excess is not None and sat_excess > 0)
    b_core = bool(logit_pos >= 0.5 * n) and bool(calib_pos >= 0.5 * n)
    c_core = (var_red is not None and var_red > 0)
    d_core = (method_dep is not None and abs(method_dep) > 0.02)

    table.append(f"| A 饱和+概率聚合 | Delta_prob>0但Delta_logit≤0;校准后≈0;raw更饱和 | 池化伪影占比{sig_A_frac:.0%};温度后Δ={calib_collapse};饱和差={sat_excess} |")
    table.append(f"| B 真实径向—角向 | mean_logit 与校准后 direction 仍稳定>0 | logit 下 {logit_pos}/{n} 为正;温度后 {calib_pos}/{n} 为正 |")
    table.append(f"| C 时间一致性 | direction 显著降方差/离群 | 方差下降均值={var_red} |")
    table.append(f"| D 混合方法依赖 | FSGAN/SadTalker 相反最优γ或相反H-B | I_pool FSGAN−SadTalker={method_dep} |")

    # Primary mechanism: calibration-collapse + raw-saturation => A is the mechanism
    # that EXPLAINS the direction>raw effect. Method-dependent SIGN => D character.
    if a_core and d_core and not b_core:
        label = ("A 为主因 + D(方法依赖):direction>raw 的视频级 mean_prob 现象本质是 sigmoid 饱和 + "
                 "概率聚合伪影(校准后归零),但其符号/幅度强依赖伪造方法(FSGAN 与 SadTalker 相反)")
        nxt = ("(1) 视频级评估必须温度校准(FF++ val 拟合 T)后再聚合,或直接用 mean_logit/median 等无饱和聚合;"
               "(2) 因符号方法依赖,不得将 mean_prob 优势解读为方向特征更好;"
               "(3) 暂停径向—角向复杂训练;若需自适应,用样本级径向可靠性门控,不得依赖测试时 method label。")
    elif a_core and not b_core:
        label, nxt = "A:sigmoid 饱和 + 概率聚合伪影(主因)", "采用温度校准 + 更合理的视频聚合(mean_logit/median),暂停径向—角向复杂训练。"
    elif b_core:
        label, nxt = "B:真实径向—角向特征作用(主因)", "继续径向—角向自适应训练;视频级用校准后的 mean_prob 上报。"
    elif c_core and not a_core:
        label, nxt = "C:Direction 提升视频内时间一致性", "采用 temporal-consistency-aware pooling / 稳健视频聚合(median/trimmed)。"
    else:
        label, nxt = "D:混合且方法依赖", "采用样本级径向可靠性门控(不依赖测试时 method label);按方法分别报告。"
    return dict(label=label, evidence=evidence, table=table, next=nxt)


_args = None
def main():
    global _args
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    _args = ap.parse_args()
    if not _args.report_only:
        temporal_run()
    out, verdict = build_report()
    print("\n=== exp7 + final report done ===")
    print("VERDICT:", verdict["label"])
    for e in verdict["evidence"]:
        print("  -", e)
    print("\nNEXT:", verdict["next"])
    print("report ->", REP / "pooling_saturation_final_report.md")


if __name__ == "__main__":
    main()
