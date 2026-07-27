# scripts/06_pooling_saturation_analysis.py
"""Pooling experiments 1 (aggregators), 2 (saturation), 3 (Jensen), gamma scan.

Parquet-only. Dry-run first (smoke), then expand. See
docs/superpowers/specs/2026-07-27-pooling-saturation-design.md (§3 experiments 1/2/3, §12 gamma).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepfake_detection.study_pooling.config import load_config
from deepfake_detection.study.io import load_frame_subset, dedup_reals
from deepfake_detection.study_pooling.aggregators import aggregate, jensen_gap, NAMES
from deepfake_detection.study_pooling.video_eval import eval_aggregator
from deepfake_detection.study_pooling.saturation import frame_stats, sat_ratios, sigmoid_grad_stats
from deepfake_detection.study_pooling.gamma import recover_wdotu, score_gamma
from deepfake_detection.study_pooling.wb_io import signed_wb
from deepfake_detection.diagnosis.checkpoint_index import discover_checkpoints

CFG = load_config(ROOT / "configs" / "pooling_saturation_config.yaml")
TAB = ROOT / CFG.paths["tables"]
FIG = ROOT / CFG.paths["figures"]
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
AGGS = ["mean_prob", "mean_logit", "sigmoid_mean_logit", "median_logit",
        "clip_prob", "trimmed_logit", "topk_logit"]


def _wide(df):
    """Merge raw + direction per image_path, keeping feature_norm + video_id + label."""
    raw = df[df.infer_mode == "raw"][["checkpoint_id", "backbone", "seed", "domain", "method_id",
        "label", "video_id", "image_path", "feature_norm", "signed_logit"]
    ].rename(columns={"signed_logit": "s_raw"})
    d = df[df.infer_mode == "direction"][["image_path", "signed_logit"]
    ].rename(columns={"signed_logit": "s_dir"})
    return raw.merge(d, on="image_path")


def run_combo(bb, sd, method, dom, fpath, ckinfo):
    df = dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode=None,
        backbones=[bb], seeds=[sd], domains=[dom], methods=[method, "real"]))
    w = _wide(df)
    if w.empty or (w.label == 1).sum() == 0 or (w.label == 0).sum() == 0:
        return [], [], [], []
    agg_rows = []; sat_rows = []; j_rows = []
    # exp1 aggregation
    for a in AGGS:
        if a == "clip_prob":
            for c in CFG.clip_values:
                m = eval_aggregator(w, a, c=float(c))
                agg_rows.append({"backbone": bb, "seed": sd, "method": method, "domain": dom,
                    "aggregation": f"clip_prob_{int(c)}", "auc_raw": m["auc_raw"],
                    "auc_direction": m["auc_dir"], "delta_auc": m["delta_auc"],
                    "num_real_videos": m["n_real"], "num_fake_videos": m["n_fake"]})
        else:
            m = eval_aggregator(w, a)
            agg_rows.append({"backbone": bb, "seed": sd, "method": method, "domain": dom,
                "aggregation": a, "auc_raw": m["auc_raw"], "auc_direction": m["auc_dir"],
                "delta_auc": m["delta_auc"], "num_real_videos": m["n_real"],
                "num_fake_videos": m["n_fake"]})
    # exp2 saturation (frame-level, raw & direction, real & fake)
    for mode, col in [("raw", "s_raw"), ("direction", "s_dir")]:
        for lbl, lbln in [(1, "fake"), (0, "real")]:
            sub = w[w.label == lbl]
            s = sub[col].to_numpy(float)
            p = 1.0 / (1.0 + np.exp(-s))
            st = frame_stats(s, p)
            sr = sat_ratios(p, CFG.sat_thresholds)
            sg = sigmoid_grad_stats(p)
            sat_rows.append({"backbone": bb, "seed": sd, "method": method, "domain": dom,
                "infer_mode": mode, "label": lbln, **st, **sr, **sg})
    # exp3 Jensen (per video, raw & direction)
    for mode, col in [("raw", "s_raw"), ("direction", "s_dir")]:
        Js = [jensen_gap(g[col].to_numpy()) for _, g in w.groupby("video_id", sort=False)]
        j_rows.append({"backbone": bb, "seed": sd, "method": method, "domain": dom,
            "infer_mode": mode, "mean_J": float(np.mean(Js)),
            "mean_abs_J": float(np.mean(np.abs(Js))), "n_videos": len(Js)})
    # gamma scan (uses real checkpoint b)
    gamma_rows = []
    try:
        if ckinfo is not None:
            wu, _ = recover_wdotu(w.s_raw.to_numpy(), w.s_dir.to_numpy(),
                                  w.feature_norm.to_numpy())
            _wv, bv = signed_wb(ckinfo)  # discard recovered weight; use real b only
            for g in CFG.gamma_values:
                s_g = score_gamma(wu, bv, w.feature_norm.to_numpy(), g)  # uses REAL b
                wg = w.assign(s_raw=s_g, s_dir=s_g)  # gamma replaces both consistently
                for a in ["mean_prob", "mean_logit"]:
                    m = eval_aggregator(wg, a)
                    gamma_rows.append({"backbone": bb, "seed": sd, "method": method,
                        "domain": dom, "gamma": float(g), "aggregation": a,
                        "video_auc": m["auc_raw"]})
    except Exception as e:  # noqa: BLE001
        print(f"[gamma] skip {bb} {sd} {method} {dom}: {e}")
    return agg_rows, sat_rows, j_rows, gamma_rows


def _self_check(A):
    """AUC(mean_logit) == AUC(sigmoid_mean_logit) at machine precision (sigmoid monotone)."""
    a = pd.DataFrame(A)
    if a.empty:
        return np.nan
    ml = a[a.aggregation == "mean_logit"].set_index(
        ["backbone", "seed", "method", "domain"])
    sml = a[a.aggregation == "sigmoid_mean_logit"].set_index(
        ["backbone", "seed", "method", "domain"])
    common = ml.index.intersection(sml.index)
    dr = (ml.loc[common, "auc_raw"] - sml.loc[common, "auc_raw"]).abs().max()
    dd = (ml.loc[common, "auc_direction"] - sml.loc[common, "auc_direction"]).abs().max()
    return float(max(dr, dd)) if len(common) else np.nan


def _delta_summary(A):
    """Compute Delta_prob, Delta_logit, I_pool = Delta_prob - Delta_logit per combo."""
    a = pd.DataFrame(A)
    if a.empty:
        return pd.DataFrame()
    dp = a[a.aggregation == "mean_prob"][["backbone", "seed", "method", "domain",
                                          "delta_auc"]].rename(columns={"delta_auc": "Delta_prob"})
    dl = a[a.aggregation == "mean_logit"][["backbone", "seed", "method", "domain",
                                           "delta_auc"]].rename(columns={"delta_auc": "Delta_logit"})
    out = dp.merge(dl, on=["backbone", "seed", "method", "domain"])
    out["I_pool"] = out["Delta_prob"] - out["Delta_logit"]
    return out


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=120)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _plots(A, S, J, G):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = pd.DataFrame(A); s = pd.DataFrame(S); j = pd.DataFrame(J); g = pd.DataFrame(G)

    # ---- Fig 1: FSGAN & SadTalker aggregators bar (mean over seeds) ----
    sub = a[(a.method.isin(["fsgan", "sadtalker"])) & (~a.aggregation.str.startswith("clip_prob"))]
    if not sub.empty:
        pvt = sub.groupby(["method", "aggregation"])["delta_auc"].mean().unstack("method")
        order = [x for x in ["mean_prob", "mean_logit", "sigmoid_mean_logit", "median_logit",
                             "trimmed_logit", "topk_logit"] if x in pvt.index]
        pvt = pvt.loc[order]
        fig, ax = plt.subplots(figsize=(10, 5))
        pvt.plot(kind="bar", ax=ax)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("delta_auc (direction - raw)")
        ax.set_title("Aggregator effect: direction - raw AUC (FSGAN vs SadTalker)")
        ax.tick_params(axis="x", rotation=30)
        _save(fig, "Q_pooling_FSGAN_SadTalker_aggregators.png")

    summ = _delta_summary(A)

    def _heatmap(col, fname, title):
        if summ.empty:
            return
        pvt = summ.groupby(["backbone", "method"])[col].mean().unstack("backbone")
        if pvt.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 9))
        im = ax.imshow(pvt.to_numpy(), aspect="auto", cmap="RdBu_r",
                       vmin=-max(abs(pvt.to_numpy().min()), abs(pvt.to_numpy().max())),
                       vmax=max(abs(pvt.to_numpy().min()), abs(pvt.to_numpy().max())))
        ax.set_yticks(range(len(pvt.index))); ax.set_yticklabels(pvt.index)
        ax.set_xticks(range(len(pvt.columns))); ax.set_xticklabels(pvt.columns)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.03)
        _save(fig, fname)

    # ---- Fig 2/3: delta heatmaps ----
    _heatmap("Delta_prob", "Q_pooling_delta_heatmap_mean_prob.png",
             "Delta_prob (mean_prob: direction - raw AUC), mean over seeds")
    _heatmap("Delta_logit", "Q_pooling_delta_heatmap_mean_logit.png",
             "Delta_logit (mean_logit: direction - raw AUC), mean over seeds")

    # ---- Fig 4: interaction term I_pool ----
    if not summ.empty:
        fig, ax = plt.subplots(figsize=(8, 9))
        pvt = summ.groupby(["backbone", "method"])["I_pool"].mean().unstack("backbone")
        if not pvt.empty:
            im = ax.imshow(pvt.to_numpy(), aspect="auto", cmap="RdBu_r",
                           vmin=-max(abs(pvt.to_numpy().min()), abs(pvt.to_numpy().max())),
                           vmax=max(abs(pvt.to_numpy().min()), abs(pvt.to_numpy().max())))
            ax.set_yticks(range(len(pvt.index))); ax.set_yticklabels(pvt.index)
            ax.set_xticks(range(len(pvt.columns))); ax.set_xticklabels(pvt.columns)
            ax.set_title("I_pool = Delta_prob - Delta_logit (pooling-induced gain), mean over seeds")
            fig.colorbar(im, ax=ax, fraction=0.03)
        _save(fig, "Q_pooling_interaction_term.png")

    if not s.empty:
        sg = s.groupby(["backbone", "infer_mode", "label"])["std"].mean().unstack("infer_mode")
        # ---- Fig 5: logit scale |s| raw vs direction ----
        labs = s.copy(); labs["abs_mean"] = labs.groupby(
            ["backbone", "infer_mode", "label"])["mean"].transform(lambda x: x.abs())
        pvt = labs.groupby(["backbone", "infer_mode", "label"])["mean"].mean().abs().unstack("infer_mode")
        if not pvt.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            pvt.plot(kind="bar", ax=ax)
            ax.set_ylabel("|mean logit|")
            ax.set_title("Logit scale |s|: raw vs direction (by backbone, fake/real)")
            ax.axhline(0, color="k", lw=0.5)
            _save(fig, "Q_pooling_logit_scale_raw_vs_direction.png")
        # ---- Fig 6: saturation ratio raw vs direction (sat_010, fake) ----
        for tcol, ttl, fnm in [("sat_010", "Saturation ratio @0.10", "Q_pooling_saturation_raw_vs_direction.png")]:
            if tcol in s.columns:
                pvt = s.groupby(["backbone", "infer_mode"])[tcol].mean().unstack("infer_mode")
                if not pvt.empty:
                    fig, ax = plt.subplots(figsize=(7, 5))
                    pvt.plot(kind="bar", ax=ax)
                    ax.set_ylabel(ttl)
                    ax.set_title(f"{ttl}: raw vs direction (mean over method/label)")
                    _save(fig, fnm)
        # ---- Fig 7: sigmoid gradient p(1-p) raw vs direction ----
        pvt = s.groupby(["backbone", "infer_mode"])["mean"].mean().unstack("infer_mode") \
            if "mean" in s.columns else pd.DataFrame()
        # use sigmoid_grad mean column (also named 'mean'); recompute from sigmoid_grad_stats output
        if "median" in s.columns:
            pvt = s.groupby(["backbone", "infer_mode"])["mean"].mean().unstack("infer_mode")
            if not pvt.empty:
                fig, ax = plt.subplots(figsize=(7, 5))
                pvt.plot(kind="bar", ax=ax)
                ax.set_ylabel("mean sigmoid gradient p(1-p)")
                ax.set_title("Sigmoid gradient (mean): raw vs direction — small => more saturation")
                _save(fig, "Q_pooling_sigmoid_gradient.png")

    # ---- Fig 8: Jensen gap distribution ----
    if not j.empty:
        fig, ax = plt.subplots(figsize=(7, 5))
        for mode, sub in j.groupby("infer_mode"):
            ax.hist(sub["mean_abs_J"].to_numpy(), bins=30, alpha=0.5, label=mode)
        ax.set_xlabel("mean |J| per backbone/seed/method/domain")
        ax.set_ylabel("count")
        ax.set_title("Jensen gap distribution: raw vs direction")
        ax.legend()
        _save(fig, "Q_pooling_jensen_gap_distribution.png")

        # ---- Fig 9: J vs score change (mean_J vs delta of |s|) ----
        js = j.groupby(["backbone", "seed", "method", "domain", "infer_mode"])["mean_J"].mean().unstack("infer_mode")
        if "raw" in js.columns and "direction" in js.columns:
            js["J_diff"] = js["direction"] - js["raw"]
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(js["raw"].abs(), js["J_diff"], s=12, alpha=0.5)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("|mean J| (raw)")
            ax.set_ylabel("mean J (direction - raw)")
            ax.set_title("Jensen gap vs direction-induced change")
            _save(fig, "Q_pooling_jensen_gap_vs_score_change.png")

        # ---- Fig 10: Jensen FSGAN vs SadTalker ----
        sub = j[j.method.isin(["fsgan", "sadtalker"])]
        if not sub.empty:
            pvt = sub.groupby(["method", "infer_mode"])["mean_abs_J"].mean().unstack("infer_mode")
            if not pvt.empty:
                fig, ax = plt.subplots(figsize=(6, 5))
                pvt.plot(kind="bar", ax=ax)
                ax.set_ylabel("mean |J|")
                ax.set_title("Jensen gap: FSGAN vs SadTalker (raw vs direction)")
                ax.axhline(0, color="k", lw=0.5)
                _save(fig, "Q_pooling_jensen_gap_FSGAN_SadTalker.png")

    # ---- Fig 11/12: gamma scan by aggregator for FSGAN / SadTalker ----
    if not g.empty:
        for method, fnm in [("fsgan", "Q_pooling_gamma_FSGAN_by_aggregator.png"),
                            ("sadtalker", "Q_pooling_gamma_SadTalker_by_aggregator.png")]:
            sub = g[g.method == method]
            if sub.empty:
                continue
            pvt = sub.groupby(["gamma", "aggregation"])["video_auc"].mean().unstack("aggregation")
            if pvt.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            pvt.plot(ax=ax, marker="o")
            ax.axvline(0.0, color="gray", ls=":", lw=0.8, label="dir (gamma=0)")
            ax.axvline(1.0, color="gray", ls="--", lw=0.8, label="raw (gamma=1)")
            ax.set_xlabel("gamma (0=direction, 1=raw)")
            ax.set_ylabel("video-level AUC")
            ax.set_title(f"Gamma scan — {method} (real b, mean over backbone/seed/domain)")
            ax.legend()
            _save(fig, fnm)

    plt.close("all")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    fpath = CFG.parquet_frame
    if a.smoke:
        targets = [(CFG.smoke["backbone"], CFG.smoke["seed"], CFG.smoke["methods"])]
    else:
        targets = [(bb, sd, CFG.methods) for bb in CFG.backbones for sd in CFG.seeds]
    A = []; S = []; J = []; G = []
    for bb, sd, methods in targets:
        cks = discover_checkpoints(CFG.ckpt_root, modes=["raw"], backbones=[bb], seeds=[sd])
        ckinfo = cks[0] if cks else None
        for m in methods:
            for dom in CFG.domains:
                ag, sa, jr, gr = run_combo(bb, sd, m, dom, fpath, ckinfo)
                A += ag; S += sa; J += jr; G += gr
    pd.DataFrame(A).to_csv(TAB / "Q_pooling_aggregation_auc.csv", index=False)
    pd.DataFrame(S).to_csv(TAB / "Q_pooling_saturation_stats.csv", index=False)
    pd.DataFrame(J).to_csv(TAB / "Q_pooling_jensen_gap.csv", index=False)
    pd.DataFrame(G).to_csv(TAB / "Q_pooling_gamma_scan.csv", index=False)
    _plots(A, S, J, G)

    # Self-check report
    max_diff = _self_check(A)
    summ = _delta_summary(A)
    print("\n=== exp1/2/3 + gamma done ===")
    print(f"rows: agg={len(A)} sat={len(S)} jensen={len(J)} gamma={len(G)}")
    print(f"[self-check] max |AUC(mean_logit) - AUC(sigmoid_mean_logit)| = {max_diff:.3e}  (must < 1e-9)")
    if not summ.empty:
        print("\n--- KEY dry-run table: delta_auc mean_prob vs mean_logit ---")
        for (meth, dom), row in summ[summ.method.isin(["fsgan", "sadtalker"])].groupby(
                ["method", "domain"], sort=False):
            print(f"  {meth:10s} {dom:5s}  Delta_prob={row['Delta_prob'].iloc[0]:+.4f}  "
                  f"Delta_logit={row['Delta_logit'].iloc[0]:+.4f}  I_pool={row['I_pool'].iloc[0]:+.4f}")


if __name__ == "__main__":
    main()
