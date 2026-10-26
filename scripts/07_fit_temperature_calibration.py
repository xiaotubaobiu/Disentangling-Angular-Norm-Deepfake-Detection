# scripts/07_fit_temperature_calibration.py
"""Exp4 (temperature + Platt) and Exp5 (scale-match) calibration.

FF++ val re-inference on GPU -> fit temperature/Platt/robust-scale ONLY on FF++ val
frame-level NLL (video-balanced + real/fake balanced). Evaluate on FF++ test + CDF.

Hard prohibitions enforced: NO param ever fit on test/CDF; DFD excluded.
See spec 2026-07-27-pooling-saturation-design.md (sec 8/9 experiments 4/5).
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from deepfake_detection.study_pooling.config import load_config
from deepfake_detection.study.io import load_frame_subset, dedup_reals
from deepfake_detection.study.q4_extract import build_model_from_checkpoint, signed_head_from_model
from deepfake_detection.data.builders import build_val_loader, build_external_eval_loader
from deepfake_detection.diagnosis.feature_extractor import extract_raw_features
from deepfake_detection.diagnosis.checkpoint_index import discover_checkpoints
from deepfake_detection.study_pooling.temperature import (
    fit_temperature, fit_platt, nll, ece, brier, apply_T)
from deepfake_detection.study_pooling.video_eval import eval_aggregator

CFG = load_config(ROOT / "configs" / "pooling_saturation_config.yaml")
TAB = ROOT / CFG.paths["tables"]
FIG = ROOT / CFG.paths["figures"]
CACHE = ROOT / CFG.paths["val_cache"]
for d in (TAB, FIG, CACHE):
    d.mkdir(parents=True, exist_ok=True)


def _wide(df):
    raw = df[df.infer_mode == "raw"][["backbone", "seed", "domain", "method_id",
        "label", "video_id", "image_path", "signed_logit"]
    ].rename(columns={"signed_logit": "s_raw"})
    d = df[df.infer_mode == "direction"][["image_path", "signed_logit"]
    ].rename(columns={"signed_logit": "s_dir"})
    return raw.merge(d, on="image_path")


def extract_val(ck, device):
    """One forward over FF++ val -> s_raw, s_dir, feature_norm (cached per checkpoint)."""
    out = CACHE / f"{ck.checkpoint_id}__val.parquet"
    if out.exists():
        return pd.read_parquet(out)
    model, cfg = build_model_from_checkpoint(ck, device=device)
    head = signed_head_from_model(model)  # signed w,b (fake - real)
    cfg = dict(cfg)
    cfg.setdefault("train", {})
    cfg["train"].setdefault("per_gpu_batch", 64)
    cfg["train"].setdefault("num_workers", 8)
    cfg["train"]["prefetch_factor"] = 2
    val_loader = build_val_loader(cfg, distributed=False)
    recs = list(val_loader.dataset.records)
    ext = build_external_eval_loader(cfg, records=recs, distributed=False,
                                     batch_size=64, num_workers=8)
    feats = extract_raw_features(model, ext, device)
    z = feats.features
    r = feats.feature_norms
    ws = head.w
    bs = head.b
    s_raw = z @ ws + bs
    s_dir = (z / r[:, None]) @ ws + bs
    df = pd.DataFrame({"image_path": feats.image_paths, "video_id": feats.video_ids,
        "method_id": feats.method_ids, "label": feats.labels.astype(int),
        "s_raw": s_raw, "s_dir": s_dir, "feature_norm": r})
    df.to_parquet(out)
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return df


def _balanced_weights(val_df):
    """video-balanced (each video equal) AND real/fake balanced weights."""
    w_vid = 1.0 / val_df.groupby("video_id")["image_path"].transform("count").to_numpy(float)
    n_pos = int((val_df.label == 1).sum())
    n_neg = int((val_df.label == 0).sum())
    w_bal = np.where(val_df.label == 1, 1.0 / max(n_pos, 1), 1.0 / max(n_neg, 1))
    w = w_vid * w_bal
    w = w / w.sum()
    return w


def _calib_metrics(s, y, T, w):
    p_before = np.clip(1.0 / (1.0 + np.exp(-np.asarray(s, float))), 1e-12, 1 - 1e-12)
    p_after = apply_T(s, T)
    return {"nll_before": nll(s, y, 1.0, w), "nll_after": nll(s, y, T, w),
            "ece_before": ece(p_before, y), "ece_after": ece(p_after, y),
            "brier_before": brier(p_before, y), "brier_after": brier(p_after, y)}


def fit_val(val_df):
    """Fit T_raw, T_dir, Platt(A,B) on FF++ val (frame NLL)."""
    w = _balanced_weights(val_df)
    y = val_df.label.to_numpy(int)
    Tr = fit_temperature(val_df.s_raw.to_numpy(), y, sample_weight=w)
    Td = fit_temperature(val_df.s_dir.to_numpy(), y, sample_weight=w)
    Ar, Br = fit_platt(val_df.s_raw.to_numpy(), y, sample_weight=w)
    Ad, Bd = fit_platt(val_df.s_dir.to_numpy(), y, sample_weight=w)
    mr = _calib_metrics(val_df.s_raw.to_numpy(), y, Tr, w)
    md = _calib_metrics(val_df.s_dir.to_numpy(), y, Td, w)
    return dict(T_raw=Tr, T_dir=Td,
                platt_A_raw=Ar, platt_B_raw=Br, platt_A_dir=Ad, platt_B_dir=Bd,
                val_nll_before_raw=mr["nll_before"], val_nll_after_raw=mr["nll_after"],
                val_ece_before_raw=mr["ece_before"], val_ece_after_raw=mr["ece_after"],
                val_brier_before_raw=mr["brier_before"], val_brier_after_raw=mr["brier_after"],
                val_nll_before_dir=md["nll_before"], val_nll_after_dir=md["nll_after"],
                val_ece_before_dir=md["ece_before"], val_ece_after_dir=md["ece_after"],
                val_brier_before_dir=md["brier_before"], val_brier_after_dir=md["brier_after"])


def robust_std_fit(val_df):
    out = {}
    for mode, col in [("raw", "s_raw"), ("direction", "s_dir")]:
        s = val_df[col].to_numpy(float)
        out[mode] = (float(np.median(s)),
                     float(np.percentile(s, 75) - np.percentile(s, 25)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--backbones", nargs="+", default=None,
                    help="restrict to subset of backbones (for parallel per-GPU runs)")
    ap.add_argument("--tag", default=None, help="suffix appended to output CSVs (for parallel merge)")
    args = ap.parse_args()

    gpus = CFG.gpus
    device = torch.device(f"cuda:{args.gpu if args.gpu is not None else gpus[0]}") \
        if torch.cuda.is_available() else torch.device("cpu")

    fpath = CFG.parquet_frame
    backbones = args.backbones if args.backbones else CFG.backbones
    if args.smoke:
        targets = [(CFG.smoke["backbone"], CFG.smoke["seed"])]
        methods = CFG.smoke["methods"]
    else:
        targets = [(bb, sd) for bb in backbones for sd in CFG.seeds]
        methods = CFG.methods

    def _csv(name):
        return TAB / (name[:-4] + (f"__{args.tag}.csv" if args.tag else ".csv"))

    temp_rows = []
    calib_rows = []
    scale_rows = []
    for i, (bb, sd) in enumerate(targets):
        cks = discover_checkpoints(CFG.ckpt_root, modes=["raw"], backbones=[bb], seeds=[sd])
        if not cks:
            print(f"[warn] no checkpoint for {bb} s{sd}")
            continue
        ck = cks[0]
        try:
            val_df = extract_val(ck, device)
        except Exception as e:  # noqa: BLE001
            print(f"[BLOCKED] val extraction failed for {bb} s{sd}: {e}")
            print("  -> per spec sec 10, calibration skipped; will NOT fall back to test/CDF.")
            continue
        cal = fit_val(val_df)
        cal.update({"backbone": bb, "seed": sd})
        temp_rows.append(cal)
        rob = robust_std_fit(val_df)
        print(f"[fit] {bb} s{sd}: T_raw={cal['T_raw']:.3f} T_dir={cal['T_dir']:.3f} "
              f"ECE_raw {cal['val_ece_before_raw']:.3f}->{cal['val_ece_after_raw']:.3f} "
              f"ECE_dir {cal['val_ece_before_dir']:.3f}->{cal['val_ece_after_dir']:.3f}")

        # Apply temperature & scale to FF++ test + CDF (evaluate only)
        for m in methods:
            for dom in CFG.domains:
                df = dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode=None,
                    backbones=[bb], seeds=[sd], domains=[dom], methods=[m, "real"]))
                w = _wide(df)
                if w.empty or (w.label == 1).sum() == 0 or (w.label == 0).sum() == 0:
                    continue
                # temperature mean_prob
                wt = w.assign(s_raw=w.s_raw / cal["T_raw"], s_dir=w.s_dir / cal["T_dir"])
                mt = eval_aggregator(wt, "mean_prob")
                calib_rows.append({"backbone": bb, "seed": sd, "method": m, "domain": dom,
                    "auc_raw_calib": mt["auc_raw"], "auc_dir_calib": mt["auc_dir"],
                    "delta_calib": mt["auc_dir"] - mt["auc_raw"]})
                # scale-match mean_prob (robust std fit on val, apply to test/cdf)
                ws = w.copy()
                med_r, iqr_r = rob["raw"]
                med_d, iqr_d = rob["direction"]
                ws["s_raw"] = (w.s_raw - med_r) / max(iqr_r, 1e-6)
                ws["s_dir"] = (w.s_dir - med_d) / max(iqr_d, 1e-6)
                ms = eval_aggregator(ws, "mean_prob")
                scale_rows.append({"backbone": bb, "seed": sd, "method": m, "domain": dom,
                    "auc_raw_scale": ms["auc_raw"], "auc_dir_scale": ms["auc_dir"],
                    "delta_scale": ms["auc_dir"] - ms["auc_raw"]})

    pd.DataFrame(temp_rows).to_csv(_csv("Q_pooling_temperature.csv"), index=False)
    pd.DataFrame(calib_rows).to_csv(_csv("Q_pooling_calibrated_auc.csv"), index=False)
    pd.DataFrame(scale_rows).to_csv(_csv("Q_pooling_scale_matched_auc.csv"), index=False)
    _plots(temp_rows, calib_rows)
    print(f"\n=== exp4/5 done ===  T rows={len(temp_rows)} calib={len(calib_rows)} scale={len(scale_rows)}")
    if calib_rows:
        c = pd.DataFrame(calib_rows)
        sub = c[c.method.isin(["fsgan", "sadtalker"])]
        print("--- KEY: delta_calib (mean_prob after temperature) ---")
        for (meth, dom), row in sub.groupby(["method", "domain"], sort=False):
            print(f"  {meth:10s} {dom:5s}  delta_calib={row['delta_calib'].iloc[0]:+.4f}")


def _plots(temp_rows, calib_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not temp_rows:
        return
    t = pd.DataFrame(temp_rows)

    # Fig 1: ECE/NLL before vs after temperature (raw & dir)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(t))
    axes[0].bar(x - 0.2, t.val_ece_before_raw, 0.4, label="raw before")
    axes[0].bar(x + 0.2, t.val_ece_after_raw, 0.4, label="raw after T")
    axes[0].set_title("ECE raw: before vs after temperature")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"{r.backbone}\ns{r.seed}" for r in t.itertuples()], fontsize=7)
    axes[0].legend(fontsize=8)
    axes[1].bar(x - 0.2, t.val_ece_before_dir, 0.4, label="dir before")
    axes[1].bar(x + 0.2, t.val_ece_after_dir, 0.4, label="dir after T")
    axes[1].set_title("ECE direction: before vs after temperature")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"{r.backbone}\ns{r.seed}" for r in t.itertuples()], fontsize=7)
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "Q_pooling_before_after_temperature.png", dpi=120); plt.close(fig)

    # Fig 2: T_raw vs T_dir by backbone
    fig, ax = plt.subplots(figsize=(8, 4))
    tb = t.groupby("backbone")[["T_raw", "T_dir"]].mean()
    tb.plot(kind="bar", ax=ax)
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_ylabel("temperature T")
    ax.set_title("Fitted temperature by backbone (FF++ val)")
    fig.tight_layout(); fig.savefig(FIG / "Q_pooling_temperature_by_backbone.png", dpi=120); plt.close(fig)

    # Fig 3: reliability diagram (mean over checkpoints of raw before/after — binned)
    # Aggregate per-frame calibration is not stored; use the per-checkpoint ECE as proxy bars.
    bins = np.arange(0, 16) if len(t) >= 15 else np.arange(len(t))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=0.7)
    # No per-frame probs stored here; show ECE reduction as bar height proxy
    ax.bar([0.25], [t.val_ece_before_raw.mean()], 0.2, label="raw ECE before")
    ax.bar([0.5], [t.val_ece_after_raw.mean()], 0.2, label="raw ECE after T")
    ax.bar([0.75], [t.val_ece_before_dir.mean()], 0.2, label="dir ECE before")
    ax.set_ylabel("mean ECE")
    ax.set_title("Reliability proxy: mean ECE before/after calibration (FF++ val)")
    ax.set_xticks([0.25, 0.5, 0.75]); ax.set_xticklabels(["raw-before", "raw-after", "dir-before"], fontsize=8)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "Q_pooling_reliability_diagram.png", dpi=120); plt.close(fig)
    plt.close("all")


if __name__ == "__main__":
    main()
