# scripts/08_pooling_rank_flip.py
"""Exp6: rank-flip H/B identity dissected BY AGGREGATOR.

For each (backbone, seed, method, domain) x 6 aggregators, run eval_aggregator on
the SAME fake-real video pairs (no sampling) and record CC/WW/H/B/identity_error.
Aggregators: mean_prob, mean_logit, median_logit, trimmed_logit,
temperature_mean_prob (divide s by fitted T then mean_prob),
standardized_mean_prob (robust-standardize s via val-fit median/IQR then mean_prob).

Params (T, robust) come ONLY from FF++ val (Task 9). DFD excluded.
See spec sec 10 experiment 6.
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
from study_pooling.video_eval import eval_aggregator

CFG = load_config(PHASE / "configs" / "pooling_saturation_config.yaml")
TAB = PHASE / CFG.paths["tables"]
FIG = PHASE / CFG.paths["figures"]
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

AGGS = ["mean_prob", "mean_logit", "median_logit", "trimmed_logit",
        "temperature_mean_prob", "standardized_mean_prob"]


def _wide(df):
    raw = df[df.infer_mode == "raw"][["backbone", "seed", "domain", "method_id",
        "label", "video_id", "image_path", "signed_logit"]
    ].rename(columns={"signed_logit": "s_raw"})
    d = df[df.infer_mode == "direction"][["image_path", "signed_logit"]
    ].rename(columns={"signed_logit": "s_dir"})
    return raw.merge(d, on="image_path")


def _load_val_params():
    """Load fitted T (per backbone,seed) and robust params (per backbone,seed) from val cache."""
    T_map = {}
    rob_map = {}
    tpath = TAB / "Q_pooling_temperature.csv"
    if tpath.exists():
        tdf = pd.read_csv(tpath)
        for r in tdf.itertuples():
            T_map[(r.backbone, r.seed)] = (r.T_raw, r.T_dir)
    # Robust median/IQR params: recompute from val cache parquets (filename parse).
    import re
    cache = PHASE / CFG.paths["val_cache"]
    for p in sorted(cache.glob("*__val.parquet")):
        m = re.match(r"(?P<bb>clip|dinov2|efficientnet)_(?P<mode>raw)_s(?P<seed>\d+)__val", p.name)
        if not m:
            continue
        key = (m.group("bb"), int(m.group("seed")))
        v = pd.read_parquet(p, columns=["s_raw", "s_dir"])
        rob_map[key] = {
            "raw": (float(v.s_raw.median()),
                    float(np.percentile(v.s_raw, 75) - np.percentile(v.s_raw, 25))),
            "direction": (float(v.s_dir.median()),
                          float(np.percentile(v.s_dir, 75) - np.percentile(v.s_dir, 25))),
        }
    return T_map, rob_map


def _transform(w, agg, T_map, rob_map, key):
    """Return a transformed copy of w for temperature/standardized aggregators."""
    w = w.copy()
    if agg == "temperature_mean_prob":
        Tr, Td = T_map.get(key, (1.0, 1.0))
        w["s_raw"] = w.s_raw / Tr
        w["s_dir"] = w.s_dir / Td
        agg = "mean_prob"
    elif agg == "standardized_mean_prob":
        rob = rob_map.get(key)
        if rob is None:
            return None, None
        mr, ir = rob["raw"]
        md, id_ = rob["direction"]
        w["s_raw"] = (w.s_raw - mr) / max(ir, 1e-6)
        w["s_dir"] = (w.s_dir - md) / max(id_, 1e-6)
        agg = "mean_prob"
    return w, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    fpath = str(ROOT / CFG.parquet_frame)
    T_map, rob_map = _load_val_params()

    if args.smoke:
        targets = [(CFG.smoke["backbone"], CFG.smoke["seed"], CFG.smoke["methods"])]
    else:
        targets = [(bb, sd, CFG.methods) for bb in CFG.backbones for sd in CFG.seeds]

    rows = []
    for bb, sd, methods in targets:
        key = (bb, sd)
        for m in methods:
            for dom in CFG.domains:
                df = dedup_reals(load_frame_subset(fpath, train_mode="raw", infer_mode=None,
                    backbones=[bb], seeds=[sd], domains=[dom], methods=[m, "real"]))
                w = _wide(df)
                if w.empty or (w.label == 1).sum() == 0 or (w.label == 0).sum() == 0:
                    continue
                for agg in AGGS:
                    wt, agg_eff = _transform(w, agg, T_map, rob_map, key)
                    if wt is None:
                        continue
                    r = eval_aggregator(wt, agg_eff)
                    rows.append({"backbone": bb, "seed": sd, "method": m, "domain": dom,
                        "aggregation": agg, "CC": r["CC"], "WW": r["WW"],
                        "H": r["H"], "B": r["B"], "H_minus_B": r["H_minus_B"],
                        "delta_auc": r["delta_auc"], "identity_error": r["identity_error"]})

    out = pd.DataFrame(rows)
    out.to_csv(TAB / "Q_pooling_rank_flip_by_aggregator.csv", index=False)

    # identity self-check
    if not out.empty:
        max_id = out.identity_error.abs().max()
        print(f"\n=== exp6 rank-flip done: {len(out)} rows ===")
        print(f"[self-check] max |identity_error| = {max_id:.3e}  (must be ~0)")
        # KEY report: FSGAN & SadTalker H-B by aggregator (mean over seeds/backbone)
        print("\n--- KEY: mean H-B by aggregator (FSGAN/SadTalker, FF++/CDF) ---")
        sub = out[out.method.isin(["fsgan", "sadtalker"])]
        pvt = sub.groupby(["method", "domain", "aggregation"])["H_minus_B"].mean().unstack("aggregation")
        print(pvt.to_string(float_format=lambda v: f"{v:+.2f}"))
        _plot(out)


def _plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if out.empty:
        return
    pvt = out.groupby(["method", "aggregation"])["H_minus_B"].mean().unstack("aggregation")
    order = [a for a in AGGS if a in pvt.columns]
    pvt = pvt[order]
    fig, ax = plt.subplots(figsize=(10, 7))
    pvt.plot(kind="bar", ax=ax)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("mean H - B (direction-fix wins minus breaks)")
    ax.set_title("Rank-flip H-B by aggregator (mean over backbone/seed/domain)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "Q_pooling_HB_by_aggregator.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
