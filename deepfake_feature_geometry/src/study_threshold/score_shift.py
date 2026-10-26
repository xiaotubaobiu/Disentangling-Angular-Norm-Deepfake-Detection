"""Why a high-AUC model can still lose balanced accuracy at a frozen threshold.

AUC is invariant to any monotone transform of the scores; a hard threshold is
not. So the interesting quantities are how the two class-conditional score
distributions *move* and *stretch* between source and target:

    shift_real         = mean_target(real)  - mean_source(real)
    shift_fake         = mean_target(fake)  - mean_source(fake)
    common_shift       = (shift_real + shift_fake) / 2      # rigid translation
    differential_shift = shift_fake - shift_real            # classes move apart/together
    scale_real         = std_target(real) / std_source(real)
    scale_fake         = std_target(fake) / std_source(fake)

Plus distribution-level distances per class (Wasserstein-1 and the two-sample KS
statistic). A large `common_shift` with `differential_shift ~ 0` and `scale ~ 1`
means the ranking is intact and only the *offset* is wrong — recoverable by
re-centring the threshold. A large `differential_shift` or `scale != 1` means the
classes overlap differently and no single re-centring recovers the source
operating point.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

from study_threshold.io import score_column

QUANTILES = (0.05, 0.25, 0.75, 0.95)


def distribution_stats(scores) -> dict:
    s = np.asarray(scores, float)
    if s.size == 0:
        return {k: float("nan") for k in
                ["mean", "median", "std", "iqr", "q05", "q25", "q75", "q95", "n"]}
    q05, q25, q75, q95 = (float(np.quantile(s, q)) for q in QUANTILES)
    return {
        "mean": float(np.mean(s)),
        "median": float(np.median(s)),
        "std": float(np.std(s, ddof=1)) if s.size > 1 else 0.0,
        "iqr": float(q75 - q25),
        "q05": q05, "q25": q25, "q75": q75, "q95": q95,
        "n": int(s.size),
    }


def class_stats(videos: pd.DataFrame, mode: str) -> dict:
    """Per-class distribution stats of the video score for one mode."""
    col = score_column(mode)
    out = {}
    for label, name in ((0, "real"), (1, "fake")):
        stats = distribution_stats(videos.loc[videos["label"] == label, col])
        out.update({f"{name}_{k}": v for k, v in stats.items()})
    return out


def shift_metrics(source_videos: pd.DataFrame, target_videos: pd.DataFrame,
                  mode: str) -> dict:
    """Source -> target shift decomposition plus per-class distances."""
    col = score_column(mode)
    src = {name: source_videos.loc[source_videos["label"] == lab, col].to_numpy(float)
           for lab, name in ((0, "real"), (1, "fake"))}
    tgt = {name: target_videos.loc[target_videos["label"] == lab, col].to_numpy(float)
           for lab, name in ((0, "real"), (1, "fake"))}

    s_src = {k: distribution_stats(v) for k, v in src.items()}
    s_tgt = {k: distribution_stats(v) for k, v in tgt.items()}

    shift_real = s_tgt["real"]["mean"] - s_src["real"]["mean"]
    shift_fake = s_tgt["fake"]["mean"] - s_src["fake"]["mean"]
    out = {
        "mode": mode,
        "shift_real": float(shift_real),
        "shift_fake": float(shift_fake),
        "common_shift": float((shift_real + shift_fake) / 2.0),
        "differential_shift": float(shift_fake - shift_real),
        "scale_real": float(s_tgt["real"]["std"] / s_src["real"]["std"])
        if s_src["real"]["std"] > 0 else float("nan"),
        "scale_fake": float(s_tgt["fake"]["std"] / s_src["fake"]["std"])
        if s_src["fake"]["std"] > 0 else float("nan"),
        "separation_source": float(s_src["fake"]["mean"] - s_src["real"]["mean"]),
        "separation_target": float(s_tgt["fake"]["mean"] - s_tgt["real"]["mean"]),
    }
    for name in ("real", "fake"):
        if src[name].size and tgt[name].size:
            out[f"wasserstein_{name}"] = float(wasserstein_distance(src[name], tgt[name]))
            out[f"ks_{name}"] = float(ks_2samp(src[name], tgt[name]).statistic)
        else:
            out[f"wasserstein_{name}"] = float("nan")
            out[f"ks_{name}"] = float("nan")
    for domain, stats in (("source", s_src), ("target", s_tgt)):
        for name in ("real", "fake"):
            out.update({f"{domain}_{name}_{k}": v for k, v in stats[name].items()})
    return out


def classify_failure(row: pd.Series | dict, ba_drop_threshold: float = 0.05) -> str:
    """Label the dominant cause of a threshold failure for one (checkpoint, mode).

    Expects `r_transfer`, `common_shift`, `differential_shift`, `scale_real`,
    `scale_fake` and `target_auc`. Deliberately coarse — the numbers are in the
    tables; this is the one-word summary used in the report.
    """
    r = float(row["r_transfer"])
    if r < ba_drop_threshold:
        return "threshold transfers"
    common = abs(float(row["common_shift"]))
    diff = abs(float(row["differential_shift"]))
    scale = max(abs(float(row.get("scale_real", 1.0)) - 1.0),
                abs(float(row.get("scale_fake", 1.0)) - 1.0))
    ranked = {
        "class-conditional (differential) shift": diff,
        "global score translation": common,
        "score scale change": scale * max(common, 1.0),
    }
    return max(ranked, key=ranked.get)
