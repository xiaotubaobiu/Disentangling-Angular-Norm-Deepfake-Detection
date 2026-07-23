#!/usr/bin/env python
"""Q4 extraction + multi-forgery common-direction alignment (spec §7.2-7.6).

Smoke pass first (clip / seed_42, ffpp + cdf) to confirm the multi-forgery
common-direction signal BEFORE the full 15-checkpoint extraction.

Pipeline (per checkpoint x domain):
  1. Build same-frame groups from the per-frame parquet: key =
     (domain, source_video_id, frame_tok); keep groups that contain a real
     frame + >=2 fake methods. Sample min(groups_target, available) groups.
  2. Extract raw features z for the union of (real + fake) frame paths in the
     sampled groups; cache to outputs/z_cache/<checkpoint_id>__<domain>.parquet
     (one row per frame: image_path, video_id, method_id, label,
     source_video_id, frame_tok, domain, z0..z{D-1}). Reuse cache if present.
  3. Align:
     - same-frame alignment: mean over groups & fake-method pairs (m,n) of
       cosine_between(d_m, d_n) for d = chord and d = tangent (FS pairs only,
       FR pairs only, and all-pairs reported SEPARATELY -- never pooled).
     - mismatch negative: shuffle real-frame assignment across groups (fixed
       seed), recompute chord cosines. Tests real-frame specificity.
     - per-method cosine matrix (mean chord-displacement cosine between each
       method pair across groups).
     - residual vs classifier w: w_signed = w_fake - w_real; for each chord
       displacement d, project out w then cosine_between(d_m_perp, d_n_perp).
       If residual collapses, the "common direction" is just classifier w.
     - p_global + LOFO + PCA (FF++ estimate, CDF test):
         p_global = mean_chord of all FF++ chord displacements
         per CDF fake displacement, alignment = cosine_between(d, p_global)
         LOFO = recompute p_global excluding one method, measure held-out
                method's mean alignment to the leave-one-out p_global
         PCA = explained variance ratio of top components of the FF++ chord
                displacement matrix.
  4. Tables: outputs/tables/Q4_multiforgery_alignment.csv,
              outputs/tables/Q4_leave_one_method_out.csv.
     Figures: outputs/figures/Q4_same_frame_vs_random_alignment.png,
              Q4_method_cosine_matrix.png, Q4_PCA_explained_variance.png,
              Q4_leave_one_method_out.png,
              Q4_residual_after_classifier_proj.png.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from deepfake_detection.data.constants import (  # noqa: E402
    FACE_SWAP_METHODS,
    REENACTMENT_METHODS,
)
from deepfake_detection.diagnosis.checkpoint_index import discover_checkpoints  # noqa: E402
from deepfake_detection.study.canonical import add_canonical_keys  # noqa: E402
from deepfake_detection.study.config import load_config  # noqa: E402
from deepfake_detection.study.io import load_frame_subset  # noqa: E402
from deepfake_detection.study.q4_alignment import (  # noqa: E402
    chord,
    cosine_between,
    mean_chord,
    project_out_w,
    tangent,
)
from deepfake_detection.study.q4_extract import (  # noqa: E402
    build_model_from_checkpoint,
    extract_checkpoint,
    signed_head_from_model,
)

CFG = load_config(ROOT / "configs" / "study_config.yaml")
CACHE = ROOT / CFG.paths["z_cache"]
TAB = ROOT / CFG.paths["tables"]
FIG = ROOT / CFG.paths["figures"]
CACHE.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

FS = set(FACE_SWAP_METHODS)
FR = set(REENACTMENT_METHODS)

PCA_TOPK = 5


# --------------------------------------------------------------------------- #
# Group construction
# --------------------------------------------------------------------------- #
def build_groups(domain: str, n_groups: int, seed: int) -> dict:
    """Build same-frame groups (real + >=2 fakes) from the per-frame parquet.

    Returns dict[group_key=(domain, source_video_id, frame_tok)] ->
        dict[method_id] -> (image_path, label_int)
    """
    df = add_canonical_keys(
        load_frame_subset(
            str(ROOT / CFG.parquet["frame"]),
            train_mode="raw",
            infer_mode="raw",
            backbones=["clip"],
            seeds=[42],
            domains=[domain],
        )
    )
    df["frame_tok"] = df["canonical_frame_id"].str.split("|").str[-1]

    grouped: dict[tuple, dict] = defaultdict(dict)
    for r in df.itertuples():
        grouped[(r.domain, r.source_video_id, r.frame_tok)][r.method_id] = (
            r.image_path,
            int(r.label),
        )

    qualify = {
        k: m
        for k, m in grouped.items()
        if "real" in m and len([x for x in m if x != "real"]) >= 2
    }
    keys = sorted(qualify.keys())  # deterministic ordering before sampling
    chosen = random.Random(seed).sample(keys, min(n_groups, len(keys)))
    return {k: qualify[k] for k in chosen}


# --------------------------------------------------------------------------- #
# Feature extraction (cached)
# --------------------------------------------------------------------------- #
def extract(ck, domain: str, groups: dict, device) -> pd.DataFrame:
    """Extract raw z for all frames in `groups`, cache per (ckpt, domain)."""
    out = CACHE / f"{ck.checkpoint_id}__{domain}.parquet"
    if out.exists():
        return pd.read_parquet(out)

    paths: list[str] = []
    for m in groups.values():
        for _meth, (p, _lbl) in m.items():
            paths.append(p)
    paths = list(dict.fromkeys(paths))  # dedup, preserve order

    ff = extract_checkpoint(ck, domain, paths, device=device, batch_size=64)
    D = ff.features.shape[1]
    recs = pd.DataFrame(
        {
            "image_path": ff.image_paths,
            "video_id": ff.video_ids,
            "method_id": ff.method_ids,
            "label": ff.labels,
            "domain": domain,
        }
    )
    recs = add_canonical_keys(recs)
    recs["frame_tok"] = recs["canonical_frame_id"].str.split("|").str[-1]
    zdf = pd.DataFrame(ff.features, columns=[f"z{i}" for i in range(D)])
    df = pd.concat([recs.reset_index(drop=True), zdf.reset_index(drop=True)], axis=1)
    df.to_parquet(out)
    return df


def signed_w(ck, device) -> np.ndarray:
    """Load classifier, return w_signed = w_fake - w_real (D,)."""
    model, _cfg = build_model_from_checkpoint(ck, device=device)
    head = signed_head_from_model(model)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return head.w


def unit(v):
    v = np.asarray(v, float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# --------------------------------------------------------------------------- #
# Alignment analysis
# --------------------------------------------------------------------------- #
def analyze(ck, domain: str, cached: pd.DataFrame, groups: dict, w_signed: np.ndarray):
    """Compute the full alignment metric bundle for one (ckpt, domain).

    Returns:
        row: dict of headline metrics for Q4_multiforgery_alignment.csv.
        method_pairs: dict[(m,n)] -> list[float] of chord cosines (m<n order).
        chords_all: (Nfake_total, D) chord displacements aligned with method_all.
        method_all: list[str] same length as chords_all rows (per-displacement method).
        pca_vec: list[float] top-K explained variance ratios.
        lofo: list[dict] leave-one-method-out rows.
    """
    ZCOL = [c for c in cached.columns if c.startswith("z")]
    path2z = {
        r.image_path: np.asarray([getattr(r, c) for c in ZCOL], float)
        for r in cached.itertuples()
    }
    wh = unit(w_signed)
    gkeys = list(groups.keys())

    # Per-group per-method chord/tangent displacements (and the base u_r).
    disp_chord: dict[tuple, dict] = defaultdict(dict)   # k -> method -> chord
    disp_tan: dict[tuple, dict] = defaultdict(dict)     # k -> method -> tangent
    real_zs = {}
    for k in gkeys:
        zr = path2z.get(groups[k]["real"][0])
        if zr is None:
            continue
        real_zs[k] = zr
        ur = unit(zr)
        for meth, (p, _lbl) in groups[k].items():
            if meth == "real":
                continue
            zf = path2z.get(p)
            if zf is None:
                continue
            uf = unit(zf)
            disp_chord[k][meth] = chord(uf, ur)
            disp_tan[k][meth] = tangent(uf, ur)

    # ---- same-frame pairwise cosines (split FS / FR / all) ---------------- #
    same_chord_fs, same_chord_fr, same_chord_all = [], [], []
    same_tan_fs, same_tan_fr, same_tan_all = [], [], []
    method_pairs: dict[tuple, list[float]] = defaultdict(list)
    res_cos_all = []
    chords_all: list[np.ndarray] = []
    method_all: list[str] = []

    valid_keys = [k for k in gkeys if k in disp_chord and disp_chord[k]]
    for k in valid_keys:
        ms = sorted(disp_chord[k].keys())
        for m in ms:
            chords_all.append(disp_chord[k][m])
            method_all.append(m)
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                ca, cb = disp_chord[k][a], disp_chord[k][b]
                ta, tb = disp_tan[k][a], disp_tan[k][b]
                cc = cosine_between(ca, cb)
                ct = cosine_between(ta, tb)
                # method_pairs keyed by sorted tuple for global aggregation
                mp = tuple(sorted((a, b)))
                method_pairs[mp].append(cc)
                same_chord_all.append(cc)
                same_tan_all.append(ct)
                res_cos_all.append(
                    cosine_between(project_out_w(ca, wh), project_out_w(cb, wh))
                )
                both_fs = a in FS and b in FS
                both_fr = a in FR and b in FR
                if both_fs:
                    same_chord_fs.append(cc)
                    same_tan_fs.append(ct)
                elif both_fr:
                    same_chord_fr.append(cc)
                    same_tan_fr.append(ct)

    # ---- mismatch negative: shuffle real-frame assignment across groups --- #
    rng = np.random.default_rng(CFG.global_seed)
    perm = rng.permutation(len(valid_keys))
    mis_chord_all = []
    for idx, k in enumerate(valid_keys):
        zr = real_zs.get(valid_keys[perm[idx]])
        if zr is None:
            continue
        ur = unit(zr)
        ms = []
        for meth, (p, _lbl) in groups[k].items():
            if meth == "real":
                continue
            zf = path2z.get(p)
            if zf is not None:
                ms.append(chord(unit(zf), ur))
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                mis_chord_all.append(cosine_between(ms[i], ms[j]))

    # ---- PCA on chord displacement matrix --------------------------------- #
    chords_arr = np.asarray(chords_all, float) if chords_all else np.empty((0, len(w_signed)))
    pca_vec = pca_explained(chords_arr, k=PCA_TOPK)

    # ---- LOFO + per-method alignment -------------------------------------- #
    lofo = lofo_and_method_alignment(ck, domain, valid_keys, disp_chord, method_pairs)

    row = dict(
        checkpoint=ck.checkpoint_id,
        backbone=ck.backbone,
        seed=ck.seed,
        domain=domain,
        n_groups=len(valid_keys),
        n_pairs=len(same_chord_all),
        same_chord_all=float(np.mean(same_chord_all)) if same_chord_all else float("nan"),
        same_chord_fs=float(np.mean(same_chord_fs)) if same_chord_fs else float("nan"),
        same_chord_fr=float(np.mean(same_chord_fr)) if same_chord_fr else float("nan"),
        same_tangent_all=float(np.mean(same_tan_all)) if same_tan_all else float("nan"),
        same_tangent_fs=float(np.mean(same_tan_fs)) if same_tan_fs else float("nan"),
        same_tangent_fr=float(np.mean(same_tan_fr)) if same_tan_fr else float("nan"),
        mismatch_chord=float(np.mean(mis_chord_all)) if mis_chord_all else float("nan"),
        residual_chord=float(np.mean(res_cos_all)) if res_cos_all else float("nan"),
        pca_top1=float(pca_vec[0]) if pca_vec else float("nan"),
    )
    return row, method_pairs, chords_arr, method_all, pca_vec, lofo


def pca_explained(M: np.ndarray, k: int = 5) -> list:
    """Top-k explained variance ratios of centered displacement matrix M."""
    if M.shape[0] < 2:
        return []
    M = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(M, compute_uv=False)
    var = s ** 2
    tot = var.sum()
    if tot <= 0:
        return []
    return (var / tot)[:k].tolist()


def lofo_and_method_alignment(ck, domain, valid_keys, disp_chord, method_pairs):
    """Leave-one-method-out + per-method alignment to p_global.

    p_global estimated per leave-one-out fold from FF++/this-domain chord
    displacements excluding the held-out method; alignment = mean cosine of
    the held-out method's chord displacements to that p_global.

    Returns list of dicts: checkpoint/backbone/seed/domain/held_out_method/
        n_groups_heldout/lofo_alignment/mean_pairwise_cosine (vs all other
        methods).
    """
    # All methods present across groups.
    methods_present = sorted({m for k in valid_keys for m in disp_chord[k]})
    if not methods_present:
        return []

    rows = []
    for held in methods_present:
        # chords excluding the held-out method
        others = [disp_chord[k][m] for k in valid_keys for m in disp_chord[k] if m != held]
        held_vecs = [disp_chord[k][held] for k in valid_keys if held in disp_chord[k]]
        if not others or not held_vecs:
            continue
        p_loo = mean_chord(np.asarray(others, float))
        align = float(np.mean([cosine_between(v, p_loo) for v in held_vecs]))
        # mean pairwise cosine of held method vs each other method
        pair_cosines = []
        for other in methods_present:
            if other == held:
                continue
            mp = tuple(sorted((held, other)))
            if mp in method_pairs and method_pairs[mp]:
                pair_cosines.append(float(np.mean(method_pairs[mp])))
        rows.append(
            dict(
                checkpoint=ck.checkpoint_id,
                backbone=ck.backbone,
                seed=ck.seed,
                domain=domain,
                held_out_method=held,
                n_heldout_groups=len(held_vecs),
                lofo_alignment=align,
                mean_pairwise_cosine=(
                    float(np.mean(pair_cosines)) if pair_cosines else float("nan")
                ),
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Per-method cosine matrix
# --------------------------------------------------------------------------- #
def method_cosine_matrix(method_pairs: dict) -> tuple[list, np.ndarray]:
    """Mean cosine between each method pair. Returns (method_list, M_sym)."""
    methods = sorted({m for pr in method_pairs for m in pr})
    idx = {m: i for i, m in enumerate(methods)}
    M = np.full((len(methods), len(methods)), np.nan)
    np.fill_diagonal(M, 1.0)
    for (a, b), vals in method_pairs.items():
        if vals:
            c = float(np.mean(vals))
            M[idx[a], idx[b]] = c
            M[idx[b], idx[a]] = c
    return methods, M


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plots(rows, lofo_rows, pca_per_run, method_matrix_runs, smoke_suffix=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not rows:
        return
    df = pd.DataFrame(rows)
    labels = [f"{r['backbone']}\n{r['domain']}" for _, r in df.iterrows()]

    # In smoke mode, figure basenames are prefixed with the backbone (passed in
    # as smoke_suffix == "__<backbone>") so backbones do not overwrite each
    # other's figures. --full (smoke_suffix="") keeps the canonical names.
    sf = smoke_suffix

    # 1. same-frame vs random alignment
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df))
    w = 0.28
    ax.bar(x - w, df["same_chord_all"], w, label="same-frame chord (all)")
    if "same_chord_fs" in df:
        ax.bar(x, df["same_chord_fs"], w, label="same-frame chord (FS)")
    ax.bar(x + w, df["mismatch_chord"], w, label="mismatch (negative)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("mean cosine between method-pair displacements")
    ax.set_title("Q4: same-frame vs random alignment")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"Q4_same_frame_vs_random_alignment{sf}.png", dpi=120)
    plt.close("all")

    # 2. residual after projecting out classifier w
    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(df))
    w = 0.35
    ax.bar(x - w / 2, df["same_chord_all"], w, label="chord cosine")
    ax.bar(x + w / 2, df["residual_chord"], w, label="residual after w-proj")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("cosine")
    ax.set_title("Q4: residual common-direction after projecting out classifier w")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"Q4_residual_after_classifier_proj{sf}.png", dpi=120)
    plt.close("all")

    # 3. PCA explained variance
    fig, ax = plt.subplots(figsize=(7, 5))
    max_k = max((len(p) for p in pca_per_run), default=0)
    for i, (label, pca_vec) in enumerate(pca_per_run):
        if not pca_vec:
            continue
        ax.plot(
            np.arange(1, len(pca_vec) + 1),
            pca_vec,
            marker="o",
            label=label,
        )
    ax.set_xlabel("principal component")
    ax.set_ylabel("explained variance ratio")
    ax.set_title("Q4: PCA of chord displacement matrix")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f"Q4_PCA_explained_variance{sf}.png", dpi=120)
    plt.close("all")

    # 4. leave-one-method-out
    if lofo_rows:
        ldf = pd.DataFrame(lofo_rows)
        fig, ax = plt.subplots(figsize=(9, 5))
        for (ck, dom), g in ldf.groupby(["checkpoint", "domain"]):
            g = g.sort_values("held_out_method")
            ax.plot(
                g["held_out_method"],
                g["lofo_alignment"],
                marker="o",
                label=f"{ck}/{dom}",
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("held-out method alignment to leave-one-out p_global")
        ax.set_xlabel("held-out method")
        ax.set_title("Q4: leave-one-method-out alignment")
        ax.legend(fontsize=7)
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(FIG / f"Q4_leave_one_method_out{sf}.png", dpi=120)
        plt.close("all")

    # 5. per-method cosine matrix (one per run; suffix by checkpoint/domain)
    for (ck, dom), (methods, M) in method_matrix_runs:
        if not methods:
            continue
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_xticks(range(len(methods)))
        ax.set_yticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(methods, fontsize=8)
        for i in range(len(methods)):
            for j in range(len(methods)):
                v = M[i, j]
                if not np.isnan(v):
                    ax.text(
                        j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.6 else "black",
                    )
        ax.set_title(f"Q4 per-method chord-cosine matrix\n{ck} / {dom}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        safe = f"{ck}__{dom}"
        fig.savefig(FIG / f"Q4_method_cosine_matrix__{safe}{sf}.png", dpi=120)
        plt.close("all")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument(
        "--backbone",
        default="clip",
        help="backbone to evaluate in --smoke mode (clip/dinov2/efficientnet); "
             "ignored in --full (all backbones/seeds)",
    )
    ap.add_argument("--n-groups", type=int, default=None,
                    help="override groups-per-domain sample size")
    a = ap.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if a.smoke:
        cks = discover_checkpoints(
            CFG.ckpt_root, modes=["raw"], backbones=[a.backbone], seeds=[42]
        )
        ngrp = a.n_groups if a.n_groups is not None else 400
        domains = list(CFG.domains)
        # In smoke mode, scope output artifacts to the chosen backbone so they
        # do not collide with / clobber other backbones' smoke results (clip is
        # already in the canonical-named files). --full keeps canonical names.
        smoke_suffix = f"__{a.backbone}"
    else:
        cks = discover_checkpoints(CFG.ckpt_root, modes=["raw"])
        ngrp = a.n_groups if a.n_groups is not None else CFG.q4["groups_target"]
        domains = list(CFG.domains)
        smoke_suffix = ""

    if not cks:
        raise RuntimeError("No checkpoints discovered.")

    print(f"[q4] mode={'smoke' if a.smoke else 'full'} "
          f"n_ckpt={len(cks)} domains={domains} n_groups={ngrp} device={device}",
          flush=True)

    rows = []
    lofo_rows = []
    pca_per_run = []          # (label, [evr...])
    method_matrix_runs = []   # ((ck_id, dom), (methods, M))
    method_matrix_first = None

    for ck in cks:
        w = signed_w(ck, device=device)
        print(f"[q4] checkpoint={ck.checkpoint_id} D={len(w)}", flush=True)
        for dom in domains:
            groups = build_groups(dom, ngrp, CFG.global_seed)
            if not groups:
                print(f"[q4]   {dom}: no qualifying groups, skipping", flush=True)
                continue
            cached = extract(ck, dom, groups, device=device)
            row, method_pairs, _chords, _method_all, pca_vec, lofo = analyze(
                ck, dom, cached, groups, w
            )
            rows.append(row)
            lofo_rows.extend(lofo)
            pca_per_run.append((f"{ck.backbone}/{dom}", pca_vec))
            methods, M = method_cosine_matrix(method_pairs)
            method_matrix_runs.append(((ck.checkpoint_id, dom), (methods, M)))
            if method_matrix_first is None:
                method_matrix_first = (methods, M)
            print(
                f"[q4]   {dom}: n_groups={row['n_groups']} "
                f"same_chord_all={row['same_chord_all']:.4f} "
                f"same_chord_fs={row['same_chord_fs']:.4f} "
                f"same_chord_fr={row['same_chord_fr']:.4f} "
                f"mismatch={row['mismatch_chord']:.4f} "
                f"residual={row['residual_chord']:.4f} "
                f"pca_top1={row['pca_top1']:.4f}",
                flush=True,
            )

    # ---- write tables ------------------------------------------------------ #
    # In smoke mode, all artifacts are scoped per-backbone (suffix) so backbones
    # do not clobber each other. --full keeps the canonical names.
    align_df = pd.DataFrame(rows)
    align_df.to_csv(
        TAB / f"Q4_multiforgery_alignment{smoke_suffix}.csv", index=False
    )

    # LOFO table: long format with per-method rows.
    pd.DataFrame(lofo_rows).to_csv(
        TAB / f"Q4_leave_one_method_out{smoke_suffix}.csv", index=False
    )

    # Per-method cosine matrix as a tidy long table (all method pairs, all runs).
    mat_rows = []
    for (ck_id, dom), (methods, M) in method_matrix_runs:
        for i, mi in enumerate(methods):
            for j, mj in enumerate(methods):
                if i <= j:
                    mat_rows.append(
                        dict(
                            checkpoint=ck_id, domain=dom,
                            method_a=mi, method_b=mj,
                            mean_chord_cosine=float(M[i, j]),
                        )
                    )
    pd.DataFrame(mat_rows).to_csv(
        TAB / f"Q4_method_cosine_matrix{smoke_suffix}.csv", index=False
    )

    _plots(rows, lofo_rows, pca_per_run, method_matrix_runs, smoke_suffix=smoke_suffix)

    # ---- console summary --------------------------------------------------- #
    print("\n" + "=" * 78, flush=True)
    print("Q4 multi-forgery alignment summary", flush=True)
    print("=" * 78, flush=True)
    show_cols = [
        "checkpoint", "domain", "n_groups", "n_pairs",
        "same_chord_all", "same_chord_fs", "same_chord_fr",
        "same_tangent_all", "mismatch_chord", "residual_chord", "pca_top1",
    ]
    print(align_df[show_cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"),
          flush=True)
    print(f"\n[q4] wrote {TAB/f'Q4_multiforgery_alignment{smoke_suffix}.csv'}", flush=True)
    print(f"[q4] wrote {TAB/f'Q4_leave_one_method_out{smoke_suffix}.csv'}", flush=True)
    print(f"[q4] wrote {TAB/f'Q4_method_cosine_matrix{smoke_suffix}.csv'}", flush=True)


if __name__ == "__main__":
    main()
