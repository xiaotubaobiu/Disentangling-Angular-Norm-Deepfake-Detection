#!/usr/bin/env python3
"""Generate black-and-white–friendly figures for thesis printing.

Usage:
    python scripts/generate_bw_figures.py \
        --checkpoint-dir experiments/2026-05-05_paper_feature_geometry/results/checkpoints \
        --output-dir figures/frame_level_bw \
        --gpu 0

Extracts features from saved checkpoints and generates all five figure types
with grayscale-safe styling (hatching, line styles, markers).
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.nn.functional as F

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BACKBONES = ["clip", "dinov2", "efficientnet"]
DOMAINS = ["ffpp", "cdf", "dfd"]
SEEDS = [7, 42, 123, 999, 2048]

BW_REAL_STYLE = {"color": "0.3", "edgecolor": "black", "linewidth": 0.8}
BW_FAKE_STYLE = {"color": "0.75", "edgecolor": "black", "linewidth": 0.8}
REAL_HATCH = "///"
FAKE_HATCH = "..."

LINE_STYLES = {
    "Accuracy": {"color": "black", "linestyle": "-", "linewidth": 1.5},
    "Balanced Accuracy": {"color": "0.3", "linestyle": "--", "linewidth": 1.5},
    "TPR (Fake Recall)": {"color": "0.5", "linestyle": "-.", "linewidth": 1.2},
    "FPR (Real → Fake)": {"color": "0.6", "linestyle": ":", "linewidth": 1.2},
}

_PHASE = Path(__file__).resolve().parents[1]   # experiments/<date>_<phase>/

TSNE_MARKERS = {"Real": ("o", "white", "black"), "Fake": ("^", "0.5", "black")}


# ─────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────

def load_model(backbone: str, ckpt_path: str, device: torch.device):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from models.clip_classifier import CLIPFineTuneBinaryClassifier
    from models.dinov2 import DINOv2BinaryClassifier
    from models.efficientnet import EfficientNetBinaryClassifier

    if backbone == "clip":
        model = CLIPFineTuneBinaryClassifier("ViT-B/16", normalize_features=False)
    elif backbone == "dinov2":
        model = DINOv2BinaryClassifier(normalize_features=False)
    elif backbone == "efficientnet":
        model = EfficientNetBinaryClassifier(normalize_features=False)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def extract_features_for_domain(model, cfg, domain, device):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from data.builders import build_eval_loader

    loader = build_eval_loader(cfg, domain=domain, distributed=False)
    all_features, all_labels, all_probs, all_norms, all_vids = [], [], [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            features = model.extract_features(images.float()).float()
            logits = model.classifier(features)
            prob_fake = torch.softmax(logits, dim=1)[:, 1]
            norm = features.norm(dim=-1)

            all_features.append(features.cpu())
            all_labels.extend(batch["label"].tolist())
            all_probs.extend(prob_fake.cpu().tolist())
            all_norms.extend(norm.cpu().tolist())
            all_vids.extend(batch["video_id"])

    features_cat = torch.cat(all_features, dim=0).numpy()
    return {
        "features": features_cat,
        "labels": np.array(all_labels),
        "probs": np.array(all_probs),
        "norms": np.array(all_norms),
        "video_ids": all_vids,
    }


# ─────────────────────────────────────────────────────────────
# Plot functions
# ─────────────────────────────────────────────────────────────

def plot_norm_histogram(data, backbone, domain, output_dir):
    labels = data["labels"]
    norms = data["norms"]
    real_norms = norms[labels == 0]
    fake_norms = norms[labels == 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(min(norms), max(norms), 51)
    ax.hist(real_norms, bins=bins, density=True, alpha=0.7,
            label="Real", hatch=REAL_HATCH, **BW_REAL_STYLE)
    ax.hist(fake_norms, bins=bins, density=True, alpha=0.7,
            label="Fake", hatch=FAKE_HATCH, **BW_FAKE_STYLE)
    ax.set_xlabel("Feature L2 Norm")
    ax.set_ylabel("Density")
    ax.set_title(f"{backbone} — {domain.upper()} Feature Norm Distribution")
    ax.legend(frameon=True, edgecolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(output_dir) / "norm_histogram" / f"{backbone}_{domain}_norm_hist.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_tsne(data, backbone, output_dir):
    from sklearn.manifold import TSNE

    features = data["features"]
    labels = data["labels"]
    norms = data["norms"]

    for feat_type, feat_name in [("raw", features), ("normalized", F.normalize(torch.tensor(features), dim=-1).numpy())]:
        tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto",
                     n_iter=1000, random_state=42)
        emb = tsne.fit_transform(feat_name)

        # By label
        fig, ax = plt.subplots(figsize=(5, 5))
        for lbl, name, (marker, face, edge) in [(0, "Real", TSNE_MARKERS["Real"]),
                                                   (1, "Fake", TSNE_MARKERS["Fake"])]:
            mask = labels == lbl
            ax.scatter(emb[mask, 0], emb[mask, 1], marker=marker,
                      facecolors=face, edgecolors=edge, s=8, linewidths=0.3,
                      label=name, alpha=0.6)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{backbone} ({feat_type}) — label")
        ax.legend(frameon=True, edgecolor="black", markerscale=2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        out = Path(output_dir) / "tsne" / f"{backbone}_raw_{feat_type}_label.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")

        # By norm
        fig, ax = plt.subplots(figsize=(5.5, 5))
        sc = ax.scatter(emb[:, 0], emb[:, 1], c=norms, cmap="gray_r",
                       s=8, alpha=0.6, edgecolors="none")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_title(f"{backbone} ({feat_type}) — norm")
        plt.colorbar(sc, ax=ax, label="L2 Norm")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        out = Path(output_dir) / "tsne" / f"{backbone}_raw_{feat_type}_norm.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")


def plot_cosine_heatmap(all_data, backbone, output_dir):
    centers = {}
    for domain in DOMAINS:
        data = all_data[domain]
        features = data["features"]
        labels = data["labels"]
        for lbl, name in [(0, "real"), (1, "fake")]:
            mask = labels == lbl
            feats = features[mask]
            normed = feats / np.linalg.norm(feats, axis=1, keepdims=True)
            center = normed.mean(axis=0)
            center = center / np.linalg.norm(center)
            centers[f"{domain}_{name}"] = center

    keys = [f"{d}_{l}" for d in DOMAINS for l in ["real", "fake"]]
    n = len(keys)
    sim = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            sim[i, j] = np.dot(centers[ki], centers[kj])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim, cmap="gray_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(keys, fontsize=9)
    for i in range(n):
        for j in range(n):
            color = "white" if sim[i, j] > 0.5 else "black"
            ax.text(j, i, f"{sim[i, j]:.3f}", ha="center", va="center",
                   fontsize=8, color=color)
    ax.set_title(f"{backbone} — Cross-Domain Cosine Similarity")
    plt.colorbar(im, ax=ax, shrink=0.8)

    out = Path(output_dir) / "cosine_heatmap" / f"{backbone}_cross_domain_cosine.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_calibration(data, backbone, domain, output_dir, n_bins=10):
    probs = data["probs"]
    labels = data["labels"]

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_means = []
    bin_accs = []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            bin_means.append((lo + hi) / 2)
            bin_accs.append(0)
        else:
            bin_means.append(probs[mask].mean())
            bin_accs.append(labels[mask].mean())

    ece = sum(abs(a - m) * ((probs >= lo) & (probs < hi)).sum()
              for (lo, hi), a, m in zip(zip(bin_edges[:-1], bin_edges[1:]), bin_accs, bin_means)) / len(probs)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(bin_centers, bin_accs, width=1.0 / n_bins * 0.9,
           color="0.7", edgecolor="black", linewidth=0.8, label="Model")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"{backbone} — {domain.upper()} Reliability Diagram\nECE = {ece:.4f}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=True, edgecolor="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(output_dir) / "calibration" / f"{backbone}_{domain}_reliability.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_threshold_curve(data, backbone, domain, output_dir):
    # Aggregate to video level
    video_scores = defaultdict(lambda: {"scores": [], "label": 0})
    for vid, prob, lbl in zip(data["video_ids"], data["probs"], data["labels"]):
        video_scores[vid]["scores"].append(prob)
        video_scores[vid]["label"] = lbl

    labels = []
    scores = []
    for v in video_scores.values():
        labels.append(v["label"])
        scores.append(np.mean(v["scores"]))
    labels = np.array(labels)
    scores = np.array(scores)

    thresholds = np.linspace(0, 1, 1001)
    accs, bal_accs, tprs, fprs = [], [], [], []
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        acc = (tp + tn) / max(tp + tn + fp + fn, 1)
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        fpr = fp / max(fp + tn, 1)
        bal_acc = (tpr + tnr) / 2
        accs.append(acc)
        bal_accs.append(bal_acc)
        tprs.append(tpr)
        fprs.append(fpr)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, vals in [("Accuracy", accs), ("Balanced Accuracy", bal_accs),
                        ("TPR (Fake Recall)", tprs), ("FPR (Real → Fake)", fprs)]:
        ax.plot(thresholds, vals, label=name, **LINE_STYLES[name])
    ax.axvline(0.5, color="0.6", linestyle=":", linewidth=0.8, label="Default threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"{backbone} — {domain.upper()} Threshold Sensitivity")
    ax.legend(frameon=True, edgecolor="black", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out = Path(output_dir) / "threshold" / f"{backbone}_{domain}_threshold_curve.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def find_checkpoint(ckpt_dir: str, backbone: str, train_mode: str, seed: int) -> str | None:
    pattern = Path(ckpt_dir) / backbone / train_mode / f"seed_{seed}_*" / "output" / "best_model.pth"
    import glob
    matches = glob.glob(str(pattern))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default=str(_PHASE / "results" / "checkpoints"))
    parser.add_argument("--output-dir", default=str(_PHASE / "figures" / "frame_level_bw"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42, help="Which seed checkpoint to use for visualization")
    parser.add_argument("--config-dir", default=str(_PHASE / "configs"))
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from config_io import load_config

    for backbone in BACKBONES:
        print(f"\n{'='*60}")
        print(f"Processing backbone: {backbone}")
        print(f"{'='*60}")

        ckpt_path = find_checkpoint(args.checkpoint_dir, backbone, "raw", args.seed)
        if ckpt_path is None:
            print(f"  WARNING: No checkpoint found for {backbone}/raw/seed_{args.seed}, skipping.")
            continue

        cfg_path = Path(args.config_dir) / f"{backbone}_base.yaml"
        cfg = load_config(str(cfg_path))

        print(f"  Loading model from {ckpt_path}")
        model = load_model(backbone, ckpt_path, device)

        all_domain_data = {}
        for domain in DOMAINS:
            print(f"  Extracting features for {domain}...")
            data = extract_features_for_domain(model, cfg, domain, device)
            all_domain_data[domain] = data

            print(f"  Plotting norm histogram for {domain}...")
            plot_norm_histogram(data, backbone, domain, args.output_dir)

            print(f"  Plotting calibration for {domain}...")
            plot_calibration(data, backbone, domain, args.output_dir)

            print(f"  Plotting threshold curve for {domain}...")
            plot_threshold_curve(data, backbone, domain, args.output_dir)

        print(f"  Plotting t-SNE for DFD...")
        plot_tsne(all_domain_data["dfd"], backbone, args.output_dir)

        print(f"  Plotting cosine heatmap...")
        plot_cosine_heatmap(all_domain_data, backbone, args.output_dir)

        del model
        torch.cuda.empty_cache()

    print(f"\nDone! All B&W figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
