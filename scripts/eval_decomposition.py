#!/usr/bin/env python3
"""Extract features from raw-train EfficientNet checkpoints and compute
AUC under three test-time decompositions: raw, direction-only, length-only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from deepfake_detection.models.factory import build_model
from deepfake_detection.data.builders import build_eval_loader

SEEDS = [42, 7, 123, 999, 2048]
DOMAINS = ["ffpp", "cdf", "dfd"]


def load_config(config_path):
    import yaml
    resolved = config_path
    cfg = None
    while True:
        with open(resolved) as f:
            c = yaml.safe_load(f)
        if "_base_" in c:
            base_ref = c.pop("_base_")
            base_path = Path(resolved).parent / base_ref
            if not base_path.exists():
                base_path = Path(resolved).parent.parent.parent / base_ref
            with open(base_path) as f:
                base = yaml.safe_load(f)
            for k, v in c.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    base[k].update(v)
                else:
                    base[k] = v
            c = base
            if "_base_" not in c:
                break
            resolved = base_path
        else:
            break
    return c


@torch.no_grad()
def extract_domain(model, dataloader, device):
    model.eval()
    all_features, all_labels, all_probs = [], [], []
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        features = model.extract_features(images.float()).float()
        logits = model.classifier(features)
        prob_fake = torch.softmax(logits, dim=1)[:, 1]
        all_features.append(features.cpu().numpy())
        all_labels.append(batch["label"].numpy())
        all_probs.append(prob_fake.cpu().numpy())
    return {
        "features": np.concatenate(all_features),
        "labels": np.concatenate(all_labels),
        "probs": np.concatenate(all_probs),
    }


def compute_decomposition_aucs(features, labels, probs):
    results = {}
    # Raw (classifier output)
    if len(set(labels)) > 1:
        results["raw"] = roc_auc_score(labels, probs)
    else:
        results["raw"] = float("nan")

    # Direction-only: L2 normalize features, recompute logits
    direction = F.normalize(torch.from_numpy(features), dim=-1).numpy()
    norms = np.linalg.norm(features, axis=1)
    # Use direction cosine to class centroids as score
    real_mean = features[labels == 0].mean(axis=0)
    fake_mean = features[labels == 1].mean(axis=0)
    d = fake_mean - real_mean
    d = d / (np.linalg.norm(d) + 1e-8)
    direction_scores = direction @ d
    if len(set(labels)) > 1:
        results["direction_only"] = roc_auc_score(labels, direction_scores)
    else:
        results["direction_only"] = float("nan")

    # Length-only: use feature norm as score
    if len(set(labels)) > 1:
        results["length_only"] = roc_auc_score(labels, norms)
    else:
        results["length_only"] = float("nan")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-features", action="store_true", help="Save extracted features to features/ dir")
    parser.add_argument("--backbone", default="all", choices=["all", "efficientnet", "dinov2", "xception"],
                        help="Which backbone to evaluate")
    args = parser.parse_args()

    BACKBONES = {
        "efficientnet": {"pattern": "efficientnet", "label": "EfficientNet-B0"},
        "dinov2": {"pattern": "dinov2", "label": "DINOv2-ViT-B/14"},
        "xception": {"pattern": "xception", "label": "Xception"},
    }

    selected = list(BACKBONES.keys()) if args.backbone == "all" else [args.backbone]

    for bb_key in selected:
        bb_info = BACKBONES[bb_key]
        exp_dir = Path(args.experiments_dir)
        raw_dirs = []
        for d in sorted(exp_dir.iterdir()):
            config_path = d / "config" / "config.yaml"
            results_path = d / "output" / "results_summary.txt"
            if not config_path.exists() or not results_path.exists():
                continue
            with open(config_path) as f:
                import yaml
                cfg = yaml.safe_load(f)
            name = cfg.get("experiment_name", "")
            if "raw" in name and bb_info["pattern"] in name and "multibb" in name:
                raw_dirs.append((name, d))

        raw_dirs.sort(key=lambda x: x[0])

        if not raw_dirs:
            print(f"No raw {bb_info['label']} experiments found.\n")
            continue

        print(f"=== {bb_info['label']} ({len(raw_dirs)} experiments) ===\n")

        all_rows = []
        for name, d in raw_dirs:
            ckpt_path = d / "output" / "best_model.pth"
            config_path = d / "config" / "config.yaml"
            if not ckpt_path.exists():
                print(f"  Skipping {name}: no checkpoint")
                continue

            cfg = load_config(str(config_path))
            model = build_model(cfg["model"])
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            model = model.to(args.device)
            model.eval()

            seed = cfg["train"]["seed"]
            row = {"experiment": name, "seed": seed}

            for domain in DOMAINS:
                loader = build_eval_loader(cfg, domain=domain, distributed=False)
                data = extract_domain(model, loader, args.device)

                aucs = compute_decomposition_aucs(data["features"], data["labels"], data["probs"])
                row[f"{domain}_raw"] = aucs["raw"]
                row[f"{domain}_direction"] = aucs["direction_only"]
                row[f"{domain}_length"] = aucs["length_only"]

                if args.save_features:
                    feat_dir = Path("features") / bb_key / f"raw_s{seed}"
                    feat_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(feat_dir / f"{domain}.npz", **data)

            all_rows.append(row)
            print(f"  {name}: done")

        # Print summary table
        print(f"\n{'Seed':<8} {'Decomposition':<18} {'FF++ AUC':>10} {'CDF AUC':>10} {'DFD AUC':>10}")
        print("-" * 60)

        for decomposition, suffix in [("raw", "raw"), ("direction-only", "direction"), ("length-only", "length")]:
            decomp_rows = []
            for row in all_rows:
                ff = row.get(f"ffpp_{suffix}", float("nan"))
                cd = row.get(f"cdf_{suffix}", float("nan"))
                df = row.get(f"dfd_{suffix}", float("nan"))
                decomp_rows.append((row["seed"], ff, cd, df))

            for seed, ff, cd, df in decomp_rows:
                print(f"{seed:<8} {decomposition:<18} {ff:>10.4f} {cd:>10.4f} {df:>10.4f}")

            ffs = [r[1] for r in decomp_rows if not np.isnan(r[1])]
            cds = [r[2] for r in decomp_rows if not np.isnan(r[2])]
            dfs = [r[3] for r in decomp_rows if not np.isnan(r[3])]
            ff_m = f"{np.mean(ffs):.4f}±{np.std(ffs):.4f}" if ffs else "—"
            cd_m = f"{np.mean(cds):.4f}±{np.std(cds):.4f}" if cds else "—"
            df_m = f"{np.mean(dfs):.4f}±{np.std(dfs):.4f}" if dfs else "—"
            print(f"{'mean':<8} {decomposition:<18} {ff_m:>10} {cd_m:>10} {df_m:>10}")
            print()

    print("Done.")


if __name__ == "__main__":
    main()
