#!/usr/bin/env python3
"""Extract features from CLIP raw checkpoints and save to features/clip/."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from deepfake_detection.models.factory import build_model
from deepfake_detection.data.builders import build_eval_loader

SEEDS = [42, 7, 123, 999, 2048]
DOMAINS = ["ffpp", "cdf", "dfd"]

BASE_CFG = {
    "dataset": {
        "root": "/Dataset/deepfake_detection/DF40_all",
        "train_videos_per_method": 586,
        "val_videos_per_method": 100,
        "frames_per_video": 8,
        "image_size": 224,
        "methods": [],
        "real_sampling": "oversample_per_method",
        "use_mask": False,
    },
    "augment": {
        "brightness_contrast_p": 0.5, "brightness_limit": 0.1,
        "compression_p": 0.2, "compression_quality_lower": 40, "compression_quality_upper": 100,
        "contrast_limit": 0.1, "horizontal_flip_p": 0.5,
        "rotate_limit": 10, "rotate_p": 0.5,
    },
}


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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_dir = Path("experiments")

    for d in sorted(exp_dir.iterdir()):
        results_path = d / "output" / "results_summary.txt"
        config_path = d / "config" / "config.yaml"
        if not results_path.exists() or not config_path.exists():
            continue
        with open(results_path) as f:
            line = f.readline()
        name = line.split(":", 1)[1].strip() if ":" in line else ""
        if "raw" not in name or "norm_shortcut" not in name:
            continue

        ckpt_path = d / "output" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"  Skip {name}: no checkpoint")
            continue

        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        seed = cfg["train"]["seed"]

        feat_dir = Path("features") / "clip" / f"raw_s{seed}"
        if feat_dir.exists() and all((feat_dir / f"{domain}.npz").exists() for domain in DOMAINS):
            print(f"  Skip {name}: features already exist")
            continue

        print(f"Processing {name} (seed={seed})...")
        model = build_model(cfg["model"])
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device)
        model.eval()

        cfg_merged = {**cfg, **BASE_CFG}
        for k in set(list(cfg.keys()) + list(BASE_CFG.keys())):
            if k in cfg and k in BASE_CFG and isinstance(cfg[k], dict) and isinstance(BASE_CFG[k], dict):
                cfg_merged[k] = {**BASE_CFG[k], **cfg[k]}

        feat_dir.mkdir(parents=True, exist_ok=True)
        for domain in DOMAINS:
            loader = build_eval_loader(cfg_merged, domain=domain, distributed=False)
            data = extract_domain(model, loader, device)
            data["norms"] = np.linalg.norm(data["features"], axis=1)
            np.savez(feat_dir / f"{domain}.npz", **data)
            print(f"  {domain}: features={data['features'].shape}")

    print("Done.")


if __name__ == "__main__":
    main()
