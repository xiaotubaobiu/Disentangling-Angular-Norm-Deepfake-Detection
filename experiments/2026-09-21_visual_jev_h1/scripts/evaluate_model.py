from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, roc_auc_score, roc_curve


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from config_io import load_config
from data.builders import build_eval_loader
from engine.ddp import barrier, init_ddp, is_main_process
from models.factory import build_model


def probability_metrics(labels: np.ndarray, scores: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fnr - fpr)))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "ap": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "acc_at_0_5": float(accuracy_score(labels, scores >= 0.5)),
        "eer": float((fpr[index] + fnr[index]) / 2.0),
        "count": int(len(labels)),
    }


@torch.no_grad()
def evaluate_domain(model, loader, device: torch.device, verify_swap: bool) -> tuple[dict, list[dict], dict | None]:
    model.eval()
    local_rows: list[tuple] = []
    swap_check = None
    started = time.time()

    for batch_index, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        logits = model(images)
        probabilities = torch.softmax(logits.float(), dim=1)[:, 1]

        if verify_swap and swap_check is None:
            features = model.extract_features(images)
            original = model.classify_features(features, (0, 1))
            swapped = model.classify_features(features, (1, 0))
            max_error = float((swapped - original[:, [1, 0]]).abs().max().cpu())
            labels_for_check = batch["label"].to(device)
            original_loss = torch.nn.functional.cross_entropy(original, labels_for_check)
            swapped_loss = torch.nn.functional.cross_entropy(swapped, 1 - labels_for_check)
            loss_error = float((swapped_loss - original_loss).abs().cpu())
            swap_check = {
                "max_logit_permutation_error": max_error,
                "remapped_loss_error": loss_error,
                "passed": bool(max_error <= 1e-6 and loss_error <= 1e-6),
            }

        labels_np = batch["label"].numpy().astype(np.int64, copy=False)
        scores_np = probabilities.cpu().numpy().astype(np.float64, copy=False)
        for sample_id, key, method, video_id, label, score in zip(
            batch["sample_id"],
            batch["video_key"],
            batch["method_id"],
            batch["video_id"],
            labels_np.tolist(),
            scores_np.tolist(),
        ):
            local_rows.append((sample_id, key, method, video_id, int(label), float(score)))

        if (batch_index + 1) % 200 == 0 and is_main_process():
            print(
                f"evaluated_local_batches={batch_index + 1}/{len(loader)} "
                f"elapsed_s={time.time() - started:.1f}",
                flush=True,
            )

    if torch.distributed.is_initialized():
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, local_rows)
        local_rows = [row for rank_rows in gathered for row in rank_rows]

    seen_sample_ids: set[str] = set()
    frame_labels: list[int] = []
    frame_scores: list[float] = []
    videos: OrderedDict[str, dict] = OrderedDict()
    for sample_id, key, method, video_id, label, score in local_rows:
        if sample_id in seen_sample_ids:
            continue
        seen_sample_ids.add(sample_id)
        frame_labels.append(label)
        frame_scores.append(score)
        current = videos.setdefault(key, {
            "video_key": key,
            "method_id": method,
            "video_id": video_id,
            "label": int(label),
            "score_sum": 0.0,
            "num_frames": 0,
        })
        if current["label"] != int(label):
            raise ValueError(f"Conflicting labels for {key}")
        current["score_sum"] += float(score)
        current["num_frames"] += 1

    labels = np.asarray(frame_labels, dtype=np.int64)
    scores = np.asarray(frame_scores, dtype=np.float64)
    video_rows = []
    for current in videos.values():
        video_rows.append({
            "video_key": current["video_key"],
            "method_id": current["method_id"],
            "video_id": current["video_id"],
            "label": current["label"],
            "prob_fake": current["score_sum"] / current["num_frames"],
            "num_frames": current["num_frames"],
        })
    video_labels = np.asarray([row["label"] for row in video_rows], dtype=np.int64)
    video_scores = np.asarray([row["prob_fake"] for row in video_rows], dtype=np.float64)
    metrics = {
        "frame": probability_metrics(labels, scores),
        "video": probability_metrics(video_labels, video_scores),
        "elapsed_seconds": time.time() - started,
    }
    return metrics, video_rows, swap_check


def write_video_rows(rows: list[dict], path: Path) -> None:
    fields = ["video_key", "method_id", "video_id", "label", "prob_fake", "num_frames"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--domains", nargs="+", default=["ffpp", "cdf"], choices=["ffpp", "cdf"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["train"]["per_gpu_batch"] = args.batch_size
    cfg["train"]["num_workers"] = args.num_workers
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_rank = init_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    distributed = torch.distributed.is_initialized()

    model = build_model(cfg["model"])
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    provenance = {
        "model_id": args.model_id,
        "model_name": cfg["model"]["name"],
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch_zero_based": int(checkpoint.get("epoch", -1)),
        "checkpoint_validation_auc": float(checkpoint.get("auc", float("nan"))),
        "total_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
        "device": str(device),
        "domains": {},
    }

    for domain in args.domains:
        if is_main_process():
            print(f"Evaluating {args.model_id} on {domain}", flush=True)
        loader = build_eval_loader(cfg, domain=domain, distributed=distributed)
        metrics, video_rows, swap_check = evaluate_domain(
            model,
            loader,
            device,
            verify_swap=cfg["model"]["name"] == "clip_visual_jev",
        )
        if is_main_process():
            provenance["domains"][domain] = {
                **metrics,
                "candidate_swap_check": swap_check,
            }
            write_video_rows(video_rows, output_dir / f"{domain}_video_predictions.csv")
            print(json.dumps({domain: provenance["domains"][domain]}, indent=2), flush=True)
        barrier()

    if is_main_process():
        with (output_dir / "evaluation.json").open("w") as handle:
            json.dump(provenance, handle, indent=2, sort_keys=True)
    if distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
