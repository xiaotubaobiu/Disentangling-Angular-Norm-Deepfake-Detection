"""Real pretrained GPU forward/backward smoke using source-training frames only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[3]
PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config_io import load_config
from data.builders import build_train_dataset
from data.semantic_language import load_language_bank
from models.smolvlm_candidate import CandidateTextLookup, SmolVLMCandidateScorer


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0,1,2,3":
        raise RuntimeError("this experiment permits only physical CUDA_VISIBLE_DEVICES=0,1,2,3")
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    pretrained = PHASE / "results/pretrained/smolvlm_256m"
    model, processor = SmolVLMCandidateScorer.from_pretrained(pretrained, score_batch_size=32)
    model.to(device)
    e2 = ROOT / "experiments/2026-09-22_semantic_visual_jev_e2"
    cfg = load_config(e2 / "configs/semantic_jev.yaml")
    cfg["train"]["seed"] = 42
    base = build_train_dataset(cfg)
    bank = load_language_bank(e2 / "configs/language_bank.yaml")
    lookup = CandidateTextLookup.build(model, bank, ("train", "fixed"), device)
    language = bank.task("T1", "train")
    question = language.questions[0]
    options = language.candidate_sets[0].options
    images, labels, records = [], [], []
    for index in range(args.batch_size):
        record = base.records[(index // 2) + (len(base.records) // 2 if index % 2 else 0)]
        images.append(Image.open(record.frame_path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
        labels.append(int(record.label))
        records.append({"path": record.frame_path, "label": int(record.label), "method": record.method})
    pixels = processor.image_processor(images=[[im] for im in images], do_image_splitting=False,
        size={"longest_edge": 512}, max_image_size={"longest_edge": 512}, return_tensors="pt")["pixel_values"][:, 0].to(device)
    qids = torch.tensor([question.text_id] * args.batch_size, device=device)
    cids = torch.tensor([[entry.text_id for entry in options]] * args.batch_size, device=device)
    mask = torch.ones_like(cids, dtype=torch.bool)
    questions, candidates = lookup.lookup(qids, cids, mask)
    # Check our pre-expanded text exactly matches the official processor.
    prompt_text = f"Question: {question.text}\nCandidate answer: {options[0].text}\nEvaluate whether this candidate correctly answers the question about the image."
    chat = processor.apply_chat_template([{"role":"user", "content":[{"type":"image"},{"type":"text","text":prompt_text}]}], add_generation_prompt=True, tokenize=False)
    official = processor(images=images[0], text=chat, do_image_splitting=False,
        size={"longest_edge":512}, max_image_size={"longest_edge":512}, return_tensors="pt")
    assert model.prompts.encode_pair(question.text, options[0].text) == official["input_ids"][0].tolist()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    for step in range(args.steps):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.float16):
            logits = model(pixels, questions, candidates, mask)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor(labels, device=device))
        assert torch.isfinite(loss), "nonfinite smoke loss"
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norms = {name:float(torch.linalg.vector_norm(p.grad)) for name,p in model.named_parameters() if p.grad is not None}
        assert all(np.isfinite(value) for value in norms.values()), "nonfinite smoke gradient"
        assert any(name.startswith("backbone.text_model") and value > 0 for name,value in norms.items())
        assert any(name.startswith("scorer") and value > 0 for name,value in norms.items())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        history.append({"step":step,"loss":float(loss),"seconds":time.monotonic()-started,
                        "peak_bytes":torch.cuda.max_memory_allocated()})
        print(json.dumps(history[-1]), flush=True)
    model.eval()
    with torch.no_grad():
        logits = model(pixels, questions, candidates, mask)
        swapped = model(pixels, questions, candidates.flip(1), mask.flip(1)).flip(1)
        error = float((logits.softmax(1)-swapped.softmax(1)).abs().max())
        assert error <= 1e-6, f"permutation smoke error {error}"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    result = {"smoke":True,"reportable":False,"visible_devices":os.environ["CUDA_VISIBLE_DEVICES"],
              "used_logical_device":0,"backbone_class":type(model.backbone).__name__,
              "pretrained_loading_info":model.pretrained_loading_info,
              "parameters":sum(p.numel() for p in model.parameters()), "trainable_parameters":trainable,
              "image_tokens":model.backbone.image_seq_len,"native_image_shape":list(pixels.shape),
              "official_tokenization_matches":True,"permutation_max_probability_error":error,
              "has_lm_head":any('lm_head' in name for name,_ in model.named_modules()),
              "history":history,"source_training_records":records}
    out = PHASE / "results/progress/backbone_smoke.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
