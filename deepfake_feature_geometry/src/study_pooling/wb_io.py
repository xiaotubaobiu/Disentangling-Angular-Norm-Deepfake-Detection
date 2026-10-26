# src/study_pooling/wb_io.py
"""Read signed head (w_fake-w_real, b_fake-b_real) from a checkpoint."""
from __future__ import annotations
from study.q4_extract import build_model_from_checkpoint

def signed_wb(ckpt_info, device="cpu"):
    model, _cfg = build_model_from_checkpoint(ckpt_info, device=device)
    w = model.classifier.weight.detach().cpu().numpy()   # (2,D)
    b = model.classifier.bias.detach().cpu().numpy()     # (2,)
    return (w[1]-w[0]).astype("float64"), float(b[1]-b[0])
