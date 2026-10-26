"""Regression tests for contrastive losses.

Bug 2: infonce_bg_face_loss masked self-pairs with
`cross_real[sample_idx, sample_idx[:B]]`, indexing with arrays of mismatched
lengths (B*K vs B), which raised IndexError for every realistic batch
(B>1, K>1). These tests lock down that the loss runs and returns a finite
scalar, and that same_frame_contrastive_loss behaves sanely.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.contrastive import (
    infonce_bg_face_loss,
    same_frame_contrastive_loss,
    image_text_contrastive_loss,
)


def _norm(x, dim):
    return F.normalize(x, dim=dim)


def test_infonce_bg_face_loss_runs_and_is_finite():
    torch.manual_seed(0)
    B, K, D = 4, 3, 8
    bg = _norm(torch.randn(B, K, D), 2)
    real = _norm(torch.randn(B, D), 1)
    fake = _norm(torch.randn(B, D), 1)
    loss = infonce_bg_face_loss(bg, real, fake, temperature=0.07)
    assert loss.ndim == 0
    assert torch.isfinite(loss), "loss must be finite (no -inf leaking through)"
    assert loss.item() > 0


def test_infonce_bg_face_loss_single_patch():
    """K=1 must also work (degenerate patch count)."""
    torch.manual_seed(1)
    B, K, D = 5, 1, 16
    bg = _norm(torch.randn(B, K, D), 2)
    real = _norm(torch.randn(B, D), 1)
    fake = _norm(torch.randn(B, D), 1)
    loss = infonce_bg_face_loss(bg, real, fake)
    assert torch.isfinite(loss)


def test_infonce_perfect_alignment_low_loss():
    """When bg aligns with its real face and anti-aligns with fake, loss is low."""
    B, K, D = 3, 2, 8
    real = _norm(torch.randn(B, D), 1)
    bg = real.unsqueeze(1).expand(B, K, D).clone()  # bg == its real face
    fake = -real  # fake anti-aligned
    loss = infonce_bg_face_loss(bg, real, fake, temperature=0.07)
    assert torch.isfinite(loss)
    assert loss.item() < 1.0


def test_same_frame_contrastive_loss_runs():
    torch.manual_seed(2)
    B, D = 6, 8
    bg = torch.randn(B, D)
    real = torch.randn(B, D)
    fake = torch.randn(B, D)
    loss = same_frame_contrastive_loss(bg, real, fake)
    assert torch.isfinite(loss) and loss.item() > 0


def test_image_text_contrastive_loss_runs():
    torch.manual_seed(3)
    B, D = 4, 8
    img = torch.randn(B, D)
    txt = torch.randn(B, D)
    labels = torch.tensor([0, 1, 0, 1])
    loss = image_text_contrastive_loss(img, txt, labels)
    assert torch.isfinite(loss)
