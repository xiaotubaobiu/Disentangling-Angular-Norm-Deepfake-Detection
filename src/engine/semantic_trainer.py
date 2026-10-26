"""CE-only training utilities for semantic Visual-Jev."""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _raw_model(model):
    return model.module if hasattr(model, "module") else model


def semantic_classification_step(model, batch, text_lookup, device):
    """Run one candidate-classification step with exact cross entropy."""
    images = batch["image"].to(device, non_blocking=True)
    question_ids = batch["question_text_id"].to(device, non_blocking=True)
    candidate_ids = batch["candidate_text_ids"].to(
        device, non_blocking=True
    )
    candidate_mask = batch["candidate_mask"].to(
        device, non_blocking=True
    )
    target = batch["target_index"].to(device, non_blocking=True)
    question, candidates = text_lookup.lookup(
        question_ids, candidate_ids, candidate_mask
    )
    logits = model(images, question, candidates, candidate_mask)
    loss = F.cross_entropy(logits, target)
    return loss, logits


def trainable_parameters(model) -> list[torch.nn.Parameter]:
    """Return only parameters participating in the semantic decision rule."""
    return [
        parameter
        for parameter in _raw_model(model).parameters()
        if parameter.requires_grad
    ]


class TrainableEMA:
    """Exponential moving average over trainable decision parameters only."""

    def __init__(self, model, decay: float = 0.999) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError("EMA decay must be between zero and one")
        self.decay = float(decay)
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in _raw_model(model).named_parameters()
            if parameter.requires_grad
        }
        if not self.shadow:
            raise ValueError("EMA requires at least one trainable parameter")
        self._backup: dict[str, torch.Tensor] | None = None

    @torch.no_grad()
    def update(self, model) -> None:
        parameters = dict(_raw_model(model).named_parameters())
        for name, shadow in self.shadow.items():
            parameter = parameters[name]
            shadow.mul_(self.decay).add_(
                parameter.detach(), alpha=1.0 - self.decay
            )

    @torch.no_grad()
    def apply(self, model) -> None:
        if self._backup is not None:
            raise RuntimeError("EMA weights are already applied")
        parameters = dict(_raw_model(model).named_parameters())
        self._backup = {
            name: parameters[name].detach().clone() for name in self.shadow
        }
        for name, shadow in self.shadow.items():
            parameters[name].copy_(shadow)

    @torch.no_grad()
    def restore(self, model) -> None:
        if self._backup is None:
            raise RuntimeError("EMA weights are not applied")
        parameters = dict(_raw_model(model).named_parameters())
        for name, backup in self._backup.items():
            parameters[name].copy_(backup)
        self._backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        if state.keys() != self.shadow.keys():
            raise ValueError("EMA state does not match trainable parameters")
        for name, value in state.items():
            self.shadow[name].copy_(value)


def _amp_enabled(device: torch.device, cfg: dict) -> bool:
    return (
        device.type == "cuda"
        and cfg.get("train", {}).get("precision", "amp") == "amp"
    )


def run_semantic_train_epoch(
    model,
    dataloader: Iterable,
    optimizer,
    scaler,
    device,
    text_lookup,
    cfg,
    ema=None,
) -> float:
    """Train one epoch using candidate cross entropy and no auxiliary loss."""
    model.train()
    raw_model = _raw_model(model)
    if hasattr(raw_model, "clip"):
        raw_model.clip.eval()
    parameters = trainable_parameters(model)
    total_loss = 0.0
    total_count = 0
    amp_enabled = _amp_enabled(device, cfg)

    for batch in dataloader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            loss, _ = semantic_classification_step(
                model, batch, text_lookup, device
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)

        batch_count = int(batch["target_index"].numel())
        total_loss += float(loss.detach().item()) * batch_count
        total_count += batch_count

    return total_loss / max(total_count, 1)


@torch.no_grad()
def run_semantic_val_epoch(model, dataloader, text_lookup, device) -> dict:
    """Return deterministic, all-rank seen-language CE and accuracy."""
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)

    for batch in dataloader:
        loss, logits = semantic_classification_step(
            model, batch, text_lookup, device
        )
        target = batch["target_index"].to(device, non_blocking=True)
        batch_count = target.numel()
        totals[0] += loss.double() * batch_count
        totals[1] += (logits.argmax(dim=-1) == target).sum().double()
        totals[2] += batch_count

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)

    count = int(totals[2].item())
    if count == 0:
        return {"loss": 0.0, "accuracy": 0.0, "count": 0}
    return {
        "loss": float((totals[0] / count).item()),
        "accuracy": float((totals[1] / count).item()),
        "count": count,
    }
