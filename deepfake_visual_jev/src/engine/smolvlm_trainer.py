"""Accumulated E3 candidate CE with exact distributed validation counts."""
from __future__ import annotations

from contextlib import nullcontext
import itertools
import json
import time

import torch
import torch.distributed as dist

from engine.semantic_trainer import semantic_classification_step, trainable_parameters


def _reduce(totals):
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals)
    return totals


def train_epoch(model, loader, lookup, optimizer, scaler, device, *,
                accumulation, amp, ema=None, max_batches=None, log_every=0):
    model.train()
    if accumulation < 1:
        raise ValueError('accumulation must be positive')
    count_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    if count_batches == 0:
        raise ValueError('empty training loader')
    params = trainable_parameters(model)
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    steps = skipped = 0
    started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    for index, batch in enumerate(itertools.islice(loader, count_batches)):
        group_start = index // accumulation * accumulation
        group_size = min(accumulation, count_batches - group_start)
        is_step = index + 1 == group_start + group_size
        sync = model.no_sync() if hasattr(model, 'no_sync') and not is_step else nullcontext()
        with sync:
            with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
                loss, _ = semantic_classification_step(model, batch, lookup, device)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite candidate loss at batch {index}')
            scaler.scale(loss / group_size).backward()
        count = batch['target_index'].numel()
        totals[0] += loss.detach().double() * count
        totals[1] += count
        if is_step:
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1.)
            if not scaler.is_enabled() and not torch.isfinite(norm):
                raise RuntimeError('nonfinite FP32 gradients')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            did_step = scaler.get_scale() >= old_scale
            steps += int(did_step)
            skipped += int(not did_step)
            if did_step and ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
        if log_every and (index + 1) % log_every == 0:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(json.dumps({'event': 'train_progress', 'batch': index + 1,
                                  'total_batches': count_batches, 'ce': float(totals[0] / totals[1]),
                                  'seconds': time.monotonic() - started,
                                  'optimizer_steps': steps, 'skipped_steps': skipped}), flush=True)
    _reduce(totals)
    return {'loss': float(totals[0] / totals[1]), 'count': int(totals[1]),
            'batches': count_batches, 'optimizer_steps': steps, 'skipped_steps': skipped,
            'seconds': time.monotonic() - started}


@torch.no_grad()
def validate_epoch(model, loader, lookup, device, *, max_batches=None):
    # Raw module, so unequal nonpadding shards never trigger DDP collectives.
    model = model.module if hasattr(model, 'module') else model
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    count_batches = min(len(loader), max_batches) if max_batches is not None else len(loader)
    started = time.monotonic()
    for batch in itertools.islice(loader, count_batches):
        loss, logits = semantic_classification_step(model, batch, lookup, device)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite source-validation candidate loss')
        target = batch['target_index'].to(device)
        totals[0] += loss.double() * target.numel()
        totals[1] += (logits.argmax(-1) == target).sum()
        totals[2] += target.numel()
    _reduce(totals)
    if not totals[2]:
        raise ValueError('empty validation loader')
    return {'loss': float(totals[0] / totals[2]), 'accuracy': float(totals[1] / totals[2]),
            'count': int(totals[2]), 'batches': count_batches,
            'seconds': time.monotonic() - started}
