"""Paired seed-42 FF++ prompt tuning, with atomic resumable checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
import yaml

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(ROOT / 'src')]
from antifake_model import AntifakeScorer
from upstream_loader import load_backbone
from experiment_data import (validate_gpu, load_records, ImageRecords, TrainingDataset,
    ExhaustiveSuite, language_bank, batch_texts, atomic_json, sha256)


def atomic_save(path, payload):
    temp = path.with_suffix('.tmp')
    torch.save(payload, temp)
    temp.replace(path)


def move(batch, device):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=('A1', 'A2'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    validate_gpu(visible)
    if len(visible.split(',')) != 1:
        raise ValueError('each training job uses exactly one visible GPU')
    cfg_path = PHASE / 'configs/stage1.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    protocol_path = PHASE / 'results/protocol/manifest.json'
    protocol = json.loads(protocol_path.read_text())
    for path, expected in protocol['files'].items():
        if sha256(path) != expected:
            raise ValueError(f'frozen protocol changed: {path}')
    mode = {'A1': 'lm', 'A2': 'scalar'}[args.arm]
    seed = cfg['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    folder = PHASE / 'results' / ('smoke_training' if args.smoke else 'training') / args.arm
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        raise RuntimeError('completed run already exists; do not overwrite')
    fingerprint = {'protocol_sha256': sha256(protocol_path), 'config_sha256': sha256(cfg_path),
                   'assets_sha256': sha256(PHASE / 'results/assets/manifest.json'),
                   'source_sha256': {str(p.relative_to(PHASE)): sha256(p)
                       for p in [Path(__file__), *(PHASE / 'src' / name for name in
                          ('antifake_model.py', 'upstream_loader.py', 'experiment_data.py'))]},
                   'seed': seed, 'arm': args.arm, 'mode': mode, 'smoke': args.smoke,
                   'reportable': not args.smoke, 'visible_devices': visible}
    atomic_json(folder / 'run.json', fingerprint)
    print('loading backbone', flush=True)
    model = AntifakeScorer(load_backbone(PHASE), mode=mode,
        gradient_checkpointing=cfg['train']['gradient_checkpointing']).cuda()
    groups = [{'params': [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('score_head')],
               'lr': cfg['train']['prompt_lr']}]
    if mode == 'scalar':
        groups.append({'params': model.score_head.parameters(), 'lr': cfg['train']['head_lr']})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg['train']['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', init_scale=128.)
    bank = language_bank()
    prep = cfg['preprocess']
    train_base = ImageRecords(load_records('train'), prep['mean'], prep['std'])
    val_base = ImageRecords(load_records('val'), prep['mean'], prep['std'])
    val = ExhaustiveSuite(val_base, bank, split='train')
    if args.smoke:
        val = Subset(val, list(range(min(12, len(val)))))
    workers = 0 if args.smoke else cfg['train']['num_workers']
    common = {'batch_size': cfg['train']['microbatch'], 'num_workers': workers, 'pin_memory': True}
    if workers:
        common.update(multiprocessing_context='spawn', persistent_workers=False, prefetch_factor=2)
    val_loader = DataLoader(val, shuffle=False, **common)
    start_epoch, best, best_epoch, history = 0, float('inf'), 0, []
    if args.resume:
        resume = torch.load(folder / 'last.pth', map_location='cpu', weights_only=False)
        for key in ('protocol_sha256', 'config_sha256', 'assets_sha256', 'source_sha256', 'arm', 'smoke'):
            if resume['fingerprint'][key] != fingerprint[key]:
                raise ValueError(f'resume fingerprint mismatch: {key}')
        model.load_trainable(resume['state'])
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        start_epoch, best, best_epoch, history = resume['epoch'], resume['best'], resume['best_epoch'], resume['history']
    else:
        if (folder / 'last.pth').exists():
            raise RuntimeError('partial run exists; specify --resume')
        atomic_save(folder / 'initial.pth', {'state': model.trainable_state_dict(), 'fingerprint': fingerprint})
    total_epochs = 1 if args.smoke else cfg['train']['epochs']
    count = 24 if args.smoke else cfg['train']['examples_per_epoch']
    effective = cfg['train']['effective_batch']
    micro = cfg['train']['microbatch']
    accumulation = effective // micro
    if effective % micro:
        raise ValueError('microbatch must divide effective batch')
    tick = time.monotonic()
    for epoch in range(start_epoch + 1, total_epochs + 1):
        dataset = TrainingDataset(train_base, bank, seed=seed, epoch=epoch, count=count)
        loader = DataLoader(dataset, shuffle=False, **common)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_total, examples, skipped = 0., 0, 0
        for step, batch in enumerate(loader):
            batch = move(batch, 'cuda')
            questions, candidates = batch_texts(batch, bank)
            loss = model.training_loss(batch['image'], questions, candidates, batch['candidate_mask'], batch['target_index'])
            if not torch.isfinite(loss):
                raise ValueError(f'nonfinite train loss epoch={epoch} step={step}')
            bsize = len(questions)
            group_start = (step // accumulation) * effective
            group_count = min(effective, count - group_start)
            scaler.scale(loss * bsize / group_count).backward()
            loss_total += float(loss.detach()) * bsize
            examples += bsize
            boundary = (step + 1) % accumulation == 0 or step + 1 == len(loader)
            if boundary:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += int(scaler.get_scale() < scale_before)
                optimizer.zero_grad(set_to_none=True)
            if step % 120 == 0:
                print(json.dumps({'epoch': epoch, 'step': step, 'steps': len(loader),
                    'train_loss': loss_total / examples, 'elapsed_s': time.monotonic() - tick}), flush=True)
        model.eval()
        val_loss, correct, val_count = 0., 0, 0
        with torch.no_grad():
            for batch in val_loader:
                batch = move(batch, 'cuda')
                questions, candidates = batch_texts(batch, bank)
                logits = model(batch['image'], questions, candidates, batch['candidate_mask'])
                val_loss += float(F.cross_entropy(logits, batch['target_index'], reduction='sum'))
                correct += int((logits.argmax(-1) == batch['target_index']).sum())
                val_count += len(questions)
        value = val_loss / val_count
        if not math.isfinite(value):
            raise ValueError('nonfinite validation')
        row = {'epoch': epoch, 'examples': examples, 'train_loss': loss_total / examples,
               'val_candidate_ce': value, 'val_accuracy': correct / val_count, 'val_count': val_count,
               'skipped_updates': skipped, 'elapsed_s': time.monotonic() - tick,
               'peak_cuda_bytes': torch.cuda.max_memory_allocated()}
        history.append(row)
        state = model.trainable_state_dict()
        if args.smoke:
            initial = torch.load(folder / 'initial.pth', map_location='cpu', weights_only=True)['state']
            prompt_change = sum(float((state[n] - initial[n]).abs().sum())
                                for n in state if 'train_' in n)
            if prompt_change == 0 or skipped == math.ceil(count / effective):
                raise ValueError('smoke did not make a real optimizer update')
            row['prompt_absolute_change'] = prompt_change
        if value < best:
            best, best_epoch = value, epoch
            atomic_save(folder / 'best.pth', {'state': state, 'epoch': epoch, 'fingerprint': fingerprint})
        atomic_save(folder / 'last.pth', {'state': state, 'epoch': epoch, 'fingerprint': fingerprint,
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
            'history': history, 'best': best, 'best_epoch': best_epoch})
        atomic_json(folder / 'history.json', history)
        print(json.dumps(row), flush=True)
    for path, expected in fingerprint['source_sha256'].items():
        if sha256(PHASE / path) != expected:
            raise ValueError(f'training source changed while running: {path}')
    atomic_json(folder / 'complete.json', {**fingerprint, 'completed': True,
        'epochs': total_epochs, 'best_epoch': best_epoch, 'best_candidate_ce': best,
        'best_checkpoint_sha256': sha256(folder / 'best.pth'), 'history': history})
    print('TRAINING COMPLETE', args.arm, flush=True)


if __name__ == '__main__':
    main()
