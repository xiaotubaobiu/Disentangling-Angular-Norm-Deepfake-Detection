"""Four-GPU training of the pretrained, non-generative E3 candidate scorer."""
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
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

ROOT = Path(__file__).resolve().parents[3]
PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from config_io import load_config
from data.builders import build_train_dataset, build_val_dataset, _dataloader_kwargs
from data.semantic_dataset import SemanticDataset, SemanticTemplateSampler, TASKS
from data.semantic_language import load_language_bank
from data.smolvlm_data import NativeImageDataset, ExactShardSampler
from engine.semantic_trainer import TrainableEMA
from engine.smolvlm_trainer import train_epoch, validate_epoch
from models.smolvlm_candidate import SmolVLMCandidateScorer, CandidateTextLookup


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=PHASE / 'configs/candidate.yaml')
    parser.add_argument('--seed', type=int, required=True, choices=(42, 7, 2048))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--max-train-batches', type=int)
    parser.add_argument('--max-val-batches', type=int)
    args = parser.parse_args(argv)
    limits = (args.max_train_batches, args.max_val_batches)
    if not args.smoke and any(n is not None for n in limits):
        parser.error('batch limits require --smoke')
    if any(n is not None and n < 1 for n in limits):
        parser.error('batch limits must be positive')
    return args


def validate_runtime(visible_devices, world_size):
    if visible_devices != '0,1,2,3':
        raise ValueError('E3 permits only physical devices 0,1,2,3')
    if world_size != 4:
        raise ValueError('E3 training requires four ranks')


def verify_protocol(manifest, root=ROOT):
    for relative, expected in manifest['files'].items():
        if sha256(root / relative) != expected:
            raise ValueError(f'frozen protocol file changed: {relative}')


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def atomic_save(path, value):
    temp = path.with_suffix('.pth.tmp')
    torch.save(value, temp)
    temp.replace(path)


def physical_digest(records):
    text = '\n'.join(f'{r.sample_id}\t{r.method}' for r in records)
    return hashlib.sha256(text.encode()).hexdigest()


def build_loaders(cfg, bank, rank, world):
    seed = cfg['train']['seed']
    train_base, val_base = build_train_dataset(cfg), build_val_dataset(cfg)
    train_frames = {r.frame_path for r in train_base.records}
    val_frames = {r.frame_path for r in val_base.records}
    train_videos = {(r.method, r.video_id) for r in train_base.records}
    val_videos = {(r.method, r.video_id) for r in val_base.records}
    if train_frames & val_frames or train_videos & val_videos:
        raise ValueError('source training/validation physical overlap')
    audit = {'train_records': len(train_base), 'val_records': len(val_base),
             'train_unique_frames': len(train_frames), 'val_unique_frames': len(val_frames),
             'train_record_sha256': physical_digest(train_base.records),
             'val_record_sha256': physical_digest(val_base.records),
             'train_val_frame_overlap': 0, 'train_val_video_overlap': 0,
             'data_definition': 'DF40 ff source split, same 16 methods as H1/E2'}
    native = ROOT / cfg['model']['pretrained']
    train_base = NativeImageDataset(train_base.records, native, augment=True, seed=seed + rank * 100003)
    val_base = NativeImageDataset(val_base.records, native, augment=False, seed=seed)
    train = SemanticDataset(train_base, SemanticTemplateSampler(bank, seed), mode='train', split='train')
    val = SemanticDataset(val_base, SemanticTemplateSampler(bank, seed), mode='suite', split='train')
    train_sampler = DistributedSampler(train, num_replicas=world, rank=rank, shuffle=True, seed=seed)
    val_sampler = ExactShardSampler(val, rank=rank, world_size=world)
    kwargs = _dataloader_kwargs(cfg['train']['num_workers'], cfg['train']['prefetch_factor'])
    train_loader = DataLoader(train, batch_size=cfg['train']['per_gpu_batch'], sampler=train_sampler,
                              drop_last=True, generator=torch.Generator().manual_seed(seed + rank), **kwargs)
    val_loader = DataLoader(val, batch_size=cfg['train']['per_gpu_batch'], sampler=val_sampler,
                            drop_last=False, generator=torch.Generator().manual_seed(seed + rank), **kwargs)
    audit['validation_semantic_examples'] = len(val)
    return train_loader, val_loader, audit


def main(argv=None):
    args = parse_args(argv)
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    validate_runtime(visible, int(os.environ.get('WORLD_SIZE', 1)))
    cfg = load_config(args.config)
    if tuple(cfg['language']['train_tasks']) != TASKS:
        raise ValueError('E3 stage 1 requires all five tasks')
    cfg['train']['seed'] = args.seed
    protocol_path = PHASE / 'configs/protocol_lock.json'
    if not args.smoke:
        protocol = json.loads(protocol_path.read_text())
        verify_protocol(protocol)
    rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    device = torch.device(f'cuda:{local_rank}')
    out = ROOT / cfg['train']['output_dir']
    if args.smoke:
        out /= 'smoke'
    out /= f'seed_{args.seed}'
    try:
        if (out / 'best_model.pth').exists() or (out / 'history.jsonl').exists():
            raise FileExistsError(f'preserving existing training run: {out}')
        out.mkdir(parents=True, exist_ok=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        model, _ = SmolVLMCandidateScorer.from_pretrained(
            ROOT / cfg['model']['pretrained'], score_batch_size=cfg['model']['score_batch_size'],
            gradient_checkpointing=cfg['model']['gradient_checkpointing'])
        model.to(device)
        bank = load_language_bank(ROOT / cfg['language']['manifest'])
        lookup = CandidateTextLookup.build(model, bank, ('train', 'fixed'), device)
        wrapped = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
        optimizer = torch.optim.AdamW([
            {'params': model.backbone.text_model.parameters(), 'lr': cfg['train']['language_lr']},
            {'params': model.scorer.parameters(), 'lr': cfg['train']['head_lr']}],
            weight_decay=cfg['train']['weight_decay'])
        total_epochs = cfg['train']['epochs']
        warmup = cfg['train']['warmup_epochs']
        def schedule(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            return .5 * (1 + math.cos(math.pi * (epoch - warmup) / max(1, total_epochs - warmup)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
        scaler = torch.amp.GradScaler('cuda')
        ema = TrainableEMA(model, cfg['train']['ema_decay'])
        train_loader, val_loader, audit = build_loaders(cfg, bank, rank, 4)
        provenance = {'schema_version': 1, 'experiment': cfg['experiment_name'], 'seed': args.seed,
                      'smoke': args.smoke, 'reportable': not args.smoke,
                      'config_sha256': sha256(args.config),
                      'language_manifest_sha256': sha256(ROOT / cfg['language']['manifest']),
                      'pretrained_manifest_sha256': sha256(PHASE / 'results/pretrained/manifest.json'),
                      'protocol_sha256': None if args.smoke else sha256(protocol_path),
                      'visible_devices': visible, 'world_size': 4, 'source_data': audit,
                      'pretrained_loading_info': model.pretrained_loading_info,
                      'parameters': sum(p.numel() for p in model.parameters()),
                      'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                      'effective_batch': cfg['train']['per_gpu_batch'] * 4 * cfg['train']['accumulation_steps'],
                      'has_lm_head': False, 'generation_used': False}
        if rank == 0:
            atomic_json(out / 'provenance.json', provenance)
            print(json.dumps({'event': 'start', **provenance}), flush=True)
        max_train = (args.max_train_batches or 8) if args.smoke else None
        max_val = (args.max_val_batches or 4) if args.smoke else None
        best_loss, best_epoch, stale = math.inf, None, 0
        started = time.monotonic()
        for epoch in range(1 if args.smoke else total_epochs):
            torch.cuda.reset_peak_memory_stats()
            train_loader.dataset.set_epoch(epoch)
            train_loader.sampler.set_epoch(epoch)
            trained = train_epoch(wrapped, train_loader, lookup, optimizer, scaler, device,
                                  accumulation=cfg['train']['accumulation_steps'], amp=True,
                                  ema=ema, max_batches=max_train, log_every=100)
            if trained['skipped_steps'] > max(10, .01 * trained['optimizer_steps']):
                raise RuntimeError('excessive AMP skipped optimizer steps')
            ema.apply(model)
            validated = validate_epoch(model, val_loader, lookup, device, max_batches=max_val)
            if not args.smoke and validated['count'] != len(val_loader.dataset):
                raise RuntimeError('validation omitted or duplicated source examples')
            diagnostics = {'rank': rank, 'train': trained, 'validation': validated,
                           'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
            rank_rows = [None] * 4
            dist.all_gather_object(rank_rows, diagnostics)
            improved = validated['loss'] < best_loss
            if improved:
                best_loss, best_epoch, stale = validated['loss'], epoch + 1, 0
                if rank == 0:
                    payload = {**provenance, 'epoch': epoch + 1, 'val_seen_ce': best_loss,
                               'val_seen_accuracy': validated['accuracy'], 'weights_kind': 'ema',
                               'model_state_dict': {name: p.detach().cpu().clone() for name, p in model.state_dict().items()}}
                    atomic_save(out / 'best_model.pth', payload)
            else:
                stale += 1
            ema.restore(model)
            scheduler.step()
            if rank == 0:
                row = {'epoch': epoch + 1, 'train_ce': trained['loss'], 'val_seen_ce': validated['loss'],
                       'val_seen_accuracy': validated['accuracy'], 'best_epoch': best_epoch,
                       'rank_diagnostics': rank_rows, 'elapsed_seconds': time.monotonic() - started}
                with (out / 'history.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
                if not args.smoke:
                    atomic_save(out / 'last_training_state.pth', {
                        **provenance, 'epoch': epoch + 1, 'best_epoch': best_epoch, 'best_val_seen_ce': best_loss,
                        'stale': stale, 'model_state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'ema': ema.state_dict()})
            dist.barrier()
            if stale >= cfg['train']['patience']:
                break
        if not args.smoke:
            verify_protocol(protocol)
        if rank == 0:
            final = {**provenance, 'completed': True, 'completed_epochs': epoch + 1,
                     'best_epoch': best_epoch, 'best_val_seen_ce': best_loss,
                     'best_checkpoint_sha256': sha256(out / 'best_model.pth'),
                     'seconds': time.monotonic() - started}
            atomic_json(out / 'completion.json', final)
            print(json.dumps({'event': 'complete', **final}), flush=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
