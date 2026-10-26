"""E3 native-token evaluation with the frozen E2 request/metric definitions."""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[3]
PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from config_io import load_config
from data.builders import build_eval_dataset, build_val_dataset
from data.datasets import FrameClassificationDataset
from data.semantic_dataset import SemanticDataset, SemanticTemplateSampler, TASKS
from data.semantic_language import load_language_bank
from data.smolvlm_data import NativeImageDataset, ExactShardSampler
from evaluation.semantic_controls import select_global_audit_indices
from evaluation.semantic_eval import deduplicate_distributed_rows
from evaluation.smolvlm_feature_cache import build_cache
from evaluation.smolvlm_diagnostics import (
    extended_language_bank, NegationDataset, MatchedTextPool, MatchedControlDataset,
)
from models.smolvlm_candidate import SmolVLMCandidateScorer, CandidateTextLookup

ADDED_CONDITIONS = ('negation', 'composition', 'meaningless_matched_candidates', 'meaningless_matched_questions')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_e2():
    path = ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py'
    spec = importlib.util.spec_from_file_location('e3_reused_e2_evaluation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_checkpoint(checkpoint, completion, expected, *, seed, checkpoint_sha):
    for entry in (checkpoint, completion):
        if entry.get('smoke') is not False or entry.get('reportable') is not True:
            raise ValueError('non-reportable checkpoint or completion')
        if entry.get('seed') != seed or entry.get('world_size') != 4 or entry.get('visible_devices') != '0,1,2,3':
            raise ValueError('checkpoint seed/runtime disagrees with formal protocol')
        for key, value in expected.items():
            if entry.get(key) != value:
                raise ValueError(f'checkpoint/completion fingerprint mismatch: {key}')
    if completion.get('completed') is not True or completion.get('best_checkpoint_sha256') != checkpoint_sha:
        raise ValueError('missing complete training or best-checkpoint fingerprint')
    if checkpoint.get('weights_kind') != 'ema' or checkpoint.get('epoch') != completion.get('best_epoch'):
        raise ValueError('checkpoint is not the selected source-validation EMA model')


def condition_dataset(base, bank, *, seed, condition, e2, pool=None):
    if condition == 'negation':
        return NegationDataset(base, bank, seed=seed)
    if condition == 'composition':
        return SemanticDataset(base, SemanticTemplateSampler(bank, seed), mode='suite',
                               tasks=('T1', 'T5'), split='composition')
    if condition.startswith('meaningless_matched_'):
        meaningful = condition_dataset(base, bank, seed=seed, condition='heldout', e2=e2)
        return MatchedControlDataset(meaningful, pool, kind=condition.removeprefix('meaningless_matched_'))
    spec = e2.CONDITION_SPECS[condition]
    if spec.fixed_h1:
        return e2.FixedH1Dataset(base, bank)
    if condition == 'permutation':
        selected = select_global_audit_indices([str(r.sample_id) for r in base.records], limit=4096)
        return SemanticDataset(Subset(base, selected), SemanticTemplateSampler(bank, seed),
                               mode='suite', tasks=TASKS, split=spec.language_split)
    if condition == 'counterfactual':
        return e2.CounterfactualEvaluationDataset(base, bank, tasks=TASKS,
                                                  split=spec.language_split, seed=seed)
    suite = SemanticDataset(base, SemanticTemplateSampler(bank, seed), mode='suite',
                             tasks=TASKS, split=spec.language_split)
    paired = e2.PairedT4EvaluationDataset(suite)
    if condition.startswith('meaningless_'):
        return e2.PairedMeaninglessEvaluationDataset(paired, bank, control=condition)
    return paired


def verify_lock(path):
    manifest = json.loads(path.read_text())
    for relative, expected in manifest['files'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'locked evaluation/training file changed: {relative}')
    return sha256(path)


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def collect_rows(rows, *, output, condition, domain, rank, world, expected_count):
    """Disk transport avoids replicating millions of Python rows on every GPU."""
    directory = output / '_rank_transport' / condition / domain
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'rank_{rank}.pkl'
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        pickle.dump(rows, stream, protocol=5)
    temp.replace(path)
    del rows
    _barrier()
    merged = None
    if rank == 0:
        merged = []
        for peer in range(world):
            # Only this run's locally written rank transport is deserialized.
            with (directory / f'rank_{peer}.pkl').open('rb') as stream:
                merged.extend(pickle.load(stream))
        if len(merged) != expected_count:
            raise ValueError(f'incomplete evaluation coverage: {len(merged)} != {expected_count}')
        merged = deduplicate_distributed_rows(merged)
    return merged


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True, choices=(42, 7, 2048))
    parser.add_argument('--cpu-smoke', action='store_true', help='source-validation records only; non-reportable')
    parser.add_argument('--cache-only', action='store_true')
    parser.add_argument('--conditions', nargs='+', choices=tuple(load_e2().CONDITION_SPECS) + ADDED_CONDITIONS, default=None)
    parser.add_argument('--domains', nargs='+', choices=('ffpp', 'cdf'), default=['ffpp', 'cdf'])
    args = parser.parse_args(argv)
    cfg_path = PHASE / 'configs/candidate.yaml'
    cfg = load_config(cfg_path)
    cfg['train']['seed'] = args.seed
    if args.cpu_smoke:
        device, rank, world = torch.device('cpu'), 0, 1
        args.domains = ['ffpp']
        output = PHASE / f'results/evaluation_smoke/seed_{args.seed}'
        training = PHASE / f'results/training/smoke/seed_{args.seed}'
        cache_root = PHASE / 'results/native_cache_smoke'
        eval_lock_sha = None
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3' or int(os.environ.get('WORLD_SIZE', 1)) != 4:
            raise ValueError('formal evaluation requires exactly physical GPUs 0,1,2,3')
        eval_lock_sha = verify_lock(PHASE / 'configs/evaluation_lock.json')
        verify_lock(PHASE / 'configs/protocol_lock.json')
        rank, local_rank = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
        world = 4
        torch.cuda.set_device(local_rank)
        dist.init_process_group('nccl', timeout=timedelta(seconds=3600))
        device = torch.device(f'cuda:{local_rank}')
        output = PHASE / f'results/evaluation/seed_{args.seed}'
        training = PHASE / f'results/training/seed_{args.seed}'
        cache_root = PHASE / 'results/native_cache'
    try:
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        e2 = load_e2()
        model, _ = SmolVLMCandidateScorer.from_pretrained(ROOT / cfg['model']['pretrained'],
                                                         score_batch_size=cfg['model']['score_batch_size'])
        checkpoint_path = training / 'best_model.pth'
        checkpoint_sha = None
        if not args.cache_only:
            checkpoint_sha = sha256(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True, mmap=True)
            completion = json.loads((training / 'completion.json').read_text())
            expected = {
                'config_sha256': sha256(cfg_path),
                'protocol_sha256': sha256(PHASE / 'configs/protocol_lock.json'),
                'language_manifest_sha256': sha256(ROOT / cfg['language']['manifest']),
                'pretrained_manifest_sha256': sha256(PHASE / 'results/pretrained/manifest.json')}
            if not args.cpu_smoke:
                validate_checkpoint(checkpoint, completion, expected, seed=args.seed, checkpoint_sha=checkpoint_sha)
                series = json.loads((PHASE / 'results/training_logs/series_status.json').read_text())
                matches = [r for r in series['runs'] if r['seed'] == args.seed]
                if len(matches) != 1 or matches[0]['exit_code'] != 0:
                    raise ValueError('formal evaluation requires a verified successful training launcher exit')
            elif checkpoint.get('smoke') is not True:
                raise ValueError('CPU smoke must load an explicitly non-reportable source smoke model')
            for name, value in model.state_dict().items():
                if name.startswith(('backbone.vision_model.', 'backbone.connector.')):
                    if not torch.equal(value, checkpoint['model_state_dict'][name]):
                        raise ValueError(f'frozen pretrained weight changed: {name}')
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            del checkpoint
        model.to(device).eval()
        bank = load_language_bank(ROOT / cfg['language']['manifest'])
        diagnostic_bank = extended_language_bank(bank, ROOT / cfg['evaluation']['diagnostics_manifest'])
        matched_pool = None
        provenance = {'seed': args.seed, 'smoke': args.cpu_smoke, 'reportable': not args.cpu_smoke,
                      'checkpoint_sha256': checkpoint_sha, 'checkpoint_path': str(checkpoint_path),
                      'evaluation_lock_sha256': eval_lock_sha, 'config_sha256': sha256(cfg_path),
                      'protocol_sha256': sha256(PHASE / 'configs/protocol_lock.json'),
                      'world_size': world, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                      'precision': 'float32', 'input_scope': 'source_validation_only' if args.cpu_smoke else 'ffpp_cdf',
                      'native_caches': {}}
        if not args.cache_only and rank == 0:
            provenance_path = output / 'run_provenance.json'
            if provenance_path.exists():
                previous = json.loads(provenance_path.read_text())
                for key in ('checkpoint_sha256', 'evaluation_lock_sha256', 'config_sha256', 'protocol_sha256'):
                    if previous[key] != provenance[key]:
                        raise ValueError('refusing to mix evaluations from different checkpoints or protocols')
                provenance['native_caches'].update(previous.get('native_caches', {}))
            atomic_json(provenance_path, provenance)
        conditions = args.conditions or cfg['evaluation']['conditions'] + list(ADDED_CONDITIONS)
        for domain in args.domains:
            if args.cpu_smoke:
                source = build_val_dataset(cfg)
                records = [r for r in source.records if r.label == 0][:2] + [r for r in source.records if r.label == 1][:2]
            else:
                records = build_eval_dataset(cfg, domain).records
            native = NativeImageDataset(records, ROOT / cfg['model']['pretrained'], augment=False, seed=args.seed)
            cache = build_cache(model, native, cache_root, domain=domain, cfg=cfg, device=device,
                                batch_size=cfg['train']['per_gpu_batch'], num_workers=0 if args.cpu_smoke else 4)
            if args.cache_only:
                continue
            provenance['native_caches'][domain] = sha256(cache_root / domain / 'manifest.json')
            if rank == 0:
                atomic_json(output / 'run_provenance.json', provenance)
            base = e2.CachedMetadataDataset(native)
            for condition in conditions:
                complete_path = output / condition / f'{domain}_completion.json'
                if complete_path.exists():
                    saved = json.loads(complete_path.read_text())
                    if saved['checkpoint_sha256'] != checkpoint_sha or saved['evaluation_lock_sha256'] != eval_lock_sha:
                        raise ValueError('completed evaluation has different provenance')
                    for relative, expected in saved['files'].items():
                        if sha256(output / relative) != expected:
                            raise ValueError('completed evaluation artifact changed')
                    continue
                active_bank = diagnostic_bank if condition in ADDED_CONDITIONS else bank
                if condition.startswith('meaningless_matched_') and matched_pool is None:
                    matched_pool = MatchedTextPool(diagnostic_bank, model.prompts,
                                                  ROOT / cfg['evaluation']['diagnostics_manifest'])
                dataset = condition_dataset(base, active_bank, seed=args.seed, condition=condition, e2=e2, pool=matched_pool)
                loader = DataLoader(dataset, batch_size=cfg['train']['per_gpu_batch'], num_workers=0,
                                    sampler=ExactShardSampler(dataset, rank=rank, world_size=world))
                if condition.startswith('meaningless_matched_'):
                    lookup = CandidateTextLookup.build(model, matched_pool.bank, ('heldout', 'matched_control'), device)
                else:
                    splits = {'negation': ('diagnostic', 'train'), 'composition': ('diagnostic',)}.get(condition)
                    lookup = CandidateTextLookup.build(model, active_bank,
                        splits or e2.required_text_splits(condition), device)
                if rank == 0:
                    print(json.dumps({'event': 'evaluation_start', 'seed': args.seed, 'domain': domain,
                                      'condition': condition, 'source_requests': len(dataset)}), flush=True)
                started = time.monotonic()
                local_rows = e2.evaluate_condition(model, loader, lookup, device, seed=args.seed,
                    domain=domain, condition=condition, feature_cache=cache)
                if condition == 'permutation':
                    rows = local_rows
                else:
                    rows = collect_rows(local_rows, output=output, condition=condition, domain=domain,
                                        rank=rank, world=world, expected_count=len(dataset))
                if rank == 0:
                    e2.write_outputs(rows, output_dir=output, condition=condition, domain=domain)
                    artifacts = sorted((output / condition).glob(f'{domain}_*'))
                    result = {'seed': args.seed, 'domain': domain, 'condition': condition,
                              'checkpoint_sha256': checkpoint_sha, 'evaluation_lock_sha256': eval_lock_sha,
                              'smoke': args.cpu_smoke, 'reportable': not args.cpu_smoke,
                              'source_requests': len(dataset), 'rows_after_deduplication': len(rows),
                              'seconds': time.monotonic() - started,
                              'files': {str(p.relative_to(output)): sha256(p) for p in artifacts}}
                    atomic_json(complete_path, result)
                    print(json.dumps({'event': 'evaluation_complete', **result}), flush=True)
                _barrier()
                if condition != 'permutation':
                    (output / '_rank_transport' / condition / domain / f'rank_{rank}.pkl').unlink()
                del rows, local_rows
            del cache
        if not args.cpu_smoke:
            verify_lock(PHASE / 'configs/evaluation_lock.json')
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
