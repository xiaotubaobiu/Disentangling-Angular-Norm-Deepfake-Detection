"""A0/A1/A2 evaluation with frozen splits and the audited semantic metrics."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader
import yaml

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(ROOT / 'src')]
from antifake_model import AntifakeScorer
from upstream_loader import load_backbone
from experiment_data import (validate_gpu, load_records, ImageRecords, language_bank,
    atomic_json, sha256, legacy_evaluation, method_family)
from evaluation_adapter import MetricAdapter, TextLookup, condition_dataset, matched_pool
from evaluation_cache import build_cache


class ProgressLoader:
    def __init__(self, loader, condition, domain):
        self.loader, self.condition, self.domain = loader, condition, domain

    def __iter__(self):
        tick = time.monotonic()
        for step, batch in enumerate(self.loader):
            if step % 200 == 0:
                print(json.dumps({'event': 'inference_progress', 'domain': self.domain,
                    'condition': self.condition, 'batch': step, 'batches': len(self.loader),
                    'seconds': time.monotonic() - tick}), flush=True)
            yield batch


def representative_records(records):
    selected, seen = [], set()
    for record in records:
        family = method_family(record.method, record.label)
        if family not in seen:
            selected.append(record)
            seen.add(family)
    if len(selected) != 3:
        raise ValueError('smoke requires three families')
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=('A0', 'A1', 'A2'))
    parser.add_argument('--domain', required=True, choices=('ffpp', 'cdf'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--conditions', nargs='+')
    args = parser.parse_args()
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    validate_gpu(visible)
    if len(visible.split(',')) != 1:
        raise ValueError('one GPU per evaluation job')
    cfg = yaml.safe_load((PHASE / 'configs/stage1.yaml').read_text())
    protocol_path = PHASE / 'results/protocol/manifest.json'
    protocol = json.loads(protocol_path.read_text())
    for path, expected in protocol['files'].items():
        if sha256(path) != expected:
            raise ValueError(f'protocol changed: {path}')
    mode = 'scalar' if args.arm == 'A2' else 'lm'
    checkpoint = None
    if args.arm != 'A0':
        train_root = PHASE / 'results' / ('smoke_training' if args.smoke else 'training') / args.arm
        completion = json.loads((train_root / 'complete.json').read_text())
        checkpoint = train_root / 'best.pth'
        if (completion.get('completed') is not True or completion.get('smoke') != args.smoke
                or completion['best_checkpoint_sha256'] != sha256(checkpoint)):
            raise ValueError('training incomplete/nonreportable/changed checkpoint')
        checkpoint_sha = sha256(checkpoint)
    else:
        checkpoint_sha = sha256(PHASE / 'results/assets/COCO_150k_SD3_SD2IP.pth')
    source_files = [Path(__file__), PHASE / 'configs/matched_controls.yaml', *(PHASE / 'src' / n for n in
        ('evaluation_adapter.py', 'evaluation_cache.py', 'antifake_model.py', 'upstream_loader.py', 'experiment_data.py')),
        ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py',
        *(ROOT / 'src/evaluation' / n for n in ('semantic_eval.py', 'semantic_controls.py', 'smolvlm_diagnostics.py'))]
    fingerprint = {'arm': args.arm, 'domain': args.domain, 'seed': 42, 'smoke': args.smoke,
        'reportable': not args.smoke, 'protocol_sha256': sha256(protocol_path),
        'checkpoint_sha256': checkpoint_sha, 'assets_sha256': sha256(PHASE / 'results/assets/manifest.json'),
        'sources': {str(p): sha256(p) for p in source_files},
        'probability_definition': 'softmax(summed answer log likelihood) for LM; softmax(shared scalar) for A2'}
    folder = PHASE / 'results' / ('smoke_evaluation' if args.smoke else 'evaluation') / args.arm
    folder.mkdir(parents=True, exist_ok=True)
    run_path = folder / f'{args.domain}_run.json'
    if run_path.exists() and json.loads(run_path.read_text()) != fingerprint:
        raise ValueError('existing evaluation has different provenance')
    atomic_json(run_path, fingerprint)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    scorer = AntifakeScorer(load_backbone(PHASE), mode=mode, gradient_checkpointing=False).cuda().eval()
    if checkpoint:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        scorer.load_trainable(state['state'])
    bank, legacy = language_bank(), legacy_evaluation()
    records = load_records(args.domain + '_full')
    if args.smoke:
        records = representative_records(records)
    prep = cfg['preprocess']
    images = ImageRecords(records, prep['mean'], prep['std'])
    cache_root = PHASE / 'results' / ('smoke_visual_cache' if args.smoke else 'visual_cache') / args.domain
    cache_identity = {'assets_sha256': fingerprint['assets_sha256'], 'preprocessing': prep,
                      'source': sha256(PHASE / 'src/antifake_model.py'),
                      'torch_version': torch.__version__, 'protocol_sha256': fingerprint['protocol_sha256']}
    cache = build_cache(scorer, images, cache_root, identity=cache_identity, device='cuda', batch_size=16,
                        workers=0 if args.smoke else 4)
    conditions = args.conditions or cfg['evaluation']['conditions']
    if any(c not in cfg['evaluation']['conditions'] for c in conditions):
        raise ValueError('unknown requested condition')
    pool = None
    for condition in conditions:
        done = folder / condition / f'{args.domain}_completion.json'
        if done.exists():
            previous = json.loads(done.read_text())
            if previous['fingerprint'] != fingerprint:
                raise ValueError('condition provenance mismatch')
            for rel, expected in previous['files'].items():
                if sha256(folder / rel) != expected:
                    raise ValueError('completed condition artifact changed')
            print('verified existing condition', condition, flush=True)
            continue
        subset = records if args.smoke or condition in ('official_binary', 'fixed_h1') else load_records(args.domain + '_semantic')
        metadata = legacy.CachedMetadataDataset(ImageRecords(subset, prep['mean'], prep['std']))
        lookup = TextLookup(bank)
        if condition.startswith('meaningless_matched_'):
            if pool is None:
                pool = matched_pool(bank, scorer)
                atomic_json(folder / f'{args.domain}_matched_pool_manifest.json', {
                    'match': 'exact_llm_and_qformer_lengths',
                    'texts': {str(e.text_id): e.text for e in pool.entries},
                    'pools': {f'{kind}:{src}': list(ids) for (kind, src), ids in pool.pools.items()}})
            lookup = TextLookup(pool.bank)
        dataset = condition_dataset(metadata, bank, condition, legacy, pool=pool)
        model = MetricAdapter(scorer, lookup).eval()
        loader = ProgressLoader(DataLoader(dataset, batch_size=cfg['evaluation']['microbatch'], num_workers=0), condition, args.domain)
        print(json.dumps({'event': 'condition_start', 'arm': args.arm, 'domain': args.domain,
                          'condition': condition, 'source_requests': len(dataset)}), flush=True)
        tick = time.monotonic()
        with torch.no_grad():
            rows = legacy.evaluate_condition(model, loader, lookup, torch.device('cuda'), seed=42,
                domain=args.domain, condition=condition, feature_cache=cache)
        if condition != 'permutation' and len(rows) != len(dataset):
            raise ValueError('evaluation missing requests')
        legacy.write_outputs(rows, output_dir=folder, condition=condition, domain=args.domain)
        outputs = sorted((folder / condition).glob(f'{args.domain}_*'))
        info = {'fingerprint': fingerprint, 'completed': True, 'source_requests': len(dataset),
                'physical_frames': len(subset), 'physical_videos': len({(r.method, r.video_id) for r in subset}),
                'seconds': time.monotonic() - tick, 'files': {str(p.relative_to(folder)): sha256(p) for p in outputs}}
        atomic_json(done, info)
        print(json.dumps({'event': 'condition_complete', 'condition': condition, 'seconds': info['seconds']}), flush=True)
    for path, expected in fingerprint['sources'].items():
        if sha256(path) != expected:
            raise ValueError('evaluation source changed while running')
    print('EVALUATION COMPLETE', args.arm, args.domain, flush=True)


if __name__ == '__main__':
    main()
