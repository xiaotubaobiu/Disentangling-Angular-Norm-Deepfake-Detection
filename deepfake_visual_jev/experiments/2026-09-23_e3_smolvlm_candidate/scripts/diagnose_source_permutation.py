"""Source-only numerical diagnosis; never a reportable semantic result."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from config_io import load_config
from data.builders import build_val_dataset
from data.semantic_dataset import SemanticDataset, SemanticTemplateSampler, TASKS
from data.semantic_language import load_language_bank
from data.smolvlm_data import NativeImageDataset
from evaluation.semantic_controls import permutation_audit
from evaluation.smolvlm_feature_cache import open_cache
from models.smolvlm_candidate import CandidateTextLookup, SmolVLMCandidateScorer


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape-controls', action='store_true')
    parser.add_argument('--fixed-scoring', action='store_true')
    args = parser.parse_args()
    if args.shape_controls and args.fixed_scoring:
        raise ValueError('run shape isolation and final scorer verification separately')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0,1,2,3':
        raise ValueError('only physical GPUs 0-3 are authorized')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    free, _ = torch.cuda.mem_get_info(device)
    if free < 6 * 1024**3:
        raise RuntimeError('insufficient headroom alongside training')
    cfg = load_config(PHASE / 'configs/candidate.yaml')
    cfg['train']['seed'] = 42
    path = PHASE / 'results/training/seed_42/best_model.pth'
    # Load one immutable file snapshot even if the trainer atomically replaces it.
    with path.open('rb') as stream:
        checkpoint_sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        stream.seek(0)
        checkpoint = torch.load(stream, map_location='cpu', weights_only=True)
    model_class = SmolVLMCandidateScorer
    if args.fixed_scoring:
        from evaluation.smolvlm_fixed_shape import FixedShapeCandidateScorer
        model_class = FixedShapeCandidateScorer
    model, _ = model_class.from_pretrained(
        ROOT / cfg['model']['pretrained'], score_batch_size=96)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.to(device).eval()
    spec = importlib.util.spec_from_file_location('source_diagnostic_e2',
        ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py')
    e2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e2)
    source = build_val_dataset(cfg)
    records = [r for r in source.records if r.label == 0][:2] + [r for r in source.records if r.label == 1][:2]
    native = NativeImageDataset(records, ROOT / cfg['model']['pretrained'], augment=False, seed=42)
    cache = open_cache(model, native, PHASE / 'results/native_cache_smoke', domain='ffpp', cfg=cfg)
    bank = load_language_bank(ROOT / cfg['language']['manifest'])
    lookup = CandidateTextLookup.build(model, bank, ('train',), device)
    suite = SemanticDataset(e2.CachedMetadataDataset(native), SemanticTemplateSampler(bank, 42),
                            mode='suite', tasks=TASKS, split='train')
    batch = next(iter(DataLoader(suite, batch_size=32, num_workers=0)))
    _, qids, cids, mask = e2._move_batch(batch, device, include_images=False)
    features = cache.lookup(batch['sample_id'], device)
    questions, candidates = lookup.lookup(qids, cids, mask)
    filename = 'source_permutation_shape_controls.json' if args.shape_controls else 'source_permutation_diagnosis.json'
    if args.fixed_scoring:
        filename = 'source_permutation_fixed_scoring.json'
    output_path = PHASE / 'results/progress' / filename
    if output_path.exists():
        raise FileExistsError('preserving earlier diagnostic evidence')
    results = {
        'started_utc': datetime.now(timezone.utc).isoformat(), 'reportable': False,
        'input_scope': 'source_validation_only', 'language_split': 'train',
        'seed': 42, 'epoch': checkpoint['epoch'], 'checkpoint_sha256': checkpoint_sha,
        'physical_samples': [str(r.sample_id) for r in records], 'request_count': len(mask),
        'threshold': 1e-6, 'checks': [],
        'preceding_probe': {'source': 'retained session summary: exec session 30753',
            'saved_json': False, 'exit_code': 1, 'max_probability_error': 1.1324882507324219e-6,
            'prediction_consistency': 1.0, 'threshold_passed': False},
    }
    shapes = []
    hook = model.backbone.register_forward_pre_hook(
        lambda _module, _args, kwargs: shapes.append(list(kwargs['input_ids'].shape)), with_kwargs=True)

    def save():
        temporary = output_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(results, indent=2) + '\n')
        temporary.replace(output_path)

    def compare(name, actual, expected, **extra):
        error = float((actual.softmax(-1) - expected.softmax(-1)).abs().max())
        row = {'name': name, 'max_probability_error': error,
               'max_logit_error': float((actual[torch.isfinite(expected)] - expected[torch.isfinite(expected)]).abs().max()),
               'prediction_consistency': float((actual.argmax(-1) == expected.argmax(-1)).float().mean()),
               'threshold_passed': error <= 1e-6, 'forward_shapes': list(shapes), **extra}
        shapes.clear()
        results['checks'].append(row)
        save()
        print(json.dumps(row), flush=True)

    if args.shape_controls:
        # Only source-language inputs; separate batch-size and sequence-padding effects.
        for pad_batch, pad_tokens in [(False, True), (True, False), (True, True)]:
            def fixed_shape(_module, _args, kwargs):
                ids, attention = kwargs['input_ids'], kwargs['attention_mask']
                images = kwargs['image_hidden_states']
                if pad_tokens:
                    padding = 128 - ids.shape[1]
                    if padding < 0:
                        raise ValueError('source diagnostic exceeds selected token shape')
                    ids = torch.nn.functional.pad(ids, (0, padding), value=model.prompts.pad_token_id)
                    attention = torch.nn.functional.pad(attention, (0, padding), value=False)
                if pad_batch:
                    count = 96 - ids.shape[0]
                    ids = torch.cat([ids, ids[-1:].expand(count, -1)])
                    attention = torch.cat([attention, attention[-1:].expand(count, -1)])
                    images = torch.cat([images, images[-1:].expand(count, -1, -1)])
                return _args, {**kwargs, 'input_ids': ids, 'attention_mask': attention,
                               'image_hidden_states': images}
            padding_hook = model.backbone.register_forward_pre_hook(fixed_shape, with_kwargs=True, prepend=True)
            shapes.clear()
            audit = permutation_audit(model, features, questions, candidates, mask, reduce_across_ranks=False)
            row = {'name': 'shape_control', 'pad_batch': pad_batch, 'pad_tokens': pad_tokens,
                   **audit, 'threshold_passed': audit['max_probability_error'] <= 1e-6,
                   'forward_shapes': list(shapes)}
            results['checks'].append(row)
            padding_hook.remove()
            save()
            print(json.dumps(row), flush=True)
        hook.remove()
        return

    reference = model.classify_features(features, questions, candidates, mask)
    results['reference_shapes'] = list(shapes)
    shapes.clear()
    if args.fixed_scoring:
        original = SmolVLMCandidateScorer.classify_features(model, features, questions, candidates, mask)
        compare('fixed_vs_original_numerical_drift', reference, original)
    repeat = model.classify_features(features, questions, candidates, mask)
    compare('identical_repeat', repeat, reference)
    # Swap candidates without changing any batch membership or maximum length.
    order = torch.arange(mask.shape[1], device=device).expand_as(mask).clone()
    for i, n in enumerate(mask.sum(1).tolist()):
        order[i, :n] = order[i, :n].flip(0)
    permuted = model.classify_features(features, questions, candidates.gather(1, order[..., None]), mask)
    compare('whole_batch_reverse', permuted.gather(1, order), reference)
    for n in (2, 3):
        selected = mask.sum(1) == n
        sub = (features[selected], questions[selected], candidates[selected], mask[selected])
        identity = model.classify_features(*sub)
        compare(f'subgroup_{n}_identity', identity, reference[selected])
        for perm in itertools.permutations(range(n)):
            full = list(perm) + list(range(n, mask.shape[1]))
            logits = model.classify_features(sub[0], sub[1], sub[2][:, full], sub[3][:, full])
            remapped = torch.empty_like(logits)
            remapped[:, full] = logits
            compare(f'subgroup_{n}_permutation', remapped, identity, permutation=list(perm))
    shapes.clear()
    frozen = permutation_audit(model, features, questions, candidates, mask, reduce_across_ranks=False)
    results['frozen_audit'] = {**frozen, 'threshold_passed': frozen['max_probability_error'] <= 1e-6,
                               'forward_shapes': list(shapes)}
    if args.fixed_scoring:
        shapes.clear()
        # Stress multiple chunks and two independent length buckets with source
        # images and one synthetic repeated instruction, no held-out templates.
        longer = model._texts[int(questions[0, 0])] + ' Check the candidate carefully.' * 30
        synthetic_id = max(model._texts) + 1
        model.register_texts([SimpleNamespace(text_id=synthetic_id, text=longer)])
        repeated_q = questions.repeat(5, 1)
        repeated_q[::3, 0] = synthetic_id
        stress = permutation_audit(model, features.repeat(5, 1, 1), repeated_q,
            candidates.repeat(5, 1, 1), mask.repeat(5, 1), reduce_across_ranks=False)
        results['multichunk_multilength_source_stress'] = {
            **stress, 'threshold_passed': stress['max_probability_error'] <= 1e-6,
            'request_count': len(repeated_q), 'synthetic_question': longer,
            'forward_shapes': sorted({tuple(shape) for shape in shapes})}
        print(json.dumps({'stress': results['multichunk_multilength_source_stress']}), flush=True)
    hook.remove()
    save()
    print(json.dumps({'frozen_audit': results['frozen_audit']}), flush=True)
    if not results['frozen_audit']['threshold_passed']:
        raise SystemExit(1)
    if args.fixed_scoring and not results['multichunk_multilength_source_stress']['threshold_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
