"""Same GPU/input, warmed batch-one inference; run after evaluation jobs exit."""
from pathlib import Path
import gc
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import default_collate
import yaml

PHASE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(PHASE.parents[1] / 'src')]
from upstream_loader import load_backbone
from antifake_model import AntifakeScorer
from experiment_data import load_records, ImageRecords, language_bank, atomic_json, sha256, ExhaustiveSuite
from evaluation_adapter import MetricAdapter, TextLookup
from permutation_eval import fixed_batch_audit
from evaluation.semantic_controls import permutation_audit

if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
    raise RuntimeError('paired benchmark uses only physical GPU0')
torch.set_num_threads(4)
cfg = yaml.safe_load((PHASE / 'configs/stage1.yaml').read_text())
prep = cfg['preprocess']
base = ImageRecords(load_records('val'), prep['mean'], prep['std'])
image = base[0]['image'][None].cuda()
bank = language_bank()
question = [bank.text(bank.text_id('fixed_h1_q'))]
candidates = [[bank.text(bank.text_id('fixed_h1_real')), bank.text(bank.text_id('fixed_h1_fake'))]]
mask = torch.ones(1, 2, dtype=torch.bool, device='cuda')
result = {'gpu': torch.cuda.get_device_name(), 'physical_gpu': 0, 'torch_version': torch.__version__,
          'batch_size': 1, 'candidate_count': 2, 'warmup_iterations': 20, 'timed_iterations': 50,
          'image': base.records[0].frame_path, 'arms': {}}
result['source_sha256'] = sha256(Path(__file__))
for arm, mode in [('A1', 'lm'), ('A2', 'scalar')]:
    torch.manual_seed(42)
    scorer = AntifakeScorer(load_backbone(PHASE), mode=mode, gradient_checkpointing=False).cuda().eval()
    checkpoint = PHASE / f'results/training/{arm}/best.pth'
    scorer.load_trainable(torch.load(checkpoint, map_location='cpu', weights_only=True)['state'])
    metrics = {'parameters': sum(p.numel() for p in scorer.parameters()),
               'trainable_parameters': sum(p.numel() for p in scorer.parameters() if p.requires_grad),
               'parameter_bytes': sum(p.numel() * p.element_size() for p in scorer.parameters()),
               'adaptation_checkpoint_bytes': checkpoint.stat().st_size,
               'checkpoint_sha256': sha256(checkpoint)}
    with torch.no_grad():
        features = scorer.extract_features(image)
        for cached in (False, True):
            for _ in range(20):
                output = scorer.classify_features(features, question, candidates, mask) if cached else scorer(image, question, candidates, mask)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            timings = []
            for _ in range(50):
                tick = time.perf_counter()
                output = scorer.classify_features(features, question, candidates, mask) if cached else scorer(image, question, candidates, mask)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - tick)
            metrics['cached' if cached else 'full'] = {
                'median_latency_ms': float(np.median(timings) * 1000),
                'p95_latency_ms': float(np.quantile(timings, .95) * 1000),
                'serial_images_per_second': float(1 / np.mean(timings)),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        # Supplementary numerical diagnosis uses a fixed validation image and
        # mixed 2/3-candidate batch. Keep the original formal G3 results intact.
        suite = ExhaustiveSuite(base, bank, split='heldout')
        probe = default_collate([suite[0], suite[len(suite.slots) - 1]])
        lookup = TextLookup(bank)
        adapter = MetricAdapter(scorer, lookup).eval()
        qs, cs = lookup.lookup(probe['question_text_id'].cuda(), probe['candidate_text_ids'].cuda(), probe['candidate_mask'].cuda())
        probe_features = scorer.extract_features(probe['image'].cuda())
        probe_mask = probe['candidate_mask'].cuda()
        metrics['permutation_numerical_probe'] = {
            'scope': 'supplementary validation-image probe; does not replace formal G3',
            'original_batch_regrouping': permutation_audit(adapter, probe_features, qs, cs, probe_mask, reduce_across_ranks=False),
            'fixed_batch': fixed_batch_audit(adapter, probe_features, qs, cs, probe_mask)}
    result['arms'][arm] = metrics
    print(arm, metrics, flush=True)
    del scorer, features, output, adapter, probe_features
    gc.collect()
    torch.cuda.empty_cache()
atomic_json(PHASE / 'results/summary/benchmark.json', result)
print('BENCHMARK COMPLETE', flush=True)
