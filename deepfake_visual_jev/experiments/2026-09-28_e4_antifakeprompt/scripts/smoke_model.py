"""Real 7B forward/backward and original-rank equivalence. Not scientific results."""
from pathlib import Path
import gc
import json
import os
import sys
import time

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(ROOT / 'src')]
from upstream_loader import load_backbone
from antifake_model import AntifakeScorer
from data.index_ffpp import load_json_index

if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
    raise RuntimeError('smoke requires explicit physical GPU 0')
torch.manual_seed(42)
torch.set_num_threads(4)
start = time.monotonic()
backbone = load_backbone(PHASE)
print('model loaded', time.monotonic() - start, flush=True)
ckpt = torch.load(PHASE / 'results/assets/COCO_150k_SD3_SD2IP.pth', map_location='cpu', weights_only=False)
processor = ckpt['config']['datasets']['textinv']['vis_processor']['eval']
transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(), transforms.Normalize(processor['mean'], processor['std'])])
records = [load_json_index('/Dataset/deepfake_detection/DF40_all', 'simswap', 'ff', 'train', label, max_videos=1)[0]
           for label in (0, 1)]
images = torch.stack([transform(Image.open(r.frame_path).convert('RGB')) for r in records]).cuda()
questions = ['Is this photo real?', 'Is this photo real?']
candidates = [['yes', 'no'], ['yes', 'no']]
mask = torch.ones(2, 2, dtype=torch.bool, device='cuda')
targets = torch.tensor([0, 1], device='cuda')
result = {'smoke': True, 'reportable': False, 'processor': processor, 'load_audit': backbone.load_audit,
          'image_paths': [r.frame_path for r in records], 'modes': {}}
for mode in ('lm', 'scalar'):
    scorer = AntifakeScorer(backbone, mode=mode, gradient_checkpointing=True).cuda().eval()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        tick = time.monotonic()
        scores = scorer(images, questions, candidates, mask)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - tick
        if not torch.isfinite(scores).all():
            raise ValueError('non-finite forward')
        rank_equal = None
        if mode == 'lm':
            ranks = backbone.predict_class({'image': images, 'prompt': 'Is this photo real [*]?'}, ['yes', 'no'])
            rank_equal = bool(torch.equal(scores.argsort(-1, descending=True), ranks))
            assert rank_equal
    scorer.train()
    loss = scorer.training_loss(images, questions, candidates, mask, targets)
    loss.backward()
    grads = {}
    for name, p in scorer.named_parameters():
        if p.requires_grad:
            if p.grad is None or not torch.isfinite(p.grad).all():
                raise ValueError(f'invalid gradient: {name}')
            # An additive bias shared by every candidate cancels in softmax;
            # its exact zero gradient is correct, not an interrupted graph.
            if name.endswith(('train_word_embeddings.weight', 'train_embed_tokens.weight', 'score_head.1.weight')) and p.grad.abs().sum() == 0:
                raise ValueError(f'missing active gradient: {name}')
            grads[name] = float(p.grad.norm())
        elif p.grad is not None:
            raise ValueError(f'frozen parameter has gradient: {name}')
    result['modes'][mode] = {'scores': scores.tolist(), 'loss': float(loss), 'gradient_norms': grads,
        'parameters': sum(p.numel() for p in scorer.parameters()),
        'trainable_parameters': sum(p.numel() for p in scorer.parameters() if p.requires_grad),
        'batch2_seconds': elapsed, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'upstream_rank_equal': rank_equal}
    print(mode, result['modes'][mode], flush=True)
    scorer.zero_grad(set_to_none=True)
    del loss, scores, scorer
    gc.collect()
    torch.cuda.empty_cache()
out = PHASE / 'results/smoke/model.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(result, indent=2) + '\n')
print('SMOKE COMPLETE', flush=True)
