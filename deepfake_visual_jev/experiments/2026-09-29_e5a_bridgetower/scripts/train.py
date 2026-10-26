"""Single-seed E5-A qualification training; selection uses validation CE only."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(ROOT / 'src')]
from bridge_scorer import SharedBridgeScorer
from data.semantic_dataset import SemanticTemplateSampler, TASKS, method_family
from data.semantic_language import load_language_bank


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class Examples(Dataset):
    def __init__(self, rows, bank, *, split, seed, epoch, count=None):
        self.rows = rows
        self.bank = bank
        self.sampler = SemanticTemplateSampler(bank, seed)
        self.split = split
        self.epoch = epoch
        if count is None:
            # One deterministic train-wording question/set per task per validation record.
            self.draws = [(i, task, None) for i in range(len(rows)) for task in TASKS]
        else:
            groups = {'camera': [], 'swap': [], 'reenact': []}
            for index, row in enumerate(rows):
                groups[method_family(row['method'], row['label'])].append(index)
            if any(not group for group in groups.values()):
                raise ValueError('missing class in train source')
            rng = random.Random(seed * 100003 + epoch)
            t4_questions = {entry.yes_means: entry.name for entry in bank.task('T4', split).questions}
            if set(t4_questions) != {'swap', 'reenact'}:
                raise ValueError('T4 training requires swap and reenact questions')
            draws = []
            for i in range(count):
                task = TASKS[i % len(TASKS)]
                force_question = None
                if task in {'T1', 'T2', 'T3'}:
                    family = 'camera' if (i // len(TASKS)) % 2 == 0 else ('swap' if rng.randrange(2) == 0 else 'reenact')
                elif task == 'T4':
                    target_yes = (i // len(TASKS)) % 2 == 0
                    family = rng.choice(('swap', 'reenact') if target_yes else ('camera', 'swap', 'reenact'))
                    polarity = (family if target_yes else rng.choice(('swap', 'reenact'))
                                if family == 'camera' else ('reenact' if family == 'swap' else 'swap'))
                    force_question = t4_questions[polarity]
                else:
                    family = ('camera', 'swap', 'reenact')[(i // len(TASKS)) % 3]
                draws.append((rng.choice(groups[family]), task, force_question))
            rng.shuffle(draws)
            self.draws = draws

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, index):
        row_index, task, force_question = self.draws[index]
        row = self.rows[row_index]
        with Image.open(row['frame_path']) as image:
            rgb = image.convert('RGB')
        sample = {'sample_id': f"{row['frame_path']}::{row['label']}",
                  'label': int(row['label']), 'method_id': row['method'],
                  'video_id': row['pair_id'], 'image': rgb}
        return self.sampler.render(sample, task, self.split, self.epoch,
                                   force_question=force_question)


def collate(batch):
    return batch


def score_batch(model, rows):
    options = [list(row['candidate_texts'][:int(row['candidate_mask'].sum())]) for row in rows]
    scores = model([r['image'] for r in rows], [r['task_id'] for r in rows],
                   [r['question_text'] for r in rows], options)
    target = torch.tensor([int(r['target_index']) for r in rows], device=scores.device)
    return torch.nn.functional.cross_entropy(scores, target)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0', '1', '2', '3'}:
        raise ValueError('bind one physical GPU 0–3')
    if not json.loads((PHASE / 'results/protocol/physical_split_audit.json').read_text())['pass']:
        raise ValueError('physical audit not passed')
    cfg_path = PHASE / 'configs/e5_a.yaml'
    cfg = yaml.safe_load(cfg_path.read_text())
    source = (PHASE / cfg['data_protocol']).resolve()
    language_path = (PHASE / cfg['language_bank']).resolve()
    source_manifest = json.loads((source / 'manifest.json').read_text())
    rows = {}
    for name in ('train', 'val'):
        path = source / f'{name}.json'
        if digest(path) != source_manifest['splits'][name]['sha256']:
            raise ValueError(f'changed source protocol: {name}')
        rows[name] = json.loads(path.read_text())
    bank = load_language_bank(language_path)
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    model = SharedBridgeScorer(PHASE / 'results/pretrained/bridgetower-base-itm-mlm').cuda()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=cfg['train']['lr'], weight_decay=cfg['train']['weight_decay'])
    root = PHASE / 'results/training'
    root.mkdir(parents=True, exist_ok=True)
    provenance = {'seed': 42, 'config_sha256': digest(cfg_path), 'language_sha256': digest(language_path),
                  'source_manifest_sha256': digest(source / 'manifest.json'), 'counts': model.counts(),
                  'tokenizer_merges_sha256': digest(PHASE / 'results/pretrained/roberta-base-tokenizer/merges.txt'),
                  'tokenizer_vocab_sha256': digest(PHASE / 'results/pretrained/roberta-base-tokenizer/vocab.json'),
                  'train_source_sha256': digest(Path(__file__)),
                  'model_source_sha256': digest(PHASE / 'src/bridge_scorer.py'),
                  'selection': 'FF++ validation candidate CE only'}
    atomic_json(root / 'run_provenance.json', provenance)
    best = math.inf
    history = []
    for epoch in range(1, cfg['train']['epochs'] + 1):
        train = Examples(rows['train'], bank, split='train', seed=42, epoch=epoch,
                         count=cfg['train']['examples_per_epoch'])
        loader = DataLoader(train, batch_size=cfg['train']['microbatch'], shuffle=False,
                            num_workers=4, collate_fn=collate)
        model.train()
        # Frozen encoders must remain in eval mode to keep their dropout fixed.
        model.model.bridgetower.vision_model.eval()
        model.model.bridgetower.text_model.eval()
        optimizer.zero_grad(set_to_none=True)
        losses, steps = [], 0
        for step, batch in enumerate(loader, 1):
            loss = score_batch(model, batch)
            if not torch.isfinite(loss):
                raise ValueError(f'nonfinite train loss at epoch {epoch}, batch {step}')
            (loss / cfg['train']['accumulation']).backward()
            losses.append(float(loss.detach()))
            if step % cfg['train']['accumulation'] == 0 or step == len(loader):
                if not all(p.grad is None for p in model.model.bridgetower.vision_model.parameters()):
                    raise ValueError('vision encoder gradient leak')
                if not all(p.grad is None for p in model.model.bridgetower.text_model.parameters()):
                    raise ValueError('text encoder gradient leak')
                norm = torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
                if not torch.isfinite(norm):
                    raise ValueError(f'nonfinite gradient norm at epoch {epoch}, batch {step}')
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
            if step % 100 == 0:
                print(json.dumps({'epoch': epoch, 'batch': step, 'batches': len(loader),
                                  'train_ce_recent': sum(losses[-100:]) / len(losses[-100:])}), flush=True)
        model.eval()
        val = Examples(rows['val'], bank, split='train', seed=42, epoch=0)
        val_loader = DataLoader(val, batch_size=cfg['train']['microbatch'], shuffle=False,
                                num_workers=4, collate_fn=collate)
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                loss = score_batch(model, batch)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite validation CE')
                total += float(loss) * len(batch)
                count += len(batch)
        val_ce = total / count
        entry = {'epoch': epoch, 'train_ce': sum(losses) / len(losses), 'val_candidate_ce': val_ce,
                 'optimizer_steps': steps, 'skipped_optimization': 0}
        history.append(entry)
        atomic_json(root / 'history.json', history)
        print(json.dumps(entry), flush=True)
        if val_ce < best:
            best = val_ce
            temp = root / 'best.pth.tmp'
            torch.save({'state': {k: v.cpu() for k, v in model.model.state_dict().items() if not k.startswith(('bridgetower.vision_model.', 'bridgetower.text_model.'))},
                        'epoch': epoch, 'validation_ce': val_ce, 'provenance': provenance}, temp)
            temp.replace(root / 'best.pth')
    atomic_json(root / 'complete.json', {'complete': True, 'epochs': cfg['train']['epochs'],
                'best_validation_ce': best, 'best_checkpoint_sha256': digest(root / 'best.pth'),
                'history_sha256': digest(root / 'history.json'), **provenance})


if __name__ == '__main__':
    main()
