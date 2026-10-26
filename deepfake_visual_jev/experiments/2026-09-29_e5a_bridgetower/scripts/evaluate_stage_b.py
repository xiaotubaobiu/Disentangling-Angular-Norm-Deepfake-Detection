"""Frozen FF++ Stage B: T2 seen/heldout, polarity, permutation."""
from __future__ import annotations

from collections import defaultdict
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(ROOT / 'src')]
from bridge_scorer import SharedBridgeScorer
from data.semantic_dataset import SemanticTemplateSampler
from data.semantic_language import load_language_bank
from train import atomic_json, digest


class T2Suite(Dataset):
    def __init__(self, records, bank, split):
        self.records = records
        self.sampler = SemanticTemplateSampler(bank, 42)
        self.split = split
        language = bank.task('T2', split)
        self.slots = [(q.name, c.name) for q in language.questions for c in language.candidate_sets]
        self._cached_path = None
        self._cached_image = None

    def __len__(self):
        return len(self.records) * len(self.slots)

    def __getitem__(self, index):
        frame, slot = divmod(index, len(self.slots))
        row = self.records[frame]
        if self._cached_path != row['frame_path']:
            with Image.open(row['frame_path']) as image:
                self._cached_image = image.convert('RGB')
            self._cached_path = row['frame_path']
        rgb = self._cached_image.copy()
        sample = {'sample_id': f"{row['frame_path']}::{row['label']}",
                  'label': int(row['label']), 'method_id': row['method'],
                  'video_id': row['pair_id'], 'image': rgb}
        q, c = self.slots[slot]
        return self.sampler.render(sample, 'T2', self.split, 0,
                                   permutation=(0, 1), force_question=q,
                                   force_candidate_set=c)


def collate(batch):
    return batch


def score(model, batch, *, reverse=False):
    candidates = [list(row['candidate_texts'][:2]) for row in batch]
    if reverse:
        candidates = [list(reversed(row)) for row in candidates]
    logits = model([row['image'] for row in batch], ['T2'] * len(batch),
                   [row['question_text'] for row in batch], candidates)
    probabilities = logits.softmax(-1)
    yes_index = [list(row['candidate_semantics'][:2]).index('yes') for row in batch]
    if reverse:
        yes_index = [1 - index for index in yes_index]
    return [float(probabilities[i, idx]) for i, idx in enumerate(yes_index)]


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        key = (row['video_key'], row['variant_id'], row['question_id'])
        groups[key].append(row)
    result = []
    for (video, variant, question), rows in groups.items():
        label = rows[0]['label']
        polarity = rows[0]['yes_means']
        if any(row['label'] != label or row['yes_means'] != polarity for row in rows):
            raise ValueError('inconsistent video aggregation group')
        result.append({'video_key': video, 'variant_id': variant, 'question_id': question,
                       'label': label, 'yes_means': polarity,
                       'p_yes': sum(row['p_yes'] for row in rows) / len(rows),
                       'frames': len(rows)})
    return result


def balanced_accuracy(rows):
    per_class = {}
    for label in (0, 1):
        selected = [row for row in rows if row['label'] == label]
        if not selected:
            raise ValueError(f'empty class {label}')
        correct = sum((row['p_yes'] >= 0.5) == (row['yes_means'] == ('real' if label == 0 else 'fake'))
                      for row in selected)
        per_class[str(label)] = correct / len(selected)
    return (per_class['0'] + per_class['1']) / 2, per_class


def polarity_consistency(rows):
    groups = defaultdict(lambda: {'real': [], 'fake': []})
    for row in rows:
        candidate_set = row['variant_id'].split('::')[-1]
        groups[(row['video_key'], candidate_set)][row['yes_means']].append(row['p_yes'])
    consistent = 0
    for values in groups.values():
        if not values['real'] or not values['fake']:
            raise ValueError('missing polarity pair')
        real_yes = sum(values['real']) / len(values['real']) >= 0.5
        fake_yes = sum(values['fake']) / len(values['fake']) >= 0.5
        consistent += real_yes != fake_yes
    return consistent / len(groups), len(groups)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_id < args.shard_count:
        raise ValueError('invalid shard assignment')
    if args.merge:
        merge(args.shard_count)
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0', '1', '2', '3'}:
        raise ValueError('bind one physical GPU 0–3')
    complete_path = PHASE / 'results/training/complete.json'
    complete = json.loads(complete_path.read_text())
    checkpoint = PHASE / 'results/training/best.pth'
    if not complete['complete'] or complete['best_checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('missing verified selected checkpoint')
    if complete['config_sha256'] != digest(PHASE / 'configs/e5_a.yaml'):
        raise ValueError('training config changed after selection')
    if complete['train_source_sha256'] != digest(PHASE / 'scripts/train.py') or (
            complete['model_source_sha256'] != digest(PHASE / 'src/bridge_scorer.py')):
        raise ValueError('training/model source changed after selection')
    for name in ('merges.txt', 'vocab.json'):
        if complete[f'tokenizer_{name.split(".")[0]}_sha256'] != digest(
                PHASE / 'results/pretrained/roberta-base-tokenizer' / name):
            raise ValueError(f'tokenizer {name} changed after selection')
    source = PHASE.parent / '2026-09-28_e4_antifakeprompt/results/protocol'
    if complete['source_manifest_sha256'] != digest(source / 'manifest.json'):
        raise ValueError('source manifest changed after selection')
    manifest = json.loads((source / 'manifest.json').read_text())
    source_rows = source / 'ffpp_full.json'
    if digest(source_rows) != manifest['splits']['ffpp_full']['sha256']:
        raise ValueError('FF++ frozen source changed')
    records = json.loads(source_rows.read_text())
    if args.shard_count > 1:
        def shard(row):
            key = f"{row['method']}::{row['pair_id']}".encode()
            return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % args.shard_count
        records = [row for row in records if shard(row) == args.shard_id]
    language_path = PHASE.parent / '2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml'
    if complete['language_sha256'] != digest(language_path):
        raise ValueError('language bank changed after selection')
    bank = load_language_bank(language_path)
    model = SharedBridgeScorer(PHASE / 'results/pretrained/bridgetower-base-itm-mlm').cuda().eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    missing, unexpected = model.model.load_state_dict(state['state'], strict=False)
    if unexpected or not all(name.startswith(('bridgetower.vision_model.', 'bridgetower.text_model.')) for name in missing):
        raise ValueError(f'checkpoint state mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}')
    out = PHASE / 'results/stage_b'
    out.mkdir(parents=True, exist_ok=True)
    summary = {'checkpoint_sha256': digest(checkpoint), 'source_sha256': digest(source_rows),
               'seed': 42, 'domain': 'ffpp', 'shard_count': args.shard_count,
               'shard_id': args.shard_id, 'conditions': {}}
    torch.set_num_threads(4)
    for split in ('train', 'heldout'):
        suite = T2Suite(records, bank, split)
        loader = DataLoader(suite, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate)
        path = out / f't2_{split}_shard{args.shard_id}_frame_predictions.csv'
        semantic_match, count, max_error = 0, 0, 0.0
        with path.open('w', newline='') as stream, torch.inference_mode():
            writer = csv.DictWriter(stream, fieldnames=['video_key', 'variant_id', 'question_id',
                'label', 'yes_means', 'p_yes']); writer.writeheader()
            for batch_index, batch in enumerate(loader, 1):
                values = score(model, batch)
                swapped = score(model, batch, reverse=True) if split == 'heldout' else None
                for i, row in enumerate(batch):
                    writer.writerow({'video_key': f"{row['method_id']}::{row['video_id']}",
                                     'variant_id': row['variant_id'], 'question_id': row['question_id'],
                                     'label': int(row['label']), 'yes_means': row['question_yes_means_name'],
                                     'p_yes': values[i]})
                    if swapped is not None:
                        semantic_match += (values[i] >= 0.5) == (swapped[i] >= 0.5)
                        max_error = max(max_error, abs(values[i] - swapped[i]))
                        count += 1
                if batch_index % 100 == 0:
                    print(json.dumps({'split': split, 'batch': batch_index,
                                      'batches': len(loader)}), flush=True)
        with path.open(newline='') as stream:
            video_rows = aggregate([{'video_key': row['video_key'], 'variant_id': row['variant_id'],
                'question_id': row['question_id'], 'label': int(row['label']),
                'yes_means': row['yes_means'], 'p_yes': float(row['p_yes'])}
                for row in csv.DictReader(stream)])
        ba, recalls = balanced_accuracy(video_rows)
        polarity, pairs = polarity_consistency(video_rows)
        summary['conditions'][split] = {'video_variant_rows': len(video_rows),
            'physical_videos': len({r['video_key'] for r in video_rows}), 't2_ba': ba,
            'class_recalls': recalls, 'polarity_consistency': polarity,
            'polarity_pairs': pairs, 'permutation_consistency': semantic_match / count if count else None,
            'permutation_comparisons': count, 'permutation_max_probability_error': max_error if count else None,
            'frame_predictions_sha256': digest(path)}
        atomic_json(out / f'stage_b_shard{args.shard_id}.json', summary)
        print(json.dumps({'split': split, **summary['conditions'][split]}), flush=True)
    if args.shard_count == 1:
        merge(1)


def merge(shard_count):
    out = PHASE / 'results/stage_b'
    shards = [json.loads((out / f'stage_b_shard{i}.json').read_text()) for i in range(shard_count)]
    reference = shards[0]
    for i, shard in enumerate(shards):
        if shard['shard_count'] != shard_count or shard['shard_id'] != i or any(
                shard[key] != reference[key] for key in ('checkpoint_sha256', 'source_sha256', 'seed', 'domain')):
            raise ValueError(f'incompatible shard {i}')
    summary = {key: reference[key] for key in ('checkpoint_sha256', 'source_sha256', 'seed', 'domain')}
    summary['shard_count'] = shard_count
    summary['conditions'] = {}
    for split in ('train', 'heldout'):
        frames = []
        video_shard = {}
        for i, shard in enumerate(shards):
            path = out / f't2_{split}_shard{i}_frame_predictions.csv'
            if digest(path) != shard['conditions'][split]['frame_predictions_sha256']:
                raise ValueError(f'changed shard predictions: {i}/{split}')
            with path.open(newline='') as stream:
                for row in csv.DictReader(stream):
                    previous = video_shard.setdefault(row['video_key'], i)
                    if previous != i:
                        raise ValueError('physical video appears in multiple evaluation shards')
                    frames.append({'video_key': row['video_key'], 'variant_id': row['variant_id'],
                        'question_id': row['question_id'], 'label': int(row['label']),
                        'yes_means': row['yes_means'], 'p_yes': float(row['p_yes'])})
        videos = aggregate(frames)
        if len(frames) != 12196 * 8:
            raise ValueError(f'incomplete T2 {split} frame coverage: {len(frames)}')
        ba, recalls = balanced_accuracy(videos)
        polarity, pairs = polarity_consistency(videos)
        comparisons = sum(shard['conditions'][split]['permutation_comparisons'] for shard in shards)
        consistency = (sum(shard['conditions'][split]['permutation_consistency'] *
                           shard['conditions'][split]['permutation_comparisons'] for shard in shards) /
                       comparisons) if comparisons else None
        summary['conditions'][split] = {'video_variant_rows': len(videos),
            'physical_videos': len({row['video_key'] for row in videos}), 't2_ba': ba,
            'class_recalls': recalls, 'polarity_consistency': polarity, 'polarity_pairs': pairs,
            'permutation_consistency': consistency, 'permutation_comparisons': comparisons,
            'permutation_max_probability_error': max(shard['conditions'][split]['permutation_max_probability_error'] or 0
                                                       for shard in shards)}
    if any(summary['conditions'][split]['physical_videos'] != 3049 for split in ('train', 'heldout')):
        raise ValueError('Stage B did not cover full frozen FF++ test set')
    # A reversed candidate list can change the last floating point bits of softmax.
    # Require every semantic decision to agree and the maximum probability drift
    # across all frame requests to remain within numerical tolerance.
    summary['permutation_probability_tolerance'] = 1e-5
    gates = {'G1_t2_heldout': summary['conditions']['heldout']['t2_ba'] >= 0.70,
             'G2_polarity': summary['conditions']['heldout']['polarity_consistency'] >= 0.90,
             'G6_permutation': (summary['conditions']['heldout']['permutation_consistency'] == 1.0 and
                 summary['conditions']['heldout']['permutation_max_probability_error'] <=
                 summary['permutation_probability_tolerance'])}
    summary['gates'] = gates
    summary['stage_b_status'] = 'PASS' if all(gates.values()) else 'FAIL'
    atomic_json(out / 'stage_b_result.json', summary)
    print(json.dumps({'stage_b_status': summary['stage_b_status'], 'gates': gates}), flush=True)


if __name__ == '__main__':
    main()
