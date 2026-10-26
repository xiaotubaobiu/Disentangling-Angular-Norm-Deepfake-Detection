"""FF++ T1/T3/T4/T5 evaluation, with an explicit post-failure diagnostic mode."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]
sys.path[:0] = [str(PHASE / 'src'), str(ROOT / 'src')]
from bridge_scorer import SharedBridgeScorer
from data.semantic_dataset import SemanticTemplateSampler
from data.semantic_language import load_language_bank
from train import atomic_json, digest

TASKS = ('T1', 'T3', 'T4', 'T5')
SEMANTICS = ('real', 'fake', 'yes', 'no', 'camera', 'swap', 'reenact')
FIELDS = ('video_key', 'variant_id', 'question_id', 'task_id', 'label', 'method_family',
          'target_semantic', 'question_yes_means', *(f'p_{name}' for name in SEMANTICS))


def collate(batch):
    return batch


class SemanticSuite(Dataset):
    def __init__(self, records, bank, split):
        self.records = records
        self.sampler = SemanticTemplateSampler(bank, 42)
        self.split = split
        self.slots = []
        for task in TASKS:
            language = bank.task(task, split)
            self.slots.extend((task, q.name, c.name, len(c.options))
                              for q in language.questions for c in language.candidate_sets)
        self._cached_path = None
        self._cached_image = None

    def __len__(self):
        return len(self.records) * len(self.slots)

    def __getitem__(self, index):
        frame, slot = divmod(index, len(self.slots))
        record = self.records[frame]
        if self._cached_path != record['frame_path']:
            with Image.open(record['frame_path']) as image:
                self._cached_image = image.convert('RGB')
            self._cached_path = record['frame_path']
        task, q, c, count = self.slots[slot]
        sample = {'sample_id': f"{record['frame_path']}::{record['label']}",
                  'label': int(record['label']), 'method_id': record['method'],
                  'video_id': record['pair_id'], 'image': self._cached_image.copy()}
        return self.sampler.render(sample, task, self.split, 0,
            permutation=tuple(range(count)), force_question=q, force_candidate_set=c)


def score(model, batch, reverse=False):
    candidate_texts = [list(row['candidate_texts'][:int(row['candidate_mask'].sum())]) for row in batch]
    if reverse:
        candidate_texts = [list(reversed(options)) for options in candidate_texts]
    logits = model([row['image'] for row in batch], [row['task_id'] for row in batch],
                   [row['question_text'] for row in batch], candidate_texts)
    probabilities = logits.softmax(-1).detach().cpu().tolist()
    mapped = []
    for row, options in zip(batch, probabilities, strict=True):
        count = int(row['candidate_mask'].sum())
        names = list(row['candidate_semantics'][:count])
        if reverse:
            names.reverse()
        mapped.append(dict(zip(names, options[:count], strict=True)))
    return mapped


def aggregate(frames):
    grouped = defaultdict(list)
    for row in frames:
        grouped[(row['video_key'], row['task_id'], row['variant_id'], row['question_id'])].append(row)
    videos = []
    for _, items in grouped.items():
        first = items[0]
        for item in items[1:]:
            if any(item[name] != first[name] for name in (
                    'label', 'method_family', 'target_semantic', 'question_yes_means')):
                raise ValueError('inconsistent frame group')
        video = {name: first[name] for name in FIELDS if not name.startswith('p_')}
        for semantic in SEMANTICS:
            values = [row[f'p_{semantic}'] for row in items if row[f'p_{semantic}'] is not None]
            video[f'p_{semantic}'] = sum(values) / len(values) if values else None
        videos.append(video)
    return videos


def metrics(videos):
    result = {}
    for split in ('train', 'heldout'):
        subset = [row for row in videos if row['split'] == split]
        tasks = {}
        for task in TASKS:
            rows = [row for row in subset if row['task_id'] == task]
            if len({row['video_key'] for row in rows}) != 3049:
                raise ValueError(f'incomplete FF++ {split}/{task} coverage')
            if task in ('T1', 'T3'):
                value = roc_auc_score([row['label'] for row in rows],
                                      [row['p_fake'] for row in rows])
                metric = 'auroc'
            elif task == 'T4':
                value = balanced_accuracy_score(
                    [row['target_semantic'] for row in rows],
                    ['yes' if row['p_yes'] >= 0.5 else 'no' for row in rows])
                metric = 'balanced_accuracy'
            else:
                value = recall_score(
                    [row['method_family'] for row in rows],
                    [max(('camera', 'swap', 'reenact'), key=lambda s: row[f'p_{s}']) for row in rows],
                    labels=['camera', 'swap', 'reenact'], average='macro', zero_division=0)
                metric = 'macro_recall'
            tasks[task] = {'metric': metric, 'value': float(value), 'rows': len(rows)}
        result[split] = tasks
    heldout_t4 = [row for row in videos if row['split'] == 'heldout' and row['task_id'] == 'T4'
                  and row['method_family'] in ('swap', 'reenact')]
    pairs = defaultdict(lambda: {'swap': [], 'reenact': [], 'family': None})
    for row in heldout_t4:
        candidate_set = row['variant_id'].split('::')[-1]
        group = pairs[(row['video_key'], candidate_set)]
        group['family'] = row['method_family']
        group[row['question_yes_means']].append(row['p_yes'])
    wins = 0
    for pair in pairs.values():
        if not pair['swap'] or not pair['reenact']:
            raise ValueError('missing T4 matched/counterfactual question')
        family = pair['family']
        other = 'reenact' if family == 'swap' else 'swap'
        wins += (sum(pair[family]) / len(pair[family]) > sum(pair[other]) / len(pair[other]))
    return result, wins / len(pairs), len(pairs)


def output_dir(exploratory):
    return PHASE / ('results/t1_t5_exploratory' if exploratory else 'results/stage_c')


def check_stage_b(stage_b, exploratory):
    expected = 'FAIL' if exploratory else 'PASS'
    if stage_b['stage_b_status'] != expected:
        raise ValueError(f'{"exploratory diagnostic" if exploratory else "Stage C"} requires Stage B {expected}')


def run_shard(shard_count, shard_id, exploratory=False):
    stage_b = json.loads((PHASE / 'results/stage_b/stage_b_result.json').read_text())
    check_stage_b(stage_b, exploratory)
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0', '1', '2', '3'}:
        raise ValueError('bind one GPU 0–3')
    complete = json.loads((PHASE / 'results/training/complete.json').read_text())
    checkpoint = PHASE / 'results/training/best.pth'
    if not complete['complete'] or complete['best_checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('training checkpoint changed')
    if stage_b['checkpoint_sha256'] != digest(checkpoint):
        raise ValueError('Stage B used different checkpoint')
    source = PHASE.parent / '2026-09-28_e4_antifakeprompt/results/protocol/ffpp_full.json'
    if stage_b['source_sha256'] != digest(source):
        raise ValueError('Stage B used different FF++ source')
    records = json.loads(source.read_text())
    def shard(row):
        key = f"{row['method']}::{row['pair_id']}".encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % shard_count
    records = [row for row in records if shard(row) == shard_id]
    bank = load_language_bank(PHASE.parent / '2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml')
    model = SharedBridgeScorer(PHASE / 'results/pretrained/bridgetower-base-itm-mlm').cuda().eval()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    missing, unexpected = model.model.load_state_dict(state['state'], strict=False)
    if unexpected or not all(name.startswith(('bridgetower.vision_model.', 'bridgetower.text_model.')) for name in missing):
        raise ValueError('selected checkpoint state mismatch')
    out = output_dir(exploratory)
    out.mkdir(parents=True, exist_ok=True)
    report = {'mode': 'post_stage_b_failure_diagnostic' if exploratory else 'stage_c_qualification',
              'shard_id': shard_id, 'shard_count': shard_count,
              'checkpoint_sha256': digest(checkpoint), 'source_sha256': digest(source), 'splits': {}}
    torch.set_num_threads(4)
    for split in ('train', 'heldout'):
        suite = SemanticSuite(records, bank, split)
        loader = DataLoader(suite, batch_size=24, shuffle=False, num_workers=4,
                            collate_fn=collate)
        path = out / f'semantic_{split}_shard{shard_id}.csv'
        consistent, comparisons, max_error = 0, 0, 0.0
        with path.open('w', newline='') as stream, torch.inference_mode():
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for batch_index, batch in enumerate(loader, 1):
                original = score(model, batch)
                swapped = score(model, batch, reverse=True)
                for row, first, second in zip(batch, original, swapped, strict=True):
                    first_best = max(first, key=first.get)
                    second_best = max(second, key=second.get)
                    consistent += first_best == second_best
                    comparisons += 1
                    max_error = max(max_error, *(abs(first[name] - second[name]) for name in first))
                    record = {'video_key': f"{row['method_id']}::{row['video_id']}",
                              'variant_id': row['variant_id'], 'question_id': row['question_id'],
                              'task_id': row['task_id'], 'label': int(row['label']),
                              'method_family': row['method_family'], 'target_semantic': row['target_semantic'],
                              'question_yes_means': row['question_yes_means_name']}
                    record.update({f'p_{name}': first.get(name) for name in SEMANTICS})
                    writer.writerow(record)
                if batch_index % 100 == 0:
                    print(json.dumps({'split': split, 'shard': shard_id,
                                      'batch': batch_index, 'batches': len(loader)}), flush=True)
        report['splits'][split] = {'rows': comparisons,
            'permutation_comparisons': comparisons,
            'permutation_consistency': consistent / comparisons,
            'permutation_max_probability_error': max_error,
            'csv_sha256': digest(path)}
        atomic_json(out / f'stage_c_shard{shard_id}.json', report)
    print(json.dumps({'shard': shard_id, 'complete': True}), flush=True)


def merge(shard_count, exploratory=False):
    stage_b = json.loads((PHASE / 'results/stage_b/stage_b_result.json').read_text())
    check_stage_b(stage_b, exploratory)
    out = output_dir(exploratory)
    shards = [json.loads((out / f'stage_c_shard{i}.json').read_text()) for i in range(shard_count)]
    all_videos = []
    permutation_count = 0
    permutation_agree = 0.0
    max_error = 0.0
    for split in ('train', 'heldout'):
        frames = []
        video_shard = {}
        for i, shard in enumerate(shards):
            if shard['mode'] != ('post_stage_b_failure_diagnostic' if exploratory else 'stage_c_qualification') or (
                    shard['shard_id'] != i or shard['shard_count'] != shard_count) or (
                    shard['checkpoint_sha256'] != stage_b['checkpoint_sha256'] or
                    shard['source_sha256'] != stage_b['source_sha256']):
                raise ValueError('incompatible Stage C shard')
            path = out / f'semantic_{split}_shard{i}.csv'
            if digest(path) != shard['splits'][split]['csv_sha256']:
                raise ValueError('changed Stage C shard predictions')
            with path.open(newline='') as stream:
                for row in csv.DictReader(stream):
                    previous = video_shard.setdefault(row['video_key'], i)
                    if previous != i:
                        raise ValueError('video in two Stage C shards')
                    row['label'] = int(row['label'])
                    for semantic in SEMANTICS:
                        cell = row[f'p_{semantic}']
                        row[f'p_{semantic}'] = float(cell) if cell else None
                    frames.append(row)
            item = shard['splits'][split]
            permutation_count += item['permutation_comparisons']
            permutation_agree += item['permutation_consistency'] * item['permutation_comparisons']
            max_error = max(max_error, item['permutation_max_probability_error'])
        if len(video_shard) != 3049:
            raise ValueError('incomplete Stage C physical video coverage')
        expected_slots = 10 if split == 'train' else 8
        if len(frames) != 12196 * expected_slots:
            raise ValueError(f'incomplete {split} frame/language coverage: {len(frames)}')
        for row in aggregate(frames):
            row['split'] = split
            all_videos.append(row)
    task_metrics, t4_rank, t4_pairs = metrics(all_videos)
    result = {'mode': 'post_stage_b_failure_diagnostic' if exploratory else 'stage_c_qualification',
              'checkpoint_sha256': stage_b['checkpoint_sha256'], 'source_sha256': stage_b['source_sha256'],
              'task_metrics': task_metrics, 't4_matched_over_counterfactual': t4_rank,
              't4_family_pairs': t4_pairs, 'permutation_consistency': permutation_agree / permutation_count,
              'permutation_comparisons': permutation_count,
              'permutation_max_probability_error': max_error}
    atomic_json(out / 'stage_c_result.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard-count', type=int, default=4)
    parser.add_argument('--shard-id', type=int, default=0)
    parser.add_argument('--merge', action='store_true')
    parser.add_argument('--exploratory-after-stage-b-fail', action='store_true',
                        help='explicitly run T1/T3/T4/T5 after failed Stage B without changing E5-A qualification')
    args = parser.parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_id < args.shard_count:
        raise ValueError('invalid shard assignment')
    if args.merge:
        merge(args.shard_count, args.exploratory_after_stage_b_fail)
    else:
        run_shard(args.shard_count, args.shard_id, args.exploratory_after_stage_b_fail)
