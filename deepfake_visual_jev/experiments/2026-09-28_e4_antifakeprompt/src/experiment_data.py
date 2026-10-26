"""Immutable physical splits and paired training/evaluation requests."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from data.index_ffpp import FrameRecord
from data.semantic_dataset import SemanticTemplateSampler, method_family, TASKS
from data.semantic_language import load_language_bank
from evaluation.smolvlm_diagnostics import extended_language_bank

PHASE = Path(__file__).resolve().parents[1]
ROOT = PHASE.parents[1]


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_gpu(visible):
    devices = visible.split(',')
    if not devices or any(d not in {'0', '1', '2', '3'} for d in devices) or len(devices) != len(set(devices)):
        raise ValueError('explicit physical GPU list must be a subset of 0,1,2,3')


def select_records(records, *, max_fake_videos, max_real_videos, frames):
    if frames < 1:
        raise ValueError('frames must be positive')
    videos = defaultdict(dict)
    for r in records:
        videos[(r.method, r.video_id)][r.frame_path] = r
    methods = defaultdict(list)
    for key in videos:
        methods[key[0]].append(key)
    result = []
    for method in sorted(methods):
        keys = sorted(methods[method], key=lambda k: hashlib.sha256('::'.join(k).encode()).hexdigest())
        limit = max_real_videos if method == 'real' else max_fake_videos
        for key in keys if limit is None else keys[:limit]:
            available = sorted(videos[key].values(), key=lambda r: (r.frame_name, r.frame_path))
            count = min(frames, len(available))
            indices = [0] if count == 1 else [round(i * (len(available) - 1) / (count - 1)) for i in range(count)]
            result.extend(available[i] for i in indices)
    return result


def assert_disjoint(splits):
    for (a, aa), (b, bb) in itertools.combinations(splits.items(), 2):
        # FF++ and CDF source video ids need not be globally unique. Check
        # physical parent paths, plus method/video within each source domain.
        av = {(r.method, r.video_id) for r in aa}
        bv = {(r.method, r.video_id) for r in bb}
        if av & bv:
            raise ValueError(f'video overlap: {a}/{b}: {sorted(av & bv)[:3]}')
        if {r.frame_path for r in aa} & {r.frame_path for r in bb}:
            raise ValueError(f'physical frame overlap: {a}/{b}')


def epoch_draws(records, *, seed, epoch, count):
    groups = defaultdict(list)
    for index, r in enumerate(records):
        groups[method_family(r.method, r.label)].append(index)
    if set(groups) != {'camera', 'swap', 'reenact'}:
        raise ValueError('training requires all three known families')
    rng = random.Random(seed * 100003 + epoch)
    draws = [(rng.choice(groups[('camera', 'swap', 'reenact')[i % 3]]), TASKS[i % 5]) for i in range(count)]
    rng.shuffle(draws)
    return draws


def load_records(name):
    manifest = json.loads((PHASE / 'results/protocol/manifest.json').read_text())
    path = PHASE / manifest['splits'][name]['path']
    if sha256(path) != manifest['splits'][name]['sha256']:
        raise ValueError('changed physical record manifest')
    return [FrameRecord(**row) for row in json.loads(path.read_text())]


def language_bank():
    bank = load_language_bank(ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/configs/language_bank.yaml')
    return extended_language_bank(bank, ROOT / 'experiments/2026-09-23_e3_smolvlm_candidate/configs/diagnostics.yaml')


class ImageRecords(Dataset):
    def __init__(self, records, mean, std):
        self.records = records
        self.augment = False
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize(mean, std)])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        with Image.open(r.frame_path) as img:
            image = self.transform(img.convert('RGB'))
        return {'image': image, 'label': torch.tensor(r.label), 'method_id': r.method,
                'video_id': r.video_id, 'video_key': f'{r.method}::{r.video_id}',
                'image_path': r.frame_path, 'sample_id': r.sample_id}


class TrainingDataset(Dataset):
    def __init__(self, base, bank, *, seed, epoch, count):
        self.base = base
        self.draws = epoch_draws(base.records, seed=seed, epoch=epoch, count=count)
        self.sampler = SemanticTemplateSampler(bank, seed)
        self.epoch = epoch

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, index):
        record, task = self.draws[index]
        return self.sampler.render(self.base[record], task, 'train', self.epoch)


class ExhaustiveSuite(Dataset):
    """All frozen question/candidate-set combinations on each selected frame."""
    def __init__(self, base, bank, *, split, tasks=TASKS, seed=42):
        self.base_dataset = base
        self.sampler = SemanticTemplateSampler(bank, seed)
        self.split, self.epoch, self.mode, self.tasks = split, 0, 'suite', tasks
        self.slots = [(task, (q.name, c.name)) for task in tasks
                      for q in bank.task(task, split).questions
                      for c in bank.task(task, split).candidate_sets]

    def __len__(self):
        return len(self.base_dataset) * len(self.slots)

    def __getitem__(self, index):
        frame, slot = divmod(index, len(self.slots))
        task, (question, candidates) = self.slots[slot]
        out = self.sampler.render(self.base_dataset[frame], task, self.split, 0,
                                  force_question=question, force_candidate_set=candidates)
        out['question_pair_id'] = f'T4::{self.split}::{candidates}' if task == 'T4' else ''
        return out


def batch_texts(batch, bank):
    # IDs are lookup keys only. Semantic ids/labels never enter the model.
    questions = [bank.text(int(i)) for i in batch['question_text_id']]
    candidates = [[bank.text(int(i)) if int(i) >= 0 else '' for i in row]
                  for row in batch['candidate_text_ids']]
    return questions, candidates


def legacy_evaluation():
    path = ROOT / 'experiments/2026-09-22_semantic_visual_jev_e2/scripts/evaluate_semantic.py'
    spec = importlib.util.spec_from_file_location('e4_reused_e2_evaluation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
