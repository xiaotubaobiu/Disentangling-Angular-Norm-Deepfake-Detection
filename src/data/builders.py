from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from data.constants import ALL_METHODS
from data.index_ffpp import load_json_index, load_dfd_json_index
from data.index_ffpp import (
    index_train_aligned_triplets,
)
from data.sampling import sample_uniform_frame_indices
from data.datasets import FrameClassificationDataset, AlignedTripletDataset, BgFaceTripletDataset
from data.semantic_dataset import TASKS, SemanticDataset, SemanticTemplateSampler


def _worker_init_fn(worker_id):
    """每个 DataLoader worker 用独立但确定的种子初始化随机状态。"""
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def _dataloader_kwargs(num_workers, prefetch_factor=4):
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "worker_init_fn": _worker_init_fn,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = prefetch_factor
        kwargs["multiprocessing_context"] = "spawn"
    return kwargs


def _get_normalization(cfg):
    model_name = cfg.get("model", {}).get("name", "")
    if "clip" in model_name:
        from data.transforms import CLIP_MEAN, CLIP_STD
        return CLIP_MEAN, CLIP_STD
    return None, None


def _subsample_records_to_frames(records, frames_per_video=8):
    from collections import defaultdict
    by_video: dict[str, list] = defaultdict(list)
    for r in records:
        by_video[r.pair_id].append(r)
    result = []
    for pair_id, frames in by_video.items():
        n = min(frames_per_video, len(frames))
        indices = sample_uniform_frame_indices(len(frames), n)
        for i in indices:
            result.append(frames[i])
    return result


def _deduplicate_videos(records):
    """Keep all frames for each unique pair_id from first-encountered method."""
    seen = set()
    claimed_by = {}
    result = []
    for r in records:
        if r.pair_id not in seen:
            seen.add(r.pair_id)
            claimed_by[r.pair_id] = r.method
        if claimed_by.get(r.pair_id) == r.method:
            result.append(r)
    return result


def build_train_dataset(cfg):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    max_videos = cfg["dataset"]["train_videos_per_method"]
    frames_per_video = cfg["dataset"]["frames_per_video"]
    seed = cfg["train"].get("seed", 42)
    require_triplets = cfg.get("data", {}).get("require_aligned_triplets", False)
    norm_mean, norm_std = _get_normalization(cfg)

    if require_triplets:
        all_triplets = []
        for method in methods:
            triplets = index_train_aligned_triplets(root, method, max_videos)
            all_triplets.extend(triplets)
        dataset = AlignedTripletDataset(all_triplets, augment=True)
    else:
        # Collect fake records from JSON: first max_videos per method
        fake_records = []
        for method in methods:
            recs = load_json_index(root, method, "ff", "train", 1, max_videos=max_videos)
            fake_records.extend(recs)
        fake_by_video = _subsample_records_to_frames(fake_records, frames_per_video)

        # Collect real records from JSON: FF++ original, aligned range
        real_records = []
        for method in methods:
            recs = load_json_index(root, method, "ff", "train", 0, max_videos=max_videos)
            real_records.extend(recs)
        real_deduped = _deduplicate_videos(real_records)
        real_by_video = _subsample_records_to_frames(real_deduped, frames_per_video)
        target_real_count = len(fake_by_video)
        rng = random.Random(42)
        real_balanced = []
        while len(real_balanced) < target_real_count:
            real_balanced.extend(rng.sample(real_by_video, min(len(real_by_video), target_real_count - len(real_balanced))))

        all_records = real_balanced[:target_real_count] + fake_by_video
        dataset = FrameClassificationDataset(all_records, augment=True,
                                            normalize_mean=norm_mean, normalize_std=norm_std,
                                            seed=seed)
    return dataset


def build_train_loader(cfg, distributed=True):
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    prefetch_factor = cfg["train"].get("prefetch_factor", 4)
    seed = cfg["train"].get("seed", 42)
    dataset = build_train_dataset(cfg)

    g = torch.Generator()
    g.manual_seed(seed)
    sampler = DistributedSampler(dataset, shuffle=True, seed=seed) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(sampler is None),
                      drop_last=True, generator=g,
                      **_dataloader_kwargs(num_workers, prefetch_factor))


def build_eval_dataset(cfg, domain="ffpp"):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    seed = cfg["train"].get("seed", 42)
    norm_mean, norm_std = _get_normalization(cfg)

    if domain == "dfd":
        real_records = load_dfd_json_index(root, "test", 0)
        fake_records = load_dfd_json_index(root, "test", 1)
        all_records = real_records + fake_records
        if not all_records:
            raise FileNotFoundError(
                f"No DFD records found under {root}/dataset_json/DeepFakeDetection.json; run scripts/prepare_dfd_df40.py first."
            )
    else:
        test_domain = "ff" if domain == "ffpp" else "cdf"

        real_records = []
        for method in methods:
            recs = load_json_index(root, method, test_domain, "test", 0)
            real_records.extend(recs)
        real_deduped = _deduplicate_videos(real_records)

        fake_records = []
        for method in methods:
            recs = load_json_index(root, method, test_domain, "test", 1)
            fake_records.extend(recs)

        all_records = real_deduped + fake_records

    dataset = FrameClassificationDataset(all_records, augment=False,
                                        normalize_mean=norm_mean, normalize_std=norm_std,
                                        seed=seed)
    return dataset


def build_eval_loader(cfg, domain="ffpp", distributed=True):
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    prefetch_factor = cfg["train"].get("prefetch_factor", 4)
    seed = cfg["train"].get("seed", 42)
    dataset = build_eval_dataset(cfg, domain=domain)
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, drop_last=False, generator=g,
                      **_dataloader_kwargs(num_workers, prefetch_factor))


def build_val_dataset(cfg):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    val_videos = cfg["dataset"].get("val_videos_per_method", 100)
    frames_per_video = cfg["dataset"]["frames_per_video"]
    seed = cfg["train"].get("seed", 42)
    norm_mean, norm_std = _get_normalization(cfg)

    # Val fake: videos [max_videos : max_videos + val_videos] per method
    max_videos = cfg["dataset"]["train_videos_per_method"]
    fake_records = []
    for method in methods:
        recs = load_json_index(root, method, "ff", "train", 1,
                               video_range=(max_videos, max_videos + val_videos))
        fake_records.extend(recs)
    fake_by_video = _subsample_records_to_frames(fake_records, frames_per_video)

    # Val real: FF++ original, same range — NO overlap with training real
    real_records = []
    for method in methods:
        recs = load_json_index(root, method, "ff", "train", 0,
                               video_range=(max_videos, max_videos + val_videos))
        real_records.extend(recs)
    real_deduped = _deduplicate_videos(real_records)
    real_by_video = _subsample_records_to_frames(real_deduped, frames_per_video)
    target_real_count = len(fake_by_video)
    rng = random.Random(42)
    real_balanced = []
    while len(real_balanced) < target_real_count:
        real_balanced.extend(rng.sample(real_by_video, min(len(real_by_video), target_real_count - len(real_balanced))))

    all_records = real_balanced[:target_real_count] + fake_by_video
    dataset = FrameClassificationDataset(all_records, augment=False,
                                        normalize_mean=norm_mean, normalize_std=norm_std,
                                        seed=seed)
    return dataset


def build_val_loader(cfg, distributed=True):
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    prefetch_factor = cfg["train"].get("prefetch_factor", 4)
    seed = cfg["train"].get("seed", 42)
    dataset = build_val_dataset(cfg)
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=False, drop_last=False, generator=g,
                      **_dataloader_kwargs(num_workers, prefetch_factor))


def build_semantic_train_loader(cfg, language_bank, distributed=True):
    """Build the dynamic five-task training loader over the H1 dataset."""
    seed = cfg["train"].get("seed", 42)
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    prefetch_factor = cfg["train"].get("prefetch_factor", 4)
    base_dataset = build_train_dataset(cfg)
    dataset = SemanticDataset(
        base_dataset,
        SemanticTemplateSampler(language_bank, seed=seed),
        mode="train",
        tasks=tuple(cfg.get("language", {}).get("train_tasks", TASKS)),
        split="train",
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = (
        DistributedSampler(dataset, shuffle=True, seed=seed)
        if distributed
        else None
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        generator=generator,
        **_dataloader_kwargs(num_workers, prefetch_factor),
    )


def build_semantic_val_loader(
    cfg, language_bank, distributed=True, split="train", tasks=TASKS
):
    """Build the deterministic FF++ validation task suite."""
    return _build_semantic_suite_loader(
        cfg,
        language_bank,
        build_val_dataset(cfg),
        tasks=tasks,
        split=split,
        distributed=distributed,
    )


def build_semantic_eval_loader(
    cfg,
    language_bank,
    domain="ffpp",
    distributed=True,
    split="heldout",
    tasks=TASKS,
):
    """Build a deterministic domain suite, omitting undefined DFD families."""
    tasks = tuple(tasks)
    if domain == "dfd":
        tasks = tuple(task for task in tasks if task in {"T1", "T2", "T3"})
    return _build_semantic_suite_loader(
        cfg,
        language_bank,
        build_eval_dataset(cfg, domain=domain),
        tasks=tasks,
        split=split,
        distributed=distributed,
    )


def _build_semantic_suite_loader(
    cfg, language_bank, base_dataset, *, tasks, split, distributed
):
    seed = cfg["train"].get("seed", 42)
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    prefetch_factor = cfg["train"].get("prefetch_factor", 4)
    dataset = SemanticDataset(
        base_dataset,
        SemanticTemplateSampler(language_bank, seed=seed),
        mode="suite",
        tasks=tuple(tasks),
        split=split,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        generator=generator,
        **_dataloader_kwargs(num_workers, prefetch_factor),
    )


def build_external_eval_loader(cfg, records, distributed=True, batch_size=None, num_workers=None):
    seed = cfg["train"].get("seed", 42)
    prefetch_factor = cfg["train"].get("prefetch_factor", 4)
    norm_mean, norm_std = _get_normalization(cfg)
    dataset = FrameClassificationDataset(records, augment=False,
                                         normalize_mean=norm_mean, normalize_std=norm_std,
                                         seed=seed)
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = DistributedSampler(dataset, shuffle=False) if distributed else None
    return DataLoader(dataset, batch_size=batch_size or cfg["train"].get("per_gpu_batch", 32), sampler=sampler,
                      shuffle=False, drop_last=False, generator=g,
                      **_dataloader_kwargs(num_workers if num_workers is not None else cfg["train"].get("num_workers", 4), prefetch_factor))


def build_bgface_train_loader(cfg, distributed=True):
    root = cfg["dataset"]["root"]
    methods = cfg["dataset"].get("methods") or ALL_METHODS
    max_videos = cfg["dataset"]["train_videos_per_method"]
    batch_size = cfg["train"].get("per_gpu_batch", 32)
    num_workers = cfg["train"].get("num_workers", 4)
    prefetch_factor = cfg["train"].get("prefetch_factor", 4)
    num_bg_patches = cfg.get("loss", {}).get("num_bg_patches", 4)
    from data.transforms import CLIP_MEAN, CLIP_STD

    all_triplets = []
    for method in methods:
        triplets = index_train_aligned_triplets(root, method, max_videos)
        all_triplets.extend(triplets)

    dataset = BgFaceTripletDataset(
        all_triplets, augment=True,
        num_bg_patches=num_bg_patches,
        normalize_mean=CLIP_MEAN, normalize_std=CLIP_STD,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      shuffle=(sampler is None), drop_last=True,
                      **_dataloader_kwargs(num_workers, prefetch_factor))
