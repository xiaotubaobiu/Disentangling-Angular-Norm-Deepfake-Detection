"""E3 native preprocessing over the exact H1/E2 physical frame records."""
from __future__ import annotations

import numpy as np
from torch.utils.data import Sampler

from data.datasets import FrameClassificationDataset


class NativeImageDataset(FrameClassificationDataset):
    def __init__(self, records, pretrained_path, *, augment, seed):
        # Keep the source resize and augmentation; defer final normalization to
        # the official image processor. Zero mean/unit std yields RGB in [0,1].
        super().__init__(records, augment=augment, img_size=224,
                         normalize_mean=(0., 0., 0.), normalize_std=(1., 1., 1.), seed=seed)
        from transformers import AutoImageProcessor
        self.image_processor = AutoImageProcessor.from_pretrained(pretrained_path, local_files_only=True)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        rgb = sample['image'].permute(1, 2, 0).numpy()
        rgb = np.rint(rgb * 255).clip(0, 255).astype(np.uint8)
        sample['image'] = self.image_processor(
            images=[[rgb]], do_image_splitting=False,
            size={'longest_edge': 512}, max_image_size={'longest_edge': 512},
            return_tensors='pt')['pixel_values'][0, 0]
        return sample


class ExactShardSampler(Sampler):
    """Disjoint validation/cache shards; unlike DDP sampler, never pads."""
    def __init__(self, dataset, *, rank, world_size):
        if not 0 <= rank < world_size:
            raise ValueError('invalid rank/world size')
        self.size, self.rank, self.world_size = len(dataset), rank, world_size

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.size, self.world_size))
