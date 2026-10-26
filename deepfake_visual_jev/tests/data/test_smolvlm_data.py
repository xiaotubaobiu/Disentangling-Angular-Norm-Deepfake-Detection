from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoProcessor

from data.index_ffpp import FrameRecord
from data.smolvlm_data import NativeImageDataset, ExactShardSampler


PRETRAINED = Path(__file__).resolve().parents[2] / 'experiments/2026-09-23_e3_smolvlm_candidate/results/pretrained/smolvlm_256m'


def test_native_dataset_matches_official_pixels_and_preserves_record_identity(tmp_path):
    # Wrong color order, double normalization, or changed frame identity must fail.
    image = np.random.default_rng(8).integers(0, 256, (83, 107, 3), dtype=np.uint8)
    path = tmp_path / 'frame.png'
    cv2.imwrite(str(path), image)
    record = FrameRecord('real', 'source_1', 'frame.png', str(path), None, 0)
    dataset = NativeImageDataset([record], PRETRAINED, augment=False, seed=42)
    sample = dataset[0]
    processor = AutoProcessor.from_pretrained(PRETRAINED, local_files_only=True)
    rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (224, 224))
    expected = processor.image_processor(images=[[rgb]], do_image_splitting=False,
        size={'longest_edge': 512}, max_image_size={'longest_edge': 512},
        return_tensors='pt')['pixel_values'][0, 0]
    torch.testing.assert_close(sample['image'], expected, rtol=0, atol=0)
    assert sample['sample_id'] == record.sample_id
    assert sample['video_key'] == 'real::source_1'
    assert sample['label'].item() == 0
    assert sample['image'].shape == (3, 512, 512)


def test_validation_shards_cover_each_sample_once_without_padding():
    shards = [list(ExactShardSampler(range(11), rank=r, world_size=4)) for r in range(4)]
    assert shards == [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7]]
    assert [len(ExactShardSampler(range(11), rank=r, world_size=4)) for r in range(4)] == [3, 3, 3, 2]
