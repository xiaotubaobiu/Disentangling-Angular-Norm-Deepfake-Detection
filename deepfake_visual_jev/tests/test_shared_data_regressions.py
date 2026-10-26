import numpy as np
from data.crops import crop_region
from data.transforms import apply_shared_transform_pair, build_shared_spatial_augment

def test_crop_region_degenerate_box():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    for box in [(50, 50, 50, 50), (80, 80, 10, 10), (-5, -5, 3, 3), (200, 200, 300, 300)]:
        out = crop_region(img, box, output_size=224)
        assert out.shape == (224, 224, 3)

def test_shared_spatial_augment_not_normalized():
    aug = build_shared_spatial_augment(seed=0)
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = aug(image=img)["image"]
    assert out.max() > 1.5

def test_shared_transform_pair_varies_across_calls():
    rng_img = (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8)
    pair = rng_img.copy()
    outs = []
    for _ in range(8):
        a, b = apply_shared_transform_pair(rng_img.copy(), pair.copy())
        assert a.shape == b.shape
        outs.append(a.astype(np.int64))
    all_identical = all(np.array_equal(outs[0], o) for o in outs[1:])
    assert not all_identical, "augmentations are frozen/identical across calls"
