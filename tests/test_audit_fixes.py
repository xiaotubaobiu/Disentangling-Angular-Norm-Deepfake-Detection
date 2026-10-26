"""Regression tests for bugs found during the full-project audit.

- diagnostics.delta_auc_per_method: must pair raw/direction AUC WITHIN the same
  backbone (previously produced a 3x3 cross-backbone cartesian product).
- crops.crop_region: must not crash on degenerate/empty boxes.
- transforms.build_shared_spatial_augment: must NOT normalize (so the caller's
  per-crop normalize is not a double-normalization) and must vary across calls.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from diagnosis.diagnostics import delta_auc_per_method
from diagnosis.hypothesis_summary import summarize_h3
from data.crops import crop_region
from data.transforms import (
    apply_shared_transform_pair,
    build_shared_spatial_augment,
)


def _method_df_two_backbones():
    rows = []
    for bb, base in [("clip", 0.80), ("dinov2", 0.90)]:
        for method in ["simswap", "fsgan"]:
            for seed in [7, 42]:
                rows.append({"domain": "cdf", "backbone": bb, "seed": seed,
                             "method_id": method, "infer_mode": "raw",
                             "auc_method": base})
                rows.append({"domain": "cdf", "backbone": bb, "seed": seed,
                             "method_id": method, "infer_mode": "direction",
                             "auc_method": base + 0.02})
    return pd.DataFrame(rows)


def test_delta_auc_pairs_within_backbone():
    df = _method_df_two_backbones()
    out = delta_auc_per_method(df, "cdf")
    # 2 backbones x 2 methods x 2 seeds = 8 rows, NOT 8*2 (cartesian).
    assert len(out) == 8, f"expected 8 paired rows, got {len(out)} (cartesian?)"
    assert "backbone" in out.columns
    assert out.groupby(["backbone", "method_id", "seed"]).size().max() == 1
    assert np.allclose(out["delta_auc"].to_numpy(), 0.02)


def test_crop_region_degenerate_box():
    img = np.zeros((100, 120, 3), dtype=np.uint8)
    for box in [(50, 50, 50, 50), (80, 80, 10, 10), (-5, -5, 3, 3), (200, 200, 300, 300)]:
        out = crop_region(img, box, output_size=224)
        assert out.shape == (224, 224, 3)


def test_shared_spatial_augment_not_normalized():
    aug = build_shared_spatial_augment(seed=0)
    img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    out = aug(image=img)["image"]
    assert out.max() > 1.5  # normalized output would be ~<3 std units, not raw range


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


def test_h3_seed_noise_is_within_backbone():
    """H3 seed-noise floor must be the WITHIN-backbone seed std, not the
    backbone-pooled std. We build 2 backbones whose per-method means differ a
    lot across backbones but are near-constant across seeds."""
    rows = []
    for bb, base in [("clip", 0.50), ("dinov2", 0.85)]:  # big cross-backbone gap
        for method in ["simswap", "fsgan"]:
            for i, seed in enumerate([7, 42, 123, 999, 2048]):
                rows.append({"domain": "cdf", "infer_mode": "direction",
                             "method_id": method, "backbone": bb, "seed": seed,
                             "fake_score_mean": base + 0.001 * i})  # tiny seed noise
    method_df = pd.DataFrame(rows)
    aggregate_df = pd.DataFrame([{"infer_mode": "direction", "D_method_cdf_mean": 0.05}])

    h = summarize_h3(aggregate_df, method_df)
    seed_noise = h["evidence"]["direction_seed_noise_floor"]
    # Within-backbone std of [base, base+.001,...] ~= 0.00158; pooled (mixing
    # 0.50 and 0.85) would be ~0.175. Assert we got the small within-backbone value.
    assert seed_noise < 0.01, f"seed noise {seed_noise} looks backbone-pooled"

