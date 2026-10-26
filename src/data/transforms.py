from __future__ import annotations

import albumentations as A

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.45762756, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_rgb_augment(mean=IMAGENET_MEAN, std=IMAGENET_STD, seed=42) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=[-10, 10], p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.FancyPCA(),
            A.HueSaturationValue(),
        ], p=0.5),
        A.ImageCompression(quality_range=(40, 100), p=0.5),
        A.Normalize(mean=mean, std=std),
    ], additional_targets={"image_pair": "image"}, seed=seed)


def build_eval_transform(mean=IMAGENET_MEAN, std=IMAGENET_STD) -> A.Compose:
    return A.Compose([
        A.Normalize(mean=mean, std=std),
    ])


def build_shared_spatial_augment(seed=None) -> A.Compose:
    """Spatial/photometric augmentation shared across a (real, fake) pair.

    Deliberately excludes A.Normalize: the caller normalizes each crop
    afterwards via build_eval_transform, so applying Normalize here (and then
    dividing by 255) is a double-normalization bug. seed defaults to None so
    each call draws fresh randomness — passing a fixed seed makes every sample
    receive the identical augmentation.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=[-10, 10], p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            A.FancyPCA(),
            A.HueSaturationValue(),
        ], p=0.5),
        A.ImageCompression(quality_range=(40, 100), p=0.5),
    ], additional_targets={"image_pair": "image"}, seed=seed)


def apply_shared_transform_pair(image, image_pair):
    """Apply the SAME random spatial augmentation to both images of a pair.

    Returns un-normalized uint8-range images; normalization is applied per-crop
    by the caller. A fresh (unseeded) Compose is built each call so successive
    samples get diverse augmentations.
    """
    transformed = build_shared_spatial_augment()(image=image, image_pair=image_pair)
    return transformed["image"], transformed["image_pair"]
