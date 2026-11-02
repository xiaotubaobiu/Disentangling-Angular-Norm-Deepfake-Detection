"""DF40 official image augmentation, pinned to Albumentations 1.3.1.

Source: third_party/provenance.json. Input already resized to224 by OpenCV
INTER_CUBIC, matching DF40 load_rgb. Preserve official IsotropicResize even
though it is normally a no-op after square resizing. Pretrained normalization
is model-specific and is applied by trainer.image_tensor afterwards.
"""
import importlib.util
from functools import lru_cache
from pathlib import Path
import random
import numpy as np
import albumentations as A
import cv2


@lru_cache(maxsize=1)
def build_transform():
    if A.__version__!='1.3.1':
        raise RuntimeError('Official DF40 augmentation requires albumentations==1.3.1')
    path=Path(__file__).resolve().parents[1]/'third_party/df40/albu.py'
    spec=importlib.util.spec_from_file_location('conference21_df40_albu',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    Resize=module.IsotropicResize
    return A.Compose([
      A.HorizontalFlip(p=.5), A.Rotate(limit=[-10,10],p=.5),
      A.GaussianBlur(blur_limit=[3,7],p=.5),
      A.OneOf([
        Resize(max_side=224,interpolation_down=cv2.INTER_AREA,interpolation_up=cv2.INTER_CUBIC),
        Resize(max_side=224,interpolation_down=cv2.INTER_AREA,interpolation_up=cv2.INTER_LINEAR),
        Resize(max_side=224,interpolation_down=cv2.INTER_LINEAR,interpolation_up=cv2.INTER_LINEAR),
      ],p=1),
      A.OneOf([A.RandomBrightnessContrast(brightness_limit=[-.1,.1],contrast_limit=[-.1,.1]),
        A.FancyPCA(),A.HueSaturationValue()],p=.5),
      A.ImageCompression(quality_lower=40,quality_upper=100,p=.5),
    ])


def augment(image,seed):
    transform=build_transform()
    py_state=random.getstate();np_state=np.random.get_state()
    try:
        random.seed(seed);np.random.seed(seed)
        return transform(image=image)['image']
    finally:
        random.setstate(py_state);np.random.set_state(np_state)
