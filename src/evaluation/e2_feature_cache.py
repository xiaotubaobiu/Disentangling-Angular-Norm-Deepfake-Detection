"""Provenance-checked float32 cache for frozen E2 evaluation visual features."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.datasets import FrameClassificationDataset, _load_frame, cv2_resize
from data.transforms import CLIP_MEAN, CLIP_STD, build_eval_transform


def _digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()


def _visual_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.clip.visual.state_dict().items()):
        raw = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(raw.dtype).encode())
        digest.update(json.dumps(list(raw.shape)).encode())
        digest.update(raw.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _record_identity(dataset) -> tuple[list[str], list[int], str]:
    if getattr(dataset, "augment", False):
        raise ValueError("training augmentations cannot use the frozen evaluation cache")
    records = getattr(dataset, "records", None)
    if records is None or len(records) != len(dataset):
        raise ValueError("feature cache requires ordered evaluation records")
    identities = []
    for record in records:
        identities.append({"sample_id": str(record.sample_id),
                           "frame_path": str(record.frame_path),
                           "label": int(record.label),
                           "method": str(record.method),
                           "video_id": str(record.video_id),
                           "landmark_path": str(getattr(record, "landmark_path", None))})
    first_identity = {}
    ids = []
    indices = []
    for index, identity in enumerate(identities):
        sample_id = identity["sample_id"]
        if sample_id in first_identity:
            if identity != first_identity[sample_id]:
                raise ValueError(f"conflicting records share sample ID {sample_id}")
        else:
            first_identity[sample_id] = identity
            ids.append(sample_id)
            indices.append(index)
    return ids, indices, _digest_json(identities)


def _image_stat_fingerprint(records) -> str:
    """Detect in-place source changes without rereading every image at open."""
    digest = hashlib.sha256()
    seen_paths = set()
    for record in records:
        path = str(record.frame_path)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        try:
            stat = Path(path).stat()
        except OSError as exc:
            raise ValueError(f"feature cache source image is unavailable: {path}") from exc
        payload = (path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                   stat.st_ino)
        digest.update(json.dumps(payload, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _expected_manifest(model, dataset, domain: str, cfg: dict) -> tuple[dict, list[str]]:
    ids, _indices, record_sha = _record_identity(dataset)
    clip_encode_image = getattr(type(model.clip), "encode_image", None)
    preprocess = {
        "model": cfg.get("model", {}),
        "dataset_image_size": cfg.get("dataset", {}).get("image_size"),
        "actual_image_size": getattr(dataset, "img_size", None),
        "clip_mean": CLIP_MEAN,
        "clip_std": CLIP_STD,
        "dataset_source": inspect.getsource(type(dataset)),
        "frame_source": inspect.getsource(_load_frame),
        "resize_source": inspect.getsource(cv2_resize),
        "eval_transform_source": inspect.getsource(build_eval_transform),
        "base_dataset_source": inspect.getsource(FrameClassificationDataset),
        "feature_extraction_source": inspect.getsource(type(model).extract_features),
        "clip_encode_image_source": (
            inspect.getsource(clip_encode_image) if clip_encode_image is not None else None
        ),
        "visual_forward_source": inspect.getsource(type(model.clip.visual).forward),
        "torch_version": torch.__version__,
    }
    return ({"version": 3, "domain": domain, "dtype": "float32",
             "shape": [len(ids), int(model.clip.visual.output_dim)],
             "index_rows": len(dataset),
             "visual_sha256": _visual_fingerprint(model),
             "preprocess_sha256": _digest_json(preprocess),
             "records_sha256": record_sha,
             "image_stat_sha256": _image_stat_fingerprint(dataset.records)}, ids)


class FrozenFeatureCache:
    def __init__(self, features, ids: list[str], manifest: dict):
        self.features = features
        self.row_for_id = {sample_id: index for index, sample_id in enumerate(ids)}
        self.manifest = manifest

    def lookup(self, sample_ids, device: torch.device) -> torch.Tensor:
        try:
            rows = [self.row_for_id[str(sample_id)] for sample_id in sample_ids]
        except KeyError as exc:
            raise ValueError(f"sample ID is absent from frozen feature cache: {exc}") from exc
        return torch.from_numpy(np.asarray(self.features[rows], dtype=np.float32)).to(
            device, non_blocking=True)


def open_feature_cache(model, dataset, cache_root: str | Path, *, domain: str,
                       cfg: dict) -> FrozenFeatureCache:
    root = Path(cache_root) / domain
    manifest_path, feature_path = root / "manifest.json", root / "features.npy"
    if not manifest_path.is_file() or not feature_path.is_file():
        raise ValueError(f"missing frozen feature cache for {domain}: {root}")
    expected, ids = _expected_manifest(model, dataset, domain, cfg)
    manifest = json.loads(manifest_path.read_text())
    if any(manifest.get(name) != value for name, value in expected.items()):
        raise ValueError(f"{domain} feature cache fingerprint disagrees with visual/preprocessing/records")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if features.dtype != np.float32 or list(features.shape) != expected["shape"]:
        raise ValueError(f"{domain} feature cache shape or dtype disagrees with manifest")
    if _digest_file(feature_path) != manifest.get("features_sha256"):
        raise ValueError(f"{domain} feature cache content fingerprint disagrees with manifest")
    return FrozenFeatureCache(features, ids, manifest)


@torch.no_grad()
def build_feature_cache(model, dataset, cache_root: str | Path, *, domain: str,
                        cfg: dict, device: torch.device, batch_size: int = 128,
                        num_workers: int = 0) -> FrozenFeatureCache:
    if batch_size < 1 or num_workers < 0:
        raise ValueError("feature cache loader sizes must be valid")
    expected, ids = _expected_manifest(model, dataset, domain, cfg)
    root = Path(cache_root) / domain
    manifest_path, feature_path = root / "manifest.json", root / "features.npy"
    if manifest_path.exists() or feature_path.exists():
        return open_feature_cache(model, dataset, cache_root, domain=domain, cfg=cfg)
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / "features.npy.tmp"
    mmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32,
                                     shape=tuple(expected["shape"]))
    options = dict(batch_size=batch_size, shuffle=False, num_workers=num_workers,
                   pin_memory=device.type == "cuda", persistent_workers=False)
    if num_workers:
        options.update(multiprocessing_context="spawn", prefetch_factor=4)
    _ids, unique_indices, _record_sha = _record_identity(dataset)
    loader = DataLoader(Subset(dataset, unique_indices), **options)
    model.eval()
    offset = 0
    for batch in loader:
        batch_ids = [str(value) for value in batch["sample_id"]]
        if batch_ids != ids[offset:offset + len(batch_ids)]:
            raise ValueError("feature cache loader changed ordered sample identities")
        features = model.extract_features(batch["image"].to(device, non_blocking=True))
        values = features.detach().float().cpu().numpy()
        if values.shape != (len(batch_ids), expected["shape"][1]) or not np.isfinite(values).all():
            raise ValueError("frozen visual model emitted malformed/nonfinite features")
        mmap[offset:offset + len(batch_ids)] = values
        offset += len(batch_ids)
    if offset != len(ids):
        raise ValueError("feature cache encoded an incomplete physical frame population")
    mmap.flush()
    del mmap
    os.replace(temporary, feature_path)
    expected["features_sha256"] = _digest_file(feature_path)
    manifest_tmp = root / "manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
    os.replace(manifest_tmp, manifest_path)
    return open_feature_cache(model, dataset, cache_root, domain=domain, cfg=cfg)
