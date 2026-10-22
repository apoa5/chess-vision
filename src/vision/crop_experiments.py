"""Experimental fixed-geometry crop transforms; production preprocessing is unchanged."""
from dataclasses import asdict, dataclass
import hashlib

import cv2
import numpy as np

from src.vision.grid_geometry import polygon_crop

MEAN = np.array([.485, .456, .406], np.float32)
STD = np.array([.229, .224, .225], np.float32)


@dataclass(frozen=True)
class CropSpec:
    name: str
    input_size: int = 224
    padding_squares: float = 2.
    detail_factor: float = 1.
    suppress_outer_context: bool = False

    def __post_init__(self):
        if type(self.input_size) is not int or self.input_size < 32:
            raise ValueError('Input size must be an integer >=32')
        if not 0 <= self.padding_squares <= 2 or not 0 < self.detail_factor <= 1:
            raise ValueError('Experiments may only remove context/detail from frozen crops')
        if self.suppress_outer_context and self.padding_squares != 2:
            raise ValueError('Context suppression preserves the original five-square canvas')

    def metadata(self):
        return asdict(self)


def transform_crop(image, polygon, spec):
    """Subcrop stored pixels, never rerun localization/warp/grid or rotate appearance.

    Detail degradation changes RGB only, preserving the original geometric mask.
    Suppression retains three-square RGB context on the five-square canvas, so
    target scale/input size stay fixed while outer neighboring information changes.
    """
    polygon = np.asarray(polygon, np.float32)
    image = image.copy()
    if spec.padding_squares != 2 or spec.suppress_outer_context:
        padding = 100 if spec.suppress_outer_context else round(100*spec.padding_squares)
        crop, _, _, bounds, local, _ = polygon_crop(image, np.full(image.shape[:2], 255, np.uint8), polygon, padding)
        if spec.suppress_outer_context:
            x0, y0, x1, y1 = bounds
            keep = np.zeros_like(image)
            keep[max(0, y0):min(image.shape[0], y1), max(0, x0):min(image.shape[1], x1)] = image[max(0, y0):min(image.shape[0], y1), max(0, x0):min(image.shape[1], x1)]
            image = keep
        else:
            image, polygon = crop, np.float32(local)
    if spec.detail_factor != 1:
        height, width = image.shape[:2]
        small = cv2.resize(image, (max(1, round(width*spec.detail_factor)), max(1, round(height*spec.detail_factor))), interpolation=cv2.INTER_AREA)
        image = cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return image, polygon


def packed_input(image, polygon, spec):
    """Lossless uint8 cache of resized RGB and binary target mask before augmentation."""
    image, polygon = transform_crop(image, polygon, spec)
    mask = np.zeros(image.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    rgb = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (spec.input_size, spec.input_size), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (spec.input_size, spec.input_size), interpolation=cv2.INTER_NEAREST)
    return np.concatenate([rgb.transpose(2, 0, 1), mask[None]], axis=0)


def tensor_from_packed(packed, *, rng=None, augmentation=None):
    """Same photometric augmentation and normalization as the frozen implementation."""
    image = packed[:3].transpose(1, 2, 0).astype(np.float32)/255
    augmentation = augmentation or {}
    if rng is not None:
        brightness = augmentation.get('brightness', .10)
        contrast = augmentation.get('contrast', .10)
        image = (image-.5)*rng.uniform(1-contrast, 1+contrast)+.5+rng.uniform(-brightness, brightness)
        if rng.random() < augmentation.get('blur_probability', .15):
            image = cv2.GaussianBlur(image, (3, 3), rng.uniform(.2, .7))
        if rng.random() < augmentation.get('noise_probability', .20):
            image += rng.normal(0, augmentation.get('noise_std', .01), image.shape).astype(np.float32)
        image = np.clip(image, 0, 1)
    rgb = (image-MEAN)/STD
    return np.ascontiguousarray(np.concatenate([rgb.transpose(2, 0, 1), packed[3:4].astype(np.float32)], axis=0), dtype=np.float32)


class CachedCropDataset:
    """Position-split memmap cache; augmentation seed matches ContextSquareDataset."""
    def __init__(self, rows, cache, *, split, seed=42, augmentation=None):
        if split not in ('train', 'val', 'test') or len(rows) != len(cache) or any(r['split'] != split for r in rows):
            raise ValueError('Cache rows must match one frozen split')
        if any(r.get('geometry_review_required') or r.get('quality_flags') for r in rows):
            raise ValueError('Unapproved targets cannot enter experiments')
        self.rows, self.cache, self.split = rows, cache, split
        self.seed, self.epoch, self.augmentation = seed, 0, augmentation or {}

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        rng = None
        if self.split == 'train':
            digest = hashlib.sha256(f'{self.seed}:{self.epoch}:{row["crop_id"]}'.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
        return torch.from_numpy(tensor_from_packed(self.cache[index], rng=rng, augmentation=self.augmentation)), row['class_id']
