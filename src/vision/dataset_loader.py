"""Context RGB + target-square-mask tensors, without chess-coordinate/FEN inputs."""
import hashlib
import json
from pathlib import Path
import cv2
import numpy as np
from src.utils.image_io import load_image


class ContextSquareDataset:
    """Map-style PyTorch DataLoader-compatible dataset; returns (4xHxW tensor, class ID).

    First three channels: ImageNet-normalized RGB. Fourth: binary target mask.
    Randomness is a pure function of seed, epoch, crop ID, independent of workers.
    """
    def __init__(self, manifest_path, split, *, input_size=224, seed=42, augmentation=None, tensor=True):
        manifest = json.loads(Path(manifest_path).read_text())
        if split not in ('train','val','test') or manifest.get('split_status')!='position_grouped' or manifest.get('preview_only'):
            raise ValueError('A position-split, approved manifest and valid split are required.')
        if type(input_size) is not int or input_size<=0:
            raise ValueError('Input size must be positive.')
        assignment={}
        for row in manifest['results']:
            for key in [('position',row['position_id']),('source',row['source_image'])]:
                if key in assignment and assignment[key]!=row['split']:
                    raise ValueError('Manifest leaks a position or source image across splits.')
                assignment[key]=row['split']
        self.rows = [r for r in manifest['results'] if r['split']==split]
        if not self.rows:
            raise ValueError('Split is empty.')
        if any(r.get('geometry_review_required') or r.get('quality_flags') for r in self.rows):
            raise ValueError('Unapproved or flagged samples cannot be loaded.')
        self.split,self.input_size,self.seed,self.epoch,self.tensor=split,input_size,seed,0,tensor
        self.augmentation = augmentation or {}

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __getitem__(self,index):
        row=self.rows[index]
        image=load_image(row['crop_path'])
        rng = None
        if self.split == 'train':
            digest = hashlib.sha256(f'{self.seed}:{self.epoch}:{row["crop_id"]}'.encode()).digest()
            rng = np.random.default_rng(int.from_bytes(digest[:8], 'big'))
        array = preprocess_context_crop(image, row['target_polygon'], self.input_size,
                                        augmentation=self.augmentation, rng=rng)
        if self.tensor:
            import torch
            return torch.from_numpy(array),row['class_id']
        return array,row['class_id']


def preprocess_context_crop(image, target_polygon, input_size=224, *, augmentation=None, rng=None):
    """Shared training/inference contract: BGR context + local target polygon."""
    from src.utils.image_io import validate_image
    validate_image(image)
    polygon = np.asarray(target_polygon, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3 or polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError('Expected BGR image and finite four-vertex target polygon.')
    if type(input_size) is not int or input_size <= 0:
        raise ValueError('Input size must be positive.')
    if (polygon < 0).any() or (polygon[:,0] >= image.shape[1]).any() or (polygon[:,1] >= image.shape[0]).any() or not cv2.isContourConvex(polygon):
        raise ValueError('Target polygon must be convex and within the crop.')
    augmentation = augmentation or {}
    target_polygon = polygon
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask=np.zeros(image.shape[:2],np.uint8)
    cv2.fillConvexPoly(mask,np.rint(target_polygon).astype(np.int32),1)
    image=cv2.resize(image,(input_size,input_size),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    mask=cv2.resize(mask,(input_size,input_size),interpolation=cv2.INTER_NEAREST).astype(np.float32)
    if rng is not None:
        brightness=augmentation.get('brightness',.10);contrast=augmentation.get('contrast',.10)
        image=(image-.5)*rng.uniform(1-contrast,1+contrast)+.5+rng.uniform(-brightness,brightness)
        if rng.random()<augmentation.get('blur_probability',.15):
            image=cv2.GaussianBlur(image,(3,3),rng.uniform(.2,.7))
        if rng.random()<augmentation.get('noise_probability',.20):
            image+=rng.normal(0,augmentation.get('noise_std',.01),image.shape).astype(np.float32)
        image=np.clip(image,0,1)
    rgb=(image-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
    array=np.ascontiguousarray(np.concatenate([rgb.transpose(2,0,1),mask[None]],axis=0),dtype=np.float32)
    return array
