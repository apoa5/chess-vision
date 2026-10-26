"""C.3 frozen historical geometry and paired-representation data helpers.

Offline only. Reuses C.1 extraction, B.2 packing and original grouped rows.
"""
import hashlib
import json
from pathlib import Path
import re

import cv2
import numpy as np

from src.vision.crop_experiments import CachedCropDataset, CropSpec, packed_input
from src.vision.dataset import chess_square_at, review_approved
from src.vision.geometry_audit import context_transform, project
from src.vision.input_representations import RepresentationSettings, height_envelope, raw_letterbox


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def historical_inputs(config):
    if config['final_holdout'] or config['historical_test_used'] or config['D3c_access_for_training_or_tuning'] or config['D2_used_for_training_or_selection']:
        raise ValueError('C.3 only allows original historical train/validation')
    manifest=read(config['manifest'])
    if not manifest['training_ready'] or manifest['split_status']!='position_grouped' or manifest['preview_only']:
        raise ValueError('Approved position-grouped manifest required')
    prepared={r['image_id']:r for r in read(config['prepared_boards'])}
    images,positions,reviews=(read(config[k]) for k in ('images','positions','reviews'))
    from scripts.run_crop_experiments import leakage_audit
    leakage=leakage_audit(manifest['results'],prepared,images,positions)
    rows={s:[r for r in manifest['results'] if r['split']==s] for s in ('train','val')}
    if len(rows['train'])!=7168 or len(rows['val'])!=1600:raise ValueError('Frozen group population changed')
    by_source={}
    for r in rows['train']+rows['val']:
        source=r['source_image'];p=prepared[source]
        if not re.fullmatch(r'position_\d{3}_var_\d{2}\.jpg',source):raise ValueError('Nonhistorical capture prohibited')
        path=Path(images[source]['path'])
        if path.parent!=Path('data/raw') or path.name!=source:raise ValueError('Unexpected historical source path')
        if not review_approved(p,reviews.get(source)):raise ValueError('Stale board geometry/label review')
        if r['board_review_signature']!=p['review_signature'] or r['geometry_review_required'] or r['quality_flags']:
            raise ValueError('Unapproved target geometry')
        if r['chess_square']!=chess_square_at(r['row'],r['col'],p['label_rotation']):raise ValueError('Semantic square mapping mismatch')
        if r['class_name']!=manifest['classes'][r['class_id']]['name']:raise ValueError('Class order mismatch')
        by_source.setdefault(source,[]).append(r)
    if any(len(rs)!=64 or len({r['chess_square'] for r in rs})!=64 for rs in by_source.values()):
        raise ValueError('Complete 64-square source boards required')
    if len({r['position_id'] for r in rows['train']})!=12 or len({r['position_id'] for r in rows['val']})!=3:
        raise ValueError('Position groups changed')
    return manifest,rows,prepared,images,leakage


def source_geometry(prepared):
    polygons=np.asarray(prepared['grid_geometry']['square_polygons'],np.float32)
    if polygons.shape!=(64,4,2) or not np.isfinite(polygons).all():raise ValueError('Finite frozen fitted polygons required')
    # Manual historical records need not contain a source_corners field. All
    # approved records already preserve these exact reviewed source polygons.
    raw=np.asarray(prepared['grid_geometry'].get('source_square_polygons'),np.float32)
    if raw.shape!=(64,4,2) or not np.isfinite(raw).all():
        raise ValueError('Frozen reviewed source polygons unavailable; do not infer new geometry')
    quad=np.float32([raw[0,0],raw[7,1],raw[63,2],raw[56,3]])
    if not cv2.isContourConvex(quad) or cv2.contourArea(quad)<16:raise ValueError('Invalid raw board geometry')
    return raw,quad


def native_input(image,row,prepared,settings=None,geometry=None):
    settings=settings or RepresentationSettings()
    raw,quad=source_geometry(prepared) if geometry is None else geometry
    index=row['row']*8+row['col'];target=raw[index]
    canvas=np.asarray(row['target_polygon'])+np.asarray(row['bounds'][:2])
    if not np.allclose(canvas,prepared['grid_geometry']['square_polygons'][index],atol=1e-3):
        raise ValueError('Stored crop and frozen grid target differ')
    if not cv2.isContourConvex(np.float32(target)):raise ValueError('Nonconvex target')
    envelope,metadata=height_envelope(quad,target,image.shape[1],image.shape[0],settings.height_squares)
    result=raw_letterbox(image,target,envelope,settings)
    if result['packed'][3].sum()==0:raise ValueError('No target indicator after native extraction')
    result.update(height_proxy=metadata,source_target_polygon=target)
    return result


def b2_cache(split):
    root=Path('outputs/evaluation/crop_b2_20261018_run1')
    return root/'cache/context1_224'/(split+'.npy'),read(root/'cache_report.json')['context1_224'][split]


def make_native_cache(rows,prepared,images,settings,expected_hashes=None):
    """Bounded uint8 RAM cache only; no duplicate crop files written."""
    cache=np.empty((len(rows),4,settings.size,settings.size),np.uint8)
    source=None;image=None;geometry=None
    for i,row in enumerate(rows):
        if source!=row['source_image']:
            source=row['source_image'];image=cv2.imread(images[source]['path']);geometry=source_geometry(prepared[source])
            if image is None:raise ValueError('Missing native source: '+source)
        result=native_input(image,row,prepared[source],settings,geometry)
        cache[i]=result['packed']
        if expected_hashes is not None and hashlib.sha256(cache[i].tobytes()).hexdigest()!=expected_hashes[row['crop_id']]['taller_sha256']:
            raise ValueError('Native representation differs from frozen preflight')
    cache.flags.writeable=False
    return cache


class TrackedDataset(CachedCropDataset):
    def begin_epoch(self,epoch):
        self.set_epoch(epoch);self.order=[]

    def __getitem__(self,index):
        if hasattr(self,'order'):self.order.append(int(index))
        return super().__getitem__(index)

    def order_sha256(self):
        return hashlib.sha256(np.asarray(self.order,dtype='<i4').tobytes()).hexdigest()


def epoch_metrics(metrics):
    cm=np.asarray(metrics['confusion_matrix']);occupied=int(cm[1:].sum())
    return dict(metrics,occupied_count=occupied,occupied_correct=int(cm.diagonal()[1:].sum()),
                occupied_accuracy=float(cm.diagonal()[1:].sum()/occupied),
                empty_count=int(cm[0].sum()),empty_false_positives=int(cm[0,1:].sum()),
                empty_false_positive_rate=float(cm[0,1:].sum()/cm[0].sum()))


def checkpoint_key(metrics,epoch):
    return (-metrics['occupied_accuracy'],metrics['empty_false_positives'],metrics['loss'],epoch)


def state_hash(model):
    digest=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        digest.update(name.encode());digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode());digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
