"""Same-frame geometry, baseline polygon crops and bounded local recognition sessions."""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import math
import time
import uuid
from io import BytesIO
from PIL import Image
import cv2
import numpy as np
from src.chess.labels import CLASS_NAMES
from src.scanning.replay import _geometry
from src.vision.perspective import order_corners,warp_board_with_context
from src.vision.grid_geometry import polygon_crop
ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'models/checkpoints/piece_classifier_best.pt'
VERSION='scan_piece_observation_v1'

@lru_cache(maxsize=1)
def model_metadata():
    baseline=json.loads((ROOT/'configs/scanning_baseline.json').read_text())
    digest=hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()
    if digest!=baseline['sha256']['models/checkpoints/piece_classifier_best.pt']:raise ValueError('Baseline checkpoint hash changed')
    preprocessing=hashlib.sha256((ROOT/'src/vision/dataset_loader.py').read_bytes()).hexdigest()
    if preprocessing!=baseline['sha256']['src/vision/dataset_loader.py']:raise ValueError('Baseline preprocessing hash changed')
    return {'checkpoint_sha256':digest,'architecture':baseline['architecture'],'label_version':baseline['label_version'],'class_names':list(CLASS_NAMES),'input_channels':4,'input_size':224,'calibration':'uncalibrated','preprocessing_sha256':preprocessing}

@lru_cache(maxsize=1)
def classifier():
    from src.vision.inference import PieceClassifier
    model_metadata()
    return PieceClassifier(CHECKPOINT,batch_size=16,cpu_threads=4)

class SessionRegistry:
    """Access serialized by the server's session lock; no growing frame history."""
    def __init__(self):self.sessions={}
    def create(self):
        now=time.monotonic();self.sessions={k:v for k,v in self.sessions.items() if now-v['touched']<1800}
        metadata=model_metadata()
        if len(self.sessions)>=32:
            oldest=min(self.sessions,key=lambda key:self.sessions[key]['touched'])
            del self.sessions[oldest]
        identity=str(uuid.uuid4());self.sessions[identity]={'epoch':0,'timestamp':-1.,'geometry_version':-1,'touched':now}
        return {'session_id':identity,'schema_version':VERSION,'model':metadata}
    def validate(self,meta):
        session=self.sessions.get(meta.get('session_id'))
        if session is None or time.monotonic()-session['touched']>1800:raise ValueError('Unknown or expired recognition session')
        for name in ('board_epoch','geometry_version','orientation_version','width','height'):
            if type(meta.get(name)) is not int or meta[name]<(1 if name in ('width','height','geometry_version') else 0):raise ValueError('Invalid '+name)
        if meta.get('schema_version')!=VERSION:raise ValueError('Unsupported recognition schema')
        if not isinstance(meta.get('frame_id'),str) or not 1<=len(meta['frame_id'])<=80:raise ValueError('Invalid frame ID')
        timestamp=meta.get('timestamp_ms')
        if isinstance(timestamp,bool) or not isinstance(timestamp,(int,float)) or not math.isfinite(timestamp) or timestamp<0:raise ValueError('Invalid timestamp')
        if meta['board_epoch']<session['epoch'] or timestamp<=session['timestamp'] or meta['geometry_version']<=session['geometry_version']:raise ValueError('Stale frame or board epoch')
        if meta['width']*meta['height']>32000000:raise ValueError('Image exceeds pixel limit')
        return session
    def accept(self,meta):
        session=self.validate(meta);session.update(epoch=meta['board_epoch'],timestamp=meta['timestamp_ms'],geometry_version=meta['geometry_version'],touched=time.monotonic())


def prepare_keyframe(image,geometry):
    """Keep camera-relative pixel appearance. Canonical IDs only select target masks."""
    height,width=image.shape[:2]
    polygons=_geometry(geometry|{'source':'manual'},width,height)
    if geometry.get('source')!='tracking':raise ValueError('Expected tracked geometry')
    seed=order_corners(geometry['corners'])
    rectified=warp_board_with_context(image,seed,800,200)
    transform=cv2.getPerspectiveTransform(seed,np.float32([[200,200],[999,200],[999,999],[200,999]]))
    crops=[]
    for square in [f'{f}{r}' for r in range(8,0,-1) for f in 'abcdefgh']:
        polygon=cv2.perspectiveTransform(np.asarray(polygons[square],np.float32).reshape(-1,1,2),transform).reshape(4,2)
        crop,coverage,_,bounds,target,valid=polygon_crop(rectified.image,rectified.valid_mask,polygon,200)
        if valid<.99:raise ValueError('Target square is clipped: '+square)
        crops.append({'image':crop,'target_polygon':target,'row':(int(square[1])-1),'col':ord(square[0])-97,'chess_square':square,'context_bounds':bounds,'target_valid_fraction':valid,'context_valid_fraction':float((coverage>0).mean())})
    return crops


def recognize_keyframe(raw,meta,provider=None):
    if not isinstance(meta,dict):raise ValueError('Invalid keyframe metadata')
    if hashlib.sha256(raw).hexdigest()!=meta.get('source_sha256'):raise ValueError('JPEG hash mismatch')
    with Image.open(BytesIO(raw)) as header:
        if header.format!='JPEG' or header.width*header.height>32000000:raise ValueError('Expected bounded JPEG image')
    if not isinstance(meta.get('geometry'),dict):raise ValueError('Missing tracked geometry')
    image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
    if image is None or list(image.shape[1::-1])!=[meta['width'],meta['height']]:raise ValueError('JPEG dimensions mismatch')
    start=time.perf_counter();crops=prepare_keyframe(image,meta['geometry'])
    predictions,logits=(provider or classifier()).predict_with_logits(crops)
    squares={}
    for crop,prediction,logit in zip(crops,predictions,logits):
        squares[crop['chess_square']]={**{k:v for k,v in prediction.metadata().items() if k not in ('row','col')},'logits':logit,'context_bounds':crop['context_bounds'],'target_polygon':crop['target_polygon'],'target_valid_fraction':crop['target_valid_fraction'],'context_valid_fraction':crop['context_valid_fraction'],'visibility':'unknown'}
    if len(squares)!=64:raise ValueError('Expected 64 square predictions')
    return {'schema_version':VERSION,**{k:meta[k] for k in ('session_id','frame_id','board_epoch','geometry_version','orientation_version','timestamp_ms','source_sha256')},'squares':squares,'model':model_metadata(),'elapsed_ms':round((time.perf_counter()-start)*1000,1),'resolved_squares':0,'completion_enabled':False}
