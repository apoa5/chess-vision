"""Compact board-font character reader and image-wide component proposals."""
from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np

CHARACTERS='ABCDEFGH12345678'
LABELS=tuple(CHARACTERS)+('reject',)
MODEL_PATH=Path(__file__).resolve().parents[2]/'models/checkpoints/board_marking_reader.pt'


def build_model():
    from torch import nn
    return nn.Sequential(nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,48,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((3,3)),nn.Flatten(),nn.Linear(432,96),nn.ReLU(),nn.Linear(96,len(LABELS)))


def normalize_component(binary):
    h,w=binary.shape;scale=min(28/w,40/h)
    glyph=cv2.resize(binary,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA)
    canvas=np.zeros((48,36),np.uint8);y=(48-glyph.shape[0])//2;x=(36-glyph.shape[1])//2
    canvas[y:y+glyph.shape[0],x:x+glyph.shape[1]]=glyph
    return canvas


def components(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    factor=min(1.,1920/max(gray.shape));working=cv2.resize(gray,None,fx=factor,fy=factor) if factor<1 else gray
    result=[];seen=[]
    for kernel in (15,25,41):
        contrast=cv2.morphologyEx(working,cv2.MORPH_TOPHAT,np.ones((kernel,kernel),np.uint8))
        _,binary=cv2.threshold(contrast,55,255,cv2.THRESH_BINARY)
        count,regions,stats,_=cv2.connectedComponentsWithStats(binary)
        for i in range(1,count):
            x,y,w,h,area=stats[i]
            if not (6<=h<=70 and 2<=w<=65 and 9<=area<=1800 and .12<=w/h<=1.8):continue
            center=np.array([x+w/2,y+h/2])/factor
            if any(np.linalg.norm(center-other)<3/factor for other in seen):continue
            seen.append(center)
            glyph=normalize_component((regions[y:y+h,x:x+w]==i).astype(np.uint8)*255)
            result.append({'glyph':glyph,'center':center.tolist(),'polygon':(np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]])/factor).tolist()})
            if len(result)>=2000:return result
    return result


@lru_cache(maxsize=1)
def load_reader():
    import torch
    checkpoint=torch.load(MODEL_PATH,map_location='cpu',weights_only=True)
    if checkpoint.get('schema_version')!='board_marking_reader_v1' or checkpoint.get('labels')!=list(LABELS):raise ValueError('Invalid marking reader contract')
    torch.set_num_threads(4);model=build_model();model.load_state_dict(checkpoint['state_dict']);model.eval()
    return model


def rotated_glyph(glyph,k):
    rotated=np.rot90(glyph,k).copy();y,x=np.where(rotated>30)
    return normalize_component(rotated[y.min():y.max()+1,x.min():x.max()+1]) if len(x) else np.zeros((48,36),np.uint8)


def read_components(image,threshold=.80):
    import torch
    proposals=components(image)
    if not proposals:return []
    # Keep every orientation hypothesis: a symmetric glyph cannot decide the side.
    arrays=np.stack([rotated_glyph(p['glyph'],k) for p in proposals for k in range(4)])[:,None].astype(np.float32)/255
    model=load_reader();results=[]
    with torch.inference_mode():
        probabilities=np.concatenate([torch.softmax(model(torch.from_numpy(arrays[i:i+128])),1).numpy() for i in range(0,len(arrays),128)])
    for index,p in enumerate(proposals):
        unique={}
        for k in range(4):
            prob=probabilities[4*index+k];label=int(prob.argmax());confidence=float(prob[label]);margin=confidence-float(np.partition(prob,-2)[-2])
            if label==16 or confidence<threshold or margin<.35:continue
            name=LABELS[label]
            if name not in unique or unique[name]['confidence']<confidence:unique[name]={'label':name,'confidence':confidence,'center':p['center'],'polygon':p['polygon'],'glyph_turns':k,'source':'board_marking_reader_v1','proposal_id':index}
        results.extend(unique.values())
    return results
