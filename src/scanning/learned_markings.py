"""Pretrained OCR observations and spatially checked board-plane registration."""
from pathlib import Path
from functools import lru_cache
import sys
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
ALPHABET='ABCDEFGH12345678'


@lru_cache(maxsize=1)
def get_reader():
    local=ROOT/'tools/ocr/python'
    if local.exists() and str(local) not in sys.path:sys.path.insert(0,str(local))
    import easyocr
    import torch
    torch.set_num_threads(4)
    # Downloads are explicit setup, never an inference-time network side effect.
    return easyocr.Reader(['en'],gpu=False,model_storage_directory=str(ROOT/'tools/ocr/models'),user_network_directory=str(ROOT/'tools/ocr/network'),download_enabled=False,verbose=False)


def read_markings(image):
    reader=get_reader()
    outputs=reader.readtext(image,allowlist=ALPHABET+ALPHABET[:8].lower(),rotation_info=[90,180,270],paragraph=False,min_size=5,text_threshold=.4,low_text=.2,link_threshold=.2,canvas_size=1920,mag_ratio=1,batch_size=16,workers=0)
    observations=[]
    for box,text,confidence in outputs:
        text=text.upper().strip()
        if len(text)!=1 or text not in ALPHABET or float(confidence)<.55:continue
        polygon=np.asarray(box,dtype=float)
        if polygon.shape!=(4,2) or not np.isfinite(polygon).all():continue
        observations.append({'label':text,'confidence':float(confidence),'center':polygon.mean(axis=0).tolist(),'polygon':polygon.tolist(),'source':'easyocr_english_g2'})
    return observations


def marker_positions(label,offset=2.):
    if label in 'ABCDEFGH':
        x=4*'ABCDEFGH'.index(label)+2
        return [(x,-offset),(x,32+offset)]
    if label in '12345678':
        y=4*(int(label)-1)+2
        return [(-offset,y),(32+offset,y)]
    raise ValueError('Unknown marking')


def fit_marking_geometry(observations,seed,image_shape,*,max_seed_distance=3.5):
    """Fit recognized labels, checking both axes; no slot-derived character labels."""
    candidates=[];height,width=image_shape[:2]
    rows=[]
    for obs in sorted(observations,key=lambda x:x['confidence'],reverse=True):
        label=obs.get('label')
        if not isinstance(label,str) or len(label)!=1 or label not in ALPHABET:continue
        center=np.asarray(obs['center'],np.float32)
        if center.shape!=(2,) or not np.isfinite(center).all():continue
        rows.append((label,center,marker_positions(label),obs.get('glyph_seed_turns',obs.get('glyph_turns') if obs.get('sampling')=='rectified_board_plane' else None)))
    if not rows:return {'status':'uncertain','reason':'insufficient_spatially_consistent_ocr','observations':observations}
    centers=np.asarray([row[1] for row in rows],np.float32)
    slots=np.asarray([row[2] for row in rows],np.float64)
    seed=np.asarray(seed,np.float32)
    camera=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),seed)
    indices=[[3,2,1,0],[0,3,2,1],[1,0,3,2],[2,1,0,3]]
    for dx in (-1,0,1):
        for dy in (-1,0,1):
            shifted=cv2.perspectiveTransform((np.float32([[0,0],[32,0],[32,32],[0,32]])+[4*dx,4*dy]).astype(np.float32)[:,None,:],camera).reshape(4,2)
            for turns,order in enumerate(indices):
                initial=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),shifted[order])
                inverse=np.linalg.inv(initial);source=[];target=[];labels=[];used=set()
                # Transform all proposals together, retaining the same confidence
                # order, pose checks and unique (label, edge) assignment as before.
                points=cv2.perspectiveTransform(centers[:,None,:],inverse).reshape(-1,2)
                distances=np.linalg.norm(points[:,None,:].astype(np.float64)-slots,axis=2)
                closest=np.argmin(distances,axis=1)
                nearby=np.flatnonzero(distances[np.arange(len(rows)),closest]<=max_seed_distance)
                for index in nearby:
                    label,center,positions,pose=rows[index];which=int(closest[index]);key=(label,which)
                    if pose is not None and pose!=(turns+2*which)%4:continue
                    if key in used:continue
                    used.add(key);source.append(positions[which]);target.append(center);labels.append(label)
                if len(source)<8:continue
                model,mask=cv2.findHomography(np.float32(source),np.float32(target),cv2.RANSAC,6.)
                if model is None:continue
                selected=mask.ravel().astype(bool)
                projected=cv2.perspectiveTransform(np.float32(source).reshape(-1,1,2),model).reshape(-1,2)
                residuals=np.linalg.norm(projected-np.asarray(target),axis=1);unique_centers=[]
                for index in np.argsort(residuals):
                    if not selected[index]:continue
                    if any(np.linalg.norm(np.asarray(target[index])-center)<3 for center in unique_centers):selected[index]=False
                    else:unique_centers.append(np.asarray(target[index]))
                good={label for label,ok in zip(labels,selected) if ok}
                if sum(selected)<8 or len(good&set('ABCDEFGH'))<3 or len(good&set('12345678'))<3:continue
                valid_source=np.asarray(source)[selected]
                # At least two intersecting edge families; reject one-line fits.
                if np.ptp(valid_source[:,0])<12 or np.ptp(valid_source[:,1])<12:continue
                p=np.column_stack((np.float32([[0,0],[32,0],[32,32],[0,32]]),np.ones(4)))@model.T
                if np.any(p[:,2]*p[0,2]<=0):continue
                corners=p[:,:2]/p[:,2:]
                if np.any(corners<0) or np.any(corners[:,0]>=width) or np.any(corners[:,1]>=height):continue
                edges=np.roll(corners,-1,axis=0)-corners
                cross=edges[:,0]*np.roll(edges,-1,axis=0)[:,1]-edges[:,1]*np.roll(edges,-1,axis=0)[:,0]
                if np.any(cross>=-1):continue
                predicted=cv2.perspectiveTransform(np.float32(source).reshape(-1,1,2),model).reshape(-1,2)
                residual=float(np.median(np.linalg.norm(predicted[selected]-np.asarray(target)[selected],axis=1)))
                entry={'corners':corners.tolist(),'inliers':int(sum(selected)),'distinct_files':len(good&set('ABCDEFGH')),'distinct_ranks':len(good&set('12345678')),'median_residual_px':residual,'rotation_quarters_ccw':turns,'labels':labels,'inlier_mask':selected.tolist(),'seed_offset_squares':[dx,dy],'source':'learned_border_markings'}
                if not any(np.max(np.linalg.norm(corners-np.asarray(c['corners']),axis=1))<8 for c in candidates):candidates.append(entry)
    candidates.sort(key=lambda c:(-c['inliers'],c['median_residual_px']))
    if not candidates:return {'status':'uncertain','reason':'insufficient_spatially_consistent_ocr','observations':observations}
    if len(candidates)>1 and candidates[0]['inliers']-candidates[1]['inliers']<3:
        return {'status':'uncertain','reason':'competing_ocr_geometries','observations':observations,'candidates':candidates}
    return {'status':'candidate','observations':observations,**candidates[0]}


def coarse_board_seed(image):
    """Optional wood-boundary proposal; never accepted without recognized markings."""
    from src.vision.perspective import order_corners
    scale=min(1.,960/max(image.shape[:2]));small=cv2.resize(image,None,fx=scale,fy=scale)
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array([0,45,20],np.uint8),np.array([40,255,255],np.uint8))
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours,key=cv2.contourArea,reverse=True):
        if cv2.contourArea(contour)<small.shape[0]*small.shape[1]*.05:break
        hull=cv2.convexHull(contour);quad=None
        for epsilon in (.025,.04,.06):
            proposed=cv2.approxPolyDP(hull,epsilon*cv2.arcLength(hull,True),True)
            if len(proposed)==4:quad=proposed;break
        if quad is None:continue
        outer=order_corners(quad.reshape(4,2)/scale)
        transform=cv2.getPerspectiveTransform(np.float32([[0,0],[39,0],[39,39],[0,39]]),outer)
        return cv2.perspectiveTransform(np.float32([[3.5,3.5],[35.5,3.5],[35.5,35.5],[3.5,35.5]])[:,None,:],transform).reshape(4,2)
    return None


def coarse_board_seeds(image,limit=6):
    """Bounded alternative wood outlines; labels, never color, establish identity.

    A single low-saturation mask can merge a wooden board with its wooden table.
    Use several contrasts and contour candidates rather than trusting the largest.
    """
    from src.vision.perspective import order_corners
    scale=min(1.,960/max(image.shape[:2]));small=cv2.resize(image,None,fx=scale,fy=scale)
    hsv=cv2.cvtColor(small,cv2.COLOR_BGR2HSV);seeds=[]
    for saturation in (80,120,60,45):
        mask=cv2.inRange(hsv,np.array([0,saturation,20],np.uint8),np.array([40,255,255],np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours,key=cv2.contourArea,reverse=True)[:3]:
            if cv2.contourArea(contour)<small.shape[0]*small.shape[1]*.05:continue
            hull=cv2.convexHull(contour)
            for epsilon in (.025,.04,.06):
                quad=cv2.approxPolyDP(hull,epsilon*cv2.arcLength(hull,True),True)
                if len(quad)!=4:continue
                outer=order_corners(quad.reshape(4,2)/scale)
                transform=cv2.getPerspectiveTransform(np.float32([[0,0],[39,0],[39,39],[0,39]]),outer)
                seed=order_corners(cv2.perspectiveTransform(np.float32([[3.5,3.5],[35.5,3.5],[35.5,35.5],[3.5,35.5]])[:,None,:],transform).reshape(4,2))
                if not any(np.max(np.linalg.norm(seed-other,axis=1))<10 for other in seeds):seeds.append(seed)
                break
            if len(seeds)>=limit:return seeds
    return seeds


def fit_alternative_seeds(observations,seeds,image_shape):
    """Compare independent outline hypotheses using the same recognized glyphs."""
    candidates=[]
    for index,seed in enumerate(seeds):
        edge=seed[1]-seed[0]
        angle=float(np.degrees(np.arctan2(edge[1],edge[0])))
        quarter=round(angle/90)
        rows=observations
        if abs(angle-90*quarter)<=22:
            rows=[o|{'glyph_seed_turns':(o['glyph_turns']-quarter)%4} if 'glyph_turns' in o else o for o in observations]
        result=fit_marking_geometry(rows,seed,image_shape)
        if result['status']!='candidate':continue
        corners=np.asarray(result['corners']);entry=result|{'geometry_seed_index':index}
        match=next((i for i,c in enumerate(candidates) if np.max(np.linalg.norm(corners-np.asarray(c['corners']),axis=1))<8),None)
        if match is None:candidates.append(entry)
        elif result['inliers']>candidates[match]['inliers']:candidates[match]=entry
    candidates.sort(key=lambda c:(-c['inliers'],c['median_residual_px']))
    if not candidates:return {'status':'uncertain','reason':'no_alternative_seed_verified'}
    if len(candidates)>1 and candidates[0]['inliers']-candidates[1]['inliers']<3:
        return {'status':'uncertain','reason':'competing_alternative_seed_geometries','candidates':candidates}
    return candidates[0]


def select_marking_candidate(candidates):
    """Merge equivalent fits, then retain the existing competing-board guard."""
    unique=[]
    for entry in candidates:
        corners=np.asarray(entry['corners'])
        match=next((i for i,c in enumerate(unique) if np.max(np.linalg.norm(corners-np.asarray(c['corners']),axis=1))<8),None)
        if match is None:unique.append(entry)
        elif entry['inliers']>unique[match]['inliers']:unique[match]=entry
    unique.sort(key=lambda c:(-c['inliers'],c['median_residual_px']))
    if not unique:return {'status':'uncertain','reason':'no_alternative_seed_verified'}
    if len(unique)>1 and unique[0]['inliers']-unique[1]['inliers']<3:
        return {'status':'uncertain','reason':'competing_alternative_seed_geometries','candidates':unique}
    return {'status':'candidate',**unique[0]}


def fast_marking_geometry(image,observations,threshold=.8):
    """Try raw recognized glyphs, then bounded board-plane glyph sampling."""
    from src.scanning.marking_model import read_plane_components
    seeds=coarse_board_seeds(image)
    raw=fit_alternative_seeds(observations,seeds,image.shape)
    if raw['status']=='candidate':return raw|{'acquisition_path':'fast_marking_outlines'}
    candidates=list(raw.get('candidates',[]))
    # Perspective/diagonal views can make raw glyph poses ambiguous. Read all
    # bounded outlines in their board planes, comparing competing fits rather
    # than accepting the first color contour or assuming any character's slot.
    for index,seed in enumerate(seeds):
        plane=read_plane_components(image,seed,threshold)
        result=fit_marking_geometry(plane,seed,image.shape,max_seed_distance=7.)
        fits=[result] if result['status']=='candidate' else result.get('candidates',[])
        candidates.extend(fit|{'geometry_seed_index':index,'acquisition_path':'fast_marking_planes'} for fit in fits)
    return select_marking_candidate(candidates)
