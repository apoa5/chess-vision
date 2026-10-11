"""Deterministic capture validation and geometry diagnostics, without recognition."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from zipfile import ZipFile
import cv2
import numpy as np
import chess

VERSION = 'guided_capture_v1'
MAX_BYTES = 100 * 1024 * 1024


def _project(h, points):
    p=np.column_stack((points,np.ones(len(points)))) @ h.T
    if np.any(p[:,2] <= 1e-8):raise ValueError('Geometry crosses projective horizon')
    return p[:,:2]/p[:,2:]


def _geometry(value, width, height):
    if not isinstance(value,dict) or value.get('source')!='manual':raise ValueError('Unsupported geometry provenance')
    if type(value.get('version')) is not int or value['version']<1:raise ValueError('Invalid geometry version')
    h=np.asarray(value.get('h_board_to_frame'),dtype=float)
    corners=np.asarray(value.get('corners'),dtype=float)
    if h.shape!=(9,) or corners.shape!=(4,2) or not np.isfinite(h).all() or not np.isfinite(corners).all():raise ValueError('Invalid geometry')
    h=h.reshape(3,3)
    if abs(np.linalg.det(h))<1e-9:raise ValueError('Singular geometry')
    if np.any(corners<0) or np.any(corners[:,0]>width) or np.any(corners[:,1]>height):raise ValueError('Corners outside source')
    edges=np.roll(corners,-1,axis=0)-corners
    following=np.roll(edges,-1,axis=0)
    crosses=edges[:,0]*following[:,1]-edges[:,1]*following[:,0]
    if np.any(crosses>=-1):raise ValueError('Mirrored, folded or degenerate corners')
    projected=_project(h,np.array([[0,0],[32,0],[32,32],[0,32]]))
    if not np.allclose(projected,corners,atol=.1):raise ValueError('Homography does not match this frame corners')
    polygons={}
    for rank in range(8):
        for file in range(8):
            x,y=file*4,rank*4
            polygons[f'{chr(97+file)}{rank+1}']=_project(h,np.array([[x,y],[x+4,y],[x+4,y+4],[x,y+4]])).tolist()
    supplied=value.get('square_polygons')
    if supplied is not None:
        if not isinstance(supplied,dict) or set(supplied)!=set(polygons):raise ValueError('Expected 64 refined square polygons')
        inverse=np.linalg.inv(h)
        for square,polygon in supplied.items():
            p=np.asarray(polygon,dtype=float)
            if p.shape!=(4,2) or not np.isfinite(p).all():raise ValueError('Invalid refined square polygon')
            # Fitting can adjust boundaries, but cannot remap square identity.
            q=np.column_stack((p,np.ones(4))) @ inverse.T
            if np.any(np.abs(q[:,2])<1e-8):raise ValueError('Invalid refined square projection')
            board=q[:,:2]/q[:,2:]
            f=ord(square[0])-97;r=int(square[1])-1
            expected=np.array([[f*4,r*4],[(f+1)*4,r*4],[(f+1)*4,(r+1)*4],[f*4,(r+1)*4]])
            # Camera-order polygons may start at a different vertex and have opposite winding.
            error=min(float(np.max(np.linalg.norm(np.roll(order,k,axis=0)-expected,axis=1))) for order in (board,board[::-1]) for k in range(4))
            if error>.8:raise ValueError('Refined polygon changes square identity')
        polygons=supplied
    return polygons


def replay_session(archive, output_dir=None):
    """Validate bytes before use; never extract archive paths to the filesystem."""
    records=[]
    with ZipFile(archive) as zipped:
        entries=zipped.infolist()
        names=[i.filename for i in entries]
        if len(entries)>65 or len(names)!=len(set(names)) or sum(i.file_size for i in entries)>MAX_BYTES:raise ValueError('Archive exceeds limits or has duplicate names')
        for name in names:
            path=PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name:raise ValueError('Unsafe archive path')
        if 'manifest.json' not in names or zipped.getinfo('manifest.json').file_size>1024*1024:raise ValueError('Missing or oversized manifest')
        manifest=json.loads(zipped.read('manifest.json'))
        if manifest.get('schema_version')!=VERSION:raise ValueError('Unsupported capture schema')
        if not isinstance(manifest.get('session_id'),str) or not manifest['session_id']:raise ValueError('Missing session ID')
        if manifest.get('board')!={'square_cm':4,'border_cm':3.5}:raise ValueError('Unsupported physical board profile')
        frames=manifest.get('frames')
        if not isinstance(frames,list) or not 1<=len(frames)<=60:raise ValueError('Expected 1–60 frames')
        events=manifest.get('events',[])
        if not isinstance(events,list) or len(events)>1000:raise ValueError('Invalid events')
        for event in events:
            if not isinstance(event,dict) or not isinstance(event.get('type'),str):raise ValueError('Invalid event')
            t=event.get('timestamp_ms');epoch=event.get('board_epoch')
            if type(t) not in (int,float) or not math.isfinite(t) or t<0 or type(epoch) is not int or epoch<0:raise ValueError('Invalid event timestamp/epoch')
        metadata=manifest.get('metadata',{})
        if not isinstance(metadata,dict):raise ValueError('Invalid session metadata')
        fen=metadata.get('fen')
        if fen:
            try:chess.Board(fen)
            except ValueError as exc:raise ValueError('Invalid ground-truth FEN') from exc
        seen=set();ids=set();paths=set()
        for frame in frames:
            path=frame.get('path');frame_id=frame.get('frame_id')
            if not isinstance(frame_id,str) or not frame_id or frame_id in ids or path in paths:raise ValueError('Duplicate or missing frame identity')
            ids.add(frame_id);paths.add(path)
            if not isinstance(path,str) or not path.startswith('frames/') or not path.endswith('.jpg') or path not in names:raise ValueError('Missing frame image')
            t=frame.get('timestamp_ms');epoch=frame.get('board_epoch')
            if type(t) not in (int,float) or not math.isfinite(t) or t<0 or type(epoch) is not int or epoch<0:raise ValueError('Invalid timestamp/epoch')
            if frame.get('source') not in ('camera','demo') or frame.get('visibility')!='unknown' or not isinstance(frame.get('view_group'),str):raise ValueError('Invalid capture provenance')
            width,height=frame.get('width'),frame.get('height')
            if any(type(v) is not int or v<=0 or v>12000 for v in (width,height)) or width*height>32000000:raise ValueError('Invalid source dimensions')
            raw=zipped.read(path);digest=hashlib.sha256(raw).hexdigest()
            if digest!=frame.get('source_sha256'):raise ValueError(f'Hash mismatch: {frame_id}')
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape[1]!=frame.get('width') or image.shape[0]!=frame.get('height'):raise ValueError('Source dimensions mismatch')
            gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            blur=float(cv2.Laplacian(gray,cv2.CV_64F).var())
            reasons=['recognition_not_connected','visibility_unknown']
            key=(epoch,digest)
            if key in seen:reasons.append('duplicate_source_in_epoch')
            seen.add(key)
            if frame['source']=='demo':reasons.append('synthetic_demo')
            polygons=None
            if frame.get('geometry') is None:reasons.append('geometry_uninitialized')
            else:polygons=_geometry(frame['geometry'],image.shape[1],image.shape[0])
            record={'frame_id':frame_id,'timestamp_ms':t,'board_epoch':epoch,'source_sha256':digest,'view_group':frame['view_group'],'laplacian_variance':blur,'geometry_status':'manual' if polygons else 'uninitialized','eligible_for_recognition_evidence':False,'reasons':reasons,'square_polygons':polygons}
            records.append(record)
            if output_dir and polygons:
                destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
                for square,polygon in polygons.items():
                    pts=np.rint(polygon).astype(np.int32)
                    cv2.polylines(image,[pts],True,(100,230,130),2)
                    center=tuple(np.rint(pts.mean(axis=0)).astype(int))
                    cv2.putText(image,square,center,cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),4)
                    cv2.putText(image,square,center,cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
                # Use our own index, never user-supplied identity in a filesystem path.
                if not cv2.imwrite(str(destination/f'frame_{len(records):04d}_overlay.jpg'),image):raise OSError('Could not save overlay')
        if set(names)!={'manifest.json',*paths}:raise ValueError('Unexpected archive entries')
    records.sort(key=lambda r:(r['timestamp_ms'],r['frame_id']))
    report={'schema_version':'guided_replay_v1','session_id':manifest['session_id'],'metadata':metadata,'frames':records,'events':manifest.get('events',[]),'automatic_completion':False,'resolved_squares':0,'recognition_connected':False}
    if output_dir:
        destination=Path(output_dir);destination.mkdir(parents=True,exist_ok=True)
        (destination/'replay.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
