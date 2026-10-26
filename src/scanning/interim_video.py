"""Small original-video adapter for D.3c. No training or production integration."""
from fractions import Fraction
import gzip
import hashlib
import json
from pathlib import Path
import re
import cv2
from src.scanning.native_pilot import decoder, pts_record, sha256_file
from src.scanning.pilot_comparison import display_image, display_transform


def match_position(path, mapping):
    match=re.fullmatch(r'pos_(\d{3})\.(mov|mp4)',Path(path).name,re.I)
    if not match: raise ValueError('Ambiguous filename; require pos_NNN.MOV/MP4')
    number=f'{int(match[1]):02d}'
    if number not in mapping: raise ValueError('No supplied FEN for position '+number)
    return number


def index_video(path,session_id):
    av=decoder();rows=[];rotations=set();first=None;previous=None
    with av.open(str(path)) as container:
        if len(container.streams.video)!=1: raise ValueError('Exactly one video stream required')
        stream=container.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=1
        meta=dict(codec=stream.codec_context.name,pixel_format=stream.codec_context.format.name,
            encoded_width=stream.width,encoded_height=stream.height,declared_frames=stream.frames,
            average_rate=str(stream.average_rate),nominal_rate=str(stream.base_rate),time_base=str(stream.time_base),
            stream_metadata=dict(stream.metadata),container_metadata=dict(container.metadata),
            container_duration_seconds=container.duration/av.time_base if container.duration is not None else None)
        for n,frame in enumerate(container.decode(stream)):
            t=Fraction(frame.pts)*frame.time_base if frame.pts is not None else None
            if t is None or (previous is not None and t<=previous):raise ValueError('Missing or unordered native PTS')
            if first is None:first=t
            rotation=int(frame.rotation);display_transform(frame.width,frame.height,rotation)
            rotations.add(rotation);previous=t
            width,height=(frame.height,frame.width) if rotation in (-90,90,270,-270) else (frame.width,frame.height)
            rows.append(dict(frame_id=f'{session_id}_f{n:06d}',decode_index=n,**pts_record(frame.pts,frame.time_base,first),
                width=width,height=height,encoded_width=frame.width,encoded_height=frame.height,display_rotation_degrees=rotation))
    if len(rows)<2:raise ValueError('Insufficient decoded frames')
    meta.update(decoded_frames=len(rows),display_rotations_degrees=sorted(rotations),display_width=rows[0]['width'],display_height=rows[0]['height'],
        first_pts=rows[0]['pts'],last_pts=rows[-1]['pts'],observed_mean_fps=(len(rows)-1)/float(previous-first),
        observed_duration_seconds=float(previous-first),encoded_to_display_matrix=display_transform(meta['encoded_width'],meta['encoded_height'],rows[0]['display_rotation_degrees']).tolist())
    return meta,rows


def decode_exact(path,row):
    """Same original PTS/right-angle BGR8 convention as D.1; no transcoding."""
    av=decoder();target=Fraction(*row['pts_time'])
    with av.open(str(path)) as container:
        stream=container.streams.video[0];stream.thread_type='SLICE';stream.codec_context.thread_count=1
        container.seek(row['pts'],stream=stream,backward=True)
        for frame in container.decode(stream):
            t=Fraction(frame.pts)*frame.time_base
            if t>target:break
            if t==target:
                if int(frame.rotation)!=row['display_rotation_degrees']:raise ValueError('Display rotation changed')
                image=display_image(frame.to_ndarray(format='bgr24'),int(frame.rotation))
                if image.shape[:2]!=(row['height'],row['width']):raise ValueError('Display size changed')
                return image
    raise ValueError('Exact original PTS could not be decoded')


def pixel_hash(image):return hashlib.sha256(image.tobytes()).hexdigest()


def write_gzip(path,obj):
    path=Path(path)
    if path.exists():raise ValueError('Fresh gzip path required')
    with path.open('wb') as stream:
        with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as zipped:
            zipped.write(json.dumps(obj,allow_nan=False,separators=(',',':')).encode())
