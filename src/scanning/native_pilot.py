"""Offline native-video pilot provenance; no recognition, tracking or verified labels."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from src.scanning.frame_selection import image_signature


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def decoder():
    local = Path(__file__).resolve().parents[2]/'tools/pilot_video/python'
    if local.is_dir() and str(local) not in sys.path:
        sys.path.insert(0, str(local))
    try:
        import av
    except ImportError as error:
        raise ValueError('Install isolated requirements-pilot-video.txt; do not estimate VFR timestamps using nominal FPS') from error
    return av


def pts_record(pts, time_base, first_time=None):
    if pts is None or type(pts) is not int:
        raise ValueError('Missing integer presentation timestamp')
    base = Fraction(time_base)
    if base <= 0:
        raise ValueError('Nonpositive time base')
    absolute = pts*base
    relative = absolute-(absolute if first_time is None else first_time)
    return dict(pts=pts, time_base=[base.numerator, base.denominator],
                pts_time=[absolute.numerator, absolute.denominator],
                timestamp_ms=float(absolute*1000), relative_timestamp_ms=float(relative*1000))


def select_candidates(timeline, max_frames=12, min_gap_ms=2000, min_blur=30., duplicate_distance=6):
    """Same A.3 time-bin/quality/gap/dHash policy, applied to native PTS samples."""
    if not timeline or not 1 <= max_frames <= 12 or min_gap_ms < 2000:
        raise ValueError('Nonempty timeline and A.3 bounded selection settings required')
    start, end = timeline[0]['timestamp_ms'], timeline[-1]['timestamp_ms']
    if end <= start:
        raise ValueError('Timeline must span positive time')
    accepted = []
    for row in timeline:
        row['selected'] = False
        row['reasons'] = []
        row['time_bin'] = min(max_frames-1, int((row['timestamp_ms']-start)/(end-start)*max_frames))
        if row['blur_score'] < min_blur:row['reasons'].append('blur_below_threshold')
        if not 20 <= row['mean_luma'] <= 235:row['reasons'].append('exposure_outside_range')
    for bucket in range(max_frames):
        for row in sorted([r for r in timeline if r['time_bin'] == bucket and not r['reasons']], key=lambda r: (-r['blur_score'], r['timestamp_ms'])):
            if any(row['decoded_pixel_sha256'] == other['decoded_pixel_sha256'] or
                   (int(row['dhash'], 16)^int(other['dhash'], 16)).bit_count() <= duplicate_distance for other in accepted):
                row['reasons'].append('near_duplicate_of_accepted_candidate')
            elif any(abs(row['timestamp_ms']-other['timestamp_ms']) < min_gap_ms for other in accepted):
                row['reasons'].append('insufficient_temporal_gap')
            else:
                row['selected'] = True; accepted.append(row);break
    for row in timeline:
        if not row['selected'] and not row['reasons']:row['reasons'].append('not_selected_in_time_bin')
    return accepted


def inspect_video(path, destination, session_id, *, sample_hz=2):
    """Decode all frames in presentation order; index every PTS, save ≤12 PNGs.

    Sampling is only an offline quality shortlist, never a replacement timeline.
    Current pilot streams have no display rotation. Fail closed for rotated
    streams until their exact display transform is separately supported/tested.
    """
    av = decoder()
    source = Path(path)
    destination = Path(destination)
    if destination.exists():raise ValueError('Refusing to overwrite native-video output')
    destination.mkdir(parents=True, exist_ok=False)
    source_sha = sha256_file(source)
    index, sampled, images = [], [], {}
    first, previous, previous_bucket = None, None, None
    with av.open(str(source)) as container:
        if len(container.streams.video) != 1:raise ValueError('Exactly one video stream required')
        stream = container.streams.video[0]
        stream.thread_type = 'SLICE';stream.codec_context.thread_count = 1
        metadata = dict(container_format=container.format.name, container_metadata=dict(container.metadata),
                        container_duration_seconds=container.duration/av.time_base if container.duration is not None else None,
                        codec=stream.codec_context.name, pixel_format=stream.codec_context.format.name,
                        encoded_width=stream.width, encoded_height=stream.height, declared_frames=stream.frames,
                        stream_time_base=str(stream.time_base), stream_start_pts=stream.start_time,
                        stream_duration_pts=stream.duration, stream_metadata=dict(stream.metadata),
                        average_rate=str(stream.average_rate), nominal_rate=str(stream.base_rate),
                        streams=[dict(index=s.index, type=s.type, codec=s.codec_context.name, time_base=str(s.time_base), duration_pts=s.duration, metadata=dict(s.metadata)) for s in container.streams])
        for number, frame in enumerate(container.decode(stream)):
            rotation = getattr(frame, 'rotation', 0)
            if rotation != 0:raise ValueError('Nonzero display rotation requires an explicit tested display-coordinate adapter')
            absolute = Fraction(frame.pts)*frame.time_base if frame.pts is not None else None
            if absolute is None:raise ValueError('Missing frame PTS; nominal-FPS substitution forbidden')
            if first is None:first = absolute
            if previous is not None and absolute <= previous:raise ValueError('Non-increasing presentation timeline')
            previous = absolute
            row = dict(frame_id=f'{session_id}_frame_{number:06d}', decode_index=number,
                       width=frame.width, height=frame.height, duration_pts=getattr(frame, 'duration', None),
                       display_rotation_degrees=rotation, **pts_record(frame.pts, frame.time_base, first))
            index.append(row)
            bucket = int((absolute-first)*sample_hz)
            if bucket == previous_bucket:continue
            previous_bucket = bucket
            image = frame.to_ndarray(format='bgr24')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            quality = row | dict(blur_score=float(cv2.Laplacian(gray, cv2.CV_64F).var()), mean_luma=float(gray.mean()),
                                 decoded_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                 dhash=f'{image_signature(image):016x}')
            sampled.append(quality);images[row['frame_id']] = image
    if len(index) < 2:raise ValueError('At least two decoded frames required')
    if metadata['declared_frames'] and len(index) != metadata['declared_frames']:
        raise ValueError('Decoded frame count disagrees with stream declaration')
    selected = select_candidates(sampled)
    candidates = []
    for row in selected:
        image = images[row['frame_id']]
        filename = row['frame_id']+'.png'
        if not cv2.imwrite(str(destination/filename), image):raise OSError('PNG encoding failed')
        candidates.append(row | dict(path=str(destination/filename), source_video_sha256=source_sha,
                          derivative_sha256=sha256_file(destination/filename), split='pilot', training_eligible=False,
                          verified_corners_px=None, verified_semantic_orientation=None, photographic_placement_review='pending',
                          camera_orientation=None, camera_elevation=None, board_visibility=None, review_status='pending'))
    times = [Fraction(*row['pts_time']) for row in index]
    intervals = [b-a for a, b in zip(times, times[1:])]
    histogram = Counter(str(value) for value in intervals)
    result = dict(schema_version='native_pilot_session_v1', session_id=session_id, split='pilot', designation='development_only',
                  source_video=dict(path=str(source), sha256=source_sha, bytes=source.stat().st_size, **metadata),
                  decoded_frames=len(index), all_frames_decoded=True, pts_strictly_increasing=True,
                  first_pts_time=str(times[0]), last_pts_time=str(times[-1]),
                  interval_seconds_histogram=dict(histogram), observed_mean_fps=float((len(index)-1)/(times[-1]-times[0])),
                  max_frame_gap_ms=float(max(intervals)*1000), multiple_frame_intervals=len(histogram)>1,
                  sample_hz=sample_hz, selection_settings=dict(max_frames=12, min_gap_ms=2000, min_blur=30,
                      min_luma=20, max_luma=235, duplicate_distance=6), candidates=candidates,
                  sampled_quality_timeline=sampled, original_frame_index=index,
                  verified_ground_truth=False, unchanged_position_review='pending',
                  limitations=['All original frames remain in the unchanged native recording.',
                    'Frame PNG pixels are lossless derivatives of decoder BGR output; native source is compressed video.',
                    'Quality/dHash heuristics do not establish camera pose, visible board, identity or geometric correctness.',
                    'Container creation_time is a metadata tag, not verified capture start time.',
                    'This native index is not a browser guided_capture_v1 ZIP or an approved capture_protocol_v2 companion.'])
    if sha256_file(source) != source_sha:raise ValueError('Original changed during decode')
    (destination/'session.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
