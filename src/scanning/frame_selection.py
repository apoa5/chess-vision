"""Offline candidate selection from immutable guided-scan sequences, without recognition."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np

from src.scanning.replay import replay_session


def image_signature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    return sum(int(bit) << index for index, bit in enumerate(bits))


def extract_candidates(archive, output_dir, *, max_frames=12, min_gap_ms=2000,
                       min_blur=30., min_luma=20., max_luma=235., duplicate_distance=6):
    """One sharp candidate per time bin; exact bytes preserved, all rejection reasons logged.

    Thresholds are development heuristics, not calibrated correctness probabilities.
    Human review must still establish view diversity, quality and ground truth.
    """
    archive, destination = Path(archive), Path(output_dir)
    if destination.exists():
        raise ValueError('Output directory already exists')
    if (type(max_frames) is not int or not 1 <= max_frames <= 12 or
            not np.isfinite([min_gap_ms, min_blur, min_luma, max_luma]).all() or
            min_gap_ms < 2000 or min_blur < 0 or not 0 <= min_luma < max_luma <= 255 or
            type(duplicate_distance) is not int or not 0 <= duplicate_distance <= 64):
        raise ValueError('Invalid candidate selection parameters')
    source_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    replay_session(archive)  # Existing bounded ZIP/path/hash/dimension validation.
    with ZipFile(archive) as zipped:
        manifest = json.loads(zipped.read('manifest.json'))
        frames = manifest['frames']
        times = [f['timestamp_ms'] for f in frames]
        if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('Expected an original time-ordered sequence, not independent stills')
        if (any(f['source'] != 'camera' or f['board_epoch'] != 0 for f in frames) or
                any(e['type'] in ('board_change', 'new_board_epoch') for e in manifest.get('events', []))):
            raise ValueError('Camera-only unchanged-position sequence required')
        timeline, signatures = [], []
        duration = times[-1]-times[0]
        for f in frames:
            raw = zipped.read(f['path'])
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur, luma = float(cv2.Laplacian(gray, cv2.CV_64F).var()), float(gray.mean())
            reasons = []
            if blur < min_blur:
                reasons.append('blur_below_threshold')
            if not min_luma <= luma <= max_luma:
                reasons.append('exposure_outside_range')
            bucket = min(max_frames-1, int((f['timestamp_ms']-times[0])/duration*max_frames))
            timeline.append({**{k: f[k] for k in ('frame_id', 'timestamp_ms', 'source_sha256', 'width', 'height', 'view_group')},
                             'time_bin': bucket, 'blur_score': blur, 'mean_luma': luma,
                             'selected': False, 'reasons': reasons})
            signatures.append(image_signature(image))
        accepted = []
        for bucket in range(max_frames):
            indexes = sorted([i for i, r in enumerate(timeline) if r['time_bin'] == bucket and not r['reasons']],
                             key=lambda i: (-timeline[i]['blur_score'], timeline[i]['timestamp_ms']))
            for i in indexes:
                row = timeline[i]
                if any(row['source_sha256'] == timeline[j]['source_sha256'] or
                       (signatures[i] ^ signatures[j]).bit_count() <= duplicate_distance for j in accepted):
                    row['reasons'].append('near_duplicate_of_accepted_candidate')
                elif any(abs(row['timestamp_ms']-timeline[j]['timestamp_ms']) < min_gap_ms for j in accepted):
                    row['reasons'].append('insufficient_temporal_gap')
                else:
                    row['selected'] = True
                    accepted.append(i)
                    break
        destination.mkdir(parents=True, exist_ok=False)
        candidates = []
        for i, row in enumerate(timeline):
            if not row['selected'] and not row['reasons']:
                row['reasons'].append('not_selected_in_time_bin')
            if row['selected']:
                filename = f'candidate_{len(candidates)+1:04d}.jpg'
                (destination/filename).write_bytes(zipped.read(frames[i]['path']))
                candidates.append({**row, 'path': filename, 'archive_frame_id': row['frame_id'],
                                   'source_session_id': manifest['session_id'], 'verified_ground_truth': False,
                                   'camera_orientation': None, 'elevation': None,
                                   'review_status': 'pending'})
    if hashlib.sha256(archive.read_bytes()).hexdigest() != source_hash:
        raise ValueError('Source archive changed during extraction')
    report = dict(schema_version='scan_frame_candidates_v1', source_archive=str(archive),
                  source_archive_sha256=source_hash, session_id=manifest['session_id'],
                  selection_settings=dict(max_frames=max_frames, min_gap_ms=min_gap_ms,
                                          min_blur=min_blur, min_luma=min_luma, max_luma=max_luma,
                                          duplicate_distance=duplicate_distance),
                  source_frame_count=len(frames), candidates=candidates, timeline=timeline,
                  ground_truth_verified=False,
                  limitations=['Image heuristics can reject useful views and miss near-duplicates.',
                               'Candidate frames are not automatic recognition evidence or verified labels.',
                               'Complete scan evaluation must use the original sequence, including rejected frames.'])
    (destination/'candidates.json').write_text(json.dumps(report, indent=2)+'\n')
    return report
