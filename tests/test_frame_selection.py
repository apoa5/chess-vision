"""Only synthetic sequence fixtures; no camera ground truth is manufactured."""
import hashlib
import json
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from src.scanning.frame_selection import extract_candidates


def sequence(tmp_path, images, times=None):
    times = times if times is not None else [i*3000 for i in range(len(images))]
    frames, files = [], []
    for i, (image, timestamp) in enumerate(zip(images, times)):
        raw = cv2.imencode('.jpg', image)[1].tobytes()
        path = f'frames/{i}.jpg'
        frames.append(dict(frame_id=f'frame_{i}', path=path, timestamp_ms=timestamp,
                           source_sha256=hashlib.sha256(raw).hexdigest(), width=image.shape[1], height=image.shape[0],
                           board_epoch=0, source='camera', visibility='unknown', view_group='other', geometry=None))
        files.append((path, raw))
    manifest = dict(schema_version='guided_capture_v1', session_id='synthetic_sequence', frames=frames,
                    board=dict(square_cm=4, border_cm=3.5), events=[], metadata={})
    path = tmp_path/'sequence.zip'
    with ZipFile(path, 'w') as zipped:
        zipped.writestr('manifest.json', json.dumps(manifest))
        for name, raw in files:
            zipped.writestr(name, raw)
    return path, manifest


def images(count):
    rng = np.random.default_rng(13)
    return [rng.integers(20, 235, (120, 120, 3), dtype=np.uint8) for _ in range(count)]


def test_selected_images_preserve_exact_bytes_timestamps_and_no_verification(tmp_path):
    archive, manifest = sequence(tmp_path, images(8))
    original = archive.read_bytes()
    result = extract_candidates(archive, tmp_path/'out', max_frames=4)
    assert len(result['candidates']) == 4
    assert len(result['timeline']) == 8
    assert not result['ground_truth_verified']
    assert result['candidates'][-1]['timestamp_ms'] >= 18000  # Uniform bins preserve late views.
    with ZipFile(archive) as zipped:
        for row in result['candidates']:
            source = next(f for f in manifest['frames'] if f['frame_id'] == row['archive_frame_id'])
            assert (tmp_path/'out'/row['path']).read_bytes() == zipped.read(source['path'])
            assert row['timestamp_ms'] == source['timestamp_ms']
            assert row['review_status'] == 'pending' and row['camera_orientation'] is None
    assert archive.read_bytes() == original
    repeated = extract_candidates(archive, tmp_path/'repeat', max_frames=4)
    assert result == repeated
    with pytest.raises(ValueError, match='already exists'):
        extract_candidates(archive, tmp_path/'out')


def test_repeated_near_identical_views_do_not_create_many_training_candidates(tmp_path):
    image = images(1)[0]
    # Different JPEG bytes with nearly identical appearance.
    almost = np.clip(image.astype(int)+1, 0, 255).astype(np.uint8)
    archive, _ = sequence(tmp_path, [image, almost, image, almost])
    result = extract_candidates(archive, tmp_path/'out')
    assert len(result['candidates']) == 1
    assert any('near_duplicate_of_accepted_candidate' in r['reasons'] for r in result['timeline'])


def test_bad_frames_remain_in_timeline_even_when_no_candidates_are_selected(tmp_path):
    archive, _ = sequence(tmp_path, [np.zeros((120, 120, 3), np.uint8), np.full((120, 120, 3), 255, np.uint8)])
    result = extract_candidates(archive, tmp_path/'out')
    assert result['candidates'] == [] and len(result['timeline']) == 2
    assert all('blur_below_threshold' in r['reasons'] for r in result['timeline'])


@pytest.mark.parametrize('times', [[0], [1000, 1000], [3000, 1000]])
def test_single_still_or_invalid_time_order_rejected(tmp_path, times):
    archive, _ = sequence(tmp_path, images(len(times)), times)
    with pytest.raises(ValueError, match='time-ordered'):
        extract_candidates(archive, tmp_path/'out')


def test_temporal_gap_and_cap_are_enforced(tmp_path):
    archive, _ = sequence(tmp_path, images(20), [i*500 for i in range(20)])
    result = extract_candidates(archive, tmp_path/'out', max_frames=12)
    times = [r['timestamp_ms'] for r in result['candidates']]
    assert len(times) <= 12
    assert all(b-a >= 2000 for a, b in zip(times, times[1:]))
