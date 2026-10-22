"""Native timestamp integrity and review/split gates; synthetic frames are not GT."""
import copy
from fractions import Fraction

import numpy as np
import pytest

from scripts.validate_native_pilot import validate_session
from src.scanning.native_pilot import pts_record, select_candidates


def fixture():
    frames = [dict(frame_id=f'f{i}', decode_index=i, width=40, height=40,
                   display_rotation_degrees=0, **pts_record(t, Fraction(1, 600), Fraction(1, 6)))
              for i, t in enumerate([100, 120, 160, 1600])]
    candidate = frames[-1] | dict(source_video_sha256='abc', split='pilot', training_eligible=False)
    return dict(schema_version='native_pilot_session_v1', split='pilot', designation='development_only',
                source_video=dict(sha256='abc'), original_frame_index=frames, decoded_frames=4, candidates=[candidate])


def test_vfr_exact_pts_and_nonzero_origin_are_preserved():
    session = fixture()
    result = validate_session(session, check_files=False)
    assert result['native_pts_integrity'] and not result['ground_truth_verified']
    assert session['original_frame_index'][0]['relative_timestamp_ms'] == 0
    assert session['original_frame_index'][2]['pts_time'] == [4, 15]
    assert session['original_frame_index'][2]['timestamp_ms'] != 2/30*1000


@pytest.mark.parametrize('pts,base', [(None, Fraction(1, 30)), (0, 0), (0, -1)])
def test_missing_pts_or_bad_time_base_never_uses_nominal_fps(pts, base):
    with pytest.raises(ValueError):pts_record(pts, base)


@pytest.mark.parametrize('field,value', [('timestamp_ms', 999), ('pts', 1700), ('source_video_sha256', 'wrong'), ('training_eligible', True)])
def test_candidate_provenance_or_training_promotion_cannot_be_silently_changed(field, value):
    session = fixture();session['candidates'][0][field] = value
    with pytest.raises(ValueError):validate_session(session, check_files=False)


def test_nonmonotonic_or_duplicate_frame_ids_rejected():
    session = fixture();session['original_frame_index'][2] = copy.deepcopy(session['original_frame_index'][1])
    with pytest.raises(ValueError):validate_session(session, check_files=False)


def test_duplicate_frame_selection_is_sparse_and_reports_rejection():
    rows = [dict(timestamp_ms=i*3000, blur_score=100, mean_luma=100, dhash='0', decoded_pixel_sha256='same') for i in range(24)]
    selected = select_candidates(rows)
    assert len(selected) == 1
    assert any('near_duplicate_of_accepted_candidate' in row['reasons'] for row in rows)


def test_all_bad_quality_frames_can_produce_no_candidates_without_fabrication():
    rows = [dict(timestamp_ms=i*3000, blur_score=0, mean_luma=100, dhash=str(i), decoded_pixel_sha256=str(i)) for i in range(4)]
    assert select_candidates(rows) == []


def test_pilot_exposes_existing_corner_anchor_validation_failure_without_fixing_policy():
    from src.vision.perspective import warp_board, order_corners
    polygon = np.float32([[93.51681, 437.57138], [214.24336, 315.75433], [406.0875, 368.97888], [307.93262, 533.2469]])
    assert not np.array_equal(polygon, order_corners(polygon))
    with pytest.raises(ValueError, match='ordered TL'):
        warp_board(np.zeros((848, 464, 3), np.uint8), polygon)
