"""Synthetic fixtures exercise intake rules; no fixture is collected ground truth."""
import copy
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import cv2
import numpy as np
import pytest

from src.scanning.capture_protocol import (
    make_plan, review_digest, sequence_review_digest, split_digest, validate_capture_manifest)

FEN = '8/8/8/8/8/8/8/K6k w - - 0 1'


def validate(plan, root, **kwargs):
    return validate_capture_manifest(plan, root, {}, set(), **kwargs)


def captured(tmp_path, split='train', *, source='camera', day=1, salt=1):
    plan = make_plan()
    session = next(s for s in plan['sessions'] if s['split'] == split)
    position = next(p for p in plan['positions'] if p['id'] == session['position_id'])
    position['fen'] = FEN
    image = np.random.default_rng(salt).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    raw = cv2.imencode('.jpg', image)[1].tobytes()
    frame_sha = hashlib.sha256(raw).hexdigest()
    created = f'2026-11-{day:02d}T10:00:00+00:00'
    original = dict(frame_id='frame_0001', path='frames/frame_0001.jpg', timestamp_ms=100,
                    width=100, height=100, source_sha256=frame_sha, board_epoch=0,
                    source=source, view_group='white-side', visibility='unknown', geometry=None)
    archive_manifest = dict(schema_version='guided_capture_v1', session_id=f'scan_{split}_{salt}',
                            created_at=created, metadata={'position_id': position['id'], 'fen': FEN},
                            board={'square_cm': 4, 'border_cm': 3.5},
                            frames=[original, {**original, 'frame_id': 'frame_0002',
                                               'path': 'frames/frame_0002.jpg', 'timestamp_ms': 3100}], events=[])
    archive = tmp_path/f'{split}_{salt}.zip'
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr('manifest.json', json.dumps(archive_manifest))
        for f in archive_manifest['frames']:
            zipped.writestr(f['path'], raw)
    session['archive'] = dict(path=archive.name, sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                              recorder_session_id=archive_manifest['session_id'])
    session['started_at'] = created
    session['view_intervals'] = [dict(start_ms=0, end_ms=5000, camera_orientation='white_near_camera',
                                     elevation='medium', notes='Synthetic fixture timeline')]
    frame = dict(id=f'{split}_{salt}_frame', session_id=session['id'], archive_frame_id='frame_0001',
                 source_sha256=frame_sha, width=100, height=100, board_epoch=0,
                 timestamp_ms=100, near_duplicate_group_id=f'{split}_{salt}_view_1',
                 camera_orientation='white_near_camera', elevation='medium', elevation_degrees=60,
                 elevation_method='estimated', selected_for_matrix=True, quality='usable', quality_reasons=[],
                 verified_corners_px=[[10, 10], [90, 10], [90, 90], [10, 90]],
                 verified_semantic_orientation='white_near_camera', reviews={})
    for kind in ('quality', 'corners', 'orientation', 'placement'):
        frame['reviews'][kind] = dict(status='verified', reviewer='synthetic-test-fixture',
                                      reviewed_at=f'2026-11-{day:02d}T11:00:00+00:00',
                                      annotation_sha256=review_digest(frame, session, position, kind))
    plan['frames'] = [frame]
    return plan, session, position, frame, archive


def resign(frame, session, position):
    for kind in frame['reviews']:
        frame['reviews'][kind]['annotation_sha256'] = review_digest(frame, session, position, kind)


def test_empty_plan_is_not_collected_data(tmp_path):
    plan = make_plan()
    result = validate(plan, tmp_path)
    assert len(plan['positions']) == 14
    assert result['planned_sessions'] == 52
    assert result['expected_matrix_cells'] == 624
    assert result['captured_sessions'] == result['frame_count'] == result['fully_reviewed_frames'] == 0
    assert result['selected_frame_ids'] == []
    assert not result['collection_complete']
    assert result['matrix_by_split'] == {s: {'expected': n, 'verified': 0}
                                         for s, n in [('pilot', 48), ('train', 384), ('val', 96), ('final_holdout', 96)]}
    assert plan == json.loads((Path(__file__).resolve().parents[1]/
                               'configs/capture/a3_plan.json').read_text())


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'type', 'version', 'duplicate_id'])
def test_invalid_manifest_shape_or_ids_rejected(tmp_path, mutation):
    plan = make_plan()
    if mutation == 'missing':
        del plan['sessions'][0]['independence_group_id']
    elif mutation == 'extra':
        plan['positions'][0]['auto_verified'] = True
    elif mutation == 'type':
        plan['positions'][0]['previously_seen'] = 1
    elif mutation == 'version':
        plan['schema_version'] = 'wrong'
    else:
        plan['positions'].append(copy.deepcopy(plan['positions'][0]))
        plan['split_lock_sha256'] = split_digest(plan)
    with pytest.raises(ValueError):
        validate(plan, tmp_path)


def test_split_lock_detects_changes(tmp_path):
    plan = make_plan()
    plan['sessions'][0]['independence_group_id'] = 'new_group'
    with pytest.raises(ValueError, match='lock'):
        validate(plan, tmp_path)


@pytest.mark.parametrize('kind', ['placement', 'session_group', 'session_split', 'seen'])
def test_leakage_rejected(tmp_path, kind):
    plan = make_plan()
    train = next(p for p in plan['positions'] if p['split'] == 'train')
    val = next(p for p in plan['positions'] if p['split'] == 'val')
    if kind == 'placement':
        train['fen'] = FEN
        val['fen'] = FEN.replace(' w ', ' b ')  # Same placement despite different game history.
    elif kind == 'session_group':
        t = next(s for s in plan['sessions'] if s['split'] == 'train')
        next(s for s in plan['sessions'] if s['split'] == 'val')['independence_group_id'] = t['independence_group_id']
    elif kind == 'session_split':
        next(s for s in plan['sessions'] if s['split'] == 'val')['split'] = 'train'
    else:
        val['previously_seen'] = True
    plan['split_lock_sha256'] = split_digest(plan)
    with pytest.raises(ValueError):
        validate(plan, tmp_path)


def test_historical_position_rejected_from_fresh_holdout(tmp_path):
    plan = make_plan()
    next(p for p in plan['positions'] if p['split'] == 'final_holdout')['fen'] = FEN
    with pytest.raises(ValueError, match='Historical placement'):
        validate_capture_manifest(plan, tmp_path, {'legacy': {'fen': FEN}}, set())


def test_real_archive_contract_and_review_selection(tmp_path):
    plan, session, position, frame, archive = captured(tmp_path)
    result = validate(plan, tmp_path, purpose='training')
    assert result['selected_frame_ids'] == [frame['id']]
    assert result['verified_matrix_cells'] == 1
    assert result['frame_count'] == result['fully_reviewed_frames'] == 1
    assert not result['collection_complete']
    with pytest.raises(ValueError, match='incomplete'):
        validate(plan, tmp_path, require_complete=True)
    with pytest.raises(ValueError, match='No fully reviewed'):
        validate(plan, tmp_path, purpose='validation')


def test_holdout_never_selected_for_training_or_validation(tmp_path):
    plan, session, position, frame, archive = captured(tmp_path, split='final_holdout')
    for purpose in ('training', 'validation'):
        with pytest.raises(ValueError, match='No fully reviewed'):
            validate(plan, tmp_path, purpose=purpose)
    with pytest.raises(ValueError, match='sealed'):
        validate(plan, tmp_path, purpose='final_evaluation')
    assert validate(plan, tmp_path, purpose='final_evaluation', release_final=True)['selected_frame_ids'] == [frame['id']]


@pytest.mark.parametrize('kind', ['corners', 'placement', 'orientation', 'quality'])
def test_reviews_are_bound_to_annotation_and_pixels(tmp_path, kind):
    plan, session, position, frame, archive = captured(tmp_path)
    if kind == 'corners':
        frame['verified_corners_px'][0][0] += 1
    elif kind == 'placement':
        position['fen'] = '8/8/8/8/8/8/7k/K7 w - - 0 1'
    elif kind == 'orientation':
        frame['verified_semantic_orientation'] = 'black_near_camera'
    else:
        frame['quality_reasons'] = ['changed quality judgment']
    with pytest.raises(ValueError, match='signature'):
        validate(plan, tmp_path)


@pytest.mark.parametrize('kind', ['hash', 'dimensions', 'demo', 'missing_review', 'outside', 'epoch', 'unknown_frame', 'corners'])
def test_bad_capture_and_false_verified_claims_rejected(tmp_path, kind):
    plan, session, position, frame, archive = captured(tmp_path, source='demo' if kind == 'demo' else 'camera')
    if kind == 'hash':
        archive.write_bytes(archive.read_bytes()+b'changed')
    elif kind == 'dimensions':
        frame['width'] = 101
    elif kind == 'missing_review':
        frame['reviews']['orientation']['reviewer'] = None
    elif kind == 'outside':
        session['archive']['path'] = '../outside.zip'
    elif kind == 'epoch':
        frame['board_epoch'] = 1
    elif kind == 'unknown_frame':
        frame['archive_frame_id'] = 'does_not_exist'
    elif kind == 'corners':
        frame['verified_corners_px'] = [[0, 0]]*4
        resign(frame, session, position)
    with pytest.raises(ValueError):
        validate(plan, tmp_path)


def test_excluded_frame_is_retained_but_not_model_eligible(tmp_path):
    plan, session, position, frame, archive = captured(tmp_path)
    frame['quality'] = 'excluded'
    frame['quality_reasons'] = ['unintended severe motion blur']
    resign(frame, session, position)
    result = validate(plan, tmp_path)
    assert result['frame_count'] == 1
    assert result['fully_reviewed_frames'] == result['verified_matrix_cells'] == 0


def test_archive_frame_aliases_cannot_inflate_coverage(tmp_path):
    plan, session, position, frame, archive = captured(tmp_path)
    plan['frames'].append({**copy.deepcopy(frame), 'id': 'alias'})
    with pytest.raises(ValueError, match='registered more than once'):
        validate(plan, tmp_path)


def test_schema_unknown_keyword_is_not_silently_ignored():
    from src.scanning.capture_protocol import _shape
    with pytest.raises(ValueError, match='Unsupported schema'):
        _shape('anything', {'unsupportedConstraint': True}, {})


def test_capture_days_cannot_cross_splits_even_with_different_group_names(tmp_path):
    plan, session, position, frame, archive = captured(tmp_path, split='train', salt=1)
    other, s2, p2, f2, a2 = captured(tmp_path, split='val', salt=2)
    # Different placement avoids conflating the two separate leakage checks.
    next(p for p in plan['positions'] if p['id'] == p2['id'])['fen'] = '8/8/8/8/8/8/7k/K7 w - - 0 1'
    next(s for s in plan['sessions'] if s['id'] == s2['id']).update(s2)
    with pytest.raises(ValueError, match='capture_day'):
        validate(plan, tmp_path)


def test_historical_image_hashes_and_missing_inventory_are_rejected(tmp_path):
    plan, session, position, frame, archive = captured(tmp_path, split='final_holdout')
    with pytest.raises(ValueError, match='Historical source image'):
        validate_capture_manifest(plan, tmp_path, {}, {frame['source_sha256']})
    with pytest.raises(ValueError, match='inventory unavailable'):
        validate_capture_manifest(plan, tmp_path, {}, None)


def test_duplicate_pixels_in_unregistered_archive_frames_cannot_cross_splits(tmp_path):
    plan, session, position, frame, archive = captured(tmp_path, split='train', day=1, salt=1)
    other, s2, p2, f2, a2 = captured(tmp_path, split='val', day=2, salt=1)
    next(p for p in plan['positions'] if p['id'] == p2['id'])['fen'] = '8/8/8/8/8/8/7k/K7 w - - 0 1'
    next(s for s in plan['sessions'] if s['id'] == s2['id']).update(s2)
    # f2 need not be registered: leakage checks include every frame inside each ZIP.
    with pytest.raises(ValueError, match='source_image'):
        validate(plan, tmp_path)


def test_cli_plan_creation_and_reports_never_overwrite(tmp_path, monkeypatch):
    from scripts import validate_capture_manifest as cli
    path = tmp_path/'plan.json'
    monkeypatch.setattr('sys.argv', ['validate', '--init-plan', str(path)])
    cli.main()
    original = path.read_bytes()
    with pytest.raises(SystemExit):
        cli.main()
    assert path.read_bytes() == original
    output = tmp_path/'audit.json'
    monkeypatch.setattr('sys.argv', ['validate', str(path), '--report', str(output)])
    cli.main()
    assert json.loads(output.read_text())['frame_count'] == 0
    with pytest.raises(SystemExit):
        cli.main()
    assert path.read_bytes() == original


def test_proposed_digest_command_never_approves_or_changes_an_annotation(tmp_path, monkeypatch, capsys):
    from scripts import validate_capture_manifest as cli
    plan, session, position, frame, archive = captured(tmp_path)
    frame['reviews']['orientation'] = dict(status='pending', reviewer=None, reviewed_at=None, annotation_sha256=None)
    frame['verified_semantic_orientation'] = None
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps(plan))
    original = path.read_bytes()
    monkeypatch.setattr('sys.argv', ['validate', str(path), '--print-review-digests'])
    cli.main()
    output = json.loads(capsys.readouterr().out)
    assert output['mode'] == 'proposed_annotation_digests_not_validation'
    assert path.read_bytes() == original
    assert validate(plan, tmp_path)['fully_reviewed_frames'] == 0


def test_full_pilot_matrix_can_be_completed_with_unique_reviewed_frames(tmp_path):
    plan = make_plan()
    position = plan['positions'][0]
    position['fen'] = FEN
    plan['positions'] = [position]
    plan['sessions'] = [s for s in plan['sessions'] if s['position_id'] == position['id']]
    plan['frames'] = []
    for index, session in enumerate(plan['sessions']):
        base, source_session, p, frame, archive = captured(tmp_path, split='pilot', salt=index+10)
        session['started_at'] = source_session['started_at']
        session['archive'] = source_session['archive']
        session['view_intervals'] = []
        with ZipFile(archive) as zipped:
            archived = json.loads(zipped.read('manifest.json'))
        archived['frames'] = []
        files = []
        for side_index, side in enumerate(plan['matrix']['orientations']):
            for elevation_index, (elevation, degrees) in enumerate([('low', 35), ('medium', 60), ('overhead', 90)]):
                k = side_index*3+elevation_index
                image = np.random.default_rng(index*12+k).integers(0, 256, (100, 100, 3), dtype=np.uint8)
                raw = cv2.imencode('.jpg', image)[1].tobytes()
                f = copy.deepcopy(frame)
                f.update(id=f'pilot_{index}_{k}', session_id=session['id'], archive_frame_id=f'frame_{k:04d}',
                         source_sha256=hashlib.sha256(raw).hexdigest(), camera_orientation=side,
                         verified_semantic_orientation=side, elevation=elevation, elevation_degrees=degrees,
                         timestamp_ms=100+k*3000, near_duplicate_group_id=f'pilot_{index}_{k}')
                resign(f, session, position)
                plan['frames'].append(f)
                source = dict(frame_id=f['archive_frame_id'], path=f'frames/{k:04d}.jpg', timestamp_ms=f['timestamp_ms'],
                              width=100, height=100, source_sha256=f['source_sha256'], board_epoch=0,
                              source='camera', view_group='other', visibility='unknown', geometry=None)
                archived['frames'].append(source)
                files.append((source['path'], raw))
                session['view_intervals'].append(dict(start_ms=k*3000, end_ms=(k+1)*3000,
                                                     camera_orientation=side, elevation=elevation, notes='Synthetic'))
        with ZipFile(archive, 'w') as zipped:
            zipped.writestr('manifest.json', json.dumps(archived))
            for name, raw in files:
                zipped.writestr(name, raw)
        session['archive']['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    plan['split_lock_sha256'] = split_digest(plan)
    result = validate(plan, tmp_path, require_complete=True)
    assert result['collection_complete']
    assert result['expected_matrix_cells'] == result['verified_matrix_cells'] == 24


def test_full_scan_selection_keeps_sequence_when_still_frame_is_unsuitable(tmp_path):
    plan, session, position, frame, archive = captured(tmp_path, split='val')
    frame['quality'] = 'excluded'
    frame['quality_reasons'] = ['blur unsuitable for recognition training']
    resign(frame, session, position)
    session['sequence_review'] = dict(status='verified', reviewer='synthetic-fixture',
                                       reviewed_at='2026-11-01T11:00:00Z',
                                       annotation_sha256=sequence_review_digest(session, position))
    result = validate(plan, tmp_path, purpose='scan_validation')
    assert result['selected_session_ids'] == [session['id']]
    assert result['selected_frame_ids'] == []
    assert result['sequence_diagnostics'][session['id']]['source_frames'] == 2
    with pytest.raises(ValueError, match='No fully reviewed'):
        validate(plan, tmp_path, purpose='validation')


def test_scan_holdout_requires_explicit_release_and_unchanged_timeline(tmp_path):
    plan, session, position, frame, archive = captured(tmp_path, split='final_holdout')
    session['sequence_review'] = dict(status='verified', reviewer='synthetic-fixture',
                                       reviewed_at='2026-11-01T11:00:00Z',
                                       annotation_sha256=sequence_review_digest(session, position))
    with pytest.raises(ValueError, match='sealed'):
        validate(plan, tmp_path, purpose='scan_final_evaluation')
    assert validate(plan, tmp_path, purpose='scan_final_evaluation', release_final=True)['selected_session_ids']
    session['view_intervals'][0]['notes'] = 'changed metadata'
    with pytest.raises(ValueError, match='signature'):
        validate(plan, tmp_path, purpose='scan_final_evaluation', release_final=True)


@pytest.mark.parametrize('problem', ['timestamp', 'interval_gap', 'view_mismatch', 'missing_intervals'])
def test_sequence_metadata_must_match_original_timeline(tmp_path, problem):
    plan, session, position, frame, archive = captured(tmp_path)
    if problem == 'timestamp':
        frame['timestamp_ms'] += 1
    elif problem == 'interval_gap':
        session['view_intervals'] = [dict(start_ms=0, end_ms=1000, camera_orientation=None, elevation=None, notes=''),
                                     dict(start_ms=2000, end_ms=5000, camera_orientation=None, elevation=None, notes='')]
    elif problem == 'view_mismatch':
        session['view_intervals'][0]['camera_orientation'] = 'black_near_camera'
    else:
        session['view_intervals'] = []
        session['sequence_review'] = dict(status='verified', reviewer='fixture', reviewed_at='2026-11-01T11:00:00Z',
                                           annotation_sha256=sequence_review_digest(session, position))
    with pytest.raises(ValueError):
        validate(plan, tmp_path)


def test_reviewed_repeats_are_not_training_selections(tmp_path):
    plan, session, position, frame, archive = captured(tmp_path)
    frame['selected_for_matrix'] = False
    assert validate(plan, tmp_path)['fully_reviewed_frames'] == 1
    with pytest.raises(ValueError, match='No fully reviewed'):
        validate(plan, tmp_path, purpose='training')


@pytest.mark.parametrize('problem', ['temporal_gap', 'near_duplicate_group', 'cap'])
def test_recognition_subset_cannot_be_inflated(tmp_path, problem):
    plan, session, position, frame, archive = captured(tmp_path)
    other = copy.deepcopy(frame)
    other.update(id='second_candidate', archive_frame_id='frame_0002', timestamp_ms=3100,
                 near_duplicate_group_id='other_group', camera_orientation='black_near_camera',
                 verified_semantic_orientation='black_near_camera')
    # Give the second source different bytes and a different legitimate view interval.
    with ZipFile(archive) as zipped:
        m = json.loads(zipped.read('manifest.json'))
        first = zipped.read(m['frames'][0]['path'])
    image = np.random.default_rng(99).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    second = cv2.imencode('.jpg', image)[1].tobytes()
    other['source_sha256'] = hashlib.sha256(second).hexdigest()
    m['frames'][1]['source_sha256'] = other['source_sha256']
    session['view_intervals'] = [dict(start_ms=0, end_ms=2000, camera_orientation='white_near_camera', elevation='medium', notes=''),
                                 dict(start_ms=2000, end_ms=5000, camera_orientation='black_near_camera', elevation='medium', notes='')]
    if problem == 'temporal_gap':
        plan['frame_selection']['min_gap_ms'] = 4000
    elif problem == 'near_duplicate_group':
        other['near_duplicate_group_id'] = frame['near_duplicate_group_id']
    else:
        plan['frame_selection']['max_frames_per_session'] = 1
    with ZipFile(archive, 'w') as zipped:
        zipped.writestr('manifest.json', json.dumps(m))
        zipped.writestr(m['frames'][0]['path'], first)
        zipped.writestr(m['frames'][1]['path'], second)
    session['archive']['sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    resign(other, session, position)
    plan['frames'].append(other)
    plan['split_lock_sha256'] = split_digest(plan)
    with pytest.raises(ValueError, match='spacing|Near-duplicate|cap exceeded'):
        validate(plan, tmp_path)
