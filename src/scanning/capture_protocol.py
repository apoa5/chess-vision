"""A.3 companion manifests: planned captures, human attestations and split isolation."""
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import itertools
import json
from pathlib import Path
import re
from zipfile import ZipFile

import chess
import cv2
import numpy as np

from src.scanning.replay import replay_session

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT/'configs/capture/capture_manifest.schema.json'
ORIENTATIONS = ['white_near_camera', 'black_near_camera', 'white_left', 'white_right']
PURPOSE_SPLIT = {'training': 'train', 'validation': 'val', 'final_evaluation': 'final_holdout'}
SCAN_PURPOSE_SPLIT = {'scan_validation': 'val', 'scan_final_evaluation': 'final_holdout'}


def pending_review():
    return dict(status='pending', reviewer=None, reviewed_at=None, annotation_sha256=None)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def split_digest(manifest):
    """Assignment lock excludes incoming image bytes and human annotations."""
    return digest({
        'positions': sorted([{k: p[k] for k in ('id', 'split', 'previously_seen')} for p in manifest['positions']], key=lambda p: p['id']),
        'sessions': sorted([{k: s[k] for k in ('id', 'position_id', 'split', 'independence_group_id',
                                             'board_id', 'camera_id', 'environment', 'lighting', 'capture_mode')}
                            for s in manifest['sessions']], key=lambda s: s['id']),
        'matrix': manifest['matrix'], 'boards': manifest['boards'], 'frame_selection': manifest['frame_selection']})


def make_plan():
    """Reserved IDs and empty sessions only; never creates photographs or labels."""
    positions = [dict(id=f'a3_{split}_{i:02d}', split=split, previously_seen=split == 'pilot', fen=None)
                 for split, count in [('pilot', 2), ('train', 8), ('val', 2), ('final_holdout', 2)]
                 for i in range(1, count+1)]
    environments = ['room_A', 'room_B']
    sessions = []
    for p in positions:
        for environment, lighting in itertools.product(environments[:1] if p['split'] == 'pilot' else environments,
                                                       ['diffuse', 'directional']):
            sessions.append(dict(id=f"{p['id']}_{environment}_{lighting}", position_id=p['id'], split=p['split'],
                                 independence_group_id=f"a3_{p['split']}_{environment}_block01",
                                 board_id='wooden_tournament_39cm', camera_id='iphone_13_pro',
                                 environment=environment, lighting=lighting, started_at=None, archive=None,
                                 capture_mode='continuous_sequence', sequence_review=pending_review(), view_intervals=[]))
    plan = dict(schema_version='capture_protocol_v2', dataset_id='a3_sequence_collection_01',
                frame_selection=dict(max_frames_per_session=12, min_gap_ms=2000),
                matrix=dict(environments=environments, lighting=['diffuse', 'directional'],
                            orientations=ORIENTATIONS.copy(), elevations=['low', 'medium', 'overhead']),
                boards=[dict(id='wooden_tournament_39cm', piece_set_id='wooden_tournament_set', square_cm=4, border_cm=3.5)],
                positions=positions, sessions=sessions, frames=[], split_lock_sha256='0'*64)
    plan['split_lock_sha256'] = split_digest(plan)
    return plan


def _shape(value, schema, definitions, path='$'):
    """Enforce the keywords used by our committed JSON Schema, without a new dependency.

    This is intentionally not a general-purpose JSON Schema implementation.
    Unknown keywords fail rather than silently skipping a schema constraint.
    """
    supported = {'$schema', 'title', 'description', '$defs', '$ref', 'type', 'enum', 'const',
                 'properties', 'required', 'additionalProperties', 'items', 'minItems', 'maxItems',
                 'uniqueItems', 'minLength', 'pattern', 'minimum', 'maximum'}
    if set(schema)-supported:
        raise ValueError(f'Unsupported schema keywords: {set(schema)-supported}')
    if '$ref' in schema:
        return _shape(value, definitions[schema['$ref'].removeprefix('#/$defs/')], definitions, path)
    types = {'object': lambda v: isinstance(v, dict), 'array': lambda v: isinstance(v, list),
             'string': lambda v: isinstance(v, str), 'integer': lambda v: type(v) is int,
             'number': lambda v: type(v) in (int, float) and np.isfinite(v),
             'boolean': lambda v: type(v) is bool, 'null': lambda v: v is None}
    wanted = schema.get('type', list(types))
    if isinstance(wanted, str):
        wanted = [wanted]
    if not any(types[t](value) for t in wanted):
        raise ValueError(f'{path}: expected {wanted}')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError(f'{path}: invalid enum value')
    if 'const' in schema and value != schema['const']:
        raise ValueError(f'{path}: incorrect schema version')
    if isinstance(value, dict):
        missing = set(schema.get('required', []))-value.keys()
        extra = value.keys()-schema.get('properties', {}).keys()
        if missing or (schema.get('additionalProperties') is False and extra):
            raise ValueError(f'{path}: missing {sorted(missing)}, unexpected {sorted(extra)}')
        for key, item in value.items():
            if key in schema.get('properties', {}):
                _shape(item, schema['properties'][key], definitions, path+'.'+key)
    elif isinstance(value, list):
        if not schema.get('minItems', 0) <= len(value) <= schema.get('maxItems', float('inf')):
            raise ValueError(f'{path}: invalid array length')
        if schema.get('uniqueItems') and len({json.dumps(v, sort_keys=True) for v in value}) != len(value):
            raise ValueError(f'{path}: duplicate array items')
        for index, item in enumerate(value):
            _shape(item, schema['items'], definitions, f'{path}[{index}]')
    elif isinstance(value, str):
        if len(value.strip()) < schema.get('minLength', 0) or ('pattern' in schema and not re.search(schema['pattern'], value)):
            raise ValueError(f'{path}: invalid string')
    elif type(value) in (int, float):
        if not schema.get('minimum', -float('inf')) <= value <= schema.get('maximum', float('inf')):
            raise ValueError(f'{path}: number out of range')


def _indexed(rows, label):
    result = {r['id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate {label} IDs')
    return result


def _time(value):
    try:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
        if date.tzinfo is None:
            raise ValueError('timezone required')
        return date
    except (ValueError, AttributeError) as exc:
        raise ValueError('Review/capture timestamps must be ISO-8601 with a timezone') from exc


def placement(fen):
    if len(fen.split()) != 6:
        raise ValueError('Supply a six-field FEN; nonvisual fields are annotation metadata')
    return chess.Board(fen).board_fen()


def review_digest(frame, session, position, kind):
    identity = {k: frame[k] for k in ('id', 'archive_frame_id', 'source_sha256', 'width', 'height', 'board_epoch', 'timestamp_ms')}
    identity.update(position_id=position['id'], board_id=session['board_id'], session_id=session['id'])
    annotation = {'quality': [frame['quality'], frame['quality_reasons']],
                  'corners': frame['verified_corners_px'],
                  'orientation': frame['verified_semantic_orientation'],
                  'placement': position['fen']}[kind]
    return digest({'source': identity, 'kind': kind, 'annotation': annotation})


def sequence_review_digest(session, position):
    return digest({'archive': session['archive'], 'position_id': position['id'], 'fen': position['fen'],
                   'board_id': session['board_id'], 'capture_mode': session['capture_mode'],
                   'view_intervals': session['view_intervals']})


def _review(review, expected, earliest):
    if review['status'] == 'pending':
        if any(review[k] is not None for k in ('reviewer', 'reviewed_at', 'annotation_sha256')):
            raise ValueError('Pending review must not claim an attestation')
    else:
        if not review['reviewer'] or not review['reviewer'].strip():
            raise ValueError('Human reviewer required')
        if _time(review['reviewed_at']) < earliest:
            raise ValueError('Review predates capture')
        if review['annotation_sha256'] != expected:
            raise ValueError('Stale or missing image-bound annotation signature')


def _archive(session, root, board):
    reference = session['archive']
    relative = Path(reference['path'])
    path = (root/relative).resolve()
    if relative.is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError('Archive path must stay under --asset-root')
    if hashlib.sha256(path.read_bytes()).hexdigest() != reference['sha256']:
        raise ValueError('Archive hash mismatch')
    replay_session(path)  # Reuse archive safety, all byte hashes, dimensions and geometry checks.
    with ZipFile(path) as zipped:
        manifest = json.loads(zipped.read('manifest.json'))
    if manifest['session_id'] != reference['recorder_session_id']:
        raise ValueError('Recorder session ID mismatch')
    if manifest.get('metadata', {}).get('position_id') != session['position_id']:
        raise ValueError('Recorder position ID mismatch')
    if _time(manifest.get('created_at')) != _time(session['started_at']):
        raise ValueError('started_at must match the recorder created_at timestamp')
    if (board['square_cm'], board['border_cm']) != (4, 3.5):
        raise ValueError('Existing replay supports only the 4 cm square / 3.5 cm border profile')
    if (any(f['source'] != 'camera' or f['board_epoch'] != 0 for f in manifest['frames']) or
            any(e['type'] in ('board_change', 'new_board_epoch') for e in manifest.get('events', []))):
        raise ValueError('Camera-only single-position archives required; no demo or changed-board epochs')
    times = [f['timestamp_ms'] for f in manifest['frames']]
    if len(times) < 2 or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Continuous sequence requires at least two strictly time-ordered frames')
    return {f['frame_id']: f for f in manifest['frames']}


def validate_capture_manifest(manifest, asset_root, legacy_positions, legacy_hashes, *,
                              purpose='audit', require_complete=False, release_final=False):
    schema = json.loads(SCHEMA_PATH.read_text())
    _shape(manifest, schema, schema['$defs'])
    if manifest['split_lock_sha256'] != split_digest(manifest):
        raise ValueError('Split/collection assignment lock mismatch')
    if purpose not in ('audit', *PURPOSE_SPLIT, *SCAN_PURPOSE_SPLIT):
        raise ValueError('Unknown purpose')
    if purpose in ('final_evaluation', 'scan_final_evaluation') and not release_final:
        raise ValueError('Final holdout is sealed; an explicit --release-final is required')
    positions = _indexed(manifest['positions'], 'position')
    sessions = _indexed(manifest['sessions'], 'session')
    boards = _indexed(manifest['boards'], 'board')
    _indexed(manifest['frames'], 'frame')
    legacy_placements = {placement(p['fen']) for p in legacy_positions.values() if p.get('fen')}
    assigned, archives, frame_sources, scan_selected = {}, {}, {}, []
    temporal = {}

    def isolated(key, split):
        if key in assigned and assigned[key] != split:
            raise ValueError(f'Cross-split leakage: {key}')
        assigned[key] = split

    for p in positions.values():
        if p['split'] in ('val', 'final_holdout') and p['previously_seen']:
            raise ValueError('Previously seen positions cannot enter validation or final holdout')
        if p['fen'] is not None:
            label = placement(p['fen'])
            isolated(('placement', label), p['split'])
            if label in legacy_placements and (not p['previously_seen'] or p['split'] in ('val', 'final_holdout')):
                raise ValueError('Historical placement reused as fresh/held-out ground truth')
    for s in sessions.values():
        if s['position_id'] not in positions or s['board_id'] not in boards:
            raise ValueError('Session references unknown position or board')
        if s['split'] != positions[s['position_id']]['split']:
            raise ValueError('Position/session split mismatch')
        isolated(('independence_group', s['independence_group_id']), s['split'])
        if s['environment'] not in manifest['matrix']['environments']:
            raise ValueError('Session environment missing from capture matrix')
        if s['archive'] is not None:
            if s['started_at'] is None:
                raise ValueError('Captured session requires started_at')
            isolated(('capture_day', _time(s['started_at']).date().isoformat()), s['split'])
            for key in ('sha256', 'recorder_session_id'):
                value = s['archive'][key]
                if value in archives:
                    raise ValueError('Archive or recorder session reused across protocol sessions')
                archives[value] = s['id']
            if legacy_hashes is None:
                raise ValueError('Historical image hash inventory unavailable for captured data')
            frames = _archive(s, Path(asset_root), boards[s['board_id']])
            frame_sources[s['id']] = frames
            times = [f['timestamp_ms'] for f in frames.values()]
            intervals = s['view_intervals']
            if any(v['end_ms'] <= v['start_ms'] for v in intervals) or any(
                    a['end_ms'] != b['start_ms'] for a, b in zip(intervals, intervals[1:])):
                raise ValueError('View intervals must be ordered, contiguous, positive-duration intervals')
            if intervals and (intervals[0]['start_ms'] > times[0] or intervals[-1]['end_ms'] <= times[-1]):
                raise ValueError('View intervals do not cover the complete source sequence')
            _review(s['sequence_review'], sequence_review_digest(s, positions[s['position_id']]),
                    _time(s['started_at'])+timedelta(milliseconds=times[-1]))
            if s['sequence_review']['status'] == 'verified':
                if not intervals or positions[s['position_id']]['fen'] is None:
                    raise ValueError('Reviewed scan requires full timeline metadata and reference placement')
                if purpose in SCAN_PURPOSE_SPLIT and s['split'] == SCAN_PURPOSE_SPLIT[purpose]:
                    scan_selected.append(s['id'])
            temporal[s['id']] = dict(source_frames=len(times), first_timestamp_ms=times[0],
                                    last_timestamp_ms=times[-1], max_gap_ms=max(b-a for a, b in zip(times, times[1:])),
                                    view_metadata_complete=bool(intervals))
            for source in frames.values():
                isolated(('source_image', source['source_sha256']), s['split'])
                if source['source_sha256'] in legacy_hashes and (
                        not positions[s['position_id']]['previously_seen'] or s['split'] in ('val', 'final_holdout')):
                    raise ValueError('Historical source image reused as fresh/held-out data')
        elif s['started_at'] is not None:
            raise ValueError('Uncaptured planned session must have null started_at')
        elif s['sequence_review'] != pending_review() or s['view_intervals']:
            raise ValueError('Planned session must not claim reviewed sequence metadata')
    selected, ready, cells, digests = [], [], Counter(), {}
    archive_references, primary_sources, primary_groups = set(), set(), set()
    primary_times = {}
    for frame in manifest['frames']:
        if frame['session_id'] not in sessions:
            raise ValueError('Frame references unknown session')
        session = sessions[frame['session_id']]
        position = positions[session['position_id']]
        source = frame_sources.get(session['id'], {}).get(frame['archive_frame_id'])
        if source is None:
            raise ValueError('Frame is not present in a validated camera archive')
        reference = (session['id'], frame['archive_frame_id'])
        if reference in archive_references:
            raise ValueError('Same archive frame registered more than once')
        archive_references.add(reference)
        for key in ('source_sha256', 'width', 'height', 'board_epoch', 'timestamp_ms'):
            if frame[key] != source[key]:
                raise ValueError(f'Frame differs from archive: {key}')
        if frame['board_epoch'] != 0:
            raise ValueError('One unchanged position per protocol session required')
        intervals = session['view_intervals']
        if intervals:
            interval = next(v for v in intervals if v['start_ms'] <= frame['timestamp_ms'] < v['end_ms'])
            for key in ('camera_orientation', 'elevation'):
                if interval[key] is not None and frame[key] != interval[key]:
                    raise ValueError('Selected frame view disagrees with source timeline')
        isolated(('source_image', frame['source_sha256']), session['split'])
        bounds = {'low': (25, 45), 'medium': (50, 70), 'overhead': (80, 90)}[frame['elevation']]
        if not bounds[0] <= frame['elevation_degrees'] <= bounds[1]:
            raise ValueError('Elevation category and degrees disagree')
        corners = frame['verified_corners_px']
        if corners is not None:
            polygon = np.asarray(corners, np.float32)
            edges = np.roll(polygon, -1, axis=0)-polygon
            crosses = edges[:, 0]*np.roll(edges[:, 1], -1)-edges[:, 1]*np.roll(edges[:, 0], -1)
            if (not cv2.isContourConvex(polygon) or np.any(crosses <= 0) or
                    int(np.argmin(polygon.sum(axis=1))) != 0 or
                    polygon[:, 0].max() >= frame['width'] or polygon[:, 1].max() >= frame['height']):
                raise ValueError('Corners must be in-bounds, convex and image-clockwise TL/TR/BR/BL')
        signatures = {kind: review_digest(frame, session, position, kind) for kind in frame['reviews']}
        digests[frame['id']] = signatures
        for kind, review in frame['reviews'].items():
            _review(review, signatures[kind], _time(session['started_at'])+timedelta(milliseconds=source['timestamp_ms']))
        verified = lambda kind: frame['reviews'][kind]['status'] == 'verified'
        if (corners is not None) != verified('corners'):
            raise ValueError('Verified corners and geometry review must be supplied together')
        if (frame['verified_semantic_orientation'] is not None) != verified('orientation'):
            raise ValueError('Verified semantic orientation requires its own review')
        if verified('orientation') and frame['verified_semantic_orientation'] != frame['camera_orientation']:
            raise ValueError('Correct the capture-side note to match verified orientation')
        if verified('placement') and position['fen'] is None:
            raise ValueError('Placement review requires ground-truth FEN')
        if verified('quality') != (frame['quality'] != 'pending'):
            raise ValueError('Usable/excluded quality decisions require human review')
        if frame['quality'] == 'excluded' and not frame['quality_reasons']:
            raise ValueError('Excluded frame requires a reason')
        eligible = all(verified(k) for k in frame['reviews']) and frame['quality'] == 'usable'
        if eligible:
            ready.append(frame['id'])
        if frame['selected_for_matrix'] and eligible:
            if frame['source_sha256'] in primary_sources:
                raise ValueError('Duplicate source pixels cannot fill independent matrix cells')
            primary_sources.add(frame['source_sha256'])
            group = (session['position_id'], frame['near_duplicate_group_id'])
            if group in primary_groups:
                raise ValueError('Near-duplicate frames cannot both enter recognition selection')
            primary_groups.add(group)
            times = primary_times.setdefault(session['id'], [])
            if any(abs(frame['timestamp_ms']-t) < manifest['frame_selection']['min_gap_ms'] for t in times):
                raise ValueError('Recognition frames violate minimum temporal spacing')
            times.append(frame['timestamp_ms'])
            if len(times) > manifest['frame_selection']['max_frames_per_session']:
                raise ValueError('Recognition frame cap exceeded')
            if purpose in PURPOSE_SPLIT and session['split'] == PURPOSE_SPLIT[purpose]:
                selected.append(frame['id'])
            cell = (session['position_id'], session['environment'], session['lighting'],
                    frame['camera_orientation'], frame['elevation'])
            cells[cell] += 1
            if cells[cell] > 1:
                raise ValueError('Duplicate primary matrix cell; mark repeats selected_for_matrix=false')
    wanted = set()
    for position in positions.values():
        environments = manifest['matrix']['environments'][:1] if position['split'] == 'pilot' else manifest['matrix']['environments']
        wanted.update(itertools.product([position['id']], environments, manifest['matrix']['lighting'],
                                        manifest['matrix']['orientations'], manifest['matrix']['elevations']))
    if set(cells)-wanted:
        raise ValueError('Primary frame lies outside the assigned matrix')
    missing = sorted(wanted-set(cells))
    if require_complete and missing:
        raise ValueError(f'Collection incomplete: {len(missing)} verified matrix cells missing')
    if purpose in PURPOSE_SPLIT and not selected:
        raise ValueError(f'No fully reviewed frames eligible for {purpose}')
    if purpose in SCAN_PURPOSE_SPLIT and not scan_selected:
        raise ValueError(f'No reviewed full sequences eligible for {purpose}')
    return {'schema_version': 'capture_validation_v2', 'dataset_id': manifest['dataset_id'],
            'purpose': purpose, 'planned_sessions': len(sessions), 'captured_sessions': len(frame_sources),
            'frame_count': len(manifest['frames']), 'fully_reviewed_frames': len(ready),
            'expected_matrix_cells': len(wanted), 'verified_matrix_cells': len(cells),
            'matrix_by_split': {split: {
                'expected': sum(positions[c[0]]['split'] == split for c in wanted),
                'verified': sum(positions[c[0]]['split'] == split for c in cells)}
                for split in ('pilot', 'train', 'val', 'final_holdout')},
            'missing_matrix_cells': [list(cell) for cell in missing], 'collection_complete': not missing,
            'selected_frame_ids': selected, 'selected_session_ids': scan_selected,
            'sequence_diagnostics': temporal, 'review_digests': digests,
            'limitations': ['Human attestations cannot be verified as truthful by software.',
                            'Hashes catch exact duplicates; session provenance must group near-duplicates.',
                            'This companion format is not consumed by the legacy training loader.']}
