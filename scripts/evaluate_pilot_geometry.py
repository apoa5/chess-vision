"""Paired geometry-only pilot audit using unchanged legacy and B.1b policies."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from scripts.ingest_native_pilot import contact_sheet
from src.scanning.native_pilot import sha256_file
from src.utils.config import load_config
from src.utils.image_io import load_image
from src.vision.dataset import chess_square_at
from src.vision.geometry_audit import context_transform, project, raw_grid_overlay
from src.vision.geometry_reliability import prepare_reliable_raw_photo
from src.vision.position_pipeline import PreparationError, prepare_raw_photo


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def evaluate_frame(image_path, config, destination, policy):
    destination.mkdir(parents=True, exist_ok=False)
    record = dict(policy=policy, completed=False, returned_polygon=None, predicted_camera_orientation=None,
                  geometry_correctness='unverified', semantic_orientation_correctness='unverified',
                  visual_review='pending', failure_stage=None, recovery_attempted=False)
    start = time.perf_counter()
    try:
        function = prepare_raw_photo if policy == 'legacy' else prepare_reliable_raw_photo
        rectified, grid, turns, provenance = function(image_path, config, destination)
        record.update(completed=True, failure_stage=None, provenance=provenance,
                      returned_polygon=provenance['source_corners'], rotation_quarters_ccw=turns,
                      predicted_camera_orientation=['white_near_camera', 'white_left', 'black_near_camera', 'white_right'][turns])
        transform = np.linalg.inv(context_transform(provenance['source_corners'], rectified.board_bounds))
        polygons = project(grid['square_polygons'], transform)
        ids = [chess_square_at(r, c, turns) for r in range(8) for c in range(8)]
        overlay = raw_grid_overlay(load_image(image_path), polygons, ids)
        record['source_square_polygons'] = polygons.tolist()
        record['semantic_square_ids'] = ids
    except PreparationError as error:
        record.update(failure_stage=error.stage, error=str(error))
        overlay = load_image(image_path)
    except (OSError, RuntimeError, ImportError, ValueError) as error:
        record.update(failure_stage='dependency_or_runtime', error=f'{type(error).__name__}: {error}')
        overlay = load_image(image_path)
    detection = destination/('legacy/detection.json' if policy == 'marked-board' else 'detection.json')
    if detection.exists():
        raw = json.loads(detection.read_text())
        record.update(localization_gate_passed=raw['success'], localization_proposal=raw['corners'])
        if not record['completed'] and raw['corners'] is not None:
            cv2.polylines(overlay, [np.rint(raw['corners']).astype(np.int32)], True, (0, 170, 255), 2)
        candidates = raw.get('diagnostics', {}).get('top_candidates', [])
        record['leading_rejected_polygon'] = candidates[0]['corners'] if not raw['success'] and candidates else None
    recovery = destination/'recovery.json'
    record['recovery_attempted'] = recovery.exists()
    if recovery.exists():
        result = json.loads(recovery.read_text())
        record.update(recovery_status=result.get('status'), recovery_reason=result.get('reason'),
                      recovery_path=record.get('provenance', {}).get('geometry_reliability', {}).get('path'))
    if not record['completed']:
        cv2.putText(overlay, record['failure_stage'], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
    record.update(elapsed_seconds=time.perf_counter()-start, overlay_path=str(destination/'geometry_overlay.png'))
    cv2.imwrite(str(destination/'geometry_overlay.png'), overlay)
    write(destination/'result.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():parser.error('New output directory required')
    manifest = json.loads((args.native_run/'manifest.json').read_text())
    if manifest['designation'] != 'development_only' or manifest['training_eligible'] or manifest['final_holdout']:
        parser.error('Development-only pilot required')
    protected = manifest['protected_sha256'] | manifest['media_sha256']
    for path, expected in protected.items():
        if sha256_file(path) != expected:raise ValueError('Protected input changed: '+path)
    config = load_config()
    cv2.setNumThreads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version='pilot_geometry_audit_v1', native_manifest_sha256=sha256_file(args.native_run/'manifest.json'),
                  config_sha256=sha256_file('config/settings.yaml'), source_sha256=sha256_file(Path(__file__)),
                  policies=['legacy', 'marked-board'], frames=[], reference_geometry_available=False,
                  limitation='Completion is not geometry or orientation correctness; image overlays require separate review. No piece classifier is loaded.')
    for session in manifest['sessions']:
        session_id = session['capture_session_id']
        for number, frame in enumerate(session['candidates'], 1):
            if sha256_file(frame['path']) != frame['derivative_sha256']:raise ValueError('Candidate image changed')
            row = {key:frame[key] for key in ['frame_id', 'path', 'pts', 'time_base', 'timestamp_ms', 'relative_timestamp_ms', 'derivative_sha256']}
            row.update(session_id=session_id, position_id=session['position_id'], candidate_number=number, policies={})
            for policy in ['legacy', 'marked-board']:
                row['policies'][policy] = evaluate_frame(frame['path'], config, args.output_dir/session_id/f'candidate_{number:02d}'/policy, policy)
            if row['policies']['legacy']['completed']:
                if not row['policies']['marked-board']['completed']:raise ValueError('Legacy success regressed under opt-in policy')
                if row['policies']['legacy']['source_square_polygons'] != row['policies']['marked-board']['source_square_polygons']:
                    raise ValueError('Legacy-success grid changed under opt-in policy')
            report['frames'].append(row)
            write(args.output_dir/'results.json', report)
            print(session['position_category'], number, [(key, value['completed'], value['failure_stage']) for key, value in row['policies'].items()], flush=True)
        for policy in ['legacy', 'marked-board']:
            rows = [r for r in report['frames'] if r['session_id'] == session_id]
            contact_sheet([(r['policies'][policy]['overlay_path'], f"{r['candidate_number']:02d} {r['timestamp_ms']/1000:.2f}s {'complete' if r['policies'][policy]['completed'] else r['policies'][policy]['failure_stage']}") for r in rows],
                          args.output_dir/f'{session_id}_{policy}_contact.png')
    report['counts'] = {policy:dict(completed=sum(r['policies'][policy]['completed'] for r in report['frames']),
                          failures=dict(Counter(r['policies'][policy]['failure_stage'] for r in report['frames'] if not r['policies'][policy]['completed'])),
                          recovery_attempts=sum(r['policies'][policy]['recovery_attempted'] for r in report['frames'])) for policy in report['policies']}
    write(args.output_dir/'results.json', report)
    for path, expected in protected.items():
        if sha256_file(path) != expected:raise ValueError('Protected input changed: '+path)
    write(args.output_dir/'preservation.json', dict(protected_inputs_unchanged=True, original_media_unchanged=True))


if __name__ == '__main__':main()
