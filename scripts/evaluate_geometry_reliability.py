"""B.1b development-only candidate inspection and paired raw-photo evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from src.chess.board_state import BoardState
from src.chess.fen import board_from_fen
from src.utils.config import load_config
from src.utils.image_io import load_image, save_image
from src.vision.board_detection import DetectionSettings, detect_board_corners, preprocess_image, _line_boundaries
from src.vision.dataset import chess_square_at, file_hash, review_approved
from src.vision.geometry_audit import context_transform, project, quadrilateral_metrics, raw_grid_overlay
from src.vision.geometry_reliability import POLICY_VERSION, prepare_reliable_raw_photo
from src.vision.inference import PieceClassifier, predict_rectified
from src.vision.position_evaluation import compare_position
from src.vision.position_pipeline import PreparationError

BASELINE = Path('outputs/evaluation/baseline_a1_20261018/positions_run1/summary.json')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def preserved_inputs():
    b1 = read('docs/evaluation/geometry_results_b1_20261018.json')
    assert file_hash(Path(b1['full_report_path'])) == b1['full_report_sha256']
    protected = dict(read(b1['full_report_path'])['protected_sha256'])
    protected.update(b1['audit_source_sha256'])
    protected.update(b1['diagnostic_sha256'])
    protected.update(read('docs/evaluation/capture_protocol_validation_a3_video_20261018.json')['files_sha256'])
    # Preserve all B.1 outputs, not merely its summary and representative images.
    for path in [*Path(b1['full_report_path']).parent.rglob('*'),
                 Path('docs/evaluation/geometry_results_b1_20261018.json'),
                 Path('docs/evaluation/geometry_alignment_audit.md')]:
        if path.is_file():
            protected[str(path)] = file_hash(path)
    verify(protected)
    return protected


def verify(protected):
    for name, expected in protected.items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f'Protected input changed: {name}')


def reference_seed(record):
    if record.get('source_corners'):
        return np.float32(record['source_corners'])
    g = record['grid_geometry']
    transform, _ = cv2.findHomography(np.float32(g['square_polygons']).reshape(-1, 2),
                                     np.float32(g['source_square_polygons']).reshape(-1, 2), 0)
    x0, y0, x1, y1 = record['board_bounds']
    return project([[x0, y0], [x1-1, y0], [x1-1, y1-1], [x0, y1-1]], transform)


def inspect_candidates(image, settings):
    """Read the frozen detector's entire local list at return, audit use ONLY.

    Profiling is restored even on error. No monkeypatch, scorer modification or
    reference annotation participates in generation/ranking. This diagnostic hook
    deliberately depends on the frozen function's local variable contract.
    """
    captured = []
    def profile(frame, event, value):
        if event == 'return' and frame.f_code is detect_board_corners.__code__ and value is not None:
            captured.extend(dict(c) for c in frame.f_locals['candidates'])
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        detection = detect_board_corners(image, settings)
    finally:
        sys.setprofile(previous)
    if len(captured) != detection.diagnostics['candidate_count']:
        raise ValueError('Frozen candidate inspection contract changed')
    return detection, captured


def seed_agreement(metrics, reference):
    scale = float(np.linalg.norm(reference-np.roll(reference, -1, axis=0), axis=1).mean()/8)
    return metrics['rms_corner_error_squares'] <= .25 and metrics['max_corner_error_px']/scale <= .5


def line_search_diagnostics(image, settings, reference):
    """Compare raw Hough lines and the frozen capped pool, without changing it."""
    gray, edges, scale = preprocess_image(image, settings)
    state = {}
    def profile(frame, event, value):
        if event == 'return' and frame.f_code is _line_boundaries.__code__ and value is not None:
            state.update({k: frame.f_locals.get(k) for k in ['groups', 'lines']})
    previous = sys.getprofile()
    sys.setprofile(profile)
    try:
        _line_boundaries(edges, settings)
    finally:
        sys.setprofile(previous)
    if state.get('lines') is None:
        return {'hough_segments': 0, 'selected_lines': 0}
    segments = np.float32(state['lines']).reshape(-1, 4)
    all_lines = np.column_stack((segments[:, 1]-segments[:, 3], segments[:, 2]-segments[:, 0],
                                 segments[:, 0]*segments[:, 3]-segments[:, 2]*segments[:, 1]))
    all_lines /= np.linalg.norm(all_lines[:, :2], axis=1)[:, None]
    selected = np.concatenate(state['groups'])
    p = np.float32(reference)/scale
    step = float(np.mean(np.linalg.norm(p-np.roll(p, -1, axis=0), axis=1))/8)
    matches = []
    for first, second in zip(p, np.roll(p, -1, axis=0)):
        pair = np.column_stack(([first, second], np.ones(2)))
        def error(lines):
            return np.mean(np.abs(pair@lines.T), axis=0)/step
        matches.append(dict(best_all_hough_line_error_squares=float(error(all_lines).min()),
                            best_selected_line_error_squares=float(error(selected).min())))
    return dict(hough_segments=len(all_lines), selected_lines=len(selected),
                edges_T_R_B_L=matches, selected_normalized_lines=selected.tolist())


def polygon_metrics(actual, reference):
    """Reviewed-reference agreement; not calibrated independent ground truth."""
    actual, reference = np.float32(actual), np.float32(reference)
    outer = lambda p: np.array([p[0, 0], p[7, 1], p[63, 2], p[56, 3]])
    metrics = quadrilateral_metrics(outer(actual), outer(reference))
    metrics['strict_outer_agreement'] = seed_agreement(metrics, outer(reference))
    ious, misses = [], []
    for i, (a, b) in enumerate(zip(actual, reference)):
        intersection, _ = cv2.intersectConvexConvex(a, b)
        union = cv2.contourArea(a)+cv2.contourArea(b)-intersection
        ious.append(float(intersection/union))
        if cv2.pointPolygonTest(b, tuple(map(float, a.mean(axis=0))), False) < 0:
            misses.append(i)
    metrics.update(min_square_iou=min(ious), median_square_iou=float(np.median(ious)),
                   camera_cell_centres_in_corresponding_reference=64-len(misses), missed_camera_cells=misses)
    return metrics


def raw_polygons(grid, corners, bounds):
    return project(grid['square_polygons'], np.linalg.inv(context_transform(corners, bounds)))


def labeled_image(image, before, after, reference, name):
    out = image.copy()
    for points, color in [(reference, (0, 220, 0)), (before, (0, 140, 255)), (after, (255, 150, 0))]:
        if points is not None:
            cv2.polylines(out, [np.rint(points).astype(np.int32)], True, color, 3)
    cv2.rectangle(out, (0, 0), (out.shape[1], 95), (245, 245, 245), -1)
    for i, line in enumerate([name, 'orange: before accepted / best rejected; blue: after fitted grid',
                              'green: reviewed grid' if reference is not None else 'NO reviewed geometry reference']):
        cv2.putText(out, line, (8, 25+28*i), cv2.FONT_HERSHEY_SIMPLEX, .50, (0, 0, 0), 1)
    return out


def summarize(rows, key):
    groups = {}
    for side in sorted({r['orientation'] for r in rows}):
        selected = [r for r in rows if r['orientation'] == side]
        outputs = [r[key] for r in selected]
        completed = [o for o in outputs if o['status'] == 'predicted']
        known = [o for o in completed if o.get('geometry_metrics') is not None]
        groups[side] = dict(attempted=len(outputs), completed=len(completed),
                            failures=dict(Counter(o['failure_stage'] for o in outputs if o['status'] != 'predicted')),
                            localization_accepted=sum(o.get('localization_accepted', False) for o in outputs),
                            known_geometry_completed=len(known),
                            strict_reference_geometry_completed=sum(o['geometry_metrics']['strict_outer_agreement'] for o in known),
                            all_64_centres_agree_completed=sum(o['geometry_metrics']['camera_cell_centres_in_corresponding_reference'] == 64 for o in known),
                            unverified_geometry_completed=len(completed)-len(known),
                            semantic_correct=sum(o.get('orientation_correct', False) for o in completed),
                            exact_boards=sum(o['exact_match'] for o in completed),
                            square_errors=sum(o['piece_errors'] for o in completed))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['candidates', 'line-pool', 'evaluate'])
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--candidate-cache', type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory must be new and unused')
    if args.mode == 'evaluate' and args.candidate_cache is None:
        parser.error('evaluate requires --candidate-cache from the candidates command')
    protected = preserved_inputs()
    prior, config = read(BASELINE), load_config()
    if set(prior['test_positions']) != {'position_006', 'position_010', 'position_016'}:
        raise ValueError('Only the already-inspected historical development images may be evaluated')
    all_prepared = {r['image_id']: r for r in read('data/classification/prepared_boards.json')}
    prepared = {name: r for name, r in all_prepared.items() if r.get('grid_geometry')}
    reviews, metadata, positions = (read('data/metadata/'+n+'.json') for n in ['day9_board_reviews', 'images', 'positions'])
    rows = [r for r in prior['records'] if r['mode'] == 'raw']
    assert len(rows) == 30
    cv2.setNumThreads(1)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = DetectionSettings(**config.get('board_detection', {}))
    report = dict(task='B.1b', policy_version=POLICY_VERSION, protected_sha256=protected, records=[],
                  evidence_scope='Development on the same 30 historical images; no unbiased generalization claim.')
    sources = {str(p): file_hash(p) for p in [Path(__file__), Path('src/vision/geometry_reliability.py'), Path('scripts/predict_position.py')]}
    if args.mode == 'evaluate':
        model = PieceClassifier(config['classifier']['checkpoint_path'], cpu_threads=4, batch_size=16)
    for old in rows:
        name = old['image_id']
        image_path = config['data']['raw']/name
        if file_hash(image_path) != all_prepared[name]['source_sha256']:
            raise ValueError('Raw image differs from frozen dataset: '+name)
        protected[str(image_path)] = all_prepared[name]['source_sha256']
        image = load_image(image_path)
        p = prepared.get(name)
        if p and (not review_approved(p, reviews.get(name)) or file_hash(image_path) != p['source_sha256']):
            raise ValueError('Missing/stale historical geometry review: '+name)
        if args.mode == 'line-pool':
            row = dict(image_id=name, reviewed_geometry_available=p is not None)
            if p:
                row.update(line_search_diagnostics(image, settings, reference_seed(p)))
            report['records'].append(row)
            print(name, 'line pool inspected', flush=True)
            continue
        if args.mode == 'candidates':
            start = time.monotonic()
            detection, candidates = inspect_candidates(image, settings)
            if p:
                ref = reference_seed(p)
                for rank, candidate in enumerate(candidates, 1):
                    candidate.update(rank=rank, reference_metrics=quadrilateral_metrics(candidate['corners'], ref))
            row = dict(image_id=name, source_sha256=file_hash(image_path), status_before=old['status'],
                       failure_stage_before=old.get('failure_stage'), success=detection.success,
                       selected_corners=detection.corners.tolist() if detection.success else None,
                       diagnostics=detection.diagnostics, candidates=candidates, seconds=time.monotonic()-start)
            write(args.output_dir/(Path(name).stem+'.json'), row)
            report['records'].append({k: v for k, v in row.items() if k not in ['candidates', 'diagnostics']})
            print(name, len(candidates), 'generated', flush=True)
            continue
        pool_path = args.candidate_cache/(Path(name).stem+'.json')
        pool = read(pool_path)
        protected[str(pool_path)] = file_hash(pool_path)
        if pool['image_id'] != name or pool['status_before'] != old['status'] or pool['failure_stage_before'] != old.get('failure_stage'):
            raise ValueError('Candidate cache belongs to another baseline attempt')
        historical_detection = read(BASELINE.parent/'raw'/Path(name).stem/'detection.json')
        if pool['success'] != historical_detection['success']:
            raise ValueError('Candidate inspection changed the baseline detection outcome')
        if pool['success'] and not np.allclose(pool['selected_corners'], historical_detection['corners'], atol=.001):
            raise ValueError('Candidate inspection changed the baseline detected corners')
        if pool.get('source_sha256', file_hash(image_path)) != file_hash(image_path):
            raise ValueError('Candidate cache belongs to another source')
        if old['status'] == 'predicted' and not np.allclose(pool['selected_corners'], old['source_corners'], atol=.001):
            raise ValueError('Before detector did not reproduce frozen successful geometry')
        before = dict(old)
        before['localization_accepted'] = pool['success']
        before['orientation_correct'] = old.get('rotation_quarters_ccw') == {'white_near_camera': 0, 'white_left': 1, 'black_near_camera': 2, 'white_right': 3}[metadata[name]['orientation']]
        before['geometry_metrics'] = None
        if old['status'] == 'predicted' and p:
            predictions = sorted(read(old['prediction_path'])['predictions'], key=lambda r: (r['row'], r['col']))
            previous_grid = {'square_polygons': [r['square_polygon'] for r in predictions]}
            before['geometry_metrics'] = polygon_metrics(raw_polygons(previous_grid, old['source_corners'], p['board_bounds']), p['grid_geometry']['source_square_polygons'])
        destination = args.output_dir/Path(name).stem
        destination.mkdir()
        after = dict(status='failed', failure_stage=None, localization_accepted=False, geometry_metrics=None)
        polygon, actual = None, None
        start = time.monotonic()
        try:
            rectified, grid, turns, provenance = prepare_reliable_raw_photo(image_path, config, destination)
            actual = raw_polygons(grid, provenance['source_corners'], rectified.board_bounds)
            polygon = np.float32([actual[0, 0], actual[7, 1], actual[63, 2], actual[56, 3]])
            after.update(localization_accepted=True, rotation_quarters_ccw=turns,
                         orientation_correct=turns == {'white_near_camera': 0, 'white_left': 1, 'black_near_camera': 2, 'white_right': 3}[metadata[name]['orientation']],
                         provenance=provenance)
            if p:
                after['geometry_metrics'] = polygon_metrics(actual, p['grid_geometry']['source_square_polygons'])
            prediction = predict_rectified(model, rectified, grid, turns, padding=200)
            comparison = compare_position(BoardState(prediction['board']['rows'], 'white_at_bottom'), board_from_fen(positions[metadata[name]['position_id']]['fen']))
            after.update(status='predicted', **comparison)
            write(destination/'predictions.json', prediction)
            save_image(destination/'canonical_grid.jpg', raw_grid_overlay(image, actual, [chess_square_at(i//8, i%8, turns) for i in range(64)], targets=('a1', 'd5', 'c3', 'g3')))
        except PreparationError as error:
            after.update(failure_stage=error.stage, reason=str(error))
            # Preserve accepted classical localization even when semantics reject it.
            after['localization_accepted'] = pool['success']
        after['seconds'] = time.monotonic()-start
        reference = None if p is None else np.float32(p['grid_geometry']['source_square_polygons'])[[0, 7, 63, 56], [0, 1, 2, 3]]
        before_polygon = pool['selected_corners'] if pool['success'] else pool['candidates'][0]['corners']
        save_image(destination/'before_after.jpg', labeled_image(image, before_polygon, polygon, reference, name))
        row = dict(image_id=name, orientation=metadata[name]['orientation'], reviewed_geometry_available=p is not None,
                   before=before, after=after)
        if p:
            ref = reference_seed(p)
            best = min(pool['candidates'], key=lambda c: c['reference_metrics']['rms_corner_error_squares'])
            good = [c for c in pool['candidates'] if seed_agreement(c['reference_metrics'], ref)]
            row['candidate_coverage'] = dict(candidate_count=len(pool['candidates']), best_rms_candidate=best,
                                             usable_count=len(good), first_usable_rank=good[0]['rank'] if good else None)
        write(destination/'result.json', row)
        report['records'].append(row)
        print(name, old['status'], '->', after['status'], after.get('piece_errors'), after.get('reason', ''), flush=True)
    if args.mode == 'evaluate':
        report.update(before=summarize(report['records'], 'before'), after=summarize(report['records'], 'after'))
        report['previous_success_regressions'] = [r['image_id'] for r in report['records'] if r['before']['status'] == 'predicted' and (r['after']['status'] != 'predicted' or r['after'].get('incorrect_squares') != r['before']['incorrect_squares'])]
    report['source_sha256'] = sources
    verify(sources)
    verify(protected)
    write(args.output_dir/'summary.json', report)


if __name__ == '__main__':
    main()
