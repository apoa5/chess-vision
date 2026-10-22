"""Audit frozen A.2 geometry and run explicit historical manual-reference ablations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from src.chess.board_state import BoardState
from src.chess.fen import board_from_fen
from src.utils.config import load_config
from src.utils.image_io import load_image, save_image
from src.vision.dataset import file_hash, chess_square_at, review_approved
from src.vision.dataset_loader import preprocess_context_crop
from src.vision.geometry_audit import context_transform, project, quadrilateral_metrics, independently_rasterized_mask, raw_grid_overlay
from src.vision.grid_geometry import polygon_crop
from src.vision.inference import PieceClassifier, predict_rectified, prediction_overlay
from src.vision.orientation import resolve_orientation, OrientationSettings
from src.vision.position_pipeline import prepare_raw_photo
from src.vision.position_evaluation import compare_position

SIDES = {'white_near_camera': 'bottom', 'white_left': 'left', 'black_near_camera': 'top', 'white_right': 'right'}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def title(image, lines):
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30*len(lines)+10), (245, 245, 245), -1)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (8, 26+30*i), cv2.FONT_HERSHEY_SIMPLEX, .55, (10, 10, 10), 1, cv2.LINE_AA)
    return out


def montage(paths, target):
    thumbs = []
    for path in paths:
        image = load_image(path)
        thumbs.append(cv2.resize(image, (480, 640), interpolation=cv2.INTER_AREA))
    canvas = np.full((((len(thumbs)+1)//2)*640, 960, 3), 245, np.uint8)
    for i, image in enumerate(thumbs):
        canvas[(i//2)*640:(i//2+1)*640, (i%2)*480:(i%2+1)*480] = image
    save_image(target, canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New unused directory')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Refusing to reuse output directory')
    config = load_config()
    cv2.setNumThreads(1)
    baseline = read('configs/baselines/a1_20261018.json')
    protected = dict(baseline['preserved_sha256'])
    a2 = read('docs/evaluation/orientation_results_a2_20261018.json')
    protected[a2['full_report_path']] = a2['full_report_sha256']
    protected.update(read(a2['full_report_path'])['source_sha256'])
    source = Path('outputs/evaluation/baseline_a1_20261018/positions_run1/summary.json')
    protected[str(source)] = file_hash(source)
    for name, expected in protected.items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    prior = read(source)
    if set(prior['test_positions']) != {'position_006', 'position_010', 'position_016'}:
        raise ValueError('This audit is restricted to the already-inspected historical test positions')
    manifest = read(config['classifier']['dataset_paths']['metadata'])
    prepared_path = config['classifier']['dataset_paths']['metadata'].parent/'prepared_boards.json'
    prepared = {r['image_id']: r for r in read(prepared_path)}
    images, positions = read('data/metadata/images.json'), read('data/metadata/positions.json')
    overrides, reviews = read('data/metadata/board_corner_overrides.json'), read('data/metadata/day9_board_reviews.json')
    prior_prepared = {r['image_id']: r for r in prior['records'] if r['mode'] == 'prepared'}
    raw = [r for r in prior['records'] if r['mode'] == 'raw']
    failures = [r for r in raw if r['status'] == 'failed']
    if len(raw) != 30 or len(failures) != 14:
        raise ValueError('Historical attempt count mismatch')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    model = PieceClassifier(config['classifier']['checkpoint_path'], cpu_threads=4, batch_size=16)
    report = dict(task='B.1', source_commit=baseline['git_commit'], baseline_id=baseline['baseline_id'],
                  protected_sha256=protected, failures=[], prepared_contract=[], rescues=[],
                  limitations=['Only historical positions 006/010/016; no new final holdout accessed.',
                               'Historical human reviews are references, not independently measured corner ground truth.',
                               'Roundtrip/crop parity measures internal consistency, not physical base accuracy.',
                               'Manual reference injection is diagnostic, not improved automatic performance.'])
    grids = []
    # Regenerate model crops from the untouched prepared canvas and compare to saved test inputs.
    for name in sorted(prior_prepared):
        p = prepared[name]
        if not review_approved(p, reviews.get(name)):
            raise ValueError(f'Stale/missing historical board review: {name}')
        for path_key, hash_key in [('context_path', 'context_sha256'), ('mask_path', 'mask_sha256')]:
            if file_hash(Path(p[path_key])) != p[hash_key]:
                raise ValueError(f'Changed prepared input: {name}')
        rect = load_image(p['context_path'])
        valid = cv2.imread(p['mask_path'], cv2.IMREAD_GRAYSCALE)
        polygons = np.float32(p['grid_geometry']['square_polygons'])
        raw_polygons = np.float32(p['grid_geometry']['source_square_polygons'])
        mapping, _ = cv2.findHomography(polygons.reshape(-1, 2), raw_polygons.reshape(-1, 2), 0)
        error = float(np.max(np.linalg.norm(project(project(polygons, mapping), np.linalg.inv(mapping))-polygons, axis=2)))
        samples = [r for r in manifest['results'] if r['source_image'] == name and r['split'] == 'test']
        issues = []
        for r in samples:
            crop, _, _, bounds, local, coverage = polygon_crop(rect, valid, polygons[r['row']*8+r['col']], 200)
            saved = load_image(r['crop_path'])
            expected = preprocess_context_crop(saved, r['target_polygon'])
            actual = preprocess_context_crop(crop, local)
            if (not np.array_equal(crop, saved) or not np.array_equal(actual, expected) or
                    not np.array_equal(actual[3], independently_rasterized_mask(crop.shape, local)) or
                    r['chess_square'] != chess_square_at(r['row'], r['col'], p['label_rotation']) or
                    coverage < .99):
                issues.append(r['crop_id'])
        ids = [chess_square_at(i//8, i%8, p['label_rotation']) for i in range(64)]
        image = load_image(config['data']['raw']/name)
        out = args.output_dir/'prepared'/Path(name).stem
        out.mkdir(parents=True)
        overlay = raw_grid_overlay(image, raw_polygons, ids, targets=('a1', 'd5', 'c3', 'g3'))
        path = out/'source_grid.jpg'
        save_image(path, title(overlay, [name+' reviewed prepared grid', 'green: canonical squares; orange: a1/d5/c3/g3']))
        grids.append(path)
        # Exact classifier-input RGB and target masks for named targets, without altering source crops.
        panels = []
        for square in ('a1', 'd5', 'c3', 'g3'):
            r = next(r for r in samples if r['chess_square'] == square)
            crop = load_image(r['crop_path'])
            tensor = preprocess_context_crop(crop, r['target_polygon'])
            rgb = cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
            mask = (tensor[3]*255).astype(np.uint8)
            cv2.polylines(rgb, [np.rint(np.asarray(r['target_polygon'])*[224/crop.shape[1], 224/crop.shape[0]]).astype(np.int32)], True, (0, 255, 0), 1)
            panels.append(title(np.concatenate([rgb, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)], axis=1), [square+' target / binary mask']))
        save_image(out/'target_masks.png', np.concatenate(panels, axis=0))
        report['prepared_contract'].append(dict(image_id=name, orientation=images[name]['orientation'],
                                                crops_checked=len(samples), mismatches=issues,
                                                roundtrip_max_context_px=error, overlay=str(path)))
        print('prepared contract', name, len(issues), flush=True)
    # Replay each unsuccessful raw photo unchanged; optional references never enter the automatic attempt.
    focus_paths = []
    for prior_record in failures:
        name = prior_record['image_id']
        image_path = config['data']['raw']/name
        image = load_image(image_path)
        out = args.output_dir/'failed_raw'/Path(name).stem
        out.mkdir(parents=True)
        automatic = out/'automatic'
        automatic.mkdir()
        observed = None
        try:
            prepare_raw_photo(image_path, config, automatic)
        except ValueError as exc:
            observed = getattr(exc, 'stage', 'preparation')
        if observed != prior_record['failure_stage']:
            raise ValueError(f'Failed raw stage did not reproduce: {name}: {observed}')
        detection = read(automatic/'detection.json')
        candidates = detection['diagnostics']['top_candidates']
        candidate = detection['corners'] if detection['success'] else (candidates[0]['corners'] if candidates else None)
        ref = overrides.get(name)
        if ref and file_hash(image_path) != ref['source_sha256']:
            raise ValueError('Stale manual corners')
        record = dict(image_id=name, orientation=images[name]['orientation'], failure_stage=observed,
                      detection=detection, reference_available=bool(ref), ablations=[])
        overlay = image.copy()
        if candidate is not None:
            points = np.rint(candidate).astype(np.int32)
            cv2.polylines(overlay, [points], True, (0, 140, 255), 3, cv2.LINE_AA)
            for label, point in zip(('TL', 'TR', 'BR', 'BL'), points):
                cv2.putText(overlay, label, tuple(point), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 140, 255), 2)
        if ref:
            cv2.polylines(overlay, [np.rint(ref['corners']).astype(np.int32)], True, (0, 255, 0), 3, cv2.LINE_AA)
            record['candidate_vs_reviewed'] = quadrilateral_metrics(candidate, ref['corners']) if candidate is not None else None
        path = out/'candidate_vs_reference.jpg'
        save_image(path, title(overlay, [name+' '+images[name]['orientation'],
                                       'A.2 failure reproduced: '+observed,
                                       'orange: '+('accepted detector' if detection['success'] else 'best REJECTED candidate'),
                                       'green: historical reviewed corners' if ref else 'NO reviewed corner reference available']))
        focus_paths.append(path)
        if observed == 'orientation':
            # Preserve the actual candidate evidence lost when prepare_raw_photo raises.
            from src.vision.pipeline import rectify_with_config
            result = resolve_orientation(rectify_with_config(image, np.float32(detection['corners']), config),
                                         OrientationSettings(**config['orientation']))
            record['automatic_orientation_evidence'] = result.metadata()
        if ref:
            if not review_approved(prepared[name], reviews.get(name)):
                raise ValueError('Manual rescue lacks current human-approved board reference')
            experiments = [('reviewed_corners_only', ref['corners'], None),
                           ('reviewed_corners_and_orientation', ref['corners'], SIDES[images[name]['orientation']])]
            if detection['success']:
                experiments.append(('automatic_corners_reference_orientation', detection['corners'], SIDES[images[name]['orientation']]))
            for mode, corners, side in experiments:
                destination = out/mode
                destination.mkdir()
                attempt = dict(image_id=name, mode=mode, status='failed', failure_stage=None)
                try:
                    rect, grid, turns, provenance = prepare_raw_photo(image_path, config, destination,
                                                                    corners=corners, white_side=side)
                    prediction = predict_rectified(model, rect, grid, turns, padding=200)
                    comparison = compare_position(BoardState(prediction['board']['rows'], 'white_at_bottom'),
                                                  board_from_fen(positions[images[name]['position_id']]['fen']))
                    attempt.update(status='predicted', **comparison, rotation_quarters_ccw=turns, **provenance,
                                   orientation_correct=turns == prepared[name]['label_rotation'],
                                   saved_prepared_piece_errors=prior_prepared[name]['piece_errors'])
                    save_image(destination/'prediction_overlay.jpg', prediction_overlay(rect.image, prediction['predictions']))
                    write(destination/'predictions.json', prediction)
                except (OSError, ValueError, RuntimeError) as exc:
                    attempt.update(failure_stage=getattr(exc, 'stage', 'classification'), reason=str(exc))
                write(destination/'result.json', attempt)
                record['ablations'].append(attempt)
                report['rescues'].append(attempt)
                print('rescue', name, mode, attempt['status'], attempt.get('piece_errors'), flush=True)
        report['failures'].append(record)
        write(out/'audit.json', record)
    montage(focus_paths, args.output_dir/'failure_gallery.jpg')
    for name, expected in protected.items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f'Protected input changed during audit: {name}')
    report['failure_counts'] = {orientation: dict(Counter(r['failure_stage'] for r in report['failures'] if r['orientation'] == orientation))
                                for orientation in SIDES}
    report['crop_count'] = sum(r['crops_checked'] for r in report['prepared_contract'])
    report['contract_mismatches'] = sum(len(r['mismatches']) for r in report['prepared_contract'])
    write(args.output_dir/'summary.json', report)
    print(json.dumps({'failure_counts': report['failure_counts'], 'checked_crops': report['crop_count'],
                      'contract_mismatches': report['contract_mismatches']}, indent=2))


if __name__ == '__main__':
    main()
