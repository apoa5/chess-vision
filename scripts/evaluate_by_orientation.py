"""Stratify a saved prepared/raw evaluation; never rerun or modify recognition."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from scripts.evaluate_classifier import contact_sheet, confusion_plot
from src.chess.labels import CLASS_NAMES
from src.utils.config import load_config
from src.vision.dataset import file_hash
from src.vision.orientation_evaluation import (
    ORIENTATIONS, annotate_attempt, expected_attempts, stratify, validate_attempts)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def export_errors(rows, destination, prepared, prediction_paths):
    """Export exact saved context windows, padded with black at source boundaries."""
    errors = []
    for row in rows:
        wrong = [p for p in row['square_predictions'] if not p['correct']]
        if not wrong:
            continue
        context = (Path(prepared[row['image_id']]['context_path']) if row['mode'] == 'prepared'
                   else prediction_paths[(row['mode'], row['image_id'])].parent/'context.png')
        image = cv2.imread(str(context))
        if image is None:
            raise ValueError(f'Gallery context image unavailable: {context}')
        for p in wrong:
            x0, y0, x1, y1 = p['context_bounds']
            crop = np.zeros((y1-y0, x1-x0, 3), np.uint8)
            sx0, sy0 = max(0, x0), max(0, y0)
            sx1, sy1 = min(image.shape[1], x1), min(image.shape[0], y1)
            crop[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = image[sy0:sy1, sx0:sx1]
            crop_id = f"{Path(row['image_id']).stem}_{p['chess_square']}"
            path = destination/row['mode']/row['orientation']/'crops'/f'{crop_id}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), crop):
                raise OSError(f'Could not write {path}')
            errors.append({'mode': row['mode'], 'orientation': row['orientation'],
                           'image_id': row['image_id'], 'position_id': row['position_id'],
                           'chess_square': p['chess_square'], 'crop_id': crop_id,
                           'crop_path': str(path), 'target_polygon': p['target_polygon'],
                           'class_name': p['true_class'], 'predicted_class': p['class_name'],
                           'confidence': p['confidence'], 'source_context': str(context),
                           'orientation_correct': row['orientation_correct']})
    for mode in ('prepared', 'raw'):
        for orientation in ORIENTATIONS:
            selected = sorted([e for e in errors if e['mode'] == mode and e['orientation'] == orientation],
                              key=lambda e: -e['confidence'])
            directory = destination/mode/orientation
            directory.mkdir(parents=True, exist_ok=True)
            write(directory/'errors.json', selected)
            for offset in range(0, len(selected), 20):
                contact_sheet(selected[offset:offset+20], directory/f'errors_{offset//20+1:02d}.png')
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positions-report', type=Path, required=True, help='Existing evaluate_positions.py summary.json')
    parser.add_argument('--baseline-manifest', type=Path, default=Path('configs/baselines/a1_20261018.json'))
    parser.add_argument('--output-dir', type=Path, required=True, help='Must not already exist')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Output directory already exists; choose a new versioned path.')
    cv2.setNumThreads(1)
    config = load_config()
    baseline = read(args.baseline_manifest)
    # Includes the original reports, checkpoint, source and split manifests.
    for name, expected in baseline['preserved_sha256'].items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f'Frozen baseline artifact changed: {name}')
    manifest_path = config['classifier']['dataset_paths']['metadata']
    images_path = config['data']['metadata']/'images.json'
    positions_path = config['data']['metadata']/'positions.json'
    manifest, images, positions = read(manifest_path), read(images_path), read(positions_path)
    source = read(args.positions_report)
    for key, expected in {
        'checkpoint_sha256': baseline['checkpoint']['sha256'],
        'manifest_sha256': file_hash(manifest_path),
        'dataset_fingerprint': baseline['dataset']['fingerprint'],
        'ground_truth_positions_sha256': file_hash(positions_path),
        'images_metadata_sha256': file_hash(images_path),
    }.items():
        if source[key] != expected:
            raise ValueError(f'Saved evaluation provenance mismatch: {key}')
    validate_attempts(source['records'], expected_attempts(manifest, images))
    prepared_path = manifest_path.parent/'prepared_boards.json'
    prepared = {r['image_id']: r for r in read(prepared_path)}
    hashes = {str(args.positions_report): file_hash(args.positions_report)}
    rows, paths = [], {}
    for record in source['records']:
        payload = None
        if record['status'] == 'predicted':
            path = Path(record['prediction_path'])
            paths[(record['mode'], record['image_id'])] = path
            hashes[str(path)] = file_hash(path)
            payload = read(path)
        rows.append(annotate_attempt(record, payload, images[record['image_id']], positions[record['position_id']]))
    report = {'schema_version': 'orientation_evaluation_v1', 'baseline_id': baseline['baseline_id'],
              'source_commit': baseline['git_commit'], 'source_sha256': hashes,
              'checkpoint_sha256': source['checkpoint_sha256'], 'manifest_sha256': source['manifest_sha256'],
              'class_names': list(CLASS_NAMES), 'evaluation': stratify(rows),
              'limitations': [
                  'Post-processing saved A.1 predictions; no inference, training or tuning performed.',
                  'Raw workflow is the historical photo pipeline, not the browser learned-marking bootstrap.',
                  'Square metrics are conditional on completed outputs; exact-board rates count every attempt.',
                  'Gate acceptance is not independently measured geometric accuracy.',
                  'Orientation/canonical mapping checks use image capture metadata, not an independent geometry audit.',
                  'Missing sessions are unknown; variation IDs are not independent capture session IDs.',
                  'Angles are approximate capture-plan elevations above the board plane, not measured camera poses.',
                  'Orientation is confounded with environment/lighting and small correlated position samples.',
                  'No calibration fitting or confidence threshold selection; no phone runtime measured.']}
    for mode, metrics in source['modes'].items():
        for key, value in metrics.items():
            if report['evaluation'][mode]['overall'][key] != value:
                raise ValueError(f'Saved aggregate disagrees with attempt records: {mode}/{key}')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    errors = export_errors(rows, args.output_dir, prepared, paths)
    report['error_examples'] = len(errors)
    csv_rows, class_rows = [], []
    for mode, data in report['evaluation'].items():
        for dimension, groups in data['groups'].items():
            for name, group in groups.items():
                recognition = group['recognition_on_completed_outputs'] or {}
                csv_rows.append({'mode': mode, 'dimension': dimension, 'group': name,
                                 **{k: v for k, v in group.items()
                                    if not isinstance(v, dict) and k != 'recognition_on_completed_outputs'},
                                 'occupied_square_accuracy': recognition.get('occupied_square_accuracy')})
                if dimension == 'orientation':
                    directory = args.output_dir/mode/name
                    matrix = recognition.get('confusion_matrix', [[0]*13 for _ in range(13)])
                    confusion_plot(matrix, directory/'confusion_matrix.png')
                    with (directory/'confusion_matrix.csv').open('w', newline='') as f:
                        writer = csv.writer(f); writer.writerow(['true / predicted', *CLASS_NAMES])
                        writer.writerows([label, *counts] for label, counts in zip(CLASS_NAMES, matrix))
                    per_class = recognition.get('per_class') or [
                        dict(class_name=label, support=0, precision=None, recall=None, f1=None)
                        for label in CLASS_NAMES]
                    class_rows.extend({'mode': mode, 'orientation': name, **r} for r in per_class)
    for filename, values in [('groups.csv', csv_rows), ('per_class.csv', class_rows)]:
        with (args.output_dir/filename).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    write(args.output_dir/'attempts.json', [{k: v for k, v in r.items() if k != 'square_predictions'} for r in rows])
    for name, expected in {**baseline['preserved_sha256'], **hashes}.items():
        if file_hash(Path(name)) != expected:
            raise ValueError(f'Input changed during evaluation: {name}')
    write(args.output_dir/'summary.json', report)
    print(json.dumps({mode: {name: {'attempts': g['attempted_boards'], 'completed': g['successful_boards'],
                                   'square_accuracy': g['square_accuracy_successful_only'],
                                   'occupied_accuracy': (g['recognition_on_completed_outputs'] or {}).get('occupied_square_accuracy'),
                                   'exact_boards': g['exact_boards']}
                            for name, g in data['groups']['orientation'].items()}
                      for mode, data in report['evaluation'].items()}, indent=2))


if __name__ == '__main__':
    main()
