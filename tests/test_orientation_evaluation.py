import copy
import json
from pathlib import Path

import pytest

from src.chess.labels import CLASS_NAMES, CLASS_TO_SYMBOL
from src.vision.dataset import chess_square_at
from src.vision.orientation_evaluation import (
    ORIENTATIONS, TURNS, annotate_attempt, expected_attempts, stratify, validate_attempts)


def attempt(orientation='white_near_camera', mode='raw', failure=None, wrong=False):
    metadata = dict(position_id='p1', orientation=orientation, environment='office',
                    lighting='overhead', camera_angle_degrees=75)
    record = dict(mode=mode, image_id='one.jpg', position_id='p1', status='failed' if failure else 'predicted',
                  ground_truth_placement='8/8/8/8/8/8/8/K6k', exact_match=not (failure or wrong),
                  failure_stage=failure, piece_errors=None if failure else int(wrong),
                  validation_status='not_evaluated' if failure else 'valid',
                  avg_confidence=1., inference_time_ms=100.)
    payload = dict(source_image='one.jpg', mode=mode, rotation_quarters_ccw=TURNS[orientation], predictions=[])
    for r in range(8):
        for c in range(8):
            square = chess_square_at(r, c, TURNS[orientation])
            label = 6 if square == 'a1' else 12 if square == 'h1' else 0
            if wrong and square == 'a1':
                label = 5
            probabilities = [float(i == label) for i in range(13)]
            payload['predictions'].append(dict(row=r, col=c, chess_square=square, class_id=label,
                                               class_name=CLASS_NAMES[label], symbol=CLASS_TO_SYMBOL[CLASS_NAMES[label]],
                                               probabilities=probabilities, confidence=1.))
    position = dict(fen=record['ground_truth_placement']+' w - - 0 1')
    return record, None if failure else payload, metadata, position


@pytest.mark.parametrize('orientation', ORIENTATIONS)
def test_four_camera_sides_preserve_canonical_labels(orientation):
    row = annotate_attempt(*attempt(orientation))
    assert row['orientation_correct'] and row['coordinate_assignment_correct']
    assert len(row['square_predictions']) == 64
    assert all(p['correct'] for p in row['square_predictions'])
    assert row['session_id'] == 'unknown'


def test_localization_failure_stays_in_every_group_denominator():
    rows = [annotate_attempt(*attempt()), annotate_attempt(*attempt(failure='localization'))]
    result = stratify(rows)['raw']
    for dimension, groups in result['groups'].items():
        assert sum(g['attempted_boards'] for g in groups.values()) == 2
        assert sum(g['evaluated_squares'] for g in groups.values()) == 64
    group = result['groups']['orientation']['white_near_camera']
    assert group['exact_board_accuracy_all_attempts'] == .5
    assert group['completed_position_rate_all_attempts'] == .5
    assert group['recognition_on_completed_outputs']['accuracy'] == 1
    assert group['failure_counts'] == {'localization': 1}
    assert group['pipeline_gates']['localization_accepted'] == 1


def test_empty_and_failed_only_groups_have_no_invented_recognition_accuracy():
    data = stratify([annotate_attempt(*attempt(failure='orientation'))])['raw']
    group = data['groups']['orientation']['white_near_camera']
    assert group['exact_board_accuracy_all_attempts'] == 0
    assert group['recognition_on_completed_outputs'] is None
    assert group['pipeline_gates']['localization_accepted'] == 1
    assert group['pipeline_gates']['orientation_resolved'] == 0
    empty = data['groups']['orientation']['white_left']
    assert empty['attempted_boards'] == 0
    assert empty['exact_board_accuracy_all_attempts'] is None


def test_confusion_and_validation_diagnostics_are_separate():
    args = attempt(wrong=True)
    args[0]['validation_status'] = 'diagnostics'
    group = stratify([annotate_attempt(*args)])['raw']['overall']
    metrics = group['recognition_on_completed_outputs']
    assert metrics['confusion_matrix'][6][5] == 1
    assert metrics['occupied_square_accuracy'] == .5
    assert metrics['accuracy'] == 63/64
    assert group['diagnostics']['classification_mismatch_boards'] == 1
    assert group['diagnostics']['position_validation_flagged_boards'] == 1


@pytest.mark.parametrize('problem', ['duplicate', 'missing', 'extra'])
def test_invalid_attempt_inventory_rejected(problem):
    expected = {'prepared': {'a.jpg'}, 'raw': {'a.jpg', 'b.jpg'}}
    records = [dict(mode=mode, image_id=name, status='predicted') for mode, names in expected.items() for name in names]
    if problem == 'duplicate':
        records.append(records[0])
    elif problem == 'missing':
        records.pop()
    else:
        records.append(dict(mode='raw', image_id='extra.jpg'))
    with pytest.raises(ValueError, match='attempts'):
        validate_attempts(records, expected)


@pytest.mark.parametrize('problem', ['orientation', 'duplicate_square', 'mapping', 'errors', 'class'])
def test_invalid_metadata_or_payload_rejected(problem):
    args = attempt()
    if problem == 'orientation':
        args[2]['orientation'] = None
    elif problem == 'duplicate_square':
        args[1]['predictions'][1] = copy.deepcopy(args[1]['predictions'][0])
    elif problem == 'mapping':
        args[1]['predictions'][0]['chess_square'], args[1]['predictions'][1]['chess_square'] = (
            args[1]['predictions'][1]['chess_square'], args[1]['predictions'][0]['chess_square'])
    elif problem == 'errors':
        args[0]['piece_errors'] = 1
    else:
        args[1]['predictions'][0]['class_name'] = 'black_king'
    with pytest.raises(ValueError):
        annotate_attempt(*args)


def test_orientation_disagreement_is_reported_separately():
    args = attempt()
    args[2]['orientation'] = 'black_near_camera'
    row = annotate_attempt(*args)
    assert not row['orientation_correct']
    assert not row['coordinate_assignment_correct']


def test_actual_frozen_counts_and_all_partition_totals():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root/'data/classification/crops.json'
    report_path = root/'outputs/evaluation/baseline_a1_20261018/positions_run1/summary.json'
    if not manifest_path.exists() or not report_path.exists():
        pytest.skip('Local A.1 dataset/evaluation assets unavailable')
    manifest = json.loads(manifest_path.read_text())
    images = json.loads((root/'data/metadata/images.json').read_text())
    records = json.loads(report_path.read_text())['records']
    expected = expected_attempts(manifest, images)
    assert {k: len(v) for k, v in expected.items()} == {'raw': 30, 'prepared': 23}
    validate_attempts(records, expected)
    for mode, counts in [('prepared', [12, 6, 2, 3]), ('raw', [18, 6, 3, 3])]:
        assert [sum(r['mode'] == mode and images[r['image_id']]['orientation'] == orientation for r in records)
                for orientation in ORIENTATIONS] == counts


def test_cli_exports_failed_only_strata_and_refuses_output_reuse(tmp_path, monkeypatch):
    import csv
    from scripts import evaluate_by_orientation as cli
    from src.vision.position_evaluation import summarize_positions
    from src.vision.dataset import file_hash

    prepared_record, payload, metadata, position = attempt(mode='prepared')
    raw_record = attempt(failure='localization')[0]
    prediction = tmp_path/'prediction.json'
    prediction.write_text(json.dumps(payload))
    prepared_record['prediction_path'] = str(prediction)
    manifest = tmp_path/'crops.json'
    manifest.write_text(json.dumps({'split_report': {'position_assignment': {'p1': 'test'}},
                                     'results': [{'split': 'test', 'source_image': 'one.jpg'}]}))
    (tmp_path/'images.json').write_text(json.dumps({'one.jpg': metadata}))
    (tmp_path/'positions.json').write_text(json.dumps({'p1': position}))
    (tmp_path/'prepared_boards.json').write_text('[]')
    baseline = tmp_path/'baseline.json'
    baseline.write_text(json.dumps({'preserved_sha256': {}, 'checkpoint': {'sha256': 'model'},
                                    'dataset': {'fingerprint': 'dataset'}, 'baseline_id': 'test', 'git_commit': 'commit'}))
    report = tmp_path/'source.json'
    records = [prepared_record, raw_record]
    source = dict(checkpoint_sha256='model', manifest_sha256=file_hash(manifest),
                  dataset_fingerprint='dataset', images_metadata_sha256=file_hash(tmp_path/'images.json'),
                  ground_truth_positions_sha256=file_hash(tmp_path/'positions.json'), records=records,
                  modes={mode: summarize_positions([r for r in records if r['mode'] == mode])
                         for mode in ('prepared', 'raw')})
    report.write_text(json.dumps(source))
    monkeypatch.setattr(cli, 'load_config', lambda: {'classifier': {'dataset_paths': {'metadata': manifest}},
                                                   'data': {'metadata': tmp_path}})
    monkeypatch.setattr(cli, 'confusion_plot', lambda *args: None)
    output = tmp_path/'result'
    monkeypatch.setattr('sys.argv', ['evaluate', '--positions-report', str(report),
                                    '--baseline-manifest', str(baseline), '--output-dir', str(output)])
    cli.main()
    summary = json.loads((output/'summary.json').read_text())
    raw = summary['evaluation']['raw']['groups']['orientation']['white_near_camera']
    assert raw['attempted_boards'] == 1 and raw['exact_board_accuracy_all_attempts'] == 0
    assert raw['recognition_on_completed_outputs'] is None
    with (output/'per_class.csv').open() as f:
        classes = list(csv.DictReader(f))
    assert len(classes) == 2*4*13
    assert all(r['support'] == '0' and r['recall'] == '' for r in classes if r['mode'] == 'raw')
    original = (output/'summary.json').read_bytes()
    with pytest.raises(SystemExit):
        cli.main()
    assert (output/'summary.json').read_bytes() == original


def test_gallery_preserves_context_padding_and_true_predicted_labels(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from scripts import evaluate_by_orientation as cli

    context = tmp_path/'context.png'
    cv2.imwrite(str(context), np.full((8, 8, 3), 128, np.uint8))
    row = dict(mode='prepared', image_id='one.jpg', position_id='p1', orientation='white_left',
               orientation_correct=True, square_predictions=[dict(
                   correct=False, context_bounds=[-2, -2, 6, 6], chess_square='a1',
                   target_polygon=[[2, 2], [3, 2], [3, 3], [2, 3]],
                   true_class='white_king', class_name='white_queen', confidence=.95)])
    sheets = []
    monkeypatch.setattr(cli, 'contact_sheet', lambda rows, path: sheets.extend(rows))
    errors = cli.export_errors([row], tmp_path/'gallery', {'one.jpg': {'context_path': str(context)}}, {})
    assert len(errors) == len(sheets) == 1
    assert sheets[0]['class_name'] == 'white_king'
    assert sheets[0]['predicted_class'] == 'white_queen'
    crop = cv2.imread(errors[0]['crop_path'])
    assert crop.shape == (8, 8, 3)
    assert not crop[:2].any() and not crop[:, :2].any()
    assert (crop[2:, 2:] == 128).all()
