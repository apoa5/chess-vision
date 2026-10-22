"""Stratify saved position predictions without changing inference or its inputs."""
from collections import Counter

import numpy as np

from src.chess.fen import board_from_fen
from src.chess.labels import CLASS_NAMES, CLASS_TO_SYMBOL, SYMBOL_TO_ID
from src.vision.classifier_evaluation import prediction_metrics
from src.vision.dataset import chess_square_at
from src.vision.position_evaluation import summarize_positions

ORIENTATIONS = ('white_near_camera', 'black_near_camera', 'white_left', 'white_right')
TURNS = dict(zip(ORIENTATIONS, (0, 2, 1, 3)))
DIMENSIONS = ('orientation', 'position_id', 'environment', 'lighting',
              'camera_angle_degrees', 'session_id')


def expected_attempts(manifest, images):
    """Raw includes excluded photos; prepared includes only approved test crops."""
    assignments = manifest['split_report']['position_assignment']
    raw = {name for name, meta in images.items() if assignments.get(meta['position_id']) == 'test'}
    prepared = {r['source_image'] for r in manifest['results'] if r['split'] == 'test'}
    if not prepared <= raw:
        raise ValueError('Prepared test images are missing from raw test membership.')
    return {'raw': raw, 'prepared': prepared}


def validate_attempts(records, expected):
    actual = [(r['mode'], r['image_id']) for r in records]
    wanted = {(mode, name) for mode, names in expected.items() for name in names}
    if len(actual) != len(set(actual)) or set(actual) != wanted:
        raise ValueError('Evaluation has duplicate, missing or extra attempts.')
    for row in records:
        if row['status'] not in ('predicted', 'failed'):
            raise ValueError('Unknown evaluation status.')
        if row['status'] == 'failed' and not row.get('failure_stage'):
            raise ValueError('Failed attempt has no failure stage.')


def annotate_attempt(record, payload, metadata, position):
    """Join capture metadata only after inference; never use it as model input."""
    orientation = metadata.get('orientation')
    if orientation not in ORIENTATIONS:
        raise ValueError('Missing or unsupported physical camera orientation.')
    if record['position_id'] != metadata['position_id']:
        raise ValueError('Position ID disagrees with image metadata.')
    row = {**record, **{key: metadata.get(key) if metadata.get(key) is not None else 'unknown'
                       for key in DIMENSIONS}, 'square_predictions': []}
    row['orientation_correct'] = None
    row['coordinate_assignment_correct'] = None
    if record['status'] != 'predicted':
        if payload is not None:
            raise ValueError('Failed attempt must not supply a successful prediction.')
        return row
    if payload['source_image'] != record['image_id'] or payload['mode'] != record['mode']:
        raise ValueError('Saved prediction identity mismatch.')
    truth = board_from_fen(position['fen'])
    if record['ground_truth_placement'] != position['fen'].split()[0]:
        raise ValueError('Ground-truth placement changed.')
    turns = payload['rotation_quarters_ccw']
    if type(turns) is not int or turns not in range(4):
        raise ValueError('Invalid saved orientation.')
    predictions = payload['predictions']
    if (len(predictions) != 64 or
            {(p['row'], p['col']) for p in predictions} != {(r, c) for r in range(8) for c in range(8)}):
        raise ValueError('Exactly 64 unique camera squares are required.')
    coordinates = [p['chess_square'] for p in predictions]
    if set(coordinates) != {f'{f}{r}' for f in 'abcdefgh' for r in range(1, 9)}:
        raise ValueError('Exactly 64 unique canonical squares are required.')
    # Check both internal mapping consistency and agreement with capture-side reference.
    if any(p['chess_square'] != chess_square_at(p['row'], p['col'], turns) for p in predictions):
        raise ValueError('Saved square identity disagrees with saved orientation.')
    row['orientation_correct'] = turns == TURNS[orientation]
    row['coordinate_assignment_correct'] = all(
        p['chess_square'] == chess_square_at(p['row'], p['col'], TURNS[orientation]) for p in predictions)
    for p in predictions:
        square = p['chess_square']
        symbol = truth.rows[8-int(square[1])][ord(square[0])-97]
        label = SYMBOL_TO_ID[symbol]
        probabilities = p['probabilities']
        predicted = int(np.argmax(probabilities))
        if (p['class_id'] != predicted or p['class_name'] != CLASS_NAMES[predicted] or
                p['symbol'] != CLASS_TO_SYMBOL[CLASS_NAMES[predicted]]):
            raise ValueError('Saved class/probability contract mismatch.')
        row['square_predictions'].append({**p, 'true_class_id': label,
                                          'true_class': CLASS_NAMES[label],
                                          'correct': predicted == label})
    # Validate probabilities and reconcile saved board errors with square-level data.
    metrics = prediction_metrics([p['true_class_id'] for p in row['square_predictions']],
                                 [p['probabilities'] for p in row['square_predictions']])
    if metrics['error_count'] != record['piece_errors'] or record['exact_match'] != (metrics['error_count'] == 0):
        raise ValueError('Saved board comparison disagrees with square predictions.')
    return row


def summarize_group(rows):
    result = summarize_positions(rows)
    successful = [r for r in rows if r['status'] == 'predicted']
    squares = [p for r in successful for p in r['square_predictions']]
    result['recognition_on_completed_outputs'] = prediction_metrics(
        [p['true_class_id'] for p in squares], [p['probabilities'] for p in squares]) if squares else None
    result['evaluated_squares'] = len(squares)
    result['completed_position_rate_all_attempts'] = len(successful)/len(rows) if rows else None
    # These are operational pipeline gates, not independently verified localization accuracy.
    result['pipeline_gates'] = {
        'localization_accepted': sum(r['status'] == 'predicted' or r.get('failure_stage') in
                                     ('orientation', 'grid', 'classification') for r in rows),
        'orientation_resolved': sum(r['status'] == 'predicted' or r.get('failure_stage') in
                                   ('grid', 'classification') for r in rows),
        'grid_accepted': sum(r['status'] == 'predicted' or r.get('failure_stage') == 'classification' for r in rows),
        'orientation_checked_against_metadata': len(successful),
        'orientation_correct_against_metadata': sum(r['orientation_correct'] for r in successful),
        'coordinate_assignment_correct_against_metadata': sum(r['coordinate_assignment_correct'] for r in successful),
    }
    for gate in ('localization_accepted', 'orientation_resolved', 'grid_accepted'):
        result['pipeline_gates'][gate+'_rate_all_attempts'] = (
            result['pipeline_gates'][gate]/len(rows) if rows else None)
    result['pipeline_gates']['orientation_agreement_rate_completed_outputs'] = (
        result['pipeline_gates']['orientation_correct_against_metadata']/len(successful) if successful else None)
    result['diagnostics'] = {
        'classification_mismatch_boards': sum(r['piece_errors'] > 0 for r in successful),
        'position_validation_flagged_boards': sum(r['validation_status'] != 'valid' for r in successful),
        'orientation_mismatch_boards': sum(not r['orientation_correct'] for r in successful),
    }
    return result


def stratify(rows):
    """All four orientations are represented even when no outputs complete."""
    report = {}
    for mode in ('prepared', 'raw'):
        selected = [r for r in rows if r['mode'] == mode]
        groups = {}
        for dimension in DIMENSIONS:
            names = ORIENTATIONS if dimension == 'orientation' else sorted({str(r[dimension]) for r in selected})
            groups[dimension] = {
                name: summarize_group([r for r in selected if str(r[dimension]) == name]) for name in names}
            assert sum(g['attempted_boards'] for g in groups[dimension].values()) == len(selected)
            assert sum(g['evaluated_squares'] for g in groups[dimension].values()) == 64*sum(
                r['status'] == 'predicted' for r in selected)
        report[mode] = {'overall': summarize_group(selected), 'groups': groups,
                        'orientation_environment_counts': {
                            orientation: dict(Counter(r['environment'] for r in selected if r['orientation'] == orientation))
                            for orientation in ORIENTATIONS}}
    return report
