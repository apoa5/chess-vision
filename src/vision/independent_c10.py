"""C.10 evaluation-only contracts and metrics. Not used by production inference."""
from pathlib import Path
import chess
import numpy as np
from src.chess.labels import CLASS_NAMES
from src.training.multidataset import position_key, digest, placement_labels, SQUARES
from src.training.manifest_tools import labels_to_placement
from src.scanning.square_evidence import classification_metrics

CHECKPOINTS = {
    'b2': {'path': 'outputs/evaluation/crop_b2_20261018_run1/training/context1_224/best.pt',
           'sha256': 'dad8eea28482ee894dfca0794d9f4415d21ac58db23c6d6bed8c4307c71b1bd8'},
    'convnext': {'path': 'outputs/evaluation/convnext_c9/export_v1/convnextv2_tiny_best.pt',
                 'sha256': '45b471edb751261e9da1ef8703f3d9c940cdaec69f5713f6847e48bf0dc26acb'},
}
FENS = [
    'rn1qkbnr/pbpp1ppp/1p6/4p3/8/1P1P1N2/PBP1PPPP/RN1QKB1R w KQkq - 0 1',
    'r4rk1/pbpn1ppp/1p1p1q2/2b1p3/4P3/2NP1NP1/PPP2PBP/R2QR1K1 w - - 0 1',
    '5r2/pb2k1p1/1p5p/8/7P/P1N3P1/1P3P1K/3R4 w - - 0 1',
]


def validate_fen(fen):
    if not isinstance(fen, str) or len(fen.split()) != 6:
        raise ValueError('Explicit full owner-supplied FEN required')
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError('Invalid reference position')
    return board.board_fen()


def independence(fen, source_sha, historical):
    placement = validate_fen(fen)
    related = [b for b in historical if position_key(b['placement']) == position_key(placement)]
    duplicate = [b['id'] for b in historical if source_sha in b.get('source_hashes', [])]
    train = [b['id'] for b in related if b['split'] == 'train']
    return dict(canonical_placement=placement, placement_key=position_key(placement),
                historical_placement_matches=[b['id'] for b in related], c9_train_placement_matches=train,
                duplicate_source_matches=duplicate, independent_eligible=not related and not duplicate)


def validate_manifest(manifest):
    if manifest['training_eligible'] or manifest['final_holdout_access']:
        raise ValueError('Evaluation-only cohort required')
    if manifest['class_names'] != list(CLASS_NAMES):
        raise ValueError('Class order changed')
    if len(manifest['photos']) != 3 or len({p['id'] for p in manifest['photos']}) != 3:
        raise ValueError('Exactly three distinct photo identities required')
    if len({position_key(p['fen']) for p in manifest['photos']}) != 3:
        raise ValueError('Three distinct physical placements required')
    for photo in manifest['photos']:
        validate_fen(photo['fen'])
        if photo['ground_truth']['source'] != 'owner_supplied_reference_FEN':
            raise ValueError('Human reference provenance required')
        if digest(photo['path']) != photo['sha256']:
            raise ValueError('Original photo changed')


def verify_checkpoint(path, expected, state=None, model_name=None):
    if digest(path) != expected:
        raise ValueError('Frozen checkpoint hash mismatch')
    if state is not None:
        if state['class_names'] != list(CLASS_NAMES) or state['input_channels'] != 4 or state['input_size'] != 224:
            raise ValueError('Checkpoint input/class contract changed')
        if model_name == 'b2' and (state['architecture'] != 'mobilenet_v3_small' or state['crop_spec']['name'] != 'context1_224'):
            raise ValueError('Not the B.2 reference representation')
        if model_name == 'convnext' and (state['updates'] != 1578 or state['representation'] != 'b2-context1-224-v1'):
            raise ValueError('Not the selected C.9 checkpoint')


def canonical_inputs(inputs):
    if len(inputs) != 64 or {i['canonical_square'] for i in inputs} != set(SQUARES):
        raise ValueError('Exactly one input for each canonical square required')
    lookup = {i['canonical_square']: i for i in inputs}
    return [lookup[s] for s in SQUARES]


def score(truth, logits):
    y = np.asarray(truth, dtype=int); z = np.asarray(logits, dtype=np.float64)
    if y.shape != (64,) or z.shape != (64, 13) or not np.isfinite(z).all():
        raise ValueError('64 aligned labels and finite 13-class logits required')
    probs = np.exp(z-z.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
    prediction = probs.argmax(1); confidence = probs.max(1); wrong = prediction != y
    m = classification_metrics(y, prediction)
    for row in m['per_class']:
        a, b = row['precision'], row['recall']
        row['f1'] = 2*a*b/(a+b) if a is not None and b is not None and a+b else (0. if row['support'] or row['predicted_count'] else None)
    occupied_pair = (y != 0) & (prediction != 0)
    m.update(empty_accuracy=1-m['empty_false_positive_rate'] if m['empty_count'] else None,
             exact_board=bool(not wrong.any()), corrections=int(wrong.sum()),
             occupied_errors=int((wrong & (y != 0)).sum()),
             wrong_colors=int((occupied_pair & ((y <= 6) != (prediction <= 6))).sum()),
             wrong_piece_types=int((occupied_pair & ((y-1) % 6 != (prediction-1) % 6)).sum()),
             color_and_type_counts_may_overlap=True,
             confidence_mean_correct=float(confidence[~wrong].mean()) if (~wrong).any() else None,
             confidence_mean_incorrect=float(confidence[wrong].mean()) if wrong.any() else None,
             confident_errors_above_0_9=int((wrong & (confidence > .9)).sum()),
             confident_empty_false_positives=int(((y == 0) & wrong & (confidence > .9)).sum()),
             uncalibrated_confidence=True)
    observations = [dict(square=s, truth=int(t), prediction=int(p), raw_logits=l.tolist(),
                         probabilities=v.tolist(), confidence=float(c), correct=bool(t == p))
                    for s, t, p, l, v, c in zip(SQUARES, y, prediction, z, probs, confidence)]
    return dict(metrics=m, observations=observations, predicted_placement=labels_to_placement(prediction.tolist()))


def cohort_metrics(rows, attempted):
    """Failed automatic photos stay in the board denominator; no fake square labels."""
    completed = [r for r in rows if r['completed'] and r['independent_eligible']]
    all_obs = [o for r in completed for o in r['observations']]
    result = dict(attempted_photos=attempted, eligible_completed_photos=len(completed),
                  exact_boards=sum(r['metrics']['exact_board'] for r in completed),
                  exact_board_denominator=attempted, scored_square_count=len(all_obs),
                  independent_exact_board_denominator=sum(r['independent_eligible'] for r in rows),
                  failed_or_excluded_photos=attempted-len(completed))
    if not all_obs:
        return result | dict(square_metrics=None, mean_corrections_completed=None)
    y = [o['truth'] for o in all_obs]; p = [o['prediction'] for o in all_obs]
    metrics = classification_metrics(y, p)
    for row in metrics['per_class']:
        a,b=row['precision'],row['recall']
        row['f1']=2*a*b/(a+b) if a is not None and b is not None and a+b else (0. if row['support'] or row['predicted_count'] else None)
    result.update(square_metrics=metrics, mean_corrections_completed=float(np.mean([r['metrics']['corrections'] for r in completed])),
                  mean_confidence_correct=float(np.mean([o['confidence'] for o in all_obs if o['correct']])) if any(o['correct'] for o in all_obs) else None,
                  mean_confidence_incorrect=float(np.mean([o['confidence'] for o in all_obs if not o['correct']])) if any(not o['correct'] for o in all_obs) else None,
                  confident_errors_above_0_9=sum(not o['correct'] and o['confidence']>.9 for o in all_obs),
                  confident_empty_false_positives=sum(o['truth']==0 and not o['correct'] and o['confidence']>.9 for o in all_obs))
    return result
