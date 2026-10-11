"""Full-position comparison and failure-inclusive aggregates."""
import numpy as np
from src.chess.board_state import BoardState
from src.chess.fen import placement_fen


def compare_position(predicted: BoardState, expected: BoardState):
    predicted_fen = placement_fen(predicted); expected_fen = placement_fen(expected)
    wrong = np.asarray(predicted.rows,dtype=object) != np.asarray(expected.rows,dtype=object)
    errors = int(wrong.sum())
    squares = [f'{chr(97+c)}{8-r}' for r,c in zip(*np.where(wrong))]
    return {'square_accuracy':(64-errors)/64,'piece_errors':errors,
            'exact_match':predicted_fen == expected_fen,'incorrect_squares':squares,
            'ground_truth_placement':expected_fen,'predicted_placement':predicted_fen}


def summarize_positions(rows):
    successes = [r for r in rows if r['status'] == 'predicted']
    count = len(rows); exact = sum(r['exact_match'] for r in successes)
    return {'attempted_boards':count,'successful_boards':len(successes),'failed_boards':count-len(successes),
            'exact_boards':exact,'exact_board_accuracy_all_attempts':exact/count if count else None,
            'exact_board_accuracy_successful_only':exact/len(successes) if successes else None,
            'square_accuracy_successful_only':sum(64-r['piece_errors'] for r in successes)/(64*len(successes)) if successes else None,
            'mean_confidence_successful_only':float(np.mean([r['avg_confidence'] for r in successes])) if successes else None,
            'mean_inference_time_ms_successful_only':float(np.mean([r['inference_time_ms'] for r in successes])) if successes else None,
            'failure_counts':{stage:sum(r.get('failure_stage')==stage for r in rows) for stage in sorted({r['failure_stage'] for r in rows if r['status']=='failed'})},
            'standard_chess_valid_boards':sum(r['validation_status']=='valid' for r in successes)}
