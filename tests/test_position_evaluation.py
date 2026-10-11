import pytest
from src.chess.board_state import BoardState
from src.vision.position_evaluation import compare_position,summarize_positions


def test_one_wrong_square_prevents_exact_board_match():
    rows=[[None]*8 for _ in range(8)];rows[7][0]='K';rows[0][7]='k'
    truth=BoardState(rows,'white_at_bottom')
    predicted=[r.copy() for r in rows];predicted[0][7]='q'
    result=compare_position(BoardState(predicted,'white_at_bottom'),truth)
    assert result['square_accuracy']==63/64
    assert result['piece_errors']==1 and not result['exact_match']
    assert result['incorrect_squares']==['h8']


def test_preprocessing_failure_remains_in_exact_board_denominator():
    rows=[{'status':'predicted','exact_match':True,'piece_errors':0,'avg_confidence':.9,
           'inference_time_ms':100.,'validation_status':'valid'},
          {'status':'failed','failure_stage':'orientation'}]
    result=summarize_positions(rows)
    assert result['exact_board_accuracy_all_attempts']==.5
    assert result['exact_board_accuracy_successful_only']==1
    assert result['square_accuracy_successful_only']==1
    assert result['failure_counts']=={'orientation':1}


def test_no_successes_does_not_report_perfect_accuracy():
    result=summarize_positions([{'status':'failed','failure_stage':'localization'}])
    assert result['exact_board_accuracy_all_attempts']==0
    assert result['square_accuracy_successful_only'] is None
