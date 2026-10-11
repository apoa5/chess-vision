import chess
from src.chess.board_state import BoardState
from src.chess.validation import validate_position,fen_result
from src.chess.fen import GameState
from src.chess.fen import board_from_fen


def test_valid_starting_board_with_explicit_rights():
    result=validate_position(board_from_fen(chess.STARTING_FEN),GameState(castling='KQkq'))
    assert result['fen_parseable'] and result['standard_chess_valid']
    assert result['diagnostics']==[]


def test_missing_and_duplicate_kings_are_reported_without_correction():
    rows=[[None]*8 for _ in range(8)];rows[7][0]='K';rows[7][1]='K'
    board=BoardState(rows,'white_at_bottom');result=validate_position(board)
    assert result['fen_parseable'] and not result['standard_chess_valid']
    assert 'white_king_count_2_expected_1' in result['diagnostics']
    assert 'no_black_king' in result['diagnostics']
    assert board.rows[7][:2]==('K','K')


def test_backrank_pawn_and_bad_rights_are_diagnostics_for_composed_positions():
    rows=[[None]*8 for _ in range(8)];rows[7][0]='K';rows[0][7]='k';rows[0][0]='P'
    result=validate_position(BoardState(rows,'white_at_bottom'),GameState(castling='K'))
    assert 'pawns_on_backrank' in result['diagnostics']
    assert 'bad_castling_rights' in result['diagnostics']


def test_game_state_provenance_does_not_claim_visual_inference():
    result=fen_result(board_from_fen(chess.STARTING_FEN),GameState(side_to_move='b'),['side_to_move'])
    assert result['game_state_fields']['side_to_move']['source']=='supplied'
    assert result['game_state_fields']['castling']=={'value':'-','source':'default'}
    assert not result['game_state_inferred_from_image']
