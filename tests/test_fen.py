import chess
import pytest
from src.chess.board_state import BoardState
from src.chess.fen import GameState,placement_fen,full_fen
from src.chess.fen import board_from_fen


def test_starting_position_round_trip():
    board=board_from_fen(chess.STARTING_FEN)
    assert placement_fen(board)==chess.STARTING_FEN.split()[0]
    assert full_fen(board,GameState(castling='KQkq'))==chess.STARTING_FEN


def test_empty_runs_at_start_middle_and_end():
    rows=[[None]*8 for _ in range(8)]
    rows[0]=[None,None,'k',None,None,'r',None,None]
    rows[7]=['K',None,None,None,None,None,None,'R']
    board=BoardState(rows,'white_at_bottom')
    assert placement_fen(board)=='2k2r2/8/8/8/8/8/8/K6R'
    state=GameState('b','-','e3',7,20)
    assert full_fen(board,state).endswith(' b - e3 7 20')
    assert chess.Board(full_fen(board,state)).board_fen()==placement_fen(board)


def test_unresolved_orientation_cannot_generate_fen():
    with pytest.raises(ValueError,match='White-at-bottom'):
        placement_fen(BoardState([[None]*8 for _ in range(8)]))


@pytest.mark.parametrize('kwargs',[{'side_to_move':'x'},{'castling':'KK'},{'castling':''},{'castling':'qK'},
                                   {'en_passant':'e4'},{'halfmove_clock':-1},{'halfmove_clock':True},{'fullmove_number':0}])
def test_invalid_game_state_is_rejected(kwargs):
    with pytest.raises(ValueError): GameState(**kwargs)
