import pytest
from src.chess.board_state import BoardState


def test_board_shape_orientation_and_independent_storage():
    rows = [[None]*8 for _ in range(8)]
    rows[0][0] = 'r'
    state = BoardState(rows)
    rows[0][0] = 'P'
    assert state.rows[0][0] == 'r'
    assert state.coordinate_system == 'image_relative'
    assert state.metadata()['rows'][0][0] == 'r'
    assert BoardState(rows, 'white_at_bottom').coordinate_system == 'white_at_bottom'


@pytest.mark.parametrize('rows', [[], [[None]*8]*7, [[None]*7]*8, ['........']*8, None])
def test_bad_shapes(rows):
    with pytest.raises(ValueError, match='8 rows'):
        BoardState(rows)


@pytest.mark.parametrize('symbol', ['.', '', 'white_pawn', 'X', 1, []])
def test_bad_symbols(symbol):
    rows = [[None]*8 for _ in range(8)]
    rows[3][4] = symbol
    with pytest.raises(ValueError, match='symbol'):
        BoardState(rows)


def test_board_does_not_claim_chess_legality():
    assert BoardState([['K']*8 for _ in range(8)]).rows[0][0] == 'K'
    with pytest.raises(ValueError, match='coordinate'):
        BoardState([[None]*8]*8, 'unknown')
