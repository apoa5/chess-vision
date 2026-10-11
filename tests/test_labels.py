import chess
import pytest
from src.chess.labels import CLASSES, CLASS_TO_SYMBOL, get_class, class_for_symbol


def test_canonical_order_and_symbols():
    assert len(CLASSES) == 13
    assert [c.class_id for c in CLASSES] == list(range(13))
    assert len({c.name for c in CLASSES}) == 13
    assert [c.symbol for c in CLASSES] == [None, 'P', 'N', 'B', 'R', 'Q', 'K', 'p', 'n', 'b', 'r', 'q', 'k']
    for label in CLASSES:
        assert CLASS_TO_SYMBOL[label.name] == label.symbol
        assert class_for_symbol(label.symbol) == label
        if label.symbol is not None:
            assert chess.Piece.from_symbol(label.symbol).symbol() == label.symbol
    with pytest.raises(TypeError):
        CLASS_TO_SYMBOL['empty'] = '.'


@pytest.mark.parametrize('value', [-1, 13, True, 1.0, 'pawn'])
def test_bad_ids(value):
    with pytest.raises(ValueError):
        get_class(value)
