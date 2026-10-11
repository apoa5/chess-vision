"""Canonical, versioned class order shared by datasets, models, and boards."""

from dataclasses import dataclass
from types import MappingProxyType

LABEL_VERSION = 'chess_pieces_v1'


@dataclass(frozen=True)
class PieceClass:
    class_id: int
    name: str
    symbol: str | None

    @property
    def display_name(self):
        return self.name.replace('_', ' ').title()


_TYPES = (('pawn', 'p'), ('knight', 'n'), ('bishop', 'b'), ('rook', 'r'), ('queen', 'q'), ('king', 'k'))
CLASSES = (PieceClass(0, 'empty', None),) + tuple(
    PieceClass(1 + color_index * 6 + index, f'{color}_{piece}', symbol.upper() if color == 'white' else symbol)
    for color_index, color in enumerate(('white', 'black')) for index, (piece, symbol) in enumerate(_TYPES))
CLASS_NAMES = tuple(label.name for label in CLASSES)
CLASS_TO_SYMBOL = MappingProxyType({label.name: label.symbol for label in CLASSES})
NAME_TO_ID = MappingProxyType({label.name: label.class_id for label in CLASSES})
SYMBOL_TO_ID = MappingProxyType({label.symbol: label.class_id for label in CLASSES})


def get_class(class_id):
    if type(class_id) is not int or not 0 <= class_id < len(CLASSES):
        raise ValueError('Class ID must be an integer from 0 to 12.')
    return CLASSES[class_id]


def class_for_symbol(symbol):
    if symbol is not None and not isinstance(symbol, str):
        raise ValueError('Piece symbol must be None or a canonical piece symbol.')
    if symbol not in SYMBOL_TO_ID:
        raise ValueError(f'Invalid piece symbol: {symbol!r}')
    return get_class(SYMBOL_TO_ID[symbol])
