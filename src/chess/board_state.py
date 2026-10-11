"""Image-relative or explicitly orientation-normalized 8×8 piece placement."""

from dataclasses import dataclass
from src.chess.labels import class_for_symbol


@dataclass(frozen=True)
class BoardState:
    """Row 0 is the image top; columns increase left-to-right.

    None means empty. Shape/symbol validation does not assert chess legality.
    white_at_bottom is an explicit orientation assertion, never inferred here.
    No side-to-move, castling, clocks, FEN, or legal-move logic is included.
    """
    rows: tuple
    coordinate_system: str = 'image_relative'

    def __post_init__(self):
        if self.coordinate_system not in ('image_relative', 'white_at_bottom'):
            raise ValueError('Unknown board coordinate system.')
        try:
            if isinstance(self.rows, (str, bytes)):
                raise ValueError('Board must have 8 rows of 8 cells.')
            rows = tuple(self.rows)
            if any(isinstance(row, (str, bytes)) for row in rows):
                raise ValueError('Board must have 8 rows of 8 cells.')
            rows = tuple(tuple(row) for row in rows)
        except TypeError as error:
            raise ValueError('Board must have 8 rows of 8 cells.') from error
        if len(rows) != 8 or any(len(row) != 8 for row in rows):
            raise ValueError('Board must have 8 rows of 8 cells.')
        for row in rows:
            for symbol in row:
                class_for_symbol(symbol)
        object.__setattr__(self, 'rows', rows)

    def metadata(self):
        return {'rows': [list(row) for row in self.rows], 'coordinate_system': self.coordinate_system}
