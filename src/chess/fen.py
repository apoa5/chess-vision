"""Piece-placement serialization with explicit, nonvisual game-state fields."""
from dataclasses import dataclass
import re
from src.chess.board_state import BoardState


@dataclass(frozen=True)
class GameState:
    side_to_move: str = 'w'
    castling: str = '-'
    en_passant: str = '-'
    halfmove_clock: int = 0
    fullmove_number: int = 1

    def __post_init__(self):
        if self.side_to_move not in ('w','b'):
            raise ValueError('Side to move must be w or b.')
        if not isinstance(self.castling,str) or not re.fullmatch(r'-|K?Q?k?q?',self.castling) or not self.castling:
            raise ValueError('Castling must be - or an ordered subset of KQkq.')
        if not isinstance(self.en_passant,str) or not re.fullmatch(r'-|[a-h][36]',self.en_passant):
            raise ValueError('En passant must be - or a square on rank 3 or 6.')
        if type(self.halfmove_clock) is not int or self.halfmove_clock < 0:
            raise ValueError('Halfmove clock must be a nonnegative integer.')
        if type(self.fullmove_number) is not int or self.fullmove_number < 1:
            raise ValueError('Fullmove number must be a positive integer.')


def placement_fen(board: BoardState) -> str:
    """Serialize a resolved White-bottom matrix: row zero is rank eight."""
    if not isinstance(board,BoardState) or board.coordinate_system != 'white_at_bottom':
        raise ValueError('FEN requires an explicitly White-at-bottom BoardState.')
    ranks = []
    for row in board.rows:
        rank = ''; empty = 0
        for symbol in row:
            if symbol is None:
                empty += 1
            else:
                if empty: rank += str(empty); empty = 0
                rank += symbol
        if empty: rank += str(empty)
        ranks.append(rank)
    return '/'.join(ranks)


def full_fen(board: BoardState, game_state: GameState | None = None) -> str:
    state = game_state or GameState()
    return f'{placement_fen(board)} {state.side_to_move} {state.castling} {state.en_passant} {state.halfmove_clock} {state.fullmove_number}'


def board_from_fen(fen: str) -> BoardState:
    """Parse a reference/game position into the canonical orientation."""
    import chess
    parsed = chess.Board(fen)
    rows = [[parsed.piece_at(chess.square(c,7-r)).symbol() if parsed.piece_at(chess.square(c,7-r)) else None
             for c in range(8)] for r in range(8)]
    return BoardState(rows,'white_at_bottom')
