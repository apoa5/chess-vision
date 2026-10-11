"""Standard-chess diagnostics; never alter or reject composed piece placements."""
from collections import Counter
import chess
from src.chess.board_state import BoardState
from src.chess.fen import GameState, placement_fen, full_fen


def validate_position(board: BoardState, game_state: GameState | None = None):
    fen = full_fen(board,game_state)
    parsed = chess.Board(fen)
    status = parsed.status()
    diagnostics = [name.removeprefix('STATUS_').lower() for name,value in vars(chess).items()
                   if name.startswith('STATUS_') and value and status & value]
    counts = Counter(symbol for row in board.rows for symbol in row if symbol is not None)
    for symbol,color in [('K','white'),('k','black')]:
        if counts[symbol] != 1:
            diagnostics.append(f'{color}_king_count_{counts[symbol]}_expected_1')
    return {'fen_parseable':True,'standard_chess_valid':parsed.is_valid(),
            'status_code':int(status),'diagnostics':diagnostics,
            'piece_counts':dict(sorted(counts.items())), 'placement_fen':placement_fen(board),
            'full_fen':fen,
            'note':'Standard-chess consistency checks are not a proof of reachability or recognition accuracy. Composed puzzles may intentionally violate them; turn-dependent checks use supplied/default game state.'}


def fen_result(board, game_state=None, supplied_fields=()):
    state = game_state or GameState()
    fields = {name:{'value':getattr(state,name),'source':'supplied' if name in supplied_fields else 'default'}
              for name in state.__dataclass_fields__}
    return {'placement_fen':placement_fen(board),'full_fen':full_fen(board,state),
            'game_state_fields':fields,'game_state_inferred_from_image':False,
            'validation':validate_position(board,state)}
