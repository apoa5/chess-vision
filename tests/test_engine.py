import chess
import pytest

from src.chess.engine import StockfishEngine, resolve_engine_path
from src.utils.config import load_config


@pytest.fixture
def engine():
    with StockfishEngine(load_config()["stockfish"]["executable"]) as instance:
        yield instance


def test_executable_found():
    assert resolve_engine_path(load_config()["stockfish"]["executable"]).is_file()


def test_missing_engine(tmp_path):
    with pytest.raises(FileNotFoundError, match="Stockfish executable not found"):
        StockfishEngine(tmp_path / "missing")


def test_nonexecutable_file(tmp_path):
    path = tmp_path / "stockfish"
    path.write_text("not an executable")
    with pytest.raises(PermissionError, match="not executable"):
        resolve_engine_path(path)


@pytest.mark.parametrize("as_fen", [False, True])
def test_valid_analysis(engine, as_fen):
    board = chess.Board()
    result = engine.analyze(board.fen() if as_fen else board, depth=6)
    assert result.best_move in board.legal_moves
    assert result.evaluation.score() is not None
    replay = board.copy()
    for move in result.principal_variation:
        assert move in replay.legal_moves
        replay.push(move)
    assert board.fen() == chess.STARTING_FEN


def test_convenience_methods(engine):
    board = chess.Board()
    assert engine.best_move(board, depth=4) in board.legal_moves
    assert engine.evaluate(board, time=0.05).score() is not None


def test_invalid_position(engine):
    with pytest.raises(ValueError):
        engine.analyze("invalid fen")
    with pytest.raises(ValueError, match="valid chess position"):
        engine.analyze(chess.Board.empty())


def test_invalid_limit(engine):
    with pytest.raises(ValueError, match="positive"):
        engine.analyze(chess.Board(), depth=0)


def test_close_is_idempotent(engine):
    engine.close()
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.analyze(chess.Board())


def test_checkmate_has_no_move(engine):
    board = chess.Board()
    for move in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push_uci(move)
    result = engine.analyze(board, depth=4)
    assert result.best_move is None
    assert result.evaluation.is_mate()
