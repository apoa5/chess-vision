"""Small synchronous Stockfish wrapper using python-chess's UCI interface."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.engine

from src.utils.config import PROJECT_ROOT


def resolve_engine_path(executable: str | Path) -> Path:
    """Find a command on PATH or validate a project-relative/absolute path."""
    value = str(executable)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        found = shutil.which(value) if candidate.parent == Path(".") else None
        candidate = Path(found) if found else PROJECT_ROOT / candidate
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Stockfish executable not found: {candidate}. "
            "Install Stockfish and set STOCKFISH_PATH or stockfish.executable."
        )
    if not os.access(candidate, os.X_OK):
        raise PermissionError(f"Stockfish file is not executable: {candidate}")
    return candidate.resolve()


@dataclass(frozen=True)
class Analysis:
    best_move: chess.Move | None
    evaluation: chess.engine.Score  # White's point of view: centipawns or mate.
    principal_variation: tuple[chess.Move, ...]


class StockfishEngine:
    def __init__(self, executable: str | Path):
        self.path = resolve_engine_path(executable)
        self._engine = chess.engine.SimpleEngine.popen_uci(str(self.path))

    def analyze(self, position: str | chess.Board, *, time: float = 0.2,
                depth: int | None = None) -> Analysis:
        if self._engine is None:
            raise RuntimeError("Stockfish engine is closed.")
        if time <= 0 or (depth is not None and depth <= 0):
            raise ValueError("Analysis time and depth must be positive.")
        board = chess.Board(position) if isinstance(position, str) else position.copy()
        if not board.is_valid():
            raise ValueError("Position is not a valid chess position.")
        limit = chess.engine.Limit(depth=depth) if depth is not None else chess.engine.Limit(time=time)
        info = self._engine.analyse(board, limit)
        pv = tuple(info.get("pv", ()))
        return Analysis(pv[0] if pv else None, info["score"].white(), pv)

    def best_move(self, position: str | chess.Board, **limits) -> chess.Move | None:
        return self.analyze(position, **limits).best_move

    def evaluate(self, position: str | chess.Board, **limits) -> chess.engine.Score:
        return self.analyze(position, **limits).evaluation

    def close(self) -> None:
        engine, self._engine = self._engine, None
        if engine is not None:
            try:
                engine.quit()
            finally:
                engine.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
