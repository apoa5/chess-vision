"""Run the Day 1 acceptance check from any working directory."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess

from src.chess.engine import StockfishEngine
from src.utils.config import DEFAULT_CONFIG, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--engine", help="Override the configured Stockfish executable")
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()
    try:
        settings = load_config(args.config)
        board = chess.Board()
        with StockfishEngine(args.engine or settings["stockfish"]["executable"]) as engine:
            result = engine.analyze(board, depth=args.depth)
        print(f"FEN: {board.fen()}")
        print(f"Best move: {result.best_move}")
        print(f"Evaluation (White POV, centipawns or #mate): {result.evaluation}")
        print("Principal variation:", " ".join(move.uci() for move in result.principal_variation))
    except (OSError, ValueError, chess.engine.EngineError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
