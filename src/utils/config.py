"""Central settings; relative paths are resolved from the project root."""

import os
import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config/settings.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load and validate settings, with an optional STOCKFISH_PATH override."""
    with Path(path).open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    engine = settings.get("stockfish", {})
    if not isinstance(engine, dict):
        raise ValueError("stockfish must be a mapping.")
    executable = os.environ.get("STOCKFISH_PATH") or engine.get("executable")
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("stockfish.executable must be a nonempty path or command.")
    settings["stockfish"] = {**engine, "executable": executable}
    board = settings.get("board", {})
    size = board.get("normalized_size") if isinstance(board, dict) else None
    if type(size) is not int or size <= 0 or size % 8:
        raise ValueError("board.normalized_size must be a positive integer divisible by 8.")
    for key, default in (("context_margin_squares", 2.0), ("crop_padding_squares", 2.0)):
        value = board.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 4:
            raise ValueError(f"board.{key} must be a finite number between 0 and 4.")
        board[key] = value
    if board["crop_padding_squares"] > board["context_margin_squares"]:
        raise ValueError("Crop padding cannot exceed the context margin.")
    for section, keys in {
        "data": ("raw", "processed", "test", "metadata"),
        "outputs": ("board_detection", "rectified_boards", "square_crops"),
    }.items():
        paths = settings.get(section)
        if not isinstance(paths, dict):
            raise ValueError(f"{section} must be a mapping.")
        for key in keys:
            value = paths.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{section}.{key} must be a nonempty path.")
            paths[key] = (PROJECT_ROOT / Path(value).expanduser()).resolve()
    return settings
