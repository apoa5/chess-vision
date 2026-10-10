"""Save 64 indexed crops and a grid preview from a normalized board."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.image_io import ensure_output_directory, load_image, save_image
from src.vision.square_extraction import extract_squares
from src.vision.pipeline import save_square_crops
from src.utils.visualization import draw_board_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input normalized, top-down board image")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Directory for crops and metadata")
    args = parser.parse_args()
    try:
        settings = load_config(args.config)
        board = load_image(args.image)
        squares = extract_squares(board)
        output = ensure_output_directory(args.output_dir or settings["outputs"]["square_crops"] / args.image.stem)
        save_square_crops(squares, output)
        save_image(output / "grid.png", draw_board_grid(board))
        print(f"Saved {len(squares)} squares and metadata to: {output}")
    except (OSError, ValueError, NotImplementedError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
