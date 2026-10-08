"""Day 2 CLI scaffold for perspective correction (scheduled for Day 6)."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.image_io import load_image, save_image
from src.vision.perspective import order_corners, warp_board


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input chessboard photograph")
    parser.add_argument("--corners", type=float, nargs=8, required=True, metavar="COORD",
                        help="Four x y pairs in any order, in original image pixels")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="Rectified image path")
    args = parser.parse_args()
    try:
        settings = load_config(args.config)
        image = load_image(args.image)
        corners = order_corners(np.asarray(args.corners).reshape(4, 2))
        board = warp_board(image, corners, settings["board"]["normalized_size"])
        output = args.output or settings["outputs"]["rectified_boards"] / f"{args.image.stem}.png"
        print(f"Saved: {save_image(output, board)}")
    except (OSError, ValueError, NotImplementedError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
