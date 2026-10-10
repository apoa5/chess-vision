"""Detect the playing area and save a perspective-corrected square board."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.image_io import load_image, save_image
from src.vision.perspective import order_corners
from src.vision.board_detection import DetectionSettings, detect_board_corners
from src.vision.pipeline import rectify_with_config, save_rectified_context


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input chessboard photograph")
    parser.add_argument("--corners", type=float, nargs=8, metavar="COORD",
                        help="Optional manual override: four x y pairs in original image pixels")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="Rectified image path")
    args = parser.parse_args()
    try:
        settings = load_config(args.config)
        image = load_image(args.image)
        if args.corners is not None:
            corners = order_corners(np.asarray(args.corners).reshape(4, 2))
        else:
            detection = detect_board_corners(image, DetectionSettings(**settings.get("board_detection", {})))
            if not detection.success:
                raise ValueError(f"Detection failed: {detection.reason}")
            corners = detection.corners
        rectified = rectify_with_config(image, corners, settings)
        board = rectified.board_image
        output = args.output or settings["outputs"]["rectified_boards"] / f"{args.image.stem}.png"
        print(f"Saved: {save_image(output, board)}")
        context = save_rectified_context(rectified, output)
        print(f"Context: {context['context_path']}")
    except (OSError, ValueError, NotImplementedError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
