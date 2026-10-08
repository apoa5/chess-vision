"""Day 2 CLI scaffold for board detection (algorithm scheduled for Day 4)."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.image_io import load_image
from src.utils.visualization import save_debug_visualization
from src.vision.board_detection import detect_board_corners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input chessboard photograph")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="Debug image path")
    args = parser.parse_args()
    try:
        settings = load_config(args.config)
        image = load_image(args.image)
        detection = detect_board_corners(image)
        if not detection.success:
            raise ValueError(detection.reason or "No reliable board candidate found.")
        corners = detection.corners
        output = args.output or settings["outputs"]["board_detection"] / f"{args.image.stem}_corners.png"
        save_debug_visualization(output, image, corners=corners)
        print(f"Corners (TL, TR, BR, BL): {corners.tolist()}")
        print(f"Saved: {output}")
    except (OSError, ValueError, NotImplementedError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
