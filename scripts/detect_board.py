"""Detect the 8×8 playing area and save corners, edges, and candidate diagnostics."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.image_io import load_image
from src.utils.detection_debug import save_detection_debug
from src.vision.board_detection import DetectionSettings, detect_board_corners


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input chessboard photograph")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, help="Debug image path")
    args = parser.parse_args()
    try:
        settings = load_config(args.config)
        image = load_image(args.image)
        options = DetectionSettings(**settings.get("board_detection", {}))
        detection = detect_board_corners(image, options)
        output = args.output or settings["outputs"]["board_detection"] / f"{args.image.stem}_corners.png"
        save_detection_debug(output, image, detection, options)
        print(f"Saved: {output}")
        if not detection.success:
            parser.exit(1, f"Detection failed: {detection.reason}\n")
        print(f"Corners (TL, TR, BR, BL): {detection.corners.tolist()}")
        print(f"Candidate score (not a probability): {detection.score:.3f}")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
