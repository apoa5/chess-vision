"""Run photo → detection → normalized board → grid preview → 64 square crops."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG, load_config
from src.vision.pipeline import process_board_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Original chessboard photograph")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Root for detection, rectified boards, and crops")
    parser.add_argument("--white-side", choices=["bottom", "left", "top", "right"],
                        help="Explicit orientation in the rectified image if markings are unreadable")
    parser.add_argument("--require-orientation", action="store_true", help="Exit 1 if automatic orientation is uncertain")
    args = parser.parse_args()
    try:
        result = process_board_image(args.image, load_config(args.config), args.output_dir, args.white_side)
        print(f"Detection preview: {result['overlay_path']}")
        if not result["board_detected"]:
            parser.exit(1, f"Detection failed: {result['reason']}\n")
        print(f"Board: {result['board_path']}\nGrid: {result['grid_path']}\n"
              f"Saved {result['square_count']} squares: {result['crop_directory']}")
        print("Inspect the grid preview to verify localization and square alignment.")
        print(f"Context board: {result['context_path']}\nContext crops: {result['context_crop_directory']}")
        print(f"Orientation: {result['orientation']['status']}; coordinates: {result['coordinate_system']}")
        if result["orientation"]["status"] == "uncertain":
            message = "Orientation unresolved. Inspect the rectified image and supply --white-side; crops remain image-relative.\n"
            if args.require_orientation:
                parser.exit(1, message)
            print(message, file=sys.stderr)
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
