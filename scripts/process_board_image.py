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
    args = parser.parse_args()
    try:
        result = process_board_image(args.image, load_config(args.config), args.output_dir)
        print(f"Detection preview: {result['overlay_path']}")
        if not result["board_detected"]:
            parser.exit(1, f"Detection failed: {result['reason']}\n")
        print(f"Board: {result['board_path']}\nGrid: {result['grid_path']}\n"
              f"Saved {result['square_count']} squares: {result['crop_directory']}")
        print("Inspect the grid preview to verify localization and square alignment.")
        print(f"Context board: {result['context_path']}\nContext crops: {result['context_crop_directory']}")
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
