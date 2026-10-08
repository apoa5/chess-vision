"""Load/save a photo and optionally annotate known corners for Day 2 checks."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.utils.image_io import load_image
from src.utils.visualization import save_debug_visualization


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument("--corners", type=float, nargs=8, metavar="COORD",
                        help="Optional four x y pairs in original image pixels, in any order")
    parser.add_argument("--max-size", type=int, help="Optional maximum preview side length")
    args = parser.parse_args()
    try:
        image = load_image(args.image)
        corners = None if args.corners is None else np.asarray(args.corners).reshape(4, 2)
        output = save_debug_visualization(args.output, image, corners=corners, max_size=args.max_size)
        print(f"Loaded: {args.image} ({image.shape[1]}×{image.shape[0]}, BGR)")
        print(f"Saved: {output}")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
