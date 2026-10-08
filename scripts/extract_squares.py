"""Day 2 CLI scaffold for square extraction (scheduled for Day 7)."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.image_io import ensure_output_directory, load_image, save_image
from src.vision.square_extraction import extract_squares


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Input normalized, top-down board image")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Directory for crops and metadata")
    args = parser.parse_args()
    try:
        settings = load_config(args.config)
        squares = extract_squares(load_image(args.image))
        output = ensure_output_directory(args.output_dir or settings["outputs"]["square_crops"] / args.image.stem)
        metadata = []
        for square in squares:
            filename = f"r{square.row}_c{square.col}.png"
            save_image(output / filename, square.image)
            metadata.append({**square.metadata(), "filename": filename})
        (output / "squares.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {len(squares)} squares and metadata to: {output}")
    except (OSError, ValueError, NotImplementedError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
