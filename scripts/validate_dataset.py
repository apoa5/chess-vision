"""Validate Day 3 photos/metadata and optionally write a coverage summary."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import PROJECT_ROOT
from src.utils.dataset import validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=PROJECT_ROOT / "data/metadata")
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument("--report", type=Path, help="Optional JSON summary output")
    args = parser.parse_args()
    try:
        report = validate_dataset(args.metadata_dir, args.raw_dir)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Validated {report['image_count']} photos and {report['valid_fen_count']} position FENs.")
        subset = report["development_subset"]
        print(f"Fixed development subset: {subset['image_count']} photos across {subset['position_count']} positions.")
        print("Capture environments:", report["images_by_environment"])
        print("Unknown image orientations:", len(report["unknown_orientation_images"]))
        target = report["collection_target"]
        print(f"Collection progress: {report['image_count']}/{target['images']} target photos; "
              f"{report['position_count']}/{target['positions']} target positions.")
        if args.report:
            print(f"Saved summary: {args.report}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
