"""Automatically rectify the fixed development subset and retain localization reviews."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.image_io import ensure_output_directory, load_image, save_image
from src.vision.board_detection import DetectionSettings, detect_board_corners
from src.vision.evaluation import apply_review, detection_signature
from src.vision.pipeline import rectify_with_config, save_rectified_context


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        settings = DetectionSettings(**config.get("board_detection", {}))
        metadata = config["data"]["metadata"]
        manifest = json.loads((metadata / "development_subset.json").read_text())
        names = manifest["image_ids"]
        if not isinstance(names, list) or len(names) != 20 or len(set(names)) != 20 or not all(isinstance(n, str) for n in names):
            raise ValueError("Expected the fixed subset of 20 unique image filenames.")
        review_path = metadata / "day5_review.json"
        reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
        output = ensure_output_directory(args.output_dir or config["outputs"]["rectified_boards"])
        records = []
        for name in names:
            source = config["data"]["raw"] / name
            image = load_image(source)
            detection = detect_board_corners(image, settings)
            review = apply_review(detection.success, detection_signature(source, detection, settings), reviews.get(name))
            record = {"image_id": name, "board_detected": detection.success,
                      "reason": detection.reason, "corners": detection.corners.tolist() if detection.success else None,
                      "output_path": None, "output_shape": None, **review}
            if detection.success:
                rectified = rectify_with_config(image, detection.corners, config)
                board = rectified.board_image
                destination = save_image(output / f"{Path(name).stem}.png", board)
                record.update(output_path=str(destination), output_shape=list(board.shape),
                              **save_rectified_context(rectified, destination))
            records.append(record)
            print(name, "saved" if detection.success else "rejected",
                  f"(localization correct: {review['board_localization_correct']})", flush=True)
        report = {"subset": manifest["name"], "normalized_size": config["board"]["normalized_size"],
                  "image_count": len(records), "warped_count": sum(r["board_detected"] for r in records),
                  "results": records}
        (output / "rectification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {report['warped_count']}/{len(records)} boards to {output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
