"""Evaluate the fixed subset, saving debug files, CSV, and review-bound results."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.detection_debug import save_detection_debug
from src.utils.image_io import ensure_output_directory, load_image
from src.vision.board_detection import DetectionSettings, detect_board_corners
from src.vision.evaluation import apply_review, detection_signature, summarize_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reviews", type=Path, help="Manual-review JSON; defaults to metadata/day5_review.json")
    parser.add_argument("--report", type=Path, help="Optional additional JSON summary destination")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        settings = DetectionSettings(**config.get("board_detection", {}))
        manifest = json.loads((config["data"]["metadata"] / "development_subset.json").read_text())
        image_ids = manifest.get("image_ids", [])
        if not isinstance(image_ids, list) or not all(isinstance(name, str) for name in image_ids) or len(image_ids) != 20 or len(set(image_ids)) != 20:
            raise ValueError("Evaluation requires the fixed manifest of 20 unique filenames.")
        review_path = args.reviews or config["data"]["metadata"] / "day5_review.json"
        if args.reviews and not review_path.exists():
            raise FileNotFoundError(f"Review file not found: {review_path}")
        reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
        if not isinstance(reviews, dict):
            raise ValueError("Reviews must be a JSON object keyed by image filename.")
        output = ensure_output_directory(args.output_dir or config["outputs"]["board_detection"] / "day5")
        records = []
        for name in image_ids:
            started = time.monotonic()
            path = config["data"]["raw"] / name
            image = load_image(path)
            result = detect_board_corners(image, settings)
            debug = save_detection_debug(output / f"{Path(name).stem}_corners.png", image, result, settings)
            signature = detection_signature(path, result, settings)
            records.append({"image_id": name, "success": result.success, "candidate_score": result.score,
                            "reason": result.reason, "corners": debug["corners"],
                            "overlay_path": debug["overlay_path"], "detection_signature": signature,
                            "elapsed_seconds": round(time.monotonic() - started, 3),
                            **apply_review(result.success, signature, reviews.get(name))})
            print(name, "candidate found" if result.success else result.reason, flush=True)
        report = {"subset": manifest["name"], "settings": asdict(settings),
                  **summarize_evaluation(records), "results": records}
        encoded = json.dumps(report, indent=2) + "\n"
        (output / "evaluation.json").write_text(encoded, encoding="utf-8")
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(encoded, encoding="utf-8")
        columns = ["image_id", "success", "candidate_score", "board_localization_correct", "review_status",
                   "failure_category", "reason", "overlay_path", "elapsed_seconds", "notes"]
        with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        print(f"Candidates: {report['candidate_found_count']}/{report['image_count']}; "
              f"unreviewed: {report['unreviewed_count']}")
        accuracy = report["localization_accuracy"]
        print("Localization accuracy:", f"{accuracy:.0%}" if accuracy is not None else "pending manual review")
        print(f"Saved evaluation: {output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
