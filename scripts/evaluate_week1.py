"""Run the full Week 1 pipeline on the fixed subset and write reviewed metrics."""

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config
from src.utils.image_io import ensure_output_directory
from src.vision.evaluation import apply_review
from src.vision.pipeline import process_board_image


def apply_warp_review(record, review):
    """Generating 64 crops is not evidence that the board was localized correctly."""
    if not record["board_detected"]:
        return {"warp_correct": False, "warp_review_status": "not_detected", "warp_notes": ""}
    if review and review.get("warp_signature") == record["warp_signature"]:
        correct = review.get("warp_correct")
        if type(correct) is not bool:
            raise ValueError("Warp review correctness must be boolean.")
        return {"warp_correct": correct, "warp_review_status": "matched", "warp_notes": review.get("notes", "")}
    return {"warp_correct": None, "warp_review_status": "stale" if review else "unreviewed", "warp_notes": ""}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, help="Artifact root; defaults to outputs/week1")
    parser.add_argument("--reviews", type=Path, help="Warp reviews; defaults to metadata/day7_review.json")
    parser.add_argument("--report", type=Path, help="Additional JSON snapshot, with a companion CSV")
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        metadata = config["data"]["metadata"]
        manifest = json.loads((metadata / "development_subset.json").read_text())
        names = manifest["image_ids"]
        if not isinstance(names, list) or not all(isinstance(n, str) and Path(n).name == n for n in names) or len(names) != 20 or len(set(names)) != 20:
            raise ValueError("Expected 20 unique image filenames in the fixed manifest.")
        localization_path = metadata / "day5_review.json"
        localization = json.loads(localization_path.read_text()) if localization_path.exists() else {}
        review_path = args.reviews or metadata / "day7_review.json"
        if args.reviews and not review_path.exists():
            raise FileNotFoundError(f"Review file not found: {review_path}")
        reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
        if not isinstance(reviews, dict) or not isinstance(localization, dict):
            raise ValueError("Reviews must be objects keyed by image filename.")
        output = ensure_output_directory(args.output_dir or config["outputs"]["rectified_boards"].parent / "week1")
        records = []
        for name in names:
            record = process_board_image(config["data"]["raw"] / name, config, output)
            record.update(apply_review(record["board_detected"], record["detection_signature"], localization.get(name)))
            record["board_detection_correct"] = record.pop("board_localization_correct")
            record.update(apply_warp_review(record, reviews.get(name)))
            for key in ("overlay_path", "board_path", "grid_path", "crop_directory", "context_path",
                        "context_grid_path", "valid_mask_path", "context_crop_directory"):
                if record[key] is not None:
                    path = Path(record[key]).resolve()
                    record[key] = str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)
            records.append(record)
            print(name, f"detected={record['board_detected']} squares={record['square_count']} warp_correct={record['warp_correct']}", flush=True)
        known = all(r["board_detection_correct"] is not None for r in records)
        accuracy = sum(r["board_detection_correct"] is True for r in records) / len(records) if known else None
        report = {"subset": manifest["name"], "image_count": len(records),
                  "normalized_size": config["board"]["normalized_size"],
                  "localization_accuracy": accuracy, "target_accuracy": .90,
                  "target_met": accuracy >= .90 if accuracy is not None else None,
                  "boards_detected": sum(r["board_detected"] for r in records),
                  "localizations_correct": sum(r["board_detection_correct"] is True for r in records),
                  "warps_correct": sum(r["warp_correct"] is True for r in records),
                  "unreviewed_warps": sum(r["warp_correct"] is None for r in records),
                  "boards_with_64_crops": sum(r["square_count"] == 64 for r in records),
                  "results": records}
        (output / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        with (output / "evaluation.csv").open("w", newline="", encoding="utf-8") as stream:
            columns = ["image_id", "board_detected", "board_detection_correct", "warp_correct", "square_count", "review_status", "warp_review_status", "reason", "notes", "warp_notes", "grid_path", "crop_directory"]
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            args.report.with_suffix(".csv").write_text((output / "evaluation.csv").read_text(), encoding="utf-8")
        print(f"Saved Week 1 evaluation: {output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
