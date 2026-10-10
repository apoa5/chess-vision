"""Run the Day 4 baseline on the fixed development subset; save all debug results."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.detection_debug import save_detection_debug
from src.utils.image_io import ensure_output_directory, load_image
from src.vision.board_detection import DetectionSettings, detect_board_corners


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        settings = DetectionSettings(**config.get("board_detection", {}))
        manifest = json.loads((config["data"]["metadata"] / "development_subset.json").read_text())
        output = ensure_output_directory(args.output_dir or config["outputs"]["board_detection"])
        records = []
        for name in manifest["image_ids"]:
            try:
                image = load_image(config["data"]["raw"] / name)
                detection = detect_board_corners(image, settings)
                record = save_detection_debug(output / f"{Path(name).stem}_corners.png", image, detection, settings)
                records.append({"image_id": name, "success": record["success"],
                                "score": record["score"], "reason": record["reason"],
                                "overlay_path": record["overlay_path"], "manual_correct": None})
            except (OSError, ValueError) as error:
                records.append({"image_id": name, "success": False, "reason": str(error),
                                "manual_correct": None})
            print(name, "candidate found" if records[-1]["success"] else records[-1]["reason"])
        summary = {"subset": manifest["name"], "candidate_found_count": sum(r["success"] for r in records),
                   "image_count": len(records), "note": "Candidate acceptance is not measured localization accuracy; inspect overlays.",
                   "results": records}
        (output / "day4_results.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Candidates found: {summary['candidate_found_count']}/{summary['image_count']}; debug files: {output}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
