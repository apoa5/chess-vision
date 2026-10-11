"""Prepare normalized boards, then build FEN-labeled crops from approved reviews."""
import argparse
import json
from pathlib import Path
import sys
import cv2
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.config import DEFAULT_CONFIG, load_config
from src.vision.dataset import prepare_boards, build_crops, apply_visibility_reviews, dataset_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--reviews', type=Path)
    parser.add_argument('--summary', type=Path, help='Compact results path; defaults to capture metadata directory.')
    parser.add_argument('--preview-crops', action='store_true', help='Generate unapproved candidate crops; never replace the training manifest.')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    # The detector scores many small candidates; thread launch overhead dominates.
    cv2.setNumThreads(1)
    try:
        config = load_config(args.config)
        output = args.output_dir or config['classifier']['dataset_paths']['metadata'].parent
        records = prepare_boards(config, output)
        if not args.prepare_only:
            review_path = args.reviews or config['data']['metadata']/'day9_board_reviews.json'
            reviews = json.loads(review_path.read_text()) if review_path.exists() else {}
            report = build_crops(records, reviews, config, output, preview_only=args.preview_crops)
            if args.preview_crops:
                print(f"Built {report['crop_count']} unapproved preview crops; review geometry before training.")
                return
            visibility_path = config['data']['metadata']/'day9_visibility_reviews.json'
            if visibility_path.exists():
                apply_visibility_reviews(report, json.loads(visibility_path.read_text()))
                (output/'crops.json').write_text(json.dumps(report, indent=2)+'\n')
            summary_text = json.dumps(dataset_summary(report, records), indent=2)+'\n'
            (output/'summary.json').write_text(summary_text)
            summary_path = args.summary or config['data']['metadata']/'day9_dataset_summary.json'
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(summary_text)
            print(f"Built {report['crop_count']} labeled crops from {report['board_count']} approved boards.")
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')

if __name__ == '__main__':
    main()
