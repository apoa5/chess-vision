"""Select sparse unverified still candidates from an original guided-scan ZIP."""
import argparse
from pathlib import Path
import sys
from zipfile import BadZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scanning.frame_selection import extract_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-frames', type=int, default=12)
    parser.add_argument('--min-gap-ms', type=float, default=2000)
    parser.add_argument('--min-blur', type=float, default=30)
    parser.add_argument('--min-luma', type=float, default=20)
    parser.add_argument('--max-luma', type=float, default=235)
    parser.add_argument('--duplicate-distance', type=int, default=6)
    args = parser.parse_args()
    try:
        report = extract_candidates(args.archive, args.output_dir, max_frames=args.max_frames,
                                    min_gap_ms=args.min_gap_ms, min_blur=args.min_blur,
                                    min_luma=args.min_luma, max_luma=args.max_luma,
                                    duplicate_distance=args.duplicate_distance)
        print(f"Preserved original sequence; selected {len(report['candidates'])}/{report['source_frame_count']} unverified candidates.")
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
        parser.exit(1, f'Frame selection failed: {exc}\n')


if __name__ == '__main__':
    main()
