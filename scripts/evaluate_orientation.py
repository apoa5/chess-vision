"""Audit marking-based orientation separately from board localization."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import DEFAULT_CONFIG, load_config
from src.utils.image_io import ensure_output_directory, load_image
from src.vision.board_detection import DetectionSettings, detect_board_corners
from src.vision.orientation import OrientationSettings, resolve_orientation
from src.vision.pipeline import rectify_with_config

EXPECTED = {'white_near_camera': 0, 'white_left': 1, 'black_near_camera': 2, 'white_right': 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--all', action='store_true', help='Audit all registered photos, not just the fixed subset')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--report', type=Path, help='Optional compact versionable JSON snapshot')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        metadata = config['data']['metadata']
        images = json.loads((metadata/'images.json').read_text())
        names = sorted(images) if args.all else json.loads((metadata/'development_subset.json').read_text())['image_ids']
        output = ensure_output_directory(args.output_dir or config['outputs']['rectified_boards'].parent/'week1/orientation_evaluation')
        rows = []
        for name in names:
            photo = load_image(config['data']['raw']/name)
            detection = detect_board_corners(photo, DetectionSettings(**config.get('board_detection', {})))
            expected = EXPECTED.get(images[name]['orientation'])
            row = {'image_id': name, 'board_detected': detection.success, 'expected_rotation_quarters_ccw': expected,
                   'status': 'not_detected', 'predicted_rotation_quarters_ccw': None, 'score': None, 'correct': False if expected is not None else None}
            if detection.success:
                result = resolve_orientation(rectify_with_config(photo, detection.corners, config),
                                             OrientationSettings(**config.get('orientation', {})))
                row.update(status=result.status, predicted_rotation_quarters_ccw=result.rotation_quarters_ccw,
                           score=result.score, correct=(result.rotation_quarters_ccw == expected) if expected is not None else None)
                (output/f'{Path(name).stem}.json').write_text(json.dumps(result.metadata(), indent=2)+'\n')
            rows.append(row)
            print(name, row['status'], row['predicted_rotation_quarters_ccw'], 'expected', expected, flush=True)
        # Calibration image is explicitly reported and excluded from headline metrics.
        reference = 'position_001_var_08.jpg'
        audit = [r for r in rows if r['image_id'] != reference and r['correct'] is not None]
        audited_resolved = [r for r in audit if r['status'] == 'resolved']
        report = {'scope': 'Development orientation audit; detector acceptance is not verified localization correctness.',
                  'calibration_image_excluded_from_metrics': reference, 'image_count': len(rows),
                  'audited_image_count': len(audit), 'board_candidates': sum(r['board_detected'] for r in rows),
                  'resolved_count': len(audited_resolved), 'correct_resolved_count': sum(r['correct'] for r in audited_resolved),
                  'wrong_resolved_count': sum(not r['correct'] for r in audited_resolved),
                  'uncertain_count': sum(r['status'] == 'uncertain' for r in audit),
                  'not_detected_count': sum(r['status'] == 'not_detected' for r in audit),
                  'resolved_accuracy': sum(r['correct'] for r in audited_resolved)/len(audited_resolved) if audited_resolved else None,
                  'end_to_end_orientation_rate': sum(r['correct'] for r in audit)/len(audit) if audit else None,
                  'results': rows}
        encoded = json.dumps(report, indent=2)+'\n'
        (output/'evaluation.json').write_text(encoded)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(encoded)
        print({k:v for k,v in report.items() if k!='results'})
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
