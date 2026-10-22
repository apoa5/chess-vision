"""Validate A.3 companion metadata, ZIP provenance, human reviews and split isolation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import BadZipFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scanning.capture_protocol import (
    ROOT, make_plan, review_digest, sequence_review_digest, split_digest, validate_capture_manifest)


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, nargs='?')
    parser.add_argument('--init-plan', type=Path, help='Write a new empty plan; refuses existing paths')
    parser.add_argument('--asset-root', type=Path, default=ROOT)
    parser.add_argument('--purpose', choices=['audit', 'training', 'validation', 'final_evaluation',
                                             'scan_validation', 'scan_final_evaluation'], default='audit')
    parser.add_argument('--release-final', action='store_true', help='Explicit final-evaluation release, never for model selection')
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--report', type=Path, help='Optional new report path; never overwrites a file')
    parser.add_argument('--print-split-lock', action='store_true', help='Print proposed lock only; not a validation result')
    parser.add_argument('--print-review-digests', action='store_true', help='Print annotation digests only; never approves a review')
    args = parser.parse_args()
    try:
        if args.init_plan:
            if args.manifest:
                parser.error('Use --init-plan separately from validation')
            args.init_plan.parent.mkdir(parents=True, exist_ok=True)
            with args.init_plan.open('x') as f:
                f.write(json.dumps(make_plan(), indent=2)+'\n')
            print('Created planning metadata only: zero captured frames, zero verified ground truth.')
            return
        if args.manifest is None:
            parser.error('Provide a manifest or --init-plan')
        manifest = read(args.manifest)
        if args.print_split_lock:
            print(json.dumps({'mode': 'proposed_split_lock_not_validation', 'split_lock_sha256': split_digest(manifest)}))
            return
        if args.print_review_digests:
            sessions = {s['id']: s for s in manifest['sessions']}
            positions = {p['id']: p for p in manifest['positions']}
            values = {frame['id']: {kind: review_digest(frame, sessions[frame['session_id']],
                                                      positions[sessions[frame['session_id']]['position_id']], kind)
                                    for kind in frame['reviews']} for frame in manifest['frames']}
            sequence_values = {s['id']: sequence_review_digest(s, positions[s['position_id']])
                               for s in manifest['sessions'] if s['archive'] is not None}
            print(json.dumps({'mode': 'proposed_annotation_digests_not_validation', 'digests': values,
                              'sequence_digests': sequence_values}, indent=2))
            return
        historical = read(ROOT/'data/metadata/positions.json')
        baseline = read(ROOT/'configs/baselines/a1_20261018.json')
        inventory = ROOT/baseline['dataset']['inventory_path']
        hashes = None
        if inventory.exists():
            if hashlib.sha256(inventory.read_bytes()).hexdigest() != baseline['dataset']['inventory_sha256']:
                raise ValueError('Historical image inventory hash changed')
            hashes = set(read(inventory).values())
        report = validate_capture_manifest(manifest, args.asset_root, historical, hashes,
                                           purpose=args.purpose, require_complete=args.require_complete,
                                           release_final=args.release_final)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open('x') as f:
                f.write(json.dumps(report, indent=2)+'\n')
        print(json.dumps({k: v for k, v in report.items() if k not in ('missing_matrix_cells', 'review_digests')}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, BadZipFile) as exc:
        parser.exit(1, f'Capture validation failed: {exc}\n')


if __name__ == '__main__':
    main()
