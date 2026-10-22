"""Inspect staged pilot IDs/FENs and media presence; does not validate native video."""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess

ROOT = Path(__file__).resolve().parents[1]


def inspect(manifest, root=ROOT):
    if (manifest.get('schema_version') != 'a3_native_pilot_intake_v1' or
            manifest.get('designation') != 'development_only' or
            manifest.get('training_eligible') is not False or manifest.get('final_holdout') is not False):
        raise ValueError('Development-only staging registry required')
    positions, sessions, placements, records = set(), set(), set(), []
    for row in manifest['records']:
        for field, seen in [('position_id', positions), ('capture_session_id', sessions)]:
            if row[field] in seen:raise ValueError('Duplicate '+field)
            seen.add(row[field])
        if row['split'] != 'pilot':raise ValueError('Pilot records must stay development-only')
        if len(row['fen'].split()) != 6:raise ValueError('Six-field user FEN required')
        board = chess.Board(row['fen'])
        if board.board_fen() != row['placement_fen']:raise ValueError('Placement/FEN mismatch')
        if board.board_fen() in placements:raise ValueError('Duplicate placement among pilot positions')
        placements.add(board.board_fen())
        media = {}
        for kind, extensions in [('video', {'.mp4', '.mov', '.m4v'}),
                                 ('reference', {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.dng', '.tif', '.tiff'})]:
            directory = (root/row[kind+'_directory']).resolve()
            if not directory.is_relative_to((root/'data/raw/a3_pilot_01').resolve()):
                raise ValueError('Media directory outside isolated pilot root')
            found = sorted(str(p.relative_to(root)) for p in directory.iterdir()
                           if p.is_file() and p.suffix.lower() in extensions) if directory.exists() else []
            media[kind] = dict(files=found, count=len(found), status='present_unvalidated' if len(found) == 1 else 'missing' if not found else 'ambiguous_multiple_files')
        records.append(dict(position_id=row['position_id'], capture_session_id=row['capture_session_id'],
                            fen_syntax_valid=True, standard_chess_valid=board.is_valid(),
                            pieces=len(board.piece_map()), historical_position_ids=row['historical_position_ids'],
                            historical_splits=row['historical_splits'], photographic_placement_review=row['photographic_placement_review'], media=media))
    ready = all(item['status'] == 'present_unvalidated' for r in records for item in r['media'].values())
    return dict(status='ready_for_media_validation' if ready else 'awaiting_or_ambiguous_media',
                records=records, missing_or_ambiguous_assets=sum(item['count'] != 1 for r in records for item in r['media'].values()),
                protocol_accepted=False, media_validation_performed=False, verified_media_frames=0,
                tools={name:shutil.which(name) for name in ['ffprobe', 'ffmpeg']},
                limitations=['Presence is not codec/PTS/quality validation or verified placement.',
                             'Existing capture_protocol_v2 validator/replay only accepts guided_capture_v1 ZIPs; native input needs a PTS-preserving adapter.',
                             'Null camera/light/pose metadata is unknown, never a measurement.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path, nargs='?', default=ROOT/'data/metadata/a3_pilot_01_intake.json')
    parser.add_argument('--report', type=Path, help='New output file, never overwrite')
    args = parser.parse_args()
    try:
        report = inspect(json.loads(args.manifest.read_text()))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            with args.report.open('x') as f:f.write(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'ready_for_media_validation' else 2
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f'Pilot intake inspection failed: {error}\n')


if __name__ == '__main__':sys.exit(main())
