"""Integrity gates for native pilot PTS/derivatives; never approves human ground truth."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import hashlib

from src.scanning.native_pilot import pts_record, sha256_file


def validate_session(session, *, check_files=True):
    if session['schema_version'] != 'native_pilot_session_v1' or session['split'] != 'pilot' or session['designation'] != 'development_only':
        raise ValueError('Native development-only pilot session required')
    frames = session['original_frame_index']
    if len(frames) < 2 or len(frames) != session['decoded_frames']:raise ValueError('Incomplete native frame index')
    first = Fraction(*frames[0]['pts_time'])
    previous, ids = None, {}
    for number, row in enumerate(frames):
        if row['frame_id'] in ids or row['decode_index'] != number:raise ValueError('Duplicate/reordered frame identity')
        exact = pts_record(row['pts'], Fraction(*row['time_base']), first)
        if any(row[key] != value for key, value in exact.items()):raise ValueError('PTS/time-base/timestamp disagreement')
        value = Fraction(*row['pts_time'])
        if previous is not None and value <= previous:raise ValueError('Nonmonotonic presentation timeline')
        if row['display_rotation_degrees'] != 0:raise ValueError('Unsupported display-coordinate transform')
        previous = value;ids[row['frame_id']] = row
    if len(session['candidates']) > 12:raise ValueError('Too many recognition candidates')
    chosen = []
    for row in session['candidates']:
        if row.get('training_eligible') is not False or row['split'] != 'pilot':raise ValueError('Candidate promoted to training')
        source = ids.get(row['frame_id'])
        if source is None or any(row[k] != source[k] for k in ['pts', 'time_base', 'pts_time', 'timestamp_ms', 'relative_timestamp_ms', 'width', 'height']):
            raise ValueError('Candidate not bound to original timestamp/dimensions')
        if row['source_video_sha256'] != session['source_video']['sha256']:raise ValueError('Wrong parent video')
        time = Fraction(*row['pts_time'])
        if any(abs(time-other) < 2 for other in chosen):raise ValueError('Near-adjacent recognition candidates')
        chosen.append(time)
        if check_files:
            if sha256_file(row['path']) != row['derivative_sha256']:raise ValueError('Changed derivative')
            image = cv2.imread(row['path'])
            if image is None or list(image.shape[:2]) != [row['height'], row['width']]:raise ValueError('Bad derivative dimensions')
            if hashlib.sha256(image.tobytes()).hexdigest() != row['decoded_pixel_sha256']:raise ValueError('PNG changed decoded pixels')
    if check_files and sha256_file(session['source_video']['path']) != session['source_video']['sha256']:
        raise ValueError('Original video changed')
    return dict(decoded_frames=len(frames), sparse_candidates=len(chosen), native_pts_integrity=True,
                ground_truth_verified=False, a3_human_reviews_complete=False)


def validate(manifest):
    if manifest['schema_version'] != 'native_pilot_dataset_v1' or manifest['designation'] != 'development_only' or manifest['training_eligible'] is not False or manifest['final_holdout'] is not False:
        raise ValueError('Pilot-only manifest required')
    sessions, groups, results = set(), {}, []
    for row in manifest['sessions']:
        if row['capture_session_id'] in sessions:raise ValueError('Duplicate recording/session ID')
        sessions.add(row['capture_session_id'])
        for key in [('position', row['position_id']), ('placement', row['placement_fen']), ('correlation', row['independence_group_id'])]:
            if groups.setdefault(key, row['split']) != row['split']:raise ValueError('Cross-split position/recording leakage')
        if row['split'] != 'pilot' or row['training_eligible']:raise ValueError('Session not development-only')
        indexed = json.loads(Path(row['index_path']).read_text())
        if indexed['session_id'] != row['capture_session_id'] or indexed['candidates'] != row['candidates']:
            raise ValueError('Session/derivative binding mismatch')
        reference = row['reference']
        if sha256_file(reference['path']) != reference['sha256']:raise ValueError('Reference photo changed')
        results.append(dict(session_id=row['capture_session_id'], **validate_session(indexed)))
    return dict(native_media_integrity_passed=True, sessions=results, original_order_preserved=True,
                training_eligible=False, final_holdout=False, ground_truth_verified=False,
                capture_protocol_v2_accepted=False,
                limitations=['Native extension validated; existing guided-ZIP schema/replay is not claimed to accept MOV.',
                             'Image-reference, unchanged-position, corners and semantic-orientation attestations remain pending.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():parser.error('New report path required')
    report = validate(json.loads(args.manifest.read_text()))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':main()
