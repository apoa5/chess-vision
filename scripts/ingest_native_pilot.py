"""Bind pilot originals/FENs and index native PTS; no model or verified-label writes."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from scripts.inspect_pilot_intake import inspect
from src.scanning.native_pilot import decoder, inspect_video, sha256_file


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def contact_sheet(images, output, columns=4, thumb_width=232):
    rows = (len(images)+columns-1)//columns
    canvas = np.full((rows*460, columns*thumb_width, 3), 245, np.uint8)
    for index, (path, caption) in enumerate(images):
        image = cv2.imread(str(path))
        if image is None:raise ValueError('Unreadable contact-sheet image')
        scale = min((thumb_width-8)/image.shape[1], 412/image.shape[0])
        resized = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)), interpolation=cv2.INTER_AREA)
        x, y = (index%columns)*thumb_width, (index//columns)*460
        canvas[y+32:y+32+resized.shape[0], x+4:x+4+resized.shape[1]] = resized
        cv2.putText(canvas, caption, (x+4, y+21), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
    if not cv2.imwrite(str(output), canvas):raise OSError('Failed to write contact sheet')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--intake', type=Path, default=Path('data/metadata/a3_pilot_01_intake.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():parser.error('Refusing to overwrite existing ingestion run')
    registry = json.loads(args.intake.read_text())
    presence = inspect(registry)
    if presence['status'] != 'ready_for_media_validation':parser.error('Require exactly one video/reference per position')
    cv2.setNumThreads(1)
    protected_paths = [Path('config/settings.yaml'), Path('configs/capture/a3_plan.json'), Path('configs/capture/capture_manifest.schema.json'),
        Path('models/checkpoints/piece_classifier_best.pt'), Path('models/checkpoints/board_marking_reader.pt'),
        Path('outputs/evaluation/crop_b2_20261018_run1/training/context1_224/best.pt'),
        *Path('src/vision').glob('*.py'), *Path('src/scanning').glob('*.py')]
    protected = {str(p):sha256_file(p) for p in protected_paths if p.exists()}
    media = {p:sha256_file(p) for r in presence['records'] for kind in ('video', 'reference') for p in r['media'][kind]['files']}
    sources = {str(p):sha256_file(p) for p in [Path(__file__), Path('src/scanning/native_pilot.py'), args.intake, Path('requirements-pilot-video.txt')]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir/'input_intake.json', registry)
    av = decoder()
    manifest = dict(schema_version='native_pilot_dataset_v1', dataset_id=registry['dataset_id'], designation='development_only',
                    split='pilot', final_holdout=False, training_eligible=False, media_sha256=media, protected_sha256=protected,
                    source_sha256=sources, decoder=dict(name='PyAV', version=av.__version__, libraries=av.library_versions),
                    sessions=[], pending_reviews=['Physical placement agreement, unchanged sequence, geometry and semantic orientation require separate review.'],
                    a3_compatibility='Preserves A.3 review/split/selection rules; native PTS index is an explicit new format, not a guided-recorder ZIP or automatic v2 acceptance.')
    write(args.output_dir/'manifest.json', manifest)
    references = []
    for row, available in zip(registry['records'], presence['records']):
        source = available['media']['video']['files'][0]
        reference = available['media']['reference']['files'][0]
        ref_image = cv2.imread(reference)
        if ref_image is None:raise ValueError('Unreadable reference: '+reference)
        result = inspect_video(source, args.output_dir/row['capture_session_id'], row['capture_session_id'])
        session = {key:row[key] for key in ['position_id', 'capture_session_id', 'position_category', 'fen', 'placement_fen',
                   'position_group_id', 'independence_group_id', 'historical_position_ids', 'historical_splits', 'fen_provenance']}
        session.update(split='pilot', training_eligible=False, reference=dict(path=reference, sha256=media[reference], width=ref_image.shape[1], height=ref_image.shape[0], photographic_placement_review='pending'),
                       source_video=result['source_video'], index_path=str(args.output_dir/row['capture_session_id']/'session.json'),
                       candidates=result['candidates'], decoded_frames=result['decoded_frames'], started_at=None,
                       camera_id=None, board_id=None, environment=None, lighting=None, unchanged_position_review='pending',
                       verified_sequence=False, view_intervals=[])
        manifest['sessions'].append(session)
        references.append((reference, row['position_category']))
        contact_sheet([(r['path'], f"{i+1:02d} {r['timestamp_ms']/1000:.3f}s") for i, r in enumerate(result['candidates'])],
                      args.output_dir/row['capture_session_id']/'candidates_contact.png')
        write(args.output_dir/'manifest.json', manifest)
        print(row['position_category'], 'decoded', result['decoded_frames'], 'selected', len(result['candidates']), flush=True)
    contact_sheet(references, args.output_dir/'references_contact.png', columns=5)
    for path, digest in (media|protected|sources).items():
        if sha256_file(path) != digest:raise ValueError('Input/source changed: '+path)
    write(args.output_dir/'preservation.json', dict(original_media_unchanged=True, protected_files_unchanged=True,
                                                   protected_file_count=len(protected), original_media_count=len(media)))
    print('Native pilot media indexed; no human reference/geometry/sequence review fabricated.', flush=True)


if __name__ == '__main__':main()
