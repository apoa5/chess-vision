"""D.1 native-video sampling, bounded geometry evaluation, and offline selection."""
import argparse
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
from scripts.evaluate_pilot_geometry import evaluate_frame
from scripts.validate_native_pilot import validate_session
from src.scanning.native_pilot import decoder, sha256_file
from src.scanning.pilot_comparison import display_image
from src.scanning.video_keyframes import temporal_candidates, validate_settings
from src.utils.config import load_config


def read(path):return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    if path.exists():raise ValueError('Refusing to overwrite: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2)+'\n')


def check_hashes(hashes):
    for path, digest in hashes.items():
        if sha256_file(path) != digest:raise ValueError('Protected input changed: '+path)


def init(args):
    if args.run.exists():raise ValueError('Fresh run directory required')
    catalog = read(args.catalog)
    if catalog['designation'] != 'development_only' or catalog['training_eligible'] or catalog['final_holdout']:
        raise ValueError('Only development pilot data permitted')
    for name in ['primary_manifest', 'comparison_manifest', 'owner_attestations']:
        if sha256_file(catalog[name]) != catalog[name+'_sha256']:raise ValueError('Catalog binding changed')
    native = read(catalog['primary_manifest'])
    settings = validate_settings(read(args.config))
    protected = dict(native['protected_sha256']) | dict(native['media_sha256'])
    paths = [args.catalog, Path(catalog['primary_manifest']), Path(catalog['owner_attestations'])]
    paths += [Path(s['index_path']) for s in native['sessions']]
    for dirname in ['outputs/scanning/pilot_01/hd_native_run2', 'outputs/scanning/pilot_01/hd_aligned_run1',
                    'outputs/scanning/pilot_01/hd_geometry_parts_run1', 'outputs/scanning/pilot_01/hd_geometry_final_run1',
                    'outputs/scanning/pilot_01/hd_comparison_run1', 'docs/evaluation', 'configs/baselines', 'configs/capture', 'web/src']:
        paths += [p for p in Path(dirname).rglob('*') if p.is_file()]
    protected.update({str(p):sha256_file(p) for p in paths})
    check_hashes(protected)
    # Short per-worker checks; complete frozen artifacts are rechecked at finalization.
    critical = {p:h for p,h in protected.items() if p.startswith(('src/', 'models/', 'config/'))}
    critical['scripts/evaluate_pilot_geometry.py'] = sha256_file('scripts/evaluate_pilot_geometry.py')
    sources = {p:sha256_file(p) for p in ['scripts/select_video_keyframes.py', 'src/scanning/video_keyframes.py', str(args.config)]}
    args.run.mkdir(parents=True, exist_ok=False)
    write(args.run/'settings.json', settings)
    manifest = dict(schema_version='offline_video_keyframe_candidates_d1_v1', designation='development_only',
                    training_eligible=False, final_holdout=False, source_catalog=str(args.catalog),
                    source_catalog_sha256=sha256_file(args.catalog), primary_manifest=catalog['primary_manifest'],
                    source_sha256=sources, protected_sha256=protected, critical_sha256=critical,
                    geometry_config_sha256=sha256_file('config/settings.yaml'), sessions=[])
    # Save header so interrupted preparation can be inspected; never silently reused.
    write(args.run/'initialization.json', {k:v for k,v in manifest.items() if k!='sessions'})
    av = decoder(); cv2.setNumThreads(1)
    manifest['decoder'] = dict(version=av.__version__, libraries=av.library_versions)
    for s, registered in zip(native['sessions'], catalog['sessions']):
        start = time.perf_counter()
        if any(s[k] != registered[k] for k in ['capture_session_id', 'position_id', 'position_group_id', 'independence_group_id']):
            raise ValueError('Session grouping mismatch')
        source = s['source_video']
        if source['sha256'] != registered['primary_source']['sha256']:raise ValueError('Wrong source representation')
        if sha256_file(source['path']) != source['sha256']:raise ValueError('Source changed')
        index = read(s['index_path']); validate_session(index, check_files=False)
        chosen = temporal_candidates(index['original_frame_index'], settings['interval_ms'], settings['candidate_limit'])
        dest = args.run/'frames'/s['capture_session_id']; dest.mkdir(parents=True, exist_ok=False)
        rows = []
        with av.open(source['path']) as container:
            stream = container.streams.video[0]; stream.thread_type='SLICE'; stream.codec_context.thread_count=1
            for row in chosen:
                container.seek(row['pts'], stream=stream, backward=True)
                image = None
                for f in container.decode(stream):
                    t = Fraction(f.pts)*f.time_base
                    if t > Fraction(*row['pts_time']):break
                    if t == Fraction(*row['pts_time']):
                        image = display_image(f.to_ndarray(format='bgr24'), int(f.rotation)); break
                if image is None or [image.shape[1],image.shape[0]] != [row['width'],row['height']]:
                    raise ValueError('Indexed PTS/display dimensions could not be reproduced')
                path = dest/(row['frame_id']+'.png')
                if not cv2.imwrite(str(path),image):raise OSError('PNG write failed')
                rows.append(dict(row, session_id=s['capture_session_id'], path=str(path),
                                 source_video_path=source['path'], source_video_sha256=source['sha256'],
                                 derivative_sha256=sha256_file(path), decoded_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                                 source_representation='high_quality_primary',
                                 encoded_to_display_matrix=source['encoded_to_display_matrix'],
                                 split='pilot', training_eligible=False))
        session = {k:s[k] for k in ['capture_session_id','position_id','position_group_id','independence_group_id']}
        session.update(source_video=source, candidates=rows, original_index_path=s['index_path'],
                       original_index_sha256=sha256_file(s['index_path']),
                       duration_seconds=source['container_duration_seconds'], extraction_seconds=time.perf_counter()-start)
        manifest['sessions'].append(session)
        write(dest/'candidates.json', session)
        print(s['capture_session_id'],len(rows),'candidates',flush=True)
    check_hashes(native['media_sha256']); check_hashes(critical)
    write(args.run/'candidates.json', manifest)
    print('Temporal candidates frozen, no FEN passed to selection.',flush=True)


def evaluate(args):
    manifest = read(args.run/'candidates.json'); check_hashes(manifest['critical_sha256'])
    if sha256_file('config/settings.yaml') != manifest['geometry_config_sha256']:raise ValueError('Geometry configuration changed')
    cv2.setNumThreads(1); config = load_config(); done = 0
    for session in manifest['sessions']:
        for frame in session['candidates']:
            parent = args.run/'geometry'/frame['session_id']/frame['frame_id']
            record_path = parent/'pair.json'
            binding = {k:frame[k] for k in ['session_id','frame_id','pts','time_base','source_video_sha256','derivative_sha256']}
            if record_path.exists():
                old = read(record_path)
                if old['binding'] != binding or old['geometry_config_sha256'] != manifest['geometry_config_sha256']:
                    raise ValueError('Resume cache does not match exact frame/configuration')
                continue
            if done >= args.limit:break
            if sha256_file(frame['path']) != frame['derivative_sha256']:raise ValueError('Candidate changed')
            parent.mkdir(parents=True,exist_ok=True)
            attempt = 1
            while (parent/f'attempt_{attempt:02d}').exists():attempt += 1
            destination = parent/f'attempt_{attempt:02d}'
            policies = {}
            for policy in ['legacy','marked-board']:
                policies[policy] = evaluate_frame(frame['path'],config,destination/policy,policy)
            if policies['legacy']['completed'] and (not policies['marked-board']['completed'] or
                    policies['legacy']['source_square_polygons'] != policies['marked-board']['source_square_polygons']):
                raise ValueError('Legacy-success parity regressed')
            write(record_path,dict(binding=binding,geometry_config_sha256=manifest['geometry_config_sha256'],
                                   evaluator_sha256=manifest['critical_sha256']['scripts/evaluate_pilot_geometry.py'],
                                   geometry_state_review='unverified',policies=policies))
            done += 1
            print(frame['session_id'],frame['relative_timestamp_ms'],[(k,v['completed'],v['failure_stage']) for k,v in policies.items()],flush=True)
        if done >= args.limit:break
    check_hashes(manifest['critical_sha256'])
    total = sum(len(s['candidates']) for s in manifest['sessions'])
    complete = sum(1 for s in manifest['sessions'] for f in s['candidates']
                   if (args.run/'geometry'/f['session_id']/f['frame_id']/'pair.json').exists())
    print('PAIRS',complete,'/',total,flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('candidates');p.add_argument('--catalog',type=Path,default=Path('data/metadata/a3_pilot_01_sources_v2.json'));p.add_argument('--config',type=Path,default=Path('configs/experiments/keyframes_d1.json'));p.add_argument('--run',type=Path,required=True)
    p = sub.add_parser('geometry');p.add_argument('--run',type=Path,required=True);p.add_argument('--limit',type=int,default=8)
    args = parser.parse_args()
    if args.command=='candidates':init(args)
    else:
        if not 1 <= args.limit <= 32:parser.error('Bounded limit 1..32 required')
        evaluate(args)

if __name__=='__main__':main()
