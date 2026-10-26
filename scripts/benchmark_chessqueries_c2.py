"""Bounded C.2 CPU smoke test and matched D.2 zero-shot benchmark."""
import argparse
from collections import Counter, defaultdict
import gzip
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.vision.chessqueries_offline import ChessQueriesCPU, memory_info, sha256
from src.scanning.square_evidence import (SQUARES, classification_metrics, validate_bundle,
                                          verify_reviews)
from scripts.characterize_square_evidence import evaluation_truths


def read(path):
    return json.loads(Path(path).read_text())


def write_fresh(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError('Refuse to overwrite: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked(path, expected):
    if sha256(path) != expected:
        raise ValueError('Registered artifact changed: ' + str(path))


def configuration(path):
    c = read(path)
    if not c['development_only'] or c['final_holdout'] or c['training'] or c['fusion'] or c['device'] != 'cpu' or c['batch_size'] != 1:
        raise ValueError('C.2 offline CPU-only configuration required')
    return c


def smoke(config_path, output):
    if Path(output).exists():
        raise ValueError('Fresh smoke-test output required')
    config = configuration(config_path)
    manifest = read(config['evidence_manifest'])
    frame = manifest['frames'][0]
    metadata = frame['d1_metadata']
    checked(metadata['path'], metadata['derivative_sha256'])
    before = memory_info()
    limit = min(config['smoke_max_rss_bytes'], int(before['system_available_bytes'] * config['smoke_max_fraction_initial_available_ram']))
    record = dict(schema_version='chessqueries_c2_smoke_v1', config_sha256=sha256(config_path),
                  release_sha256=sha256(config['release']), frame_id=frame['frame_id'],
                  frame_sha256=metadata['derivative_sha256'], before=before, peak_rss_limit_bytes=limit,
                  frame_selection='first registered D2 frame; no labels or predictions consulted',
                  cpu_threads=config['cpu_threads'], batch_size=1)
    start = time.perf_counter()
    try:
        adapter = ChessQueriesCPU(config['release'], threads=config['cpu_threads'])
        record.update(model_load_seconds=adapter.model_load_seconds, model_parameter_count=adapter.parameter_count,
                      metadata=adapter.metadata, after_loading=memory_info())
        if memory_info()['peak_process_rss_bytes'] > limit:
            raise ValueError('Model loading exceeds conservative RSS budget')
        runs = []
        for _ in range(config['smoke_repeat_count']):
            prediction = adapter.predict(metadata['path'])
            if prediction['forward_seconds'] > config['smoke_max_forward_seconds']:
                raise ValueError('Smoke forward latency exceeds offline practicality bound')
            if memory_info()['peak_process_rss_bytes'] > limit:
                raise ValueError('Smoke inference exceeds conservative RSS budget')
            runs.append(prediction)
        if any(r['raw_logits'] != runs[0]['raw_logits'] or r['input_tensor_sha256'] != runs[0]['input_tensor_sha256'] for r in runs[1:]):
            raise ValueError('Repeated CPU predictions are not deterministic')
        record.update(status='passed', after=memory_info(),
                      repeated_logits_exact=True, output_shape=runs[0]['output_shape'],
                      class_names=runs[0]['class_names'], canonical_squares=runs[0]['canonical_squares'],
                      runs=[{k:v for k,v in r.items() if k not in ('raw_logits','probabilities','predicted_class_ids','class_names','canonical_squares')} for r in runs],
                      representative_prediction=runs[0], elapsed_seconds=time.perf_counter()-start)
    except Exception as error:
        record.update(status='blocked', error_type=type(error).__name__, error=str(error), after=memory_info())
        write_fresh(output, record)
        raise
    write_fresh(output, record)
    print(json.dumps({k:record[k] for k in ('status','model_load_seconds','model_parameter_count','output_shape','after','elapsed_seconds')}, indent=2))
    print('Forward seconds:', [r['forward_seconds'] for r in record['runs']], flush=True)


def load_cohort(config):
    manifest = read(config['evidence_manifest'])
    for key in ('observations', 'packet', 'reviews', 'config'):
        checked(manifest[key+'_path'], manifest[key+'_sha256'])
    observations = [json.loads(line) for line in Path(manifest['observations_path']).read_text().splitlines()]
    validate_bundle(manifest, observations)
    packet = read(manifest['packet_path'])
    verify_reviews(packet, read(manifest['reviews_path']), manifest['packet_sha256'])
    catalog_path = read(packet['config_path'])['source_catalog']
    checked(catalog_path, packet['protected_sha256'][catalog_path])
    truths, fens = evaluation_truths(read(catalog_path), sha256(catalog_path), read(config['ground_truth_corrections']))
    by_frame = defaultdict(dict)
    for observation in observations:
        by_frame[(observation['session_id'], observation['frame_id'])][observation['canonical_square']] = observation
    frames = []
    for f in manifest['frames']:
        if f['geometry_status'] != 'human_verified' or not f['observations']:
            raise ValueError('Requested comparison contains unverified or unusable geometry')
        rows = by_frame[(f['session_id'], f['frame_id'])]
        if set(rows) != set(SQUARES):
            raise ValueError('Incomplete or duplicate square frame')
        ordered = [rows[s] for s in SQUARES]
        first = ordered[0]
        checked(first['frame_path'], first['frame_sha256'])
        checked(first['geometry_reference'], first['geometry_reference_sha256'])
        if any(o['semantic_orientation'] != first['semantic_orientation'] or o['frame_sha256'] != first['frame_sha256'] for o in ordered):
            raise ValueError('Inconsistent frame orientation or image binding')
        frames.append(dict(session_id=f['session_id'], position_id=f['position_id'], frame_id=f['frame_id'],
                           timestamp_ms=f['timestamp_ms'], frame_path=first['frame_path'], frame_sha256=first['frame_sha256'],
                           semantic_orientation=first['semantic_orientation'], review_id=f['review_id'],
                           geometry_status=f['geometry_status'], conditions=f['conditions'],
                           source_video_sha256=first['source_video_sha256'], original_pts=first['original_pts'],
                           time_base=first['time_base'], geometry_reference=first['geometry_reference'],
                           geometry_reference_sha256=first['geometry_reference_sha256'],
                           d1_view_metadata=first['d1_raw_quality_and_view'],
                           baseline_class_ids=[o['predicted_class_id'] for o in ordered],
                           baseline_observation_ids=[o['observation_id'] for o in ordered],
                           truth=truths[f['session_id']]))
    if len(frames) != 55 or len({f['session_id'] for f in frames}) != 5:
        raise ValueError('Registered C.2 comparison requires exactly the 55-frame five-session D2 cohort')
    return manifest, frames, fens


def environment():
    names = ['torch','torchvision','numpy','Pillow','timm','safetensors','huggingface-hub','requests','tqdm']
    packages = {}
    for name in names:
        m = importlib.metadata.metadata(name)
        packages[name] = dict(version=m['Version'], license=m.get('License-Expression') or m.get('License'))
    cpu = next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), 'unknown')
    return dict(python=sys.version, platform=platform.platform(), cpu=cpu, packages=packages,
                device='cpu', cuda_used=False, memory=memory_info())


def benchmark(config_path, smoke_path, output):
    output = Path(output)
    if output.exists():
        raise ValueError('Fresh benchmark output directory required')
    config = configuration(config_path)
    passed = read(smoke_path)
    if passed['status'] != 'passed' or passed['config_sha256'] != sha256(config_path) or passed['release_sha256'] != sha256(config['release']):
        raise ValueError('Successful matching smoke test required')
    manifest, frames, fens = load_cohort(config)
    if passed['frame_id'] != frames[0]['frame_id'] or passed['frame_sha256'] != frames[0]['frame_sha256']:
        raise ValueError('Smoke input does not match registered benchmark')
    output.mkdir(parents=True)
    source_paths = ['src/vision/chessqueries_offline.py', 'scripts/benchmark_chessqueries_c2.py',
                    str(config_path), config['evidence_manifest'], config['ground_truth_corrections']]
    write_fresh(output/'registration.json', dict(schema_version='chessqueries_c2_registration_v1',
        frame_ids=[f['frame_id'] for f in frames], frame_count=55, session_count=5,
        label_version='owner_corrected_c7_supplement_v1', evaluated_fens=fens,
        observations_sha256=manifest['observations_sha256'],
        input_sha256={p:sha256(p) for p in source_paths},
        inference_policy=config['orientation_policy'], config=config,
        smoke_path=str(smoke_path), smoke_sha256=sha256(smoke_path),
        inference_has_ground_truth_argument=False, frozen_before_model_outputs=True))
    start = time.perf_counter()
    adapter = ChessQueriesCPU(config['release'], threads=config['cpu_threads'])
    predictions = []
    for i, frame in enumerate(frames):
        prediction = adapter.predict(frame['frame_path'])
        predictions.append(dict(frame_id=frame['frame_id'], session_id=frame['session_id'],
                                frame_sha256=frame['frame_sha256'], **prediction))
        if memory_info()['peak_process_rss_bytes'] > passed['peak_rss_limit_bytes']:
            raise ValueError('Benchmark exceeded tested memory budget; do not continue')
        print(f'{i+1}/55 {frame["frame_id"]} {prediction["forward_seconds"]:.3f}s', flush=True)
    record = dict(schema_version='chessqueries_c2_predictions_v1',
                  model_load_seconds=adapter.model_load_seconds, elapsed_seconds=time.perf_counter()-start,
                  model_parameter_count=adapter.parameter_count, environment=environment(),
                  release_sha256=sha256(config['release']), config_sha256=sha256(config_path),
                  predictions=predictions)
    (output/'predictions.json.gz').write_bytes(gzip.compress(json.dumps(record, allow_nan=False).encode(), mtime=0))
    print('Complete:', output/'predictions.json.gz', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['smoke','benchmark'])
    parser.add_argument('--config', type=Path, default=Path('configs/experiments/chessqueries_c2.json'))
    parser.add_argument('--smoke', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'smoke':
        smoke(args.config, args.output)
    else:
        if args.smoke is None:
            parser.error('--smoke is required for the benchmark')
        benchmark(args.config, args.smoke, args.output)
