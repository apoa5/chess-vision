"""B.3 CPU costs; run sequentially after training/tests, without other benchmarks."""
import argparse
import csv
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

from scripts.analyze_crop_experiments import preprocessing_latency
from scripts.run_augmentation_experiments import read, write, sha, load_model
from src.vision.augmentation_experiments import FAMILIES, augment_packed
from src.vision.crop_experiments import tensor_from_packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('configs/experiments/b3_augmentation.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True, help='Completed run for interleaved forward confirmation')
    args = parser.parse_args()
    if args.output.exists():parser.error('Refusing to overwrite benchmark')
    cv2.setNumThreads(1); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    plan = read(args.plan)
    all_rows = read('data/classification/crops.json')['results']
    test_rows = [r for r in all_rows if r['split'] == 'test']
    cache = np.load(Path(plan['cache_root'])/'train.npy', mmap_mode='r')
    # RAM copies remove disk/page faults from this transform-only timing.
    samples = np.array(cache[:128], copy=True)
    results = {}
    for family in FAMILIES:
        times, applied, rejected = [], 0, 0
        for repeat in range(4):
            start = time.perf_counter()
            for index, packed in enumerate(samples):
                rng = np.random.default_rng(index+1000*repeat)
                if family in ('none', 'legacy_visual'):
                    transformed, trace = packed, dict(applied=family == 'legacy_visual')
                else:
                    transformed, trace = augment_packed(packed, family, rng)
                tensor_from_packed(transformed, rng=rng if family == 'legacy_visual' else None)
                if repeat:
                    applied += family == 'legacy_visual' or trace['applied']
                    rejected += trace.get('rejected_guard', False)
            if repeat:times.append((time.perf_counter()-start)*1000/len(samples))
        results[family] = dict(ms_per_crop=times, median_ms_per_crop=float(np.median(times)),
                               applied_among_384=applied, guard_rejected_among_384=rejected)
    run_results = read(args.run_dir/'results.json')['results']
    models = {name: load_model(torch.load(item['checkpoint_path'], map_location='cpu', weights_only=True)).eval()
              for name, item in run_results.items()}
    test_cache = np.load(Path(plan['cache_root'])/'test.npy', mmap_mode='r')
    tensors = torch.stack([torch.from_numpy(tensor_from_packed(test_cache[i])) for i in range(64)])
    timings = {name: [] for name in models}
    orders = []
    with torch.inference_mode():
        for model in models.values():
            for _ in range(2):
                for i in range(0, 64, 16):model(tensors[i:i+16])
        for repeat in range(8):
            order = np.random.default_rng(900+repeat).permutation(list(models)).tolist()
            orders.append(order)
            for name in order:
                start = time.perf_counter()
                for i in range(0, 64, 16):models[name](tensors[i:i+16])
                timings[name].append((time.perf_counter()-start)*1000)
    confirmation = {name: dict(board_ms=values, median_board_ms=float(np.median(values)), q25_q75_ms=np.percentile(values, [25, 75]).tolist())
                    for name, values in timings.items()}
    del models
    result = dict(training_transform_cost=results, interleaved_forward=confirmation, interleaved_orders=orders,
                  interleaved_scope='All 13 loaded models, identical 64 clean tensors in RAM, CPU4 batch16, two warmup boards/model, eight measured rounds with fixed randomized model order. No disk/transform/geometry timing. Initial raw per-model times are retained separately.',
                  inference_preprocessing_plus_forward=preprocessing_latency(dict(checkpoint_path=plan['initialization'],
                    spec=dict(name='context1_224', input_size=224, padding_squares=1)), test_rows),
                  scope='Training transform-only loaded 128 crop inputs, one warmup/three measured passes; no backprop. Inference uses identical B.2 tighter crop for every family, 64 loaded source crops, two warmups/four repeats, batch16 CPU4; excludes disk, geometry and phone.',
                  source_sha256={str(Path(__file__)):sha(Path(__file__))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, result)
    with args.output.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model', 'initial_forward_ms', 'confirmed_forward_ms', 'q25_ms', 'q75_ms'])
        writer.writeheader()
        for name, item in confirmation.items():
            writer.writerow(dict(model=name, initial_forward_ms=run_results[name]['latency']['median_model_only_board_ms'],
                                 confirmed_forward_ms=item['median_board_ms'], q25_ms=item['q25_q75_ms'][0], q75_ms=item['q25_q75_ms'][1]))


if __name__ == '__main__':main()
