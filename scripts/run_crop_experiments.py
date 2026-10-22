"""B.2 fixed-geometry crop/resolution experiments; never writes production assets."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.chess.labels import CLASS_NAMES, LABEL_VERSION
from src.utils.config import load_config
from src.utils.image_io import load_image
from src.vision.classifier import create_classifier
from src.vision.classifier_evaluation import prediction_metrics
from src.vision.crop_experiments import CropSpec, CachedCropDataset, packed_input, tensor_from_packed, transform_crop
from src.vision.dataset import file_hash, review_approved
from src.vision.dataset_loader import preprocess_context_crop
from src.vision.training import run_epoch
from scripts.evaluate_classifier import confusion_plot


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(path)


def verify(hashes):
    for name, expected in hashes.items():
        if file_hash(Path(name)) != expected:
            raise ValueError('Protected input changed: '+name)


def frozen_inputs():
    b1b = read('docs/evaluation/geometry_reliability_results_b1b_20261018.json')
    assert file_hash(Path(b1b['full_report_path'])) == b1b['full_report_sha256']
    hashes = dict(read(b1b['full_report_path'])['protected_sha256'])
    hashes.update(b1b['source_sha256'])
    hashes.update(b1b['test_source_sha256'])
    hashes.update(b1b['diagnostic_sha256'])
    for path in [*Path(b1b['full_report_path']).parent.rglob('*'),
                 Path('docs/evaluation/geometry_reliability_results_b1b_20261018.json'),
                 Path('docs/evaluation/geometry_reliability_b1b.md')]:
        if path.is_file():
            hashes[str(path)] = file_hash(path)
    verify(hashes)
    return hashes


def leakage_audit(rows, prepared, images, positions):
    assignment, identifiers = {}, set()
    unknown = set()
    for row in rows:
        if row['crop_id'] in identifiers:
            raise ValueError('Duplicate crop ID')
        identifiers.add(row['crop_id'])
        if row.get('geometry_review_required') or row.get('quality_flags'):
            raise ValueError('Unapproved target')
        source, position = row['source_image'], row['position_id']
        keys = [('position', position), ('source', source),
                ('source_bytes', prepared[source]['source_sha256']),
                ('placement', positions[position]['fen'].split()[0])]
        session = images[source].get('capture_session_id') or images[source].get('session_id')
        if session:
            keys.append(('session', session))
        else:
            unknown.add(source)
        for key in keys:
            if assignment.setdefault(key, row['split']) != row['split']:
                raise ValueError('Split leakage: '+str(key))
    return dict(crops=len(rows), positions=len({r['position_id'] for r in rows}),
                sources=len({r['source_image'] for r in rows}), unknown_session_sources=len(unknown),
                checks='Position, source image, identical source bytes, identical placement, known sessions; derivatives inherit split.',
                limitation='Historical session IDs/timestamps are unavailable; independent-session or cross-position near-duplicate leakage cannot be certified. No new frames/splits are introduced.')


def spec_for(variant):
    return CropSpec(**{k: v for k, v in variant.items() if k != 'train'})


def cache_path(root, spec, split):
    return root/'cache'/spec.name/(split+'.npy')


def make_cache(root, spec, rows, split):
    path = cache_path(root, spec, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('Refusing to overwrite cache: '+str(path))
    array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                     shape=(len(rows), 4, spec.input_size, spec.input_size))
    dimensions, target_widths, parity = [], [], 0
    for i, row in enumerate(rows):
        if file_hash(Path(row['crop_path'])) != row['crop_sha256']:
            raise ValueError('Changed crop: '+row['crop_id'])
        image = load_image(row['crop_path'])
        packed = packed_input(image, row['target_polygon'], spec)
        array[i] = packed
        transformed, polygon = transform_crop(image, row['target_polygon'], spec)
        dimensions.append(transformed.shape[0])
        target_widths.append(float(np.sqrt(packed[3].sum())))
        if i % 67 == 0:
            np.testing.assert_array_equal(tensor_from_packed(packed), preprocess_context_crop(transformed, polygon, spec.input_size))
            parity += 1
        if i % 1024 == 0:
            print('cache', spec.name, split, i, '/', len(rows), flush=True)
    array.flush()
    del array
    return dict(rows=len(rows), shape=[len(rows), 4, spec.input_size, spec.input_size],
                sha256=file_hash(path), file_bytes=path.stat().st_size, tensor_parity_checks=parity,
                crop_width_min_median_max=np.percentile(dimensions, [0, 50, 100]).tolist(),
                effective_mask_width_min_median_max=np.percentile(target_widths, [0, 50, 100]).tolist())


def dataset(root, spec, rows, split, augmentation, seed=42):
    cache = np.load(cache_path(root, spec, split), mmap_mode='r')
    return CachedCropDataset(rows, cache, split=split, seed=seed, augmentation=augmentation)


def load_model(checkpoint):
    model = create_classifier(pretrained=False, mode='fine_tune')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def train_arm(root, spec, rows, settings, config, initial, provenance):
    dest = root/'training'/spec.name
    dest.mkdir(parents=True, exist_ok=False)
    seed = settings['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = load_model(initial)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    train = dataset(root, spec, rows['train'], 'train', config['classifier']['augmentation'], seed=seed)
    val = dataset(root, spec, rows['val'], 'val', {})
    loader = DataLoader(train, batch_size=settings['batch_size'], shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed))
    validation = DataLoader(val, batch_size=settings['batch_size'], shuffle=False, num_workers=0)
    weights = torch.tensor([provenance['class_weights'][name] for name in CLASS_NAMES], dtype=torch.float32)
    history = []
    best = float('inf')
    checkpoint_path = dest/'best.pt'
    report = dict(status='running', spec=spec.metadata(), settings=settings, initialization_sha256=provenance['checkpoint_sha256'],
                  manifest_sha256=provenance['manifest_sha256'], test_used_for_training_or_selection=False, history=history)
    start = time.perf_counter()
    for epoch in range(settings['epochs']+1):
        verify(provenance['source_sha256'])
        train_metrics = None
        if epoch:
            train.set_epoch(epoch-1)
            print('train', spec.name, 'epoch', epoch, '/', settings['epochs'], flush=True)
            def progress(step, total):
                report.update(epoch=epoch, step=step, total_steps=total, elapsed_seconds=time.perf_counter()-start)
                write(dest/'status.json', report)
                print(spec.name, epoch, step, '/', total, flush=True)
            train_metrics = run_epoch(model, loader, weights, optimizer=optimizer, mode='fine_tune', progress=progress)
        metrics = run_epoch(model, validation, weights)
        history.append(dict(epoch=epoch, train=train_metrics, validation=metrics, elapsed_seconds=time.perf_counter()-start))
        if metrics['loss'] < best:
            best = metrics['loss']
            state = dict(model_state_dict=model.state_dict(), epoch=epoch, origin_epoch=initial['epoch'], mode='fine_tune',
                         validation=metrics, architecture='mobilenet_v3_small', input_channels=4, input_size=spec.input_size,
                         label_version=LABEL_VERSION, class_names=list(CLASS_NAMES), dataset_fingerprint=initial['dataset_fingerprint'],
                         manifest_sha256=provenance['manifest_sha256'], settings=settings, experiment_only=True,
                         crop_spec=spec.metadata(), initialization_sha256=provenance['checkpoint_sha256'])
            temporary = checkpoint_path.with_suffix('.tmp')
            torch.save(state, temporary); temporary.replace(checkpoint_path)
            report.update(best_epoch=epoch, best_validation_loss=best)
        report.update(elapsed_seconds=time.perf_counter()-start)
        write(dest/'status.json', report)
        print('validation', spec.name, epoch, 'loss', metrics['loss'], 'accuracy', metrics['accuracy'], flush=True)
    report.update(status='complete', checkpoint_path=str(checkpoint_path), checkpoint_sha256=file_hash(checkpoint_path))
    write(dest/'status.json', report)
    del model, optimizer, train, val, loader, validation
    gc.collect()


def metrics_by_board(rows, probabilities, images):
    labels = np.array([r['class_id'] for r in rows])
    result = prediction_metrics(labels, probabilities)
    predicted = probabilities.argmax(axis=1)
    boards = sorted({r['source_image'] for r in rows})
    exact = []
    for name in boards:
        indices = [i for i, r in enumerate(rows) if r['source_image'] == name]
        if len(indices) != 64 or len({rows[i]['chess_square'] for i in indices}) != 64:
            raise ValueError('Incomplete board cannot enter exact-board evaluation')
        exact.append(dict(image_id=name, exact=bool(np.all(predicted[indices] == labels[indices])),
                          errors=int(np.sum(predicted[indices] != labels[indices]))))
    result.update(boards=len(boards), exact_boards=sum(r['exact'] for r in exact), per_board=exact)
    for grouping, key in [('orientation', lambda r: images[r['source_image']]['orientation']),
                          ('position', lambda r: r['position_id'])]:
        groups = {}
        for name in sorted({key(r) for r in rows}):
            indices = [i for i, row in enumerate(rows) if key(row) == name]
            group = prediction_metrics(labels[indices], probabilities[indices])
            source_ids = {rows[i]['source_image'] for i in indices}
            group.update(boards=len(source_ids), exact_boards=sum(r['exact'] for r in exact if r['image_id'] in source_ids))
            groups[name] = group
        result['by_'+grouping] = groups
    return result


def predict(model, loader):
    probabilities, logits = [], []
    model.eval()
    with torch.inference_mode():
        for tensors, _ in loader:
            output = model(tensors)
            logits.append(output.numpy())
            probabilities.append(output.softmax(dim=1).numpy())
    return np.concatenate(probabilities), np.concatenate(logits)


def latency(model, ds):
    tensors = torch.stack([ds[i][0] for i in range(64)])
    elapsed = []
    with torch.inference_mode():
        for repeat in range(10):
            start = time.perf_counter()
            for i in range(0, 64, 16):
                model(tensors[i:i+16])
            duration = time.perf_counter()-start
            if repeat >= 2:
                elapsed.append(duration*1000)
    return dict(threads=torch.get_num_threads(), batch_size=16, board_squares=64, warmups=2, measured_boards=8,
                model_only_board_ms=elapsed, median_model_only_board_ms=float(np.median(elapsed)),
                q25_q75_model_only_board_ms=np.percentile(elapsed, [25, 75]).tolist(),
                parameter_count=sum(p.numel() for p in model.parameters()),
                parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
                scope='Loaded-model CPU forward only; excludes disk, transforms, geometry and phone runtime.')


def evaluate(root, plan, rows, config, initial, images):
    report = dict(evidence='Historical development diagnostic; no independent final holdout used.', results={})
    selection = []
    for variant in plan['variants']:
        spec = spec_for(variant)
        checkpoints = [('frozen', initial, Path(config['classifier']['checkpoint_path']))]
        if variant['train']:
            path = root/'training'/spec.name/'best.pt'
            checkpoints.append(('adapted', torch.load(path, map_location='cpu', weights_only=True), path))
        for mode, state, path in checkpoints:
            name = spec.name+'_'+mode
            dest = root/'evaluation'/name
            dest.mkdir(parents=True, exist_ok=False)
            model = load_model(state).eval()
            result = dict(spec=spec.metadata(), mode=mode, checkpoint_path=str(path), checkpoint_sha256=file_hash(path),
                          checkpoint_file_bytes=path.stat().st_size, adaptation_best_epoch=state['epoch'] if mode == 'adapted' else None)
            for split in ['val', 'test']:
                ds = dataset(root, spec, rows[split], split, {})
                loader = DataLoader(ds, batch_size=16, num_workers=0, shuffle=False)
                probabilities, logits = predict(model, loader)
                result[split] = metrics_by_board(rows[split], probabilities, images)
                np.save(dest/(split+'_probabilities.npy'), probabilities)
                np.save(dest/(split+'_logits.npy'), logits)
                confusion_plot(result[split]['confusion_matrix'], dest/(split+'_confusion.png'))
                if split == 'test':
                    result['latency'] = latency(model, ds)
                del ds, loader
            if variant['train'] and mode == 'adapted':
                selection.append(name)
            report['results'][name] = result
            write(dest/'metrics.json', result)
            write(root/'evaluation_progress.json', report)
            print('evaluate', name, 'val occupied', result['val']['occupied_square_accuracy'], 'test occupied', result['test']['occupied_square_accuracy'], flush=True)
            del model
            gc.collect()
    ranked = sorted(selection, key=lambda n: (-report['results'][n]['val']['occupied_square_accuracy'],
                                             -report['results'][n]['val']['macro_f1'],
                                             report['results'][n]['latency']['median_model_only_board_ms']))
    report.update(validation_selected_configuration=ranked[0], validation_ranked_adapted_configurations=ranked,
                  selection_uses_historical_test=False)
    write(root/'results.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('configs/experiments/b2_crop_resolution.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Refusing to reuse output directory')
    plan, config = read(args.plan), load_config()
    settings = plan['training']
    torch.set_num_threads(settings['cpu_threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); cv2.setNumThreads(1)
    protected = frozen_inputs()
    initial = torch.load(config['classifier']['checkpoint_path'], map_location='cpu', weights_only=True)
    manifest_path = config['classifier']['dataset_paths']['metadata']
    manifest = read(manifest_path)
    if not manifest.get('training_ready') or manifest['split_status'] != 'position_grouped':
        raise ValueError('Frozen approved position-grouped dataset required')
    if initial['manifest_sha256'] != file_hash(manifest_path):
        raise ValueError('Initialization does not match frozen dataset')
    prepared = {r['image_id']: r for r in read('data/classification/prepared_boards.json')}
    reviews, images, positions = (read('data/metadata/'+n+'.json') for n in ['day9_board_reviews', 'images', 'positions'])
    for name in {r['source_image'] for r in manifest['results']}:
        if not review_approved(prepared[name], reviews.get(name)):
            raise ValueError('Stale prepared geometry review: '+name)
    leakage = leakage_audit(manifest['results'], prepared, images, positions)
    rows = {s: [r for r in manifest['results'] if r['split'] == s] for s in ['train', 'val', 'test']}
    sources = {str(p): file_hash(p) for p in [Path(__file__), Path('src/vision/crop_experiments.py'), args.plan]}
    provenance = dict(plan=plan, protected_sha256=protected, source_sha256=sources, leakage=leakage,
                      checkpoint_sha256=file_hash(config['classifier']['checkpoint_path']),
                      manifest_sha256=file_hash(manifest_path), class_weights=manifest['split_report']['class_weights'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir/'provenance.json', provenance)
    cache_report = {}
    for variant in plan['variants']:
        spec = spec_for(variant)
        cache_report[spec.name] = {}
        for split in (['train', 'val', 'test'] if variant['train'] else ['val', 'test']):
            cache_report[spec.name][split] = make_cache(args.output_dir, spec, rows[split], split)
        write(args.output_dir/'cache_report.json', cache_report)
        gc.collect()
    for variant in plan['variants']:
        if variant['train']:
            train_arm(args.output_dir, spec_for(variant), rows, settings, config, initial, provenance)
    # All model/checkpoint selection has finished before historical test predictions.
    evaluate(args.output_dir, plan, rows, config, initial, images)
    verify(protected); verify(sources)
    write(args.output_dir/'preservation.json', dict(unchanged_files=len(protected), protected_inputs_match=True, code_matches=True))
    print('B.2 experiments complete; frozen artifacts preserved.', flush=True)


if __name__ == '__main__':
    main()
