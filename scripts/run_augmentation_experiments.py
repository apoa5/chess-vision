"""B.3 isolated matched augmentation ablations; historical validation selection only."""
import argparse
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

from scripts.run_crop_experiments import read, write, load_model, leakage_audit, metrics_by_board, predict, latency
from scripts.evaluate_classifier import confusion_plot
from src.chess.labels import CLASS_NAMES
from src.utils.config import load_config
from src.vision.augmentation_experiments import AugmentedCropDataset, FAMILIES, GEOMETRY_RECIPE, VISUAL_RECIPE, occupancy_metrics
from src.vision.dataset import review_approved
from src.vision.training import run_epoch


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify(hashes):
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError('Protected input changed: '+path)


def protected_inputs(plan):
    export = read(plan['b2_export'])
    hashes = export['provenance']['protected_sha256'] | export['provenance']['source_sha256']
    hashes.update(export['source_sha256']); hashes.update(export['export_sha256'])
    hashes[plan['b2_export']] = sha(plan['b2_export'])
    b2root = Path(plan['cache_root']).parents[1]
    for path in b2root.rglob('*'):
        if path.is_file() and 'cache' not in path.relative_to(b2root).parts:
            hashes[str(path)] = sha(path)
    for name, splits in read(b2root/'cache_report.json').items():
        for split, item in splits.items():
            hashes[str(b2root/'cache'/name/(split+'.npy'))] = item['sha256']
    for path in Path('outputs/evaluation').glob('crop_b2_*'):
        if path.is_dir() and path != b2root:
            for file in path.rglob('*'):
                if file.is_file():
                    hashes[str(file)] = sha(file)
    for item in export['analysis']['results'].values():
        hashes[item['checkpoint_path']] = item['checkpoint_sha256']
    verify(hashes)
    return hashes


def enrich(metrics):
    metrics.update(occupancy_metrics(metrics))
    for key in ['by_orientation', 'by_position']:
        for value in metrics.get(key, {}).values():
            value.update(occupancy_metrics(value))
    return metrics


def safe_against(candidate, reference):
    """Count-based validation gates, deliberately conservative at small supports."""
    c, r = np.array(candidate['confusion_matrix']), np.array(reference['confusion_matrix'])
    reasons = []
    for field in ['empty_false_positives', 'missed_occupied']:
        if candidate[field] > reference[field]+1:
            reasons.append(field)
    if np.trace(c) < np.trace(r)-2:
        reasons.append('overall_correct')
    for index in [10, 11, 12]:
        if c[index, index] < r[index, index]-1:
            reasons.append(CLASS_NAMES[index]+'_recall')
    if candidate['exact_boards'] < reference['exact_boards']:
        reasons.append('exact_boards')
    for name, group in reference['by_orientation'].items():
        before = np.trace(np.array(group['confusion_matrix'])[1:, 1:])
        after = np.trace(np.array(candidate['by_orientation'][name]['confusion_matrix'])[1:, 1:])
        if after < before-1:
            reasons.append('orientation/'+name)
    return reasons


def select_family(plan, validation):
    """Accepts validation metrics only; no test predictions or latency tie-break."""
    eligibility, ranked = {}, []
    anchor = validation['b2_anchor']
    for family in plan['families']:
        if family == 'none':
            continue
        reasons, gains, f1s = [], [], []
        for seed in plan['training']['seeds']:
            candidate = validation[f'{family}_seed{seed}']
            control = validation[f'none_seed{seed}']
            reasons += [f'seed{seed}/control/'+x for x in safe_against(candidate, control)]
            reasons += [f'seed{seed}/anchor/'+x for x in safe_against(candidate, anchor)]
            gain = candidate['occupied_square_accuracy']-control['occupied_square_accuracy']
            gains.append(gain); f1s.append(candidate['macro_f1'])
            if gain < -1e-12:
                reasons.append(f'seed{seed}/occupied_regression')
        if np.mean(gains) < .01:
            reasons.append('mean_occupied_gain_under_one_percentage_point')
        eligibility[family] = dict(eligible=not reasons, reasons=reasons, mean_occupied_gain=float(np.mean(gains)), mean_macro_f1=float(np.mean(f1s)))
        if not reasons:
            ranked.append(family)
    ranked.sort(key=lambda f: (-eligibility[f]['mean_macro_f1'], -eligibility[f]['mean_occupied_gain'], f))
    return dict(selected_family=ranked[0] if ranked else 'retain_b2', eligible_ranked=ranked,
                eligibility=eligibility, representative_seed=42, selection_input='validation only; completed before test prediction')


def make_dataset(rows, cache, split, family='none', seed=42, augmentation=None):
    return AugmentedCropDataset(rows[split], cache[split], split=split, family=family, seed=seed, augmentation=augmentation)


def train(root, family, seed, rows, cache, plan, initial, weights, visual, sources):
    name = f'{family}_seed{seed}'
    dest = root/'training'/name
    dest.mkdir(parents=True, exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = load_model(initial)
    settings = plan['training']
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'])
    dataset = make_dataset(rows, cache, 'train', family, seed, visual)
    loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(seed))
    val = DataLoader(make_dataset(rows, cache, 'val'), batch_size=16, shuffle=False)
    status = dict(family=family, seed=seed, status='running', history=[], initialization_sha256=sha(plan['initialization']))
    best, start = float('inf'), time.perf_counter()
    for epoch in range(settings['epochs']+1):
        verify(sources)
        train_metrics = None
        if epoch:
            dataset.set_epoch(epoch-1)
            def progress(step, total):
                status.update(epoch=epoch, step=step, total_steps=total, elapsed_seconds=time.perf_counter()-start)
                write(dest/'status.json', status)
                print(name, epoch, step, '/', total, flush=True)
            train_metrics = run_epoch(model, loader, weights, optimizer=optimizer, mode='fine_tune', progress=progress)
        metrics = run_epoch(model, val, weights)
        status['history'].append(dict(epoch=epoch, train=train_metrics, validation=metrics))
        if metrics['loss'] < best:
            best = metrics['loss']
            state = {key: value for key, value in initial.items() if key != 'model_state_dict'}
            state.update(model_state_dict=model.state_dict(), epoch=epoch, origin_b2_epoch=initial['epoch'], experiment_only=True,
                         augmentation_family=family, seed=seed, settings=settings, initialization_sha256=status['initialization_sha256'],
                         validation=metrics)
            temp = dest/'best.tmp'
            torch.save(state, temp); temp.replace(dest/'best.pt')
            status.update(best_epoch=epoch, best_validation_loss=best)
        status.update(elapsed_seconds=time.perf_counter()-start)
        write(dest/'status.json', status)
        print('validation', name, epoch, 'loss', metrics['loss'], flush=True)
    status.update(status='complete', checkpoint_sha256=sha(dest/'best.pt'))
    write(dest/'status.json', status)
    del model, optimizer, dataset, loader, val
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('configs/experiments/b3_augmentation.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Refusing to reuse output directory')
    plan = read(args.plan)
    if 'none' not in plan['families'] or len(set(plan['families'])) != len(plan['families']) or set(plan['families'])-set(FAMILIES):
        raise ValueError('Require augmentation-off control and unique supported families')
    if plan['geometry'] != GEOMETRY_RECIPE or plan['visual'] != VISUAL_RECIPE:
        raise ValueError('Strength metadata must match the tested implementation recipe')
    if 42 not in plan['training']['seeds'] or len(set(plan['training']['seeds'])) != len(plan['training']['seeds']):
        raise ValueError('Require fixed representative seed 42 and unique seeds')
    settings = plan['training']
    torch.set_num_threads(settings['cpu_threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); cv2.setNumThreads(1)
    protected = protected_inputs(plan)
    config = load_config()
    manifest_path = config['classifier']['dataset_paths']['metadata']
    manifest = read(manifest_path)
    initial = torch.load(plan['initialization'], map_location='cpu', weights_only=True)
    if not manifest.get('training_ready') or manifest.get('split_status') != 'position_grouped' or initial['manifest_sha256'] != sha(manifest_path):
        raise ValueError('Approved frozen position-grouped dataset required')
    if initial['crop_spec']['padding_squares'] != 1 or initial['input_size'] != 224:
        raise ValueError('B.2 selected tighter 224 initialization required')
    prepared = {r['image_id']: r for r in read('data/classification/prepared_boards.json')}
    reviews, images, positions = (read('data/metadata/'+n+'.json') for n in ['day9_board_reviews', 'images', 'positions'])
    for name in {r['source_image'] for r in manifest['results']}:
        if not review_approved(prepared[name], reviews.get(name)):
            raise ValueError('Stale reviewed geometry')
    leakage = leakage_audit(manifest['results'], prepared, images, positions)
    rows = {split: [r for r in manifest['results'] if r['split'] == split] for split in ['train', 'val', 'test']}
    cache = {split: np.load(Path(plan['cache_root'])/(split+'.npy'), mmap_mode='r') for split in rows}
    sources = {str(p): sha(p) for p in [Path(__file__), Path('src/vision/augmentation_experiments.py'), args.plan]}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write(args.output_dir/'provenance.json', dict(plan=plan, protected_sha256=protected, source_sha256=sources, leakage=leakage,
                                               manifest_sha256=sha(manifest_path), initialization_sha256=sha(plan['initialization']),
                                               runtime=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__)))
    weights = torch.tensor([manifest['split_report']['class_weights'][name] for name in CLASS_NAMES], dtype=torch.float32)
    # All arms start from the same B.2 weights. No test tensors enter training.
    for seed in settings['seeds']:
        for family in plan['families']:
            train(args.output_dir, family, seed, rows, cache, plan, initial, weights, config['classifier']['augmentation'], sources)
    paths = {'b2_anchor': Path(plan['initialization'])}
    paths.update({f'{family}_seed{seed}': args.output_dir/'training'/f'{family}_seed{seed}'/'best.pt'
                  for seed in settings['seeds'] for family in plan['families']})
    results, validation = {}, {}
    # Finish validation and the safety decision before generating test predictions.
    for name, path in paths.items():
        model = load_model(torch.load(path, map_location='cpu', weights_only=True)).eval()
        dest = args.output_dir/'evaluation'/name; dest.mkdir(parents=True, exist_ok=False)
        probabilities, logits = predict(model, DataLoader(make_dataset(rows, cache, 'val'), batch_size=16))
        metrics = enrich(metrics_by_board(rows['val'], probabilities, images))
        np.save(dest/'val_probabilities.npy', probabilities); np.save(dest/'val_logits.npy', logits)
        confusion_plot(metrics['confusion_matrix'], dest/'val_confusion.png')
        validation[name] = metrics
        results[name] = dict(checkpoint_path=str(path), checkpoint_sha256=sha(path), checkpoint_bytes=path.stat().st_size, val=metrics)
        del model
    selection = select_family(plan, validation)
    write(args.output_dir/'selection.json', selection)
    print('validation selection', selection['selected_family'], flush=True)
    for name, item in results.items():
        state = torch.load(item['checkpoint_path'], map_location='cpu', weights_only=True)
        model = load_model(state).eval()
        ds = make_dataset(rows, cache, 'test')
        probabilities, logits = predict(model, DataLoader(ds, batch_size=16))
        metrics = enrich(metrics_by_board(rows['test'], probabilities, images))
        dest = args.output_dir/'evaluation'/name
        np.save(dest/'test_probabilities.npy', probabilities); np.save(dest/'test_logits.npy', logits)
        confusion_plot(metrics['confusion_matrix'], dest/'test_confusion.png')
        item.update(test=metrics, latency=latency(model, ds), adaptation_best_epoch=state['epoch'] if name != 'b2_anchor' else None)
        write(dest/'metrics.json', item)
        write(args.output_dir/'evaluation_progress.json', results)
        print('test', name, 'occupied', metrics['occupied_square_accuracy'], 'empty_fp', metrics['empty_false_positives'], flush=True)
        del model
        gc.collect()
    write(args.output_dir/'results.json', dict(results=results, selection=selection, evidence='Historical development only; two seeds, one warm-start epoch, three physical test positions.'))
    verify(protected); verify(sources)
    write(args.output_dir/'preservation.json', dict(protected_files=len(protected), protected_match=True, source_match=True))
    print('B.3 complete; baseline and B.2 artifacts preserved.', flush=True)


if __name__ == '__main__':
    main()
