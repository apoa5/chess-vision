"""B.3 actual training previews, paired square changes and descriptive exports."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from scripts.run_augmentation_experiments import read, write, sha, verify
from scripts.analyze_crop_experiments import paired_positions
from src.chess.labels import CLASS_NAMES
from src.vision.augmentation_experiments import FAMILIES, augment_packed
from src.vision.crop_experiments import MEAN, STD, tensor_from_packed


def render(packed):
    rgb = packed[:3].transpose(1, 2, 0).copy()
    rgb[packed[3].astype(bool)] = np.uint8(.85*rgb[packed[3].astype(bool)]+.15*np.array([0, 255, 0]))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def preview(output, rows, cache):
    indices = [next(i for i, r in enumerate(rows) if r['class_id'] == c) for c in [0, 2, 10, 12]]
    canvas = np.full((4*285, len(FAMILIES)*245, 3), 245, np.uint8)
    traces = []
    for row_index, index in enumerate(indices):
        row, packed = rows[index], cache[index]
        for col, family in enumerate(FAMILIES):
            seed = 0
            while True:
                transformed, trace = augment_packed(packed, family, np.random.default_rng(seed))
                if trace['applied'] or family in ('none', 'legacy_visual'):
                    break
                seed += 1
            if family == 'legacy_visual':
                tensor = tensor_from_packed(packed, rng=np.random.default_rng(42))
                transformed[:3] = np.rint(np.clip(tensor[:3].transpose(1, 2, 0)*STD+MEAN, 0, 1)*255).astype(np.uint8).transpose(2, 0, 1)
                trace.update(applied=True, legacy_visual_rng_seed=42)
            x, y = col*245, row_index*285
            canvas[y+40:y+264, x+10:x+234] = render(transformed)
            cv2.putText(canvas, family, (x+5, y+17), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 1)
            cv2.putText(canvas, CLASS_NAMES[row['class_id']], (x+5, y+33), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 1)
            traces.append(dict(crop_id=row['crop_id'], split=row['split'], recipe_rng_seed=seed, **trace))
    cv2.imwrite(str(output/'augmentation_examples.png'), canvas)
    write(output/'augmentation_traces.json', traces)


def csv_write(path, records):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)


def diagnostic(output, changes, rows, cache, probabilities):
    selected = []
    for family in ['perspective_affine', 'framing', 'lighting', 'video_quality']:
        for kind in ['rescued', 'broken']:
            examples = [r for r in changes if r['model'] == family+'_seed42' and r['change_kind'] == kind]
            selected.extend(examples[:2])
    if not selected:
        write(output/'diagnostic_examples.json', [])
        return
    canvas = np.full((len(selected)*270, 650, 3), 245, np.uint8)
    by_id = {r['crop_id']: i for i, r in enumerate(rows)}
    for j, change in enumerate(selected):
        index = by_id[change['crop_id']]
        y = j*270
        canvas[y+35:y+259, 8:232] = render(cache[index])
        text = [change['model']+' '+change['change_kind'], change['crop_id'],
                'truth: '+change['truth'], 'control: '+change['control_prediction'], 'variant: '+change['variant_prediction']]
        for line, value in enumerate(text):
            cv2.putText(canvas, value, (245, y+30+line*25), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 1)
    cv2.imwrite(str(output/'paired_prediction_examples.png'), canvas)
    write(output/'diagnostic_examples.json', selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--plan', type=Path, default=Path('configs/experiments/b3_augmentation.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()
    if args.output_dir.exists():parser.error('Refusing to reuse output directory')
    if not args.preview_only and args.run_dir is None:parser.error('--run-dir required for analysis')
    plan = read(args.plan)
    all_rows = read('data/classification/crops.json')['results']
    split = 'train' if args.preview_only else 'test'
    rows = [r for r in all_rows if r['split'] == split]
    cache = np.load(Path(plan['cache_root'])/(split+'.npy'), mmap_mode='r')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    if args.preview_only:
        preview(args.output_dir, rows, cache)
        return
    provenance = read(args.run_dir/'provenance.json')
    verify(provenance['protected_sha256']); verify(provenance['source_sha256'])
    summary = read(args.run_dir/'results.json')
    truth = np.array([r['class_id'] for r in rows])
    probabilities = {n: np.load(args.run_dir/'evaluation'/n/'test_probabilities.npy') for n in summary['results']}
    anchor = probabilities['b2_anchor']
    b2root = Path(plan['cache_root']).parents[1]
    b2 = np.load(b2root/'evaluation/context1_224_adapted/test_probabilities.npy')
    difference = float(np.abs(anchor-b2).max())
    if difference > 1e-7:raise ValueError('B.2 anchor did not reproduce')
    epoch_zero_differences = {}
    for name, result in summary['results'].items():
        if result['adaptation_best_epoch'] == 0:
            delta = float(np.abs(probabilities[name]-anchor).max())
            if delta > 1e-7:raise ValueError('Epoch-zero fallback changed B.2 probabilities: '+name)
            epoch_zero_differences[name] = delta
    changes, table, classes = [], [], []
    for name, result in summary['results'].items():
        if name == 'b2_anchor':
            control_name = name
        else:
            control_name = 'none_seed'+name.rsplit('seed', 1)[1]
        control = probabilities[control_name].argmax(1)
        prediction = probabilities[name].argmax(1)
        before, after = control == truth, prediction == truth
        for i in np.flatnonzero(prediction != control):
            kind = 'rescued' if not before[i] and after[i] else 'broken' if before[i] and not after[i] else 'changed_wrong_prediction'
            changes.append(dict(model=name, control_model=control_name, crop_id=rows[i]['crop_id'], source_image=rows[i]['source_image'],
                                position_id=rows[i]['position_id'], chess_square=rows[i]['chess_square'], truth=CLASS_NAMES[truth[i]],
                                control_prediction=CLASS_NAMES[control[i]], variant_prediction=CLASS_NAMES[prediction[i]], change_kind=kind,
                                occupied=bool(truth[i] != 0)))
        result['paired_control'] = dict(control=control_name, rescued=int((~before & after).sum()), broken=int((before & ~after).sum()),
                                       occupied_rescued=int((~before & after & (truth != 0)).sum()),
                                       occupied_broken=int((before & ~after & (truth != 0)).sum()),
                                       paired_positions=paired_positions(rows, probabilities[control_name], probabilities[name]))
        result['paired_anchor'] = paired_positions(rows, anchor, probabilities[name])
        for split_name in ['val', 'test']:
            m = result[split_name]
            for cls in m['per_class']:classes.append(dict(model=name, split=split_name, **cls))
            table.append(dict(model=name, split=split_name, best_epoch=result['adaptation_best_epoch'], accuracy=m['accuracy'],
                              occupied_accuracy=m['occupied_square_accuracy'], empty_fp=m['empty_false_positives'],
                              empty_fpr=m['empty_false_positive_rate'], missed_occupied=m['missed_occupied'],
                              missed_occupied_rate=m['missed_occupied_rate'], black_rook_recall=m['per_class_recall'][10],
                              black_queen_recall=m['per_class_recall'][11], black_king_recall=m['per_class_recall'][12],
                              macro_f1=m['macro_f1'], exact_boards=m['exact_boards'], boards=m['boards'],
                              forward_ms=result['latency']['median_model_only_board_ms'], checkpoint_bytes=result['checkpoint_bytes']))
    diagnostic(args.output_dir, changes, rows, cache, probabilities)
    csv_write(args.output_dir/'comparison.csv', table); csv_write(args.output_dir/'per_class.csv', classes)
    if changes:csv_write(args.output_dir/'paired_square_changes.csv', changes)
    families = {}
    for family in plan['families']:
        family_results = [summary['results'][f'{family}_seed{seed}'] for seed in plan['training']['seeds']]
        families[family] = {}
        for split_name in ['val', 'test']:
            fields = ['accuracy', 'occupied_square_accuracy', 'empty_false_positives', 'empty_false_positive_rate',
                      'missed_occupied', 'missed_occupied_rate', 'macro_f1', 'exact_boards']
            values = {field: [r[split_name][field] for r in family_results] for field in fields}
            for index, label in [(10, 'black_rook_recall'), (11, 'black_queen_recall'), (12, 'black_king_recall')]:
                values[label] = [r[split_name]['per_class_recall'][index] for r in family_results]
            families[family][split_name] = {field: dict(mean=float(np.mean(v)), minimum=float(min(v)), maximum=float(max(v))) for field, v in values.items()}
    summary.update(b2_max_probability_difference=difference, epoch_zero_max_probability_differences=epoch_zero_differences,
                   paired_changed_squares=changes, family_summaries=families,
                   source_sha256={str(Path(__file__)):sha(Path(__file__))},
                   limitation='Three physical test positions; two seeds conditional on the same B.2 initialization; one adaptation epoch; historical sessions unknown; no unbiased holdout claim.')
    write(args.output_dir/'analysis.json', summary)
    verify(provenance['protected_sha256']); verify(provenance['source_sha256'])


if __name__ == '__main__':
    main()
