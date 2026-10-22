"""B.2 paired-position uncertainty, error associations and inference-cost export."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch

from src.chess.labels import CLASS_NAMES
from src.utils.image_io import load_image
from src.vision.crop_experiments import packed_input, tensor_from_packed, transform_crop
from src.vision.dataset import file_hash
from src.vision.dataset_loader import preprocess_context_crop
from scripts.run_crop_experiments import load_model, spec_for, verify, read, write


def paired_positions(rows, baseline, candidate):
    """Three physical positions, not 1472 independent square observations."""
    truth = np.array([r['class_id'] for r in rows])
    before, after = baseline.argmax(axis=1) == truth, candidate.argmax(axis=1) == truth
    groups = {}
    for position in sorted({r['position_id'] for r in rows}):
        indices = np.array([i for i, r in enumerate(rows) if r['position_id'] == position and truth[i] != 0])
        groups[position] = dict(occupied=len(indices), before_correct=int(before[indices].sum()),
                                after_correct=int(after[indices].sum()),
                                occupied_accuracy_delta=float((after[indices].sum()-before[indices].sum())/len(indices)))
    leave_one_out = {}
    for excluded in groups:
        kept = [v for k, v in groups.items() if k != excluded]
        count = sum(v['occupied'] for v in kept)
        leave_one_out[excluded] = sum(v['after_correct']-v['before_correct'] for v in kept)/count
    return dict(physical_position_groups=len(groups), by_position=groups,
                leave_one_position_out_occupied_deltas=leave_one_out,
                limitation='Descriptive sensitivity range, not a confidence interval; three clusters and one seed cannot establish generalization.')


def annotations(rows, prepared, visibility_reviews):
    boards = {}
    for row in rows:
        boards.setdefault(row['source_image'], np.zeros((8, 8), int))[row['row'], row['col']] = row['class_id']
    result = []
    for row in rows:
        index = row['row']*8+row['col']
        polygon = np.float32(prepared[row['source_image']]['grid_geometry']['source_square_polygons'][index])
        edges = np.linalg.norm(polygon-np.roll(polygon, -1, axis=0), axis=1)
        board = boards[row['source_image']]
        r, c = row['row'], row['col']
        neighbor = board[max(0, r-1):min(8, r+2), max(0, c-1):min(8, c+2)]
        front = board[r+1, max(0, c-1):min(8, c+2)] if r < 7 else np.zeros(1)
        review = visibility_reviews.get(row['crop_id'], {})
        current = (review.get('crop_sha256') == row['crop_sha256'] and
                   review.get('board_review_signature') == row['board_review_signature'])
        result.append(dict(crop_id=row['crop_id'], raw_mean_target_edge_px=float(edges.mean()),
                           raw_min_target_edge_px=float(edges.min()),
                           raw_target_area_sqrt_px=float(np.sqrt(cv2.contourArea(polygon))),
                           occupied_neighbors=int((neighbor > 0).sum())-int(row['class_id'] != 0),
                           front_neighbor_occupied=bool((front > 0).any()),
                           visibility=review.get('status', 'unreviewed') if current else 'unreviewed'))
    return result


def associations(rows, probabilities, metadata):
    truth = np.array([r['class_id'] for r in rows])
    correct = probabilities.argmax(axis=1) == truth
    groups = {}
    for i, row in enumerate(rows):
        if row['class_id'] == 0:
            continue
        a = metadata[i]
        size = 'under_50px' if a['raw_mean_target_edge_px'] < 50 else '50_to_70px' if a['raw_mean_target_edge_px'] < 70 else 'at_least_70px'
        for key in ['source_detail/'+size, 'front_neighbor/'+str(a['front_neighbor_occupied']),
                    'neighbor_count/'+('dense_4plus' if a['occupied_neighbors'] >= 4 else 'sparse_under4'),
                    'visibility/'+a['visibility']]:
            group = groups.setdefault(key, dict(support=0, correct=0))
            group['support'] += 1
            group['correct'] += int(correct[i])
    for group in groups.values():
        group['occupied_accuracy'] = group['correct']/group['support']
    return dict(groups=groups, limitation='Neighborhood occupancy is a clutter/possible-occlusion proxy, not verified occlusion. Source size, class, position and orientation are confounded.')


def source_detail_inventory(rows, prepared):
    result = {}
    for split in ['train', 'val', 'test']:
        values, minimum_edges, area_widths = [], [], []
        for row in rows:
            if row['split'] != split:
                continue
            polygon = np.float32(prepared[row['source_image']]['grid_geometry']['source_square_polygons'][row['row']*8+row['col']])
            edges = np.linalg.norm(polygon-np.roll(polygon, -1, axis=0), axis=1)
            values.append(float(edges.mean()))
            minimum_edges.append(float(edges.min()))
            area_widths.append(float(np.sqrt(cv2.contourArea(polygon))))
        result[split] = dict(samples=len(values), raw_mean_target_edge_px_percentiles=np.percentile(values, [0, 25, 50, 75, 100]).tolist(),
                             raw_min_target_edge_px_percentiles=np.percentile(minimum_edges, [0, 25, 50, 75, 100]).tolist(),
                             raw_target_area_sqrt_px_percentiles=np.percentile(area_widths, [0, 25, 50, 75, 100]).tolist())
    return result


def preprocessing_latency(result, rows):
    spec = spec_for(result['spec'])
    model = load_model(torch.load(result['checkpoint_path'], map_location='cpu', weights_only=True)).eval()
    images = [load_image(r['crop_path']) for r in rows[:64]]
    timings = []
    with torch.inference_mode():
        for repeat in range(6):
            start = time.perf_counter()
            tensors = []
            for image, row in zip(images, rows[:64]):
                if spec.padding_squares == 2 and spec.detail_factor == 1 and not spec.suppress_outer_context:
                    transformed, polygon = image, row['target_polygon']
                else:
                    transformed, polygon = transform_crop(image, row['target_polygon'], spec)
                tensors.append(torch.from_numpy(preprocess_context_crop(transformed, polygon, spec.input_size)))
            tensors = torch.stack(tensors)
            for i in range(0, 64, 16):
                model(tensors[i:i+16])
            if repeat >= 2:
                timings.append((time.perf_counter()-start)*1000)
    return dict(board_ms=timings, median_board_ms=float(np.median(timings)),
                q25_q75_ms=np.percentile(timings, [25, 75]).tolist(),
                scope='64 loaded crop images: crop/detail transform, RGB/mask resize, normalization, tensor stacking and CPU forward. Excludes PNG loading, geometry and model load; not phone latency.')


def error_sheet(rows, results, probabilities, output):
    baseline = probabilities['context2_224_frozen'].argmax(axis=1)
    truth = np.array([r['class_id'] for r in rows])
    indices = []
    for class_id in [10, 12, 1, 2]:
        selected = [i for i, r in enumerate(rows) if truth[i] == class_id and baseline[i] != truth[i]]
        indices.extend(selected[:1])
    names = ['context2_224_frozen', 'context2_320_adapted', 'context1_224_adapted', 'target_only_224_frozen', 'quarter_detail_320_frozen']
    width, height = 350, 430
    canvas = np.full((len(indices)*height, len(names)*width, 3), 245, np.uint8)
    for r, index in enumerate(indices):
        row = rows[index]
        original = load_image(row['crop_path'])
        for c, name in enumerate(names):
            spec = spec_for(results[name]['spec'])
            packed = packed_input(original, row['target_polygon'], spec)
            rgb = cv2.cvtColor(packed[:3].transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            mask = packed[3]
            contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (0, 200, 0), 1)
            x, y = c*width, r*height
            left = x+(width-spec.input_size)//2
            canvas[y+55:y+55+spec.input_size, left:left+spec.input_size] = rgb
            prediction = int(probabilities[name][index].argmax())
            lines = [name, row['crop_id'], 'true '+row['class_name'],
                     'pred '+CLASS_NAMES[prediction]+f' {probabilities[name][index, prediction]:.3f}']
            for offset, line in enumerate(lines):
                line_y = y+18 if offset == 0 else y+38 if offset == 1 else y+395+(offset-2)*20
                cv2.putText(canvas, line, (x+4, line_y), cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1)
    cv2.imwrite(str(output), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--baseline-report', type=Path, required=True, help='Fresh unchanged baseline classifier predictions.json')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('Refusing to reuse output directory')
    cv2.setNumThreads(1); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    provenance = read(args.run_dir/'provenance.json')
    verify(provenance['protected_sha256']); verify(provenance['source_sha256'])
    summary = read(args.run_dir/'results.json')
    all_rows = read('data/classification/crops.json')['results']
    rows = [r for r in all_rows if r['split'] == 'test']
    prepared = {r['image_id']: r for r in read('data/classification/prepared_boards.json')}
    metadata = annotations(rows, prepared, read('data/metadata/day9_visibility_reviews.json'))
    probabilities = {name: np.load(args.run_dir/'evaluation'/name/'test_probabilities.npy') for name in summary['results']}
    baseline = probabilities['context2_224_frozen']
    fresh = read(args.baseline_report)
    assert [r['crop_id'] for r in fresh] == [r['crop_id'] for r in rows]
    difference = float(np.max(np.abs(baseline-np.float32([r['probabilities'] for r in fresh]))))
    if difference > 1e-7:
        raise ValueError('Experimental control does not reproduce fresh baseline probabilities')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    cache_report = read(args.run_dir/'cache_report.json')
    for variant, splits in cache_report.items():
        for split, metadata_for_cache in splits.items():
            if file_hash(args.run_dir/'cache'/variant/(split+'.npy')) != metadata_for_cache['sha256']:
                raise ValueError('Experiment cache changed: '+variant+'/'+split)
    report = dict(run_results_path=str(args.run_dir/'results.json'), run_results_sha256=file_hash(args.run_dir/'results.json'),
                  baseline_max_probability_difference=difference, current_visibility_review_count=sum(m['visibility'] != 'unreviewed' for m in metadata),
                  test_position_count=3, test_boards=23, test_squares=1472, selected_by_validation=summary['validation_selected_configuration'],
                  source_detail=source_detail_inventory(all_rows, prepared), cache_integrity_verified=True, results={})
    per_class = []
    table = []
    for name, result in summary['results'].items():
        probabilities_for_model = probabilities[name]
        truth = np.array([r['class_id'] for r in rows])
        before_correct = baseline.argmax(axis=1) == truth
        after_correct = probabilities_for_model.argmax(axis=1) == truth
        confusion = np.asarray(result['test']['confusion_matrix'])
        wrong_color = int(confusion[1:7, 7:13].sum()+confusion[7:13, 1:7].sum())
        occupied_identity = int(confusion[1:, 1:].sum()-np.trace(confusion[1:, 1:]))
        details = dict(paired_positions=paired_positions(rows, baseline, probabilities_for_model),
                       associations=associations(rows, probabilities_for_model, metadata),
                       error_kinds=dict(empty_to_piece=int(confusion[0, 1:].sum()),
                                        occupied_to_empty=int(confusion[1:, 0].sum()),
                                        occupied_wrong_color=wrong_color,
                                        occupied_same_color_wrong_type=occupied_identity-wrong_color),
                       fixed_baseline_errors_rescued=int((~before_correct & after_correct).sum()),
                       previously_correct_squares_broken=int((before_correct & ~after_correct).sum()),
                       preprocessing_plus_forward=preprocessing_latency(result, rows))
        report['results'][name] = result | details
        for split in ['val', 'test']:
            for c in result[split]['per_class']:
                per_class.append(dict(variant=name, split=split, **c))
        table.append(dict(variant=name, input_size=result['spec']['input_size'], padding_squares=result['spec']['padding_squares'],
                          best_adaptation_epoch=result['adaptation_best_epoch'],
                          validation_occupied_accuracy=result['val']['occupied_square_accuracy'],
                          test_accuracy=result['test']['accuracy'], test_occupied_accuracy=result['test']['occupied_square_accuracy'],
                          exact_boards=result['test']['exact_boards'], errors=result['test']['error_count'],
                          model_only_board_ms=result['latency']['median_model_only_board_ms'],
                          preprocessing_plus_forward_ms=details['preprocessing_plus_forward']['median_board_ms'],
                          checkpoint_bytes=result['checkpoint_file_bytes'], parameter_bytes=result['latency']['parameter_bytes']))
        print('analyzed', name, flush=True)
    error_sheet(rows, summary['results'], probabilities, args.output_dir/'error_context_examples.png')
    for name, values in [('comparison.csv', table), ('per_class.csv', per_class)]:
        with (args.output_dir/name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    report['source_sha256'] = {str(Path(__file__)): file_hash(Path(__file__))}
    write(args.output_dir/'analysis.json', report)
    verify(provenance['protected_sha256']); verify(provenance['source_sha256'])


if __name__ == '__main__':
    main()
