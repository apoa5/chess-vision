"""Evaluate the validation-selected classifier on the fixed held-out test split."""
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
from torch.utils.data import DataLoader
from src.chess.labels import CLASS_NAMES, LABEL_VERSION
from src.utils.config import load_config
from src.vision.classifier import create_classifier
from src.vision.classifier_evaluation import prediction_metrics
from src.vision.dataset import ContextSquareDataset, file_hash


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def contact_sheet(rows, path):
    if not rows:
        return
    width, height = 350, 420
    sheet = np.full((((len(rows)+3)//4)*height, width*4, 3), 245, np.uint8)
    for index, row in enumerate(rows):
        image = cv2.imread(row['crop_path'])
        ratio = min(330/image.shape[1], 330/image.shape[0])
        image = cv2.resize(image, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_AREA)
        polygon = np.rint(np.asarray(row['target_polygon'])*ratio).astype(np.int32)
        cv2.polylines(image, [polygon], True, (0, 200, 0), 2)
        x, y = (index%4)*width, (index//4)*height
        sheet[y:y+image.shape[0], x:x+image.shape[1]] = image
        lines = [row['crop_id'], f"True: {row['class_name']}",
                 f"Pred: {row['predicted_class']} ({row['confidence']:.3f})"]
        for offset, line in enumerate(lines):
            cv2.putText(sheet, line, (x+5, y+350+offset*24), cv2.FONT_HERSHEY_SIMPLEX, .45, (20,20,20), 1, cv2.LINE_AA)
    cv2.imwrite(str(path), sheet)


def confusion_plot(matrix, path):
    cell, left, top = 60, 190, 200
    canvas = np.full((top+13*cell+40, left+13*cell+20, 3), 255, np.uint8)
    for i, name in enumerate(CLASS_NAMES):
        cv2.putText(canvas, name, (5, top+i*cell+35), cv2.FONT_HERSHEY_SIMPLEX, .48, (0,0,0), 1, cv2.LINE_AA)
        label = np.full((cell, top-25, 3), 255, np.uint8)
        cv2.putText(label, name, (3, 35), cv2.FONT_HERSHEY_SIMPLEX, .48, (0,0,0), 1, cv2.LINE_AA)
        label = cv2.rotate(label, cv2.ROTATE_90_COUNTERCLOCKWISE)
        canvas[25:top, left+i*cell:left+(i+1)*cell] = label
        for j in range(13):
            value = int(matrix[i][j]); fraction = value/max(sum(matrix[i]), 1)
            color = (255, int(255-160*fraction), int(255-220*fraction))
            x, y = left+j*cell, top+i*cell
            cv2.rectangle(canvas, (x,y), (x+cell-1,y+cell-1), color, -1)
            cv2.putText(canvas, str(value), (x+8,y+35), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Rows: truth; columns: prediction. Color normalized per true class.', (10, canvas.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,0), 1)
    cv2.imwrite(str(path), canvas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/week2/classifier_evaluation'))
    args = parser.parse_args()
    config = load_config(args.config) if args.config else load_config()
    settings = config['classifier']
    torch.set_num_threads(settings['cpu_threads']); cv2.setNumThreads(1)
    torch.use_deterministic_algorithms(True)
    manifest_path = settings['dataset_paths']['metadata']
    manifest_sha = file_hash(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    checkpoint_path = settings['checkpoint_path']
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if (checkpoint['manifest_sha256'] != manifest_sha or
        checkpoint['dataset_fingerprint'] != manifest['split_report']['dataset_fingerprint']):
        raise ValueError('Checkpoint and dataset version do not match.')
    if (checkpoint['class_names'] != list(CLASS_NAMES) or checkpoint['label_version'] != LABEL_VERSION
        or checkpoint['input_channels'] != 4 or checkpoint['architecture'] != 'mobilenet_v3_small'):
        raise ValueError('Unsupported checkpoint input/label contract.')
    dataset = ContextSquareDataset(manifest_path, 'test', input_size=checkpoint['input_size'])
    # Verify the actual crop bytes, not just the manifest identifiers.
    for row in dataset.rows:
        if file_hash(Path(row['crop_path'])) != row['crop_sha256']:
            raise ValueError(f"Crop changed: {row['crop_id']}")
    model = create_classifier(pretrained=False); model.load_state_dict(checkpoint['model_state_dict']); model.eval()
    loader = DataLoader(dataset, batch_size=settings['batch_size'], shuffle=False, num_workers=0)
    probabilities = []; labels = []; start = time.perf_counter()
    with torch.inference_mode():
        for images, truth in loader:
            output = model(images)
            if not torch.isfinite(output).all(): raise ValueError('Nonfinite model output.')
            probabilities.append(output.softmax(dim=1).numpy()); labels.extend(truth.tolist())
    elapsed = time.perf_counter()-start
    if file_hash(manifest_path) != manifest_sha: raise ValueError('Manifest changed during evaluation.')
    probabilities = np.concatenate(probabilities)
    metrics = prediction_metrics(labels, probabilities)
    rows = []
    for row, probability in zip(dataset.rows, probabilities):
        predicted = int(probability.argmax())
        rows.append({**row, 'predicted_class': CLASS_NAMES[predicted], 'predicted_class_id': predicted,
                     'confidence': float(probability[predicted]), 'correct': predicted == row['class_id'],
                     'probabilities': probability.tolist()})
    confusion = metrics['confusion_matrix']
    pairs = sorted([{'true': CLASS_NAMES[i], 'predicted': CLASS_NAMES[j], 'count': confusion[i][j]}
                    for i in range(13) for j in range(13) if i != j and confusion[i][j]], key=lambda r:-r['count'])
    report = {'split': 'test', 'samples': len(dataset), 'boards': len({r['source_image'] for r in rows}),
              'positions': sorted({r['position_id'] for r in rows}), 'checkpoint_path': str(checkpoint_path),
              'checkpoint_sha256': file_hash(checkpoint_path), 'best_epoch': checkpoint['epoch'],
              'architecture': checkpoint['architecture'], 'input_size': checkpoint['input_size'],
              'input_channels': 4, 'label_version': LABEL_VERSION, 'class_names': list(CLASS_NAMES),
              'training_settings': checkpoint['settings'], 'validation': checkpoint['validation'],
              'dataset_fingerprint': checkpoint['dataset_fingerprint'], 'manifest_sha256': manifest_sha,
              'elapsed_seconds_including_crop_loading': elapsed, 'metrics': metrics, 'confusion_pairs': pairs,
              'model_modified': False, 'limitations': ['Only three held-out positions; same board, pieces and capture protocol.',
                  'Only successfully localized, reviewed boards are evaluated; this is not end-to-end accuracy.',
                  'Confidence bins are descriptive; no test-driven calibration or model selection performed.']}
    destination = args.output_dir; destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'evaluation.json', report); write_json(destination/'predictions.json', rows)
    write_json(config['data']['metadata']/'day12_classifier_evaluation.json', report)
    with (destination/'per_class.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['class_name','support','precision','recall','f1'])
        writer.writeheader(); writer.writerows(metrics['per_class'])
    with (destination/'confusion_matrix.csv').open('w', newline='') as handle:
        writer = csv.writer(handle); writer.writerow(['true / predicted', *CLASS_NAMES])
        for name, counts in zip(CLASS_NAMES, confusion): writer.writerow([name, *counts])
    confusion_plot(confusion, destination/'confusion_matrix.png')
    errors = sorted([r for r in rows if not r['correct']], key=lambda r:-r['confidence'])
    for offset in range(0, len(errors), 20):
        contact_sheet(errors[offset:offset+20], destination/f'errors_{offset//20+1:02d}.png')
    contact_sheet(sorted(rows, key=lambda r:r['confidence'])[:20], destination/'lowest_confidence.png')
    correct_examples = []
    for name in CLASS_NAMES:
        correct_examples.extend(sorted([r for r in rows if r['correct'] and r['class_name'] == name],
                                       key=lambda r:-r['confidence'])[:1])
    contact_sheet(correct_examples, destination/'correct_high_confidence.png')
    print(json.dumps({'samples':len(dataset), 'accuracy':metrics['accuracy'], 'macro_f1':metrics['macro_f1'],
                      'occupied_square_accuracy':metrics['occupied_square_accuracy'], 'errors':metrics['error_count'],
                      'output_dir':str(destination)}, indent=2))


if __name__ == '__main__': main()
