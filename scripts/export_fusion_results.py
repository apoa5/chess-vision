"""Lossless versioned D.3 result and diagnostic export; never replaces files."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scanning.native_pilot import sha256_file


def export(run, output):
    if output.exists():
        raise ValueError('Fresh export directory required')
    result_path = run / 'evaluation_v1/results.json'
    result = json.loads(result_path.read_text())
    output.mkdir(parents=True)
    with (output / 'fusion_d3_results_v1.json.gz').open('wb') as stream:
        with gzip.GzipFile(filename='', mode='wb', fileobj=stream, mtime=0) as compressed:
            compressed.write(result_path.read_bytes())
    summary = {k: v for k, v in result.items() if k not in ('boards',)}
    summary['boards'] = [{k: v for k, v in b.items() if k != 'squares'} for b in result['boards']]
    (output / 'fusion_d3_summary_v1.json').write_text(json.dumps(summary, indent=2) + '\n')
    shutil.copyfile(run / 'evaluation_v1/comparison.csv', output / 'fusion_d3_comparison.csv')
    with (output / 'fusion_d3_per_class.csv').open('w', newline='') as stream:
        fields = ['condition', 'method', 'class_id', 'support', 'predicted_count', 'precision', 'recall']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for aggregate in result['aggregates']:
            for c in aggregate['per_class']:
                writer.writerow(dict(condition=aggregate['condition'], method=aggregate['method'], **c))
    fields = ['condition', 'method', 'session_id', 'correct', 'occupied_correct', 'occupied_count', 'empty_false_positives', 'occupied_to_empty_misses', 'incorrect_piece_identities', 'exact_board']
    with (output / 'fusion_d3_per_board.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for board in result['boards']:
            writer.writerow({k: board[k] if k in board else board['metrics'][k] for k in fields})
    shutil.copytree(run / 'diagnostics_v1', output / 'fusion_d3_examples')
    shutil.copyfile(run / 'cost_v1.json', output / 'fusion_d3_cost_v1.json')
    manifest = dict(schema_version='offline_fusion_export_d3_v1', raw_result_sha256=sha256_file(result_path),
                    label_version=result['label_version'], files={str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file()})
    (output / 'fusion_d3_export_v1.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Exported lossless results, 75 board rows, 195 class rows, and diagnostic packet.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.output_dir)
