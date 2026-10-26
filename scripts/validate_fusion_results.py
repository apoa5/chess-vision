"""Independent consistency/preservation checks for an existing D.3 export."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.select_video_keyframes import check_hashes
from src.scanning.native_pilot import sha256_file
from src.scanning.square_evidence import SQUARES


def validate(run, export):
    inputs = json.loads((run / 'inputs.json').read_text())
    check_hashes(inputs['protected_sha256'])
    check_hashes(inputs['source_sha256'])
    index = json.loads(Path('data/metadata/evidence_d2_index_v1.json').read_text())
    check_hashes(index['artifacts'])
    bundle = json.loads((export / 'fusion_d3_export_v1.json').read_text())
    check_hashes({str(export / p): h for p, h in bundle['files'].items()})
    raw = gzip.decompress((export / 'fusion_d3_results_v1.json.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == bundle['raw_result_sha256'] == sha256_file(run / 'evaluation_v1/results.json')
    result = json.loads(raw)
    assert len(result['boards']) == 75 and len(result['aggregates']) == 15
    assert not result['exclusions'] and result['validated_camera_index_canonical_mappings'] == 3520
    for board in result['boards']:
        assert [s['canonical_square'] for s in board['squares']] == list(SQUARES)
        assert len(set(board['input_frame_ids'])) == board['input_frame_count']
        assert board['metrics']['correct'] == sum(s['correct'] for s in board['squares'])
        assert board['exact_board'] == all(s['correct'] for s in board['squares'])
        for square in board['squares']:
            score = square['score_distribution']
            assert len(score) == 13 and all(0 <= p <= 1 for p in score) and abs(sum(score) - 1) < 1e-10
            assert abs(score[square['predicted_class_id']] - max(score)) < 1e-12
            assert (square['operational_class_id'] is None) == square['abstained']
            assert square['correct'] == (square['predicted_class_id'] == square['true_class_id'])
    for aggregate in result['aggregates']:
        boards = [b for b in result['boards'] if (b['condition'], b['method']) == (aggregate['condition'], aggregate['method'])]
        confusion = [[0] * 13 for _ in range(13)]
        for board in boards:
            for square in board['squares']:
                confusion[square['true_class_id']][square['predicted_class_id']] += 1
        assert confusion == aggregate['confusion_matrix']
        assert sum(confusion[i][i] for i in range(13)) == aggregate['correct']
        assert sum(b['exact_board'] for b in boards) == aggregate['exact_boards']
    cost = json.loads((run / 'cost_v1.json').read_text())
    assert len(cost['timings']) == 75 and sum(len(t['seconds']) for t in cost['timings']) == 225
    return dict(schema_version='offline_fusion_validation_d3_v1', protected_files_unchanged=len(inputs['protected_sha256']),
                D2_index_artifacts_unchanged=len(index['artifacts']), validated_camera_index_canonical_mappings=3520,
                board_outputs_checked=75, square_outputs_checked=4800, aggregate_confusions_independently_recounted=15,
                portable_export_hashes_checked=len(bundle['files']), lossless_result_sha256=bundle['raw_result_sha256'],
                benchmark_replays=225, final_holdout_accessed=False, classifier_rerun=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--export-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh validation output required')
    result = validate(args.run, args.export_dir)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
