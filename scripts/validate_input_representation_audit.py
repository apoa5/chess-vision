"""Validate C.1 provenance and preservation; never reads the final holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.scanning.native_pilot import sha256_file
from scripts.select_video_keyframes import check_hashes


def validate():
    packet = Path('docs/evaluation/input_representation_c1_examples')
    audit_bytes = (packet / 'audit.json').read_bytes()
    audit = json.loads(audit_bytes)
    analysis = json.loads((packet / 'analysis.json').read_text())
    annotations = json.loads((packet / 'visible_extents.json').read_text())
    audit_sha = hashlib.sha256(audit_bytes).hexdigest()
    assert analysis['audit_sha256'] == annotations['audit_sha256'] == audit_sha
    assert analysis['annotations_sha256'] == sha256_file(packet / 'visible_extents.json')
    check_hashes(audit['source_sha256'])
    assert analysis['source_sha256'] == sha256_file('scripts/analyze_input_representations.py')
    protected = json.loads(Path('outputs/research/input_c1_audit_v1/preserved_inputs.json').read_text())['protected_sha256']
    assert not any('holdout' in str(path).lower() for path in protected)
    check_hashes(protected)
    sessions = json.loads(Path('outputs/scanning/d3c_interim_v1/registration.json').read_text())['sessions']
    original_videos = {s['source']['path']: s['source']['sha256'] for s in sessions}
    check_hashes(original_videos)
    source_hashes = {c['source']['path']: c['source']['sha256'] for c in audit['cases']}
    check_hashes(source_hashes)
    original = json.loads(Path('outputs/research/input_c1_audit_v1/audit.json').read_text())
    assert audit == original and len(audit['cases']) == 14
    assert sum(c['verification'].startswith('D2_human') for c in audit['cases']) == 10
    assert len(analysis['visible_feature_cases']) == 7
    assert audit['no_training'] and not audit['new_accuracy_measurements']
    assert not analysis['accuracy_claims']
    for case in audit['cases']:
        assert len(case['frozen_probabilities']) == 13
        assert abs(sum(case['frozen_probabilities']) - 1) < 1e-5
        assert case['canonical_square'][0] in 'abcdefgh' and case['canonical_square'][1] in '12345678'
        assert case['two_branch']['arrays_shared'] and not case['two_branch']['model_implemented']
        for name in ('b2', 'taller', 'expanded'):
            assert case['metrics'][name]['input_shape'] == [4, 224, 224]
        for suffix in ('raw', 'source', 'variants'):
            assert (packet / (case['case_id'] + '_' + suffix + '.jpg')).is_file()
    public = json.loads(Path('docs/evaluation/input_representation_c1_public_sources_v1.json').read_text())
    assert len(public['sources']) == 16
    probe = public['weight_artifact_probe']
    assert probe['http_status'] == 206 and not probe['full_weights_downloaded'] and not probe['full_sha_verified']
    assert probe['tensor_elements'] == 31560077
    return dict(schema_version='input_representation_c1_preservation_v1', status='passed',
                inherited_protected_files=len(protected), d3c_original_videos=len(original_videos),
                diagnostic_source_files=len(source_hashes), frozen_input_exact_matches=14,
                reviewed_D2_cases=10, provisional_D3c_cases=4,
                new_classifier_inference=False, training=False, final_holdout_access=False,
                packet_files=sum(p.is_file() for p in packet.iterdir()),
                packet_bytes=sum(p.stat().st_size for p in packet.iterdir() if p.is_file()),
                audit_sha256=audit_sha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = validate()
    if args.output:
        if args.output.exists():
            raise ValueError('Fresh output path required')
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
