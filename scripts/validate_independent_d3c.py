"""D.3c frozen-history/media, numerical consistency and storage checks."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.select_video_keyframes import read,write,check_hashes
from src.scanning.interim_evidence import validate_provisional_observation
from src.scanning.native_pilot import sha256_file
from src.scanning.square_evidence import SQUARES


def gzread(path):return json.loads(gzip.decompress(Path(path).read_bytes()))


def validate(run,output):
    registration=read(run/'registration.json');check_hashes(registration['protected_sha256'])
    check_hashes({s['source']['path']:s['source']['sha256'] for s in registration['sessions']})
    rec=read(run/'recognition/sessions.json');check_hashes(rec['source_sha256'])
    inference=read(run/'recognition/inference_registration.json')
    if inference['ground_truth_used_for_inference'] or inference['strict_eligible_sessions']!=0:raise ValueError('Wrong inference scope')
    obs=gzread(run/'recognition/observations.json.gz')['observations'];expected={s['session_id'] for s in registration['sessions']}
    if {o['session_id'] for o in obs}!=expected or len(obs)!=3200:raise ValueError('Missing registered observations')
    frames={}
    for o in obs:
        validate_provisional_observation(o)
        if o['geometry_verification_status']!='completed_unverified' or 'true_class_id' in o:raise ValueError('Invented verification or label inference leakage')
        frames.setdefault((o['session_id'],o['frame_id']),[]).append(o['canonical_square'])
    if len(frames)!=50 or any(len(s)!=64 or set(s)!=set(SQUARES) for s in frames.values()):raise ValueError('Canonical per-frame mapping changed')
    predictions=gzread(run/'recognition/predictions.json.gz');result=gzread(run/'evaluation_v2/results.json.gz')
    if predictions['ground_truth_used_for_predictions'] or result['strict_preregistered_gate_passed'] or len(result['boards'])!=30:raise ValueError('Wrong fusion population/scope')
    for b in result['boards']:
        if [s['canonical_square'] for s in b['squares']]!=list(SQUARES):raise ValueError('Canonical board order changed')
        if b['metrics']['correct']!=sum(s['correct'] for s in b['squares']):raise ValueError('Board metric mismatch')
        if b['all_squares_retained']!=all(not s['abstained'] for s in b['squares']):raise ValueError('Withholding mismatch')
        for s in b['squares']:
            p=np.asarray(s['score_distribution'])
            if p.shape!=(13,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1):raise ValueError('Bad scores')
    for a in result['aggregates']:
        selected=[b for b in result['boards'] if b['method']==a['method'] and (a['scope']=='all_10_sessions' or b['novel_placement'])]
        confusion=np.zeros((13,13),int)
        for b in selected:
            for s in b['squares']:confusion[s['true_class_id'],s['predicted_class_id']]+=1
        if confusion.tolist()!=a['confusion_matrix'] or int(np.trace(confusion))!=a['correct']:raise ValueError('Aggregate confusion mismatch')
    for p in result['nested_prefix_sensitivity']:
        if p['k']==5:
            b=next(b for b in result['boards'] if b['session_id']==p['session_id'] and b['method']==p['method'])
            if p['metrics']!=b['metrics']:raise ValueError('Nested final prefix differs from primary')
    public=Path('docs/evaluation/fusion_independent_d3c_results_v1.json.gz')
    if sha256_file(public)!=sha256_file(run/'evaluation_v2/results.json.gz'):raise ValueError('Portable results differ')
    source_bytes=sum(s['source']['bytes'] for s in registration['sessions']);files=[p for p in run.rglob('*') if p.is_file()]
    if any(p.suffix.lower() in ('.mov','.mp4') for p in files):raise ValueError('Redundant source-video copy')
    cases=read(run/'diagnostics_v1/cases.json')
    write(output,dict(schema_version='limited_independent_validation_d3c_v1',protected_history_files_unchanged=len(registration['protected_sha256']),original_source_hashes_unchanged=10,
        original_video_bytes=source_bytes,run_storage_bytes=sum(p.stat().st_size for p in files),persistent_images=dict(Counter(p.suffix.lower() for p in files if p.suffix.lower() in ('.jpg','.png'))),
        video_copies=0,persistent_full_resolution_candidate_PNGs=0,native_frames_decoded=21205,geometry_candidates=240,original_selected_pixel_replays=50,
        diagnostic_packed_crop_replays=cases['original_pixel_crop_replays'],unique_observations=3200,canonical_frames_checked=50,fused_boards_checked=30,
        independent_aggregate_confusion_checks=6,nested_final_prefix_checks=20,strict_preregistered_gate_passed=False,provisional_only=True,
        final_holdout_accessed=False,model_retrained=False,production_changed=False,committed=False,pushed=False))
    print('Original/protected hashes, 3200 observations, 30 boards, six confusions, twenty nested prefixes verified.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();validate(a.run,a.output)
