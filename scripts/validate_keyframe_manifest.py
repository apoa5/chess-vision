"""Validate D.1 selected-frame provenance and geometry before offline consumption."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from src.scanning.native_pilot import sha256_file
from src.scanning.video_keyframes import geometry_state, validate_settings


def validate_manifest(obj, cache=None):
    cache = {} if cache is None else cache
    def checked(path, expected):
        if path not in cache: cache[path] = sha256_file(path)
        if cache[path] != expected: raise ValueError('Hash mismatch: '+path)
    def read(path): return json.loads(Path(path).read_text())
    if obj['schema_version'] != 'offline_selected_keyframes_d1_v1': raise ValueError('Unknown schema')
    if (obj['designation'] != 'development_only' or obj['split'] != 'pilot' or
        obj['training_eligible'] or obj['final_holdout'] or obj['ground_truth_FEN_used_for_selection'] or
        obj['piece_recognition_evaluated'] or obj['per_square_evidence_or_fusion_implemented']):
        raise ValueError('Invalid experiment scope')
    settings = validate_settings(obj['settings'])
    if obj['budget'] not in settings['budgets']: raise ValueError('Invalid budget')
    if obj['policy'] not in settings['geometry_policies']: raise ValueError('Unknown policy')
    primary = obj['dataset'] == 'primary_3s_high'
    if obj['dataset'] not in ('primary_3s_high','paired_frozen_high','paired_frozen_compressed'):
        raise ValueError('Unknown dataset')
    if obj['D2_input_role'] != ('primary_candidates' if primary else 'robustness_comparison_only'):
        raise ValueError('Incorrect downstream role')
    checked(obj['source_catalog'], obj['source_catalog_sha256'])
    catalog = read(obj['source_catalog']); registered = {s['capture_session_id']:s for s in catalog['sessions']}
    role = 'compressed_comparison_source' if obj['dataset'] == 'paired_frozen_compressed' else 'primary_source'
    seen = set(); count = 0
    for session in obj['sessions']:
        sid = session['capture_session_id']
        if sid in seen: raise ValueError('Duplicate session')
        seen.add(sid); original = registered[sid]
        if any(session[k] != original[k] for k in ('position_id','position_group_id','independence_group_id')):
            raise ValueError('Session/position grouping mismatch')
        source = session['source_video']
        if any(source[k] != original[role][k] for k in ('path','sha256')): raise ValueError('Wrong source version')
        checked(source['path'],source['sha256'])
        rows = session['selected']
        if len(rows) > obj['budget'] or len({r['frame_id'] for r in rows}) != len(rows): raise ValueError('Invalid selection count')
        last = None
        for row in rows:
            if row['source_video_sha256'] != source['sha256']: raise ValueError('Wrong frame source')
            t = Fraction(row['pts'])*Fraction(*row['time_base'])
            if (t != Fraction(*row['pts_time']) or abs(float(t)*1000-row['timestamp_ms']) > 1e-6 or
                (last is not None and t <= last)): raise ValueError('Invalid timestamp/order')
            last = t
            if not np.isfinite(row['quality_score']) or not np.isfinite(row['view_descriptor']).all():
                raise ValueError('Nonfinite features')
            checked(row['path'],row['derivative_sha256'])
            image = cv2.imread(row['path'])
            if image is None or list(image.shape[:2]) != [row['height'],row['width']]: raise ValueError('Wrong dimensions')
            if hashlib.sha256(image.tobytes()).hexdigest() != row['decoded_pixel_sha256']: raise ValueError('Wrong pixels')
            pointer = row['geometry_reference']; checked(pointer,row['geometry_reference_sha256']); raw = read(pointer)
            if primary:
                if (raw['geometry_config_sha256'] != obj['geometry_config_sha256'] or
                    raw['evaluator_sha256'] != obj['geometry_evaluator_sha256']): raise ValueError('Geometry version mismatch')
                binding = raw['binding']; g = raw['policies'][obj['policy']]
            else:
                if raw['config_sha256'] != obj['geometry_config_sha256'] or raw['source_sha256'] != obj['geometry_evaluator_sha256']:
                    raise ValueError('Cached geometry version mismatch')
                candidates = [f for f in raw['frames'] if f['session_id']==sid and f['frame_id']==row['frame_id']]
                if len(candidates)!=1: raise ValueError('Missing cached geometry')
                f=candidates[0]
                if any(f[k]!=row[k] for k in ('pts','time_base','derivative_sha256')): raise ValueError('Cached frame mismatch')
                binding = {k:row[k] for k in ('frame_id','pts','time_base','source_video_sha256','derivative_sha256')}|{'session_id':sid}
                g=f['policies'][obj['policy']]
            state,_ = geometry_state(dict(row,session_id=sid),g,binding,row['review_annotation'])
            if state not in ('completed_unverified','visually_verified') or state != row['geometry_state']:
                raise ValueError('Unusable or misrepresented geometry')
            count += 1
    return dict(sessions=len(seen),selected_frames=count,valid=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('manifest',type=Path);args=p.parse_args()
    print(json.dumps(validate_manifest(json.loads(args.manifest.read_text()))))

if __name__=='__main__':main()
