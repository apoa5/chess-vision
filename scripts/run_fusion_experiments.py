"""D.3 frozen-evidence validation and GT-blind predictions, separate from evaluation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import tracemalloc

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.select_video_keyframes import read,write,check_hashes
from src.chess.labels import CLASS_NAMES
from src.scanning.native_pilot import sha256_file
from src.scanning.square_evidence import verify_reviews,validate_bundle,validate_observation
from src.scanning.video_keyframes import geometry_state
from src.scanning.offline_fusion import validate_config,fuse_board


def load_evidence(path):
    """Validate bindings first; any corruption fails closed before predictions."""
    e=read(path);packet=read(e['packet_path']);reviews=read(e['reviews_path'])
    check_hashes({e['packet_path']:e['packet_sha256'],e['reviews_path']:e['reviews_sha256'],e['config_path']:e['config_sha256'],e['observations_path']:e['observations_sha256'],e['model']['checkpoint_path']:e['model']['checkpoint_sha256']})
    check_hashes(e['model']['source_sha256']);statuses=verify_reviews(packet,reviews,e['packet_sha256'])
    obs=[json.loads(x) for x in Path(e['observations_path']).read_text().splitlines()];validate_bundle(e,obs)
    frame_lookup={(f['session_id'],f['frame_id']):f for f in e['frames']};packet_lookup={f['review_id']:f for f in packet['frames']}
    sessions={s['capture_session_id']:s for s in e['sessions']};geometry={};excluded=[];usable=[]
    for f in e['frames']:
        row=packet_lookup[f['review_id']];original=row['frame'];sid=f['session_id'];fid=f['frame_id']
        if row['session']['capture_session_id']!=sid or original['frame_id']!=fid:raise ValueError('Frame/session review mismatch')
        if statuses[f['review_id']]['human_review']!='human_verified':
            excluded.append(dict(session_id=sid,frame_id=fid,reason=statuses[f['review_id']]['geometry_status']));continue
        check_hashes({original['path']:original['derivative_sha256'],original['geometry_reference']:original['geometry_reference_sha256']})
        pair=read(original['geometry_reference']);g=pair['policies']['marked-board']
        candidate=dict(original,session_id=sid);state,_=geometry_state(candidate,g,pair['binding'])
        if state!='completed_unverified':
            excluded.append(dict(session_id=sid,frame_id=fid,reason=state));continue
        if g['source_square_polygons']!=row['source_square_polygons'] or g['semantic_square_ids']!=row['canonical_square_ids'] or g['rotation_quarters_ccw']!=row['rotation_quarters_ccw']:raise ValueError('Reviewed orientation/grid changed')
        geometry[(sid,fid)]=g
    for s in sessions.values():check_hashes({s['source_video']['path']:s['source_video']['sha256']})
    for o in obs:
        key=(o['session_id'],o['frame_id'])
        if key not in geometry:
            excluded.append(dict(observation_id=o['observation_id'],reason='frame_geometry_unusable'));continue
        validate_observation(o);f=frame_lookup[key];g=geometry[key];s=sessions[o['session_id']];d=f['d1_metadata']
        if (o['position_id']!=s['position_id'] or o['source_video_sha256']!=s['source_video']['sha256'] or
            o['frame_sha256']!=d['derivative_sha256'] or o['original_pts']!=d['pts'] or o['time_base']!=d['time_base'] or
            o['geometry_reference_sha256']!=d['geometry_reference_sha256'] or o['semantic_orientation']!=g['predicted_camera_orientation'] or
            o['rotation_quarters_ccw']!=g['rotation_quarters_ccw'] or o['class_names']!=list(CLASS_NAMES)):
            raise ValueError('Observation coordinate/orientation/source identity mismatch')
        for m in o['condition_memberships']:
            if m['condition'] not in f['conditions']:raise ValueError('Condition/frame mismatch')
        usable.append(o)
    return e,packet,usable,excluded


def snapshot(packet):
    protected=dict(packet['protected_sha256']);paths=[]
    for directory in ['src','scripts','tests','configs','config','data/metadata','docs/evaluation','web/src','web/tests','outputs/scanning/d2_pilot_run1']:
        paths += [p for p in Path(directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    protected.update({str(p):sha256_file(p) for p in paths})
    check_hashes(protected);return protected


def run(config_path,output):
    if output.exists():raise ValueError('Fresh D.3 directory required')
    start=time.perf_counter();config=validate_config(read(config_path));e,packet,obs,excluded=load_evidence(config['evidence_manifest'])
    protected=snapshot(packet);output.mkdir(parents=True,exist_ok=False)
    sources={p:sha256_file(p) for p in ['src/scanning/offline_fusion.py','scripts/run_fusion_experiments.py',str(config_path)]}
    write(output/'inputs.json',dict(schema_version='offline_fusion_inputs_d3_v1',config_path=str(config_path),config_sha256=sha256_file(config_path),
        evidence_path=config['evidence_manifest'],evidence_sha256=sha256_file(config['evidence_manifest']),observations_path=e['observations_path'],
        observations_sha256=e['observations_sha256'],protected_sha256=protected,source_sha256=sources,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),excluded=excluded))
    write(output/'settings.json',config)
    boards=[];replays=0;timings=[];tracemalloc.start();fusion_start=time.perf_counter()
    for condition in config['conditions']:
        for session in e['sessions']:
            rows=[o for o in obs if o['session_id']==session['capture_session_id'] and any(m['condition']==condition for m in o['condition_memberships'])]
            if not rows:raise ValueError('No usable condition/session observations; report exclusions before retry')
            for method in config['methods']:
                t=time.perf_counter();board=fuse_board(rows,method,config);timings.append(time.perf_counter()-t)
                if board!=fuse_board(list(reversed(rows)),method,config):raise ValueError('Input-order nondeterminism')
                replays+=1;board['condition']=condition;boards.append(board)
            print(condition,session['capture_session_id'],len(rows)//64,'frames',flush=True)
    current,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    result=dict(schema_version='offline_fusion_predictions_d3_v1',designation='development_only',final_holdout=False,
        evidence_sha256=sha256_file(config['evidence_manifest']),config_sha256=sha256_file(config_path),
        ground_truth_loaded_for_predictions=False,calibration_applied=False,class_names=list(CLASS_NAMES),boards=boards,
        input_observations=len(obs),excluded=excluded,deterministic_replays=replays,
        timing=dict(fusion_plus_replays_seconds=time.perf_counter()-fusion_start,first_pass_board_call_seconds=timings,
                    python_tracemalloc_peak_bytes=peak,memory_scope='Fusion allocations only; excludes preloaded JSON/NumPy/native libraries'),
        elapsed_including_validation_seconds=time.perf_counter()-start,source_sha256=sources)
    write(output/'predictions.json',result);check_hashes(protected)
    write(output/'preservation.json',dict(protected_files_unchanged=len(protected),D2_unchanged=True,production_unchanged=True,classifier_rerun=False))
    print('Saved',len(boards),'board predictions; replays',replays,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path('configs/experiments/fusion_d3.json'));p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.config,a.output_dir)
