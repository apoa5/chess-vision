"""Bounded-storage D.3c native intake/geometry adapter. Frozen components are reused."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from fractions import Fraction
import gzip
import json
from pathlib import Path
import sys
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import chess
import cv2
import numpy as np
from scripts.select_video_keyframes import read,write,check_hashes
from scripts.evaluate_independent_fusion import known_history
from src.scanning.interim_video import match_position,index_video,decode_exact,pixel_hash,write_gzip
from src.scanning.native_pilot import sha256_file
from src.scanning.video_keyframes import temporal_candidates,geometry_state,board_features,select_keyframes
from src.utils.config import load_config
from src.vision.position_pipeline import PreparationError
from src.vision.geometry_reliability import prepare_reliable_raw_photo
from src.vision.geometry_audit import context_transform,project,raw_grid_overlay
from src.vision.dataset import chess_square_at

CONFIG=Path('configs/experiments/fusion_independent_d3c.json')


def register(run):
    if run.exists():raise ValueError('Fresh run required')
    config=read(CONFIG);mapping=read(config['mapping'])['positions'];known=known_history();sessions=[];seen=set()
    protected=read('outputs/scanning/fusion_d3_run1/inputs.json')['protected_sha256']
    for idx in ['data/metadata/fusion_d3_index_v1.json','data/metadata/fusion_reliability_d3b_index_v1.json']:
        protected.update(read(idx)['artifacts']);protected[idx]=sha256_file(idx)
    check_hashes(protected)
    structural={}
    for number,fen in mapping.items():
        if len(fen.split())!=6:raise ValueError('Complete FEN required')
        board=chess.Board(fen)
        structural[number]=dict(structurally_valid=True,standard_chess_valid=board.is_valid(),status=int(board.status()),
                                placement_fen=board.board_fen(),piece_count=len(board.piece_map()))
    files=sorted(p for p in Path(config['source_directory']).rglob('*') if p.suffix.lower() in ('.mov','.mp4'))
    if len(files)!=config['expected_sessions']:raise ValueError('Expected exactly ten uploaded recordings')
    for p in files:
        number=match_position(p,mapping)
        if number in seen:raise ValueError('Ambiguous multiple recordings for position '+number)
        seen.add(number);digest=sha256_file(p)
        photos=[x for x in p.parent.glob(p.stem+'.*') if x.suffix.lower() in ('.jpg','.jpeg','.png','.heic')]
        sessions.append(dict(session_id='d3c_pos_'+number,position_id='d3c_position_'+number,position_number=number,position_group_id='d3c_position_'+number,
            independence_group_id='living_room_midnight_artificial_20261026_owner_cohort',source=dict(path=str(p),sha256=digest,bytes=p.stat().st_size),
            fen=mapping[number],fen_validation=structural[number],reference_photos=[dict(path=str(x),sha256=sha256_file(x)) for x in photos],
            reference_agreement='pending_visual_review' if photos else 'unverified_missing_reference',
            unchanged_position=dict(status='owner_declared',source='D.3c assignment continuous unchanged physical position'),
            capture_conditions=dict(environment='living_room',time_of_day='approximately_midnight',lighting='artificial_indoor',date='unverified_until_native_metadata',
                viewpoints='owner_reported_varied_starting_sides_overhead_to_oblique',movement='owner_reported_clockwise_and_counterclockwise',precise_pose=None),
            known_source_overlap=digest in known['sources'],known_exact_placement_overlap=structural[number]['placement_fen'] in known['placements'],
            geometry_verification='pending',semantic_orientation_verification='pending'))
    if any(s['known_source_overlap'] for s in sessions):raise ValueError('Original pilot source reused')
    run.mkdir(parents=True)
    sources={str(p):sha256_file(p) for p in [CONFIG,Path(config['mapping']),Path(__file__),Path('src/scanning/interim_video.py'),Path('docs/evaluation/fusion_independent_d3c_design.md')]}
    write(run/'registration.json',dict(schema_version='limited_independent_cohort_d3c_v1',scope=config['scope'],strict_preregistered_gate_passed=False,
        reason=['one shared lighting/environment cohort','reference photographs absent or require review','per-frame human geometry verification pending'],
        config=config,source_sha256=sources,protected_sha256=protected,all_supplied_fen_structural_checks=structural,sessions=sessions,recognition_outcomes_inspected=False))
    print('Registered ten sessions before recognition. All 20 FENs parsed.',flush=True)


def geometry_record(image,frame,config):
    start=time.perf_counter();g=dict(completed=False,failure_stage=None,geometry_correctness='unverified',semantic_orientation_correctness='unverified')
    with tempfile.TemporaryDirectory(prefix='chess_d3c_') as tmp:
        d=Path(tmp);path=d/'input.png';cv2.imwrite(str(path),image)
        try:
            rectified,grid,turns,prov=prepare_reliable_raw_photo(path,config,d)
            inverse=np.linalg.inv(context_transform(prov['source_corners'],rectified.board_bounds))
            polygons=project(grid['square_polygons'],inverse)
            g.update(completed=True,provenance=prov,rotation_quarters_ccw=turns,source_square_polygons=polygons.tolist(),
                semantic_square_ids=[chess_square_at(r,c,turns) for r in range(8) for c in range(8)],
                predicted_camera_orientation=['white_near_camera','white_left','black_near_camera','white_right'][turns],
                legacy_completed=prov['geometry_reliability']['path']=='legacy')
        except PreparationError as error:g.update(failure_stage=error.stage,error=str(error),legacy_completed=False)
        except (ValueError,RuntimeError,OSError,ImportError) as error:g.update(failure_stage='rectification_validation' if 'Corners must be ordered' in str(error) else 'processing_exception',error=type(error).__name__+': '+str(error),legacy_completed=False)
        g['recovery_attempted']=(d/'recovery.json').exists()
        for name,file in [('legacy_detection','legacy/detection.json'),('recovery','recovery.json')]:
            if (d/file).exists():
                x=read(d/file)
                if name=='legacy_detection':g[name]=dict(success=x.get('success'),corners=x.get('corners'),reason=x.get('reason'),leading_candidates=x.get('diagnostics',{}).get('top_candidates',[])[:1])
                else:g[name]={k:x.get(k) for k in ('status','reason','rotation_quarters_ccw','grid_evidence_fraction','orientation_source')}
    g['elapsed_seconds']=time.perf_counter()-start
    return g


def process_session(args):
    run,s=args;run=Path(run);dest=run/s['session_id'];dest.mkdir(exist_ok=True)
    settings=read('configs/experiments/keyframes_d1.json');config=load_config()
    cv2.setNumThreads(1);start=time.perf_counter();source=s['source']['path']
    if sha256_file(source)!=s['source']['sha256']:raise ValueError('Source changed')
    if (dest/'index.json.gz').exists():
        index=json.loads(gzip.decompress((dest/'index.json.gz').read_bytes()));meta=index['metadata'];timeline=index['frames']
    else:
        meta,timeline=index_video(source,s['session_id']);write_gzip(dest/'index.json.gz',dict(metadata=meta,frames=timeline))
    candidates=temporal_candidates(timeline,settings['interval_ms'],settings['candidate_limit']);records=[]
    for number,f in enumerate(candidates):
        target=dest/f'candidate_{number:03d}.json'
        if target.exists():records.append(read(target));continue
        image=decode_exact(source,f);f=dict(f,session_id=s['session_id'],source_video_sha256=s['source']['sha256'],
            derivative_sha256=pixel_hash(image),decoded_pixel_sha256=pixel_hash(image))
        g=geometry_record(image,f,config)
        binding={k:f[k] for k in ('session_id','frame_id','pts','time_base','source_video_sha256','derivative_sha256')}
        state,reasons=geometry_state(f,g,binding);row=dict(f,geometry=g,geometry_state=state,unusable_reasons=reasons,eligible=False,quality_score=None)
        if state=='completed_unverified':
            try:
                row.update(board_features(image,g,settings));row['eligible']=not row['quality_rejection_reasons'];row['unusable_reasons']+=row['quality_rejection_reasons']
            except ValueError as e:row.update(geometry_state='unresolved',unusable_reasons=['selector_invalid_outer_boundary',str(e)])
        write(target,row);records.append(row)
        print(s['session_id'],number+1,'/',len(candidates),state,'legacy',g['legacy_completed'],flush=True)
    result=select_keyframes(records,'uniform',5,settings)
    selected={r['frame_id']:r for r in records};reviews=[]
    for i,fid in enumerate(result['selected']):
        f=selected[fid];image=decode_exact(source,f)
        if pixel_hash(image)!=f['decoded_pixel_sha256']:raise ValueError('Original selected pixels do not reproduce')
        overlay=raw_grid_overlay(image,np.float32(f['geometry']['source_square_polygons']),f['geometry']['semantic_square_ids'])
        scale=min(1.,1200/max(overlay.shape[:2]));overlay=cv2.resize(overlay,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
        name=f'review_{i+1:02d}.jpg';cv2.imwrite(str(dest/name),overlay,[cv2.IMWRITE_JPEG_QUALITY,90])
        reviews.append(dict(review_id=s['session_id']+f'_V{i+1}',frame_id=fid,pts=f['pts'],time_base=f['time_base'],timestamp_ms=f['timestamp_ms'],
            source_video_sha256=s['source']['sha256'],pixel_sha256=f['decoded_pixel_sha256'],overlay_path=str(dest/name),overlay_sha256=sha256_file(dest/name),
            human_review='pending',geometry_status='completed_unverified',predicted_orientation=f['geometry']['predicted_camera_orientation']))
    final=dict(session_id=s['session_id'],source=s['source'],metadata=meta,candidate_count=len(records),geometry_completed=sum(r['geometry']['completed'] for r in records),
        selector_eligible=sum(r['eligible'] for r in records),legacy_completed=sum(r['geometry']['legacy_completed'] for r in records),
        failures=dict(Counter(r['geometry']['failure_stage'] for r in records if not r['geometry']['completed'])),selection=result,reviews=reviews,
        elapsed_seconds=time.perf_counter()-start,adapter_source_sha256=sha256_file(Path(__file__)))
    write(dest/'session.json',final);return final


def geometry(run,workers):
    registration=read(run/'registration.json');check_hashes(registration['protected_sha256'])
    todo=[(str(run),s) for s in registration['sessions'] if not (run/s['session_id']/'session.json').exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(process_session,todo):print('SESSION DONE',result['session_id'],result['geometry_completed'],'/',result['candidate_count'],flush=True)
    sessions=[read(run/s['session_id']/'session.json') for s in registration['sessions']]
    write(run/'geometry_summary.json',dict(sessions=sessions,review_status='pending',source_sha256=sha256_file(Path(__file__))))
    html=['<!doctype html><meta charset="utf-8"><h1>D.3c selected geometry review</h1><p>Completion is not correctness. Verify actual playing corners, semantic orientation, canonical labels and crop alignment. No piece predictions are shown.</p>']
    for s in sessions:
        html.append('<h2>'+s['session_id']+'</h2>')
        for f in s['reviews']:
            rel=Path(f['overlay_path']).relative_to(run);html.append(f'<h3>{f["review_id"]} {f["timestamp_ms"]/1000:.3f}s {f["predicted_orientation"]}</h3><img style="max-width:100%;width:900px" src="{rel}">')
    (run/'geometry_review.html').write_text(''.join(html))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['register','geometry']);p.add_argument('--run',type=Path,required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    if a.stage=='register':register(a.run)
    else:geometry(a.run,a.workers)
