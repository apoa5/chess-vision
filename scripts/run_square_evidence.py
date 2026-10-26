"""Extract frozen B.2 per-square logits after D.2 human geometry verification."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from scripts.select_video_keyframes import read,write,check_hashes
from src.chess.labels import CLASS_NAMES
from src.scanning.native_pilot import sha256_file
from src.scanning.square_evidence import (digest_json,verify_reviews,uncertainty,validate_observation,square_records,validate_bundle)
from src.scanning.video_keyframes import geometry_state
from src.vision.crop_experiments import CropSpec,packed_input,tensor_from_packed
from src.vision.geometry_audit import context_transform,project
from src.vision.grid_geometry import polygon_crop
from src.vision.perspective import warp_board_with_context


def make_inputs(image,geometry,config):
    """Reuse the frozen D.1 polygons and B.2 transforms, with camera-facing RGB."""
    corners=geometry['provenance']['source_corners']
    board=warp_board_with_context(image,np.float32(corners),config['board_size'],config['context_margin'])
    matrix=context_transform(corners,board.board_bounds)
    polygons=project(geometry['source_square_polygons'],matrix)
    spec=CropSpec(**config['crop_spec']);result=[]
    for square,polygon in zip(geometry['semantic_square_ids'],polygons):
        crop,coverage,_,bounds,local,valid=polygon_crop(board.image,board.valid_mask,polygon,config['initial_context_padding'])
        if valid<.99:raise ValueError('Verified target clipped: '+square)
        packed=packed_input(crop,local,spec)
        result.append(dict(canonical_square=square,packed=packed,original_context_bounds=list(bounds),
                           original_target_polygon=np.asarray(local).tolist(),target_valid_fraction=float(valid),
                           context_valid_fraction=float((coverage>0).mean())))
    if len(result)!=64 or len({r['canonical_square'] for r in result})!=64:raise ValueError('One input per canonical square required')
    return result


def run(packet_path,reviews_path,output):
    if output.exists():raise ValueError('Fresh output directory required')
    begin=time.perf_counter();packet=read(packet_path);reviews=read(reviews_path)
    statuses=verify_reviews(packet,reviews,sha256_file(packet_path))  # Fail before model load or pixels.
    config=read(packet['config_path'])
    if sha256_file(packet['config_path'])!=packet['config_sha256']:raise ValueError('Configuration changed after review preparation')
    if (config['fusion_enabled'] or config['completion_enabled'] or config['final_holdout'] or
        config['calibration']!={'status':'not_available','artifact':None,'fit_on_pilot':False}):raise ValueError('Unapproved experiment scope/calibration')
    if sha256_file(config['checkpoint'])!=config['checkpoint_sha256']:raise ValueError('B.2 model changed')
    check_hashes(packet['protected_sha256'])
    import torch
    from src.vision.inference import PieceClassifier
    torch.set_num_threads(config['cpu_threads']);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    checkpoint=torch.load(config['checkpoint'],map_location='cpu',weights_only=True)
    if checkpoint.get('crop_spec')!=config['crop_spec']:raise ValueError('Crop spec must match approved B.2 checkpoint')
    classifier=PieceClassifier(config['checkpoint'],batch_size=config['batch_size'],cpu_threads=config['cpu_threads'])
    output.mkdir(parents=True,exist_ok=False);(output/'inputs').mkdir();obs=[];frames=[];sessions={}
    model_identity=dict(checkpoint_path=config['checkpoint'],checkpoint_sha256=config['checkpoint_sha256'],class_names=list(CLASS_NAMES),
        architecture=checkpoint['architecture'],epoch=checkpoint['epoch'],crop_spec=config['crop_spec'],calibration_status='not_available',
        raw_softmax_computation='stable float64 softmax of unchanged float32 model logits',
        source_sha256={p:sha256_file(p) for p in ['src/vision/crop_experiments.py','src/vision/dataset_loader.py','src/vision/inference.py','src/vision/grid_geometry.py','src/vision/perspective.py','src/scanning/square_evidence.py',str(Path(__file__))]})
    feature_path=Path(config['conditions'][0]['manifest']).parents[5]/'features.json'
    feature_key=str(feature_path)
    if feature_key not in packet['protected_sha256'] or sha256_file(feature_path)!=packet['protected_sha256'][feature_key]:raise ValueError('D.1 features changed')
    features=read(feature_path)
    feature_lookup={(s['capture_session_id'],r['frame_id']):r for d in features['datasets'] if d['name']=='primary_3s_high'
                    for s in d['sessions'] for r in s['policies']['marked-board']}
    model_identity['d1_feature_reference']=feature_key;model_identity['d1_features_sha256']=sha256_file(feature_path)
    frame_start=time.perf_counter()
    for row in packet['frames']:
        sid=row['session']['capture_session_id'];f=row['frame'];fid=f['frame_id'];state=statuses[row['review_id']]
        frame=dict(review_id=row['review_id'],session_id=sid,position_id=row['session']['position_id'],frame_id=fid,
            timestamp_ms=f['timestamp_ms'],conditions=row['conditions'],**state,geometry_reference=f['geometry_reference'],
            d1_metadata=f,observations=[],extraction_seconds=None,forward_seconds=None,normalization_seconds=None)
        frames.append(frame);sessions[sid]=row['session']
        if state['human_review']!='human_verified':continue
        t=time.perf_counter()
        if sha256_file(f['path'])!=f['derivative_sha256'] or sha256_file(f['geometry_reference'])!=f['geometry_reference_sha256']:raise ValueError('Review inputs changed')
        raw=read(f['geometry_reference']);g=raw['policies']['marked-board']
        b={k:row['binding'][k] for k in ('session_id','frame_id','pts','time_base','source_video_sha256','derivative_sha256')}
        if raw['binding']!=b or geometry_state(dict(f,session_id=sid),g,b)[0]!='completed_unverified':raise ValueError('Geometry structural gate failed')
        if row['source_square_polygons']!=g['source_square_polygons']:raise ValueError('Reviewed grid changed')
        image=cv2.imread(f['path']);inputs=make_inputs(image,g,config)
        packed=np.stack([x['packed'] for x in inputs]);cache=output/'inputs'/(fid+'.npz');np.savez_compressed(cache,packed=packed)
        frame['input_cache']=str(cache);frame['input_cache_sha256']=sha256_file(cache);frame['extraction_seconds']=time.perf_counter()-t
        t=time.perf_counter();tensors=torch.from_numpy(np.stack([tensor_from_packed(x) for x in packed]));frame['normalization_seconds']=time.perf_counter()-t;logits=[];t=time.perf_counter()
        with torch.inference_mode():
            for first in range(0,64,config['batch_size']):logits.extend(classifier.model(tensors[first:first+config['batch_size']]).tolist())
        frame['forward_seconds']=time.perf_counter()-t
        memberships=[]
        for condition in row['conditions']:
            original=next(c for c in packet['conditions'] if c['name']==condition);selection=read(original['manifest'])
            if sha256_file(original['manifest'])!=original['manifest_sha256']:raise ValueError('D.1 selection changed')
            selected_session=next(s for s in selection['sessions'] if s['capture_session_id']==sid)
            trace=next(x for x in selected_session['selection_trace'] if x['frame_id']==fid)
            memberships.append(dict(condition=condition,keyframe_strategy=selection['strategy'],selection_budget=selection['budget'],
                selected_manifest=original['manifest'],selected_manifest_sha256=original['manifest_sha256'],selection_trace=trace))
        for i,(inp,z) in enumerate(zip(inputs,logits)):
            square=inp['canonical_square'];values=uncertainty(z)
            identity=dict(session_id=sid,source_video_sha256=f['source_video_sha256'],frame_id=fid,canonical_square=square,
                          checkpoint_sha256=config['checkpoint_sha256'],crop_config_sha256=packet['config_sha256'])
            observation=dict(identity,observation_id=digest_json(identity),position_id=row['session']['position_id'],
                position_group_id=row['session']['position_group_id'],independence_group_id=row['session']['independence_group_id'],
                source_video_path=row['session']['source_video']['path'],original_pts=f['pts'],time_base=f['time_base'],pts_time=f['pts_time'],
                timestamp_ms=f['timestamp_ms'],relative_timestamp_ms=f['relative_timestamp_ms'],frame_path=f['path'],frame_sha256=f['derivative_sha256'],
                condition_memberships=memberships,d1_selection_quality_metadata=f,
                d1_raw_quality_and_view={k:feature_lookup[(sid,fid)][k] for k in ('board_sharpness','board_contrast_p90_p10','extreme_luma_fraction',
                    'board_area_fraction','edge_ratio_proxy','direction_anisotropy','view_descriptor','appearance_dhash','image_plane_file_axis_degrees')},
                geometry_policy='marked-board',geometry_completed=True,geometry_verification_status='human_verified',
                geometry_reference=f['geometry_reference'],geometry_reference_sha256=f['geometry_reference_sha256'],
                human_review_id=row['review_id'],human_review_artifact=str(reviews_path),human_review_sha256=sha256_file(reviews_path),
                semantic_orientation=g['predicted_camera_orientation'],rotation_quarters_ccw=g['rotation_quarters_ccw'],
                raw_logits=z,**values,predicted_class=CLASS_NAMES[values['predicted_class_id']],class_names=list(CLASS_NAMES),
                calibrated_probabilities=None,calibration_status='not_available',calibration_artifact=None,
                crop_spec=config['crop_spec'],input_cache=str(cache),input_index=i,input_packed_sha256=hashlib.sha256(inp['packed'].tobytes()).hexdigest(),
                visibility='unknown',elevation_degrees=None,
                **{k:inp[k] for k in ('original_context_bounds','original_target_polygon','target_valid_fraction','context_valid_fraction')})
            validate_observation(observation);obs.append(observation);frame['observations'].append(observation['observation_id'])
        print(fid,'64 observations',flush=True)
    containers=[]
    for sid,meta in sessions.items():
        these=[o for o in obs if o['session_id']==sid]
        containers.append(dict(meta,squares=square_records(these)))
    manifest=dict(schema_version='offline_square_evidence_d2_v1',designation='development_only',training_eligible=False,final_holdout=False,
        packet_path=str(packet_path),packet_sha256=sha256_file(packet_path),reviews_path=str(reviews_path),reviews_sha256=sha256_file(reviews_path),
        config_path=packet['config_path'],config_sha256=packet['config_sha256'],conditions=packet['conditions'],model=model_identity,
        geometry_gate='explicit human review; no substitutions',sessions=containers,frames=frames,
        observations_path=str(output/'observations.jsonl'),observation_count=len(obs),unique_inferred_frames=sum(bool(f['observations']) for f in frames),
        calibration_applied=False,fusion_implemented=False,completion_enabled=False,ground_truth_used_for_inference=False,
        inference_loop_seconds=time.perf_counter()-frame_start,elapsed_before_serialization_seconds=time.perf_counter()-begin,
        environment=dict(python=sys.version,torch=torch.__version__,opencv=cv2.__version__,numpy=np.__version__,cpu_threads=config['cpu_threads']))
    with (output/'observations.jsonl').open('w') as h:
        for o in obs:h.write(json.dumps(o,allow_nan=False)+'\n')
    manifest['observations_sha256']=sha256_file(output/'observations.jsonl');validate_bundle(manifest,obs);write(output/'evidence.json',manifest)
    check_hashes(packet['protected_sha256']);write(output/'preservation.json',dict(protected_files_unchanged=len(packet['protected_sha256']),checkpoint_unchanged=True))
    write(output/'run_summary.json',dict(elapsed_including_final_hash_check_seconds=time.perf_counter()-begin,
        storage_bytes_excluding_this_summary=sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
        checkpoint_bytes=Path(config['checkpoint']).stat().st_size))
    print('Stored',len(obs),'unique observations')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True);p.add_argument('--reviews',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();run(a.packet,a.reviews,a.output_dir)
