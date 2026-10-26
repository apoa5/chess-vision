"""Frozen inference on exact original selected PTS; compact provisional D.3c evidence."""
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
from scripts.run_square_evidence import make_inputs
from src.scanning.interim_video import decode_exact,pixel_hash,write_gzip
from src.scanning.interim_evidence import fuse_provisional_board
from src.scanning.native_pilot import sha256_file
from src.scanning.square_evidence import uncertainty,digest_json
from src.chess.labels import CLASS_NAMES
from src.vision.crop_experiments import tensor_from_packed


def run(run_dir):
    output=run_dir/'recognition'
    if output.exists():raise ValueError('Fresh recognition directory required')
    reg=read(run_dir/'registration.json');summary=read(run_dir/'geometry_summary.json');cfg=read('configs/experiments/evidence_d2.json');fusion=read('configs/experiments/fusion_d3.json')
    check_hashes(reg['protected_sha256']);check_hashes({s['source']['path']:s['source']['sha256'] for s in reg['sessions']})
    check_hashes({cfg['checkpoint']:cfg['checkpoint_sha256']})
    output.mkdir();sources={p:sha256_file(p) for p in ['scripts/recognize_independent_d3c.py','src/scanning/interim_evidence.py','src/scanning/interim_video.py','scripts/run_square_evidence.py','configs/experiments/evidence_d2.json','configs/experiments/fusion_d3.json']}
    write(output/'inference_registration.json',dict(source_sha256=sources,cohort_sha256=sha256_file(run_dir/'registration.json'),geometry_summary_sha256=sha256_file(run_dir/'geometry_summary.json'),
        geometry_review_waiver='explicit provisional-only comparison, human_verified status never assigned',strict_eligible_sessions=0,ground_truth_used_for_inference=False))
    import torch
    from src.vision.inference import PieceClassifier
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(1)
    classifier=PieceClassifier(cfg['checkpoint'],batch_size=16,cpu_threads=4)
    checkpoint=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    if checkpoint.get('crop_spec')!=cfg['crop_spec']:raise ValueError('Crop specification changed')
    allobs=[];predictions=[];sessions=[];start=time.perf_counter()
    for session,g in zip(reg['sessions'],summary['sessions']):
        if session['session_id']!=g['session_id']:raise ValueError('Session order mismatch')
        sid=session['session_id'];folder=run_dir/sid
        lookup={}
        for p in sorted(folder.glob('candidate_*.json')):
            r=read(p);lookup[r['frame_id']]=r
        obs=[];errors=[];frames=[]
        selected=[lookup[f] for f in g['selection']['selected']]
        for f in selected:
            t=time.perf_counter();image=decode_exact(session['source']['path'],f)
            if pixel_hash(image)!=f['decoded_pixel_sha256']:raise ValueError('Selected original pixels changed')
            try:inputs=make_inputs(image,f['geometry'],cfg)
            except ValueError as e:errors.append(dict(frame_id=f['frame_id'],stage='crop_preprocessing',error=str(e)));continue
            packed=[x['packed'] for x in inputs];tensors=torch.from_numpy(np.stack([tensor_from_packed(x) for x in packed]));zs=[];forward=time.perf_counter()
            with torch.inference_mode():
                for first in range(0,64,16):zs.extend(classifier.model(tensors[first:first+16]).tolist())
            forward=time.perf_counter()-forward
            for i,(inp,z) in enumerate(zip(inputs,zs)):
                u=uncertainty(z);identity=dict(session_id=sid,position_id=session['position_id'],source_video_sha256=session['source']['sha256'],
                    frame_id=f['frame_id'],canonical_square=inp['canonical_square'],checkpoint_sha256=cfg['checkpoint_sha256'],crop_config_sha256=sha256_file('configs/experiments/evidence_d2.json'))
                obs.append(dict(**identity,observation_id=digest_json(identity),original_pts=f['pts'],time_base=f['time_base'],pts_time=f['pts_time'],timestamp_ms=f['timestamp_ms'],
                    geometry_verification_status='completed_unverified',semantic_orientation=f['geometry']['predicted_camera_orientation'],rotation_quarters_ccw=f['geometry']['rotation_quarters_ccw'],
                    geometry_policy='marked-board',class_names=list(CLASS_NAMES),raw_logits=z,**u,predicted_class=CLASS_NAMES[u['predicted_class_id']],
                    calibration_status='not_available',calibrated_probabilities=None,condition_memberships=[dict(condition='uniform_5',selection_budget=5,keyframe_strategy='uniform')],
                    d1_selection_quality_metadata={k:f[k] for k in ('quality_score','quality_components','view_descriptor')},input_index=i,
                    input_packed_sha256=hashlib.sha256(inp['packed'].tobytes()).hexdigest(),target_valid_fraction=inp['target_valid_fraction']))
            frames.append(dict(frame_id=f['frame_id'],timestamp_ms=f['timestamp_ms'],original_pts=f['pts'],time_base=f['time_base'],forward_seconds=forward,total_seconds=time.perf_counter()-t))
            # At most one frame/session is retained as 64 small crop thumbnails,
            # sufficient for compact failure diagnosis, not a training dataset.
            if not (output/(sid+'_crop_contact.jpg')).exists():
                sheet=np.full((8*118,8*118,3),245,np.uint8)
                for inp in inputs:
                    square=inp['canonical_square'];r=8-int(square[1]);c='abcdefgh'.index(square[0]);bgr=cv2.cvtColor(inp['packed'][:3].transpose(1,2,0),cv2.COLOR_RGB2BGR)
                    contours,_=cv2.findContours(inp['packed'][3].astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(bgr,contours,-1,(30,255,30),2)
                    thumb=cv2.resize(bgr,(112,100),interpolation=cv2.INTER_AREA);y=r*118;x=c*118;sheet[y+16:y+116,x:x+112]=thumb
                    cv2.putText(sheet,square,(x+3,y+12),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),1)
                cv2.imwrite(str(output/(sid+'_crop_contact.jpg')),sheet,[cv2.IMWRITE_JPEG_QUALITY,90])
            print(sid,f['frame_id'],'64 original-pixel inputs',flush=True)
        if len(frames)==5:
            for method in reg['config']['methods']:
                pred=fuse_provisional_board(obs,method,fusion,explicit_review_waiver=True)
                if pred!=fuse_provisional_board(obs[::-1],method,fusion,explicit_review_waiver=True):raise ValueError('Nondeterministic fusion')
                predictions.append(pred)
        sessions.append(dict(session_id=sid,nominated_frames=len(selected),inferred_frames=len(frames),errors=errors,frames=frames,
            comparison_eligible=len(frames)==5,verification='provisional_unverified_geometry_and_photo_agreement'))
        allobs.extend(obs)
    write_gzip(output/'observations.json.gz',dict(class_names=list(CLASS_NAMES),observations=allobs))
    write_gzip(output/'predictions.json.gz',dict(boards=predictions,ground_truth_used_for_predictions=False,strict_preregistered_gate_passed=False))
    write(output/'sessions.json',dict(sessions=sessions,observation_count=len(allobs),elapsed_seconds=time.perf_counter()-start,source_sha256=sources,
        model_checkpoint=cfg['checkpoint'],model_sha256=cfg['checkpoint_sha256'],classifier_configuration=cfg,provisional_only=True))
    check_hashes(reg['protected_sha256']);print('Saved compact evidence; provisional comparisons only.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();run(a.run)
