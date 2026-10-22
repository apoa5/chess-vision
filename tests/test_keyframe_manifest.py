"""Selected-manifest integrity failures must block downstream use."""
import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from scripts.validate_keyframe_manifest import validate_manifest
from src.scanning.native_pilot import sha256_file


@pytest.fixture
def manifest(tmp_path):
    def write(name,obj):
        p=tmp_path/name;p.write_text(json.dumps(obj));return str(p),sha256_file(p)
    video=tmp_path/'video';video.write_bytes(b'synthetic source');vsha=sha256_file(video)
    p=tmp_path/'frame.png';im=np.zeros((1024,1024,3),np.uint8);cv2.imwrite(str(p),im)
    b=dict(session_id='s',frame_id='f',pts=600,time_base=[1,600],source_video_sha256=vsha,derivative_sha256=sha256_file(p))
    polys=[]
    for r in range(8):
        for c in range(8):
            x=200+c*64;y=200+r*64;polys.append([[x,y],[x+64,y],[x+64,y+64],[x,y+64]])
    g=dict(completed=True,rotation_quarters_ccw=0,semantic_square_ids=[f'{f}{r}' for r in range(8,0,-1) for f in 'abcdefgh'],source_square_polygons=polys)
    gp,gh=write('geometry.json',dict(binding=b,geometry_config_sha256='config',evaluator_sha256='eval',policies={'legacy':g}))
    source=dict(path=str(video),sha256=vsha);groups=dict(capture_session_id='s',position_id='p',position_group_id='pg',independence_group_id='ig')
    cp,ch=write('catalog.json',dict(sessions=[dict(groups,primary_source=source)]))
    row=dict(b,pts_time=[1,1],timestamp_ms=1000,relative_timestamp_ms=1000,path=str(p),width=1024,height=1024,
        decoded_pixel_sha256=hashlib.sha256(im.tobytes()).hexdigest(),quality_score=.8,view_descriptor=[0,0,0,0],
        geometry_reference=gp,geometry_reference_sha256=gh,geometry_state='completed_unverified',review_annotation=None)
    row.pop('session_id')
    settings=json.load(open('configs/experiments/keyframes_d1.json'))
    return dict(schema_version='offline_selected_keyframes_d1_v1',dataset='primary_3s_high',source_catalog=cp,source_catalog_sha256=ch,
        designation='development_only',split='pilot',training_eligible=False,final_holdout=False,ground_truth_FEN_used_for_selection=False,
        piece_recognition_evaluated=False,per_square_evidence_or_fusion_implemented=False,settings=settings,budget=5,policy='legacy',
        D2_input_role='primary_candidates',geometry_config_sha256='config',geometry_evaluator_sha256='eval',sessions=[dict(groups,source_video=source,selected=[row])])


def test_valid_manifest_retains_unverified_state(manifest):
    assert validate_manifest(manifest)==dict(sessions=1,selected_frames=1,valid=True)


@pytest.mark.parametrize('defect',['timestamp','source','group','verified','nan','pixels','budget','scope','role','geometry_hash'])
def test_tampering_is_rejected(manifest,defect):
    m=copy.deepcopy(manifest);s=m['sessions'][0];r=s['selected'][0]
    if defect=='timestamp':r['pts']+=1
    if defect=='source':r['source_video_sha256']='wrong'
    if defect=='group':s['position_group_id']='other'
    if defect=='verified':r['geometry_state']='visually_verified'
    if defect=='nan':r['view_descriptor'][0]=float('nan')
    if defect=='pixels':r['decoded_pixel_sha256']='wrong'
    if defect=='budget':s['selected']*=6
    if defect=='scope':m['final_holdout']=True
    if defect=='role':m['D2_input_role']='robustness_comparison_only'
    if defect=='geometry_hash':r['geometry_reference_sha256']='wrong'
    with pytest.raises(ValueError):validate_manifest(m)
