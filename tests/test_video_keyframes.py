"""D.1 safeguards and explainable-selection regressions; synthetic data are not GT."""
import copy
from fractions import Fraction
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.scanning.video_keyframes import (STRATEGIES, board_features, geometry_state,
    near_duplicate, select_keyframes, selection_metrics, temporal_candidates, validate_settings)
from src.scanning.native_pilot import pts_record


def settings():return json.loads(Path('configs/experiments/keyframes_d1.json').read_text())


def geometry_fixture():
    ids=[f'{f}{r}' for r in range(8,0,-1) for f in 'abcdefgh']
    polys=[]
    for r in range(8):
        for col in range(8):
            x=200+64*col;y=200+64*r;polys.append([[x,y],[x+64,y],[x+64,y+64],[x,y+64]])
    g=dict(completed=True,rotation_quarters_ccw=0,semantic_square_ids=ids,
           source_square_polygons=polys,predicted_camera_orientation='white_near_camera')
    f=dict(session_id='s',frame_id='f',pts=20,time_base=[1,600],source_video_sha256='v',
           derivative_sha256='i',width=1024,height=1024)
    return f,g,dict((k,f[k]) for k in ('session_id','frame_id','pts','time_base','source_video_sha256','derivative_sha256'))


def row(i,angle=0,score=.9):
    return dict(frame_id=str(i),relative_timestamp_ms=i*3000,eligible=True,geometry_state='completed_unverified',
                quality_score=score,view_descriptor=[angle,0,0,0],appearance_dhash=f'{np.random.default_rng(i).integers(0,2**63):016x}',
                decoded_pixel_sha256='pixels_'+str(i),predicted_orientation=str(round(angle/(math.pi/2))%4),
                edge_ratio_proxy=.8,unusable_reasons=[])


def test_native_sampling_preserves_exact_irregular_pts_and_is_deterministic():
    frames=[dict(frame_id=str(i),**pts_record(p,Fraction(1,600),Fraction(1,6))) for i,p in enumerate([100,120,160,720,1300,1900])]
    a=temporal_candidates(frames,1000)
    assert a==temporal_candidates(copy.deepcopy(frames),1000)
    assert [r['pts'] for r in a]==[100,720,1300,1900]
    assert a[1]['timestamp_ms'] == 1200  # Actual PTS, not index / nominal FPS.
    with pytest.raises(ValueError):temporal_candidates(frames,1,limit=2)
    with pytest.raises(ValueError):temporal_candidates(frames[::-1],1000)


@pytest.mark.parametrize('field,value',[('pts',21),('source_video_sha256','other'),('derivative_sha256','other'),('session_id','other'),('time_base',[1,30])])
def test_stale_or_other_frame_geometry_is_unusable(field,value):
    f,g,b=geometry_fixture();b[field]=value
    assert geometry_state(f,g,b)[0]=='invalid_geometry_binding'


def test_unreviewed_completion_is_not_verified_and_reviews_cannot_repair_failures():
    f,g,b=geometry_fixture();assert geometry_state(f,g,b)[0]=='completed_unverified'
    review=dict(binding=b,status='visually_verified',reviewer='test reviewer',reviewer_kind='human')
    assert geometry_state(f,g,b,review)[0]=='visually_verified'
    g.update(completed=False,failure_stage='dependency_or_runtime',error='ValueError: Corners must be ordered TL, TR, BR, BL')
    state,reasons=geometry_state(f,g,b,review)
    assert state=='unresolved' and reasons[0]=='rectification_validation'


def test_incorrect_geometry_review_is_distinct_and_requires_exact_binding():
    f,g,b=geometry_fixture();review=dict(binding=b,status='rejected_incorrect')
    assert geometry_state(f,g,b,review)[0]=='rejected_incorrect'
    review['binding']={}
    with pytest.raises(ValueError):geometry_state(f,g,b,review)


@pytest.mark.parametrize('defect',['bounds','ids','nan','orientation','polygon'])
def test_completed_flag_cannot_override_invalid_geometry(defect):
    f,g,b=geometry_fixture()
    if defect=='bounds':g['source_square_polygons'][0][0][0]=-10
    if defect=='ids':g['semantic_square_ids'][0]='h1'
    if defect=='nan':g['source_square_polygons'][0][0][0]=float('nan')
    if defect=='orientation':g['rotation_quarters_ccw']=None
    if defect=='polygon':g['source_square_polygons'][0]=[[1,1]]*4
    assert geometry_state(f,g,b)[0]=='unresolved'


@pytest.mark.parametrize('strategy',STRATEGIES)
def test_identical_repeats_are_not_selected_to_fill_budget(strategy):
    rows=[row(i) for i in range(25)]
    for r in rows:r['decoded_pixel_sha256']='same';r['appearance_dhash']='a'
    assert len(select_keyframes(rows,strategy,20,settings())['selected'])==1


@pytest.mark.parametrize('strategy',STRATEGIES)
def test_invalid_state_or_quality_never_selected_even_if_eligible_flag_is_wrong(strategy):
    rows=[row(0),row(1),row(2)];rows[0]['geometry_state']='unresolved';rows[1]['geometry_state']='rejected_incorrect';rows[2]['quality_score']=float('nan')
    s=select_keyframes(rows,strategy,5,settings());assert not s['selected']
    assert all(r['reasons'] for r in s['candidate_decisions'])


def test_geometrically_different_useful_view_is_not_rejected_by_dhash_alone():
    a,b=row(0,0),row(1,math.pi);b['appearance_dhash']=a['appearance_dhash']
    assert not near_duplicate(a,b,settings())


def test_diversity_avoids_quality_only_side_collapse_with_matched_budget():
    rows=[row(i,0,.99) for i in range(8)]+[row(8,math.pi/2),row(9,math.pi),row(10,-math.pi/2)]
    pure=select_keyframes(rows,'quality',4,settings());diverse=select_keyframes(rows,'quality_diverse',4,settings())
    assert selection_metrics(rows,pure,settings())['selected_predicted_orientations']==1
    assert selection_metrics(rows,diverse,settings())['selected_predicted_orientations']==4
    assert len(pure['selected'])==len(diverse['selected'])==4


def test_temporal_separation_can_legitimately_underfill_budget():
    rows=[row(i) for i in range(8)];s=select_keyframes(rows,'quality_temporal',20,settings())
    assert len(s['selected'])<8
    assert selection_metrics(rows,s,settings())['min_gap_seconds']>=5


@pytest.mark.parametrize('strategy',STRATEGIES)
def test_fen_labels_and_input_order_do_not_influence_selection(strategy):
    rows=[row(i,i/5) for i in range(12)];expected=select_keyframes(rows,strategy,5,settings())
    changed=copy.deepcopy(rows)
    for r in changed:r['fen']='not even a valid FEN';r['piece_classes']=['random']
    assert select_keyframes(changed[::-1],strategy,5,settings())==expected


def test_board_quality_excludes_sharp_background_and_detects_blur():
    _,g,_=geometry_fixture();c=settings();rng=np.random.default_rng(4)
    image=rng.integers(0,255,(1024,1024,3),dtype=np.uint8);image[200:713,200:713]=128
    flat=board_features(image,g,c)
    assert flat['board_sharpness']<1 and flat['quality_rejection_reasons']
    image[200:713,200:713]=rng.integers(0,255,(513,513,3),dtype=np.uint8)
    sharp=board_features(image,g,c);blur=board_features(cv2.GaussianBlur(image,(19,19),6),g,c)
    assert sharp['board_sharpness']>blur['board_sharpness']*10
    assert sharp['elevation_degrees'] is None and sharp['visibility']=='unknown_occlusion'


@pytest.mark.parametrize('key,value',[('interval_ms',0),('candidate_limit',-1),('min_sharpness',float('nan')),('budgets',[0]),('diversity_weight',2)])
def test_bad_experiment_settings_rejected(key,value):
    c=settings();c[key]=value
    with pytest.raises(ValueError):validate_settings(c)


@pytest.mark.parametrize('change',[{'quality_weights':{'sharpness':float('nan')}},{'quality_warp_size':512.5},{'development_only':False},{'geometry_policies':['marked-board']}])
def test_settings_cannot_silently_change_scope_or_score_contract(change):
    c=settings();c.update(change)
    with pytest.raises(ValueError):validate_settings(c)
