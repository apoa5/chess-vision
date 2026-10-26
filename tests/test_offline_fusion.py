"""D.3 ground-truth-blind fusion invariants and uncertainty regressions."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from src.chess.labels import CLASS_NAMES
from src.scanning.square_evidence import SQUARES,uncertainty
from src.scanning.offline_fusion import METHODS,fuse_square,fuse_board,groups,choose_single_frame,validate_config
from src.scanning.fusion_metrics import reliability,legality


def config():return json.loads(Path('configs/experiments/fusion_d3.json').read_text())


def observation(i,p=None,angle=None,time=None,square='a8',quality=.8):
    p=np.asarray(p if p is not None else [.76]+[.02]*12,dtype=float);p=p/p.sum();z=np.log(np.maximum(p,1e-100));u=uncertainty(z)
    return dict(observation_id=f'o{i}_{square}',position_id='pos',session_id='s',source_video_sha256='a'*64,canonical_square=square,
        checkpoint_sha256='b'*64,crop_config_sha256='c'*64,frame_id='f'+str(i),original_pts=i*600+1,time_base=[1,600],pts_time=[i*600+1,600],
        timestamp_ms=(i*600+1)/600*1000 if time is None else time,
        geometry_verification_status='human_verified',class_names=list(CLASS_NAMES),predicted_class=CLASS_NAMES[u['predicted_class_id']],
        raw_logits=z.tolist(),**u,calibration_status='not_available',calibrated_probabilities=None,
        condition_memberships=[dict(selection_budget=5)],d1_selection_quality_metadata=dict(quality_score=quality,view_descriptor=[float(i if angle is None else angle),0,0,0]))


def row(i,p=None,**kwargs):
    o=observation(i,p,**kwargs)
    # Keep exact PTS consistent with explicit synthetic timestamps.
    if 'time' in kwargs:
        o['original_pts']=int(round(kwargs['time']*600));o['time_base']=[1,600000];o['pts_time']=[o['original_pts'],600000]
    return o


@pytest.mark.parametrize('method',METHODS)
def test_order_invariance_and_traceability(method):
    rows=[row(i,time=i*10000) for i in range(4)];a=fuse_square(rows,method,config(),single_frame_id='f1')
    assert a==fuse_square(rows[::-1],method,config(),single_frame_id='f1')
    assert len(a['score_distribution'])==13 and sum(a['score_distribution'])==pytest.approx(1)
    assert set(a['observation_ids'])=={r['observation_id'] for r in rows}
    assert not a['conditional_independence_assumed']


@pytest.mark.parametrize('method',METHODS)
def test_identical_repetition_does_not_sharpen_scores(method):
    p=[.4,.3]+[.3/11]*11;rows=[row(i,p,time=i*10000,angle=0) for i in range(7)]
    a=fuse_square(rows[:1],method,config(),single_frame_id='f0');b=fuse_square(rows,method,config(),single_frame_id='f0')
    assert np.allclose(a['score_distribution'],b['score_distribution'])
    assert b['heuristic_group_count']==1


def test_single_quality_is_one_frame_for_entire_canonical_board():
    obs=[]
    for i in range(3):
        for sq in SQUARES:obs.append(row(i,square=sq,quality=[.6,.9,.7][i],time=i*10000))
    b=fuse_board(obs,'single_quality',config());assert b['single_frame_id']=='f1'
    assert {s['single_observation_id'].split('_')[0] for s in b['squares']}=={'o1'}
    assert [s['canonical_square'] for s in b['squares']]==list(SQUARES)
    frames=[dict(frame_id='later',quality_score=.9,timestamp_ms=2),dict(frame_id='earlier',quality_score=.9,timestamp_ms=1)]
    assert choose_single_frame(frames)=='earlier'


def test_majority_tie_breaks_on_mean_probability_then_class_order():
    a=[.6,.3]+[.1/11]*11;b=[.1,.8]+[.1/11]*11
    result=fuse_square([row(0,a),row(1,b)],'majority_vote',config())
    assert result['vote_ties']==[0,1] and result['predicted_class_id']==1
    c=[.8,.1]+[.1/11]*11;result=fuse_square([row(0,c),row(1,b)],'majority_vote',config());assert result['predicted_class_id']==0


def test_mean_probability_and_mean_logit_precise_definitions():
    a=[.85,.1]+[.05/11]*11;b=[.1,.8]+[.1/11]*11;rows=[row(0,a),row(1,b)]
    p=fuse_square(rows,'mean_probabilities',config())['score_distribution'];assert np.allclose(p,np.mean([r['probabilities'] for r in rows],axis=0))
    z=np.mean([r['raw_logits'] for r in rows],axis=0);expected=np.exp(z-z.max());expected/=expected.sum()
    assert np.allclose(fuse_square(rows,'mean_logits',config())['score_distribution'],expected)
    shifted=copy.deepcopy(rows)
    for r in shifted:r['raw_logits']=(np.asarray(r['raw_logits'])+1234).tolist()
    assert np.allclose(fuse_square(shifted,'mean_logits',config())['score_distribution'],expected)


def test_grouping_avoids_temporal_single_link_chains():
    rows=[row(i,time=i*6000,angle=i*1.0) for i in range(3)]
    assert groups(rows,config())==[[0,1],[2]]


def test_grouped_quality_median_discounts_repeated_correlated_error():
    wrong=[.02,.96]+[.02/11]*11;right=[.96,.02]+[.02/11]*11
    rows=[row(i,wrong,time=i*1000,angle=0,quality=.5) for i in range(3)]+[row(3,right,time=20000,angle=1.5),row(4,right,time=40000,angle=3)]
    assert fuse_square(rows,'majority_vote',config())['predicted_class_id']==1
    result=fuse_square(rows,'clustered_quality_median',config());assert result['predicted_class_id']==0 and result['heuristic_group_count']==3
    assert result['strong_conflict'] and result['abstained']  # Never hide a conflict simply because robust argmax won.


def test_strong_contradiction_is_visible_and_abstains():
    a=[.99,.005]+[.005/11]*11;b=[.005,.99]+[.005/11]*11
    result=fuse_square([row(0,a,time=0),row(1,b,time=15000)],'mean_logits',config())
    assert result['strong_conflict'] and 'contradictory_strong_observations' in result['abstention_reasons']
    assert result['operational_class_id'] is None


@pytest.mark.parametrize('defect',['session','square','model','source','probability','geometry','class_order','duplicate'])
def test_invalid_or_cross_square_evidence_is_rejected(defect):
    a=row(0);b=row(1)
    if defect=='session':b['session_id']='other'
    if defect=='square':b['canonical_square']='b8'
    if defect=='model':b['checkpoint_sha256']='d'*64
    if defect=='source':b['source_video_sha256']='e'*64
    if defect=='probability':b['probabilities'][0]=float('nan')
    if defect=='geometry':b['geometry_verification_status']='unresolved'
    if defect=='class_order':b['class_names'].reverse()
    if defect=='duplicate':b=copy.deepcopy(a);b['observation_id']='clone';b['frame_id']='clone'
    with pytest.raises(ValueError):fuse_square([a,b],'mean_probabilities',config())


def test_camera_orientation_metadata_never_rotates_canonical_square_identity():
    rows=[row(i,time=i*10000,square='h1') for i in range(4)]
    for i,r in enumerate(rows):r['rotation_quarters_ccw']=i;r['semantic_orientation']=['white_near_camera','white_left','black_near_camera','white_right'][i]
    assert fuse_square(rows,'mean_probabilities',config())['canonical_square']=='h1'


def test_ground_truth_fields_cannot_change_any_prediction():
    rows=[row(i,time=i*10000) for i in range(3)];altered=copy.deepcopy(rows)
    for r in altered:r['ground_truth']='black_queen';r['FEN']='invented';r['oracle_correct']=False
    for m in METHODS:assert fuse_square(rows,m,config(),single_frame_id='f0')==fuse_square(altered,m,config(),single_frame_id='f0')


def test_incomplete_board_rejected_instead_of_reusing_neighbor_square():
    with pytest.raises(ValueError):fuse_board([row(0)],'mean_probabilities',config())


def test_calibration_diagnostics_do_not_claim_calibration_and_zero_scores_are_explicit():
    p=np.zeros((2,13));p[:,0]=1;r=reliability([0,1],p,[0,0],[False,True],config())
    assert not r['score_calibrated'] and r['zero_true_score_count']==1 and np.isfinite(r['nll_with_floor'])
    assert r['coverage']==.5 and r['accepted_error_count']==0 and r['high_score_wrong_count']==1


def test_legality_is_separate_and_never_repairs_prediction_vector():
    p=[0]*64;before=p.copy();result=legality(p);assert p==before
    assert not result['valid_under_either_assumed_turn'] and 'white_to_move' in result and 'black_to_move' in result


def test_fusion_configuration_is_offline_and_untuned():
    c=validate_config(config());c['completion_enabled']=True
    with pytest.raises(ValueError):validate_config(c)


def test_label_version_changes_evaluation_only_and_preserves_original_catalog():
    from scripts.characterize_square_evidence import evaluation_truths
    from src.scanning.square_evidence import classification_metrics
    original='8/8/8/8/8/8/8/8 w - - 0 1';corrected='8/2p5/8/8/8/8/8/8 w - - 0 1'
    catalog=dict(sessions=[dict(capture_session_id='s',position_id='pos',fen=original)])
    correction=dict(schema_version='offline_pilot_ground_truth_corrections_v1',source_catalog_sha256='hash',corrections=[dict(
        session_id='s',position_id='pos',original_fen=original,corrected_fen=corrected,reviewer_kind='human',approved_by='synthetic test human',approval_statement='fixture confirmation')])
    p=[.01]*13;p[7]=.88;rows=[row(i,p,square='c7',time=i*10000) for i in range(3)];prediction=fuse_square(rows,'mean_probabilities',config())
    before,_=evaluation_truths(catalog,'hash');after,_=evaluation_truths(catalog,'hash',correction);index=SQUARES.index('c7')
    assert classification_metrics([before['s'][index]],[prediction['predicted_class_id']])['correct']==0
    assert classification_metrics([after['s'][index]],[prediction['predicted_class_id']])['correct']==1
    assert catalog['sessions'][0]['fen']==original
    assert prediction==fuse_square(rows,'mean_probabilities',config())
    correction['source_catalog_sha256']='wrong'
    with pytest.raises(ValueError):evaluation_truths(catalog,'hash',correction)
