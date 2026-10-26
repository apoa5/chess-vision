"""D.2 gate, frozen preprocessing and GT-only diagnostics on synthetic fixtures."""
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.chess.labels import CLASS_NAMES
from src.scanning.square_evidence import (SQUARES,CHECKS,verify_reviews,uncertainty,validate_observation,square_records,
    classification_metrics,multi_view_metrics,paired_errors,descriptive_correlation,digest_json,validate_bundle)
from scripts.run_square_evidence import make_inputs,run
from scripts.characterize_square_evidence import paired_coverage
from src.vision.crop_experiments import CropSpec,transform_crop,tensor_from_packed
from src.vision.dataset_loader import preprocess_context_crop
from src.vision.grid_geometry import polygon_crop
from src.vision.dataset import chess_square_at


def review_fixture():
    row=dict(review_id='R001',binding={'frame':'exact'},conditions=['diversity_5','uniform_5'],geometry_completed=True)
    review=dict(review_id='R001',binding=row['binding'],decision='pending',reviewer=None,reviewer_kind='human',checks={k:False for k in CHECKS},notes='')
    return dict(frames=[row]),dict(schema_version='offline_evidence_geometry_reviews_d2_v1',packet_sha256='packet',reviews=[review])


def verified_reviews():
    packet,reviews=review_fixture();r=reviews['reviews'][0];r.update(decision='human_verified',reviewer='Synthetic test reviewer',checks={k:True for k in CHECKS});return packet,reviews


def observation(square='a8',frame='f',pts=600):
    values=uncertainty(np.arange(13))
    o=dict(observation_id=frame+square,position_id='position',session_id='session',frame_id=frame,canonical_square=square,
        source_video_sha256='a'*64,checkpoint_sha256='b'*64,crop_config_sha256='c'*64,original_pts=pts,time_base=[1,600],pts_time=[pts,600],timestamp_ms=pts/600*1000,
        geometry_verification_status='human_verified',calibration_status='not_available',calibrated_probabilities=None,
        raw_logits=list(range(13)),**values,condition_memberships=[dict(selection_budget=5)],semantic_orientation='white_near_camera',
        class_names=list(CLASS_NAMES),predicted_class=CLASS_NAMES[values['predicted_class_id']])
    return o


def test_pending_primary_blocks_before_model_load_or_output(tmp_path):
    packet,reviews=review_fixture();p=tmp_path/'packet.json';p.write_text(json.dumps(packet));reviews['packet_sha256']=__import__('hashlib').sha256(p.read_bytes()).hexdigest();r=tmp_path/'review.json';r.write_text(json.dumps(reviews))
    with pytest.raises(ValueError,match='Primary geometry review pending'):run(p,r,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_rejected_or_uncertain_frames_remain_explicit_never_replaced():
    packet,reviews=review_fixture();r=reviews['reviews'][0];r.update(decision='rejected',reviewer='human',notes='wrong board boundary')
    assert verify_reviews(packet,reviews,'packet')['R001']['geometry_status']=='geometry_rejected'
    r['decision']='unresolved';assert verify_reviews(packet,reviews,'packet')['R001']['geometry_status']=='geometry_unresolved'


@pytest.mark.parametrize('defect',['hash','binding','reviewer','kind','checks','duplicate','missing'])
def test_incomplete_stale_or_nonhuman_attestations_cannot_pass(defect):
    packet,reviews=verified_reviews();r=reviews['reviews'][0]
    if defect=='hash':reviews['packet_sha256']='wrong'
    if defect=='binding':r['binding']={}
    if defect=='reviewer':r['reviewer']=''
    if defect=='kind':r['reviewer_kind']='assistant'
    if defect=='checks':r['checks']['semantic_orientation']=False
    if defect=='duplicate':reviews['reviews'].append(copy.deepcopy(r))
    if defect=='missing':reviews['reviews']=[]
    with pytest.raises(ValueError):verify_reviews(packet,reviews,'packet')


def test_primary_approval_does_not_verify_sensitivity_only_frames():
    packet,reviews=verified_reviews();row=copy.deepcopy(packet['frames'][0]);row['review_id']='R002';row['conditions']=['diversity_10'];packet['frames'].append(row)
    r=copy.deepcopy(reviews['reviews'][0]);r.update(review_id='R002',decision='pending');reviews['reviews'].append(r)
    states=verify_reviews(packet,reviews,'packet');assert states['R001']['human_review']=='human_verified' and states['R002']['human_review']=='pending'


@pytest.mark.parametrize('bad',[[],[1]*12,[float('nan')]*13,[float('inf')]*13])
def test_invalid_logits_fail(bad):
    with pytest.raises(ValueError):uncertainty(bad)


def test_raw_distribution_uncertainty_preserved_and_stable():
    a=uncertainty([10000]*13);assert sum(a['probabilities'])==pytest.approx(1);assert a['top_two_probability_margin']==0
    assert a['predictive_entropy']==pytest.approx(np.log(13))
    assert uncertainty([-10000]*12+[10000])['maximum_probability']==1
    assert uncertainty(np.arange(13))==uncertainty(np.arange(13)+999)
    o=observation();assert validate_observation(json.loads(json.dumps(o)))==o


@pytest.mark.parametrize('defect',['prob','timestamp','geometry','class','calibration','entropy'])
def test_invalid_or_misrepresented_observation_fails(defect):
    o=observation()
    if defect=='prob':o['probabilities'][0]=.8
    if defect=='timestamp':o['original_pts']+=1
    if defect=='geometry':o['geometry_verification_status']='completed_unverified'
    if defect=='class':o['class_names'].reverse()
    if defect=='calibration':o['calibrated_probabilities']=o['probabilities']
    if defect=='entropy':o['predictive_entropy']=0
    with pytest.raises(ValueError):validate_observation(o)


def test_containers_roundtrip_64_squares_without_a_pooled_distribution():
    obs=[observation('a8'),observation('a8','f2',1200)];records=square_records(obs)
    assert len(records)==64;assert records['a8']['state']=='candidate';assert records['a8']['class_distribution'] is None
    assert records['a8']['last_observation_timestamp_ms']==2000
    assert records['h1']['state']=='unknown' and records['h1']['class_distribution'] is None
    assert json.loads(json.dumps(records))==records
    with pytest.raises(ValueError):square_records(obs+[obs[0]])
    wrong=copy.deepcopy(obs[0]);wrong['position_id']='another'
    with pytest.raises(ValueError):square_records([obs[0],wrong])


def test_clone_frame_id_does_not_make_independent_evidence():
    a=observation();b=copy.deepcopy(a);b['frame_id']='clone';b['observation_id']='new'
    with pytest.raises(ValueError,match='Duplicate physical observation'):square_records([a,b])


def test_metrics_partition_errors_and_handle_absent_class_support():
    m=classification_metrics([0,0,1,2,10],[0,3,0,6,10])
    assert m['correct']==2 and m['empty_false_positives']==1 and m['occupied_to_empty_misses']==1 and m['incorrect_piece_identities']==1
    assert m['per_class'][10]['recall']==1 and m['per_class'][11]['recall'] is None
    assert m['empty_false_positive_rate']==.5 and m['occupied_to_empty_miss_rate']==pytest.approx(1/3)


def test_oracle_measures_information_availability_not_majority_vote():
    y=np.zeros(64,int);p=np.zeros((5,64),int);p[:4,0]=1;p[4,1]=1;p[:,2]=1
    m=multi_view_metrics(y,p)
    assert m['ever_correct_squares']==63 and m['persistently_wrong_squares']==1 and m['oracle_incorrect_squares']==1
    assert m['squares'][0]['correct_in_only_one_view'];assert m['squares'][1]['correct_in_majority']
    assert not m['oracle_exact_board'] and m['exact_single_frame_boards']==0
    p[4,2]=0;assert multi_view_metrics(y,p)['oracle_exact_board']  # Five-view upper bound; no frame is exact.
    assert multi_view_metrics(y,p)['exact_single_frame_boards']==0


def test_no_views_is_unknown_not_an_oracle_failure_or_perfect_board():
    m=multi_view_metrics(np.zeros(64,int),np.empty((0,64),int))
    assert m['ever_correct_squares'] is None and m['oracle_exact_board'] is None


def test_error_overlap_distinguishes_wrong_diversity_from_useful_complementarity():
    y=[0,1,2,3];a=[0,2,2,1];b=[0,3,0,3];m=paired_errors(y,a,b)
    assert m['shared_errors']==1 and m['error_union']==3 and m['error_jaccard']==pytest.approx(1/3)
    assert m['useful_one_correct_other_wrong']==2 and m['both_wrong_different_predictions']==1
    assert paired_errors(y,y,y)['binary_error_phi'] is None
    assert paired_errors(y,a,a)['binary_error_phi']==1


def test_strategy_pairing_reports_physical_square_gains_and_losses():
    y=np.zeros(64,int);a=np.zeros((5,64),int);b=a.copy();a[:,0]=1;b[:,1]=1
    delta=paired_coverage(multi_view_metrics(y,a),multi_view_metrics(y,b))
    assert delta['b_minus_a_ever_correct']==0 and delta['rescued_by_b']==['a8'] and delta['lost_by_b']==['b8']
    assert delta['equal_usable_frame_budget']


def test_correlation_is_descriptive_and_constant_error_vectors_are_undefined():
    assert descriptive_correlation([1,2,3],[2,4,6])['pearson']==1
    assert descriptive_correlation([1,1,1],[1,2,3])['pearson'] is None
    assert descriptive_correlation([1,None],[1,2])['pair_count']==1


@pytest.mark.parametrize('turns',range(4))
def test_frozen_crop_preprocessing_and_canonical_identity_across_orientations(turns):
    config=json.loads(Path('configs/experiments/evidence_d2.json').read_text())
    image=np.random.default_rng(42).integers(0,256,(1200,1200,3),dtype=np.uint8)
    polygons=[]
    for r in range(8):
        for c in range(8):
            x=200+c*100;y=200+r*100;polygons.append([[x,y],[x+99,y],[x+99,y+99],[x,y+99]])
    geometry=dict(provenance=dict(source_corners=[[200,200],[999,200],[999,999],[200,999]]),source_square_polygons=polygons,
                  semantic_square_ids=[chess_square_at(r,c,turns) for r in range(8) for c in range(8)])
    made=make_inputs(image,geometry,config);assert {r['canonical_square'] for r in made}==set(SQUARES)
    for index in (0,27,63):
        crop,_,_,_,local,_=polygon_crop(image,np.full((1200,1200),255,np.uint8),np.float32(polygons[index]),200)
        transformed,p=transform_crop(crop,local,CropSpec(**config['crop_spec']))
        reference=preprocess_context_crop(transformed,p,224)
        assert np.array_equal(tensor_from_packed(made[index]['packed']),reference)
        assert made[index]['canonical_square']==chess_square_at(index//8,index%8,turns)


def test_bundle_rejects_missing_square_and_cross_model_observations():
    obs=[observation(s) for s in SQUARES]
    manifest=dict(schema_version='offline_square_evidence_d2_v1',designation='development_only',training_eligible=False,final_holdout=False,
        fusion_implemented=False,completion_enabled=False,ground_truth_used_for_inference=False,observation_count=64,
        model=dict(checkpoint_sha256='b'*64),config_sha256='c'*64,
        frames=[dict(session_id='session',frame_id='f',human_review='human_verified',observations=[o['observation_id'] for o in obs])],
        sessions=[dict(capture_session_id='session',squares=square_records(obs))])
    assert validate_bundle(manifest,obs)['valid']
    bad=copy.deepcopy(obs);bad[0]['checkpoint_sha256']='d'*64
    with pytest.raises(ValueError):validate_bundle(manifest,bad)
    with pytest.raises(ValueError):validate_bundle(manifest,obs[:-1])


def test_characterization_cli_end_to_end_on_synthetic_evidence(tmp_path):
    """No pilot photos/labels: exercise serialized records, diagnostics and oracle exports."""
    from scripts.characterize_square_evidence import run as characterize
    from src.scanning.native_pilot import sha256_file
    import hashlib
    def write(name,obj):
        p=tmp_path/name;p.write_text(json.dumps(obj));return p
    catalog=write('catalog.json',dict(sessions=[dict(capture_session_id='session',fen='8/8/8/8/8/8/8/8 w - - 0 1')]))
    config=write('config.json',dict(source_catalog=str(catalog)))
    packet=write('packet.json',dict(config_path=str(config),protected_sha256={str(catalog):sha256_file(catalog)}))
    image=tmp_path/'synthetic.png';cv2.imwrite(str(image),np.zeros((192,108,3),np.uint8))
    packed=np.zeros((64,4,224,224),np.uint8);cache=tmp_path/'synthetic_inputs.npz';np.savez_compressed(cache,packed=packed)
    observations=[];frames=[]
    for frame_number in range(3):
        fid='f'+str(frame_number);these=[]
        for i,square in enumerate(SQUARES):
            o=observation(square,fid,600*(frame_number+1))
            o.update(input_cache=str(cache),input_index=i,input_packed_sha256=hashlib.sha256(packed[i].tobytes()).hexdigest());these.append(o)
        observations+=these
        frames.append(dict(session_id='session',position_id='position',frame_id=fid,timestamp_ms=1000*(frame_number+1),
            conditions=['diversity_5','uniform_5','diversity_10'],human_review='human_verified',geometry_status='human_verified',
            observations=[o['observation_id'] for o in these],
            d1_metadata=dict(view_descriptor=[frame_number,0,0,0],path=str(image))))
    op=tmp_path/'obs.jsonl';op.write_text(''.join(json.dumps(o)+'\n' for o in observations))
    manifest=dict(schema_version='offline_square_evidence_d2_v1',designation='development_only',training_eligible=False,final_holdout=False,
        fusion_implemented=False,completion_enabled=False,ground_truth_used_for_inference=False,observation_count=192,
        model=dict(checkpoint_sha256='b'*64),config_sha256='c'*64,frames=frames,
        sessions=[dict(capture_session_id='session',position_id='position',squares=square_records(observations))],
        packet_path=str(packet),packet_sha256=sha256_file(packet),observations_path=str(op),observations_sha256=sha256_file(op))
    ep=write('evidence.json',manifest);characterize(ep,tmp_path/'analysis')
    result=json.loads((tmp_path/'analysis/characterization.json').read_text())
    assert len(result['conditions'])==3 and len(result['pairwise_diagnostics'])==9
    assert all(r['ever_correct_squares']==0 and r['oracle_incorrect_squares']==64 for r in result['conditions'])
    assert result['paired_session_comparisons'][0]['uniform5_to_diversity5']['b_minus_a_ever_correct']==0
    assert all(r['pairwise_summary']['mean_prediction_disagreement_rate']==0 for r in result['conditions'])
    assert (tmp_path/'analysis/comparison.csv').exists()


def test_ground_truth_corrections_require_explicit_human_approval_and_original_binding():
    from scripts.characterize_square_evidence import evaluation_truths
    fen='8/8/8/8/8/8/8/8 w - - 0 1';updated='8/2p5/8/8/8/8/8/8 w - - 0 1'
    catalog=dict(sessions=[dict(capture_session_id='s',position_id='p',fen=fen)])
    correction=dict(schema_version='offline_pilot_ground_truth_corrections_v1',source_catalog_sha256='source',corrections=[dict(
        session_id='s',position_id='p',original_fen=fen,corrected_fen=updated,reviewer_kind='human',approved_by='synthetic human fixture',approval_statement='explicit fixture approval')])
    before,_=evaluation_truths(catalog,'source');after,_=evaluation_truths(catalog,'source',correction)
    assert before['s'][SQUARES.index('c7')]==0 and after['s'][SQUARES.index('c7')]==7
    assert catalog['sessions'][0]['fen']==fen  # The registered labels never mutate.
    for field,value in [('reviewer_kind','assistant'),('original_fen',updated),('approval_statement','')]:
        bad=copy.deepcopy(correction);bad['corrections'][0][field]=value
        with pytest.raises(ValueError):evaluation_truths(catalog,'source',bad)
