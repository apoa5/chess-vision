"""D.3b uncertainty, tie-safe coverage, board risk and registration invariants."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from src.scanning.fusion_reliability import uncertainty_signals, risk_curve, matched_coverage, board_summary
from src.scanning.independent_fusion import validate_registry
from src.scanning.square_evidence import SQUARES


def probabilities(k, confidence=.95):
    p=np.full(13,(1-confidence)/12);p[k]=confidence;return p


def test_identical_views_have_zero_disagreement_and_dispersion():
    p=probabilities(2);s=uncertainty_signals(p,[p]*5,2)
    assert s['js_disagreement']==pytest.approx(0,abs=1e-12)
    assert s['variation_ratio']==0 and s['predicted_class_dispersion']==pytest.approx(0)
    assert s['strong_support_fraction']==1 and s['unanimous_argmax']
    assert s['margin']==pytest.approx(.95-.05/12)
    assert s['entropy']==pytest.approx(-sum(p*np.log(p)))


def test_confident_consensus_can_hide_a_wrong_class():
    p=probabilities(0,.999);views=[p]*4+[probabilities(1,.644)]
    s=uncertainty_signals(p,views,0)
    assert s['strong_support_fraction']==.8 and s['variation_ratio']==pytest.approx(.2)
    assert s['predicted_class_dispersion']>0 and s['js_disagreement']>0
    assert s['maximum_probability']>.99  # No ground truth is needed to compute these features.


@pytest.mark.parametrize('bad',['shape','nan','negative','nonunit','class'])
def test_invalid_uncertainty_inputs_fail_closed(bad):
    p=probabilities(1);v=np.asarray([p,p]);k=1
    if bad=='shape':p=p[:12]
    elif bad=='nan':v[0,0]=np.nan
    elif bad=='negative':v[0,0]=-1
    elif bad=='nonunit':p=p*2
    else:k=0
    with pytest.raises(ValueError): uncertainty_signals(p,v,k)


def ranking_rows():
    return [dict(session_id='s',canonical_square=s,signal=v,correct=c) for s,v,c in [('a1',.9,False),('b1',.9,True),('c1',.7,True),('d1',.1,False)]]


def test_threshold_curve_keeps_ties_together_and_zero_coverage_has_no_risk():
    r=risk_curve(ranking_rows(),'signal',True)
    assert r[0]['risk'] is None
    assert [p['retained'] for p in r]==[0,2,3,4]
    assert r[1]['risk']==.5 and r[-1]['risk']==.5
    assert risk_curve([],'signal',False)==[]


def test_matched_coverage_replay_is_deterministic_with_honest_tie_bounds():
    rows=ranking_rows();r=matched_coverage(rows,'signal',True,.25)
    assert r==matched_coverage(rows[::-1],'signal',True,.25)
    assert r['retained']==1 and r['boundary_tie_count']==2
    assert r['tie_risk_min']==0 and r['tie_risk_max']==1
    flipped=copy.deepcopy(rows)
    for x in flipped:x['correct']=not x['correct']
    assert matched_coverage(flipped,'signal',True,.25)['errors']==0  # ID, never truth, determined tie selection.


def test_low_uncertainty_direction_and_invalid_rankings():
    assert matched_coverage(ranking_rows(),'signal',False,.25)['errors']==1
    with pytest.raises(ValueError):matched_coverage(ranking_rows()*2,'signal',True,.5)
    with pytest.raises(ValueError):matched_coverage(ranking_rows(),'signal',True,0)
    rows=ranking_rows();rows[0]['signal']=float('nan')
    with pytest.raises(ValueError):matched_coverage(rows,'signal',True,.5)


def test_board_mean_confidence_does_not_imply_exact_or_retained_board():
    rows=[dict(canonical_square=s,maximum_probability=.999,margin=.99,entropy=.01,js_disagreement=0.,abstained=False,correct=True) for s in SQUARES]
    rows[0].update(correct=False,maximum_probability=.95,abstained=True)
    b=board_summary(rows)
    assert b['mean_probability']>.99 and not b['exact_board'] and not b['all_squares_retained']
    assert b['incorrect_squares']==1 and b['withheld_squares']==1
    with pytest.raises(ValueError):board_summary(rows[:-1])


def registry():
    s=dict(session_id='new_s',position_id='new_p',position_group_id='new_pg',independence_group_id='new_cohort',capture_date='2026-10-23',
        fen='8/5pk1/6p1/3n3p/2B1P1PP/8/5PK1/8 w - - 0 1',label_version='owner_v1',novel_position_attested=True,
        source_video=dict(path='video.mov',sha256='a'*64),reference=dict(path='reference.jpg',sha256='b'*64),native_index=dict(path='index.json',sha256='c'*64),
        attestation=dict(reviewer='owner',reviewer_kind='human',photo_fen_agreement=True,unchanged_position=True),board_id='board',lighting='diffuse',environment='room',
        device_metadata={'device':'unknown'},viewpoint_coverage={'elevation':'unknown'})
    return dict(schema_version='fusion_independent_registry_d3b_v1',final_holdout=False,training_eligible=False,outcomes_inspected=False,used_for_tuning=False,sessions=[s])


def known():return dict(sessions=set(),sources=set(),placements=set())


def test_registration_preserves_position_groups_and_counts_sessions_not_frames():
    r=registry();s=copy.deepcopy(r['sessions'][0]);s['session_id']='new_take';s['source_video']['sha256']='d'*64;r['sessions'].append(s)
    out=validate_registry(r,known());assert out['sessions']==2 and out['distinct_position_groups']==1
    s['position_group_id']='split_group'
    with pytest.raises(ValueError):validate_registry(r,known())


@pytest.mark.parametrize('defect',['holdout','training','inspected','tuning','agreement','stability','reviewer','novelty','source','reference','fen','date','duplicate','empty'])
def test_missing_or_ineligible_independent_provenance_rejected(defect):
    r=registry();s=r['sessions'][0]
    if defect=='holdout':r['final_holdout']=True
    elif defect=='training':r['training_eligible']=True
    elif defect=='inspected':r['outcomes_inspected']=True
    elif defect=='tuning':r['used_for_tuning']=True
    elif defect=='agreement':s['attestation']['photo_fen_agreement']=False
    elif defect=='stability':s['attestation']['unchanged_position']=False
    elif defect=='reviewer':s['attestation']['reviewer_kind']='assistant'
    elif defect=='novelty':s['novel_position_attested']=False
    elif defect=='source':s['source_video']['sha256']=None
    elif defect=='reference':s['reference']['path']=None
    elif defect=='fen':s['fen']='bad'
    elif defect=='date':s['capture_date']='unknown'
    elif defect=='duplicate':r['sessions']*=2
    else:r['sessions']=[]
    with pytest.raises((ValueError,TypeError)):validate_registry(r,known())


@pytest.mark.parametrize('kind',['sessions','sources','placements'])
def test_pilot_cannot_be_rebranded_as_independent(kind):
    r=registry();s=r['sessions'][0];k=known()
    k[kind].add({'sessions':s['session_id'],'sources':s['source_video']['sha256'],'placements':s['fen'].split()[0]}[kind])
    with pytest.raises(ValueError):validate_registry(r,k)
    assert validate_registry(r,k,functional_development=True)['sessions']==1


def test_empty_template_does_not_claim_collected_validation():
    r=json.loads(Path('configs/capture/fusion_independent_d3b_registry_template_v1.json').read_text())
    with pytest.raises(ValueError,match='pending'):validate_registry(r,known())


def test_position_id_cannot_change_its_ground_truth():
    r=registry();s=copy.deepcopy(r['sessions'][0]);s['session_id']='take2';s['source_video']['sha256']='d'*64
    s['fen']='8/5pk1/6p1/3n3p/2B1P2P/6P1/5PK1/8 w - - 0 1';r['sessions'].append(s)
    with pytest.raises(ValueError,match='inconsistent FEN'):validate_registry(r,known())


def test_development_mode_can_only_claim_functionality_for_inspected_sessions():
    r=registry();r['outcomes_inspected']=True;r['used_for_tuning']=True
    with pytest.raises(ValueError):validate_registry(r,known())
    assert validate_registry(r,known(),functional_development=True)['sessions']==1


def test_evaluation_rejects_unsupported_or_misrepresented_registration(tmp_path):
    from scripts.evaluate_independent_fusion import evaluate
    lock=tmp_path/'lock.json';lock.write_text(json.dumps({'schema_version':'unknown'}))
    with pytest.raises(ValueError,match='schema'):evaluate(lock,tmp_path/'absent.json',tmp_path/'out')
    lock.write_text(json.dumps({'schema_version':'fusion_independent_registration_d3b_v1','functional_development':True}))
    with pytest.raises(ValueError,match='mode mismatch'):evaluate(lock,tmp_path/'absent.json',tmp_path/'out')


def test_stale_registration_source_is_rejected_before_evidence_access(tmp_path):
    from scripts.evaluate_independent_fusion import evaluate
    source=tmp_path/'source';source.write_text('new contents')
    lock=tmp_path/'lock.json';lock.write_text(json.dumps(dict(schema_version='fusion_independent_registration_d3b_v1',functional_development=False,source_sha256={str(source):'0'*64})))
    with pytest.raises(ValueError,match='Protected input changed'):evaluate(lock,tmp_path/'absent.json',tmp_path/'out')


def test_preregistration_freezes_methods_code_and_protocol_before_new_outcomes():
    from scripts.select_video_keyframes import check_hashes
    r=json.loads(Path('data/metadata/fusion_independent_d3b_preregistration_v1.json').read_text())
    check_hashes(r['source_sha256'])
    assert r['independent_validation']=='pending' and r['eligible_new_sessions_collected']==0
    assert r['methods']==['single_quality','mean_probabilities','mean_logits']
    assert r['condition']=='uniform_5' and not r['completion_enabled'] and not r['final_holdout']
