"""D.3c minimal video adapter, explicit review status, and frozen-kernel equivalence."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
import pytest
from src.scanning.interim_video import match_position,write_gzip
from src.scanning.pilot_comparison import display_image,display_transform
from src.scanning.interim_evidence import fuse_provisional_board,validate_provisional_observation
from src.scanning import offline_fusion
from src.scanning.square_evidence import validate_observation


@pytest.mark.parametrize('name,expected',[('pos_001.MOV','01'),('pos_019.mp4','19'),('pos_003.mov','03')])
def test_position_mapping_uses_number_not_file_order(name,expected):
    assert match_position(name,{'01':'a','03':'b','19':'c'})==expected


@pytest.mark.parametrize('name',['001.MOV','pos_021.MOV','pos_001_copy.MOV','pos_001.jpg'])
def test_ambiguous_or_unmapped_sources_fail_closed(name):
    with pytest.raises(ValueError):match_position(name,{'01':'x'})


@pytest.mark.parametrize('rotation',[0,-90,90,180])
def test_display_rotation_matches_source_pixel_transform(rotation):
    image=np.arange(3*4*3,dtype=np.uint8).reshape(3,4,3);out=display_image(image,rotation);matrix=display_transform(4,3,rotation)
    for y in range(3):
        for x in range(4):
            xx,yy,_=matrix@np.array([x,y,1]);assert np.array_equal(out[int(yy),int(xx)],image[y,x])


def test_deterministic_lossless_compact_artifact_never_overwrites(tmp_path):
    a=tmp_path/'a.gz';b=tmp_path/'b.gz';obj={'pts':[0,600,601], 'probabilities':[.1,.9]}
    write_gzip(a,obj);write_gzip(b,obj);assert a.read_bytes()==b.read_bytes()
    with pytest.raises(ValueError):write_gzip(a,obj)


@pytest.fixture(scope='module')
def evidence():
    e=json.loads(Path('data/metadata/evidence_d2_manifest_v1.json').read_text())
    obs=[json.loads(x) for x in Path(e['observations_path']).read_text().splitlines()]
    return obs,json.loads(Path('configs/experiments/fusion_d3.json').read_text())


@pytest.mark.parametrize('method',['single_quality','mean_probabilities','mean_logits'])
def test_provisional_adapter_is_exact_frozen_kernel_and_never_forges_review(method,evidence):
    obs,config=evidence;rows=[o for o in obs if o['session_id']==obs[0]['session_id'] and any(x['condition']=='uniform_5' for x in o['condition_memberships'])]
    original=offline_fusion.fuse_board(rows,method,config);unverified=copy.deepcopy(rows)
    for o in unverified:o['geometry_verification_status']='completed_unverified'
    with pytest.raises(ValueError):offline_fusion.fuse_board(unverified,method,config)
    provisional=fuse_provisional_board(unverified,method,config,explicit_review_waiver=True)
    assert provisional==original
    assert all(o['geometry_verification_status']=='completed_unverified' for o in unverified)
    assert offline_fusion.validate_observation is validate_observation
    with pytest.raises(ValueError):fuse_provisional_board(unverified,method,config)


@pytest.mark.parametrize('defect',['rejected','unresolved','probabilities','pts','class_order','calibration'])
def test_waiver_does_not_admit_invalid_or_rejected_observations(defect,evidence):
    obs,_=evidence;o=copy.deepcopy(obs[0]);o['geometry_verification_status']='completed_unverified'
    if defect in ('rejected','unresolved'):o['geometry_verification_status']=defect
    elif defect=='probabilities':o['probabilities'][0]=float('nan')
    elif defect=='pts':o['original_pts']+=1
    elif defect=='class_order':o['class_names'].reverse()
    else:o['calibrated_probabilities']=o['probabilities']
    with pytest.raises(ValueError):validate_provisional_observation(o)


def test_scoped_validator_is_restored_after_exception(evidence):
    obs,c=evidence;rows=[copy.deepcopy(obs[0])];rows[0]['geometry_verification_status']='unresolved'
    with pytest.raises(ValueError):fuse_provisional_board(rows,'mean_logits',c,explicit_review_waiver=True)
    assert offline_fusion.validate_observation is validate_observation


def test_supplied_twenty_fens_are_structurally_valid_without_relabeling():
    import chess
    m=json.loads(Path('data/metadata/fusion_d3c_supplied_fens_v1.json').read_text())
    assert len(m['positions'])==20
    for fen in m['positions'].values():assert len(fen.split())==6 and chess.Board(fen).is_valid()
