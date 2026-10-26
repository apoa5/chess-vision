"""C.3 representation parity, frozen groups and matched training safeguards."""
import copy
import json
from pathlib import Path
import hashlib

import cv2
import numpy as np
import pytest
import torch

from src.vision.crop_training_c3 import (source_geometry,native_input,historical_inputs,
    make_native_cache,checkpoint_key,epoch_metrics,state_hash,TrackedDataset)
from src.vision.geometry_audit import context_transform,project
from src.vision.input_representations import extract_representations,RepresentationSettings
from src.vision.classifier import create_classifier
from test_input_representations import scene


def fixture():
    image,g,config,*_=scene()
    raw=np.float32(g['source_square_polygons']);corners=g['provenance']['source_corners']
    H=context_transform(corners,[200,200,1000,1000]);canvas=project(raw,H)
    prepared=dict(source_corners=corners,board_bounds=[200,200,1000,1000],
        grid_geometry=dict(square_polygons=canvas.tolist(),source_square_polygons=raw.tolist()))
    row=dict(row=3,col=3,target_polygon=canvas[27].tolist(),bounds=[0,0,1200,1200],
             crop_id='synthetic_d5',class_id=6,split='train',geometry_review_required=False,quality_flags=[])
    return image,g,config,prepared,row


def test_native_input_matches_unmodified_c1_pixels_and_mask():
    image,g,config,p,row=fixture();before=image.copy();prior=copy.deepcopy(p)
    result=native_input(image,row,p)
    old=extract_representations(image,g,g['semantic_square_ids'][27],config)['taller']
    assert np.array_equal(result['packed'],old['packed'])
    assert result['packed'].shape==(4,224,224)
    assert set(np.unique(result['packed'][3]))<={0,1}
    assert np.array_equal(image,before) and p==prior


def test_manual_records_use_saved_raw_polygons_without_new_corner_estimation():
    image,_,_,p,row=fixture();del p['source_corners']
    raw,q=source_geometry(p);out=native_input(image,row,p)
    assert raw.shape==(64,4,2) and q.shape==(4,2) and out['packed'][3].sum()>0


@pytest.mark.parametrize('mutation',['missing','nonfinite','nonconvex'])
def test_invalid_source_geometry_blocks_instead_of_dropping_target(mutation):
    _,_,_,p,_=fixture()
    if mutation=='missing':del p['grid_geometry']['source_square_polygons']
    elif mutation=='nonfinite':p['grid_geometry']['source_square_polygons'][0][0][0]=float('nan')
    else:p['grid_geometry']['source_square_polygons'][0][0]=p['grid_geometry']['source_square_polygons'][7][1]
    with pytest.raises(ValueError):source_geometry(p)


def test_wrong_saved_target_mapping_is_rejected():
    image,_,_,p,row=fixture();row['target_polygon'][0][0]+=2
    with pytest.raises(ValueError):native_input(image,row,p)


def test_native_extraction_is_label_blind():
    image,_,_,p,row=fixture();a=native_input(image,row,p)['packed']
    row['class_id']=0;row['class_name']='empty'
    assert np.array_equal(a,native_input(image,row,p)['packed'])


def test_native_cache_is_ram_only_readonly_and_checks_frozen_pixels(tmp_path):
    image,_,_,p,row=fixture();path=tmp_path/'source.png';cv2.imwrite(str(path),image)
    row['source_image']='source';expected=native_input(image,row,p)['packed']
    hashes={row['crop_id']:{'taller_sha256':hashlib.sha256(expected.tobytes()).hexdigest()}}
    cache=make_native_cache([row],{'source':p},{'source':{'path':str(path)}},RepresentationSettings(),hashes)
    assert np.array_equal(cache[0],expected) and not cache.flags.writeable
    assert list(tmp_path.iterdir())==[path]
    hashes[row['crop_id']]['taller_sha256']='0'*64
    with pytest.raises(ValueError):make_native_cache([row],{'source':p},{'source':{'path':str(path)}},RepresentationSettings(),hashes)


def test_primary_checkpoint_ties_use_fp_loss_then_earlier_epoch():
    a=dict(occupied_accuracy=.9,empty_false_positives=2,loss=.4)
    b=dict(occupied_accuracy=.9,empty_false_positives=3,loss=.1)
    assert checkpoint_key(a,3)<checkpoint_key(b,1)
    assert checkpoint_key(a,2)<checkpoint_key(a,3)
    c=dict(a,occupied_accuracy=.91)
    assert checkpoint_key(c,6)<checkpoint_key(a,1)


def test_occupied_metrics_do_not_hide_empty_false_positives():
    cm=np.zeros((13,13),int);cm[0,0]=90;cm[0,4]=10;cm[6,6]=5;cm[6,4]=3;cm[6,0]=2
    m=epoch_metrics(dict(confusion_matrix=cm.tolist(),loss=.5))
    assert m['occupied_accuracy']==.5 and m['empty_false_positives']==10
    assert m['empty_false_positive_rate']==.1


def test_seed_matched_architecture_initialization_hash_is_reproducible():
    torch.manual_seed(42);a=create_classifier(pretrained=False)
    torch.manual_seed(42);b=create_classifier(pretrained=False)
    assert state_hash(a)==state_hash(b)
    assert torch.count_nonzero(a.features[0][0].weight[:,3:])==0


def test_loader_tracks_actual_sample_order_and_preserves_binary_mask():
    rows=[dict(crop_id=str(i),split='train',class_id=i,geometry_review_required=False,quality_flags=[]) for i in range(2)]
    cache=np.zeros((2,4,224,224),np.uint8);cache[:,3,50:100,60:110]=1
    ds=TrackedDataset(rows,cache,split='train',seed=42,augmentation={'brightness':.1})
    ds.begin_epoch(0);a,_=ds[1];ds[0]
    assert ds.order==[1,0] and set(torch.unique(a[3]).tolist())<={0.,1.}
    digest=ds.order_sha256();ds.begin_epoch(0);b,_=ds[1];ds[0]
    assert digest==ds.order_sha256() and torch.equal(a,b)


def test_frozen_groups_remain_original_train_validation_only():
    config=json.loads(Path('configs/experiments/crop_representation_c3_v1.json').read_text())
    _,rows,_,_,leakage=historical_inputs(config)
    assert len(rows['train'])==7168 and len(rows['val'])==1600
    assert {r['position_id'] for r in rows['val']}=={'position_003','position_007','position_011'}
    assert not {r['position_id'] for r in rows['train']} & {r['position_id'] for r in rows['val']}
    assert leakage['unknown_session_sources']==160


@pytest.mark.parametrize('key',['final_holdout','historical_test_used','D3c_access_for_training_or_tuning','D2_used_for_training_or_selection'])
def test_prohibited_cohorts_are_rejected_before_data_loading(key):
    c={k:False for k in ['final_holdout','historical_test_used','D3c_access_for_training_or_tuning','D2_used_for_training_or_selection']};c[key]=True
    with pytest.raises(ValueError):historical_inputs(c)
