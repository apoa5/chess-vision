"""C.1 geometry-only prototypes: original pixels, alignment and visibility diagnostics."""
import copy
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
from src.vision.input_representations import (RepresentationSettings,extract_representations,height_envelope,raw_letterbox,visible_bbox_coverage,jacobian_anisotropy)
from scripts.run_square_evidence import make_inputs
from src.vision.geometry_audit import project


def scene():
    width,height=1200,1000;K=np.array([[900.,0,width/2],[0,900.,height/2],[0,0,1.]])
    eye=np.array([-6.,-10.,12.]);forward=np.array([4.,4.,0.])-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right);down=np.cross(forward,right);R=np.stack([right,down,forward]);t=-R@eye
    t=t+8*R[:,1];R=R@np.diag([1.,-1.,1.])
    H=K@np.column_stack([R[:,:2],t]);quad=project(np.float32([[0,0],[8,0],[8,8],[0,8]]),H)
    polygons=[project(np.float32([[c,r],[c+1,r],[c+1,r+1],[c,r+1]]),H).tolist() for r in range(8) for c in range(8)]
    g=dict(source_square_polygons=polygons,semantic_square_ids=[f'{f}{r}' for r in range(8,0,-1) for f in 'abcdefgh'],provenance=dict(source_corners=quad.tolist()))
    yy,xx=np.indices((height,width));image=np.stack([xx%251,yy%251,(xx+yy)%251],axis=-1).astype(np.uint8)
    config=json.loads(Path('configs/experiments/evidence_d2.json').read_text())
    return image,g,config,K,R,t


def test_baseline_matches_unchanged_frozen_extractor_exactly():
    image,g,c,*_=scene();before=image.copy();gbefore=copy.deepcopy(g)
    x=extract_representations(image,g,'d4',c);expected=next(i for i in make_inputs(image,g,c) if i['canonical_square']=='d4')
    assert np.array_equal(x['b2']['packed'],expected['packed'])
    assert np.array_equal(image,before) and g==gbefore
    for n in ('b2','expanded','taller'):
        p=x[n]['packed'];assert p.shape==(4,224,224) and p.dtype==np.uint8 and set(np.unique(p[3]))<={0,1}
    assert x['two_branch']['local'] is x['taller'] and x['two_branch']['context'] is x['expanded']
    assert not x['two_branch']['classifier_implemented']


def test_perspective_envelope_recovers_known_synthetic_top_projection():
    image,g,c,K,R,t=scene();q=np.float32([g['source_square_polygons'][0][0],g['source_square_polygons'][7][1],g['source_square_polygons'][63][2],g['source_square_polygons'][56][3]])
    target=np.float32(g['source_square_polygons'][27]);env,meta=height_envelope(q,target,image.shape[1],image.shape[0],2.)
    assert meta['mode']=='assumed_pinhole_normal' and not meta['metric_pose_verified']
    assert meta['focal_px']==pytest.approx(900,rel=.001)
    xy=np.float32([[3,3],[4,3],[4,4],[3,4]]);points=np.column_stack([xy,np.full(4,2.)]);camera=points@R.T+t;uv=camera@K.T;uv=uv[:,:2]/uv[:,2:]
    assert np.allclose(env[4:],uv,atol=.1)


def test_overhead_falls_back_explicitly_instead_of_inventing_pose():
    q=np.float32([[100,100],[900,100],[900,900],[100,900]]);target=np.float32([[300,300],[400,300],[400,400],[300,400]])
    env,m=height_envelope(q,target,1000,1000,2)
    assert m['mode']=='image_up_fallback' and m['focal_px'] is None and not m['metric_pose_verified']
    assert np.array_equal(env[:4],target)


def test_letterbox_does_not_assume_piece_is_on_board_plane():
    image=np.full((400,500,3),80,np.uint8);target=np.float32([[200,220],[250,220],[250,260],[200,260]])
    envelope=np.vstack([target,target+[0,-140]])
    x=raw_letterbox(image,target,envelope,RepresentationSettings())
    assert x['aspect_preserving'] and set(np.unique(x['packed'][3]))<={0,1}
    assert jacobian_anisotropy(x['raw_to_input'],[225,240])<1.02
    assert visible_bbox_coverage([210,90,240,250],x['source_support'])==pytest.approx(1)


def test_coverage_is_bbox_proxy_not_unseen_silhouette():
    support=np.float32([[0,50],[100,50],[100,100],[0,100]])
    assert visible_bbox_coverage([0,0,100,100],support)==pytest.approx(.5)
    assert visible_bbox_coverage([0,60,100,90],support)==pytest.approx(1)
    with pytest.raises(ValueError):visible_bbox_coverage([1,1,0,0],support)


def test_projective_warp_can_distort_nonplanar_features_with_perfect_base_geometry():
    # A generic projective map has position-dependent anisotropy, whereas raw
    # letterboxing is affine and nearly isotropic. This does not test accuracy.
    M=np.array([[1.,0,0],[0,1.,0],[0,.004,1.]])
    assert jacobian_anisotropy(M,[100,100])!=pytest.approx(jacobian_anisotropy(M,[100,400]))


@pytest.mark.parametrize('changes',[{'size':0},{'height_squares':float('nan')},{'height_squares':0},{'raw_margin_squares':-1}])
def test_invalid_settings_rejected(changes):
    with pytest.raises(ValueError):RepresentationSettings(**changes)


def test_duplicate_or_nonfinite_source_geometry_rejected():
    image,g,c,*_=scene();bad=copy.deepcopy(g);bad['semantic_square_ids'][0]=bad['semantic_square_ids'][1]
    with pytest.raises(ValueError):extract_representations(image,bad,'d4',c)
    bad=copy.deepcopy(g);bad['source_square_polygons'][0][0][0]=float('nan')
    with pytest.raises(ValueError):extract_representations(image,bad,'d4',c)


def test_deterministic_outputs_and_no_truth_argument():
    image,g,c,*_=scene();a=extract_representations(image,g,'d4',c);b=extract_representations(image,g,'d4',c)
    for name in ('b2','taller','expanded'):
        assert np.array_equal(a[name]['packed'],b[name]['packed'])
        assert np.array_equal(a[name]['raw_to_input'],b[name]['raw_to_input'])


def test_canonical_labels_do_not_rotate_original_piece_pixels():
    image,g,c,*_=scene();a=extract_representations(image,g,'d4',c)
    index=g['semantic_square_ids'].index('d4');ids=g['semantic_square_ids'];g=copy.deepcopy(g);g['semantic_square_ids']=ids[16:]+ids[:16]
    b=extract_representations(image,g,g['semantic_square_ids'][index],c)
    assert np.array_equal(a['b2']['packed'],b['b2']['packed'])
