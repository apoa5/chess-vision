"""Offline display-coordinate and timestamp pairing regressions."""
from fractions import Fraction
import cv2
import numpy as np
import pytest
from src.scanning.pilot_comparison import display_image,display_transform,correspondence_score,nearest_time,register_pair


@pytest.mark.parametrize('angle',[-90,0,90,180])
def test_display_pixels_and_explicit_coordinate_transform_agree(angle):
    image=np.arange(3*5*3,dtype=np.uint8).reshape(3,5,3)
    rendered=display_image(image,angle);matrix=display_transform(5,3,angle)
    for x,y in [(0,0),(4,0),(4,2),(0,2),(2,1)]:
        xx,yy,_=matrix@np.array([x,y,1]);np.testing.assert_array_equal(rendered[int(yy),int(xx)],image[y,x])
    assert image.shape==(3,5,3)  # source pixels remain unmodified


def test_unknown_display_transform_is_rejected():
    with pytest.raises(ValueError):display_image(np.zeros((3,5,3),np.uint8),45)
    with pytest.raises(ValueError):display_transform(5,3,45)


def test_pairing_uses_exact_pts_instead_of_nominal_frame_index():
    times=[Fraction(0),Fraction(1,30),Fraction(1,10)]
    assert nearest_time(times,Fraction(9,100))==2
    assert nearest_time(times,Fraction(1,60))==0
    with pytest.raises(ValueError):nearest_time([],0)


def test_correspondence_is_invariant_to_linear_brightness_but_rejects_wrong_frame():
    rng=np.random.default_rng(4);a=rng.uniform(30,180,(212,116)).astype(np.float32)
    assert correspondence_score(a,a*.8+20)>.999
    assert correspondence_score(a,np.roll(a,10,axis=0))<.1
    with pytest.raises(ValueError):correspondence_score(np.ones((4,4)),np.ones((4,4)))


def test_cross_version_registration_requires_visual_support():
    result=register_pair(np.zeros((192,108,3),np.uint8),np.zeros((84,46,3),np.uint8))
    assert result['status']=='unavailable'


def test_paired_geometry_compares_semantic_ids_after_cross_version_registration():
    from scripts.compare_pilot_resolutions import agreement
    low=dict(completed=True,returned_polygon=[[10,10],[30,10],[30,30],[10,30]],semantic_square_ids=['a1','b1'],source_square_polygons=[[[10,10],[20,10],[20,20],[10,20]],[[20,10],[30,10],[30,20],[20,20]]],rotation_quarters_ccw=0)
    high=dict(completed=True,returned_polygon=(np.asarray(low['returned_polygon'])*2).tolist(),semantic_square_ids=['b1','a1'],source_square_polygons=(np.asarray(low['source_square_polygons'])[::-1]*2).tolist(),rotation_quarters_ccw=0)
    registered=dict(status='registered',high_display_to_low_pixels=np.diag([.5,.5,1]).tolist())
    result=agreement(low,high,registered)
    assert result['corner_rms_low_pixels']==0
    assert result['semantic_square_centre_rms_low_pixels']==0
    high['semantic_square_ids'].reverse()
    assert agreement(low,high,registered)['semantic_square_centre_rms_low_pixels']==10
    assert agreement(low,high,{'status':'unavailable'}) is None
