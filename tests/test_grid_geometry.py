import cv2
import numpy as np
from src.vision.grid_geometry import fit_grid, polygon_crop


def test_grid_recovers_visible_edges_with_an_offset_seed_and_occlusion():
    image = np.full((480,480,3),100,np.uint8)
    for r in range(8):
        for c in range(8):
            image[80+r*40:80+(r+1)*40,80+c*40:80+(c+1)*40] = 200 if (r+c)%2 else 50
    cv2.circle(image,(180,140),15,(15,15,15),-1)
    result=fit_grid(image,[85,85,406,406])
    points=np.array(result['intersections'])
    assert len(result['square_polygons'])==64
    assert np.max(np.abs(points[2:7,2:7,0]-(80+np.arange(2,7)*40)[None,:]))<5
    assert np.max(np.abs(points[2:7,2:7,1]-(80+np.arange(2,7)*40)[:,None]))<5
    assert result['review_required']


def test_polygon_context_keeps_target_geometry_and_source_coverage():
    image=np.full((120,120,3),100,np.uint8)
    mask=np.full((120,120),255,np.uint8)
    polygon=[[5,6],[25,4],[28,26],[3,28]]
    crop, valid, tight, bounds, local, coverage=polygon_crop(image,mask,polygon,30)
    assert np.allclose(np.array(local)+bounds[:2],polygon)
    assert coverage==1
    assert np.any(valid==0)
    assert tight.shape[:2]==(25,26)


def test_review_expires_when_square_polygon_changes():
    from src.vision.dataset import board_review_signature
    record={'grid_geometry':{'square_polygons':[[[1,2],[3,2],[3,4],[1,4]]]}}
    previous=board_review_signature(record)
    record['grid_geometry']['square_polygons'][0][0][0]=2
    assert board_review_signature(record)!=previous


def test_known_checker_phase_prevents_one_rank_slip():
    image=np.full((480,480,3),100,np.uint8)
    for r in range(8):
        for c in range(8):
            image[80+r*40:80+(r+1)*40,80+c*40:80+(c+1)*40]=200 if (r+c)%2 else 50
    result=fit_grid(image,[80,104,401,425],light_top_left=False)
    points=np.array(result['intersections'])
    assert np.max(np.abs(points[2:7,2:7,1]-(80+np.arange(2,7)*40)[:,None]))<5
