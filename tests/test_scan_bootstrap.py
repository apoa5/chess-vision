import cv2
import numpy as np
import pytest
from src.scanning.bootstrap import semantic_mapping,bootstrap_image
from src.scanning.replay import _geometry
from src.vision.orientation import OrientationResult
from src.vision.perspective import RectifiedBoard


@pytest.mark.parametrize('turns,a1_index,h8_index',[(0,56,7),(1,0,63),(2,7,56),(3,63,0)])
def test_camera_side_mapping(turns,a1_index,h8_index):
    ids=semantic_mapping(turns)
    assert ids[7,0]==a1_index and ids[0,7]==h8_index
    assert set(ids.ravel())==set(range(64))


def test_uncertain_markings_never_produce_square_ids(monkeypatch):
    import src.scanning.bootstrap as b
    class Detection:
        success=True;reason=None;score=.9
        corners=np.float32([[20,20],[180,20],[180,180],[20,180]])
    monkeypatch.setattr(b,'detect_board_corners',lambda *args:Detection())
    monkeypatch.setattr(b,'rectify_with_config',lambda *args:RectifiedBoard(np.zeros((200,200,3),np.uint8),np.full((200,200),255,np.uint8),(20,20,181,181)))
    monkeypatch.setattr(b,'resolve_orientation',lambda *args:OrientationResult('uncertain'))
    result=bootstrap_image(np.zeros((200,200,3),np.uint8),{})
    assert result['reason']=='orientation_uncertain'
    assert result['square_polygons'] is None and result['geometry'] is None


def test_manual_and_markings_conflict_requires_review(monkeypatch):
    import src.scanning.bootstrap as b
    monkeypatch.setattr(b,'rectify_with_config',lambda *args:None)
    monkeypatch.setattr(b,'resolve_orientation',lambda *args:OrientationResult('resolved',2))
    result=bootstrap_image(np.zeros((200,200,3),np.uint8),{},[[20,180],[180,180],[180,20],[20,20]])
    assert result['reason']=='manual_orientation_conflicts_with_markings'
    assert result['geometry'] is None


@pytest.mark.parametrize('turns',[0,1,2,3])
def test_refined_geometry_preserves_semantic_corners_and_source(monkeypatch,turns):
    import src.scanning.bootstrap as b
    image=np.zeros((200,200,3),np.uint8);before=image.copy()
    seed=np.float32([[20,20],[180,20],[180,180],[20,180]])
    order=[[3,2,1,0],[0,3,2,1],[1,0,3,2],[2,1,0,3]][turns]
    monkeypatch.setattr(b,'rectify_with_config',lambda *args:RectifiedBoard(image,np.full((200,200),255,np.uint8),(20,20,181,181)))
    monkeypatch.setattr(b,'resolve_orientation',lambda *args:OrientationResult('resolved',turns))
    yy,xx=np.meshgrid(np.linspace(20,180,9),np.linspace(20,180,9),indexing='ij');points=np.stack([xx,yy],axis=-1)
    polys=[[points[r,c],points[r,c+1],points[r+1,c+1],points[r+1,c]] for r in range(8) for c in range(8)]
    monkeypatch.setattr(b,'fit_grid',lambda *args,**kwargs:{'fit_status':'fitted','boundary_evidence_fraction':1.,'intersections':points.tolist(),'square_polygons':np.asarray(polys).tolist(),'method':'test_fixture'})
    result=bootstrap_image(image,{},seed[order])
    assert result['status']=='candidate'
    np.testing.assert_allclose(result['geometry']['corners'],seed[order],atol=.01)
    assert len(result['square_polygons'])==64
    np.testing.assert_array_equal(image,before)
    _geometry(result['geometry']|{'square_polygons':result['square_polygons']},200,200)


def test_competing_shifted_marking_hypotheses_do_not_assign_squares(monkeypatch):
    import src.scanning.bootstrap as b
    class Detection:
        success=True;reason=None;score=.9
        corners=np.float32([[40,40],[200,40],[200,200],[40,200]])
    image=np.zeros((300,300,3),np.uint8)
    monkeypatch.setattr(b,'detect_board_corners',lambda *args:Detection())
    monkeypatch.setattr(b,'rectify_with_config',lambda *args:RectifiedBoard(image,np.full((300,300),255,np.uint8),(40,40,201,201)))
    calls=iter([OrientationResult('uncertain')]+[OrientationResult('resolved',0,score=.75)]*8)
    monkeypatch.setattr(b,'resolve_orientation',lambda *args:next(calls))
    result=bootstrap_image(image,{})
    assert result['reason']=='competing_geometry_marking_hypotheses'
    assert result['square_polygons'] is None


@pytest.mark.parametrize('localized',[True,False])
def test_learned_reader_uses_image_labels_and_recovers_orientation(monkeypatch,localized):
    import src.scanning.bootstrap as b
    import src.scanning.marking_model as reader
    import src.scanning.learned_markings as spatial
    image=np.zeros((200,200,3),np.uint8)
    seed=np.float32([[20,20],[180,20],[180,180],[20,180]])
    class Detection:
        success=localized;reason=None;score=.9;corners=seed if localized else None
    monkeypatch.setattr(spatial,'coarse_board_seed',lambda *args:seed)
    monkeypatch.setattr(b,'detect_board_corners',lambda *args:Detection())
    monkeypatch.setattr(reader,'read_components',lambda *args,**kwargs:[{'label':'A','center':[30,190],'confidence':.9}])
    monkeypatch.setattr(reader,'read_plane_components',lambda *args,**kwargs:pytest.fail('A working original reader must not be replaced by the plane reader'))
    monkeypatch.setattr(spatial,'fit_marking_geometry',lambda *args:{'status':'candidate','corners':seed[[3,2,1,0]].tolist(),'inliers':12})
    monkeypatch.setattr(b,'rectify_with_config',lambda *args:RectifiedBoard(image,np.full((200,200),255,np.uint8),(20,20,181,181)))
    monkeypatch.setattr(b,'resolve_orientation',lambda *args:OrientationResult('uncertain'))
    yy,xx=np.meshgrid(np.linspace(20,180,9),np.linspace(20,180,9),indexing='ij');points=np.stack([xx,yy],axis=-1)
    polys=[[points[r,c],points[r,c+1],points[r+1,c+1],points[r+1,c]] for r in range(8) for c in range(8)]
    monkeypatch.setattr(b,'fit_grid',lambda *args,**kwargs:{'fit_status':'fitted','boundary_evidence_fraction':1.,'intersections':points.tolist(),'square_polygons':np.asarray(polys).tolist(),'method':'test_fixture'})
    result=bootstrap_image(image,{'learned_markings':{'enabled':True}})
    assert result['status']=='candidate' and result['orientation_source']=='learned_border_markings'
    assert result['rotation_quarters_ccw']==0 and len(result['square_polygons'])==64
    assert result['implementation_version']=='board_reader_multi_seed_v5'
    assert result['marking_attempts']==[{'path':'original_raw','status':'candidate','reason':None}]


def test_diagonal_detector_seed_is_reordered_before_perspective_correction(monkeypatch):
    import src.scanning.bootstrap as b
    from src.vision.perspective import order_corners
    # A refined detector seed can change which vertex minimizes x+y at 45 degrees.
    seed=np.float32([[373.24,813.28],[790.38,386.06],[1256.65,790.61],[805.17,1274.56]])
    assert not np.array_equal(seed,order_corners(seed))
    class Detection:
        success=True;reason=None;score=.9;corners=seed
    image=np.zeros((1600,1600,3),np.uint8)
    monkeypatch.setattr(b,'detect_board_corners',lambda *args:Detection())
    def rectify(image,corners,config):
        np.testing.assert_array_equal(corners,order_corners(corners))
        return RectifiedBoard(np.zeros((200,200,3),np.uint8),np.full((200,200),255,np.uint8),(20,20,181,181))
    monkeypatch.setattr(b,'rectify_with_config',rectify)
    monkeypatch.setattr(b,'resolve_orientation',lambda *args:OrientationResult('resolved',0))
    yy,xx=np.meshgrid(np.linspace(20,180,9),np.linspace(20,180,9),indexing='ij');points=np.stack([xx,yy],axis=-1)
    polys=[[points[r,c],points[r,c+1],points[r+1,c+1],points[r+1,c]] for r in range(8) for c in range(8)]
    monkeypatch.setattr(b,'fit_grid',lambda *args,**kwargs:{'fit_status':'fitted','boundary_evidence_fraction':1.,'intersections':points.tolist(),'square_polygons':np.asarray(polys).tolist(),'method':'test_fixture'})
    result=bootstrap_image(image,{})
    assert result['status']=='candidate' and len(result['square_polygons'])==64


def test_fast_acquisition_uses_recognized_markings_and_skips_checker_search(monkeypatch):
    import src.scanning.bootstrap as b
    import src.scanning.marking_model as reader
    import src.scanning.learned_markings as spatial
    image = np.zeros((600, 600, 3), np.uint8)
    seed = np.float32([[120,120],[440,120],[440,440],[120,440]])
    h = cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),seed[[1,0,3,2]])
    rows = [{'label':label,'confidence':.95,'center':cv2.perspectiveTransform(np.float32(point).reshape(1,1,2),h).reshape(2).tolist()} for label in 'ABCDEFGH12345678' for point in spatial.marker_positions(label)]
    monkeypatch.setattr(reader,'read_components',lambda *args,**kwargs:rows)
    monkeypatch.setattr(spatial,'coarse_board_seeds',lambda *args:[seed])
    monkeypatch.setattr(b,'detect_board_corners',lambda *args:pytest.fail('Verified markings should skip the checker candidate search'))
    monkeypatch.setattr(b,'rectify_with_config',lambda *args:RectifiedBoard(image,np.full((600,600),255,np.uint8),(120,120,441,441)))
    monkeypatch.setattr(b,'resolve_orientation',lambda *args:OrientationResult('uncertain'))
    yy,xx=np.meshgrid(np.linspace(120,440,9),np.linspace(120,440,9),indexing='ij');points=np.stack([xx,yy],axis=-1)
    polys=[[points[r,c],points[r,c+1],points[r+1,c+1],points[r+1,c]] for r in range(8) for c in range(8)]
    monkeypatch.setattr(b,'fit_grid',lambda *args,**kwargs:{'fit_status':'fitted','boundary_evidence_fraction':1.,'intersections':points.tolist(),'square_polygons':np.asarray(polys).tolist(),'method':'test_fixture'})
    result=bootstrap_image(image,{'learned_markings':{'enabled':True},'scan_fast_acquisition':True})
    assert result['status']=='candidate'
    assert result['orientation_source']=='learned_border_markings'
    assert result['detection']['source']=='verified_marking_outlines'
    assert result['rotation_quarters_ccw']==2
    np.testing.assert_allclose(result['geometry']['corners'],seed[[1,0,3,2]],atol=.01)


def test_fast_acquisition_keeps_classical_fallback_when_markings_are_uncertain(monkeypatch):
    import src.scanning.bootstrap as b
    import src.scanning.marking_model as reader
    import src.scanning.learned_markings as spatial
    class Detection:
        success=False;reason='no_board';score=None;corners=None
    calls=[]
    monkeypatch.setattr(reader,'read_components',lambda *args,**kwargs:[])
    monkeypatch.setattr(spatial,'coarse_board_seeds',lambda *args:[])
    monkeypatch.setattr(spatial,'coarse_board_seed',lambda *args:None)
    monkeypatch.setattr(b,'detect_board_corners',lambda *args:(calls.append(True) or Detection()))
    result=bootstrap_image(np.zeros((200,200,3),np.uint8),{'learned_markings':{'enabled':True},'scan_fast_acquisition':True})
    assert calls==[True]
    assert result['status']=='uncertain' and result['geometry'] is None
