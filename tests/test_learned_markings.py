import cv2
import numpy as np
import pytest
from src.scanning.learned_markings import marker_positions,fit_marking_geometry


def observations(h):
    result=[]
    for label in 'ABCDEFGH12345678':
        for point in marker_positions(label):
            p=cv2.perspectiveTransform(np.float32(point).reshape(1,1,2),h).reshape(2)
            result.append({'label':label,'center':p.tolist(),'confidence':.95})
    return result


@pytest.mark.parametrize('turns',[0,1,2,3])
def test_ocr_registration_preserves_all_four_semantic_orientations(turns):
    canonical=np.float32([[0,0],[32,0],[32,32],[0,32]])
    seed=np.float32([[120,120],[440,100],[460,430],[100,450]])
    order=[[3,2,1,0],[0,3,2,1],[1,0,3,2],[2,1,0,3]][turns]
    h=cv2.getPerspectiveTransform(canonical,seed[order])
    result=fit_marking_geometry(observations(h),seed,(600,600,3))
    assert result['status']=='candidate'
    np.testing.assert_allclose(result['corners'],seed[order],atol=.1)
    assert result['distinct_files']==8 and result['distinct_ranks']==8


def test_single_axis_cannot_establish_board_geometry():
    seed=np.float32([[120,120],[440,120],[440,440],[120,440]])
    h=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),seed[[3,2,1,0]])
    labels=[o for o in observations(h) if o['label'] in 'ABCDEFGH']
    assert fit_marking_geometry(labels,seed,(600,600,3))['status']=='uncertain'


def test_corrects_shifted_seed_using_recognized_marking_positions():
    seed=np.float32([[120,120],[440,120],[440,440],[120,440]])
    h=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),seed[[3,2,1,0]])
    result=fit_marking_geometry(observations(h),seed+[40,40],(600,600,3))
    assert result['status']=='candidate'
    np.testing.assert_allclose(result['corners'],seed[[3,2,1,0]],atol=.1)


def test_incorrect_letters_do_not_create_a_consistent_grid():
    seed=np.float32([[120,120],[440,120],[440,440],[120,440]])
    rows=[{'label':'A','center':[200+i*8,200+i*3],'confidence':.99} for i in range(20)]
    assert fit_marking_geometry(rows,seed,(600,600,3))['status']=='uncertain'


def test_repeated_rotations_of_one_glyph_do_not_count_as_many_observations():
    seed=np.float32([[120,120],[440,120],[440,440],[120,440]])
    rows=[{'label':label,'center':[180,460],'confidence':.99} for label in 'ABCDEFGH12345678']*3
    assert fit_marking_geometry(rows,seed,(600,600,3))['status']=='uncertain'


def test_mirrored_board_mapping_is_rejected():
    seed=np.float32([[120,120],[440,120],[440,440],[120,440]])
    h=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),seed[[2,3,0,1]])
    assert fit_marking_geometry(observations(h),seed,(600,600,3))['status']=='uncertain'


def test_glyph_rotations_have_one_input_shape():
    from src.scanning.marking_model import rotated_glyph
    glyph=np.zeros((48,36),np.uint8);glyph[8:40,15:20]=255
    for k in range(4):assert rotated_glyph(glyph,k).shape==(48,36)
