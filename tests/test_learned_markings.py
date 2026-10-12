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


def test_black_side_reads_reversed_files_and_upright_left_ranks():
    # Camera left-to-right is H..A; bottom-to-top on left is 8..1.
    seed=np.float32([[120,120],[440,120],[440,440],[120,440]])
    rows=[]
    for index,label in enumerate('HGFEDCBA'):
        rows.append({'label':label,'center':[140+40*index,460],'confidence':.95,'glyph_seed_turns':0})
    for rank in range(1,9):
        rows.append({'label':str(rank),'center':[100,100+40*rank],'confidence':.95,'glyph_seed_turns':0})
    result=fit_marking_geometry(rows,seed,(600,600,3))
    assert result['status']=='candidate' and result['rotation_quarters_ccw']==2
    np.testing.assert_allclose(result['corners'],seed[[1,0,3,2]],atol=.1)


@pytest.mark.parametrize('turns',[0,1,2,3])
def test_printed_edge_pose_is_checked_for_each_player_side(turns):
    seed=np.float32([[120,120],[440,120],[440,440],[120,440]])
    h=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),seed[[[3,2,1,0],[0,3,2,1],[1,0,3,2],[2,1,0,3]][turns]])
    rows=observations(h)
    for index,row in enumerate(rows):row['glyph_seed_turns']=(turns+2*(index%2))%4
    result=fit_marking_geometry(rows,seed,(600,600,3))
    assert result['status']=='candidate' and result['rotation_quarters_ccw']==turns
    wrong=[r|{'glyph_seed_turns':(r['glyph_seed_turns']+1)%4} for r in rows]
    assert fit_marking_geometry(wrong,seed,(600,600,3))['status']=='uncertain'


def test_perspective_rectified_observations_map_back_to_original_pixels(monkeypatch):
    import src.scanning.marking_model as reader
    seed=np.float32([[70,80],[510,110],[480,490],[110,530]])
    def read(plane,threshold,retain_poses):
        assert retain_poses and plane.shape==(1000,1000,3)
        return [{'label':'H','confidence':.99,'center':[150.,950.],'glyph_turns':0,'polygon':[[140,940],[160,940],[160,960],[140,960]]}]
    monkeypatch.setattr(reader,'read_components',read)
    rows=reader.read_plane_components(np.zeros((600,600,3),np.uint8),seed)
    assert len(rows)==1 and rows[0]['label']=='H' and rows[0]['glyph_turns']==0
    transform=cv2.getPerspectiveTransform(seed,np.float32([[100,100],[900,100],[900,900],[100,900]]))
    center=cv2.perspectiveTransform(np.float32(rows[0]['center']).reshape(1,1,2),transform).reshape(2)
    np.testing.assert_allclose(center,[150,950],atol=.01)


def test_wooden_table_outline_cannot_hide_the_marked_board():
    from src.scanning.learned_markings import coarse_board_seed,coarse_board_seeds,fit_alternative_seeds
    # Low-saturation wood table merges with the board under the old single mask.
    hsv=np.full((800,800,3),[15,50,190],np.uint8)
    hsv[215:605,215:605]=[15,140,160]
    image=cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    playing=np.float32([[250,250],[570,250],[570,570],[250,570]])
    h=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),playing[[1,0,3,2]])
    rows=observations(h)
    wrong=coarse_board_seed(image)
    assert fit_marking_geometry(rows,wrong,image.shape)['status']=='uncertain'
    seeds=coarse_board_seeds(image)
    assert len(seeds)<=6
    result=fit_alternative_seeds(rows,seeds,image.shape)
    assert result['status']=='candidate' and result['rotation_quarters_ccw']==2
    np.testing.assert_allclose(result['corners'],playing[[1,0,3,2]],atol=.1)


def test_two_equally_supported_boards_require_review_instead_of_selecting_by_color():
    from src.scanning.learned_markings import fit_alternative_seeds
    first=np.float32([[120,120],[440,120],[440,440],[120,440]])
    second=first+[650,0]
    canonical=np.float32([[0,0],[32,0],[32,32],[0,32]])
    rows=observations(cv2.getPerspectiveTransform(canonical,first[[3,2,1,0]]))+observations(cv2.getPerspectiveTransform(canonical,second.astype(np.float32)[[3,2,1,0]]))
    result=fit_alternative_seeds(rows,[first,second],(600,1300,3))
    assert result['status']=='uncertain' and result['reason']=='competing_alternative_seed_geometries'


def test_component_proposals_keep_narrow_printed_characters_at_every_quarter_turn():
    from src.scanning.marking_model import components
    image=np.zeros((100,100,3),np.uint8)
    image[35:53,46:49]=230  # A narrow numeral remains a valid glyph when horizontal.
    for turns in range(4):
        rows=components(np.rot90(image,turns).copy())
        assert len(rows)==1
        assert rows[0]['glyph'].shape==(48,36)


@pytest.mark.parametrize('turns',[0,1,2,3])
def test_fast_plane_sampling_resolves_raw_pose_uncertainty_without_checker_search(monkeypatch,turns):
    import src.scanning.learned_markings as spatial
    import src.scanning.marking_model as reader
    seed=np.float32([[120,120],[440,120],[440,440],[120,440]])
    order=[[3,2,1,0],[0,3,2,1],[1,0,3,2],[2,1,0,3]][turns]
    h=cv2.getPerspectiveTransform(np.float32([[0,0],[32,0],[32,32],[0,32]]),seed[order])
    rows=observations(h)
    for index,row in enumerate(rows):row.update(sampling='rectified_board_plane',glyph_turns=(turns+2*(index%2))%4)
    monkeypatch.setattr(spatial,'coarse_board_seeds',lambda image:[seed])
    monkeypatch.setattr(reader,'read_plane_components',lambda *args:rows)
    result=spatial.fast_marking_geometry(np.zeros((600,600,3),np.uint8),[])
    assert result['status']=='candidate' and result['acquisition_path']=='fast_marking_planes'
    np.testing.assert_allclose(result['corners'],seed[order],atol=.1)


def test_fast_plane_sampling_still_rejects_two_equally_supported_boards(monkeypatch):
    import src.scanning.learned_markings as spatial
    import src.scanning.marking_model as reader
    first=np.float32([[120,120],[440,120],[440,440],[120,440]])
    second=(first+[650,0]).astype(np.float32)
    monkeypatch.setattr(spatial,'coarse_board_seeds',lambda image:[first,second])
    canonical=np.float32([[0,0],[32,0],[32,32],[0,32]])
    monkeypatch.setattr(reader,'read_plane_components',lambda image,seed,threshold:observations(cv2.getPerspectiveTransform(canonical,seed[[3,2,1,0]])))
    result=spatial.fast_marking_geometry(np.zeros((600,1300,3),np.uint8),[])
    assert result['status']=='uncertain' and result['reason']=='competing_alternative_seed_geometries'
