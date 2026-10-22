"""B.1 guards for image geometry versus semantic chess coordinates."""
from itertools import permutations
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.scanning.recognition import prepare_keyframe
from src.vision.dataset import chess_square_at
from src.vision.dataset_loader import preprocess_context_crop
from src.vision.geometry_audit import context_transform, project, quadrilateral_metrics
from src.vision.perspective import order_corners, warp_board_with_context


@pytest.mark.parametrize('corners', [
    [[210, 170], [815, 240], [900, 850], [100, 780]],
    [[500, 100], [900, 500], [500, 900], [100, 500]],
])
def test_all_corner_permutations_keep_winding_and_exclusive_context_endpoints(corners):
    expected = np.float32(corners)
    for shuffled in permutations(corners):
        np.testing.assert_array_equal(order_corners(shuffled), expected)
        transform = context_transform(shuffled, (200, 200, 1000, 1000))
        np.testing.assert_allclose(project(expected, transform),
                                   [[200, 200], [999, 200], [999, 999], [200, 999]], atol=1e-4)
        np.testing.assert_allclose(project(project(expected, transform), np.linalg.inv(transform)),
                                   expected, atol=1e-4)


@pytest.mark.parametrize('turns', range(4))
def test_projective_live_targets_match_all_64_canonical_squares_without_rotating_pixels(turns):
    # Semantic corners a1/h1/h8/a8 deliberately have the reverse image winding.
    camera = np.float32([[210, 170], [815, 240], [900, 850], [100, 780]])
    semantic = camera[[[3, 2, 1, 0], [0, 3, 2, 1], [1, 0, 3, 2], [2, 1, 0, 3]][turns]]
    homography = cv2.getPerspectiveTransform(np.float32([[0, 0], [32, 0], [32, 32], [0, 32]]), semantic)
    # Asymmetric image makes a silent pixel rotation detectable.
    image = np.random.default_rng(12).integers(0, 256, (1000, 1000, 3), dtype=np.uint8)
    geometry = dict(source='tracking', version=1, corners=semantic.tolist(),
                    h_board_to_frame=homography.ravel().tolist())
    crops = prepare_keyframe(image, geometry)
    rect = warp_board_with_context(image, camera)
    padded = cv2.copyMakeBorder(rect.image, 300, 300, 300, 300, cv2.BORDER_CONSTANT)
    transform = context_transform(camera, rect.board_bounds)
    camera_ids = {(r, c): chess_square_at(r, c, turns) for r in range(8) for c in range(8)}
    for crop in crops:
        square = crop['chess_square']
        f, r = ord(square[0])-97, int(square[1])-1
        board_centre = np.float32([[(f+.5)*4, (r+.5)*4]])
        centre = project(project(board_centre, homography), transform)[0]
        col, row = np.floor((centre-200)/(799/8)).astype(int)
        assert camera_ids[row, col] == square
        bounds = crop['context_bounds']
        local_centre = centre-np.asarray(bounds[:2])
        # Projective camera image -> rectification makes a planar square regular again.
        np.testing.assert_allclose(np.mean(crop['target_polygon'], axis=0), local_centre, atol=2e-4)
        x0, y0, x1, y1 = bounds
        # floor/ceil of projected vertices may request one pixel beyond context;
        # polygon_crop intentionally zero-pads rather than wrapping negative indices.
        np.testing.assert_array_equal(crop['image'], padded[y0+300:y1+300, x0+300:x1+300])
        tensor = preprocess_context_crop(crop['image'], crop['target_polygon'])
        pixel = np.floor(local_centre*224/np.array([x1-x0, y1-y0])).astype(int)
        assert tensor[3, pixel[1], pixel[0]] == 1


def test_geometry_comparison_does_not_hide_localization_error():
    reference = np.float32([[100, 100], [900, 100], [900, 900], [100, 900]])
    same = quadrilateral_metrics(reference[[2, 0, 3, 1]], reference)
    assert same['iou'] == pytest.approx(1)
    assert same['rms_corner_error_squares'] == pytest.approx(0)
    shifted = quadrilateral_metrics(reference+[100, 0], reference)
    assert shifted['iou'] == pytest.approx(7/9)
    assert shifted['rms_corner_error_squares'] == pytest.approx(1)


@pytest.mark.parametrize('stem', ['006_var_05', '006_var_06', '006_var_07',
                                  '010_var_05', '010_var_06', '010_var_07', '016_var_06'])
def test_historical_reviewed_raw_square_centres_have_correct_semantic_identity(stem):
    from src.vision.dataset import file_hash, review_approved

    root = Path(__file__).resolve().parents[1]
    manifest = root/'data/classification/prepared_boards.json'
    if not manifest.exists():
        pytest.skip('Historical reviewed dataset is required')
    name = f'position_{stem}.jpg'
    prepared = next(r for r in json.loads(manifest.read_text()) if r['image_id'] == name)
    metadata = json.loads((root/'data/metadata/images.json').read_text())[name]
    reference = json.loads((root/'data/metadata/board_corner_overrides.json').read_text())[name]
    reviews = json.loads((root/'data/metadata/day9_board_reviews.json').read_text())
    assert review_approved(prepared, reviews[name])
    assert file_hash(root/'data/raw'/name) == reference['source_sha256']
    transform = context_transform(reference['corners'], prepared['board_bounds'])
    centres = project(prepared['grid_geometry']['source_square_polygons'], transform).mean(axis=1)
    camera_cells = np.floor((centres-200)/(799/8)).astype(int)
    # Explicit file/rank formulas are independent of dataset rotation helpers.
    mapping = {
        'white_near_camera': lambda r, c: (c, 8-r),
        'black_near_camera': lambda r, c: (7-c, r+1),
        'white_left': lambda r, c: (r, c+1),
        'white_right': lambda r, c: (7-r, 8-c),
    }[metadata['orientation']]
    for i, (col, row) in enumerate(camera_cells):
        assert (row, col) == divmod(i, 8)
        file, rank = mapping(row, col)
        assert chess_square_at(i//8, i%8, prepared['label_rotation']) == f'{chr(97+file)}{rank}'
