import cv2
import numpy as np
import pytest

from src.vision.orientation import (OrientationSettings, load_templates, resolve_orientation,
                                    rotate_rectified)
from src.vision.perspective import RectifiedBoard


def marked_board():
    image = np.full((1200, 1200, 3), (45, 65, 95), np.uint8)
    bank = load_templates()
    for edge in ('bottom', 'top', 'left', 'right'):
        for i in range(8):
            label = 'ABCDEFGH'[i] if edge in ('bottom', 'top') else str(8-i)
            glyph = cv2.resize(bank[label], (18, 24), interpolation=cv2.INTER_AREA)
            if edge in ('top', 'right'):
                glyph = np.rot90(glyph, 2)
            cx, cy = {'bottom': (250+i*100, 1050), 'top': (250+i*100, 150),
                      'left': (150, 250+i*100), 'right': (1050, 250+i*100)}[edge]
            patch = image[cy-12:cy+12, cx-9:cx+9]
            patch[glyph > 100] = (230, 240, 250)
    image[200:1000, 200:1000] = (90, 100, 110)
    image[220:250, 220:250] = (10, 20, 30)
    return RectifiedBoard(image, np.full((1200, 1200), 255, np.uint8), (200, 200, 1000, 1000))


@pytest.mark.parametrize('rotation', range(4))
def test_markings_resolve_all_four_rotations_without_piece_information(rotation):
    original = marked_board()
    rotated = rotate_rectified(original, rotation)
    result = resolve_orientation(rotated)
    assert result.status == 'resolved', result.metadata()
    assert result.rotation_quarters_ccw == (-rotation) % 4
    normalized = rotate_rectified(rotated, result.rotation_quarters_ccw)
    np.testing.assert_array_equal(normalized.image, original.image)
    np.testing.assert_array_equal(normalized.valid_mask, original.valid_mask)
    assert normalized.board_bounds == original.board_bounds


def test_blank_or_missing_border_is_uncertain():
    original = marked_board()
    # Pieces/grid colors cannot determine orientation when border markings vanish.
    original.image[:200] = 90
    original.image[1000:] = 90
    original.image[:, :200] = 90
    original.image[:, 1000:] = 90
    assert resolve_orientation(original).status == 'uncertain'
    original = marked_board()
    original.valid_mask[:200] = 0
    original.valid_mask[1000:] = 0
    original.valid_mask[:, :200] = 0
    original.valid_mask[:, 1000:] = 0
    assert resolve_orientation(original).status == 'uncertain'


def test_single_readable_edge_does_not_establish_orientation():
    original = marked_board()
    original.image[:200] = 90
    original.image[:, :200] = 90
    original.image[:, 1000:] = 90
    assert resolve_orientation(original).status == 'uncertain'


def test_ambiguous_repeated_markings_are_uncertain():
    original = marked_board()
    original.image[...] = 90
    assert resolve_orientation(original).rotation_quarters_ccw is None


def test_rotate_asymmetric_bounds_and_coverage():
    image = np.zeros((100, 140, 3), np.uint8)
    mask = np.zeros((100, 140), np.uint8)
    mask[10:50, 20:60] = 255
    board = RectifiedBoard(image, mask, (20, 10, 60, 50))
    rotated = rotate_rectified(board, 1)
    assert rotated.board_bounds == (10, 80, 50, 120)
    assert rotated.image.shape == (140, 100, 3)
    assert (rotated.valid_mask[80:120, 10:50] == 255).all()
    np.testing.assert_array_equal(rotate_rectified(rotated, 3).valid_mask, mask)


@pytest.mark.parametrize('options', [{'min_matches': 0}, {'min_score_gap': 0}, {'min_similarity': 2}])
def test_invalid_orientation_settings(options):
    with pytest.raises(ValueError):
        OrientationSettings(**options)


def test_mirrored_input_is_not_silently_assigned_chess_coordinates():
    original = marked_board()
    mirrored = RectifiedBoard(original.image[:, ::-1].copy(), original.valid_mask[:, ::-1].copy(), original.board_bounds)
    assert resolve_orientation(mirrored).status == 'uncertain'


def test_missing_template_bank_returns_uncertainty(monkeypatch):
    import src.vision.orientation as orientation
    def unavailable():
        raise FileNotFoundError('No bank')
    monkeypatch.setattr(orientation, 'load_templates', unavailable)
    result = orientation.resolve_orientation(marked_board())
    assert result.status == 'uncertain'
    assert 'unavailable' in result.reason


@pytest.mark.parametrize('angle', [17, 43, 95, 187, 275])
def test_arbitrary_camera_roll_is_resolved_after_rectification(angle):
    from src.vision.perspective import order_corners, warp_board_with_context
    original = marked_board()
    # Simulate a rolled camera view; geometry supplies image-relative corners.
    affine = cv2.getRotationMatrix2D((600, 600), angle, 1)
    affine[:, 2] += 300
    photograph = cv2.warpAffine(original.image, affine, (1800, 1800))
    source_corners = np.float32([[200, 200], [999, 200], [999, 999], [200, 999]])
    corners = cv2.transform(source_corners[None], affine)[0]
    rectified = warp_board_with_context(photograph, order_corners(corners), 800, 200)
    result = resolve_orientation(rectified)
    assert result.status == 'resolved', result.metadata()
    corrected = rotate_rectified(rectified, result.rotation_quarters_ccw)
    # Unique corner patch must land at a8 again, not another corner.
    center = corrected.board_image[30:40, 30:40].mean(axis=(0, 1))
    np.testing.assert_allclose(center, [10, 20, 30], atol=3)
