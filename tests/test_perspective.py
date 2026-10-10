from itertools import permutations

import cv2
import numpy as np
import pytest

from src.vision.perspective import order_corners, warp_board


@pytest.mark.parametrize("expected", [
    [[10, 10], [90, 10], [90, 90], [10, 90]],
    [[30, 10], [120, 30], [100, 110], [10, 90]],
    [[50, 0], [100, 50], [50, 100], [0, 50]],
])
def test_corner_order_is_independent_of_input_order(expected):
    for shuffled in permutations(expected):
        source = np.asarray(shuffled, dtype=np.float32)
        original = source.copy()
        ordered = order_corners(source)
        np.testing.assert_array_equal(ordered, expected)
        np.testing.assert_array_equal(source, original)
        assert ordered.dtype == np.float32


@pytest.mark.parametrize("points", [
    [[0, 0], [1, 1], [2, 2]],
    [[0, 0], [1, 1], [2, 2], [3, 3]],
    [[0, 0], [10, 0], [10, 10], [0, 0]],
    [[0, 0], [10, 0], [10, 10], [8, 2]],
    [[0, 0], [10, 0], [10, 10], [0, float("nan")]],
    [[0, 0], [10, 0], [10, 10], [0, float("inf")]],
    [["bad", 0], [10, 0], [10, 10], [0, 10]],
])
def test_invalid_corner_geometry(points):
    with pytest.raises(ValueError, match="Corners"):
        order_corners(points)


@pytest.mark.parametrize("channels", [None, 3, 4])
def test_identity_warp_preserves_pixels_and_source(channels):
    shape = (80, 80) if channels is None else (80, 80, channels)
    image = np.random.default_rng(1).integers(0, 256, shape, dtype=np.uint8)
    original = image.copy()
    corners = order_corners([[0, 0], [79, 0], [79, 79], [0, 79]])
    result = warp_board(image, corners, 80)
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(image, original)


def test_projective_checkerboard_restores_all_square_centers():
    rows, cols = np.indices((160, 160))
    board = (((rows // 20 + cols // 20) % 2) * 200 + 20).astype(np.uint8)
    flat = np.float32([[0, 0], [159, 0], [159, 159], [0, 159]])
    corners = np.float32([[60, 30], [250, 60], [280, 260], [20, 230]])
    photo = cv2.warpPerspective(board, cv2.getPerspectiveTransform(flat, corners), (320, 300))
    rectified = warp_board(photo, corners, 800)
    assert rectified.shape == (800, 800)
    assert rectified.dtype == np.uint8
    for row in range(8):
        for col in range(8):
            assert abs(int(rectified[row * 100 + 50, col * 100 + 50]) - int(board[row * 20 + 10, col * 20 + 10])) < 3


@pytest.mark.parametrize("size", [0, -8, 801, 800.0, True])
def test_invalid_warp_size(size):
    with pytest.raises(ValueError, match="positive integer divisible by 8"):
        warp_board(np.zeros((80, 80, 3), np.uint8), [[0, 0], [79, 0], [79, 79], [0, 79]], size)


def test_out_of_bounds_warp_corners():
    with pytest.raises(ValueError, match="source image bounds"):
        warp_board(np.zeros((80, 80, 3), np.uint8), [[0, 0], [80, 0], [80, 79], [0, 79]])


def test_extended_warp_preserves_real_pixels_outside_board():
    from src.vision.perspective import warp_board_with_context
    photo = np.zeros((160, 160, 3), np.uint8)
    photo[40:120, 40:120] = 90
    # A tall piece extends above its edge square, outside the playing area.
    photo[15:55, 45:55] = (10, 20, 250)
    corners = np.float32([[40, 40], [119, 40], [119, 119], [40, 119]])
    rectified = warp_board_with_context(photo, corners, 80, 30)
    assert rectified.board_bounds == (30, 30, 110, 110)
    assert rectified.image.shape == (140, 140, 3)
    np.testing.assert_array_equal(rectified.board_image, warp_board(photo, corners, 80))
    # Original y=20 maps to context y=10, and is not black padding.
    np.testing.assert_array_equal(rectified.image[10, 40], photo[20, 50])
    assert rectified.valid_mask[10, 40] == 255


def test_context_mask_distinguishes_missing_source_pixels():
    from src.vision.perspective import warp_board_with_context
    photo = np.full((80, 80, 3), 90, np.uint8)
    corners = np.float32([[0, 0], [79, 0], [79, 79], [0, 79]])
    rectified = warp_board_with_context(photo, corners, 80, 20)
    assert rectified.valid_mask[0, 0] == 0
    assert (rectified.image[0, 0] == 0).all()
    assert (rectified.valid_mask[20:100, 20:100] == 255).all()


@pytest.mark.parametrize("margin", [-1, True, 1.5])
def test_invalid_context_margin(margin):
    with pytest.raises(ValueError, match="Context margin"):
        warp_board(np.zeros((80, 80, 3), np.uint8), [[0, 0], [79, 0], [79, 79], [0, 79]], margin_pixels=margin)
