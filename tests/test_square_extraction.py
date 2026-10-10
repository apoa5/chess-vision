import numpy as np
import pytest

from src.vision.square_extraction import extract_squares
from src.utils.visualization import draw_board_grid


@pytest.mark.parametrize("size,channels", [(800, 3), (160, None), (64, 4)])
def test_crops_cover_source_exactly_and_preserve_indices(size, channels):
    shape = (size, size) if channels is None else (size, size, channels)
    source = np.random.default_rng(2).integers(0, 256, shape, dtype=np.uint8)
    original = source.copy()
    squares = extract_squares(source)
    assert len(squares) == 64
    rebuilt = np.empty_like(source)
    side = size // 8
    for index, crop in enumerate(squares):
        assert (crop.row, crop.col) == divmod(index, 8)
        x0, y0, x1, y1 = crop.bounds
        assert crop.bounds == (crop.col * side, crop.row * side, (crop.col + 1) * side, (crop.row + 1) * side)
        assert crop.image.shape[:2] == (side, side)
        rebuilt[y0:y1, x0:x1] = crop.image
    np.testing.assert_array_equal(rebuilt, source)
    squares[0].image[...] = 0
    np.testing.assert_array_equal(source, original)


@pytest.mark.parametrize("shape", [(80, 81, 3), (81, 81, 3), (0, 0, 3)])
def test_invalid_board_shape(shape):
    with pytest.raises(ValueError):
        extract_squares(np.zeros(shape, np.uint8))


def test_grid_matches_crop_boundaries_without_changing_board():
    board = np.full((800, 800, 3), 90, np.uint8)
    grid = draw_board_grid(board)
    assert grid.shape == board.shape
    for coordinate in range(100, 800, 100):
        np.testing.assert_array_equal(grid[50, coordinate], [0, 255, 0])
        np.testing.assert_array_equal(grid[coordinate, 50], [0, 255, 0])
    assert (board == 90).all()


def test_grid_rejects_non_normalized_board():
    with pytest.raises(ValueError, match="square"):
        draw_board_grid(np.zeros((80, 81, 3), np.uint8))


def test_context_crops_keep_target_coordinates_and_edge_piece():
    from src.vision.perspective import warp_board_with_context
    from src.vision.square_extraction import extract_context_squares
    image = np.zeros((160, 160, 3), np.uint8)
    image[30:50, 45:55] = (10, 20, 250)
    rectified = warp_board_with_context(image, np.float32([[40, 40], [119, 40], [119, 119], [40, 119]]), 80, 20)
    crops = extract_context_squares(rectified, 10)
    assert len(crops) == 64
    for index, (crop, mask) in enumerate(crops):
        assert (crop.row, crop.col) == divmod(index, 8)
        assert crop.image.shape == (30, 30, 3)
        assert mask.shape == (30, 30)
        np.testing.assert_array_equal(crop.image[10:20, 10:20],
                                      rectified.board_image[crop.row*10:(crop.row+1)*10, crop.col*10:(crop.col+1)*10])
    np.testing.assert_array_equal(crops[0][0].image[0, 15], image[30, 45])
    with pytest.raises(ValueError, match="context margin"):
        extract_context_squares(rectified, 21)
