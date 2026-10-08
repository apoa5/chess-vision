import numpy as np
import pytest

from src.utils.image_io import load_image
from src.utils.visualization import (draw_board_corners, draw_points, draw_polygon,
                                     resize_for_preview, save_debug_visualization)


def test_debug_overlay_preserves_original_and_saves(tmp_path):
    image = np.full((160, 240, 3), 30, dtype=np.uint8)
    original = image.copy()
    corners = [[20, 20], [220, 20], [220, 140], [20, 140]]
    overlay = draw_board_corners(image, corners)
    np.testing.assert_array_equal(image, original)
    assert overlay.shape == image.shape
    assert not np.array_equal(overlay, image)
    assert np.any(overlay[:, :, 1] > original[:, :, 1])  # green outline
    assert np.any(np.all(overlay > 200, axis=2))  # white labels
    path = save_debug_visualization(tmp_path / "debug" / "corners.png", image, corners=corners)
    np.testing.assert_array_equal(load_image(path), overlay)


def test_point_coordinates_use_x_then_y():
    image = np.zeros((60, 90, 3), dtype=np.uint8)
    result = draw_points(image, [[70, 20]])
    np.testing.assert_array_equal(result[20, 70], [0, 0, 255])
    assert not image.any()


def test_polygon_closes_and_does_not_mutate_source():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    result = draw_polygon(image, [[10, 10], [60, 10], [60, 60], [10, 60]])
    assert result[35, 10, 1] > 0
    assert not image.any()


def test_preview_preserves_aspect_and_does_not_upscale():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    assert resize_for_preview(image, 80).shape == (40, 80, 3)
    unchanged = resize_for_preview(image, 300)
    assert unchanged.shape == image.shape
    assert not np.shares_memory(unchanged, image)
    assert resize_for_preview(image, 1).shape == (1, 1, 3)


def test_grayscale_overlay_is_colored_bgr():
    image = np.zeros((50, 50), dtype=np.uint8)
    result = draw_points(image, [[20, 20]])
    assert result.shape == (50, 50, 3)
    np.testing.assert_array_equal(result[20, 20], [0, 0, 255])


@pytest.mark.parametrize("points", [[], [[0, 1, 2]], [[float("nan"), 1]]])
def test_bad_points_fail_clearly(points):
    with pytest.raises(ValueError, match="[Pp]oint"):
        draw_points(np.zeros((50, 50, 3), dtype=np.uint8), points)


def test_invalid_preview_size():
    with pytest.raises(ValueError, match="positive integer"):
        resize_for_preview(np.zeros((20, 20, 3), dtype=np.uint8), 0)
