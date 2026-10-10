import cv2
import numpy as np
import pytest

from src.vision.board_detection import DetectionSettings, detect_board_corners
from src.vision.perspective import order_corners


def checker_image():
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    # Larger plain background rectangle must not win by area alone.
    image[15:465, 15:625] = 245
    image[95:401, 145:451] = (45, 80, 110)
    y, x = np.indices((256, 256))
    board = image[120:376, 170:426]
    board[(x // 32 + y // 32) % 2 == 0] = (210, 225, 235)
    board[(x // 32 + y // 32) % 2 == 1] = (45, 70, 95)
    return image


@pytest.mark.parametrize("max_size", [640, 400])
def test_detects_checker_in_original_coordinates_and_preserves_input(max_size):
    image = checker_image()
    original = image.copy()
    result = detect_board_corners(image, DetectionSettings(max_image_size=max_size))
    assert result.success, result.reason
    assert result.corners.shape == (4, 2)
    assert result.corners.dtype == np.float32
    np.testing.assert_array_equal(result.corners, order_corners(result.corners))
    expected = np.array([[170, 120], [425, 120], [425, 375], [170, 375]])
    assert np.max(np.linalg.norm(result.corners - expected, axis=1)) < 25
    np.testing.assert_array_equal(image, original)
    assert result.diagnostics["candidate_count"] > 0


def test_perspective_board_has_ordered_corners():
    image = checker_image()
    source = np.array([[0, 0], [639, 0], [639, 479], [0, 479]], dtype=np.float32)
    dest = np.array([[80, 20], [560, 40], [620, 450], [20, 460]], dtype=np.float32)
    transformed = cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, dest), (640, 480))
    result = detect_board_corners(transformed)
    assert result.success, result.reason
    np.testing.assert_array_equal(result.corners, order_corners(result.corners))


@pytest.mark.parametrize("kind", ["blank", "rectangle", "noise"])
def test_nonboard_images_fail_without_fabricated_corners(kind):
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    if kind == "rectangle":
        cv2.rectangle(image, (100, 80), (540, 400), (20, 20, 20), -1)
    if kind == "noise":
        image = np.random.default_rng(42).integers(0, 256, image.shape, dtype=np.uint8)
    result = detect_board_corners(image)
    assert not result.success
    assert result.corners is None
    assert result.reason


@pytest.mark.parametrize("options", [{"blur_kernel": 4}, {"canny_low": 200, "canny_high": 100},
                                     {"min_area_fraction": 0.95}, {"border_insets": [0.5]}])
def test_invalid_detection_settings(options):
    with pytest.raises(ValueError):
        DetectionSettings(**options)


def test_plain_wood_frame_without_grid_is_rejected():
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    image[100:400, 150:450] = (45, 80, 110)
    result = detect_board_corners(image)
    assert not result.success
    assert result.corners is None


def test_aligned_grid_scores_better_than_one_file_shift():
    from src.vision.board_detection import _checker_evidence
    gray = cv2.cvtColor(checker_image(), cv2.COLOR_BGR2GRAY)
    corners = np.array([[170, 120], [425, 120], [425, 375], [170, 375]], dtype=np.float32)
    aligned, _ = _checker_evidence(gray, corners)
    shifted, _ = _checker_evidence(gray, corners + np.array([32, 0], dtype=np.float32))
    assert aligned > shifted
