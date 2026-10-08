import json

import numpy as np
import pytest

from src.vision.board_detection import BoardDetection, detect_board_corners
from src.vision.perspective import order_corners, warp_board
from src.vision.square_extraction import SquareCrop, extract_squares


def test_square_metadata_preserves_image_indices_and_exclusive_bounds():
    board = np.zeros((800, 800, 3), dtype=np.uint8)
    crop = SquareCrop(row=7, col=7, bounds=(700, 700, 800, 800), image=board[700:800, 700:800])
    metadata = json.loads(json.dumps(crop.metadata()))
    assert metadata == {"row": 7, "col": 7, "bounds": [700, 700, 800, 800]}
    assert crop.image.shape == (100, 100, 3)


def test_detection_failure_has_reason_and_no_corners():
    failure = BoardDetection(corners=None, reason="No reliable board candidate found.")
    assert not failure.success
    assert failure.reason


def test_unimplemented_stages_fail_explicitly():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    corners = order_corners([[0, 0], [79, 0], [79, 79], [0, 79]])
    with pytest.raises(NotImplementedError, match="Day 4"):
        detect_board_corners(image)
    with pytest.raises(NotImplementedError, match="Day 6"):
        warp_board(image, corners)
    with pytest.raises(NotImplementedError, match="Day 7"):
        extract_squares(image)


def test_warp_requires_ordered_corners():
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="ordered TL, TR, BR, BL"):
        warp_board(image, [[79, 79], [0, 79], [0, 0], [79, 0]])


@pytest.mark.parametrize("shape", [(80, 81, 3), (81, 81, 3)])
def test_extraction_rejects_unnormalized_image(shape):
    with pytest.raises(ValueError, match="square.*divisible by 8"):
        extract_squares(np.zeros(shape, dtype=np.uint8))
