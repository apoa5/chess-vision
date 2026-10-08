from itertools import permutations

import numpy as np
import pytest

from src.vision.perspective import order_corners


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
