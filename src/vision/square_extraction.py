"""Day 7 interface and metadata contract for normalized-board square crops.

Indices are row 0..7 (top to bottom) and column 0..7 (left to right).
Return order is row-major: r0_c0, r0_c1, ..., r7_c7. These are image indices;
they are not chess coordinates until board orientation has been resolved.
"""

from dataclasses import dataclass

import numpy as np

from src.utils.image_io import validate_image


@dataclass(frozen=True)
class SquareCrop:
    """A crop plus its location on the normalized board.

    Bounds are (x0, y0, x1, y1), with exclusive right/bottom endpoints:
    image = board_image[y0:y1, x0:x1]. For an 800×800 board, r0_c0 has
    bounds (0, 0, 100, 100); r7_c7 has (700, 700, 800, 800).
    """

    row: int
    col: int
    bounds: tuple[int, int, int, int]
    image: np.ndarray

    def metadata(self) -> dict:
        """Return JSON-compatible location metadata without the image array."""
        return {"row": self.row, "col": self.col, "bounds": list(self.bounds)}


def extract_squares(board_image: np.ndarray) -> list[SquareCrop]:
    """Return 64 independent crops, covering every pixel once in row-major order."""
    validate_image(board_image)
    height, width = board_image.shape[:2]
    if height != width or height % 8:
        raise ValueError("Normalized board must be square with a side length divisible by 8.")
    side = height // 8
    return [SquareCrop(row, col, (col * side, row * side, (col + 1) * side, (row + 1) * side),
                       board_image[row * side:(row + 1) * side,
                                   col * side:(col + 1) * side].copy())
            for row in range(8) for col in range(8)]


def extract_context_squares(rectified, padding_pixels=100):
    """Extract overlapping windows centered on the 64 original grid squares.

    Bounds are in the context canvas, not the 800×800 playing-area image.
    Each crop includes a coverage mask: black padding is not observed content.
    """
    if type(padding_pixels) is not int or padding_pixels < 0:
        raise ValueError("Crop padding must be a nonnegative integer.")
    x0, y0, x1, y1 = rectified.board_bounds
    core = rectified.board_image
    squares = extract_squares(core)
    height, width = rectified.image.shape[:2]
    if min(x0, y0, width - x1, height - y1) < padding_pixels:
        raise ValueError("Crop padding cannot exceed the available context margin.")
    result = []
    for square in squares:
        sx0, sy0, sx1, sy1 = square.bounds
        bounds = (x0 + sx0 - padding_pixels, y0 + sy0 - padding_pixels,
                  x0 + sx1 + padding_pixels, y0 + sy1 + padding_pixels)
        left, top, right, bottom = bounds
        result.append((SquareCrop(square.row, square.col, bounds,
                                   rectified.image[top:bottom, left:right].copy()),
                       rectified.valid_mask[top:bottom, left:right].copy()))
    return result
