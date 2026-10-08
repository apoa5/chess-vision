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
    """Day 7 interface: return exactly 64 crops with metadata in row-major order."""
    validate_image(board_image)
    height, width = board_image.shape[:2]
    if height != width or height % 8:
        raise ValueError("Normalized board must be square with a side length divisible by 8.")
    raise NotImplementedError("Square crop extraction is scheduled for Day 7.")
