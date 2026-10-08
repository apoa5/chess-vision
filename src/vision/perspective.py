"""Board geometry in image coordinates (x rightwards, y downwards).

The shared corner representation is a float32 array of shape (4, 2), ordered
top-left, top-right, bottom-right, bottom-left (TL, TR, BR, BL).
This describes image geometry, not the orientation of White's pieces.
"""

import cv2
import numpy as np

from src.utils.image_io import validate_image

CORNER_LABELS = ("TL", "TR", "BR", "BL")


def order_corners(points) -> np.ndarray:
    """Order four distinct convex vertices clockwise, starting at image TL.

    TL is the vertex with smallest x+y; ties prefer smaller y, then smaller x.
    This gives a deterministic anchor even for a diamond-shaped board. A photo
    alone cannot determine chess orientation. Reject collinear/concave inputs
    rather than inventing a fourth corner. The input is never mutated.
    """
    try:
        corners = np.asarray(points, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Corners must contain four numeric (x, y) pairs.") from error
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Corners must have shape (4, 2) and contain finite coordinates.")
    hull = cv2.convexHull(corners, clockwise=False).reshape(-1, 2)
    if len(hull) != 4 or cv2.contourArea(hull) <= 0:
        raise ValueError("Corners must be four distinct vertices of a nondegenerate convex quadrilateral.")
    # OpenCV's counterclockwise Cartesian hull is clockwise on an image (y down).
    start = min(range(4), key=lambda i: (float(hull[i, 0]) + float(hull[i, 1]),
                                         float(hull[i, 1]), float(hull[i, 0])))
    return np.roll(hull, -start, axis=0).copy()


def warp_board(image: np.ndarray, corners: np.ndarray, output_size: int = 800) -> np.ndarray:
    """Day 6 interface: warp TL/TR/BR/BL corners to output_size×output_size.

    Callers should pass the configured normalized size. No resizing is used as a
    substitute for perspective correction in this Day 2 scaffold.
    """
    validate_image(image)
    if type(output_size) is not int or output_size <= 0 or output_size % 8:
        raise ValueError("Output size must be a positive integer divisible by 8.")
    ordered = order_corners(corners)
    if not np.array_equal(np.asarray(corners, dtype=np.float32), ordered):
        raise ValueError("Corners must be ordered TL, TR, BR, BL; use order_corners first.")
    raise NotImplementedError("Perspective warping is scheduled for Day 6.")
