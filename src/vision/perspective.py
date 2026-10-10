"""Board geometry in image coordinates (x rightwards, y downwards).

The shared corner representation is a float32 array of shape (4, 2), ordered
top-left, top-right, bottom-right, bottom-left (TL, TR, BR, BL).
This describes image geometry, not the orientation of White's pieces.
"""

import cv2
import numpy as np
from dataclasses import dataclass

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


def warp_board(image: np.ndarray, corners: np.ndarray, output_size: int = 800,
               *, margin_pixels: int = 0) -> np.ndarray:
    """Map the playing area's ordered corners directly to a square image.

    Sample the original photograph once with cubic interpolation. This corrects
    the board plane, not the perspective of pieces standing above that plane.
    """
    validate_image(image)
    if type(output_size) is not int or output_size <= 0 or output_size % 8:
        raise ValueError("Output size must be a positive integer divisible by 8.")
    if type(margin_pixels) is not int or margin_pixels < 0:
        raise ValueError("Context margin must be a nonnegative integer.")
    ordered = order_corners(corners)
    if not np.array_equal(np.asarray(corners, dtype=np.float32), ordered):
        raise ValueError("Corners must be ordered TL, TR, BR, BL; use order_corners first.")
    height, width = image.shape[:2]
    if (ordered < 0).any() or (ordered[:, 0] > width - 1).any() or (ordered[:, 1] > height - 1).any():
        raise ValueError("Corners must lie within the source image bounds.")
    end = output_size - 1
    destination = np.asarray([[0, 0], [end, 0], [end, end], [0, end]], dtype=np.float32)
    destination += margin_pixels
    transform = cv2.getPerspectiveTransform(ordered, destination)
    if not np.isfinite(transform).all() or np.linalg.matrix_rank(transform) < 3:
        raise ValueError("Corners do not define a valid perspective transform.")
    canvas_size = output_size + 2 * margin_pixels
    return cv2.warpPerspective(image, transform, (canvas_size, canvas_size),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT if margin_pixels else cv2.BORDER_REPLICATE)


@dataclass(frozen=True)
class RectifiedBoard:
    """Context canvas with explicit playing-area bounds and source coverage."""

    image: np.ndarray
    valid_mask: np.ndarray
    board_bounds: tuple[int, int, int, int]

    @property
    def board_image(self):
        x0, y0, x1, y1 = self.board_bounds
        return self.image[y0:y1, x0:x1].copy()

    def metadata(self):
        return {"board_bounds": list(self.board_bounds), "canvas_shape": list(self.image.shape),
                "bounds_convention": "exclusive right/bottom", "valid_mask_values": {"source": 255, "outside_photo": 0}}


def warp_board_with_context(image, corners, output_size=800, margin_pixels=200):
    """Extend the same board-plane transform, sampling real pixels outside the grid.

    Black outside-source pixels are explicitly identified by valid_mask. Never
    divide this entire canvas into eight: only board_bounds describe the grid.
    """
    canvas = warp_board(image, corners, output_size, margin_pixels=margin_pixels)
    end = output_size - 1
    destination = np.float32([[0, 0], [end, 0], [end, end], [0, end]]) + margin_pixels
    transform = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), destination)
    mask = cv2.warpPerspective(np.full(image.shape[:2], 255, np.uint8), transform,
                               (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_CONSTANT)
    return RectifiedBoard(canvas, mask, (margin_pixels, margin_pixels,
                                       margin_pixels + output_size, margin_pixels + output_size))
