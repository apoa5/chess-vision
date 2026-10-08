"""Non-destructive BGR debug overlays; draw first, then resize for preview.

All coordinates refer to the original image. Colors use OpenCV BGR order.
Grayscale/BGRA inputs are converted to BGR for colored annotations.
"""

from pathlib import Path

import cv2
import numpy as np

from src.utils.image_io import save_image, validate_image
from src.vision.perspective import CORNER_LABELS, order_corners


def _canvas(image: np.ndarray) -> np.ndarray:
    validate_image(image)
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _points(points, minimum: int = 1) -> np.ndarray:
    try:
        values = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Points must be numeric (x, y) pairs.") from error
    if values.ndim != 2 or values.shape[1] != 2 or len(values) < minimum:
        raise ValueError(f"Points must have shape (N, 2), with at least {minimum} points.")
    if not np.isfinite(values).all() or np.abs(values).max() > np.iinfo(np.int32).max - 1:
        raise ValueError("Point coordinates must be finite and fit in int32.")
    return np.rint(values).astype(np.int32)


def draw_points(image: np.ndarray, points, *, color=(0, 0, 255), radius: int = 5) -> np.ndarray:
    """Return a copy with filled point markers."""
    if type(radius) is not int or radius <= 0:
        raise ValueError("Point radius must be a positive integer.")
    canvas = _canvas(image)
    for point in _points(points):
        cv2.circle(canvas, tuple(point), radius, color, -1, cv2.LINE_AA)
    return canvas


def draw_polygon(image: np.ndarray, points, *, color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Return a copy with a closed polygon; vertices must already be cyclic."""
    if type(thickness) is not int or thickness <= 0:
        raise ValueError("Polygon thickness must be a positive integer.")
    canvas = _canvas(image)
    vertices = _points(points, minimum=3)
    cv2.polylines(canvas, [vertices], True, color, thickness, cv2.LINE_AA)
    return canvas


def draw_board_corners(image: np.ndarray, corners) -> np.ndarray:
    """Order corners and draw the board outline, dots, and TL/TR/BR/BL labels."""
    ordered = order_corners(corners)
    canvas = draw_points(draw_polygon(image, ordered), ordered)
    height, width = canvas.shape[:2]
    for label, (x, y) in zip(CORNER_LABELS, _points(ordered)):
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        anchor = (max(0, min(int(x) + 8, width - text_width - 1)),
                  max(text_height, min(int(y) - 8, height - baseline - 1)))
        cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, label, anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def resize_for_preview(image: np.ndarray, max_size: int = 1200) -> np.ndarray:
    """Fit within max_size×max_size, preserving aspect ratio without upscaling."""
    validate_image(image)
    if type(max_size) is not int or max_size <= 0:
        raise ValueError("Preview max_size must be a positive integer.")
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale == 1:
        return image.copy()
    dimensions = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)


def save_debug_visualization(path: str | Path, image: np.ndarray, *, corners=None,
                             max_size: int | None = None) -> Path:
    """Save an optional corner overlay/preview without changing the source."""
    debug = draw_board_corners(image, corners) if corners is not None else _canvas(image)
    if max_size is not None:
        debug = resize_for_preview(debug, max_size)
    return save_image(path, debug)
