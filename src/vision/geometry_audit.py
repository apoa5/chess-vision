"""Read-only B.1 diagnostics; no detector, crop or inference behavior is changed."""
import cv2
import numpy as np

from src.vision.perspective import order_corners


def context_transform(corners, bounds):
    """Original image -> padded context; exclusive bounds end at x1-1/y1-1."""
    x0, y0, x1, y1 = bounds
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Invalid context bounds')
    return cv2.getPerspectiveTransform(order_corners(corners),
                                      np.float32([[x0, y0], [x1-1, y0], [x1-1, y1-1], [x0, y1-1]]))


def project(points, transform):
    points = np.asarray(points, np.float32)
    return cv2.perspectiveTransform(points.reshape(-1, 1, 2), np.asarray(transform, np.float64)).reshape(points.shape)


def quadrilateral_metrics(candidate, reference):
    """Unlabeled geometry comparison; semantic square assignment is checked separately."""
    a, b = order_corners(candidate), order_corners(reference)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a)+cv2.contourArea(b)-intersection
    errors = min((np.linalg.norm(a-np.roll(b, k, axis=0), axis=1) for k in range(4)),
                 key=lambda e: float(np.mean(e**2)))
    square_scale = float(np.linalg.norm(b-np.roll(b, -1, axis=0), axis=1).mean()/8)
    return dict(iou=float(intersection/union), rms_corner_error_px=float(np.sqrt(np.mean(errors**2))),
                max_corner_error_px=float(errors.max()), rms_corner_error_squares=float(np.sqrt(np.mean(errors**2))/square_scale))


def independently_rasterized_mask(shape, polygon, input_size=224):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(polygon).astype(np.int32), 1)
    return cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_NEAREST)


def raw_grid_overlay(image, polygons, square_ids, *, targets=()):
    """Polygons must already be in original-photo pixels, never canvas pixels."""
    out = image.copy()
    for polygon, square in zip(polygons, square_ids):
        points = np.rint(polygon).astype(np.int32)
        color = (0, 100, 255) if square in targets else (80, 240, 80)
        cv2.polylines(out, [points], True, color, 3 if square in targets else 1, cv2.LINE_AA)
        centre = tuple(np.rint(np.asarray(polygon).mean(axis=0)).astype(int))
        cv2.putText(out, square, centre, cv2.FONT_HERSHEY_SIMPLEX, .40, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, square, centre, cv2.FONT_HERSHEY_SIMPLEX, .40, color, 1, cv2.LINE_AA)
    return out
