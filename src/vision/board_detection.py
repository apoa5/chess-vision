"""Classical contour/line detector for the 8×8 playing area, excluding its border."""

from dataclasses import dataclass, field
from itertools import combinations

import cv2

import numpy as np

from src.utils.image_io import validate_image
from src.vision.perspective import order_corners


@dataclass(frozen=True)
class DetectionSettings:
    max_image_size: int = 1280
    blur_kernel: int = 5
    canny_low: int = 40
    canny_high: int = 120
    normalize_contrast: bool = False
    min_area_fraction: float = 0.05
    max_area_fraction: float = 0.90
    polygon_epsilon: float = 0.02
    min_checker_correlation: float = 0.40
    min_checker_contrast: float = 8.0
    min_score: float = 0.55
    # Candidate insets account for the wooden frame around the playing area.
    border_insets: tuple[float, ...] = (0.0, 0.04, 0.07, 0.09, 0.11)
    use_wood_color_mask: bool = True
    wood_hue_min: int = 0
    wood_hue_max: int = 40
    wood_min_saturation: int = 60
    refine_candidates: int = 3
    use_line_candidates: bool = True

    def __post_init__(self):
        for option in (self.normalize_contrast, self.use_wood_color_mask, self.use_line_candidates):
            if type(option) is not bool:
                raise ValueError("Detection feature switches must be boolean.")
        if type(self.max_image_size) is not int or self.max_image_size < 64:
            raise ValueError("max_image_size must be an integer >= 64.")
        if type(self.blur_kernel) is not int or self.blur_kernel < 1 or self.blur_kernel % 2 == 0:
            raise ValueError("blur_kernel must be a positive odd integer.")
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255.")
        if not 0 < self.min_area_fraction < self.max_area_fraction < 1:
            raise ValueError("Area fractions must satisfy 0 < min < max < 1.")
        if not 0 < self.polygon_epsilon < 0.1:
            raise ValueError("polygon_epsilon must be between 0 and 0.1.")
        if not 0 <= self.min_checker_correlation <= 1 or not 0 <= self.min_score <= 1:
            raise ValueError("Correlation and score thresholds must be between 0 and 1.")
        if not 0 <= self.min_checker_contrast <= 255:
            raise ValueError("min_checker_contrast must be between 0 and 255.")
        if not self.border_insets or not all(0 <= inset < 0.25 for inset in self.border_insets):
            raise ValueError("border_insets must contain fractions between 0 and 0.25.")
        if not 0 <= self.wood_hue_min <= self.wood_hue_max <= 179 or not 0 <= self.wood_min_saturation <= 255:
            raise ValueError("Invalid wood-color HSV thresholds.")
        if type(self.refine_candidates) is not int or not 0 <= self.refine_candidates <= 8:
            raise ValueError("refine_candidates must be an integer between 0 and 8.")


@dataclass(frozen=True)
class BoardDetection:
    """Detector result; failed detections have no corners and a reason.

    Successful corners are float32 (4, 2) in TL/TR/BR/BL order. An optional
    candidate score is a diagnostic, not a calibrated probability.
    """

    corners: np.ndarray | None
    reason: str | None = None
    score: float | None = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.corners is not None


def preprocess_image(image: np.ndarray, settings: DetectionSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return working grayscale, Canny edges, and (x, y) scale to original pixels."""
    validate_image(image)
    height, width = image.shape[:2]
    scale = min(1.0, settings.max_image_size / max(height, width))
    working = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA) if scale < 1 else image
    if working.ndim == 2:
        gray = working.copy()
    else:
        gray = cv2.cvtColor(working, cv2.COLOR_BGRA2GRAY if working.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
    if settings.normalize_contrast:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(gray, (settings.blur_kernel, settings.blur_kernel), 0)
    edges = cv2.Canny(blurred, settings.canny_low, settings.canny_high)
    return gray, edges, np.array([width / gray.shape[1], height / gray.shape[0]], dtype=np.float32)


def _checker_evidence(gray: np.ndarray, corners: np.ndarray) -> tuple[float, float]:
    """Measure an 8×8 alternating pattern in a small candidate-scoring sample.

    This temporary 160×160 warp is only for candidate scoring, not the Day 6
    normalized-board pipeline. Cell-center medians reduce grid-edge and piece
    influence; subtracting row/column means reduces smooth lighting gradients.
    """
    destination = np.array([[0, 0], [159, 0], [159, 159], [0, 159]], dtype=np.float32)
    sample = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(corners, destination), (160, 160))
    tiles = sample.reshape(8, 20, 8, 20).transpose(0, 2, 1, 3)
    cells = np.median(tiles[:, :, 4:16, 4:16], axis=(2, 3)).astype(np.float32)
    residual = cells - cells.mean(axis=0, keepdims=True) - cells.mean(axis=1, keepdims=True) + cells.mean()
    checker = (np.indices((8, 8)).sum(axis=0) % 2) * 2 - 1
    contrast = abs(float(np.mean(residual * checker)))
    correlation = min(1.0, contrast / max(float(np.std(residual)), 1e-6))
    central = residual[2:6, 2:6]
    central_contrast = abs(float(np.mean(central * checker[2:6, 2:6])))
    central_correlation = min(1.0, central_contrast / max(float(np.std(central)), 1e-6))
    # Dense opening pieces mostly obscure the outer two ranks. Central cells
    # supply supporting evidence without allowing it to replace all-board evidence.
    correlation = max(correlation, 0.75 * central_correlation + 0.25 * correlation)
    contrast = max(contrast, 0.75 * central_contrast + 0.25 * contrast)
    # Pixel agreement distinguishes aligned square boundaries from candidates
    # whose cell centers look alternating but whose edges cut through squares.
    pixels = sample.astype(np.float32)
    highpass = pixels - cv2.GaussianBlur(pixels, (0, 0), 20)
    yy, xx = np.indices((160, 160))
    pixel_checker = ((xx // 20 + yy // 20) % 2) * 2 - 1
    pixel_contrast = abs(float(np.mean(highpass * pixel_checker)))
    pixel_correlation = pixel_contrast / max(float(np.std(highpass)), 1e-6)
    center_pixels = highpass[40:120, 40:120]
    center_contrast = abs(float(np.mean(center_pixels * pixel_checker[40:120, 40:120])))
    center_correlation = center_contrast / max(float(np.std(center_pixels)), 1e-6)
    alignment = max(pixel_correlation, .75 * center_correlation + .25 * pixel_correlation)
    unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    extension = np.array([[-.0625, -.0625], [1.0625, -.0625],
                          [1.0625, 1.0625], [-.0625, 1.0625]], dtype=np.float32)
    extended_corners = cv2.perspectiveTransform(extension[None], cv2.getPerspectiveTransform(unit, corners))[0]
    extended = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(
        extended_corners, np.array([[0, 0], [179, 0], [179, 179], [0, 179]], dtype=np.float32)), (180, 180))
    outside_patches = []
    for side in range(4):
        for cell in range(8):
            lo, hi = 10 + cell*20 + 4, 10 + cell*20 + 16
            patch = (extended[lo:hi, 2:8] if side == 0 else extended[lo:hi, 172:178] if side == 1
                     else extended[2:8, lo:hi] if side == 2 else extended[172:178, lo:hi])
            outside_patches.append(patch if side < 2 else patch.T)
    strip_values = np.median(np.stack(outside_patches), axis=(1, 2)).reshape(4, 8)
    outside = np.abs(np.mean(strip_values[:, 2:6] * np.array([-1, 1, -1, 1]), axis=1))
    # A strip still alternating outside a proposed boundary indicates that
    # the candidate clips the playing area rather than enclosing all 8×8 cells.
    continuation = min(1.0, float(max(outside)) / max(contrast, 8))
    return (.25 * correlation + .75 * alignment) * (1 - .5*continuation), contrast


def _refine_candidate(gray, record, to_original, settings):
    """Bounded coordinate search aligns a contour hypothesis to the visible grid."""
    corners = np.asarray(record["corners"], dtype=np.float32) / to_original
    original = corners.copy()
    best, contrast = _checker_evidence(gray, corners)
    base_step = float(np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1).mean()) / 32
    for step in (base_step, base_step / 2, base_step / 4):
        for _ in range(2):
            changed = False
            for vertex in range(4):
                for axis in range(2):
                    optimum = corners
                    for direction in (-1, 1):
                        proposal = corners.copy()
                        proposal[vertex, axis] += direction * step
                        if not cv2.isContourConvex(proposal) or (proposal < 2).any() or (proposal[:, 0] >= gray.shape[1]-2).any() or (proposal[:, 1] >= gray.shape[0]-2).any():
                            continue
                        correlation, proposed_contrast = _checker_evidence(gray, proposal)
                        if correlation > best + 0.001:
                            best, contrast, optimum = correlation, proposed_contrast, proposal
                    if optimum is not corners:
                        corners = optimum
                        changed = True
            if not changed:
                break
    fraction = cv2.contourArea(corners) / gray.size
    score = .70 * best + .20 * min(contrast / 40, 1.0) + .10 * min(fraction / .5, 1.0)
    lengths = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
    rectangle = cv2.minAreaRect(corners)[1]
    aspect = max(lengths[[0, 2]].mean(), lengths[[1, 3]].mean()) / min(lengths[[0, 2]].mean(), lengths[[1, 3]].mean())
    rectangularity = cv2.contourArea(corners) / max(rectangle[0]*rectangle[1], 1e-6)
    valid_geometry = settings.min_area_fraction <= fraction <= settings.max_area_fraction and aspect <= 3 and rectangularity >= .55 and lengths.max()/lengths.min() <= 4.5
    return {**record, "corners": (corners * to_original).tolist(), "score": score,
            "area_fraction": fraction, "aspect_ratio": float(aspect), "rectangularity": rectangularity,
            "checker_correlation": best, "checker_contrast": contrast,
            "refinement_max_shift": float(np.linalg.norm(corners-original, axis=1).max()),
            "accepted": bool(valid_geometry and best >= settings.min_checker_correlation and contrast >= settings.min_checker_contrast and score >= settings.min_score)}


def _line_boundaries(edges, settings):
    """Intersect long Hough edges when piece occlusion prevents closed contours."""
    height, width = edges.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=min(height, width) * .18, maxLineGap=25)
    if lines is None:
        return []
    groups = [[], []]
    for segment in sorted(lines.reshape(-1, 4), key=lambda s: np.hypot(s[2]-s[0], s[3]-s[1]), reverse=True):
        x1, y1, x2, y2 = map(float, segment)
        a, b, c = y1-y2, x2-x1, x1*y2-x2*y1
        length = np.hypot(a, b)
        line = np.array([a, b, c]) / length
        family = 0 if abs(x2-x1) >= abs(y2-y1) else 1
        # Use the same normal sign within each family. Nearly horizontal lines
        # must not flip sign merely because their slope crosses zero.
        if line[1 if family == 0 else 0] < 0:
            line = -line
        group = groups[family]
        if len(group) < 8 and not any(np.linalg.norm(line[:2]-old[:2]) < .08 and abs(line[2]-old[2]) < 15 for old in group):
            group.append(line)
    result = []
    for horizontal in combinations(groups[0], 2):
        if abs(horizontal[0][2]-horizontal[1][2]) < min(height, width) * .18:
            continue
        for vertical in combinations(groups[1], 2):
            if abs(vertical[0][2]-vertical[1][2]) < min(height, width) * .18:
                continue
            vertices = []
            for first, second in ((horizontal[0], vertical[0]), (horizontal[0], vertical[1]),
                                  (horizontal[1], vertical[1]), (horizontal[1], vertical[0])):
                point = np.cross(first, second)
                if abs(point[2]) < 1e-6:
                    break
                vertices.append(point[:2] / point[2])
            if len(vertices) != 4:
                continue
            points = np.asarray(vertices, dtype=np.float32)
            if (points < 2).any() or (points[:, 0] >= width-2).any() or (points[:, 1] >= height-2).any():
                continue
            try:
                ordered = order_corners(points)
            except ValueError:
                continue
            fraction = cv2.contourArea(ordered)/(width*height)
            if settings.min_area_fraction <= fraction <= settings.max_area_fraction:
                result.append(ordered)
    return result


def detect_board_corners(image: np.ndarray, settings: DetectionSettings | None = None) -> BoardDetection:
    """Select a convex quadrilateral with plausible geometry and checker evidence.

    Detection must report failure explicitly if no reliable board is found;
    image boundaries must not be returned as fabricated board corners.
    """
    settings = settings or DetectionSettings()
    gray, edges, to_original = preprocess_image(image, settings)
    height, width = gray.shape
    image_area = height * width
    masks = [("edges", edges), ("closed_edges", cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)))]
    if settings.use_wood_color_mask and image.ndim == 3:
        # On this wooden set, the warm frame remains continuous where pieces
        # and shadows break the inner-grid edge contour. Color proposes geometry;
        # checker evidence still decides whether the region is a board.
        working = cv2.resize(image[:, :, :3], (width, height), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
        for saturation, max_value in ((settings.wood_min_saturation, 255),
                                       (min(255, settings.wood_min_saturation + 40), 255),
                                       (settings.wood_min_saturation, 180)):
            warm = cv2.inRange(hsv, (settings.wood_hue_min, saturation, 30),
                               (settings.wood_hue_max, 255, max_value))
            warm = cv2.morphologyEx(warm, cv2.MORPH_OPEN, np.ones((7, 7), dtype=np.uint8))
            warm = cv2.morphologyEx(warm, cv2.MORPH_CLOSE, np.ones((11, 11), dtype=np.uint8))
            masks.append(("wood_color", warm))
    candidates = []
    seen = []
    contour_count = 0
    line_quads = []
    if settings.use_line_candidates:
        # Reuse the same geometry and evidence filters for line intersections.
        # This is a candidate source, never a fallback that bypasses scoring.
        line_quads = _line_boundaries(edges, settings)
        masks.append(("line_geometry", line_quads))
    for source, mask in masks:
        if source == "line_geometry":
            contours = [points.reshape(-1, 1, 2) for points in mask]
        else:
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if source != "line_geometry":
            contour_count += len(contours)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True):
            area_fraction = cv2.contourArea(contour) / image_area
            if not settings.min_area_fraction <= area_fraction <= settings.max_area_fraction:
                continue
            boundary = cv2.convexHull(contour) if source == "wood_color" else contour
            epsilon = max(settings.polygon_epsilon, 0.04) if source == "wood_color" else settings.polygon_epsilon
            polygon = boundary if source == "line_geometry" else cv2.approxPolyDP(boundary, epsilon * cv2.arcLength(boundary, True), True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            corners = order_corners(polygon.reshape(4, 2))
            if any(np.max(np.linalg.norm(corners - previous, axis=1)) < 5 for previous in seen):
                continue
            seen.append(corners)
            if (corners[:, 0] <= 1).any() or (corners[:, 1] <= 1).any() or (corners[:, 0] >= width - 2).any() or (corners[:, 1] >= height - 2).any():
                continue
            lengths = np.linalg.norm(corners - np.roll(corners, -1, axis=0), axis=1)
            aspect = max(lengths[[0, 2]].mean(), lengths[[1, 3]].mean()) / min(lengths[[0, 2]].mean(), lengths[[1, 3]].mean())
            rectangle = cv2.minAreaRect(corners)[1]
            rectangularity = cv2.contourArea(corners) / max(rectangle[0] * rectangle[1], 1e-6)
            if aspect > 3 or lengths.max() / max(lengths.min(), 1e-6) > 4.5 or rectangularity < 0.55:
                continue
            unit = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            transform = cv2.getPerspectiveTransform(unit, corners)
            for inset in settings.border_insets:
                # Propose the inner playing area in the contour's board plane.
                # Only candidates with measured checker evidence can succeed.
                inner = np.array([[inset, inset], [1-inset, inset],
                                  [1-inset, 1-inset], [inset, 1-inset]], dtype=np.float32)
                candidate = cv2.perspectiveTransform(inner[None], transform)[0]
                correlation, contrast = _checker_evidence(gray, candidate)
                fraction = cv2.contourArea(candidate) / image_area
                score = 0.70 * correlation + 0.20 * min(contrast / 40, 1.0) + 0.10 * min(fraction / 0.5, 1.0)
                accepted = settings.min_area_fraction <= fraction <= settings.max_area_fraction and correlation >= settings.min_checker_correlation and contrast >= settings.min_checker_contrast and score >= settings.min_score
                candidates.append({"corners": (candidate * to_original).tolist(), "score": score,
                                   "area_fraction": fraction, "aspect_ratio": float(aspect), "border_inset": inset, "source": source,
                                   "rectangularity": rectangularity, "checker_correlation": correlation,
                                   "checker_contrast": contrast, "accepted": accepted})
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    # Refine only the strongest hypotheses; contour discovery and checker
    # evidence remain necessary even when the detector ultimately reports failure.
    for candidate in candidates[:settings.refine_candidates]:
        candidates.append(_refine_candidate(gray, candidate, to_original, settings))
    candidates.sort(key=lambda candidate: candidate["score"], reverse=True)
    diagnostics = {"contour_count": contour_count, "line_quadrilateral_count": len(line_quads),
                   "candidate_count": len(candidates), "accepted_candidate_count": sum(c["accepted"] for c in candidates),
                   "working_size": [width, height], "top_candidates": candidates[:10]}
    reliable = [candidate for candidate in candidates if candidate["accepted"]]
    if not reliable:
        reason = "No plausible board quadrilateral found." if not candidates else "No candidate passed the checkerboard evidence thresholds."
        return BoardDetection(corners=None, reason=reason, diagnostics=diagnostics)
    best = reliable[0]
    diagnostics["selected_source"] = best["source"]
    return BoardDetection(corners=np.asarray(best["corners"], dtype=np.float32),
                          score=best["score"], diagnostics=diagnostics)
