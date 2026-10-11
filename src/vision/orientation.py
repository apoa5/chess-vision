"""Resolve chess orientation from this board's printed coordinates, never pieces.

Supported layout: files A..H on both horizontal borders; ranks 8..1 on
both vertical borders. Bottom/left characters are upright; top/right are
rotated 180 degrees. Mirrored inputs and other layouts are not supported.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.vision.perspective import RectifiedBoard

TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "models/board_markings.npz"


@dataclass(frozen=True)
class OrientationSettings:
    min_similarity: float = .60
    min_matches: int = 8
    min_score_gap: float = .12

    def __post_init__(self):
        if (type(self.min_similarity) not in (int, float) or type(self.min_score_gap) not in (int, float)
                or not 0 < self.min_similarity <= 1 or not 0 < self.min_score_gap <= 1):
            raise ValueError("Orientation similarity and score gap must lie in (0, 1].")
        if type(self.min_matches) is not int or not 4 <= self.min_matches <= 32:
            raise ValueError("Orientation min_matches must be an integer from 4 to 32.")


@dataclass(frozen=True)
class OrientationResult:
    status: str
    rotation_quarters_ccw: int | None = None
    score: float | None = None
    reason: str | None = None
    candidates: list = field(default_factory=list)
    source: str = "border_markings"

    def metadata(self):
        return asdict(self)


def rotate_rectified(rectified, quarters_ccw):
    """Rotate pixels, coverage, and exclusive playing-area bounds together."""
    if type(quarters_ccw) is not int or quarters_ccw not in range(4):
        raise ValueError("Rotation must be 0, 1, 2, or 3 quarter turns counterclockwise.")
    x0, y0, x1, y1 = rectified.board_bounds
    height, width = rectified.image.shape[:2]
    for _ in range(quarters_ccw):
        x0, y0, x1, y1 = y0, width - x1, y1, width - x0
        height, width = width, height
    return RectifiedBoard(np.rot90(rectified.image, quarters_ccw).copy(),
                          np.rot90(rectified.valid_mask, quarters_ccw).copy(), (x0, y0, x1, y1))


def marking_patches(rectified):
    """Sample border slots near the expected centers; exclude the playing grid."""
    x0, y0, x1, y1 = rectified.board_bounds
    side = (x1 - x0) / 8
    height, width = rectified.image.shape[:2]
    for edge in ("bottom", "top", "left", "right"):
        for index in range(8):
            lo, hi = (index + .1) * side, (index + .9) * side
            near, far = .15 * side, .85 * side
            if edge == "bottom": bounds = (x0 + lo, y1 + near, x0 + hi, y1 + far)
            elif edge == "top": bounds = (x0 + lo, y0 - far, x0 + hi, y0 - near)
            elif edge == "left": bounds = (x0 - far, y0 + lo, x0 - near, y0 + hi)
            else: bounds = (x1 + near, y0 + lo, x1 + far, y0 + hi)
            left, top, right, bottom = map(round, bounds)
            label = "ABCDEFGH"[index] if edge in ("bottom", "top") else str(8 - index)
            if min(left, top) < 0 or right > width or bottom > height:
                continue
            patch = rectified.image[top:bottom, left:right]
            mask = rectified.valid_mask[top:bottom, left:right]
            if patch.size == 0 or (mask > 0).mean() < .95:
                continue
            if edge in ("top", "right"):
                patch = np.rot90(patch, 2).copy()
            yield edge, index, label, patch


def glyph_from_patch(patch):
    """Isolate a small light printed component, then normalize its silhouette."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) if patch.ndim == 3 else patch
    h, w = gray.shape
    kernel = max(3, round(min(h, w) * .55) | 1)
    contrast = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, np.ones((kernel, kernel), np.uint8))
    _, binary = cv2.threshold(contrast, max(25, float(contrast.max()) * .35), 255, cv2.THRESH_BINARY)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    candidates = []
    for index in range(1, count):
        x, y, cw, ch, area = stats[index]
        if not (.12*h <= ch <= .8*h and .03*w <= cw <= .8*w and area >= max(6, h*w*.008)):
            continue
        if x == 0 or y == 0 or x + cw == w or y + ch == h:
            continue
        distance = np.linalg.norm((centers[index] - [w/2, h/2]) / [w, h])
        candidates.append((distance, index))
    if not candidates:
        return None
    _, index = min(candidates)
    x, y, cw, ch, _ = stats[index]
    glyph = ((labels[y:y+ch, x:x+cw] == index) * 255).astype(np.uint8)
    # Preserve aspect ratio on a common canvas rather than stretching every letter.
    scale = min(28/cw, 40/ch)
    glyph = cv2.resize(glyph, (max(1, round(cw*scale)), max(1, round(ch*scale))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((48, 36), np.uint8)
    top, left = (48-glyph.shape[0])//2, (36-glyph.shape[1])//2
    canvas[top:top+glyph.shape[0], left:left+glyph.shape[1]] = glyph
    return canvas


def load_templates(path=TEMPLATE_PATH):
    with np.load(path, allow_pickle=False) as bank:
        templates = {label: bank[label].copy() for label in "ABCDEFGH12345678"}
    if any(v.shape != (48, 36) or v.dtype != np.uint8 for v in templates.values()):
        raise ValueError("Invalid board-marking template bank.")
    return templates


def resolve_orientation(rectified, settings=None, templates=None):
    """Score four rotations; return uncertain instead of guessing.

    Score is matched-slot agreement, not a calibrated probability. At least
    two edges and three distinct files/ranks must agree. No filenames, capture
    metadata, or FENs are read.
    """
    settings = settings or OrientationSettings()
    if templates is None:
        try:
            templates = load_templates()
        except (OSError, ValueError, KeyError) as error:
            return OrientationResult("uncertain", reason=f"Marking templates unavailable: {error}")
    candidates = []
    for turns in range(4):
        evidence = []
        for edge, index, expected, patch in marking_patches(rotate_rectified(rectified, turns)):
            glyph = glyph_from_patch(patch)
            if glyph is None:
                continue
            scores = {label: float(cv2.matchTemplate(glyph, template, cv2.TM_CCOEFF_NORMED)[0, 0])
                      for label, template in templates.items()}
            predicted = max(scores, key=scores.get)
            similarity = scores[predicted]
            if similarity < settings.min_similarity:
                continue
            evidence.append({"edge": edge, "index": index, "expected": expected,
                             "predicted": predicted, "similarity": round(similarity, 3),
                             "matches": predicted == expected})
        matches = [e for e in evidence if e["matches"]]
        score = len(matches) / 32
        candidates.append({"rotation_quarters_ccw": turns, "score": score,
                           "match_count": len(matches), "matched_edges": sorted({e['edge'] for e in matches}),
                           "evidence": evidence})
    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    best = ranked[0]
    gap = best["score"] - ranked[1]["score"]
    matched = {e["expected"] for e in best["evidence"] if e["matches"]}
    # Several distinct files AND ranks establish both axes. Symmetric glyphs
    # such as A/H/1/8 alone must not approve a mirrored or obscured photograph.
    files = matched & set("ABCDEFGH")
    ranks = matched & set("12345678")
    if (best["match_count"] < settings.min_matches or len(best["matched_edges"]) < 2
            or len(files) < 3 or len(ranks) < 3 or gap < settings.min_score_gap):
        return OrientationResult("uncertain", score=best["score"], reason="Insufficient or ambiguous border markings.", candidates=candidates)
    return OrientationResult("resolved", best["rotation_quarters_ccw"], best["score"], candidates=candidates)
