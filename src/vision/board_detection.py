"""Day 4 interface for locating the outer board boundary in a BGR image."""

from dataclasses import dataclass

import numpy as np

from src.utils.image_io import validate_image


@dataclass(frozen=True)
class BoardDetection:
    """Future detector result; failed detections have no corners and a reason.

    Successful corners are float32 (4, 2) in TL/TR/BR/BL order. An optional
    candidate score is a diagnostic, not a calibrated probability.
    """

    corners: np.ndarray | None
    reason: str | None = None
    score: float | None = None

    @property
    def success(self) -> bool:
        return self.corners is not None


def detect_board_corners(image: np.ndarray) -> BoardDetection:
    """Return a result with TL/TR/BR/BL corners or a failure reason when implemented.

    Detection must report failure explicitly if no reliable board is found;
    image boundaries must not be returned as fabricated board corners.
    """
    validate_image(image)
    raise NotImplementedError("Automatic board detection is scheduled for Day 4.")
