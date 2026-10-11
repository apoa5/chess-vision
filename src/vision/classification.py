"""Day 8 input/prediction contracts; no model loading, inference, or training."""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from src.chess.labels import CLASSES, get_class, LABEL_VERSION
from src.utils.image_io import validate_image


def validate_probabilities(values):
    try:
        values = tuple(values)
        if any(not isinstance(value, Real) or isinstance(value, (bool, np.bool_)) for value in values):
            raise ValueError('Probabilities must be numeric, not boolean.')
        probabilities = tuple(float(value) for value in values)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('Probabilities must contain 13 finite numbers.') from error
    if len(probabilities) != len(CLASSES) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError('Probabilities must contain 13 finite numbers in [0, 1].')
    if not math.isclose(sum(probabilities), 1, abs_tol=1e-6):
        raise ValueError('Class probabilities must sum to 1.')
    return probabilities


@dataclass(frozen=True)
class SquareInput:
    """BGR uint8 context crop plus exclusive target-square bounds in that crop.

    Contains no row, column, FEN, or game state. Neighboring squares are context;
    the label describes only target_bounds. Tensor preprocessing belongs later.
    """
    image: np.ndarray
    target_bounds: tuple[int, int, int, int]

    def __post_init__(self):
        validate_image(self.image)
        if self.image.ndim != 3 or self.image.shape[2] != 3:
            raise ValueError('Classifier crop must be a uint8 BGR image.')
        try:
            bounds = tuple(self.target_bounds)
        except TypeError as error:
            raise ValueError('Target bounds must be four integer coordinates.') from error
        if len(bounds) != 4 or any(type(v) is not int for v in bounds):
            raise ValueError('Target bounds must be four integer coordinates.')
        x0, y0, x1, y1 = bounds
        height, width = self.image.shape[:2]
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) or x1-x0 != y1-y0:
            raise ValueError('Target bounds must describe a square within the crop.')
        image = self.image.copy()
        image.flags.writeable = False
        object.__setattr__(self, 'image', image)
        object.__setattr__(self, 'target_bounds', bounds)


@dataclass(frozen=True)
class SquarePrediction:
    row: int
    col: int
    class_id: int
    confidence: float
    probabilities: tuple | None = None

    def __post_init__(self):
        if any(type(v) is not int or v not in range(8) for v in (self.row, self.col)):
            raise ValueError('Prediction row and column must be integers from 0 to 7.')
        get_class(self.class_id)
        if type(self.confidence) not in (float, int) or not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError('Confidence must be a finite number in [0, 1].')
        if self.probabilities is not None:
            probabilities = validate_probabilities(self.probabilities)
            winner = max(range(len(CLASSES)), key=probabilities.__getitem__)
            if self.class_id != winner or not math.isclose(self.confidence, probabilities[winner], abs_tol=1e-6):
                raise ValueError('Class and confidence must match the probability argmax (ties use lowest ID).')
            object.__setattr__(self, 'probabilities', probabilities)

    @property
    def class_name(self):
        return get_class(self.class_id).name

    @property
    def symbol(self):
        return get_class(self.class_id).symbol

    @classmethod
    def from_probabilities(cls, row, col, probabilities):
        probabilities = validate_probabilities(probabilities)
        winner = max(range(len(CLASSES)), key=probabilities.__getitem__)
        return cls(row, col, winner, probabilities[winner], probabilities)

    def metadata(self):
        result = {'row': self.row, 'col': self.col, 'class_id': self.class_id,
                  'class_name': self.class_name, 'symbol': self.symbol, 'confidence': float(self.confidence),
                  'label_version': LABEL_VERSION}
        if self.probabilities is not None:
            result['probabilities'] = list(self.probabilities)
        return result
