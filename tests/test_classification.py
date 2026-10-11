import json
import numpy as np
import pytest
from src.vision.classification import SquareInput, SquarePrediction


def test_prediction_uses_canonical_mapping_and_preserves_probabilities():
    probabilities = [0.0]*13
    probabilities[1], probabilities[0] = .94, .06
    prediction = SquarePrediction.from_probabilities(4, 3, probabilities)
    probabilities[1] = 0
    result = json.loads(json.dumps(prediction.metadata()))
    assert result['symbol'] == 'P' and result['class_name'] == 'white_pawn'
    assert result['confidence'] == .94 and result['probabilities'][1] == .94
    assert (result['row'], result['col']) == (4, 3)


@pytest.mark.parametrize('values', [[0.0]*12, [0.0]*13, [float('nan')]+[0]*12, [-1]+[2]+[0]*11, [True]+[0]*12])
def test_invalid_distributions(values):
    with pytest.raises(ValueError):
        SquarePrediction.from_probabilities(0, 0, values)


def test_ties_and_inconsistent_predictions():
    values = [.5, .5]+[0]*11
    assert SquarePrediction.from_probabilities(0, 0, values).class_id == 0
    with pytest.raises(ValueError, match='argmax'):
        SquarePrediction(0, 0, 1, .5, values)
    with pytest.raises(ValueError, match='Confidence'):
        SquarePrediction(0, 0, 0, float('nan'))
    with pytest.raises(ValueError, match='row and column'):
        SquarePrediction(8, 0, 0, 1)


def test_square_input_preserves_central_target_and_owns_pixels():
    image = np.full((500, 500, 3), 90, np.uint8)
    crop = SquareInput(image, (200, 200, 300, 300))
    image[...] = 0
    assert (crop.image == 90).all()
    assert not crop.image.flags.writeable
    assert crop.target_bounds == (200, 200, 300, 300)


@pytest.mark.parametrize('bounds', [(0, 0, 501, 500), (0, 0, 100, 50), (1, 2, 3), (-1, 0, 99, 100)])
def test_invalid_target_bounds(bounds):
    with pytest.raises(ValueError, match='bounds'):
        SquareInput(np.zeros((500, 500, 3), np.uint8), bounds)
