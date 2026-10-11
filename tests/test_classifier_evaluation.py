import numpy as np
import pytest
pytest.importorskip('torch')
from src.vision.classifier_evaluation import prediction_metrics


def test_confidence_one_is_included_and_confident_errors_are_counted():
    probabilities = np.zeros((3,13))
    probabilities[0,0] = 1
    probabilities[1,0] = .95; probabilities[1,1] = .05
    probabilities[2,1] = .6; probabilities[2,0] = .4
    result = prediction_metrics([0,1,1], probabilities)
    assert result['accuracy'] == pytest.approx(2/3)
    assert result['occupied_square_accuracy'] == .5
    assert result['high_confidence_errors'] == 1
    assert sum(b['count'] for b in result['calibration_bins']) == 3
    assert result['expected_calibration_error'] == pytest.approx((.475*2+.4)/3)
    assert result['per_class'][1]['recall'] == .5
    assert result['per_class'][1]['precision'] == 1


def test_invalid_probabilities_are_rejected():
    with pytest.raises(ValueError):
        prediction_metrics([0], np.zeros((1,13)))
