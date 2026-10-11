"""Classification and confidence diagnostics for a fixed held-out prediction set."""
import numpy as np
from src.chess.labels import CLASSES
from src.vision.training import confusion_metrics


def prediction_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.shape != (len(labels), len(CLASSES)) or not len(labels):
        raise ValueError('Expected nonempty labels and N x 13 probabilities.')
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
            or (labels < 0).any() or (labels >= len(CLASSES)).any()):
        raise ValueError('Invalid probabilities or labels.')
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = predicted == labels
    matrix = np.bincount(labels * 13 + predicted, minlength=169).reshape(13, 13)
    result = confusion_metrics(matrix)
    per_class = []
    for c in CLASSES:
        support = int(matrix[c.class_id].sum())
        predicted_count = int(matrix[:, c.class_id].sum())
        per_class.append({'class_name': c.name, 'support': support,
                          'precision': float(matrix[c.class_id, c.class_id] / predicted_count) if predicted_count else 0.,
                          'recall': result['per_class_recall'][c.class_id],
                          'f1': result['per_class_f1'][c.class_id]})
    bins = []
    for index in range(10):
        selected = (confidence >= index / 10) & (confidence < (index + 1) / 10 if index < 9 else confidence <= 1)
        if selected.any():
            bins.append({'lower': index / 10, 'upper': (index + 1) / 10,
                         'count': int(selected.sum()), 'accuracy': float(correct[selected].mean()),
                         'mean_confidence': float(confidence[selected].mean())})
    result.update(per_class=per_class, calibration_bins=bins,
                  expected_calibration_error=sum(b['count'] / len(labels) * abs(b['accuracy'] - b['mean_confidence']) for b in bins),
                  mean_confidence=float(confidence.mean()),
                  high_confidence_errors=int(((confidence >= .9) & ~correct).sum()),
                  high_confidence_count=int((confidence >= .9).sum()),
                  occupied_square_accuracy=float(correct[labels != 0].mean()) if (labels != 0).any() else None,
                  error_count=int((~correct).sum()))
    return result
