from dataclasses import replace

import numpy as np
import pytest

from src.vision.board_detection import BoardDetection, DetectionSettings
from src.vision.evaluation import apply_review, detection_signature, summarize_evaluation


def test_review_signature_invalidates_changed_photo_settings_and_corners(tmp_path):
    photo = tmp_path / "photo.jpg"
    photo.write_bytes(b"original photo")
    result = BoardDetection(np.array([[1, 1], [90, 1], [90, 90], [1, 90]], dtype=np.float32))
    settings = DetectionSettings()
    signature = detection_signature(photo, result, settings)
    assert detection_signature(photo, result, replace(settings, min_score=.6)) != signature
    changed = BoardDetection(result.corners + 3)
    assert detection_signature(photo, changed, settings) != signature
    photo.write_bytes(b"different photo")
    assert detection_signature(photo, result, settings) != signature


def test_stale_or_missing_reviews_do_not_count_as_correct():
    assert apply_review(True, "new", {"detection_signature": "old", "board_localization_correct": True})["board_localization_correct"] is None
    assert apply_review(True, "new", None)["board_localization_correct"] is None
    assert apply_review(False, "new", None)["board_localization_correct"] is False


def test_invalid_approval_of_failure_is_rejected():
    with pytest.raises(ValueError, match="cannot approve"):
        apply_review(False, "same", {"detection_signature": "same", "board_localization_correct": True})


def test_summary_requires_all_outcomes_to_report_accuracy():
    records = [{"success": True, "board_localization_correct": True},
               {"success": True, "board_localization_correct": None}]
    assert summarize_evaluation(records)["localization_accuracy"] is None
    records[1]["board_localization_correct"] = False
    summary = summarize_evaluation(records)
    assert summary["candidate_found_count"] == 2
    assert summary["localization_accuracy"] == .5
    assert summary["target_met"] is False
