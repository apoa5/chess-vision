"""Bind manual localization reviews to the exact photos and detector outputs."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path


def detection_signature(image_path, detection, settings) -> str:
    """Invalidate reviews if the photo, settings, or predicted corners change."""
    content = {
        "image_sha256": hashlib.sha256(Path(image_path).read_bytes()).hexdigest(),
        "settings": asdict(settings), "success": detection.success,
        "corners": detection.corners.round(1).tolist() if detection.success else None,
    }
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def apply_review(success: bool, signature: str, review: dict | None) -> dict:
    """No detection is a known failure; accepted candidates need a matching review."""
    if review is not None and review.get("detection_signature") == signature:
        correct = review.get("board_localization_correct")
        if type(correct) is not bool or (correct and not success):
            raise ValueError("Review correctness must be boolean and cannot approve a failed detection.")
        return {"board_localization_correct": correct, "review_status": "matched",
                "failure_category": review.get("failure_category", ""), "notes": review.get("notes", "")}
    return {"board_localization_correct": None if success else False,
            "review_status": "stale" if review else ("unreviewed" if success else "not_detected"),
            "failure_category": "" if success else "no_accepted_candidate", "notes": ""}


def summarize_evaluation(records: list[dict]) -> dict:
    total = len(records)
    known = sum(record["board_localization_correct"] is not None for record in records)
    correct = sum(record["board_localization_correct"] is True for record in records)
    return {"image_count": total, "candidate_found_count": sum(record["success"] for record in records),
            "known_correct_count": correct, "known_outcome_count": known,
            "unreviewed_count": total - known,
            "localization_accuracy": correct / total if total and known == total else None,
            "target_accuracy": .90,
            "target_met": correct / total >= .90 if total and known == total else None}
