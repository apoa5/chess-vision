"""Save inspectable detection output, including failures and candidate evidence."""

from dataclasses import asdict
import json
from pathlib import Path

import cv2

from src.utils.image_io import save_image
from src.utils.visualization import draw_board_corners
from src.vision.board_detection import BoardDetection, DetectionSettings, preprocess_image


def save_detection_debug(path: str | Path, image, result: BoardDetection,
                         settings: DetectionSettings) -> dict:
    """Save original-resolution overlay, working-resolution edges, and JSON."""
    path = Path(path)
    overlay = draw_board_corners(image, result.corners) if result.success else image.copy()
    if not result.success:
        cv2.putText(overlay, "Detection failed; see JSON for reason", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    _, edges, _ = preprocess_image(image, settings)
    edge_path = path.with_name(f"{path.stem}_edges.png")
    json_path = path.with_suffix(".json")
    save_image(path, overlay)
    save_image(edge_path, edges)
    record = {"success": result.success, "reason": result.reason, "score": result.score,
              "corners": result.corners.tolist() if result.success else None,
              "corner_order": ["TL", "TR", "BR", "BL"],
              "settings": asdict(settings), "diagnostics": result.diagnostics,
              "overlay_path": str(path), "edges_path": str(edge_path)}
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
