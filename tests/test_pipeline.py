import json
from pathlib import Path

import numpy as np

from src.utils.config import load_config
from src.utils.image_io import load_image, save_image
from src.vision.board_detection import BoardDetection
from src.vision.pipeline import process_board_image
from scripts.evaluate_week1 import apply_warp_review


def test_pipeline_saves_unannotated_crops_and_review_signature(tmp_path, monkeypatch):
    import src.vision.pipeline as pipeline
    config = load_config()
    config["board"]["normalized_size"] = 80
    image = np.random.default_rng(3).integers(0, 256, (80, 80, 3), np.uint8)
    source = save_image(tmp_path / "source.png", image)
    corners = np.float32([[0, 0], [79, 0], [79, 79], [0, 79]])
    monkeypatch.setattr(pipeline, "detect_board_corners", lambda *args: BoardDetection(corners))
    result = process_board_image(source, config, tmp_path / "outputs")
    assert result["square_count"] == 64
    np.testing.assert_array_equal(load_image(result["board_path"]), image)
    metadata = json.loads((Path(result["crop_directory"]) / "squares.json").read_text())
    assert len(metadata) == 64
    for item in metadata:
        x0, y0, x1, y1 = item["bounds"]
        crop = load_image(Path(result["crop_directory"]) / item["filename"])
        np.testing.assert_array_equal(crop, image[y0:y1, x0:x1])
    review = {"warp_signature": result["warp_signature"], "warp_correct": True}
    assert apply_warp_review(result, review)["warp_correct"] is True
    assert apply_warp_review(result, None)["warp_correct"] is None
    image[40, 40] = 0
    save_image(source, image)
    changed = process_board_image(source, config, tmp_path / "outputs")
    assert apply_warp_review(changed, review)["warp_review_status"] == "stale"


def test_pipeline_failure_has_debug_but_no_crops(tmp_path, monkeypatch):
    import src.vision.pipeline as pipeline
    source = save_image(tmp_path / "blank.png", np.zeros((80, 80, 3), np.uint8))
    monkeypatch.setattr(pipeline, "detect_board_corners", lambda *args: BoardDetection(None, "No board"))
    output = tmp_path / "outputs"
    result = process_board_image(source, load_config(), output)
    assert not result["board_detected"]
    assert result["square_count"] == 0
    assert result["board_path"] is None and result["grid_path"] is None
    assert (output / "board_detection" / "blank_corners.png").is_file()
    assert not (output / "square_crops").exists()
    assert apply_warp_review(result, None)["warp_correct"] is False


def test_context_metadata_locates_target_and_marks_source_coverage(tmp_path, monkeypatch):
    import src.vision.pipeline as pipeline
    config = load_config()
    config["board"]["normalized_size"] = 80
    source = save_image(tmp_path / "source.png", np.full((80, 80, 3), 120, np.uint8))
    corners = np.float32([[0, 0], [79, 0], [79, 79], [0, 79]])
    monkeypatch.setattr(pipeline, "detect_board_corners", lambda *args: BoardDetection(corners))
    result = process_board_image(source, config, tmp_path / "outputs")
    assert result["board_bounds"] == [20, 20, 100, 100]
    assert result["context_square_count"] == 64
    context = load_image(result["context_path"])
    assert context.shape == (120, 120, 3)
    directory = Path(result["context_crop_directory"])
    metadata = json.loads((directory / "squares.json").read_text())
    assert len(metadata) == 64
    first = metadata[0]
    assert first["target_bounds"] == [20, 20, 30, 30]
    assert first["bounds"] == [0, 0, 50, 50]
    assert first["valid_fraction"] < 1
    crop = load_image(directory / first["filename"])
    np.testing.assert_array_equal(crop[20:30, 20:30], np.full((10, 10, 3), 120, np.uint8))
    assert (crop[:20, :20] == 0).all()
