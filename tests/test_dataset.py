import json

import chess
import numpy as np
import pytest

from src.utils.dataset import validate_dataset
from src.utils.image_io import save_image


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def dataset(tmp_path):
    raw, metadata = tmp_path / "raw", tmp_path / "metadata"
    raw.mkdir()
    metadata.mkdir()
    variations = {f"var_{i:02}": {"camera_angle_degrees": 90, "environment": "living_room",
                                "lighting": "overhead"} for i in range(1, 11)}
    positions = {f"position_{i:03}": {"fen": chess.STARTING_FEN} for i in range(1, 3)}
    images = {}
    for position_id in positions:
        for variation_id in variations:
            name = f"{position_id}_{variation_id}.png"
            path = save_image(raw / name, np.zeros((16, 16, 3), dtype=np.uint8))
            images[name] = {"path": str(path), "position_id": position_id,
                            "variation_id": variation_id, **variations[variation_id],
                            "orientation": "white_near_camera", "development_subset": True}
    write_json(metadata / "positions.json", positions)
    write_json(metadata / "images.json", images)
    write_json(metadata / "variations.json", variations)
    write_json(metadata / "development_subset.json", {"image_ids": list(images)})
    write_json(metadata / "capture_setup.json", {"capture_plan": {
        "target_position_count": 20, "target_variations_per_position": 10}})
    return metadata, raw


def test_dataset_validation_and_collection_target_are_separate(dataset):
    report = validate_dataset(*dataset)
    assert report["decoded_image_count"] == 20
    assert report["valid_fen_count"] == 2
    assert report["development_subset"]["image_count"] == 20
    assert report["collection_target"]["images"] == 200
    assert not any(report["collection_target"]["missing_variations_by_position"].values())


def test_unregistered_raw_photo_is_reported(dataset):
    metadata, raw = dataset
    save_image(raw / "position_003_var_01.png", np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="missing metadata.*position_003"):
        validate_dataset(metadata, raw)


def test_corrupt_photo_is_not_accepted(dataset):
    metadata, raw = dataset
    (raw / "position_001_var_01.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="could not decode"):
        validate_dataset(metadata, raw)


def test_changed_subset_membership_is_rejected(dataset):
    metadata, raw = dataset
    images = json.loads((metadata / "images.json").read_text())
    images["position_001_var_01.png"]["development_subset"] = False
    write_json(metadata / "images.json", images)
    with pytest.raises(ValueError, match="does not match"):
        validate_dataset(metadata, raw)


def test_invalid_position_fen_is_rejected(dataset):
    metadata, raw = dataset
    write_json(metadata / "positions.json", {
        "position_001": {"fen": "8/8/8/8/8/8/8/8 w - - 0 1"},
        "position_002": {"fen": chess.STARTING_FEN}})
    with pytest.raises(ValueError, match="position_001: invalid chess position"):
        validate_dataset(metadata, raw)
