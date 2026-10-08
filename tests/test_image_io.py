import cv2
import numpy as np
import pytest

from src.utils.image_io import ensure_output_directory, load_image, save_image


def test_png_round_trip_creates_parents_and_preserves_bgr(tmp_path):
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :32] = (255, 0, 0)  # blue in BGR
    image[:, 32:] = (0, 0, 255)  # red in BGR
    path = tmp_path / "nested" / "board.png"
    assert save_image(path, image) == path
    np.testing.assert_array_equal(load_image(path), image)


def test_grayscale_loads_as_bgr(tmp_path):
    image = np.full((16, 24), 120, dtype=np.uint8)
    path = save_image(tmp_path / "gray.png", image)
    loaded = load_image(path)
    assert loaded.shape == (16, 24, 3)
    assert np.all(loaded == 120)


def test_missing_image(tmp_path):
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        load_image(tmp_path / "missing.png")


def test_directory_is_not_image(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        load_image(tmp_path)


def test_corrupt_image(tmp_path):
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="could not decode"):
        load_image(path)


@pytest.mark.parametrize("image", [None, np.empty((0, 10, 3), dtype=np.uint8),
                                   np.zeros((10, 10, 2), dtype=np.uint8),
                                   np.zeros((10, 10, 3), dtype=np.float32)])
def test_invalid_image_cannot_be_saved(tmp_path, image):
    path = tmp_path / "invalid.png"
    with pytest.raises(ValueError):
        save_image(path, image)
    assert not path.exists()


def test_unsupported_output_format(tmp_path):
    with pytest.raises(OSError, match="could not save image"):
        save_image(tmp_path / "board.unsupported", np.zeros((10, 10, 3), dtype=np.uint8))


def test_encoder_false_is_not_silent_success(tmp_path, monkeypatch):
    monkeypatch.setattr(cv2, "imwrite", lambda *args: False)
    with pytest.raises(OSError, match="could not save image"):
        save_image(tmp_path / "board.png", np.zeros((10, 10, 3), dtype=np.uint8))


def test_output_directory_blocked_by_file(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("existing file")
    with pytest.raises(OSError, match="Could not create output directory"):
        ensure_output_directory(blocker / "nested")
