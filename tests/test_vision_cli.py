import subprocess
import sys

import numpy as np
import pytest

from src.utils.config import PROJECT_ROOT
from src.utils.image_io import load_image, save_image


def run_script(name, *args, cwd):
    return subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / name),
                           *map(str, args)], cwd=cwd, capture_output=True, text=True, timeout=15)


def test_preview_cli_from_another_directory(tmp_path):
    image = np.zeros((80, 120, 3), dtype=np.uint8)
    source = save_image(tmp_path / "source.png", image)
    output = tmp_path / "nested" / "preview.png"
    result = run_script("preview_image.py", source, "--output", output, "--max-size", 60,
                        "--corners", 10, 10, 110, 10, 110, 70, 10, 70, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert load_image(output).shape == (40, 60, 3)
    np.testing.assert_array_equal(load_image(source), image)


@pytest.mark.parametrize("script", ["preview_image.py", "detect_board.py",
                                   "rectify_board.py", "extract_squares.py", "process_board_image.py"])
def test_cli_help(script, tmp_path):
    result = run_script(script, "--help", cwd=tmp_path)
    assert result.returncode == 0
    assert "image" in result.stdout


def test_preview_cli_missing_input(tmp_path):
    result = run_script("preview_image.py", tmp_path / "missing.png", "--output",
                        tmp_path / "output.png", cwd=tmp_path)
    assert result.returncode == 1
    assert "Image file not found" in result.stderr
    assert "Traceback" not in result.stderr


def test_detection_failure_cli_saves_debug_files(tmp_path):
    source = save_image(tmp_path / "blank.png", np.zeros((80, 80, 3), dtype=np.uint8))
    output = tmp_path / "debug" / "blank.png"
    result = run_script("detect_board.py", source, "--output", output, cwd=tmp_path)
    assert result.returncode == 1
    assert "Detection failed" in result.stderr
    assert "Traceback" not in result.stderr
    assert output.is_file()
    assert output.with_suffix(".json").is_file()
    assert output.with_name("blank_edges.png").is_file()


def test_rectification_cli_from_another_directory(tmp_path):
    source = save_image(tmp_path / "source.png", np.full((80, 80, 3), 123, dtype=np.uint8))
    output = tmp_path / "nested" / "board.png"
    result = run_script("rectify_board.py", source, "--corners", 79, 79, 0, 79, 0, 0, 79, 0,
                        "--output", output, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert load_image(output).shape == (800, 800, 3)


def test_rectification_cli_rejects_failed_detection(tmp_path):
    source = save_image(tmp_path / "blank.png", np.zeros((80, 80, 3), dtype=np.uint8))
    output = tmp_path / "board.png"
    result = run_script("rectify_board.py", source, "--output", output, cwd=tmp_path)
    assert result.returncode == 1
    assert "Detection failed" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("script", ["rectify_board.py", "process_board_image.py"])
def test_cli_automatically_detects_board(tmp_path, script):
    image = np.full((480, 640, 3), 180, dtype=np.uint8)
    image[15:465, 15:625] = 245
    image[95:401, 145:451] = (45, 80, 110)
    y, x = np.indices((256, 256))
    board = image[120:376, 170:426]
    board[(x // 32 + y // 32) % 2 == 0] = (210, 225, 235)
    board[(x // 32 + y // 32) % 2 == 1] = (45, 70, 95)
    source = save_image(tmp_path / "checker.png", image)
    output = tmp_path / "rectified.png"
    if script == "process_board_image.py":
        result = run_script(script, source, "--output-dir", tmp_path / "pipeline", cwd=tmp_path)
        output = tmp_path / "pipeline" / "rectified_boards" / "checker.png"
        assert len(list((tmp_path / "pipeline" / "square_crops" / "checker").glob("r*_c*.png"))) == 64
    else:
        result = run_script(script, source, "--output", output, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    assert load_image(output).shape == (800, 800, 3)


def test_extraction_cli_saves_all_crops_and_metadata(tmp_path):
    import json
    source = save_image(tmp_path / "board.png", np.full((160, 160, 3), 123, np.uint8))
    output = tmp_path / "crops"
    result = run_script("extract_squares.py", source, "--output-dir", output, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    metadata = json.loads((output / "squares.json").read_text())
    assert len(metadata) == 64
    assert metadata[-1] == {"row": 7, "col": 7, "bounds": [140, 140, 160, 160], "filename": "r7_c7.png"}
    assert len(list(output.glob("r*_c*.png"))) == 64
    for item in metadata:
        np.testing.assert_array_equal(load_image(output / item["filename"]), np.full((20, 20, 3), 123, np.uint8))
    assert (output / "grid.png").is_file()


def test_full_pipeline_cli_failed_detection(tmp_path):
    source = save_image(tmp_path / "blank.png", np.zeros((80, 80, 3), np.uint8))
    output = tmp_path / "pipeline"
    result = run_script("process_board_image.py", source, "--output-dir", output, cwd=tmp_path)
    assert result.returncode == 1
    assert "Detection failed" in result.stderr
    assert (output / "board_detection" / "blank_corners.png").is_file()
    assert not (output / "square_crops").exists()
