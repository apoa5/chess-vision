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


@pytest.mark.parametrize("script,extra,day", [
    ("detect_board.py", [], 4),
    ("rectify_board.py", ["--corners", 0, 0, 79, 0, 79, 79, 0, 79], 6),
    ("extract_squares.py", [], 7),
])
def test_stage_scaffolds_report_pending_work_without_outputs(tmp_path, script, extra, day):
    source = save_image(tmp_path / "source.png", np.zeros((80, 80, 3), dtype=np.uint8))
    output = tmp_path / "output"
    option = "--output-dir" if script == "extract_squares.py" else "--output"
    result = run_script(script, source, *extra, option, output, cwd=tmp_path)
    assert result.returncode == 1
    assert f"scheduled for Day {day}" in result.stderr
    assert "Traceback" not in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("script", ["preview_image.py", "detect_board.py",
                                   "rectify_board.py", "extract_squares.py"])
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
