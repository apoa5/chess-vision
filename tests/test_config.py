import pytest

from src.utils.config import DEFAULT_CONFIG, PROJECT_ROOT, load_config


def test_default_config():
    settings = load_config()
    assert settings["board"]["normalized_size"] == 800
    assert settings["data"]["raw"] == PROJECT_ROOT / "data/raw"
    assert settings["outputs"]["square_crops"].is_absolute()


def test_engine_override(monkeypatch):
    monkeypatch.setenv("STOCKFISH_PATH", "/tmp/custom-stockfish")
    assert load_config()["stockfish"]["executable"] == "/tmp/custom-stockfish"


def test_invalid_board_size(tmp_path):
    config = tmp_path / "settings.yaml"
    config.write_text(DEFAULT_CONFIG.read_text().replace("800", "801"))
    with pytest.raises(ValueError, match="divisible by 8"):
        load_config(config)


def test_config_from_another_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert load_config()["data"]["raw"] == PROJECT_ROOT / "data/raw"
