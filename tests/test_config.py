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


@pytest.mark.parametrize("key,value", [("context_margin_squares", -1), ("crop_padding_squares", 3),
                                      ("context_margin_squares", ".nan"), ("crop_padding_squares", "true")])
def test_invalid_context_settings(tmp_path, key, value):
    path = tmp_path / "settings.yaml"
    text = DEFAULT_CONFIG.read_text()
    old = "2.0"
    path.write_text(text.replace(f"{key}: {old}", f"{key}: {value}"))
    with pytest.raises(ValueError):
        load_config(path)


def test_classifier_config_paths_and_defaults():
    classifier = load_config()['classifier']
    assert classifier['architecture'] == 'mobilenet_v3_small'
    assert classifier['input_size'] == 224
    assert classifier['dataset_paths']['train'] == PROJECT_ROOT/'data/classification/train'
    assert classifier['checkpoint_path'].is_absolute()


@pytest.mark.parametrize('replacement', ['batch_size: 0', 'learning_rate: .nan', 'epochs: true'])
def test_invalid_classifier_config(tmp_path, replacement):
    key = replacement.split(':')[0]
    import yaml
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    data['classifier'][key] = yaml.safe_load(replacement)[key]
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=key):
        load_config(path)
