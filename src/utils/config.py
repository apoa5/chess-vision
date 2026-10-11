"""Central settings; relative paths are resolved from the project root."""

import os
import math
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config/settings.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict:
    """Load and validate settings, with an optional STOCKFISH_PATH override."""
    with Path(path).open(encoding="utf-8") as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    engine = settings.get("stockfish", {})
    if not isinstance(engine, dict):
        raise ValueError("stockfish must be a mapping.")
    executable = os.environ.get("STOCKFISH_PATH") or engine.get("executable")
    if not isinstance(executable, str) or not executable.strip():
        raise ValueError("stockfish.executable must be a nonempty path or command.")
    settings["stockfish"] = {**engine, "executable": executable}
    board = settings.get("board", {})
    size = board.get("normalized_size") if isinstance(board, dict) else None
    if type(size) is not int or size <= 0 or size % 8:
        raise ValueError("board.normalized_size must be a positive integer divisible by 8.")
    for key, default in (("context_margin_squares", 2.0), ("crop_padding_squares", 2.0)):
        value = board.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 4:
            raise ValueError(f"board.{key} must be a finite number between 0 and 4.")
        board[key] = value
    if board["crop_padding_squares"] > board["context_margin_squares"]:
        raise ValueError("Crop padding cannot exceed the context margin.")
    learned=settings.get('learned_markings',{})
    if not isinstance(learned,dict) or type(learned.get('enabled',False)) is not bool:
        raise ValueError('learned_markings.enabled must be boolean')
    threshold=learned.get('min_character_confidence',.80)
    if type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<threshold<=1:
        raise ValueError('learned_markings.min_character_confidence must be in (0,1]')
    from src.vision.orientation import OrientationSettings
    orientation = settings.get("orientation", {})
    if not isinstance(orientation, dict):
        raise ValueError("orientation must be a mapping.")
    try:
        OrientationSettings(**orientation)
    except TypeError as error:
        raise ValueError(f"Invalid orientation settings: {error}") from error
    for section, keys in {
        "data": ("raw", "processed", "test", "metadata"),
        "outputs": ("board_detection", "rectified_boards", "square_crops"),
    }.items():
        paths = settings.get(section)
        if not isinstance(paths, dict):
            raise ValueError(f"{section} must be a mapping.")
        for key in keys:
            value = paths.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{section}.{key} must be a nonempty path.")
            paths[key] = (PROJECT_ROOT / Path(value).expanduser()).resolve()
    classifier = settings.get("classifier")
    if not isinstance(classifier, dict):
        raise ValueError("classifier must be a mapping.")
    if classifier.get("architecture") not in ("mobilenet_v3_small", "resnet18"):
        raise ValueError("Unsupported classifier architecture.")
    for key in ("input_size", "batch_size", "epochs"):
        if type(classifier.get(key)) is not int or classifier[key] <= 0:
            raise ValueError(f"classifier.{key} must be a positive integer.")
    if type(classifier.get("seed")) is not int or classifier["seed"] < 0:
        raise ValueError("classifier.seed must be a nonnegative integer.")
    for key in ("learning_rate", "confidence_threshold"):
        value = classifier.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"classifier.{key} must be a finite number in (0, 1].")
    for key,default in [('head_epochs',5),('early_stopping_patience',5),('cpu_threads',4)]:
        value=classifier.setdefault(key,default)
        if type(value) is not int or value<=0:
            raise ValueError(f'classifier.{key} must be a positive integer.')
    for key,default in [('fine_tune_learning_rate',.0001),('early_stopping_min_delta',.0001)]:
        value=classifier.setdefault(key,default)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=1:
            raise ValueError(f'classifier.{key} must be in (0,1].')
    fractions = classifier.get('split_fractions', [.7,.15,.15])
    if (not isinstance(fractions,list) or len(fractions)!=3 or
            any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in fractions) or
            not math.isclose(sum(fractions),1)):
        raise ValueError('classifier.split_fractions must be three positive fractions summing to one.')
    classifier['split_fractions'] = fractions
    trials = classifier.setdefault('split_search_trials',4000)
    if type(trials) is not int or trials<=0:
        raise ValueError('classifier.split_search_trials must be positive.')
    if classifier.setdefault('input_channels',4)!=4:
        raise ValueError('Context classifier requires RGB plus target-mask channel.')
    if classifier.setdefault('imbalance_strategy','sqrt_inverse_frequency_loss')!='sqrt_inverse_frequency_loss':
        raise ValueError('Unsupported class imbalance strategy.')
    augmentation = classifier.setdefault('augmentation',{})
    if not isinstance(augmentation,dict):
        raise ValueError('classifier.augmentation must be a mapping.')
    for key,default in [('brightness',.1),('contrast',.1),('blur_probability',.15),('noise_probability',.2),('noise_std',.01)]:
        value = augmentation.setdefault(key,default)
        if type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError(f'classifier.augmentation.{key} must be in [0,1].')
    dataset_paths = classifier.get("dataset_paths")
    if not isinstance(dataset_paths, dict):
        raise ValueError("classifier.dataset_paths must be a mapping.")
    for container, keys in ((dataset_paths, ("train", "val", "test", "metadata")),
                            (classifier, ("checkpoint_path",))):
        for key in keys:
            value = container.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Classifier path {key} must be nonempty.")
            container[key] = (PROJECT_ROOT / Path(value).expanduser()).resolve()
    return settings
