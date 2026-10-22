"""Frozen A.1 contract; local-asset checks skip explicitly on source-only installs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.chess.labels import CLASS_NAMES, LABEL_VERSION

ROOT = Path(__file__).resolve().parents[1]
BASELINE = json.loads((ROOT / 'configs/baselines/a1_20261018.json').read_text())


def test_frozen_label_and_preprocessing_contract():
    assert list(CLASS_NAMES) == BASELINE['class_names']
    assert LABEL_VERSION == BASELINE['checkpoint']['label_version']
    for name in ('src/chess/labels.py', 'src/vision/dataset_loader.py'):
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == BASELINE['preserved_sha256'][name]


def test_real_baseline_split_isolation():
    path = ROOT / 'data/classification/crops.json'
    if not path.exists():
        pytest.skip('Local ignored baseline dataset manifest is unavailable')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASELINE['dataset']['manifest_sha256']
    manifest = json.loads(path.read_text())
    assignment = {}
    for row in manifest['results']:
        for key in (('position', row['position_id']), ('source', row['source_image'])):
            assert assignment.setdefault(key, row['split']) == row['split']
        assert row['split'] == BASELINE['dataset']['split_report']['position_assignment'][row['position_id']]
    assert len(manifest['results']) == 10240


def test_real_checkpoint_load_and_repeatability():
    torch = pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    from src.vision.inference import PieceClassifier
    path = ROOT / BASELINE['checkpoint']['path']
    if not path.exists():
        pytest.skip('Local ignored epoch-16 checkpoint is unavailable')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASELINE['checkpoint']['sha256']
    model = PieceClassifier(path, cpu_threads=4)
    assert model.checkpoint_epoch == 16
    crop = dict(image=np.full((80, 80, 3), 128, np.uint8),
                target_polygon=[[20, 20], [59, 20], [59, 59], [20, 59]], row=0, col=0)
    first = model.predict_with_logits([crop])[1]
    second = model.predict_with_logits([crop])[1]
    assert np.asarray(first).shape == (1, 13)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-7)
    assert not model.model.training
