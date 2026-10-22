"""Evaluation reruns can preserve historical reports without changing defaults."""
import json
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('kind', ['classifier', 'positions'])
@pytest.mark.parametrize('preserve', [False, True])
def test_evaluator_metadata_report_option(tmp_path, monkeypatch, kind, preserve):
    torch = pytest.importorskip('torch')
    pytest.importorskip('torchvision')
    from scripts import evaluate_classifier, evaluate_positions
    from src.chess.labels import CLASS_NAMES, LABEL_VERSION

    module = evaluate_classifier if kind == 'classifier' else evaluate_positions
    historical = tmp_path / ('day12_classifier_evaluation.json' if kind == 'classifier'
                             else 'day14_position_evaluation.json')
    historical.write_text('historical report\n')
    output = tmp_path / 'rerun'
    manifest = tmp_path / 'crops.json'
    manifest.write_text(json.dumps({'results': [], 'split_report': {'dataset_fingerprint': 'dataset',
                                                   'position_assignment': {}}}))
    for name, value in [('prepared_boards.json', []), ('images.json', {}), ('positions.json', {})]:
        (tmp_path / name).write_text(json.dumps(value))
    config = {'classifier': {'dataset_paths': {'metadata': manifest}, 'cpu_threads': 1,
                             'checkpoint_path': tmp_path / 'model.pt', 'batch_size': 1},
              'data': {'metadata': tmp_path}}
    checkpoint = dict(manifest_sha256='hash', dataset_fingerprint='dataset',
                      class_names=list(CLASS_NAMES), label_version=LABEL_VERSION,
                      input_channels=4, architecture='mobilenet_v3_small', input_size=32,
                      epoch=16, settings={}, validation={}, model_state_dict={})
    monkeypatch.setattr(module, 'load_config', lambda: config)
    monkeypatch.setattr(module, 'file_hash', lambda path: 'hash')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: checkpoint)
    if kind == 'classifier':
        class Dataset:
            rows = [dict(crop_path='unused.png', crop_sha256='hash', class_id=0,
                         class_name='empty', crop_id='square',
                         source_image='board.jpg', position_id='position_006')]

            def __len__(self):
                return 1

            def __getitem__(self, index):
                return torch.zeros(4, 32, 32), 0

        class Model(torch.nn.Module):
            def load_state_dict(self, state):
                pass

            def forward(self, images):
                return torch.zeros(len(images), 13)

        monkeypatch.setattr(module, 'ContextSquareDataset', lambda *args, **kwargs: Dataset())
        monkeypatch.setattr(module, 'create_classifier', lambda **kwargs: Model())
        monkeypatch.setattr(module, 'contact_sheet', lambda *args: None)
        monkeypatch.setattr(module, 'confusion_plot', lambda *args: None)
    else:
        monkeypatch.setattr(module, 'PieceClassifier', lambda *args, **kwargs:
                            SimpleNamespace(checkpoint_epoch=16))
    argv = ['evaluate', '--output-dir', str(output)]
    if preserve:
        argv.append('--no-metadata-report')
    monkeypatch.setattr('sys.argv', argv)
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        module.main()
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)
    report = output / ('evaluation.json' if kind == 'classifier' else 'summary.json')
    assert report.exists()
    if preserve:
        assert historical.read_text() == 'historical report\n'
    else:
        assert json.loads(historical.read_text()) == json.loads(report.read_text())
