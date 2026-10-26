"""C.2 mapping, preprocessing, deterministic inference and cohort safeguards."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torchvision.io import write_png
from torchvision import transforms

from src.scanning.square_evidence import SQUARES
from src.vision.chessqueries_offline import ChessQueriesCPU, SYMBOLS, map_outputs, official_modules
from scripts.benchmark_chessqueries_c2 import configuration, write_fresh, load_cohort


def test_known_logit_positions_preserve_canonical_orientation_and_classes():
    z = np.full((1,64,13), -100.)
    z[0,:,0] = 0
    z[0,0,10] = 100  # a8 black rook
    z[0,63,6] = 100  # h1 white king
    out = map_outputs(z)
    assert out['canonical_squares'] == list(SQUARES)
    assert out['canonical_squares'][0] == 'a8' and out['canonical_squares'][-1] == 'h1'
    assert out['predicted_class_ids'][0] == 10 and out['predicted_class_ids'][-1] == 6
    assert out['class_names'][10] == 'black_rook' and out['class_names'][6] == 'white_king'


@pytest.mark.parametrize('bad', [np.zeros((64,13)), np.zeros((1,63,13)),
                                np.full((1,64,13), np.nan), np.full((1,64,13), np.inf)])
def test_invalid_or_nonfinite_logits_are_rejected(bad):
    with pytest.raises(ValueError):
        map_outputs(bad)


def test_softmax_stability_and_all_classes_retained():
    z = np.tile(np.arange(13), (1,64,1)).astype(float) * 1000
    out = map_outputs(z)
    assert np.isfinite(out['probabilities']).all()
    assert np.allclose(np.sum(out['probabilities'],axis=1),1)
    assert np.array_equal(out['raw_logits'],z[0])
    assert len(out['probabilities'][0]) == 13 and not out['confidence_calibrated']


def test_wrong_order_is_rejected_without_best_rotation_search():
    z = np.zeros((1,64,13))
    with pytest.raises(ValueError):
        map_outputs(z, symbols=SYMBOLS[::-1])
    with pytest.raises(ValueError):
        map_outputs(z, squares=SQUARES[::-1])


@pytest.mark.parametrize('threads', [0,5,True,1.5])
def test_cpu_thread_bound_rejected_before_loading(threads):
    with pytest.raises(ValueError):
        ChessQueriesCPU('does-not-exist.json', threads=threads)


class FrozenModel(torch.nn.Module):
    def forward(self,x):
        assert not self.training and torch.is_inference_mode_enabled()
        assert x.shape == (1,3,644,644) and x.dtype == torch.float32 and x.device.type == 'cpu'
        assert not x.requires_grad
        return torch.arange(13,dtype=torch.float32).view(1,1,13).expand(1,64,13)


def test_predict_is_inference_mode_batch_one_and_deterministic(tmp_path):
    image = torch.arange(3*17*25).remainder(256).to(torch.uint8).reshape(3,17,25)
    path = tmp_path/'source.png'
    write_png(image,path)
    original = path.read_bytes()
    adapter = object.__new__(ChessQueriesCPU)
    adapter.torch = torch
    adapter.model = FrozenModel().eval()
    adapter.transform = transforms.Compose([
        transforms.Lambda(lambda x:x/255), transforms.Resize((644,644),antialias=True),
        transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    a,b = adapter.predict(path),adapter.predict(path)
    assert a['raw_logits'] == b['raw_logits'] and a['input_tensor_sha256'] == b['input_tensor_sha256']
    assert a['orientation_policy'] == 'as_decoded_canonical_queries_no_rotation_or_GT_alignment'
    assert path.read_bytes() == original


def test_outputs_are_not_overwritten(tmp_path):
    p = tmp_path/'out.json'
    write_fresh(p, {'original':True})
    with pytest.raises(ValueError):
        write_fresh(p, {'overwrite':True})
    assert json.loads(p.read_text()) == {'original':True}


def test_configuration_rejects_training_or_holdout(tmp_path):
    original = json.loads(Path('configs/experiments/chessqueries_c2.json').read_text())
    for key in ('final_holdout','training','fusion'):
        c = dict(original);c[key] = True
        p=tmp_path/(key+'.json');p.write_text(json.dumps(c))
        with pytest.raises(ValueError):configuration(p)


def test_registered_cohort_grouping_and_versioned_label_correction():
    c = configuration('configs/experiments/chessqueries_c2.json')
    _,frames,fens = load_cohort(c)
    assert len(frames) == 55 and len({f['session_id'] for f in frames}) == 5
    assert len({f['frame_id'] for f in frames}) == 55
    assert all(len(f['truth']) == len(f['baseline_class_ids']) == 64 for f in frames)
    middle = next(f for f in frames if 'middlegame' in f['session_id'])
    assert middle['truth'][SQUARES.index('c7')] == 7
    assert '1bpnbpp1' in fens[middle['session_id']]
    original = json.loads(Path('data/metadata/evidence_d2_ground_truth_corrections_v1.json').read_text())
    assert '1b1nbpp1' in original['corrections'][0]['original_fen']


def test_official_transform_and_vocab_match_adapter_contract():
    # Optional-source integration test, no weights/model or network required.
    release_path=Path('outputs/evaluation/chessqueries_c2_release_v1/release.json')
    if not release_path.exists():pytest.skip('Pinned optional source unavailable')
    pytest.importorskip('timm');pytest.importorskip('safetensors')
    previous={n for n in sys.modules if n=='chessqueries' or n.startswith('chessqueries.')}
    if previous:pytest.skip('Run official import check in a fresh test process')
    try:
        _,official = official_modules(json.loads(release_path.read_text()))
        image=torch.arange(3*23*19).remainder(256).reshape(3,23,19).float()
        expected=transforms.Normalize([.485,.456,.406],[.229,.224,.225])(
            transforms.Resize((644,644),antialias=True)(image/255))
        actual=official.build_transform(644,train=False)(image)
        assert torch.equal(expected,actual)
    finally:
        for name in list(sys.modules):
            if name=='chessqueries' or name.startswith('chessqueries.'):
                del sys.modules[name]
