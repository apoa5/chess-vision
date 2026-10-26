"""C.2 CPU-only, zero-shot adapter. Not imported by production inference.

Uses unmodified, hash-pinned upstream modules. Narrow package namespaces avoid
upstream __init__ reexports of unrelated LLM/training integrations.
"""
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import types

import numpy as np

from src.chess.labels import CLASS_NAMES
from src.scanning.square_evidence import SQUARES

WEIGHT_SHA = '486d10b258ba57d43d8ca861d20547093ca6591c0ad53da53a492716ef1c7705'
SYMBOLS = tuple('.PNBRQKpnbrqk')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def memory_info():
    """Linux RSS high-water mark and current system availability, bytes."""
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, value = line.split(':', 1)
        fields[name] = int(value.strip().split()[0]) * 1024
    return dict(peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                system_total_bytes=fields['MemTotal'], system_available_bytes=fields['MemAvailable'])


def map_outputs(logits, symbols=SYMBOLS, squares=SQUARES):
    if tuple(symbols) != SYMBOLS or tuple(squares) != SQUARES:
        raise ValueError('Unsupported class or canonical square ordering')
    z = np.asarray(logits, dtype=np.float64)
    if z.shape != (1, 64, 13) or not np.isfinite(z).all():
        raise ValueError('Expected finite [1,64,13] logits')
    z = z[0]
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return dict(raw_logits=z.tolist(), probabilities=p.tolist(),
                predicted_class_ids=p.argmax(axis=1).tolist(),
                class_names=list(CLASS_NAMES), canonical_squares=list(SQUARES),
                confidence_calibrated=False)


def official_modules(release):
    root = Path(release['sources'][0]['path']).parent
    for row in release['sources']:
        if sha256(row['path']) != row['sha256']:
            raise ValueError('Upstream source changed: ' + row['path'])
    vendor = root.resolve()
    # Never allow an installed/different package to shadow the pinned modules.
    if any(n == 'chessqueries' or n.startswith('chessqueries.') for n in sys.modules):
        raise ValueError('Load one pinned ChessQueries adapter per fresh process')
    for name in ('chessqueries', 'chessqueries.core', 'chessqueries.models', 'chessqueries.data'):
        package = types.ModuleType(name)
        package.__path__ = [str(vendor.joinpath(*name.split('.')))]
        package.__package__ = name
        sys.modules[name] = package
    pieces = importlib.import_module('chessqueries.core.pieces')
    square = importlib.import_module('chessqueries.core.square')
    core = sys.modules['chessqueries.core']
    for name in ('NUM_PIECES', 'Color', 'Piece', 'PieceType'):
        setattr(core, name, getattr(pieces, name))
    core.ALL_SQUARES = square.ALL_SQUARES
    core.Square = square.Square
    if tuple(p.symbol for p in pieces.Piece) != SYMBOLS or tuple(s.name for s in square.ALL_SQUARES) != SQUARES:
        raise ValueError('Official vocabulary differs from Chess-Vision')
    checkpoint = importlib.import_module('chessqueries.models.checkpoint')
    transforms = importlib.import_module('chessqueries.data.transforms')
    return checkpoint, transforms


class ChessQueriesCPU:
    def __init__(self, release_path, *, threads=4):
        if type(threads) is not int or not 1 <= threads <= 4:
            raise ValueError('C.2 permits one to four CPU threads')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_XET'] = '1'
        import torch
        self.torch = torch
        torch.set_num_threads(threads)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        release = json.loads(Path(release_path).read_text())
        weights = release['weights']
        if weights['sha256'] != WEIGHT_SHA or sha256(weights['path']) != WEIGHT_SHA:
            raise ValueError('Only the pinned original-paper ViT-S release is allowed')
        self.release = release
        start = time.perf_counter()
        checkpoint, transforms = official_modules(release)
        metadata = checkpoint.read_safetensors_metadata(Path(weights['path']))
        if metadata['encoder_name'] != 'vit_small_patch14_dinov2.lvd142m' or metadata['resolution'] != '644' or metadata['normalization'] != 'imagenet':
            raise ValueError('Unexpected architecture or preprocessing metadata')
        self.model = checkpoint.load_safetensors_model(Path(weights['path']), device='cpu').float().eval()
        self.transform = transforms.build_transform(resolution=644, train=False)
        self.model_load_seconds = time.perf_counter() - start
        self.metadata = metadata
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        if any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError('CPU FP32 model required')

    def predict(self, image_path):
        """As-decoded RGB, canonical learned queries; no GT or geometry argument."""
        from torchvision.io import ImageReadMode, read_image
        torch = self.torch
        start = time.perf_counter()
        with torch.inference_mode():
            image = read_image(str(image_path), mode=ImageReadMode.RGB).float()
            tensor = self.transform(image).unsqueeze(0).to(device='cpu', dtype=torch.float32)
            if tensor.shape != (1, 3, 644, 644) or not torch.isfinite(tensor).all():
                raise ValueError('Invalid official preprocessed tensor')
            preprocessed = time.perf_counter()
            logits = self.model(tensor)
            forwarded = time.perf_counter()
            output = map_outputs(logits.cpu().numpy())
            output['input_tensor_sha256'] = hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()
        output.update(preprocessing_seconds=preprocessed-start, forward_seconds=forwarded-preprocessed,
                      inference_and_preprocessing_seconds=forwarded-start,
                      model_eval=not self.model.training, torch_inference_mode=True,
                      input_shape=[1, 3, 644, 644], output_shape=list(logits.shape),
                      orientation_policy='as_decoded_canonical_queries_no_rotation_or_GT_alignment')
        return output
