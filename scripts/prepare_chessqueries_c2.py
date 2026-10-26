"""Download only the pinned ViT-S release and minimal official inference sources.

Run explicitly; no package import or automatic model download occurs elsewhere.
Artifacts belong under ignored outputs/evaluation/, never production models.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = 'f5aaaf38212cc7839b03ccf383455053905b3f7e'
WEIGHT_SHA = '486d10b258ba57d43d8ca861d20547093ca6591c0ad53da53a492716ef1c7705'
WEIGHT_BYTES = 63147914
WEIGHT_NAME = 'chessquerieslite-vits-644-safetensors-fp16-r1.safetensors'
WEIGHT_URL = 'https://huggingface.co/joelseytre/chessqueries/resolve/eadee700749fd2930a651982c9f8a753097f81dd/' + WEIGHT_NAME
SOURCES = [
    'LICENSE', 'chessqueries/__init__.py', 'chessqueries/core/__init__.py',
    'chessqueries/core/board.py', 'chessqueries/core/pieces.py',
    'chessqueries/core/square.py', 'chessqueries/models/__init__.py',
    'chessqueries/models/base.py', 'chessqueries/models/checkpoint.py',
    'chessqueries/models/chessqueries_model.py', 'chessqueries/data/__init__.py',
    'chessqueries/data/transforms.py',
]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, *, expected=None, expected_bytes=None, maximum=2_000_000):
    if path.exists():
        if expected is None or sha(path) == expected:
            return
        raise ValueError('Existing artifact hash mismatch: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    count = 0
    with urllib.request.urlopen(url, timeout=60) as response, temporary.open('wb') as out:
        if response.status != 200:
            raise ValueError('Full download required; got HTTP ' + str(response.status))
        while chunk := response.read(1024 * 1024):
            count += len(chunk)
            if count > maximum:
                raise ValueError('Download exceeds declared bound')
            out.write(chunk)
    if expected_bytes is not None and count != expected_bytes:
        raise ValueError('Incorrect artifact length')
    if expected is not None and sha(temporary) != expected:
        raise ValueError('Incorrect artifact SHA-256')
    temporary.replace(path)


def run(root):
    if (root / 'release.json').exists():
        raise ValueError('Release already registered; do not overwrite provenance')
    prior = json.loads(Path('docs/evaluation/input_representation_c1_public_sources_v1.json').read_text())
    pinned = {row['url']: row['sha256'] for row in prior['sources']}
    records = []
    for name in SOURCES:
        url = f'https://raw.githubusercontent.com/JSeytre/chessqueries/{REVISION}/{name}'
        path = root / 'vendor' / name
        download(url, path, expected=pinned.get(url))
        records.append(dict(path=str(path), url=url, sha256=sha(path), bytes=path.stat().st_size))
        print('Source:', name, flush=True)
    weights = root / 'weights' / WEIGHT_NAME
    download(WEIGHT_URL + '?download=true', weights, expected=WEIGHT_SHA,
             expected_bytes=WEIGHT_BYTES, maximum=WEIGHT_BYTES)
    record = dict(schema_version='chessqueries_c2_release_v1', revision=REVISION,
                  sources=records, weights=dict(path=str(weights), url=WEIGHT_URL,
                  sha256=WEIGHT_SHA, bytes=WEIGHT_BYTES),
                  larger_models_downloaded=False, third_party_code_modified=False)
    (root / 'release.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Verified ViT-S:', WEIGHT_SHA, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    run(parser.parse_args().root)
