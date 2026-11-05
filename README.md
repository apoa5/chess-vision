# Chess-Vision

Chess-Vision reconstructs physical chessboard positions from camera images and
smartphone scans. It combines board localization, semantic orientation, RGB plus
target-square-mask classification, and a browser-based board preview.

The product goal is **guided scan → editable position → user confirmation →
Stockfish analysis**. Recognition is imperfect: a predicted position must be
reviewed, and camera images cannot determine move history or every full-FEN field.

## Current capabilities

- Phone camera capture, automatic board proposals, manual alignment fallback,
  tracking, and selected-frame recognition.
- Canonical 8×8 board preview with confirm/retry review and capture/replay tools.
- Offline video keyframe selection, per-square evidence analysis, and fusion
  experiments, separate from the live application.
- Position/session-grouped model training and evaluation, including bounded Modal
  GPU workflows. The live scanner retains its original MobileNetV3 profile;
  B.2 and C.9 are separate offline experimental profiles.
- A Python Stockfish/UCI wrapper and FEN utilities. Board editing and the integrated
  scan-to-engine experience remain planned work.

### Latest recognition experiment

C.9 fine-tuned ConvNeXt V2 Tiny using a new dataset of **179 photographs and
184 reviewed video frames**, covering 29 distinct positions. On its retrospective
internal test (59 board views from four positions):

| Model | Overall accuracy | Occupied-square accuracy | Exact board views |
|---|---:|---:|---:|
| Frozen B.2 MobileNetV3 | 79.00% | 72.09% | 11/59 |
| C.9 ConvNeXt V2 Tiny | 99.18% | 99.22% | 41/59 |

These are development results, not independent generalization evidence. Data and
training differ between models, some examples were historically exposed to B.2,
and video frames are correlated. ConvNeXt has **not** replaced the live model; it
is slower on CPU and its pretrained weights have noncommercial restrictions.
See the [C.9 report](docs/evaluation/convnext_v2_c9.md) for splits, licensing,
per-position results, failure cases, and reproduction commands.

## Local setup

Use Python 3.12 and Node.js 22.16 or later.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# CPU classifier dependencies; needed for recognition and model-related tests:
python -m pip install -r requirements-training.txt
```

Model weights and personal media are not included in Git. See
[model assets](models/README.md) and the relevant experiment report for required
checkpoints; changing weights also requires matching preprocessing and class maps.
Settings live in [config/settings.yaml](config/settings.yaml).

### Browser prototype

```bash
npm --prefix web ci --ignore-scripts
npm --prefix web run build
python scripts/serve_scan_web.py
```

For phone camera access, use trusted HTTPS on your local network. See
[phone setup](docs/guided_scan_prototype.md#iphone-without-a-mac),
[recognition](docs/guided_scan_recognition.md), and
[capture/replay](docs/guided_scan_capture.md).

### Stockfish

Install [Stockfish](https://stockfishchess.org/download/) separately, then configure
its executable path or set the local override:

```bash
export STOCKFISH_PATH=/absolute/path/to/stockfish
python scripts/test_stockfish.py
```

This checks the Python engine integration; it does not enable engine analysis in
the browser. The assisted MVP plan tracks remaining engine/runtime work.

## Validation

```bash
python -m pytest
npm --prefix web test
npm --prefix web run build
```

Some Python tests require private datasets, model checkpoints, or a working
Stockfish executable. Individual experiment reports specify focused suites and
required artifacts.

Keep environment files, API credentials, private TLS keys, original recordings,
and checkpoints out of Git. Configure credentials locally; commit only placeholder
examples. The repository's ignore rules cover common local credential files.

## Project documentation

- [Assisted MVP implementation plan](docs/MVP_ASSISTED_IMPLEMENTATION_PLAN.md)
- [Research roadmap](IMPLEMENTATION_PLAN.md) and [technical research](VISION_RESEARCH.md)
- [Vision conventions](docs/vision_pipeline.md)
- [Geometry reliability audit](docs/evaluation/geometry_reliability_b1b.md)
- [Offline keyframe selection](docs/evaluation/keyframes_d1.md)
- [Multi-view evidence](docs/evaluation/evidence_d2.md) and
  [fusion reliability](docs/evaluation/fusion_reliability_d3b.md)
- [Dataset metadata](data/metadata/README.md)

Historical day-by-day evaluations remain under `docs/`; the reports describe their
original models and dataset boundaries and should not be treated as current
production guarantees.
