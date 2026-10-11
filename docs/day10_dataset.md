# Day 10: independent evaluation and model inputs

The approved dataset has 160 boards and 10,240 crops from 18 positions. The 180
raw photographs remain registered; 20 have no usable board localization and are
excluded. Label approval records reflect the user's completed review of the
corrected polygon previews. Individual visibility assessments are separate.

## Reproduce the split

```bash
.venv/bin/python scripts/split_dataset.py
```

Seed 42 assigns whole positions to 12 training, 3 validation and 3 test positions.
This approximates 70/15/15 while keeping both held-out sets at three positions.
No source position or image crosses splits. Duplicate piece placements and
identical source-photo hashes are grouped together. The position used to calibrate
border-marking templates is restricted to training to avoid preprocessing leakage.
The search uses class counts to balance coverage, never classifier performance.

| Split | Positions | Boards | Crops |
| --- | ---: | ---: | ---: |
| Training | 12 | 112 | 7,168 |
| Validation | 3 | 25 | 1,600 |
| Test | 3 | 23 | 1,472 |

All 13 classes appear in every split. `data/metadata/day10_split_report.json`
records assignments, dataset fingerprint, seed, counts and loss weights.
`day10_class_distribution.csv` is the complete class-count table. Split manifests
live in `data/classification/{train,val,test}/manifest.json` and reference existing
crop files; no image duplication is needed. Regenerate splits after rebuilding or
changing the dataset. Keep the test set fixed during model development.

## Class imbalance

Empty squares make up roughly 70% of the samples. Use square-root inverse-frequency
class weights for cross-entropy loss, computed from training counts only and
normalized to mean one across represented classes. This moderates rare-class
weights without duplicating images. The weights are prepared here; applying the
loss is part of Day 11. Report per-class metrics as well as overall accuracy.

## Context and the target-square mask

```bash
.venv/bin/python -m pip install -r requirements-training.txt
```

The optional requirements file installs the CPU PyTorch build tested here. GPU
training setup can be chosen when starting Day 11.

```python
from torch.utils.data import DataLoader
from src.utils.config import load_config
from src.vision.dataset import ContextSquareDataset

config = load_config()['classifier']
dataset = ContextSquareDataset(
    config['dataset_paths']['metadata'], 'train',
    input_size=config['input_size'], seed=config['seed'],
    augmentation=config['augmentation'],
)
dataset.set_epoch(0)
images, labels = next(iter(DataLoader(dataset, batch_size=8, num_workers=0)))
assert images.shape == (8, 4, 224, 224)
```

Each item returns a float32 tensor and an integer class ID. Channels 0–2 are
ImageNet-normalized RGB context; channel 3 is a binary mask of the target-square
polygon. The label describes the occupant of this polygon only. Neither FEN nor
chess coordinates are model inputs. Rasterization occurs before resizing, and
nearest-neighbor resizing preserves mask values. The model sees clean image
pixels, not the green diagnostic outlines.

Day 11 must adapt the baseline's three-channel input layer to four channels and
allow the mask-input parameters to train. Otherwise the supplied target location
would not influence classification. Context cannot recover hidden piece parts.

Training-only augmentation uses mild brightness/contrast changes, occasional
3×3 blur, and small noise. No rotations, flips, translations or crops are applied:
these could cut off tall pieces or change their association with the target square.
Masks are unchanged by photometric augmentation. Randomness depends on seed,
epoch and crop ID, making it reproducible independently of sample access order.
Call `set_epoch` before each training epoch; use nonpersistent workers if adding
multiprocess loading. Validation and test preprocessing is deterministic.

The loader follows the map-style dataset contract described in the
[PyTorch dataset documentation](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html).

## Validation

```bash
.venv/bin/python -m pytest tests/test_dataset_splits.py tests/test_classification_dataset.py tests/test_grid_geometry.py tests/test_config.py -q
```

Tests cover reproducibility, split independence, duplicate grouping, calibration
isolation, training-only weighting, target-mask preservation, deterministic
held-out preprocessing, unsafe-manifest rejection and actual tensor batching.
No classifier has been trained or evaluated yet. Shared board, camera and lighting
still limit generalization claims; unseen positions do not measure new equipment.
