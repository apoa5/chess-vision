# Task A.1 baseline — 2026-10-18

Baseline ID: `a1_20261018_15a5a879`. Source commit:
`15a5a8792a41f1a11c9287f6061f90fb126858a3`.
The supplied research and implementation plan were found at repository root,
as untracked `VISION_RESEARCH.md` and `IMPLEMENTATION_PLAN.md`, rather than in
`docs/`. They were read without moving or modifying them.

The versioned machine-readable record is
[`configs/baselines/a1_20261018.json`](../../configs/baselines/a1_20261018.json).
It records installed package versions, platform/Python, full position assignment,
split counts, dataset fingerprint, input availability, and SHA-256 identifiers.
Only the evaluator CLI report-writing option differs from the recorded commit;
the recognition pipeline, preprocessing, labels and model are unchanged.

## Existing implementation

The repository already implements board detection, padded perspective correction,
image-supported square polygons, border-coordinate orientation, masked-context
MobileNet classification, batched inference, canonical board reconstruction,
FEN serialization, chess consistency diagnostics and a Stockfish interface.
Dataset building requires board review signatures; splitting groups physical
positions, identical placements and duplicate source hashes. Training is already
implemented with validation-selected checkpoints and refuses an existing checkpoint.
None of these components was rebuilt or trained for A.1.

The browser already has camera capture/export and replay, perspective tracking in
a worker, marking-based acquisition/recovery, same-frame canonical geometry,
selected-frame recognition, digital preview, confirm/retry and stale-response
rejection. Recognition freezes one observation for review; retry requires another
explicit capture. The current flow does not fuse views or enable automatic scan
completion. The scan recognition API returns raw logits/probabilities with
`completion_enabled: false`. The historical raw-photo evaluator uses classical
border templates through `position_pipeline.py`; it does not exercise the newer
browser learned-marking bootstrap. These workflows should remain distinct in A.2.

## Model and input contract

- Checkpoint: `models/checkpoints/piece_classifier_best.pt`, 6,261,635 bytes,
  validation-selected epoch **16**, fine-tuning mode.
- SHA-256: `ff792985ecf22d5d1c6871c0e5d6154dd45ffdd4a767b497edee8726c69ca2f8`.
- Architecture: MobileNetV3 Small, four-channel stem, 13-class head. Loading uses
  saved weights on CPU with `pretrained=False`; no weights download is necessary.
- Label version: `chess_pieces_v1`. IDs 0–12, in order: empty, white pawn,
  knight, bishop, rook, queen, king, black pawn, knight, bishop, rook, queen, king.
- `src/vision/dataset_loader.py::preprocess_context_crop`: BGR → RGB, resize with
  `INTER_AREA` to 224×224, divide by 255; normalize RGB by ImageNet mean
  `[.485,.456,.406]` and standard deviation `[.229,.224,.225]`. Rasterize the
  target polygon before resize; resize this binary fourth channel with
  `INTER_NEAREST`. Output is contiguous float32 CHW. Evaluation has no augmentation.
- Context padding is two squares (200 pixels at the 800-pixel playing area).
  Polygon crops are approximately 500×500; the target is approximately 45 pixels
  wide after input resize. Actual polygons, rather than a fixed target rectangle,
  determine the mask.
- Camera-facing pixels are preserved. Output matrices rotate to White-bottom
  using CCW quarter turns: bottom=0, left=1, top=2, right=3. Semantic square IDs
  follow orientation, not image coordinates. See `docs/day9_dataset.md`.

## Dataset and evaluation configuration

`data/classification/crops.json` is the authoritative split manifest; prepared
geometry is in `data/classification/prepared_boards.json`. Per-split copies live
at `data/classification/{train,val,test}/manifest.json`. Ground truth and raw
photo membership are `data/metadata/positions.json` and `images.json`.

Dataset format: `camera_view_polygon_context_v3`.
Dataset fingerprint: `06f2b135d209af237e15dcb4f61321c1dcf0c392eef9ab07c7bc5a20f9bd7c29`.
Manifest SHA: `d366a5084d6fda9b38e3e19dd39b5e136bb3e18788f491b0bc4d5fbeb982435e`.

| Split | Positions | Boards | Crops |
|---|---:|---:|---:|
| Train | 12 | 112 | 7,168 |
| Validation | 3 | 25 | 1,600 |
| Test | 3 | 23 | 1,472 |

Validation positions: 003, 007, 011. Test positions: 006, 010, 016.
All remaining positions are training; orientation-template calibration position
001 is training-only. Every split contains all 13 classes. There are 180 source
photos; 20 were excluded from approved crops. Board-level approval does not mean
every crop has an individual visibility review; unreviewed visibility is retained.

Evaluation uses `config/settings.yaml`: CPU, four PyTorch threads, one OpenCV
thread, batch size 16, no shuffle, zero DataLoader workers, model eval/inference
mode. The classifier evaluator enables deterministic PyTorch algorithms.
The existing position evaluator does not explicitly enable that global setting;
repeatability is checked empirically. Threshold .70 affects overlay coloring,
not argmax classifications or evaluation denominators. FEN diagnostics use
explicit defaults `w - - 0 1`; these are not image observations.

Runtime: Python 3.12.3, Linux x86-64 / WSL2, PyTorch 2.14.1+cpu,
torchvision 0.29.1+cpu, OpenCV package 5.0.0.93, NumPy 2.5.3.
Exact installed distributions are in the JSON manifest and local `environment.txt`.
Pinned base/training requirements match the local environment. Optional EasyOCR
dependencies are unnecessary for these evaluators; phone performance and Stockfish
analysis are outside this baseline measurement.

## Preservation and reproduction

Historical tracked Day 12/14 reports, existing generated Week 2 reports, split
manifests, checkpoint, templates, config and preprocessing/label sources are
identified by hash in the manifest and copied under the local ignored directory
`outputs/evaluation/baseline_a1_20261018/snapshot/`. Images remain in their original
locations; `input_sha256.json` inventories 10,740 raw/crop/context/coverage inputs.
All were present, and all 10,240 stored crop hashes matched. No rebuilding,
resplitting, relabeling, training or calibration was performed.

The snapshot and image datasets are **local ignored assets**, not available from
a Git clone alone. Preserve/archive them with this manifest for another machine.
The existing manifests embed absolute `/home/apoa/repos/chess-vision/...` paths.
For byte-identical manifest/checkpoint compatibility, restore at that path or mount
there. Rewriting paths changes the manifest SHA and fails checkpoint provenance
checks. Do not regenerate manifests to bypass that failure.

From repository root, use the existing `.venv`, or install the pinned
`requirements.txt` and `requirements-training.txt` into Python 3.12. Restore
the ignored assets first. Check hashes before running:

```bash
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
m = json.loads(Path('configs/baselines/a1_20261018.json').read_text())
for name, expected in m['preserved_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('Preserved artifacts match')
PY
```

Choose a **new, unused** run directory for every invocation. The original
evaluators allow output reuse, so the `mkdir` guard below is essential. Always
pass `--no-metadata-report`: `--output-dir` alone still writes the historical
`data/metadata/day12_classifier_evaluation.json` or Day 14 report by default.

```bash
set -e
mkdir outputs/evaluation/a1_review_run
.venv/bin/python scripts/evaluate_classifier.py --no-metadata-report \
  --output-dir outputs/evaluation/a1_review_run/classifier
.venv/bin/python scripts/evaluate_positions.py --no-metadata-report \
  --output-dir outputs/evaluation/a1_review_run/positions
.venv/bin/python -m pytest tests/test_baseline.py tests/test_classifier_evaluation.py \
  tests/test_dataset_splits.py tests/test_inference.py \
  tests/test_position_evaluation.py tests/test_training.py tests/test_evaluation_reports.py -q
```

Classifier outputs include probabilities, per-class metrics, confusion matrices
and error galleries. Position outputs include per-image records, overlays,
failure reasons and failure-inclusive exact-board rates. Compare accuracy,
counts, confusion matrices, per-image status and placement exactly; allow absolute
`1e-7` for floating metrics/probabilities in this runtime. Timing and output paths
are intentionally excluded from deterministic comparisons. Timing is descriptive
CPU timing, not a phone benchmark. This tolerance is not a cross-platform guarantee.

To compare a new run against the preserved historical reports:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
def read(path):
    return json.loads(Path(path).read_text())
old = read('data/metadata/day12_classifier_evaluation.json')
new = read('outputs/evaluation/a1_review_run/classifier/evaluation.json')
assert old['metrics'] == new['metrics']
old = read('data/metadata/day14_position_evaluation.json')
new = read('outputs/evaluation/a1_review_run/positions/summary.json')
for mode, metrics in old['modes'].items():
    for key, value in metrics.items():
        if 'time' not in key:
            assert value == new['modes'][mode][key], (mode, key)
print('Historical non-timing metrics match exactly')
PY
```

## Reproduced results

Full numeric metrics and historical/reproduced position summaries are recorded in
[`baseline_results_a1_20261018.json`](baseline_results_a1_20261018.json).

| Metric | Historical | Reproduced |
|---|---:|---:|
| Classifier overall accuracy | 1,393/1,472 (94.6332%) | Same |
| Occupied-square accuracy | 377/456 (82.6754%) | Same |
| Macro F1 | 79.6760% | Same |
| Classifier errors | 79 | Same |
| Prepared completed / attempted | 23/23 | Same |
| Prepared exact boards | 0/23 | Same |
| Raw completed / attempted | 16/30 | Same |
| Raw square accuracy on completed outputs | 95.0195% | Same |
| Raw exact boards over all attempts | 0/30 | Same |
| Raw localization / orientation failures | 12 / 2 | Same |
| Chess-consistent prepared / raw completed | 4/23, 2/16 | Same |

Every classifier metric, including the confusion matrix and calibration bins,
matches the historical JSON exactly. Two full classifier runs had identical
metrics and **zero maximum probability difference** across all 1,472 crops.
All 53 prepared/raw position records match historical status, predicted placement,
error count, exact-match decision, failure stage and validation status. Position
non-timing summaries also match exactly. No accuracy discrepancy was found.

Timing differs materially: mean preprocessing + inference was 2,338 ms prepared
and 2,764 ms raw, versus historical 480 ms and 581 ms. Evaluations and tests shared
CPU resources; these runs do not establish an isolated latency regression or
performance equivalence. Hardware/runtime details are frozen for interpretation.

Runs used these exact commands (large outputs are ignored by Git):

```bash
.venv/bin/python scripts/evaluate_classifier.py --no-metadata-report --output-dir outputs/evaluation/baseline_a1_20261018/classifier_run1
.venv/bin/python scripts/evaluate_classifier.py --no-metadata-report --output-dir outputs/evaluation/baseline_a1_20261018/classifier_run2
.venv/bin/python scripts/evaluate_positions.py --no-metadata-report --output-dir outputs/evaluation/baseline_a1_20261018/positions_run1
.venv/bin/python -m pip freeze
.venv/bin/python -m pytest tests/test_baseline.py tests/test_classifier_evaluation.py tests/test_dataset_splits.py tests/test_inference.py tests/test_position_evaluation.py tests/test_training.py -q
.venv/bin/python -m pytest tests/test_evaluation_reports.py -q
.venv/bin/python -m pytest --ignore=tests/test_engine.py -q
.venv/bin/python -m pytest tests/test_vision_cli.py::test_cli_requires_orientation_when_markings_are_missing -q
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
git diff --check
```

Read-only inspection used `rg`, `cat`, `sed`, Git status/log/diff and Python
checkpoint/manifest inspection. A local Python audit copied 58 preserved artifacts
into the snapshot, hashed the 10,740 image inputs and recorded installed package
versions; a post-evaluation audit verified all 58 artifacts remained unchanged.
No dependency install, external download, commit or push was performed.

Validation outcomes:

- Focused baseline/inference/split/metric tests: **34 passed**.
- Report destination tests: **4 passed**, exercising both flag values in both
  evaluator entry points and preserving a sentinel historical report when requested.
- Final Python regression: **300 passed**, excluding `tests/test_engine.py` because
  Stockfish execution is outside A.1. Log: local `python_tests_final.log`.
- Browser regression: **39 passed** across the eight existing test entry points.
- `git diff --check`: passed.

The first broad Python run had 295 passes and one existing CLI subprocess timeout
at its 15-second limit while evaluation was also running. That test passed alone
in 12.08 seconds, and the final full regression passed with evaluations finished.
Its timeout and production code were not changed. Original and final logs remain
in the local baseline output directory.

## Acceptance review

- Baseline identifiable and reproducible: checkpoint loaded, input/class contract,
  code commit, config, dependencies, dataset identity and split assignment recorded;
  existing crop/prepared/raw evaluation reproduced.
- Historical metrics preserved: all 58 recorded artifact hashes still match,
  including the checkpoint, split manifests, scanning baseline and original reports.
- Unavailable assets explicitly reported: none required for this local evaluation
  were missing. Source-only clones lack ignored models and image datasets; exact
  restoration currently requires the original absolute manifest paths.
- No training or model replacement: satisfied. No recognition or browser workflow
  changes; only an optional report-writing switch was added to evaluation scripts.

Task A.1 stops here. Recommend A.2 after owner review: extend the existing evaluators
to stratify orientation and pipeline failures, preserving these inputs and keeping
the historical raw-photo path distinct from live browser acquisition. Do not infer
an orientation accuracy gap from these aggregate results or tune against this
already-inspected three-position test split.
