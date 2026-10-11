# Day 13: inference and board reconstruction

`PieceClassifier` loads the epoch-16 MobileNetV3 Small checkpoint once and runs
batched inference over 64 polygon-based context crops. Training, evaluation and
inference share `preprocess_context_crop`: BGR source pixels become ImageNet-normalized
RGB plus a binary target-square mask at 224×224. No ground-truth classes, FEN,
rank/file coordinates or game-state fields enter the model.

## Run on a raw photograph

```bash
.venv/bin/python scripts/predict_position.py data/raw/position_016_var_02.jpg
```

The command reuses Week 1 detection and context rectification, resolves orientation
from border markings, fits image-supported square polygons and extracts the same
context/mask inputs as the dataset. Input pixels remain in camera view. Only the
predicted board and confidence matrices rotate to White-at-bottom coordinates.
This preserves upright appearance relative to the camera and handles all four
White-side orientations without forcing pixels through another warp.

If the markings are unreadable, inspect the saved context image and explicitly
supply its White side:

```bash
.venv/bin/python scripts/predict_position.py data/raw/example.jpg --white-side top
```

If automatic localization fails, `--corners corners.json` accepts four playing-area
vertices in original-photo coordinates, using the existing corner-ordering and
rectification code. JSON format: `{"corners": [[x,y], [x,y], [x,y], [x,y]]}`.
The corners describe the 32×32 cm playing area, not the outer border. Overrides
are reported as supplied input, not automatic success. No capture metadata or
stored corner overrides are automatically looked up by filename.

Unresolved orientation, rejected grid fits and target squares with less than 99%
source coverage fail explicitly. A fitted grid can still be inaccurate; inspect
the overlay before trusting a new photograph. The classical detector and marking
matcher remain limitations, independent of piece classification.

## Run on an existing rectified context board

```bash
.venv/bin/python scripts/predict_position.py \
  data/classification/boards/position_010_var_06_context.png \
  --mode rectified --geometry data/classification/prepared_boards.json
```

The prepared manifest supplies only geometry, orientation and image/mask hashes;
its ground-truth FEN and labels are never used. This mode retains the reviewed
polygon grid and the image borders. A tight 8×8 image without context cannot
recover clipped piece bodies, so rectified input requires a geometry description.

A single geometry JSON can instead contain `board_bounds` (exclusive right/bottom),
`mask_path`, `rotation_quarters_ccw` (0..3 to White-bottom), `orientation_source`,
and optional `grid_geometry`. Without polygons, the grid is fitted from the image;
without orientation, border recognition runs or `--white-side` is required.
The current input format assumes a square playing area on the camera-view context
canvas. Saved geometry includes hashes and can be reused:

```bash
.venv/bin/python scripts/predict_position.py \
  outputs/week2/predictions/position_016_var_02/context.png \
  --mode rectified \
  --geometry outputs/week2/predictions/position_016_var_02/geometry.json
```

## Outputs

Under `outputs/week2/predictions/<input-stem>/`:

- `predictions.json`: all 64 class predictions, symbols, probabilities, confidence,
  camera row/column, chess square, square polygon and context/coverage information;
  camera-view and White-bottom board matrices; both confidence matrices; checkpoint
  provenance and orientation source.
- `prediction_overlay.png`: full context canvas with square outlines, chess-square
  coordinates, piece symbols and confidence. Uppercase symbols are White pieces;
  lowercase are Black; `.` means empty. Orange outlines flag confidence below the
  configured threshold. Green indicates higher confidence, not verified correctness.
- `context.png`, `context_valid.png`, `geometry.json`: reusable unannotated pixels,
  source coverage and geometry.
- `status.json`: explicit current-run success/failure; older predictions in a reused
  directory must not be treated as successful output when this status says failed.
- `detection.png` and detector debug artifacts for automatic raw input.

The terminal prints a complete White-bottom 8×8 matrix, with ranks 8 down to 1.
Confidence remains descriptive; the Day 12 test found confident incorrect predictions.
No legal-move correction is performed. FEN generation and chess diagnostics were
subsequently added on [Day 14](day14_fen_evaluation.md).

## Verification and limitations

Four prepared held-out boards covering rotations 0, 1, 2 and 3 produced all 64
predictions and probability vectors exactly matching Day 12 evaluation (maximum
probability difference 0). `position_016_var_02.jpg` also completed raw localization,
automatic border orientation and 64-square inference. This confirms integration on
an existing held-out photograph, not generalization to a newly captured photo.
The compact record is `data/metadata/day13_inference_verification.json`.

The model remains unchanged. Its known errors appear in reconstructed boards:
for example, the a1 black king in position 016 variation 02 is predicted as a queen.
Inference completeness is not exact board correctness. Day 14 measures full-position
performance and implements FEN plus validity diagnostics.

```bash
.venv/bin/python -m pytest tests/test_inference.py tests/test_classification_dataset.py \
  tests/test_dataset_splits.py tests/test_classifier.py tests/test_classifier_evaluation.py -q
```

Tests cover preprocessing consistency, binary target masks, batching, all four
rank/file rotations, confidence alignment, duplicate/missing squares, invalid
polygons, clipped source coverage, unresolved orientation and no FEN dependence.

No new input is needed to complete Day 13 or to begin Day 14's existing test-set
checks. A completely new photograph with its expected position would be useful
for a later independent generalization check; side-to-move and any other full-FEN
fields will need explicit input or clearly documented defaults on Day 14.
