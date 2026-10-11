# Learned board-coordinate reader

Automatic still-frame alignment now uses a separate compact neural reader for this
wooden board's printed **A–H and 1–8**, plus a reject class. The piece classifier
and its preprocessing/checkpoint remain unchanged. Restart the Python HTTPS server
and reload the phone page to activate the new reader. Freeze a sharp full-board
view and choose **Find board automatically**. A successful learned result reports
how many spatially consistent markings supported it; inspect the proposed grid.

## What changed

The original orientation module searched fixed slots around a candidate grid and
matched glyphs to templates. A displaced grid could miss every marking. The new
path generates candidate character components across the original image using
local contrast at several scales, then classifies each candidate with the trained
reader in four rotations. **Component proposal generation is classical image
processing; the character reader is learned.** It does not assume that each
predicted square already has a correctly positioned border slot.

Recognized identities and pixel locations are matched to this board's coordinate
layout. Registration tests semantic rotations and nearby offsets, robustly fits a
board-plane homography, and requires at least eight distinct consistent observations
covering three different files and three different ranks. Rotated guesses from the
same physical glyph cannot count as independent observations. Competing mappings,
mirrored/folded geometry, missing coverage and conflicts with template evidence
are rejected. Current paint centers are modeled two centimeters outside the
playing-area edge; grid refinement then uses actual checker transitions. This
board-specific offset remains approximate, not lens/physical calibration.

The existing playing-area detector provides an initial geometric proposal. A
coarse wooden-border proposal can support OCR registration when it fails, but is
never accepted by color alone. If learned registration is uncertain/unavailable,
the original marking/template path remains a fallback. Both paths preserve the
source image and its borders, return 64 source-coordinate square polygons and
require review. They do not identify pieces or follow camera movement yet.

## Pretrained benchmark and training

Pretrained [EasyOCR](https://github.com/JaidedAI/EasyOCR) CRAFT detection plus English
recognition was benchmarked first. On the aligned phone reference it took about
14 seconds for the read operation and missed most file letters, including merging
some neighboring letters. It did not supply enough file evidence for registration.
The optional benchmark dependencies and official weights were installed under
ignored `tools/ocr/`, leaving the existing Python environment's packages unchanged.

We instead trained a small three-convolution-layer character model with:

- 6,772 examples: 5,558 training and 1,214 validation.
- Positives isolated from character slots in user-reviewed, hash-bound board geometry.
- 269 ambiguous glyphs removed using expected-label/template agreement (NCC ≥ .65).
- Reject examples from playing-area components, including piece/wood edges.
- Rotation, anisotropic scale, shear, translation, noise and blur augmentation.
- Validation grouped by positions 003, 007 and 011; positions 006, 010 and 016 excluded.
- No FEN input or phone scan frames used for training.

The label-example montage was visually inspected; this is not a claim that every
character was individually annotated by a human. The validation crops use the
same quality filter, so they favor readable characters and do not measure raw
image detection recall or robustness to unreadable markings.

Training ran for 25 epochs; epoch 22 was selected by validation macro recall.
Its filtered-crop accuracy was **97.20%**, with **99.39% macro recall** including
reject. These numbers are character-crop metrics, not board accuracy, orientation
accuracy or a probability of correct scan completion. The model is specific to
this board's font and light-colored printing.

Checkpoint: `models/checkpoints/board_marking_reader.pt` (ignored, available locally).
Training examples/history: `outputs/scanning/marking_training/` (ignored).
Tracked summary: [learned_markings_evaluation.json](../data/metadata/learned_markings_evaluation.json).

## End-to-end development results

On the same 11 unique sampled phone images, accepted automatic **candidates increased
from 2 to 4**. The newly accepted grids were visually checked; seven images still
remain uncertain or fail coverage/geometry checks. This is a small development
comparison from two scans of the same previously seen position, not an independent
accuracy benchmark. Repeated tuning on these examples makes new scans necessary
before general reliability claims.

Four older photos of validation position 003 were also checked from White-bottom,
White-left, Black-bottom and White-right viewpoints. Three returned the expected
orientation; the White-right example remained uncertain. The coordinate mapping
itself is tested in all four orientations, but learned recognition from every side
has not been established. Smaller/blurred characters, low angles, shadows and
clipped borders remain weaknesses. Class probabilities and rejection thresholds
are engineering settings, not calibrated confidence estimates.

The result is useful progress toward automatic bootstrap; it does not complete
Step 04's full all-side acceptance gate. Keep the manual fallback and collect new
clean views across all sides while developing local tracking (Step 05), rather than
rerunning slow still-frame OCR on every video frame. A frozen analyzed grid must
never be treated as tracked geometry after the camera moves.

## Reproduce

```bash
# Dataset review and training; requires the existing optional torch environment.
.venv/bin/python scripts/train_marking_reader.py --prepare-only
# Inspect outputs/scanning/marking_training/label_examples.jpg.
.venv/bin/python scripts/train_marking_reader.py --from-prepared

# Evaluation without passing FEN or manual geometry to automatic inference.
.venv/bin/python scripts/bootstrap_scan.py outputs/scanning/captures/YOUR_SCAN.zip \
  --output-dir outputs/scanning/bootstrap_learned/YOUR_SCAN --stride 6
```

`config/settings.yaml` controls `learned_markings.enabled` and the character-score
threshold. The learned path needs torch and the marking checkpoint; its default
runtime does **not** need EasyOCR. Missing reader weights/dependencies are recorded
explicitly and fall back to templates. Inference never downloads weights.

Optional pretrained OCR reproduction:

```bash
.venv/bin/python -m pip install --target tools/ocr/python --no-deps -r requirements-ocr.txt
.venv/bin/python scripts/setup_marking_ocr.py
.venv/bin/python scripts/benchmark_marking_ocr.py outputs/scanning/captures/YOUR_SCAN.zip \
  --frame 30 --output outputs/scanning/ocr/pretrained.json
```

Restart the server after retraining, as the reader is cached in memory. Model and
source hashes are retained in reports so comparisons can identify the weights used.
