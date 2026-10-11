# Resolve orientation from printed coordinates

The pipeline now attempts orientation resolution after perspective correction
and before square extraction. It uses the printed A–H and 1–8 markings in the
preserved border, not the arrangement of pieces, FEN, or filename metadata.

## Implementation

`src/vision/orientation.py` extracts small border slots for all 32 coordinates,
normalizes light glyph silhouettes, and compares them with the 16 board-specific
templates in `models/board_markings.npz`. The reference image is
`position_001_var_08.jpg`; calibration provenance is saved alongside the bank.
The template bank is approximately 6 KB and requires no OCR service or new
Python dependencies. This is a board-specific classical recognizer, not a
trained neural piece-recognition model.

For each of four candidate rotations, it compares the decoded coordinates with
the known print layout of this physical board: bottom/top A–H left-to-right;
left/right ranks 8–1 top-to-bottom. Top/right characters face the opposite
player and are individually rotated 180 degrees before comparison.

The winner must have at least eight matching markings across two edges, at
least three distinct files and three distinct ranks, and a score advantage of
0.12 over the next rotation. Individual template similarity must reach 0.60.
These thresholds are configured under `orientation` in `config/settings.yaml`.
Scores are diagnostics, not calibrated probabilities. Both coordinate axes
must be established; symmetric glyphs alone cannot approve a mirrored input.

On success, the context image and coverage mask are rotated together, with
playing-area bounds transformed correctly. No additional resizing is involved.
Crops are then extracted in White-at-bottom orientation: r0/c0 = a8,
r7/c7 = h1. Both exact and context crop metadata record the coordinate system
and chess-square assignment. This does not classify pieces or generate FEN.

On uncertainty, there is no guessed rotation or chess-square mapping. The
image and its crops remain image-relative. An uncertain result includes the
reason, candidate scores, recognized markings, and rotation hypotheses in
pipeline JSON. Later dataset construction and FEN generation must require
resolved or explicitly provided orientation.

## Use

```bash
python scripts/process_board_image.py data/raw/position_004_var_04.jpg --require-orientation
python scripts/rectify_board.py data/raw/position_004_var_04.jpg
```

The full pipeline prints orientation status. `--require-orientation` exits with
status 1 if unresolved; diagnostic images and image-relative crops are still
saved for inspection. Without that flag, uncertainty prints a warning.

If markings are obscured, inspect the rectified view and explicitly provide
White's side **in that rectified view**:

```bash
python scripts/process_board_image.py data/raw/example.jpg --white-side right
```

Allowed sides are bottom, left, top, and right. This is recorded as `provided`
with source `manual`, rather than pretending it was visually inferred. Turns
are recorded as counterclockwise quarter turns: bottom 0, left 1, top 2, right 3.
The side means White's home rank, not the location of individual white pieces.

## Calibration and evaluation

Rebuild templates for this board from a clear photo with known orientation:

```bash
python scripts/calibrate_board_orientation.py data/raw/position_001_var_08.jpg --white-side bottom
python scripts/evaluate_orientation.py --all --report data/metadata/orientation_evaluation.json
```

All 16 bottom/left coordinates must be extractable for calibration. Visually
verify the reference's orientation and markings before using a new bank.
A different font or coordinate-print layout requires a different calibration
and potentially changes to the layout logic. The 180-photo audit compares
automatic decisions with capture orientation metadata and reports detection
failures, uncertain cases, and wrong resolved cases separately. The calibration
photo is excluded from headline rates. This is a development audit, not a
held-out piece-classification experiment. Detector acceptance does not establish
correct localization.

Tests cover all four rotations with an uninformative piece area, synchronized
mask/bounds rotation, missing or occluded markings, insufficient evidence,
mirrored-input abstention, and explicit orientation before crop extraction.

## Limits

This implementation supports rotations of the current marked board. Arbitrary
camera rotation is handled by the upstream corner geometry; four quarter turns
then establish chess coordinates. It does not guarantee a decision for every
photo. Blur, shadows, occlusion, small distant markings, or incorrect detected
corners can lead to uncertainty. Rotating a crop does not repair bad board
localization. Mirrored inputs and different border layouts are unsupported;
manual orientation assumes an ordinary, unmirrored photograph.

Already clipped board-only images lack the markings. Reprocess the original
photograph to obtain context, or provide orientation explicitly. No live video
tracking, piece model, or FEN logic is implemented here. Orientation can be
reused per video frame later; temporal stabilization remains future work.

Adding rotation changes normalized pixels for some photos, so historical warp
reviews must be refreshed. Existing pixel-bound signatures correctly mark
changed warps as stale. Rebuild any labeled crops after changing normalization.

## Current 180-photo development audit

The calibration photo is excluded, leaving 179 audited photos:

| Outcome | Count |
| --- | --- |
| Correctly resolved orientation | 115 |
| Wrong resolved orientation | 0 |
| Detected candidate, uncertain orientation | 16 |
| Board detection failed | 48 |

Resolved decisions matched capture orientations in 115/115 cases. Automatic
end-to-end orientation coverage was 115/179 (64.2%); these conditional accuracy
and coverage figures must be reported together. They do not establish that
every accepted board has precise crop alignment. The separate calibration
photo resolved correctly as well. The real position 004 variation 04 context
was also tested under all four quarter-turn rotations, with correct recovery
in each case. Synthetic camera-roll tests cover 17°, 43°, 95°, 187°, and 275°.

All 127 non-engine tests passed after integration. Detailed marking evidence
is saved under `outputs/week1/orientation_evaluation/`; the versionable summary
is `data/metadata/orientation_evaluation.json`. Improve detection and unreadable
marking cases before claiming orientation support for every photo.
