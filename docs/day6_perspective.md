# Day 6: perspective correction

`warp_board` maps ordered TL/TR/BR/BL corners of the inner 8×8 playing area
onto a square, defaulting to 800×800 from configuration. OpenCV's
`getPerspectiveTransform` and `warpPerspective` sample the original photograph
once using cubic interpolation; detector preprocessing does not resize the
source supplied to the warp. Outputs use PNG to avoid additional JPEG loss.

```bash
python scripts/rectify_board.py data/raw/position_014_var_01.jpg
python scripts/rectify_development_subset.py
```

Single-image detection failures exit with a useful error and write no board.
`--corners` is an optional manual override for geometry experiments, not required
by the automatic pipeline. `--output` overrides the single-image destination;
`--output-dir` overrides the batch destination. The normalized size is configured
in `config/settings.yaml` and must be a positive integer divisible by eight.

## Development results

The fixed 20-image subset produced 16 images in `outputs/rectified_boards/`.
The other four failed localization and were not warped. `rectification.json`
records source filenames, ordered corners, output dimensions, failure reasons,
and signature-matched Day 5 localization reviews. Fifteen accepted detections
have approved localization; `position_001_var_07` is a known incorrect detection
and its warp includes the right wooden frame while losing a file on the left.
It is retained as a diagnostic example, not counted as a correct board.
The existing 75% localization result remains below the 90% target.

All 16 saved outputs were visually inspected in a contact sheet, with additional
full-resolution inspection of `position_013_var_03` and `position_004_var_04`.
For the approved detections, the board plane generally has straight horizontal
and vertical grid boundaries and recognizable pieces. Some outputs retain narrow
frame strips or slight edge offsets. This is a coarse visual check, not a measured
pixel-level alignment guarantee for all 64 future crops.

Perspective correction applies to the flat board. Tall pieces retain parallax:
in angled photographs they lean, stretch, overlap adjacent squares, and can
extend past the playing-area boundary and be cropped. An 800×800 image alone
does not guarantee correct localization or restore detail missing from a photo.
White/Black orientation is preserved from the photograph, not inferred.

Inspect outputs for eight rows and columns, straight grid boundaries, alignment
with image edges, missing squares, frame inclusion, and recognizable pieces.
Day 7 will add an indexed grid overlay and 64-square extraction.

## Tests

Tests exercise all permutations of corner ordering, malformed/degenerate corners,
out-of-bounds coordinates, invalid output sizes, identity transforms across
uint8 grayscale/BGR/BGRA, and restoration of all 64 centers of a synthetic
perspective-distorted checkerboard. CLI tests cover automatic detection, manual
overrides, execution from another directory, and failed detection without output.

```bash
python -m pytest tests/test_perspective.py tests/test_vision_cli.py tests/test_vision_contracts.py
```
