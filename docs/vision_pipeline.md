# Week 1 vision contracts

Day 2 provides working image I/O, visualization, and corner ordering. Day 4 adds
a classical contour-based detector with checkerboard candidate scoring.
Day 5 adds frame-color and long-line candidates, pixel-grid/border checks,
bounded corner refinement, and evaluation with version-bound visual reviews.
Day 6 adds perspective warping directly from original-image corners.
Day 7 adds 64-square extraction, indexed grid overlays, and full-pipeline CLIs.

## Images and paths

- `load_image(path)` returns a uint8 array shaped `(height, width, 3)` in BGR.
- `save_image(path, image)` accepts uint8 grayscale, BGR, or BGRA arrays,
  creates parent directories, and returns the written `Path`.
- Missing files, decoding failures, and failed writes raise useful errors.
- File arguments use the current working directory. Directory paths loaded
  from `config/settings.yaml` use the repository root.
- Visualization helpers return copies. Point/polygon colors use BGR;
  grayscale and BGRA inputs become BGR debug images.
- Annotate at the original resolution before resizing a preview. Preview
  resizing preserves aspect ratio and never enlarges the original image.

## Corners

Coordinates are `(x, y)` in original-image pixels: x increases to the right,
y increases downwards. The shared representation is a NumPy float32 array
with shape `(4, 2)` in this order:

| Index | Label | Image corner |
| --- | --- | --- |
| 0 | TL | Top-left |
| 1 | TR | Top-right |
| 2 | BR | Bottom-right |
| 3 | BL | Bottom-left |

`order_corners(points)` accepts four unordered points and returns their convex
boundary in clockwise image order. It anchors TL at the smallest x+y; ties
prefer smaller y, then smaller x. A diamond therefore starts at its top vertex.
Duplicate, collinear, concave, malformed, and nonfinite inputs fail explicitly.
This deterministic geometric convention cannot identify White's side or
resolve chess orientation; highly rotated boards can have ambiguous names.

`draw_board_corners(image, corners)` uses this ordering for the polygon, dots,
and TL/TR/BR/BL labels. It does not detect corners itself.

## Stage responsibilities

| Interface | Input | Output |
| --- | --- | --- |
| `detect_board_corners(image)` | Original BGR image | `BoardDetection`, with ordered corners on success or a reason on failure |
| `order_corners(points)` | Four boundary vertices | Ordered float32 `(4, 2)` corners |
| `warp_board(image, corners, output_size=800)` | Original image and ordered corners | Square, top-down board image |
| `extract_squares(board_image)` | Square normalized image, side divisible by 8 | 64 `SquareCrop` objects in row-major order |

`BoardDetection.success` is true when corners are present. Its optional `score`
is a diagnostic candidate score, not a probability. Failed detection must
not fabricate corners from image boundaries.

`BoardDetection.diagnostics` records contour/candidate counts and the ten
highest-scoring candidates with their geometry and checker evidence. Detector
options live in the `board_detection` section of `config/settings.yaml`.
For the Day 4 method, outputs, and limitations, see [day4_baseline.md](day4_baseline.md).
For the current detector and measured results, see [day5_evaluation.md](day5_evaluation.md).

The rectify CLI passes `board.normalized_size` from configuration to the warp
interface. Corners must be ordered, convex, finite, and inside the source image.
The transform maps them to `(0,0)` through `(size-1,size-1)` using one cubic
resampling operation. Resizing an image is not a substitute for perspective correction.

## Square indices and metadata

Rows increase from top to bottom (0–7). Columns increase from left to right
(0–7). Order is `r0_c0`, `r0_c1`, …, `r0_c7`, `r1_c0`, …, `r7_c7`.
Do not label a crop `a8` or another chess coordinate until orientation is known.

`SquareCrop` preserves `row`, `col`, `bounds`, and `image`. Bounds are
`(x0, y0, x1, y1)` in the normalized board, with exclusive right/bottom ends:
`board_image[y0:y1, x0:x1]`. Square size is derived from the board size,
rather than hard-coded as 100.

`SquareCrop.metadata()` omits the image array and returns JSON-compatible data.
For example, the bottom-right square of an 800×800 board has:

```json
{"row": 7, "col": 7, "bounds": [700, 700, 800, 800]}
```

The extraction CLI saves `r<row>_c<col>.png` crops
and a `squares.json` metadata file with the corresponding filenames.

Each crop owns a copy of its pixels; changing it cannot modify the board.
`draw_board_grid` overlays the same integer boundaries used for extraction.
Grid annotations never enter the normalized board or saved crops.

`process_board_image` in `src/vision/pipeline.py` runs detection once, warps
original pixels once, and saves detection diagnostics, a plain board, grid, and
64 crops. On detection failure its record has null board/crop paths and zero
squares. Existing outputs from previous runs are retained; consult the current
run's JSON rather than treating old files as successful current results.

`scripts/evaluate_week1.py` records detection correctness separately from warp
correctness and crop count. Warp reviews are bound to source/detector signatures
and actual normalized pixels; changed output becomes unreviewed. A successful
64-crop extraction does not imply correct localization. See
[day7_integration.md](day7_integration.md).

## Context outside the playing area

`warp_board_with_context` returns a `RectifiedBoard` with the extended image,
source-coverage mask, and explicit exclusive `board_bounds`. Its `board_image`
is the normalized playing area. `extract_context_squares` returns 64 overlapping
windows with masks, centered on the original grid squares. Their bounds refer
to the context canvas; saved metadata additionally locates each target square
inside its window. Never extract the 8×8 grid by dividing the entire context
canvas. See [board_context.md](board_context.md).
