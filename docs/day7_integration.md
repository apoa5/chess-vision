# Day 7: square extraction and Week 1 integration

The full automatic pipeline is now photo → ordered playing-area corners →
800×800 board → indexed grid preview → 64 crops. No corner editing is required.

```bash
source .venv/bin/activate
python scripts/process_board_image.py data/raw/position_014_var_01.jpg
python scripts/evaluate_week1.py --report data/metadata/week1_evaluation.json
```

The single-image outputs use the configured `board_detection`, `rectified_boards`,
and `square_crops` directories. Use `--output-dir PATH` for a separate artifact
root. Batch artifacts and JSON/CSV reports default to `outputs/week1/`.
The `--report` option also saves a versionable JSON snapshot and companion CSV.
Paths inside the repository are recorded relative to the repository root.

## Square extraction

`extract_squares` validates a square uint8 image whose side is divisible by eight.
It derives the tile size from that side, returning exactly 64 `SquareCrop` objects
in row-major order. For 800×800, each crop is 100×100. Every pixel belongs to
one crop, with no overlap or omitted final row/column. Crops own independent
copies. `squares.json` records row, column, exclusive bounds, and filename.

`r0_c0.png` refers to the top-left image square; `r7_c7.png` to the bottom-right.
Chess-coordinate mapping is unresolved. `draw_board_grid` uses those exact
boundaries and overlays row/column labels on a copy. Crop pixels come from the
plain normalized board, never from the annotated grid.

To extract a previously normalized image independently:

```bash
python scripts/extract_squares.py outputs/rectified_boards/position_014_var_01.png
```

## Evaluation and visual checks

The fixed `week1_development_v1` subset contains 20 photos. Results:

| Measure | Result |
| --- | --- |
| Accepted board detections | 16/20 |
| Day 5 reviewed correct localizations | 15/20 (75%) |
| Coarsely approved grid/warps | 15/20 |
| Boards producing exactly 64 crops | 16/20 |
| Saved square crops | 1,024 |
| Localization target of 90% | Unmet |

All 16 grid previews were visually reviewed in a contact sheet, with additional
full-resolution review of `position_004_var_04`, `position_007_var_04`, and the
known incorrect `position_001_var_07`. Coarse approval requires eight rows and
columns, approximately straight grid boundaries, no missing file/rank, and no
large frame inclusion. Small offsets and narrow frame strips are allowed;
this is not a pixel-accurate alignment measurement or a piece-recognition test.
In particular, `position_004_var_04` retains a thin bottom strip, and
`position_007_var_04` has more visible drift toward the right edge. Tall-piece
parallax remains visible in angled photographs.

The four rejected images are `position_001_var_05`, `position_005_var_03`,
`position_006_var_01`, and `position_009_var_03`. They save detection diagnostics
and have zero crops. `position_001_var_07` produces 64 crops but is incorrect:
the playing area loses the left file and includes the right wooden frame.
It remains a diagnostic failure, not a successful Week 1 result.

Every saved crop across all 16 boards was decoded and compared exactly with
its recorded rectangle in the plain normalized board. All 1,024 matched, all
were 100×100, and each board had all 64 distinct row/column pairs.

Localization labels reuse signature-matched Day 5 reviews. Separate Day 7
warp reviews in `data/metadata/day7_review.json` bind to the source/detector
signature and actual normalized pixels. A changed warp is marked unreviewed
until inspected again. Generating 64 crops never establishes correctness.
Failures retain old artifacts from prior runs; current-run JSON paths/counts
identify what actually succeeded.

## Test it yourself

Open the corner preview first, confirming the outline follows the inner playing
area and excludes the frame. Open the `_grid.png` preview and check all eight
rows and columns, straight lines, edge coverage, and alignment with physical
square boundaries. Finally inspect several crops, including `r0_c0.png` and
`r7_c7.png`, and compare their bounds in `squares.json`.

```bash
python -m pytest --ignore=tests/test_engine.py
```

The 96 passing tests cover configuration, data validation, detection/review
contracts, transforms, exact extraction coverage, non-destructive overlays,
CLI execution, and pipeline success/failure behavior. Stockfish tests were not
rerun for these vision changes; its integration was verified earlier.

## Remaining Week 1 work

Day 7's implementation and evaluation tasks are complete. The Week 1 reliability
definition of done remains open because localization is 75%, below 90%, and
precise crop alignment needs further measurement. Improve the documented
occlusion and boundary-selection failures before relying on every photograph.
The dataset remains an indoor starter collection; broader capture coverage and
outdoor evaluation are still pending. There is no piece classifier, automatic
FEN generation, orientation inference, or video pipeline.
