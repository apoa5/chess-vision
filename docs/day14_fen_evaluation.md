# Day 14: FEN, position diagnostics and complete-position evaluation

The single-photo pipeline now converts a resolved board to piece-placement FEN,
optionally appends supplied/default game-state fields and reports python-chess
consistency diagnostics. The held-out evaluation measures both reviewed-board
recognition and fully automatic raw-photo processing. The Day 14 deliverables
are implemented, but the measured baseline does not yet reconstruct complete
positions reliably: no held-out board was an exact match.

## Predict a photograph

```bash
.venv/bin/python scripts/predict_position.py data/raw/position_016_var_02.jpg \
  --side-to-move b
```

The command prints the board, piece-placement FEN, full FEN, diagnostics and output
paths. `predictions.json` preserves all probabilities, confidences, square geometry,
orientation provenance and the newly generated FEN/diagnostics.

The image supplies piece placement only. Full FEN fields have explicit provenance:

| Field | CLI option | Default |
| --- | --- | --- |
| Side to move | `--side-to-move w` or `b` | `w` |
| Castling rights | `--castling KQkq` or an ordered subset | `-` |
| En passant | `--en-passant e3` | `-` |
| Halfmove clock | `--halfmove-clock 0` | `0` |
| Fullmove number | `--fullmove-number 1` | `1` |

These defaults do not claim knowledge of game history. Every output field is
marked `supplied` or `default`; `game_state_inferred_from_image` is false. Placement
serialization requires a White-at-bottom BoardState; unresolved orientation cannot
silently become FEN. Empty runs are compressed manually and ranks serialized 8→1.

`src/chess/validation.py` checks the parseable full FEN with python-chess and reports
missing/duplicate kings, excess pawns/material, backrank pawns, inconsistent castling
or en passant, and other standard-chess status flags. Structural shape/symbol
validation comes from BoardState. Invalid composed positions are retained with
diagnostics. No piece is changed to satisfy chess rules. Turn-dependent diagnostics
use the supplied/default turn. Passing these checks does not establish recognition
accuracy or prove historical reachability.

## Evaluation results

```bash
.venv/bin/python scripts/evaluate_positions.py
```

The saved epoch-16 checkpoint is loaded once and unchanged. Test positions are
006, 010 and 016. Prepared evaluation uses the 23 boards admitted to the existing
test crop dataset, with reviewed geometry and known orientation. Raw evaluation
attempts all 30 photographs of those positions, including photos excluded during
dataset preparation. It uses automatic detection and border orientation without
stored corner overrides, filename orientation, ground-truth FEN or manual fallbacks.
Ground truth is read only for comparison after inference.

| Result | Prepared boards | Automatic raw photos |
| --- | ---: | ---: |
| Attempted | 23 | 30 |
| Completed predictions | 23 | 16 |
| Failed before position output | 0 | 14 |
| Square accuracy on completed outputs | 94.63% | 95.02% |
| Exact board / placement-FEN matches | 0/23 | 0/30 |
| Standard-chess-consistent outputs | 4/23 | 2/16 |
| Mean confidence on completed outputs | 97.26% | 97.69% |
| Mean preprocessing + model inference | 480 ms | 581 ms |

The raw success rate is 53.33%. Its failures comprise 12 localization failures and
two uncertain orientations. Failed photos count against exact-board accuracy over
all attempts, while square accuracy has only completed outputs in its denominator.
The higher raw square score does not indicate a better model: this path evaluates
a smaller, easier subset and independently fitted geometry.

Prepared boards contain 2–7 square errors each. Raw completed outputs contain 1–6.
Exact-board and exact placement-FEN comparisons are equivalent once orientation
is resolved; full game-state FEN equality is not measured because these fields
are not recognized visually. Reported `piece_errors` counts every mismatched
square, including false empty/occupied predictions if they occur.

The example command produces placement `8/8/1KP5/3p4/8/8/8/q7` and full FEN
`8/8/1KP5/3p4/8/8/8/q7 b - - 0 1`. These are model outputs, not corrected ground
truth: the a1 king is predicted as a queen and the d5 rook as a pawn. Diagnostics
identify the missing black king. Several other wrong outputs pass standard-chess
checks, demonstrating that validity is insufficient to establish the correct board.

Timing is local CPU measurement, not phone performance. Model timing includes
crop tensor preprocessing and batch inference, excluding checkpoint loading and
raw localization. Mean complete processing time for successful cases, including
preparation and artifact writing, was about 718 ms prepared and 2,537 ms raw.
The run shared CPU resources with checks, so these are descriptive timings.

## Saved artifacts and reproducibility

- `outputs/evaluation/week2/week2_positions.csv`: mode, image/position ID, expected
  and predicted placement, square accuracy, errors, exact match, mean confidence,
  timings, validation status and failure reason.
- `outputs/evaluation/week2/summary.json`: aggregates, provenance and every result.
- Per-image directories under `prepared/` and `raw/`: predictions, overlays and
  evaluation records; raw automatic detector debug artifacts and context images.
- `data/metadata/day14_position_evaluation.json`: tracked report with checkpoint,
  dataset and reference-metadata hashes, default game-state values and limitations.

The checkpoint/manifest version is verified. Prepared image/mask hashes are checked;
ground-truth consistency with prepared metadata is verified. Evaluation does not
train, calibrate, modify labels or select a model using test results.

```bash
.venv/bin/python -m pytest --ignore=tests/test_engine.py -q
```

The regression suite passed 224 tests. New tests cover FEN round trips, empty runs,
orientation requirements, supplied/default fields, composed-position diagnostics,
one-error exact-match failure and failure-inclusive denominators. Engine tests are
separate; this change does not modify engine integration.

## Current limits and next phase

Week 2's software path exists from photograph to FEN with confidence and diagnostics.
Its broader reliability goal is not met by this baseline. Priorities are automatic
board geometry/orientation and repeated piece-recognition confusions documented
on Day 12. Do not interpret the 94.63% square score as reliable complete positions.
The three-position holdout has already been inspected; later test-informed changes
need newly reserved positions for a fresh final evaluation.

No new photo or labels were required for these deliverables. The next agreed phase
is [guided scanning and AR](guided_scan_roadmap.md): accumulate different views of
a stationary board, resolve disagreements and then anchor move overlays. Before
that evaluation, reserve completely new positions/captures and collect continuous
scan sequences with known piece placement. We have not implemented those future
features in Day 14.
