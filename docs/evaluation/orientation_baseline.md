# Task A.2 — Orientation-stratified baseline

Date: 2026-10-18. Baseline: `a1_20261018_15a5a879`, epoch-16 checkpoint.
The inference source commit remains `15a5a8792a41f1a11c9287f6061f90fb126858a3`.
No model, labels, splits, preprocessing, inference, or live browser behavior changed.

## Method and provenance

The new `scripts/evaluate_by_orientation.py` post-processes the verified A.1
prepared/raw position evaluation. It reuses `prediction_metrics`,
`summarize_positions`, canonical square mapping, confusion plotting and contact
sheets rather than rerunning inference or duplicating the recognition pipeline.
Capture metadata is joined **after** prediction and is never passed to the model.

Inputs are the A.1 `positions_run1/summary.json` and its 39 successful per-image
prediction files, plus the frozen split manifest, image metadata, FEN labels and
prepared-board geometry. The evaluator verifies the 58 frozen A.1 artifact hashes,
report checkpoint/dataset/ground-truth identity, exact prepared/raw attempt sets,
64 unique squares per successful board, class/probability consistency, semantic
mapping, and agreement with saved errors and aggregate metrics. It hashes its
source prediction JSON files and checks them and the frozen artifacts again after
export. Output directories must be new; historical reports are never written.

The compact versioned numeric report is
[`orientation_results_a2_20261018.json`](orientation_results_a2_20261018.json).
Full results and images are local ignored artifacts under
`outputs/evaluation/orientation_a2_20261018_run2/`:

- `summary.json`: all groups, metrics, calibration bins, per-class precision/recall,
  confusion matrices, pipeline gates, confounding cross-tabs and input hashes.
- `groups.csv`: orientation, position, environment, lighting, elevation and session
  groups, with explicit attempted/completed denominators.
- `per_class.csv`: every orientation/mode/class, including zero-support classes.
- `attempts.json`: every attempted photograph, its metadata, failure stage,
  recognition errors, orientation agreement and position-validation diagnostics.
- `{prepared,raw}/{orientation}/confusion_matrix.{csv,png}`: 13×13 matrices with
  truth on rows and predictions on columns. Zero matrices mean no evaluated squares.
- Matching `errors.json`, `crops/*.png`, and `errors_*.png`: **130 square errors**
  (79 prepared, 51 raw), with canonical square, true/predicted labels, confidence,
  provenance and green target polygons. All errors are exported across 12 contact
  sheets. Original context windows and black padding are preserved. A Black-near
  contact sheet was visually checked for readable labels and target overlays.

This is a stratification of the historical photo pipeline, which uses classical
border templates. It does not measure the newer browser learned-marking bootstrap,
live tracking, multi-view recognition, phone latency or scan completion.

## Prepared geometry results

All prepared boards have reviewed geometry and known orientation; their 100%
preparation rate is supplied by dataset selection, not automatic detector accuracy.

| Physical camera side | Boards | Square accuracy | Occupied-square accuracy | Exact boards |
|---|---:|---:|---:|---:|
| White-near | 12 | 94.66% (727/768) | 81.02% (175/216) | 0/12 |
| Black-near | 6 | 94.01% (361/384) | 83.09% (113/136) | 0/6 |
| White-left | 2 | 94.53% (121/128) | 80.56% (29/36) | 0/2 |
| White-right | 3 | 95.83% (184/192) | 88.24% (60/68) | 0/3 |
| All | 23 | 94.63% (1,393/1,472) | 82.68% (377/456) | 0/23 |

Compared with White-near, overall accuracy changes by −0.65 percentage points
Black-near, −0.13 White-left and +1.17 White-right. Occupied-square differences are
+2.07, −0.46 and +7.22 points, respectively. These small, unbalanced groups do
**not** demonstrate a consistent Black-side or sideways classifier deficit.
Every board nevertheless has classification errors; exact reconstruction remains
zero in all groups.

## Raw end-to-end results

| Physical camera side | Completed / attempted | Localization accepted | Localization failures | Orientation failures | Square accuracy* | Occupied accuracy* | Exact boards |
|---|---:|---:|---:|---:|---:|---:|---:|
| White-near | 12/18 (66.67%) | 12/18 | 6 | 0 | 94.66% | 81.02% | 0/18 |
| Black-near | 2/6 (33.33%) | 4/6 | 2 | 2 | 97.66% | 62.50% | 0/6 |
| White-left | 2/3 (66.67%) | 2/3 | 1 | 0 | 94.53% | 80.56% | 0/3 |
| White-right | 0/3 (0%) | 0/3 | 3 | 0 | N/A | N/A | 0/3 |
| All | 16/30 (53.33%) | 18/30 | 12 | 2 | 95.02% | 80.38% | 0/30 |

*Square and occupied-square metrics cover completed outputs only. Every failed
photo remains in completion and exact-board denominators. No successful predictions
means N/A, not 0% recognition accuracy. In CSV, these values are blank; in JSON,
they are null. All 30 attempts are accounted for.

The raw Black-near score covers **only two sparse boards from position 016**:
125/128 squares correct, but only 5/8 occupied squares correct. Its high overall
accuracy does not establish better recognition. The model has zero raw White-right
outputs to evaluate; all three attempts failed localization. Prepared White-right
recognition is measurable and is shown separately above.

Relative to White-near, completion is 33.33 percentage points lower Black-near,
unchanged White-left, and 66.67 points lower White-right. This locates a major
observed difference before classification, while sample selection prevents a
controlled causal orientation comparison.

## Pipeline and class diagnostics

The report separates localization rejection, unresolved orientation, grid rejection,
classification mismatches and position-validation flags. These are separate
observations, not mutually exclusive causes of a wrong board. For example, a board
may contain recognition errors and still pass chess consistency checks.

- Raw localization was accepted for 18/30 attempts; 16/30 resolved orientation
  and completed. There were no recorded grid or inference exceptions.
- All 23 prepared and 16 completed raw predictions agree with the capture metadata
  on orientation and canonical square assignment. This checks semantic mapping,
  not physical corner/polygon accuracy. Unresolved photos do not enter this
  successful-only agreement statistic.
- Every completed board has at least one square error. Chess consistency flags
  occur on 19/23 prepared and 14/16 raw outputs; 4 prepared and 2 raw wrong boards
  still pass those checks.
- Automatic gate acceptance is not independently reviewed localization accuracy.
  Actual boundary alignment and piece-base association remain geometry-audit work.

Class recall examples from prepared boards:

| Class | White-near | Black-near | White-left | White-right |
|---|---:|---:|---:|---:|
| Black rook | 7/18 (38.89%) | 6/10 (60%) | 2/3 (66.67%) | 3/5 (60%) |
| Black king | 5/12 (41.67%) | 4/6 (66.67%) | 1/2 (50%) | 2/3 (66.67%) |
| White knight | 8/12 (66.67%) | 6/8 (75%) | 1/2 (50%) | 3/4 (75%) |

Confusion matrices and galleries identify rook/king/queen and pawn/knight errors
across orientations. The numeric report includes all class supports, precision,
recall and F1, plus confidence bins and descriptive calibration error. Classes
with no support are not evidence of either success or failure: existing metric
helpers use zero for undefined class scores; fixed 13-class macro F1 therefore
needs care in sparse groups. For groups with no outputs at all, CSV class scores
are explicitly blank. No thresholds or calibration parameters were fitted.

## Confounding and missing metadata

Raw-photo orientation/environment counts are:

| Orientation | Living room | Office | Outdoors |
|---|---:|---:|---:|
| White-near | 9 | 0 | 9 |
| Black-near | 0 | 6 | 0 |
| White-left | 3 | 0 | 0 |
| White-right | 0 | 3 | 0 |

Black-near and White-right occur only in the office. White-left occurs only in
the living room; White-near occurs in the living room/outdoors. Lighting follows
environment: overhead indoors, even natural outdoors. Black-near elevations are
90°/60°, whereas left/right are 75°. Elevations are approximate capture-plan
values above the board plane, not independently measured poses.

No image has a session ID or timestamp. The evaluator explicitly groups all
photos under session `unknown`; it does not substitute variation or position IDs
for capture sessions. Session independence cannot be established. Position IDs,
environments, lighting and elevations are grouped individually in the full report,
but these strata cannot remove the missing combinations in the collection design.
The three test positions and repeated photos are correlated, and the raw completed
subset has a different piece distribution from the prepared set.

## Reproduce and verify

Run from repository root with the A.1 Python environment and local ignored assets.
This command uses the existing A.1 outputs and refuses an existing destination:

```bash
.venv/bin/python scripts/evaluate_by_orientation.py \
  --positions-report outputs/evaluation/baseline_a1_20261018/positions_run1/summary.json \
  --output-dir outputs/evaluation/orientation_a2_review_run
.venv/bin/python -m pytest tests/test_orientation_evaluation.py -q
.venv/bin/python -m pytest --ignore=tests/test_engine.py -q
git diff --check
```

The executed evaluation used `orientation_a2_20261018_run2` as its destination.
An initial development export reached the gallery stage but failed CSV serialization
for an all-failed group; the corrected exporter and CLI regression test preserve
that group's null metrics. Its incomplete `run1` directory has no summary report.
If A.1 outputs are absent, reproduce them first using
[`baseline_manifest.md`](baseline_manifest.md), always with a fresh output path
and `--no-metadata-report`. Then pass the newly generated `summary.json` here.
Source-only clones lack the ignored model/data/prediction artifacts; missing assets
cause a clear failure instead of invented results.

Validation: **19 focused tests passed**, covering four-side canonical mapping,
missing/duplicate/extra attempts, malformed predictions, failure-inclusive rates,
zero-output groups, CSV serialization, refusal to overwrite an output directory,
gallery labels/black padding and actual frozen sample counts. The broader Python
regression passed **318 tests**, excluding Stockfish; it collected before the last
gallery test was added, which passed in the focused run. Logs are saved beside the
full report. A final audit verified all 58 A.1 artifact hashes, all recorded source
prediction hashes, all 130 exported errors, and that each partition's attempt,
completion, exact-board and confusion-matrix totals reconcile. `git diff --check`
passed. No browser code changed, so browser tests were not repeated for A.2.

## Acceptance review and next step

- Orientation accuracy and exact-board rates are available for all four camera
  sides in both modes; failed-only recognition groups explicitly report N/A.
- Black-side/sideways completion and recognition gaps are quantified separately.
- Per-class metrics, confusion matrices and every misclassified square are exported.
- Counts reconcile to 23 prepared boards / 1,472 squares, 30 raw attempts /
  16 completed boards / 1,024 squares, with all 14 raw failures retained.
- Environment/lighting/elevation confounding and unavailable sessions are explicit.
- Frozen A.1 artifacts and live functionality remain unchanged.

A.2 stops here. Recommend A.3 next: define a balanced independent capture protocol
with real session IDs and each orientation represented within environments and
elevations. The current data supports prioritizing localization/orientation audit,
but does not justify a model replacement or a general Black-side classifier claim.
No new holdout, training, calibration, YOLO, browser change, commit or push occurred.
