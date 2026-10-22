# Implementation progress

## A.1 — Freeze and reproduce the baseline

- Date: 2026-10-18.
- Source commit: `15a5a8792a41f1a11c9287f6061f90fb126858a3`; no new commit made.
- Hypothesis: existing epoch-16 model and frozen inputs reproduce Week 2 results.
- Changes: `scripts/evaluate_classifier.py` and `scripts/evaluate_positions.py`
  add optional `--no-metadata-report`; default behavior is unchanged.
  Added `configs/baselines/a1_20261018.json`,
  `docs/evaluation/baseline_manifest.md`,
  `docs/evaluation/baseline_results_a1_20261018.json`,
  `tests/test_baseline.py`, `tests/test_evaluation_reports.py`, and this log.
- Implementation: reused existing evaluators; preserved 58 artifacts in a separate
  local snapshot and verified 10,740 image inputs. All required local assets exist.
- Before/after metrics: unchanged, 94.6332% overall, 82.6754% occupied-square,
  79.6760% macro F1; 0/23 exact prepared boards, 16/30 completed raw photos and
  0/30 exact raw boards. Full classifier repeats have identical probabilities.
- Validation: 34 focused tests, four report-destination tests, final 300-test Python
  regression excluding Stockfish, and 39 browser tests pass. Initial CLI timeout
  under concurrent evaluation passed in isolation and in the final full regression.
  `git diff --check` passes; all 58 preservation hashes still match. See the baseline
  report for commands and artifacts.
- Limits: ignored models/images require separate archival; absolute manifest paths
  restrict relocation; timings increased under shared CPU load; no phone benchmark
  or new independent holdout. Orientation-stratified evaluation is not part of A.1.
- Decision: A.1 implementation complete, awaiting owner review/acceptance.
- Next approved task: none. A.2 is recommended only after review.

## A.2 — Orientation-stratified evaluation

- Date: 2026-10-18. Owner authorized A.2 in the follow-up to A.1.
- Source commit: `15a5a8792a41f1a11c9287f6061f90fb126858a3`; changes remain uncommitted.
- Files added: `scripts/evaluate_by_orientation.py`,
  `src/vision/orientation_evaluation.py`, `tests/test_orientation_evaluation.py`,
  `docs/evaluation/orientation_baseline.md`,
  `docs/evaluation/orientation_results_a2_20261018.json`. Updated this log.
- Hypothesis: quantify whether orientation gaps arise before recognition or in
  identity classification, preserving the exact measured baseline.
- Implementation: stratified verified A.1 saved predictions using existing metrics,
  canonical mapping and visualization helpers. Includes all attempts, stage gates,
  class confusion/precision/recall/calibration, metadata partitions and 130 labeled
  error examples. No inference or training rerun was needed.
- Before/after aggregate metrics: unchanged; 94.6332% prepared square accuracy,
  82.6754% occupied accuracy, 0/23 prepared and 0/30 raw exact positions.
- Findings: raw completion White-near 12/18, Black-near 2/6, White-left 2/3,
  White-right 0/3. Prepared occupied accuracy 81.02%, 83.09%, 80.56%, 88.24%,
  respectively. These do not demonstrate a general Black-side classifier deficit.
- Validation: 19 focused tests passed; 318 broader regression tests passed before
  the final gallery test was added (also covered by the focused run). All 58 A.1
  preservation hashes and all grouping/confusion totals verified; diff check clean.
- Limits: no session IDs; orientation/environment/lighting/elevation confounding;
  raw Black-near completed outputs contain only eight occupied squares. Gate
  acceptance does not establish physical geometry accuracy. Browser bootstrap and
  phone performance are not measured by the historical raw-photo workflow.
- Artifacts: `outputs/evaluation/orientation_a2_20261018_run2/` (local ignored).
- Decision: A.2 acceptance criteria satisfied; owner approved before assigning A.3.
- Next approved task: none. Recommend A.3 for a balanced independent capture
  protocol; do not start it without authorization.

## A.3 — Initial capture proposal (superseded by revision 2 below)

- Date: 2026-10-18. Source commit remains
  `15a5a8792a41f1a11c9287f6061f90fb126858a3`; no commit or push performed.
- Files added: `docs/evaluation/capture_protocol.md`,
  `docs/evaluation/capture_protocol_validation_a3_20261018.json`,
  `configs/capture/capture_manifest.schema.json`, `configs/capture/a3_plan.json`,
  `src/scanning/capture_protocol.py`, `scripts/validate_capture_manifest.py`,
  `tests/test_capture_protocol.py`. Updated this progress log.
- Hypothesis: crossing all viewpoints within environments/lights/elevations and
  separating reference geometry, semantic orientation and placement will support
  causal diagnosis better than the confounded historical collection.
- Implementation: 48-image engineering pilot, then 576 main images over 12 new
  positions (8 train / 2 val / 2 final), 52 planned ZIP sessions total. Companion
  schema reuses original recorder archives/replay; image-bound human attestations,
  split/day/group/placement/hash isolation, coverage checks and purpose gates.
- Metrics before/after: A.1/A.2 results unchanged. Zero new camera sessions, frames
  or verified labels. Empty-plan acceptance is not physical capture acceptance.
- Validation: 53 focused/replay/split tests and 353 broader tests passed (the focused
  run includes one test added after broad collection). Expected incomplete/training/
  sealed-holdout rejections confirmed. Diff check and frozen provenance audit pass.
- Limits: actual captures/reviews still require the owner; software cannot certify
  truthful annotations, detect all near-duplicates, or replace holdout access control.
  Replay retains its existing wooden-board profile; no legacy-loader adapter added.
- Decision: superseded by the owner's continuous, video-first collection revision below.
- Next approved task: none. Recommend the specific B.1 localization, orientation
  and target-polygon investigations in the capture protocol; none was implemented.

## A.3 revision 2 — Continuous multi-angle sequence collection

- Date: 2026-10-18. Owner requested a video-first revision of A.3 before acceptance.
- Primary unit: original uninterrupted camera sequence of one physical position;
  sparse high-quality image derivatives serve the existing MobileNetV3 model.
- Revised schema/plan to `capture_protocol_v2`: original timestamps, full view
  intervals (unknown during transitions), sequence-level human review, locked frame
  caps/spacing and near-duplicate groups. Added separate full-scan validation/final
  purpose gates; no final data can be used for model selection.
- Added `src/scanning/frame_selection.py`, `scripts/extract_scan_frames.py` and
  `tests/test_frame_selection.py`. Updated existing A.3 schema, plan, validator,
  capture-protocol module/tests and guide. Current validation record:
  `docs/evaluation/capture_protocol_validation_a3_video_20261018.json`.
- Plan: four continuous pilot recordings and 48 main recordings, 12 new main
  positions (8/2/2), crossing rooms/lights/sides/elevations. At most 12 reviewed
  recognition frames per recording; up to 624 coverage targets including pilot.
  Whole original timelines, including unsuitable stills, remain scan evaluation data.
- Infrastructure: existing approximately 1 Hz / 60-frame guided JPEG recorder and
  replay reused without live changes. Native full-rate phone videos should be
  preserved separately; native codec/PTS import and high-rate tracking evaluation
  are not implemented or claimed by these ZIP tools.
- Validation: 70 focused and 371 broader Python tests pass, excluding Stockfish;
  diff check passes. Historical 30-frame camera ZIP yields 12 unverified candidate
  JPEGs with exact bytes/timestamps and no source changes. No new capture/labels.
- A.1/A.2 preservation hashes still match; recognition architecture, model,
  preprocessing, live scanning and dataset splits remain unchanged. No training,
  commit or push performed.
- Decision: revised A.3 protocol/tooling complete, awaiting owner review. No next
  task authorized. B.1 recommendations retained; physical pilot remains an owner task.

## B.1 — Board geometry and square alignment audit

- Date: 2026-10-18. Owner approved moving to B.1 conditional on A.3 completion.
  A.3 protocol/tooling acceptance and recorded checks are complete; actual physical
  pilot capture remains an explicit owner task. Its artifacts were preserved.
- Report: `docs/evaluation/geometry_alignment_audit.md`; numeric evidence:
  `docs/evaluation/geometry_results_b1_20261018.json`; five representative images
  in `docs/evaluation/geometry_examples_b1/`. Full local diagnostics and predictions:
  `outputs/evaluation/geometry_b1_20261018_run1/`.
- All 14 A.2 failed raw-image gates reproduce: 12 localization, two orientation.
  All three White-right failures have wrong leading rejected polygons. One
  Black-near orientation failure follows accepted visibly inaccurate geometry;
  forcing semantic orientation alone then fails grid fitting.
- Seven historical reviewed references support controlled interventions. Corners
  alone restore 6/7; corners plus known orientation restore 7/7. Those results have
  27/448 square errors and 0/7 exact boards, matching saved prepared error counts.
  The other seven failed images lack reviewed corners; no new ground truth invented.
- All 1,472 prepared crop pixels/preprocessed inputs/target masks agree exactly.
  Four-side synthetic and reviewed physical-coordinate checks pass. No confirmed
  arithmetic mapping defect; no production algorithm or runtime behavior changed.
- Added read-only geometry audit module/CLI and 14 focused tests. Validation:
  45 geometry/perspective/grid/live tests passed before the last seven reviewed-case
  tests were added; all 14 final audit tests pass. Broader run: 378 tests passed
  (collected before those last seven; excludes Stockfish engine tests). All 39
  browser tests pass, including full-circle synthetic tracking. Frozen A.1/A.2
  and A.3 source hashes remain intact; output reuse is rejected; diff check passes.
- Decision: B.1 audit acceptance criteria met within documented historical-data
  limits; awaiting owner review. Geometry should be prioritized for raw completion,
  while oracle-geometry recognition errors still require later investigation.
  No training, holdout use, UX/fusion change, commit or push. B.2 has not started.

## B.1b — Geometry reliability improvements

- Date: 2026-10-18. Owner approved B.1 and authorized only this follow-up before B.2.
- Inspected full candidate pools for all 30 historical raw inputs. Three of seven
  referenced failures contain seeds within the declared .25-square RMS/.5-square
  maximum tolerance; four do not. A useful White-right seed is ranked 103rd.
  Raw Hough lines near board edges are lost in the capped line pool; scalar checker
  evidence also accepts an inaccurate Black-near polygon. No threshold relaxation.
- Added opt-in `marked-board` raw recovery in `src/vision/geometry_reliability.py`
  and `scripts/predict_position.py`. Reuses the existing frozen board-font reader,
  bootstrap registration and geometry/orientation guards. Successful legacy inputs
  remain identical. Default legacy and all browser behavior remain unchanged.
- Added `scripts/evaluate_geometry_reliability.py`, 47 focused regression tests,
  `docs/evaluation/geometry_reliability_b1b.md`, versioned numeric results and four
  diagnostic images under `docs/evaluation/geometry_examples_b1b/`.
- Legacy → opt-in development completion: White-near 12/18→18/18, Black-near
  2/6→6/6, White-left 2/3→3/3, White-right 0/3→3/3. Exact boards 0/30→1/30.
  Reviewed-reference geometry and all-64-centre agreement: completed 16/23→23/23.
  Seven remaining geometries are unverified, not newly labeled ground truth.
- All 16 old successful placements, crop geometry and probabilities remain exact.
  Two complete runs agree on placements/probabilities. The seven recovered
  referenced boards have 21/448 classification errors; all 30 outputs have 99/1920
  errors. Development-set gains do not establish unbiased generalization.
- Validation: 47 focused, 432 broader Python (excluding Stockfish) and 39 browser
  tests pass. Opt-in CLI and expected legacy/conflicting-input/overwrite refusals
  verified. Packaged inspector reproduces all 538 candidates on a real White-right
  input. All 357 protected files match after tests; diff/syntax/whitespace pass.
- Full final artifacts: `outputs/evaluation/geometry_b1b_20261018_run2/`.
  Frozen A.1/A.2, all B.1 outputs and A.3 sources preserved. No training, new final
  holdout use, fusion/UX changes, commit or push.
- Decision: B.1b complete, awaiting owner review. Recommend B.2 with prepared
  geometry, and fresh A.3 pilot/development validation before default recovery
  rollout. Existing font/marking dependence and seven missing geometry references
  remain limitations. B.2 has not started.

## B.2 — Crop and resolution experiments (complete; awaiting review)

- Completed 2026-10-19; registered run 2026-10-18. Only B.2 performed.
- Report: `docs/evaluation/crop_resolution_b2.md`; design and locked config:
  `docs/evaluation/crop_resolution_design_b2.md`, `configs/experiments/b2_crop_resolution.json`.
- Eight frozen-weight probes plus three matched three-epoch adaptation arms;
  validation loss selects epochs 0/3/3 for baseline/320/tighter 224 respectively.
  Validation occupied accuracy selects tighter 224; test is descriptive only.
- Baseline reproduced exactly: 1393/1472 overall, 377/456 occupied, 0/23 exact boards.
  Tighter 224: 1410/1472 overall, 404/456 occupied, 0/23 exact; 40 old errors rescued,
  23 previously correct squares broken, including 10 new empty false positives.
- One-square padding at 224 is the experimental recommendation, with baseline
  retained as control. Three test positions, one seed, bounded adaptation and
  missing historical session provenance limit conclusions. Production stays unchanged.
- Main CPU forward ms/board: baseline 367.5, tighter 364.6, 320 611.5; identical
  parameter counts. All per-class/orientation metrics, uncertainty and costs exported.
- Runtime checkpoints/results: `outputs/evaluation/crop_b2_20261018_run1/`;
  analysis: `outputs/evaluation/crop_b2_20261018_analysis/`; reproduction baseline:
  `outputs/evaluation/crop_b2_20261018_baseline/`. Versioned numeric report, CSVs
  and representative native-input/confusion figures accompany the written report.
- Validation: 454 Python tests (excluding external Stockfish), 22 focused tests,
  39 browser tests; syntax/whitespace checks pass. Original per-crop probabilities
  exactly match; all caches and 624 protected inputs hash-match. No new data,
  geometry/recovery/UX change, A.3 final holdout access, commit or push.
- B.2 acceptance criteria satisfied with documented evidence limitations. Stop for
  review. Recommend isolated joint RGB/mask augmentation in B.3, then validation-only
  calibration/abstention in B.4; neither implemented.

## B.3 — Geometric and visual augmentation (complete; awaiting review)

- Completed 2026-10-19. Report: `docs/evaluation/augmentation_b3.md`; registered
  design/config: `docs/evaluation/augmentation_design_b3.md`, `configs/experiments/b3_augmentation.json`.
- Six separate families × two seeds × one full-network warm-start epoch, from the
  unchanged approved B.2 tighter-224 checkpoint. Literal off and legacy visual
  controls; validation-loss epoch selection and validation-only safety gates.
- Decision: `retain_b2`. Retain the approved B.2 tighter-224 checkpoint without adding a new augmentation recipe.
- All 13 models report clean validation/test overall/occupied, empty false positives,
  missed pieces, full per-class confusion/recall/precision, orientation, exact boards,
  paired square/position changes, CPU costs and checkpoint size. Historical evidence
  only: three test positions, two conditional seeds, one epoch; unknown old sessions.
- No synthetic occlusion without verified piece/visibility support; no combined family.
- Validation: 482 Python tests (including 28 new focused), 39 browser tests, syntax/
  whitespace checks. B.2 probabilities exactly reproduce; 764 protected files and
  experimental checkpoint/source hashes match. No production or holdout changes.
- Isolated run/analysis: `outputs/evaluation/augmentation_b3_20261019_run1/` and
  `outputs/evaluation/augmentation_b3_20261019_analysis/`. Versioned numeric report,
  per-class/comparison/paired-change CSVs and reviewed diagnostic figures retained.
- Acceptance criteria satisfied. Recommend validation-only calibration/abstention
  in B.4 with the recommended checkpoint/preprocessing; B.4 not started. No commit/push.
