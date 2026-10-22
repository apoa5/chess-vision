# B.2 — Crop and resolution experiments

Completed 2026-10-19; run/configuration registered 2026-10-18. **Development evidence only.** Recommend the adapted one-square-padding, 224-pixel input for further testing; retain production assets pending fresh development captures and review. Exact-board recognition remains unsolved.

## Design and reproduction

The [registered design](crop_resolution_design_b2.md) and `configs/experiments/b2_crop_resolution.json` specify eight frozen-weight probes and three matched adaptation arms. All use the same approved stored crops, semantic square IDs, labels and position-grouped split: 7,168 train / 1,600 validation / 1,472 test crops; 12/3/3 physical positions. No localization, homography, orientation, B.1b recovery or final A.3 holdout enters these comparisons.

Every adapted arm independently starts from the untouched epoch-16 MobileNetV3 Small checkpoint, with fresh Adam, learning rate 0.0001, batch 16, seed 42 and three full-network epochs. Original photometric augmentation and class weighting are reused. Select each checkpoint by minimum weighted validation loss over epochs 0–3; rank configurations by validation occupied accuracy, then macro F1, then inference cost. No test-based checkpoint or configuration selection. This is a bounded adaptation screen, not training to convergence or a multi-seed comparison.

The unchanged-crop arm selects epoch zero (loss 0.23349); its later losses are 0.27159, 0.31048, 0.28150. Both changed-crop arms select epoch three (320: 0.17006; tighter 224: 0.13692). Including epoch zero prevents claiming an improvement by arbitrarily choosing a later control epoch.

Run from repository root with the existing virtual environment. Each output directory must be new; the tools refuse overwrite. Checkpoints and caches remain isolated in ignored outputs.

```bash
.venv/bin/python scripts/evaluate_classifier.py --no-metadata-report --output-dir outputs/evaluation/crop_b2_20261018_baseline
.venv/bin/python scripts/run_crop_experiments.py --output-dir outputs/evaluation/crop_b2_20261018_run1
.venv/bin/python scripts/analyze_crop_experiments.py --run-dir outputs/evaluation/crop_b2_20261018_run1 --baseline-report outputs/evaluation/crop_b2_20261018_baseline/predictions.json --output-dir outputs/evaluation/crop_b2_20261018_analysis
.venv/bin/python -m pytest -q --ignore=tests/test_engine.py
```

For a rerun substitute unused output names consistently. Runtime environment: Python 3.12.3, torch 2.14.1+cpu, torchvision 0.29.1+cpu, OpenCV 5.0.0, NumPy 2.5.3; CPU threads 4, inter-op/OpenCV threads 1. No package installation or model download was needed. Full provenance, cache hashes, split audit and training history are in the run directory.

## Current representation and controlled factors

The canonical board canvas is 1,200×1,200 with an 800×800 playing area and 200-pixel border. A roughly 100-pixel target square plus two squares of padding per side produces approximately 500×500 crops (test widths 496–512, median 502). At 224×224, the measured median target-mask area-equivalent width is 45 pixels. BGR→RGB, area interpolation, ImageNet normalization and nearest-neighbor binary target-mask resizing match the frozen four-channel preprocessing. Masks describe ground squares, not complete tall-piece silhouettes.

One-square padding reduces the crop to about three square widths and raises nominal target width to 75 input pixels; the 320 arm retains five-square context and raises it to 64. The outer-context suppression probe keeps scale/mask/canvas fixed while removing outer RGB context. The quarter-detail probes downsample RGB by four with area interpolation, then restore with linear interpolation before final 224/320 resizing; they share exactly the same degraded information and unchanged geometry. Tighter context and target scale change jointly; the separate controls help interpretation but do not fully factorially identify their contributions.

Native raw-photo target mean-edge length on test is 36.9–86.9 pixels, median 68.1; shortest-edge median is 62.3 pixels. Across the full dataset mean edges span approximately 27–91 pixels. Existing normalization already upsamples many targets. Enlarging an input cannot create missing source detail; 320 can preserve more existing samples in some views while merely resampling others. Ground-square size is only a proxy for available piece detail.

## Same-example comparison

Test: 1,472 squares, 456 occupied, 23 boards. All exact-board denominators are 23. Frozen variants are distribution-sensitivity diagnostics, not independently trained alternatives. Three adapted arms receive identical budgets. Latency is loaded-model CPU milliseconds per 64-square board, batch 16, two warmups and eight repeats (median).

| Variant | Best adaptation epoch | Val occupied % | Test overall % | Test occupied % | Exact boards | Errors | Forward ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| context2_224_frozen | — | 91.03 | 94.63 | 82.68 | 0/23 | 79 | 367.5 |
| context2_224_adapted | 0 | 91.03 | 94.63 | 82.68 | 0/23 | 79 | 343.4 |
| context2_320_frozen | — | 45.94 | 79.42 | 33.55 | 0/23 | 303 | 654.1 |
| context2_320_adapted | 3 | 92.52 | 95.38 | 85.09 | 0/23 | 68 | 611.5 |
| context1_224_frozen | — | 31.20 | 77.99 | 29.17 | 0/23 | 324 | 265.0 |
| context1_224_adapted | 3 | 95.09 | 95.79 | 88.60 | 0/23 | 62 | 364.6 |
| context_half_224_frozen | — | 8.12 | 69.09 | 3.51 | 0/23 | 455 | 362.1 |
| target_only_224_frozen | — | 10.04 | 13.04 | 9.65 | 0/23 | 1280 | 350.5 |
| suppress_outer_224_frozen | — | 24.57 | 72.15 | 20.39 | 0/23 | 410 | 347.5 |
| quarter_detail_224_frozen | — | 86.32 | 93.34 | 79.17 | 0/23 | 98 | 340.8 |
| quarter_detail_320_frozen | — | 33.55 | 76.77 | 26.97 | 0/23 | 342 | 812.8 |

The original evaluator reproduces 1,393/1,472 correct, 377/456 occupied correct, macro F1 0.79676 and 0/23 exact boards. The experimental control is checked against its per-crop probabilities. The selected tighter model gives 1,410/1,472 and 404/456: +1.15 percentage points overall, +5.92 occupied. It rescues 40 baseline errors but breaks 23 previously correct squares. Its 62 errors include 10 empty→piece false positives and 52 occupied identity errors; baseline had 79 identity errors and no occupancy errors. No exact board is rescued.

Simply applying a changed crop/input to frozen weights fails severely. Adaptation is essential; the poor zero-shot results do not prove those representations are intrinsically worse. Quarter-detail 224 falls to 79.17% occupied, consistent with some detail sensitivity. Quarter-detail 320 is additionally affected by input-scale distribution shift, so it does not establish that 320 is intrinsically harmful or beneficial. Outer-context removal also changes appearance substantially: its failure establishes learned context dependence, not that all removed context is necessary after training.

## Orientation and position sensitivity

| Camera orientation | Test boards | Frozen occupied % | Adapted 320 % | Adapted tighter 224 % |
|---|---:|---:|---:|---:|
| black_near_camera | 6 | 83.09 | 86.03 | 92.65 |
| white_left | 2 | 80.56 | 75.00 | 80.56 |
| white_near_camera | 12 | 81.02 | 86.57 | 87.50 |
| white_right | 3 | 88.24 | 83.82 | 88.24 |

Tighter-crop occupied gains by physical position: 006 +3/96 (+3.13 pp), 010 +24/320 (+7.50 pp), 016 +0/40. Leaving out one position gives +2.21 to +6.67 pp; these are descriptive sensitivity bounds, **not confidence intervals**. The 320 arm loses 2/40 on position 016 and has a leave-one-position-out range of −0.74 to +3.13 pp. These observations favor testing tighter 224 before spending more on 320. Three physical test positions, one seed and unequal view/class support do not establish independent generalization.

Historical records lack session IDs/timestamps for 160 sources. Position, known-session, identical-source-byte and identical-placement leakage checks pass, and no frames/splits are added or moved. True session independence and all cross-position near-duplicates cannot be certified retrospectively. No final holdout is opened or used for selection.

## Per-class and error analysis

Rows below show test precision/recall percentages. Full validation/test per-class F1, support and confusion matrices for all eleven variants are in the [numeric export](crop_resolution_results_b2_20261018.json) and [per-class CSV](crop_resolution_per_class_b2.csv). The [comparison CSV](crop_resolution_comparison_b2.csv) covers all eleven rows. [Baseline confusion](crop_examples_b2/baseline_test_confusion.png) and [tighter confusion](crop_examples_b2/tighter_test_confusion.png) are retained for review.

| Class | Support | Baseline precision | Baseline recall | Tighter precision | Tighter recall |
|---|---:|---:|---:|---:|---:|
| empty | 1016 | 100.00 | 100.00 | 100.00 | 99.02 |
| white_pawn | 114 | 88.24 | 92.11 | 94.17 | 99.12 |
| white_knight | 26 | 62.07 | 69.23 | 95.83 | 88.46 |
| white_bishop | 26 | 90.91 | 76.92 | 92.00 | 88.46 |
| white_rook | 26 | 100.00 | 80.77 | 83.33 | 96.15 |
| white_queen | 13 | 85.71 | 92.31 | 85.71 | 92.31 |
| white_king | 23 | 95.65 | 95.65 | 87.50 | 91.30 |
| black_pawn | 104 | 92.31 | 92.31 | 94.50 | 99.04 |
| black_knight | 26 | 66.67 | 92.31 | 91.67 | 84.62 |
| black_bishop | 26 | 82.61 | 73.08 | 86.21 | 96.15 |
| black_rook | 36 | 78.26 | 50.00 | 93.75 | 41.67 |
| black_queen | 13 | 52.63 | 76.92 | 47.37 | 69.23 |
| black_king | 23 | 52.17 | 52.17 | 40.62 | 56.52 |

Black-rook recall regresses 50.0→41.7%; black-king recall only improves 52.2→56.5%, and black-queen/king precision remains low. Gains in pawns and several White-piece classes do not solve intrinsically similar dark silhouettes. Same-color identity confusion remains the central problem; no wrong-color predictions or occupied→empty misses occur in the selected test outputs.

For native target mean edges <50 / 50–70 / ≥70 pixels, baseline occupied accuracy is 78.57 / 83.85 / 82.43% (supports 42/192/222); tighter becomes 80.95 / 86.98 / 91.44%. This is compatible with using available detail more effectively, but class/view/position confounding prevents a causal detail threshold. Crowded-neighborhood accuracy is 82.63→87.43%, sparse 82.70→89.27%; occupancy in front is a proxy, not a verified occlusion annotation. No current matching per-crop visibility review exists; stale review signatures are not transferred as ground truth.

The [native-input examples](crop_examples_b2/error_context_examples.png) show actual 224/320 inputs with target masks overlaid only after prediction. Tall crowns and overlapping pawn/knight silhouettes can lie outside target ground squares, so target-only crops can clip identity features. Sparse-position king/queen errors remain without surrounding clutter, suggesting ambiguity/detail and learned class bias as additional causes. These are qualitative observations, not verified causal labels or newly reviewed photographs.

## Inference cost and artifacts

All arms retain 1,531,325 parameters / 6,125,300 parameter bytes. The frozen file is 6,261,635 bytes; each experimental checkpoint is 6,247,145 bytes (serialization/metadata differences, not compression). Baseline/tighter/320 forward medians are 367.5/364.6/611.5 ms per board: tighter has comparable forward cost; 320 costs about 1.66×. Repeats/IQR and preprocessing-plus-forward timings are exported separately. The initial supplemental preprocessing pass overlapped the final tests; a sequential post-test confirmation gives baseline/tighter/320 medians 657.9/560.8/1247.6 ms per board (two warmups, four measured repeats, loaded crop images). These short host timings remain variable; use the main isolated forward benchmark for architecture-cost comparison. Confirmation raw times and scope are in the numeric export and `outputs/evaluation/crop_b2_20261018_analysis/timing_confirmation.json`. Timing variability means minor differences among identical 224 shapes are not architecture speed gains. Disk decoding, geometry, model loading and phone execution are excluded; no end-to-end or browser throughput claim is made.

Experimental checkpoints: `outputs/evaluation/crop_b2_20261018_run1/training/{context2_224,context2_320,context1_224}/best.pt`. Each records initialization hash, manifest fingerprint, class mapping, four channels and required crop specification. Use matching preprocessing; none replaces production assets. The unchanged arm contains epoch-zero weights. Full training histories, raw logits, probabilities, per-board/orientation metrics and all confusion figures are under that run. Analysis adds `comparison.csv`, `per_class.csv`, paired-position deltas, detail/clutter proxies and a diagnostic sheet. The versioned numeric export records hashes and results without copying large caches/checkpoints into source control.

## Recommendation for B.3 and B.4

Carry adapted one-square-padding / 224 input forward as the experimental candidate, with the frozen baseline as control. Keep production unchanged. Before promotion, validate on fresh A.3 development sessions with more independent positions, sparse dark pieces and reviewed visibility; retain the final holdout untouched. Additional seeds and a longer matched adaptation budget may change the ranking. Do not introduce extra model branches from these results.

For B.3, isolate modest joint RGB/mask rotation, scale, translation and perspective perturbations while holding this crop configuration fixed; retain an augmentation-off arm and report per-orientation/empty false-positive regressions. For B.4, fit calibration only on validation data for the selected checkpoint and exact preprocessing, retain raw logits, and evaluate abstention separately. Confidence calibration cannot fix argmax identity errors or guarantee a complete board. Neither task is implemented here.

## Validation, changes and acceptance

B.2 creates `configs/experiments/b2_crop_resolution.json`, `src/vision/crop_experiments.py`, `scripts/run_crop_experiments.py`, `scripts/analyze_crop_experiments.py`, `tests/test_crop_experiments.py`, this report, the design, numeric results and representative figures; updates only the B.2 section of `docs/implementation_progress.md`. Existing uncommitted work is preserved.

- [x] Baseline unchanged: original checkpoint and 624 protected files hash-match before/after the run; no production pipeline change.
- [x] Consistent holdouts: identical position-grouped historical evaluation examples; no A.3 final holdout; known leakage checks enforced, missing historical session provenance documented.
- [x] Occupied-square and exact-board metrics recorded for every variant, with per-class confusion/precision/recall and orientation groups.
- [x] Extra inference cost measured, with parameter/checkpoint sizes and explicit CPU timing scope.
- [x] Reproducible configuration, isolated versioned checkpoints and bounded-budget limitations recorded.

Validation: **454 Python tests passed in 219.89 s**, excluding external Stockfish integration (`tests/test_engine.py`); includes all 22 focused crop/augmentation parity, immutability, split/leakage, initialization independence and paired-position tests (also 22/22 in a separate focused run). `cd web && npm test`: **39 passed**. Syntax compilation and `git diff --check` pass. Fresh baseline probabilities match exactly (maximum difference 0.0); all experiment caches hash-match; zero current visibility labels are inferred. Original protected files and source guards pass after analysis. No commit/push or B.3/B.4 work.
