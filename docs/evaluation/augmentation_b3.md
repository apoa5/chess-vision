# B.3 — Geometric and visual augmentation

Completed 2026-10-19. **Historical development evidence only.** Retain the approved B.2 tighter-224 checkpoint without adding a new augmentation recipe. Production behavior remains unchanged.

## Design and reproduction

The [registered design](augmentation_design_b3.md) and `configs/experiments/b3_augmentation.json` define six separate families × seeds 42/123 × **one full-network adaptation epoch**, independently initialized from the approved B.2 tighter-224 checkpoint. This is a bounded warm-start screen, not convergence. Each arm sees all 7,168 training crops with fresh Adam 0.0001, batch 16 and four CPU threads; sample order/dropout seed is matched across families. Both seeds share the same B.2 initialization.

A literal augmentation-off control and the existing visual recipe distinguish augmentation effects from continued training. New families do not stack the existing photometric recipe. Retain the minimum weighted validation-loss checkpoint over epochs zero/one; selecting epoch zero means the full training pass was performed but rejected. All arms finish before family safety selection, whose saved decision precedes historical test prediction. No test metric can alter the decision.

Configuration SHA-256: `7968d3946c8c95fedd4d9c5e8e65c75ca28e342940fd86953df2c5a754ac2f50`. Initializer SHA-256: `dad8eea28482ee894dfca0794d9f4415d21ac58db23c6d6bed8c4307c71b1bd8`. Full source, data, class-weight, split/cache and checkpoint provenance is recorded with the run.

Run from repository root using the existing environment. Substitute unused output directories for reproduction; tools refuse overwrite. Run timing measurements sequentially after training/tests.

```bash
.venv/bin/python scripts/analyze_augmentation_experiments.py --preview-only --output-dir outputs/evaluation/augmentation_b3_20261019_preview2
.venv/bin/python scripts/run_augmentation_experiments.py --output-dir outputs/evaluation/augmentation_b3_20261019_run1
.venv/bin/python scripts/analyze_augmentation_experiments.py --run-dir outputs/evaluation/augmentation_b3_20261019_run1 --output-dir outputs/evaluation/augmentation_b3_20261019_analysis
.venv/bin/python scripts/benchmark_augmentation_experiments.py --run-dir outputs/evaluation/augmentation_b3_20261019_run1 --output outputs/evaluation/augmentation_b3_20261019_analysis/cost.json
.venv/bin/python -m pytest -q --ignore=tests/test_engine.py
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
```

The initial browser command picked an unavailable host Node installation (WSL startup failure); the bundled toolchain command above passed. No package/model download or dependency installation. Runtime versions are recorded in the numeric provenance.

## Physical interpretation and label safety

Perspective/affine: 65% probability, corner jitter ±2% width, shear ±2°. Framing: 65%, rotation ±5°, shifts ±3%, scale .95–1.05. Lighting: 65%, gamma/contrast .85–1.15, exposure ±.06 and smooth gradient ±.15. Video quality: 65%, mild 3×3 Gaussian blur, three-pixel horizontal/vertical motion blur or JPEG 75–95, with ≤1.5/255 noise. `legacy_visual` exactly reuses the existing brightness/contrast/blur/noise implementation. Configuration enables/disables families; fixed tested strengths are checked against implementation constants and cannot silently drift.

The same forward homography transforms RGB and binary target mask, using linear/nearest interpolation respectively. A 25-pixel guard around the ground-square mask bounding box must remain inside the crop, otherwise the transform is rejected. There are no flips, quarter turns, orientation relabels or color inversion. Small replicated edge wedges are synthetic framing artifacts, outside the protected center; the guard is not a verified 3-D silhouette annotation. All evaluation uses unaugmented original approved crops.

The [actual training previews](augmentation_examples_b3/augmentation_examples.png) were visually inspected before training; targets stayed aligned and transforms were modest. Green masks are diagnostic overlays only. A homography cannot generate genuinely new 3-D viewpoints, reveal hidden crowns or create missing detail. Augmentation complements real multi-angle captures.

**Partial occlusion is deferred.** Ground-square masks are not whole-piece masks, and no matching current visibility reviews establish safe erasures. Random erasing could remove identity features or corrupt empty examples. No unverified occlusion example is trained or labeled as valid ground truth. No combined family is trained in this separate-family screen.

## Same-example comparisons

The B.2 anchor reproduces its saved probabilities exactly (maximum absolute difference **0.0**). Splits remain 7,168/1,600/1,472 crops and 12/3/3 physical positions. Test support is 1,472 squares, 456 occupied, 1,016 empty and 23 boards. All models use the identical frozen prepared geometry, one-square-padding RGB/mask 224 input, class mappings and normalization. No geometry/orientation/recovery or production change enters the comparison.

| Model | Best epoch | Val occupied % | Test overall % | Test occupied % | Empty FP /1016 | Missed /456 | Black-rook recall % | Exact /23 | Confirmed forward ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b2_anchor | anchor | 95.09 | 95.79 | 88.60 | 10 | 0 | 41.67 | 0 | 282.4 |
| none_seed42 | 1 | 95.73 | 95.99 | 87.94 | 4 | 1 | 38.89 | 0 | 274.2 |
| legacy_visual_seed42 | 1 | 95.51 | 95.79 | 87.28 | 4 | 0 | 38.89 | 0 | 274.8 |
| perspective_affine_seed42 | 1 | 94.87 | 95.79 | 87.28 | 4 | 1 | 41.67 | 0 | 283.6 |
| framing_seed42 | 1 | 96.58 | 96.20 | 88.38 | 3 | 0 | 44.44 | 0 | 273.2 |
| lighting_seed42 | 0 | 95.09 | 95.79 | 88.60 | 10 | 0 | 41.67 | 0 | 270.8 |
| video_quality_seed42 | 1 | 96.37 | 95.72 | 87.94 | 8 | 1 | 38.89 | 0 | 279.7 |
| none_seed123 | 0 | 95.09 | 95.79 | 88.60 | 10 | 0 | 41.67 | 0 | 281.5 |
| legacy_visual_seed123 | 0 | 95.09 | 95.79 | 88.60 | 10 | 0 | 41.67 | 0 | 287.9 |
| perspective_affine_seed123 | 0 | 95.09 | 95.79 | 88.60 | 10 | 0 | 41.67 | 0 | 270.5 |
| framing_seed123 | 0 | 95.09 | 95.79 | 88.60 | 10 | 0 | 41.67 | 0 | 271.4 |
| lighting_seed123 | 0 | 95.09 | 95.79 | 88.60 | 10 | 0 | 41.67 | 0 | 277.5 |
| video_quality_seed123 | 0 | 95.09 | 95.79 | 88.60 | 10 | 0 | 41.67 | 0 | 266.6 |

The [full comparison CSV](augmentation_comparison_b3.csv) includes initial and confirmed CPU times as separate columns, along with both validation/test false-positive and missed-piece rates, Black rook/queen/king recall, macro F1, model bytes and exact boards. Denominators are true empty and true occupied respectively; per-class empty precision is also exported and must not be confused with the empty false-positive rate.

## Validation-only decision

Retain the approved B.2 tighter-224 checkpoint without adding a new augmentation recipe.

To qualify, both seeds must pass count-based gates against matched `none` and B.2: empty FP/missed counts ≤reference+1; overall correct ≥reference−2; each Black rook/queen/king correct ≥reference−1; each orientation occupied correct ≥reference−1; exact boards not lower. Occupied accuracy cannot regress versus matched control in either seed, and mean gain must reach one percentage point. Among eligible families rank mean validation macro F1, then mean occupied gain. These small count tolerances are preregistered screening rules, not statistically established non-inferiority margins.

| Family | Mean val occupied gain vs control (pp) | Eligible | Gate violations |
|---|---:|---|---|
| legacy_visual | -0.11 | no | seed42/control/orientation/black_near_camera; seed42/anchor/orientation/black_near_camera; seed42/occupied_regression; mean_occupied_gain_under_one_percentage_point |
| perspective_affine | -0.43 | no | seed42/control/empty_false_positives; seed42/control/overall_correct; seed42/control/black_king_recall; seed42/control/exact_boards; seed42/control/orientation/white_near_camera; seed42/control/orientation/white_right; seed42/anchor/black_king_recall; seed42/anchor/exact_boards; seed42/anchor/orientation/white_left; seed42/anchor/orientation/white_right; seed42/occupied_regression; mean_occupied_gain_under_one_percentage_point |
| framing | +0.43 | no | mean_occupied_gain_under_one_percentage_point |
| lighting | -0.32 | no | seed42/control/empty_false_positives; seed42/control/overall_correct; seed42/control/black_queen_recall; seed42/control/exact_boards; seed42/control/orientation/white_near_camera; seed42/occupied_regression; mean_occupied_gain_under_one_percentage_point |
| video_quality | +0.32 | no | mean_occupied_gain_under_one_percentage_point |

Seven arm selections retain epoch-zero B.2 weights (all seed-123 arms and lighting seed42). Their probabilities are checked against the anchor, so these rows are not seven independent learned models. Apparent lighting seed42 rescues/breaks against control42 reflect fallback to B.2, not learned lighting robustness.

The decision is saved in `selection.json` before any test inference. Historical test gains or a favorable single seed do not override safety violations. No best-test-seed, ensemble, longer-run or combined-recipe rescue is introduced.

## Orientation and paired square changes

Cells are occupied accuracy percentages. Group sizes remain Black-near 6 boards, White-left 2, White-near 12 and White-right 3. Full per-orientation confusion/occupancy/exact metrics are in the numeric export.

| Model | Black-near | White-left | White-near | White-right |
|---|---:|---:|---:|---:|
| b2_anchor | 92.65 | 80.56 | 87.50 | 88.24 |
| none_seed42 | 87.50 | 83.33 | 87.50 | 92.65 |
| legacy_visual_seed42 | 86.03 | 83.33 | 87.04 | 92.65 |
| perspective_affine_seed42 | 87.50 | 83.33 | 86.11 | 92.65 |
| framing_seed42 | 89.71 | 83.33 | 87.96 | 89.71 |
| lighting_seed42 | 92.65 | 80.56 | 87.50 | 88.24 |
| video_quality_seed42 | 88.24 | 83.33 | 87.96 | 89.71 |
| none_seed123 | 92.65 | 80.56 | 87.50 | 88.24 |
| legacy_visual_seed123 | 92.65 | 80.56 | 87.50 | 88.24 |
| perspective_affine_seed123 | 92.65 | 80.56 | 87.50 | 88.24 |
| framing_seed123 | 92.65 | 80.56 | 87.50 | 88.24 |
| lighting_seed123 | 92.65 | 80.56 | 87.50 | 88.24 |
| video_quality_seed123 | 92.65 | 80.56 | 87.50 | 88.24 |

Paired changes below use the matching augmentation-off seed; counts are per model, not summed as independent observations. The anchor/control rows are self comparisons.

| Model | Errors rescued | Correct squares broken | Occupied rescued | Occupied broken | Empty FP | Missed occupied | Identity errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| b2_anchor | 0 | 0 | 0 | 0 | 10 | 0 | 52 |
| none_seed42 | 0 | 0 | 0 | 0 | 4 | 1 | 54 |
| legacy_visual_seed42 | 8 | 11 | 7 | 10 | 4 | 0 | 58 |
| perspective_affine_seed42 | 7 | 10 | 6 | 9 | 4 | 1 | 57 |
| framing_seed42 | 12 | 9 | 11 | 9 | 3 | 0 | 53 |
| lighting_seed42 | 14 | 17 | 14 | 11 | 10 | 0 | 52 |
| video_quality_seed42 | 5 | 9 | 5 | 5 | 8 | 1 | 54 |
| none_seed123 | 0 | 0 | 0 | 0 | 10 | 0 | 52 |
| legacy_visual_seed123 | 0 | 0 | 0 | 0 | 10 | 0 | 52 |
| perspective_affine_seed123 | 0 | 0 | 0 | 0 | 10 | 0 | 52 |
| framing_seed123 | 0 | 0 | 0 | 0 | 10 | 0 | 52 |
| lighting_seed123 | 0 | 0 | 0 | 0 | 10 | 0 | 52 |
| video_quality_seed123 | 0 | 0 | 0 | 0 | 10 | 0 | 52 |

The [changed-square CSV](augmentation_paired_square_changes_b3.csv) records source image, position, canonical square ID, truth and both predictions, including changed wrong predictions. It identifies exactly which squares improve or regress. Occupied rescues/breaks and the occupancy error decomposition distinguish piece identity gains from occupancy tradeoffs.

The [paired diagnostic examples](augmentation_examples_b3/paired_prediction_examples.png) show up to two rescues and two regressions for each new family at preregistered representative seed 42. Original prepared RGB is displayed at native input size with target overlay; no synthetic transform is applied to evaluation examples. These examples are qualitative and are not new occlusion/visibility annotations.

Framing seed42 versus matched control changes occupied correctness by −1/96 on position006, +3/320 on position010 and 0/40 on position016. Leaving out one position gives −0.74 to +0.83 percentage points; this sensitivity interval is descriptive, not a confidence interval. Against B.2, Black-near occupied accuracy falls 92.65→89.71%, while the other three orientations rise slightly. The effect is not uniform viewpoint robustness.

Concrete examples: framing rescues `position_006_var_05_r7_c2` (f8, Black rook: bishop→rook), but breaks `position_006_var_05_r4_c5` (c5, Black pawn: pawn→bishop). Perspective introduces a false Black pawn at `position_006_var_07_r5_c4` (d6, truly empty). Video quality introduces another at `position_006_var_06_r0_c0` (h8). These same-input changes illustrate context/identity and occupancy tradeoffs, not newly verified occlusion causes.

Paired per-position counts and leave-one-position-out deltas for every model are in the numeric export; only three physical test groups exist. Seed ranges and position sensitivities are descriptive, not narrow square-IID confidence intervals.

## Per-class analysis

All classes, both seeds and both splits have precision/recall/F1/support in the [per-class CSV](augmentation_per_class_b3.csv). Full confusion matrices are in the numeric export and each run evaluation directory. Representative [control](augmentation_examples_b3/none_seed42_test_confusion.png), [perspective](augmentation_examples_b3/perspective_affine_seed42_test_confusion.png) and [framing](augmentation_examples_b3/framing_seed42_test_confusion.png) matrices are retained for review.

| Model | Black-rook P/R % | Black-queen P/R % | Black-king P/R % | Empty P/R % |
|---|---:|---:|---:|---:|
| b2_anchor | 93.75/41.67 | 47.37/69.23 | 40.62/56.52 | 100.00/99.02 |
| none_seed42 | 93.33/38.89 | 50.00/76.92 | 52.00/56.52 | 99.90/99.61 |
| legacy_visual_seed42 | 100.00/38.89 | 47.62/76.92 | 50.00/56.52 | 100.00/99.61 |
| perspective_affine_seed42 | 100.00/41.67 | 50.00/84.62 | 52.17/52.17 | 99.90/99.61 |
| framing_seed42 | 88.89/44.44 | 52.63/76.92 | 50.00/56.52 | 100.00/99.70 |
| lighting_seed42 | 93.75/41.67 | 47.37/69.23 | 40.62/56.52 | 100.00/99.02 |
| video_quality_seed42 | 100.00/38.89 | 47.62/76.92 | 48.15/56.52 | 99.90/99.21 |
| none_seed123 | 93.75/41.67 | 47.37/69.23 | 40.62/56.52 | 100.00/99.02 |
| legacy_visual_seed123 | 93.75/41.67 | 47.37/69.23 | 40.62/56.52 | 100.00/99.02 |
| perspective_affine_seed123 | 93.75/41.67 | 47.37/69.23 | 40.62/56.52 | 100.00/99.02 |
| framing_seed123 | 93.75/41.67 | 47.37/69.23 | 40.62/56.52 | 100.00/99.02 |
| lighting_seed123 | 93.75/41.67 | 47.37/69.23 | 40.62/56.52 | 100.00/99.02 |
| video_quality_seed123 | 93.75/41.67 | 47.37/69.23 | 40.62/56.52 | 100.00/99.02 |

Framing seed42 versus B.2 fixes seven empty false positives but adds one occupied identity error: six net overall-square gains, with occupied accuracy slightly lower (403/456 versus 404/456). Against its matched no-augmentation control, it has one fewer empty false positive, one fewer missed occupied square and one fewer identity error (56 versus 59 total errors). Its overall gain versus B.2 is therefore mainly an occupancy tradeoff, not a broad improvement in piece identity. Black-rook recall rises only one test example (15/36 to 16/36), and exact boards remain 0/23.

B.2 has 14 Black-rook→Black-king and 9 Black-king→Black-queen errors. Framing seed42 reduces those to 10 and 8, but increases Black-rook→Black-knight errors from 1 to 5; total Black-rook recall gains only one square and Black-king recall is unchanged. This is a redistribution of difficult dark-piece confusions rather than a decisive resolution.

The difficult-class tradeoffs above must be considered alongside occupied accuracy. Overall accuracy is dominated by empty squares. No improvement in confidence, exact reconstruction or unseen-view robustness is inferred solely from a square accuracy gain. Two-dimensional augmentation also cannot disambiguate fully occluded or intrinsically similar dark crowns without additional evidence.

## Cost and checkpoints

All models have 1,531,325 parameters / 6,125,300 parameter bytes. Checkpoint file sizes are exported per model; small serialization/metadata differences do not change architecture size. CPU forward times above use batch 16, four threads, two warmup boards per model/eight measured 64-square passes, with all thirteen models loaded and measured in interleaved fixed-randomized order. The initial raw per-model timing pass varied from roughly 279 to 983 ms despite identical shapes, motivating this measurement confirmation after training/test results. Initial times remain in the comparison CSV; the [cost CSV](augmentation_cost_b3.csv) gives initial/confirmed times and IQR. No selection or prediction changes follow from this measurement repair. Host timing variation is not an algorithmic speed gain.

Shared inference preprocessing + forward median: **469.9 ms per 64 loaded crops**, two warmups/four measured passes. Every family uses the same unchanged B.2 tighter-crop preprocessing, with no inference-time augmentation. Excludes disk, geometry, model loading and phone execution. Training-only transform costs below exclude backprop and data loading (128 loaded cache samples, one warmup/three measured passes).

| Family | Transform + normalization median ms/crop | Guard rejects /384 measured |
|---|---:|---:|
| none | 0.171 | 0 |
| legacy_visual | 0.977 | 0 |
| perspective_affine | 2.464 | 0 |
| framing | 2.201 | 0 |
| lighting | 0.829 | 0 |
| video_quality | 2.274 | 0 |

This small timing sample does not estimate real phone video performance or the full raw pipeline. No extra inference branch, augmentation, multi-view fusion or UX work is introduced.

Twelve isolated checkpoints and their status/history files are under `outputs/evaluation/augmentation_b3_20261019_run1/training/<family>_seed<seed>/best.pt`. They record B.2 initialization SHA, seed/family, validation selection, required crop specification, class mapping and experiment-only flag. Run provenance supplies full configuration/source hashes. Frozen A.1 and all B.2 checkpoints remain unchanged.

## Limitations and B.4 recommendation

These are already-inspected historical development results, not unbiased generalization estimates. Only three physical test positions, one already-selected B.2 initialization and two conditional training seeds are available. A single warm-start epoch can reject or miss slower-acting benefits; this does not rule out other strengths/schedules. Camera orientation, physical position, class support and lighting are confounded.

Historical session IDs/timestamps are absent for 160 sources. Existing position/source/source-byte/placement/known-session leakage checks pass; no frames or split assignments are introduced. Independent-session and all cross-position near-duplicate leakage cannot be certified retrospectively. The untouched A.3 final holdout is never loaded or used for development/selection. Fresh balanced development captures with reviewed geometry/visibility remain necessary before promotion.

For B.4, bind calibration to the recommended checkpoint and exact tighter-crop preprocessing; fit temperature or other calibration only on validation, retain raw logits, and evaluate reliability/abstention with empty false positives, missed pieces and per-class resolution coverage visible. Calibration cannot repair argmax identity errors or guarantee exact boards. If B.2 is retained, use its existing unmodified checkpoint. No B.4 calibration, thresholds or consumer behavior are implemented here.

## Validation, files and acceptance

[Full numeric results and provenance](augmentation_results_b3_20261019.json).

Validation: **482 Python tests passed in 218.29 seconds**, including 28 new focused alignment, direction, clipping, label/cache preservation, deterministic seed/epoch, augmentation-off, metric-denominator and safety-gate tests. External Stockfish integration (`tests/test_engine.py`) is excluded. Initial focused run passed 24 before four additional regressions; all 28 are covered by the broader run and the final standalone focused run (28 passed in 3.44 seconds). **39 browser tests passed** using bundled Node. Syntax and whitespace/diff checks pass. No dependency installation.

Original protected files: **764**, plus three run-source guards; hashes pass before/after and at final export. All evaluated checkpoint hashes match. B.2 per-crop probabilities reproduce exactly. No original model, split, geometry, orientation, preprocessing, recovery policy or scanning UI was modified.

New files: `configs/experiments/b3_augmentation.json`; `src/vision/augmentation_experiments.py`; `scripts/run_augmentation_experiments.py`, `scripts/analyze_augmentation_experiments.py`, `scripts/benchmark_augmentation_experiments.py`; `tests/test_augmentation_experiments.py`; registered design; this report; versioned numeric report, four CSVs and diagnostic assets. Only the B.3 section of `docs/implementation_progress.md` is updated. Existing uncommitted work is preserved.

- [x] Geometry can be enabled/disabled by experiment configuration; production defaults unchanged.
- [x] Augmentation effects measured separately from crop changes with matched augmentation-off controls, identical geometry/splits and budgets, two seeds.
- [x] RGB/mask alignment, label preservation, deterministic behavior and valid tensors tested; actual previews manually inspected.
- [x] Each orientation reports improvements/regressions; all requested occupancy, class, exact-board, paired-square and cost metrics recorded.
- [x] Frozen baseline and B.2 artifacts/checkpoints preserved; final holdout untouched; validation-only safety selection.
- [x] Occlusion deferred with explicit label-validity reason; no unsupported combined recipe or generalization claim.
- [x] Reproducible isolated checkpoints/configuration and B.4 recommendation supplied.

**B.3 complete. Stop for owner review. No commit, push or B.4 work.**
