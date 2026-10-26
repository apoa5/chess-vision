# D.2 — Per-square multi-view evidence characterization

**Complete for development review, 2026-10-22. Uniform-5 outperforms diversity-5 on recognition evidence in this pilot.** Diversity-5's reduced D.1 proxy redundancy did not translate into a better recognition comparison. There is substantial cross-view complementarity, and diversity-10 raises the ground-truth oracle ceiling to 5/5 positions, but **no fusion, completed scan or full-board multi-view recognition has been implemented or measured**.

## Geometry gate and label provenance

Diversity-5 and uniform-5 each select 25 frames and share ten: their union has **40 unique frames**. Diversity-10 contains 43 actual selections and adds **15 unique frames** to that union. All **55** full-resolution frame/grid overlays were prepared without recognition results in the [review packet](evidence_d2_geometry_review_v1/index.html). The owner explicitly confirmed all four requested checks for all 55: true playing-area corners, semantic orientation, canonical a1–h8 mapping, and alignment sufficient for square crops.

[Human review artifact](../../data/metadata/evidence_d2_geometry_reviews_v1.json) binds exact frame/session/source hashes, original PTS/time base, geometry-result hashes and packet SHA. Human-verified **55**, rejected **0**, unresolved **0**, replacements **0**; all 40 primary-union frames passed. D.1's original completed-unverified records remain unchanged; verification is a separate D.2 supplement. This qualitative human approval is not an independent subpixel corner-error benchmark. Physical camera elevation and per-square occlusion remain unknown.

A separate **ground-truth error** was discovered: the supplied middlegame FEN marked **c7 empty**, while the reference and video showed a black pawn there. The owner explicitly confirmed c7 and approved a versioned correction. [Correction artifact](../../data/metadata/evidence_d2_ground_truth_corrections_v1.json) changes only rank 7 `1b1nbpp1`→`1bpnbpp1`, preserving the original FEN, reference hash and approval. Corrected full FEN:

```text
r2q1rk1/1bpnbpp1/p2ppn1p/1p4P1/3NP2P/2N1BP2/PPPQ4/2KR1B1R w - - 1 13
```

**All primary metrics below use this owner-approved correction consistently across strategies.** Original A.3 catalogs, earlier baseline labels/results and the original-label D.2 analysis remain intact. No image inference was rerun for the correction; it affects evaluation only. Earlier c7 false-positive diagnostics are preserved as a label-discrepancy example, not a confirmed classifier defect. Historical labels for the matching placement merit a future audit; none were changed here.

For transparency, the preserved original-label results were:

| Condition | Ever-correct squares | Persistently wrong | Oracle exact boards |
|---|---:|---:|---:|
| diversity_5 | 317/320 | 3 | 3/5 |
| uniform_5 | 318/320 | 2 | 3/5 |
| diversity_10 | 319/320 | 1 | 4/5 |

## Frozen extraction and evidence artifacts

The unchanged B.2 tighter-224 checkpoint SHA is `dad8eea28482ee894dfca0794d9f4415d21ac58db23c6d6bed8c4307c71b1bd8`; B.3 recommended retaining it. B.4 has no completed approved artifact in this checkout. No calibration was fitted; raw float32 logits and full uncalibrated 13-class softmax vectors remain available for later reproducible calibration. Stable float64 softmax is used and recorded; versus historical B.2 torch-float32 softmax the maximum probability difference on 64 existing prepared crops was **2.46e-7**, with **identical logits**.

The pipeline reuses D.1's frozen source corners/polygons and the existing perspective, crop and B.2 packing/normalization functions. No localization/orientation refit, RGB semantic rotation, geometry threshold, recovery policy or recognition model change occurred. All **55 saved D.1 context images matched pixel-for-pixel**, and all **3,520 packed square inputs reproduced exactly** in a separate read-only replay. Target coverage uses the unchanged .99 gate.

The model ran once per unique frame: **55 × 64 = 3,520 observations**. Strategy membership expands these to **5,952 comparison memberships**, not independent examples. There are **320 canonical square containers**, one per square/session. Containers reference observations with `unknown`/`candidate` state and null pooled distribution; no square is automatically resolved or fused.

[Evidence manifest](../../data/metadata/evidence_d2_manifest_v1.json), [artifact index](../../data/metadata/evidence_d2_index_v1.json), and [observation schema](../../configs/experiments/evidence_d2_observation.schema.json) preserve source/session/position/group IDs, exact PTS/time base, strategy/budget/trace, quality/view descriptors, geometry/human-review identities, canonical square, all logits/probabilities, predicted class, max probability, margin, entropy, crop/input/config identities and checkpoint hash. Shared observations are deduplicated; renamed copies at the same physical timestamp are identifiable. Ground truth is absent from image inference and from raw observation records.

## Main comparison — corrected ground truth

Counts are descriptive. FP denominator = true empty observations; miss denominator = true occupied observations. Identity errors are occupied→nonempty wrong-class predictions. Empty precision is a different metric. Diversity-10 has varying per-position counts (9/10/10/8/6); its pooled accuracy is not an equal-view-cost comparison. All three conditions retain their original D.1 selections with no replacement.

| Condition | Frames | Overall accuracy | Occupied accuracy | Empty FP | Occupied→empty | Identity errors | Exact single-frame boards | Ever-correct | Always wrong | Oracle exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diversity_5 | 25 | 87.12% | 68.32% | 27/1035 (2.61%) | 15/565 (2.65%) | 164 | 1/25 | 318/320 | 2 | 4/5 |
| uniform_5 | 25 | 88.81% | 73.81% | 31/1035 (3.00%) | 9/565 (1.59%) | 139 | 1/25 | 319/320 | 1 | 4/5 |
| diversity_10 | 43 | 84.27% | 66.51% | 82/1704 (4.81%) | 27/1048 (2.58%) | 324 | 1/43 | 320/320 | 0 | 5/5 |

Ever-correct percentages: **diversity-5 99.375%, uniform-5 99.6875%, diversity-10 100%**. The oracle is a **ground-truth diagnostic upper bound for choosing among observed per-view argmax labels**, not an implemented fusion rule, attainable performance guarantee, or universal mathematical ceiling for all future methods using full distributions/RGB. A square with every argmax wrong has no correct choice under this oracle; it does not prove the pixels or non-winning probabilities contain no useful information.

At equal budget, uniform-5 has **27 fewer observation errors** (179 versus 206), **31 more correct occupied observations**, six fewer occupied→empty misses, and 25 fewer identity errors. Diversity-5 has four fewer empty false positives. Their oracle board count ties at 4/5. Uniform is not uniformly better: the endgame favors diversity, while the opening favors uniform. With five sessions, these are descriptive paired outcomes, not significance/generalization evidence.

### Per-position availability and persistent errors

| Position | Diversity-5 ever correct | Uniform-5 ever correct | Diversity-10 ever correct |
|---|---:|---:|---:|
| starting | 64/64 | 64/64 | 64/64 (9 views) |
| opening | 62/64 | 64/64 | 64/64 (10 views) |
| middlegame | 64/64 | 64/64 | 64/64 (10 views) |
| endgame | 64/64 | 63/64 | 64/64 (8 views) |
| puzzle | 64/64 | 64/64 | 64/64 (6 views) |

Paired uniform-5→diversity-5 changes: starting 0, opening **−2** (b1 White knight, d1 White queen), middlegame 0, endgame **+1** (d5 Black knight), puzzle 0. There are three ties, one position favoring each strategy; net diversity loss is one physical square. No claim that zero D.1 proxy-similar pairs establishes greater recognition usefulness.

Diversity-5→diversity-10 adds **18 condition frame memberships (+72%)** for **two** newly recoverable squares, both opening b1/d1: +0.625 percentage points in ever-correct coverage and one additional oracle-exact board. Four of five positions gain no oracle coverage. Because the main-comparator union shares three of these frames, only fifteen extra unique inferences were needed in this audit. Budget 10 therefore has diminishing square-level returns but a material oracle-board effect on the opening. Its added views also increase false positives and reduce pooled per-view accuracy: more observations are not automatically better.

After correction, persistently wrong under diversity-5: opening b1 and d1 only. Under uniform-5: endgame d5 only. Under diversity-10: **none**. No persistent empty false-positive square remains under these corrected labels, though many individual views still hallucinate pieces.

### Correct-view multiplicity

Flags overlap deliberately: every-view correct is also a majority; raw k/N and all view predictions are retained. Rare correct views matter: diversity-5 has nine physical squares correct in **only one** of its five views. This is diagnostic complementarity, not a proposal to vote or choose a view using ground truth.

| Position | Condition | Every usable view | Majority | Only one view | Wrong every view |
|---|---|---:|---:|---:|---:|
| starting | diversity_5 | 31 | 57 | 4 | 0 |
| starting | uniform_5 | 33 | 61 | 1 | 0 |
| starting | diversity_10 | 25 | 58 | 0 | 0 |
| opening | diversity_5 | 36 | 55 | 2 | 2 |
| opening | uniform_5 | 35 | 58 | 2 | 0 |
| opening | diversity_10 | 23 | 53 | 0 | 0 |
| middlegame | diversity_5 | 35 | 59 | 2 | 0 |
| middlegame | uniform_5 | 37 | 59 | 3 | 0 |
| middlegame | diversity_10 | 23 | 50 | 0 | 0 |
| endgame | diversity_5 | 55 | 62 | 1 | 0 |
| endgame | uniform_5 | 54 | 61 | 1 | 1 |
| endgame | diversity_10 | 44 | 60 | 1 | 0 |
| puzzle | diversity_5 | 56 | 61 | 0 | 0 |
| puzzle | uniform_5 | 56 | 62 | 0 | 0 |
| puzzle | diversity_10 | 54 | 60 | 0 | 0 |

## Error complementarity and similarity

All pairs are compared on the **same physical canonical squares**. Disagreement can be useful (one view correct) or useless (both wrong with different labels); both are recorded. Jaccard measures overlap of the error-square sets. Binary error phi, occupied-only versions and temporal-gap/view-distance associations are in the [numeric results](evidence_d2_results_corrected_v1.json). Constant error vectors have undefined/null association, rather than fabricated zero correlation.

| Position | Condition | Mean prediction disagreement | Mean error Jaccard | View-distance vs error-Jaccard Pearson |
|---|---|---:|---:|---:|
| starting | diversity_5 | 30.16% | 0.190 | -0.234 |
| starting | uniform_5 | 25.31% | 0.110 | -0.237 |
| starting | diversity_10 | 27.56% | 0.159 | 0.067 |
| opening | diversity_5 | 24.53% | 0.312 | 0.033 |
| opening | uniform_5 | 23.91% | 0.195 | -0.246 |
| opening | diversity_10 | 27.43% | 0.250 | -0.151 |
| middlegame | diversity_5 | 27.50% | 0.157 | -0.409 |
| middlegame | uniform_5 | 22.66% | 0.190 | 0.352 |
| middlegame | diversity_10 | 31.87% | 0.226 | -0.321 |
| endgame | diversity_5 | 7.03% | 0.233 | -0.450 |
| endgame | uniform_5 | 7.50% | 0.347 | -0.738 |
| endgame | diversity_10 | 14.29% | 0.262 | -0.278 |
| puzzle | diversity_5 | 6.72% | 0.161 | -0.195 |
| puzzle | uniform_5 | 6.25% | 0.137 | 0.123 |
| puzzle | diversity_10 | 8.02% | 0.164 | -0.264 |

The similarity–error relationship is mixed and position-dependent; projected-view novelty is not a calibrated measure of independent recognition evidence. Nearby or geometrically similar views can repeat errors, but even different semantic sides can repeat the same piece bias. Conversely, two views within the same coarse camera-side bin can differ sharply: opening b1 is correctly a knight at **9.002 s** (diversity-10) and **18.003 s** (uniform-5), while the diversity-5 views all call it a rook. Camera-side novelty alone misses useful within-side changes in elevation, occlusion and piece appearance. Exact elevation/occlusion causes remain unverified hypotheses, not new annotations.

A direct proximity check is in [pair strata](evidence_d2_pair_summary.csv). Uniform-5's six proxy-similar pairs have mean defined error phi **0.374**, versus **0.270** in its other pairs; disagreement is **10.16% versus 18.08%**. Diversity-5 has no pairs below the registered 0.12 view-distance threshold, so a similar-pair estimate is unavailable there. Diversity-10's similar pairs do **not** show higher association (0.275 versus 0.293): geometric proximity is not a universal dependence model. Temporally nearby pairs (≤7 s, matching the D.1 sparse-motion diagnostic horizon) have mean phi **0.393 versus 0.299** for diversity-5 and **0.344 versus 0.280** for diversity-10; uniform has only one such pair, too little for a stable inference. These are pair-weighted descriptive strata with small, overlapping support; null/constant vectors are excluded explicitly.

Pair rows overlap frames and squares and are **not independent statistical samples**. Correlation is descriptive within each session, without p-values or frame-IID confidence intervals. Four positions overlap historical training placements; this remains one correlated development pilot, not an independent five-session generalization benchmark.

## Per-class and difficult-class behavior

Cells show **precision/recall percentages; ever-correct physical squares/support**. Per-class denominators differ across the variable-budget condition; full observation counts, per-position metrics, all confusion matrices and physical-square counts are exported in [per-class CSV](evidence_d2_per_class.csv) and the numeric report. Null precision/recall is used for unsupported classes in a session.

| Class | Diversity-5 P/R; physical coverage | Uniform-5 P/R; physical coverage | Diversity-10 P/R; physical coverage |
|---|---|---|---|
| empty | 98.5/97.4; 207/207 | 99.1/97.0; 207/207 | 98.4/95.2; 207/207 |
| white_pawn | 89.1/78.6; 29/29 | 87.1/83.4; 29/29 | 84.7/75.2; 29/29 |
| white_knight | 81.2/37.1; 6/7 | 84.2/45.7; 7/7 | 83.3/46.9; 7/7 |
| white_bishop | 54.5/34.3; 7/7 | 60.0/51.4; 7/7 | 58.1/37.9; 7/7 |
| white_rook | 37.0/100.0; 6/6 | 44.1/100.0; 6/6 | 36.5/98.3; 6/6 |
| white_queen | 62.5/33.3; 2/3 | 75.0/40.0; 3/3 | 52.4/37.9; 3/3 |
| white_king | 86.7/52.0; 5/5 | 66.7/64.0; 5/5 | 61.3/44.2; 5/5 |
| black_pawn | 80.4/79.3; 28/28 | 84.4/85.0; 28/28 | 83.7/76.3; 28/28 |
| black_knight | 87.0/57.1; 7/7 | 71.4/57.1; 6/7 | 72.5/56.1; 7/7 |
| black_bishop | 46.9/50.0; 6/6 | 46.9/50.0; 6/6 | 47.1/41.4; 6/6 |
| black_rook | 48.5/91.4; 7/7 | 55.9/94.3; 7/7 | 39.7/90.6; 7/7 |
| black_queen | 72.7/53.3; 3/3 | 80.0/53.3; 3/3 | 77.8/48.3; 3/3 |
| black_king | 35.1/52.0; 5/5 | 51.7/60.0; 5/5 | 29.3/51.2; 5/5 |

Black-rook recall is **32/35 (91.43%)** versus uniform's **33/35 (94.29%)**, with low precision **32/66 (48.48%)** versus **33/59 (55.93%)**. Diversity-10 gives **58/64 (90.63%)**, precision **58/146 (39.73%)**. All seven physical Black-rook squares have a correct view under every condition; more Black-rook predictions do not mean better evidence. These are a different input population from historical B.2 recall and should not be presented as a controlled improvement over its 41.67% historical result.

Examples: starting h8 is misread as Black king at **21.003 s**, while other views correctly identify the rook. Middlegame f8 varies among rook, knight, bishop and king; **33.005 s** wrongly predicts bishop with probability **0.952**. Strong uncalibrated confidence does not guarantee a useful identity observation. Wrong predictions with max probability≥0.9: diversity-5 **42**, uniform-5 **33**, diversity-10 **71**. No calibration transformation or abstention threshold was fitted from these counts.

## Diagnostics

[Multi-view examples and matrices](evidence_d2_examples/) include fifteen selected-frame contacts annotated with errors/occupied correctness, fifteen full 64-square view-by-view HTML matrices, and actual 224 RGB context crops with target-mask outlines and all thirteen probability bars. Overlays are display-only, after inference; packed input hashes identify their pixels.

- [Opening b1, diversity-5 persistent rook confusion](evidence_d2_examples/a3_pilot_01_02_opening_take01_persistent_b1.png): visual-side diversity produces no correct argmax.
- [Opening b1, added views rescue the knight](evidence_d2_examples/opening_b1_diversity_10_rescue.png).
- [Opening d1, added views rescue the queen](evidence_d2_examples/opening_d1_diversity_10_rescue.png): 9.002 s correct at .998 versus 21.003 s wrong rook at .992.
- [Endgame d5, diversity-5 rescue](evidence_d2_examples/endgame_d5_diversity_5_rescue.png): 3.000/9.002 s correctly identify the Black knight, while uniform-5 misses it in every selected view.
- [Starting h8 Black-rook behavior](evidence_d2_examples/a3_pilot_01_01_starting_take01_black_rook_h8.png).
- [Starting empty e6 false-positive views](evidence_d2_examples/a3_pilot_01_01_starting_take01_empty_false_positive_e6.png).
- [c7 original-label discrepancy](evidence_d2_examples/middlegame_c7_original_label_discrepancy.png): caption retains the original erroneous empty label for audit; the main analysis uses the approved pawn correction.

Absent example categories are explicitly recorded, not fabricated. Full contacts/matrices and per-candidate identifiers remain in the ignored analysis directory as well.

Per-frame errors, in original chronological order (see matrices/manifest for exact timestamps):

| Position | Diversity-5 errors/view | Uniform-5 errors/view | Diversity-10 errors/view |
|---|---|---|---|
| starting | 1, 16, 5, 28, 11 | 2, 10, 28, 7, 1 | 1, 2, 16, 5, 28, 12, 11, 24, 1 |
| opening | 7, 11, 23, 6, 10 | 7, 6, 21, 6, 10 | 7, 8, 14, 11, 26, 23, 9, 6, 6, 10 |
| middlegame | 1, 13, 26, 10, 6 | 3, 13, 13, 6, 12 | 1, 14, 20, 13, 31, 26, 6, 10, 6, 12 |
| endgame | 3, 3, 4, 4, 3 | 4, 4, 6, 4, 3 | 3, 3, 6, 4, 18, 12, 4, 3 |
| puzzle | 0, 4, 7, 1, 3 | 0, 4, 3, 3, 3 | 0, 4, 7, 1, 6, 3 |

## Runtime, storage and reproduction

CPU batch 16, four torch threads, one OpenCV thread. Original native videos/PNGs are unchanged; D.2 reads D.1's decoded HD PNGs and original-source hashes. HLG→BGR8 is the inherited A.3 convention without new tone mapping: no phone/browser equivalence or pure resolution inference is claimed.

Measured unique-frame work: context/crop extraction plus cache compression **54.17 s**, normalization **12.93 s**, model forward **19.12 s** (median **336 ms per 64-square frame**). Inference loop **89.33 s**; full run including model setup, serialization and final protected-file check **128.99 s**. Corrected characterization/rendering **16.98 s**; separate read-only input replay **33.39 s**. These are host observations, not real-time phone estimates or standalone per-strategy benchmarks.

The evidence run initially stores **422.43 MB**: **398.77 MB** lossless packed-input caches, **22.80 MB** JSONL observations, **0.86 MB** evidence manifest plus small summaries. Checkpoint remains **6,247,145 bytes**. The full geometry packet is **87.79 MB**; corrected analysis/diagnostics add about **21.18 MB**, before copied review exports. Archive ignored run assets with the original videos/D.1 run; the versioned manifest alone does not contain its linked caches/logits. No video/model binaries are automatically staged or committed.

From repository root, using the existing `.venv` and accessible frozen D.1 assets:

```bash
# For fresh captures/reruns, build a NEW packet and obtain its own bound human review.
.venv/bin/python scripts/prepare_evidence_review.py --output-dir docs/evaluation/evidence_d2_geometry_review_new

# Reproduce these approved 55 frames, using an unused output directory.
.venv/bin/python scripts/run_square_evidence.py --packet docs/evaluation/evidence_d2_geometry_review_v1/packet.json --reviews data/metadata/evidence_d2_geometry_reviews_v1.json --output-dir outputs/scanning/d2_reproduce
.venv/bin/python scripts/validate_square_evidence.py outputs/scanning/d2_reproduce/evidence.json --reproduce-inputs --report outputs/scanning/d2_reproduce/input_validation.json

# Original-label audit; then the owner-approved corrected-label comparison.
.venv/bin/python scripts/characterize_square_evidence.py --evidence outputs/scanning/d2_reproduce/evidence.json --output-dir outputs/scanning/d2_reproduce/original_labels
.venv/bin/python scripts/characterize_square_evidence.py --evidence outputs/scanning/d2_reproduce/evidence.json --ground-truth-corrections data/metadata/evidence_d2_ground_truth_corrections_v1.json --output-dir outputs/scanning/d2_reproduce/corrected_labels

.venv/bin/python -m pytest -q tests/test_square_evidence.py tests/test_video_keyframes.py tests/test_keyframe_manifest.py tests/test_crop_experiments.py tests/test_geometry_audit.py tests/test_geometry_reliability.py tests/test_scan_recognition.py tests/test_inference.py
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
```

Tools refuse existing run/analysis directories. Existing reviews cannot be applied to different pixels/grids/packet versions. New D.2 implementation files are outside the original packet's protected-input snapshot; inference/evaluation source hashes are recorded separately. The initial pending status/template and expected gate-rejection check remain as an audit trail, not the current review status.

[Validation and preservation details](evidence_d2_validation.json). Exact boards mean 64-square piece placement; side-to-move, castling rights and clocks are not inferred. Frozen D.2 source/config/test copies are archived under the run’s `source_snapshot/`.

Validation: **188 relevant Python tests passed** (including **36 focused D.2 tests**); **39 browser tests passed**; syntax and whitespace checks pass. Synthetic tests cover pending/stale/nonhuman reviews, all four orientations with bit-exact preprocessing parity, duplicates and renamed clones, 64-square serialization, error decomposition, unknown/no-view behavior, GT-only oracle semantics, paired coverage, dependent-pair summaries, end-to-end characterization and explicit-human ground-truth correction binding. Actual-data validation covers all 55 frames/3,520 observations/input replays. **3,966 protected files** hash-match; production, D.1 selections, original media, A.1/A.2/B.1/B.1b/B.2/B.3 artifacts/checkpoints remain unchanged.

## D.3 readiness — conclusions, not implementation

1. **Complementarity exists.** Many per-frame errors disappear from ever-correct coverage, including nine diversity-5 squares whose sole correct observation would be lost if treated as ordinary repeated consensus.
2. **Diversity-5 does not outperform uniform-5 here.** Keep uniform-5 as the strong equal-budget baseline; do not promote D.1's proxy winner based on zero similar-view pairs.
3. **Five views are not uniformly sufficient under the selection oracle.** Each five-view condition reaches 4/5 boards; diversity-10 reaches 5/5, at 72% more condition memberships for two additional squares.
4. Persistent five-view errors are opening b1/d1 for diversity, endgame d5 for uniform. None survives the diversity-10 oracle, but that result depends on correct ground truth; c7 illustrates why label audit matters.
5. View-dependent identity/occupancy changes are plausible D.3 targets. Persistent same-label bias across sides and high-confidence mistakes require more than counting diverse frames.
6. Similarity/error associations vary by position; coarse camera sides do not establish independent views. Retain temporal gaps, source/position/cohort identity, exact hashes, raw view descriptors and pairwise observed agreement for evaluation.
7. Candidate D.3 metadata: human geometry status/policy, source/target/context coverage, board size/sharpness/perspective, temporal proximity, duplicate identity, projected-view descriptors, full logits/probabilities, entropy/margin and explicit calibration status. Occlusion/elevation are unknown and cannot be invented as weights. D.1 quality scores are not calibrated recognition reliability.
8. **Proceed to controlled offline D.3 experiments after review**, preserving uniform-5/diversity-5 comparisons and the coverage-sensitivity condition. Approved B.4 calibration remains outstanding for calibrated weighting; raw logits allow later application without image inference. Fresh independent reviewed positions/sessions remain necessary before production claims.

**Probabilities from correlated video frames must not automatically be treated as conditionally independent evidence.** D.2 does not multiply probabilities, vote, average class distributions, select an operational fused label, or implement scan completion. Oracle results use ground truth only for evaluation and must not be fed into D.3 inference.

## Files changed and acceptance checklist

All D.2 changes are new files: `src/scanning/square_evidence.py`; `scripts/prepare_evidence_review.py`, `run_square_evidence.py`, `validate_square_evidence.py`, `characterize_square_evidence.py`; `configs/experiments/evidence_d2.json` and observation schema; `tests/test_square_evidence.py`; this report/design/CSV (including pair strata)/numeric exports/example and geometry-review folders; `data/metadata/evidence_d2_geometry_reviews_v1.json`, `evidence_d2_ground_truth_corrections_v1.json`, `evidence_d2_manifest_v1.json`, `evidence_d2_index_v1.json`, historical pending `evidence_d2_status_v1.json` and current `evidence_d2_status_v2.json`. Large generated inference/cache/analysis assets remain under ignored `outputs/scanning/d2_pilot_run1/`.

- [x] Deduplicated primary union, full overlays, explicit human verification and all four checks; separate completion/review/rejection/unresolved states; no substitutions.
- [x] Traceable 64-square evidence, every raw class logit/probability and uncertainty output, exact B.2 input/model identity; no forced resolved label or pooled distribution.
- [x] Fixed D.1 strategies, equal-budget main comparison and actual-count diversity-10 sensitivity; no selection tuning.
- [x] Ordinary/per-class/Black-rook metrics, per-frame errors/exact boards, ever-correct coverage, multiplicity, error overlap/disagreement/association and clearly scoped oracle diagnostics.
- [x] Corrected label separately owner-approved; original FEN/artifacts/results preserved; no automatic photograph labeling.
- [x] Per-position matrices/contacts, real multi-view crop/probability examples, runtime/storage, reproducible scripts and tests.
- [x] Development/session grouping retained; no final holdout, training, B.4 fitting, model replacement, production/default-recovery/browser change, fusion or completion logic.
- [x] Protected artifacts preserved; stop after D.2. **No commit or push.**
