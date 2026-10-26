# D.3 — Controlled offline multi-view fusion

Completed 2026-10-22. Development experiment only; five physical sessions, no independent generalization estimate. **Uniform-5 mean logits achieved 311/320 correct squares, 104/113 occupied squares and 1/5 exact boards.** It improves classification over the ground-truth-blind single-frame baseline, but exact-board count remains 1/5 and a confidently missed pawn defeats the diagnostic abstention rule. Do not enable automatic scan completion from this evidence.

## Boundaries and provenance

This experiment consumes D.2's existing logits/probabilities; no video decoding, crop regeneration, classifier inference, retraining, geometry changes, production integration or holdout access. D.1 selection remains unchanged. All 55 frames retain explicit human geometry verification, including corners, orientation, canonical mapping and crop alignment. This is qualitative human verification, not a quantitative corner-error bound.

Validated 3,520 unique observations and their class order, source/frame/session identities, model/configuration identities, finite logits and normalized probabilities, source hashes, geometry and review bindings. Explicitly checked all 3,520 camera-index-to-canonical-square and orientation mappings. Only observations of the same physical square, session, source and configuration may aggregate. There were **zero exclusions**. Invalid records fail closed; unreviewed, rejected or unresolved geometry is excluded explicitly. No frame replacement.

Primary labels use the registered source catalog plus the owner-confirmed versioned c7 correction: middlegame rank 7 `1b1nbpp1` becomes `1bpnbpp1`. The correction SHA-256 is `bd68d33dacd2dfc77ec322da97d953cc9d89bf7a99ace11d39b1786956a57202`. Original catalog, FENs and D.2 results remain untouched. Label provenance and all five evaluated FENs are in the results. Ground truth is loaded only by the separate evaluator after predictions exist. No target-position knowledge enters aggregation or frame selection.

B.2 tighter-224 checkpoint SHA-256 remains `dad8eea28482ee894dfca0794d9f4415d21ac58db23c6d6bed8c4307c71b1bd8`. No compatible approved B.4 calibration artifact was found; **all scores are uncalibrated**. No new calibration parameters were fitted.

## Prespecified methods

See [experimental design](fusion_d3_design.md) and [configuration](../../configs/experiments/fusion_d3.json). Parameters were fixed before inspecting D.3 results; there was no tuning or search. Each method outputs the complete 13-class score vector and square-level disagreement/traceability diagnostics.

| Method | Definition |
|---|---|
| Single quality | One whole-board frame with maximum D.1 quality; ties use earliest timestamp then frame identity. No per-square best-frame selection. |
| Majority vote | Argmax class counts; tied counts use mean input probability among tied classes, then lowest class ID. Vote fractions are scores, not calibrated probabilities. |
| Mean probabilities | Arithmetic mean of original 13-class softmax distributions. |
| Mean logits | Arithmetic mean of original logits, then numerically stable softmax at temperature 1. This is a mean, not a sum. |
| Clustered quality median | Chronological complete-link groups of similar/nearby views; quality-weighted probability mean inside each group, coordinate-wise median across groups, then normalization. Zero-mass median falls back to equal-group mean. |

Grouping joins a frame only if it is close to every existing member: D.1 view-proxy distance below 0.12 **or** timestamp separation at most 7 seconds. Quality weights are clipped to [0.25, 1]. These heuristic groups discount redundancy; they are not independent samples or reliable physical camera poses. Mean logits is mathematically related to a normalized geometric mean of probabilities, but no conditional-independence interpretation is assumed. Naïve probability multiplication is not used.

Diagnostic withholding uses fixed maximum-score <0.8, margin <0.2, fewer than two supporting groups, less than two-thirds supporting groups, or contradictory input labels each with score ≥0.8. Single-frame withholding uses only its score and margin. These are exploratory rules, not an approved scan-completion policy. Each square retains tentative argmax, withholding reasons, group distributions, support, vote counts, entropy, margin and Jensen–Shannon disagreement. Withheld operational class is null. No board completion logic is implemented.

## Comparable experimental conditions

Uniform-5 is primary; diversity-5 is the equal-budget comparator. Both use 25 frame memberships over five sessions. Diversity-10 uses **43 actual memberships** (9, 10, 10, 8, 6 per session), 72% more than 25; it is a coverage sensitivity condition, not an equal-cost comparison. There are 55 distinct frames across all strategies.

Each result below is one tentative 64-square board per session, pooled over five boards: 320 squares, 113 occupied and 207 empty. **Exact boards use all tentative argmaxes, including withheld squares; they are not accepted scans.** D.2's per-view averages are a different aggregation unit and are not the single-frame baseline below.

## Recognition results

| Condition | Method | Overall | Occupied | Exact boards | Empty FP | Occupied→empty | Wrong piece identity |
|---|---|---:|---:|---:|---:|---:|---:|
| Uniform-5 | Single quality | 95.00% | 85.84% | 1/5 | 0 | 2 | 14 |
| Uniform-5 | Vote | 95.63% | 87.61% | 0/5 | 0 | 1 | 13 |
| Uniform-5 | Mean probabilities | 96.56% | 90.27% | 0/5 | 0 | 1 | 10 |
| Uniform-5 | Mean logits | **97.19%** | **92.04%** | **1/5** | 0 | 1 | 8 |
| Uniform-5 | Clustered median | 96.56% | 90.27% | 1/5 | 0 | 1 | 10 |
| Diversity-5 | Single quality | 93.13% | 80.53% | 0/5 | 0 | 2 | 20 |
| Diversity-5 | Vote | 94.38% | 84.07% | 0/5 | 0 | 1 | 17 |
| Diversity-5 | Mean probabilities | 95.31% | 86.73% | 0/5 | 0 | 1 | 14 |
| Diversity-5 | Mean logits | 95.94% | 88.50% | 1/5 | 0 | 1 | 12 |
| Diversity-5 | Clustered median | 95.00% | 85.84% | 0/5 | 0 | 1 | 15 |
| Diversity-10 | Single quality | 93.13% | 80.53% | 0/5 | 0 | 2 | 20 |
| Diversity-10 | Vote | 96.88% | 91.15% | 0/5 | 0 | 1 | 9 |
| Diversity-10 | Mean probabilities | 97.19% | 92.04% | 1/5 | 0 | 1 | 8 |
| Diversity-10 | Mean logits | 96.56% | 90.27% | 1/5 | 0 | 1 | 10 |
| Diversity-10 | Clustered median | 95.00% | 85.84% | 0/5 | 0 | 1 | 15 |

Uniform-5 mean logits improves by 7 squares over its blind single frame, 5 over vote, and 2 over mean probabilities. It does not improve exact-board count over the single-frame baseline. Diversity-5 is worse on overall/occupied accuracy for every matched method; reduced D.1 proxy redundancy did not translate into superior fusion here. Diversity-10 mean probabilities matches uniform-5 mean logits's counts at greater input cost. The robust method does not consistently improve on simple aggregation.

### Paired physical-session results

Errors per 64-square board, uniform-5; lower is better:

| Position/session | Single | Vote | Mean probabilities | Mean logits | Clustered median |
|---|---:|---:|---:|---:|---:|
| Starting | 2 | 2 | 1 | **0** | **0** |
| Opening | 7 | 4 | 3 | 3 | 4 |
| Middlegame | 3 | 4 | 4 | 3 | 4 |
| Endgame | 4 | 3 | 2 | 2 | 2 |
| Puzzle | **0** | 1 | 1 | 1 | 1 |

Thus fusion rescues the starting position while losing the previously exact puzzle. [All 75 per-board results](fusion_d3_per_board.csv) and 225 paired session/method/reference comparisons in the machine-readable results retain actual rescued and broken square coordinates, including comparisons with uniform-5 mean probabilities and vote.

### Difficult classes

[Per-class precision/recall](fusion_d3_per_class.csv) contains all 195 condition/method/class rows. Full 13×13 confusion matrices are in the summary and full results; per-board matrices are also retained. Uniform-5 single/mean-probability/mean-logit black-rook recall and precision are 7/7 (100%); vote and clustered median retain recall but introduce a false rook (precision 7/8). Diversity-5 single misses one of seven black rooks; its fused methods recover recall. Seven examples cannot establish robust rare-class performance.

No method introduces an empty-square false positive on these five boards. Occupied-to-empty errors fall from two to one with fusion, but that remaining miss is highly confident. Knights, bishops and queens remain important identity confusions; the full matrices expose these rather than reducing recognition to occupied/empty detection.

## Confidence, disagreement and abstention

Uniform-5, 320 tentative predictions:

| Method | ECE, all squares | ECE, occupied only | Wrong score ≥0.9 | Retained by diagnostic rule | Retained errors |
|---|---:|---:|---:|---:|---:|
| Single | .014 | .043 | 3 | 293 | 4 |
| Vote | .055 | .101 | 0 | 263 | 3 |
| Mean probabilities | .093 | .192 | 0 | 241 | 1 |
| Mean logits | .020 | .059 | 2 | 266 | 4 |
| Clustered median | .034 | .090 | 3 | 257 | 2 |

ECE uses 10 fixed bins and is descriptive, not evidence of calibration. The 207 empty squares dominate pooled reliability. Uniform-5 mean logits has NLL .122 and multiclass Brier .051, versus .188/.085 for mean probabilities and .177/.076 for single. Vote zeros use an explicit 1e-12 NLL floor and report clipped counts. Comparing these different score constructions requires caution. Full reliability bins, selective coverage/error rates and occupied-only supplement are preserved.

Mean logits's two extra correct squares over mean probabilities come with two versus zero errors scoring ≥0.9, and four versus one errors retained by the diagnostic withholding rule. There is no clear winner across accuracy and reliability risks.

### Concrete visual evidence

[Diagnostic index](fusion_d3_examples/index.html) includes all 15 condition/session board comparisons and per-square tables. Images show the **actual frozen B.2 RGB/context inputs with mask**, original full class distributions and method score vectors. Examples were selected after evaluation for explanation only.

- [Starting g1 rescue](fusion_d3_examples/starting_g1_rescue.png): only one of five input argmaxes is the correct white knight. Mean logits recovers it at score .485, while mean probabilities chooses bishop. It is correctly withheld by the diagnostic rule. This is genuine complementary evidence; the starting board becomes tentative-exact despite no exact individual uniform-5 frame.
- [Opening c7 rescue](fusion_d3_examples/opening_c7_rescue.png): the quality-selected frame predicts black bishop; four other views correctly show black pawn and aggregation corrects it. This opening-square example is separate from the middlegame c7 annotation correction.
- [Puzzle d7 regression](fusion_d3_examples/puzzle_d7_regression.png): the single frame correctly recognizes white knight, but three wrong rook argmaxes defeat two correct knight views. All aggregation methods choose rook; withholding catches the disagreement.
- [Endgame h4 confident miss](fusion_d3_examples/endgame_h4_confident_miss.png): four highly confident empty predictions overwhelm the one correct white-pawn observation (score .644). Mean logits predicts empty near 1.0 and **does not abstain**. The weaker correct view falls below the strong-conflict cutoff. Group discounting also fails; several viewpoints repeat the bias. Reviewed crop imagery visibly contains the pawn, but this experiment does not establish a new geometry defect.
- [Opening d1 correct minority lost](fusion_d3_examples/opening_d1_minority.png): one correct white-queen observation is overwhelmed; withholding catches uncertainty. [Opening b1 repeated error](fusion_d3_examples/opening_b1_repeated_bias.png) shows diversity-5 views all missing the correct identity.

The evaluator records 139 method/condition/square instances where a correct minority argmax is overwhelmed; these repeat physical squares and are not independent failure samples. Mean-logit uniform-5 retained errors are opening c5, middlegame e3, endgame d5 and h4. Disagreement helps detect some failures but cannot detect uniformly biased predictions reliably.

## Chess legality, separately

No piece counts, legal history, known position or chess rules change predictions. Existing validation is reported under both assumed turns, without castling/en-passant history. This is a limited plausibility diagnostic, not legal-position reconstruction. Uniform-5 single/vote/mean-probabilities/mean-logits/clustered median yield 2/2/3/3/3 boards valid under at least one assumed turn; mean logits has only one correct board despite three plausible boards. No repair experiment was performed.

## Oracle diagnostics — ground-truth upper bounds only

D.2-style ever-correct argmax coverage is reproduced separately: uniform-5 319/320 with 4/5 oracle-exact boards; diversity-5 318/320 with 4/5; diversity-10 320/320 with 5/5. These are **not achieved fusion**. They describe ground-truth selection among existing hard predictions, not a universal mathematical bound for methods using complete distributions. The additional best-individual-frame diagnostic uses ground truth explicitly and is never the single-frame baseline. None of these diagnostics influences prediction, parameters or frame choice.

## Runtime and storage

Aggregation reads cached observations only. With full per-square validation, traceability and disagreement output, median per-board CPU costs across three repeats per case were approximately: single 91 ms, vote 75 ms, mean probabilities 79 ms, mean logits 82 ms, clustered median 94 ms. These prototype audit costs exclude file loading, classifier inference and phone execution; they are not optimized deployment estimates. All 225 benchmark replays reproduce predictions exactly.

The instrumented complete experiment took 76.18 s including validation; fusion plus reverse-order reproducibility checks took 55.13 s. Python allocation-tracing peak was 31,694,544 bytes after evidence loading, excluding preloaded data/native allocations. Predictions JSON is 33,839,902 bytes; evaluated JSON is 36,158,328 bytes. The portable lossless compressed results plus summary, CSVs, costs and diagnostic packet total about 24.2 MB. No source videos or checkpoints are duplicated into the report.

## Reproduction and validation

From repository root with the existing environment and protected local D.2 caches, choose **fresh output paths**; scripts refuse replacement:

```bash
.venv/bin/python scripts/run_fusion_experiments.py --config configs/experiments/fusion_d3.json --output-dir outputs/scanning/fusion_d3_reproduction
.venv/bin/python scripts/evaluate_fusion_experiments.py --run outputs/scanning/fusion_d3_reproduction --output-dir outputs/scanning/fusion_d3_reproduction/evaluation_v1
.venv/bin/python scripts/render_fusion_diagnostics.py --evaluation outputs/scanning/fusion_d3_reproduction/evaluation_v1/results.json --output-dir outputs/scanning/fusion_d3_reproduction/diagnostics_v1
.venv/bin/python scripts/benchmark_fusion.py --run outputs/scanning/fusion_d3_reproduction --output outputs/scanning/fusion_d3_reproduction/cost_v1.json
.venv/bin/python scripts/export_fusion_results.py --run outputs/scanning/fusion_d3_reproduction --output-dir outputs/scanning/fusion_d3_reproduction/export_v1
.venv/bin/python scripts/validate_fusion_results.py --run outputs/scanning/fusion_d3_reproduction --export-dir outputs/scanning/fusion_d3_reproduction/export_v1 --output outputs/scanning/fusion_d3_reproduction/final_validation.json
.venv/bin/python -m pytest -q tests/test_offline_fusion.py tests/test_square_evidence.py tests/test_video_keyframes.py tests/test_keyframe_manifest.py tests/test_crop_experiments.py tests/test_geometry_audit.py tests/test_geometry_reliability.py tests/test_scan_recognition.py tests/test_inference.py tests/test_validation.py tests/test_fen.py
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
```

Python: **234 passed**, including 31 focused D.3 tests. Browser: **39 passed**. Focused checks cover coordinate/session/config alignment, tie breaks, aggregation definitions, input permutation determinism, duplicates/invalid observations, grouping, repeated observations, withholding failures, legality separation and original/corrected label handling. There are 75 reverse-order experiment replays and 225 benchmark replays. See [validation record](fusion_d3_validation.json) for final byte-preservation, export, compilation and metric checks.

## Artifacts and files changed

All D.3 additions; no existing tracked files modified:

- Core: `src/scanning/offline_fusion.py`, `src/scanning/fusion_metrics.py`.
- Scripts: `run_fusion_experiments.py`, `evaluate_fusion_experiments.py`, `render_fusion_diagnostics.py`, `benchmark_fusion.py`, `export_fusion_results.py`, `validate_fusion_results.py` under `scripts/`.
- Configuration: `configs/experiments/fusion_d3.json`; tests: `tests/test_offline_fusion.py`.
- This report, design, comparison/per-board/per-class CSVs, summary, lossless results `.json.gz`, cost/export/validation JSONs and `fusion_d3_examples/` under `docs/evaluation/`.
- [Versioned artifact index](../../data/metadata/fusion_d3_index_v1.json); local reproducibility details under `outputs/scanning/fusion_d3_run1/`.

The portable [full results](fusion_d3_results_v1.json.gz) include all 75 boards and 4,800 method/square diagnostics with complete score distributions, source observation IDs, grouping, uncertainty, truth/evaluation fields and confusion matrices. [Readable summary](fusion_d3_summary_v1.json) omits only the verbose square lists. Original D.2 logits/probabilities remain in their protected source artifact, referenced by identity/hash rather than copied or altered.

## Acceptance and decision gate

- [x] Frozen evidence validated; canonical alignment, source/session/geometry and label version explicit; zero exclusions.
- [x] Five transparent, deterministic, ground-truth-blind methods; prespecified rules, no tuning.
- [x] Three comparable conditions; extra-frame cost distinguished.
- [x] Per-session/class/confusion/exact-board metrics; paired improvements and regressions retained.
- [x] Confidence, disagreement, abstention and high-score failures exposed; no invented calibration.
- [x] Legality and oracle diagnostics separate from primary predictions; no repairs.
- [x] Machine-readable traceability, visual examples, runtime/storage and tests supplied.
- [x] Historical artifacts/checkpoint/production preserved; no holdout, training, browser changes, commit or push.
- [x] D.3 only; no D.4 scan-completion implementation.

**Recommendation:** retain uniform-5 mean logits as an offline accuracy candidate, with uniform-5 mean probabilities as the simpler, less confident-error-prone comparator. Neither is approved for production or completion decisions. Fusion demonstrates useful complementarity, but it does not improve aggregate exact-board count over the blind single frame, and its confident miss survives the proposed withholding rule. Revise reliability evaluation within D.3 and collect more independent, geometry-reviewed sessions before using D.4 as an automatic completion gate. Evaluate any future approved B.4 calibration out of session; do not fit it to these five recordings. Preserve black-rook, knight/rook, bishop/pawn and boundary/empty-confusion examples in future evaluation.

Only five correlated physical sessions are available; four placements overlap historical training data. Frames, squares and repeated crops are not independent experimental samples. Report paired session outcomes, not significance claims. Correlated video probabilities must never automatically be treated as independent evidence. This task establishes an auditable offline baseline, not generalization or reliable full-board reconstruction.
