# D.3c — Limited independent-session evaluation and assisted MVP transition

2026-10-26. **Complete as a limited, provisional interim evaluation—not a passed D.3b preregistered validation gate.** All ten recordings were processed. Quality-selected single frame and mean-logit fusion both achieve **84.53% overall accuracy and 0/10 exact boards** against the supplied FENs. The next milestone is an assisted, editable scan-to-analysis flow; automatic completion, retraining and further recording collection are not prerequisites.

## Cohort registration, truth and protocol integrity

The requested research/roadmap documents actually reside at repository root, `VISION_RESEARCH.md` and `IMPLEMENTATION_PLAN.md`. Read these and D.1–D.3b reports, source/result manifests and actual decoding, geometry, selection, inference and evaluation implementation before adding the adapter. All historical files remain unchanged.

Discovered ten originals in `data/raw/new_test/`: `pos_001.MOV`, `pos_003.MOV`, `pos_005.MOV`, `pos_007.MOV`, `pos_009.MOV`, `pos_011.MOV`, `pos_013.MOV`, `pos_015.MOV`, `pos_017.MOV`, `pos_019.MOV`. Their numbers unambiguously select FENs 01, 03, 05, 07, 09, 11, 13, 15, 17, 19 from the owner's list. No file was guessed or excluded for ambiguous mapping. All 20 supplied FENs are preserved separately, parse structurally and pass python-chess standard-position validity checks. This validates syntax/legality, not photographic agreement or historical reachability.

The [versioned cohort manifest](../../data/metadata/fusion_independent_d3c_cohort_v1.json) records source paths, SHA-256, bytes, session/position/correlation groups, full FENs, native metadata, selected PTS, geometry references and verification states. New source hashes differ from prior pilot sources. **Position 03 exactly matches historical `position_003` placement**, despite being a genuinely new recording. Do not call it a novel-position generalization sample. Report all ten sessions and a separate nine-novel-placement subset.

There are **no matching reference photographs** in the uploaded cohort directory. The owner's unchanged-position declaration is recorded; it does not independently verify each piece/FEN or all frames. No model predictions establish or correct ground truth. Assistant inspection found the ten representative initial grid overlays broadly consistent with printed coordinates, and inspected ten five-view target-crop examples. This is limited visual diagnosis, **not attributed human verification of all fifty grids or full photo/FEN agreement**. No labels or geometry were automatically certified or corrected. Strict verified-geometry/photo-qualified primary sessions: **0/10**. Algorithm-completed five-frame provisional sessions: **10/10**.

The original D.3b protocol calls for ten new positions over at least two dates with lighting variation, references and human geometry review. This cohort has ten recordings but only nine novel placements, one native-metadata date, one living-room/nighttime/artificial-light setup, no references and pending human review. The original protocol/config/preregistration were preserved. The interim execution design and new source registration were saved **before recognition outcomes**. Its deviations are explicit; the strict gate did not pass. Do not equate 640 physical squares, 3,200 observations or fifty frames with hundreds of independent sessions. No untouched final holdout was accessed.

### Native media validation

All ten fully decode: **21,205 frames**, approximately **30 fps**, HEVC, **1920×1080 encoded / 1080×1920 displayed**, constant **−90°** display rotation. Actual rational PTS/time bases are preserved, not estimated from nominal FPS. Native creation-time tags span 2026-10-26 00:00:59–00:59:11 UTC; tags support the declared capture period but are not independently verified capture clocks. Device metadata is retained in the manifest rather than guessed.

| Position | Native frames | PTS span (s) | Legacy completed | Recovery completed | Uniform selected |
|---|---:|---:|---:|---:|---:|
| 01 | 2,328 | 77.58 | 2/26 | 16/26 | 5 |
| 03, placement overlap | 1,990 | 66.31 | 9/23 | 19/23 | 5 |
| 05 | 2,077 | 69.21 | 8/24 | 14/24 | 5 |
| 07 | 2,101 | 70.01 | 0/24 | 13/24 | 5 |
| 09 | 1,980 | 65.98 | 0/22 | 10/22 | 5 |
| 11 | 2,062 | 68.71 | 0/23 | 14/23 | 5 |
| 13 | 1,868 | 62.24 | 0/21 | 9/21 | 5 |
| 15 | 2,155 | 71.81 | 0/24 | 9/24 | 5 |
| 17 | 2,255 | 75.14 | 24/26 | 25/26 | 5 |
| 19 | 2,389 | 79.61 | 23/27 | 25/27 | 5 |

Frame rotation and exact original pixels were reproduced for every selected frame. The adapter's geometry-binding derivative digest denotes decoded BGR8 pixels; it is explicitly an interim pixel binding, not a claim that a persistent PNG with that file hash exists. Temporary lossless PNGs are only an interface to existing path-based geometry code.

## Frozen pipeline and minimal adapter

The existing `evaluate_independent_fusion.py register/evaluate` CLI consumes **reviewed D.2 evidence**, not raw MOV files. It correctly requires reference and human attestations absent here; it was not weakened. The old pilot importer/secondary comparison paths are also pilot-specific and would create unnecessary PNG derivatives.

New offline-only adapter:

1. Stream-decode and index native PTS using the existing isolated PyAV decoder and tested `display_image`/`display_transform`; no video transcoding/copy.
2. Call unchanged D.1 `temporal_candidates`: nearest original PTS on a **3,000 ms** grid, same candidate limit. On **240 candidates**, invoke unchanged opt-in B.1b wrapper, including its original legacy attempt. Keep thresholds/configuration unchanged.
3. Reuse unchanged finite/in-frame canonical geometry safeguards, board quality features, uniform-5 selector and hard near-duplicate guard. All 154 completions passed those quality gates; this does not prove absence of blur or occlusion. All ten sessions yield five selected frames; no after-the-fact replacement.
4. Re-decode selected native PTS, verify original pixel hashes and call frozen D.2 `make_inputs`: **800-pixel board, 200-pixel context margin, original 200-pixel initial padding**, B.2 `context1_224` (`padding_squares=1`, 224 input), unchanged packing/normalization and .99 target-coverage gate. Keep camera-relative RGB; semantic mapping changes IDs/masks, not piece appearance. All 3,200 crops pass coverage.
5. Run unchanged B.2 MobileNetV3 checkpoint `outputs/evaluation/crop_b2_20261018_run1/training/context1_224/best.pt`, SHA-256 **`dad8eea28482ee894dfca0794d9f4415d21ac58db23c6d6bed8c4307c71b1bd8`**. Preserve all logits and original float64 softmax probabilities, 13-class order, mask/input hashes and source provenance. No training or calibration.
6. Execute original D.3 single-quality / arithmetic mean-probability / arithmetic mean-logit→softmax functions and diagnostic abstention. To evaluate unverified inputs honestly, `interim_evidence.py` explicitly scopes **only the human-review input gate waiver** in this standalone process, retaining `completed_unverified` in every record. It never assigns `human_verified`. The original validator is restored on exit, including exceptions; all numeric checks, grouping, scores, ties and abstention thresholds remain unchanged. Tests prove exact equivalence with the strict frozen kernel on verified D.2 inputs. These outputs cannot enter a strict verified population.

This review waiver is a declared protocol deviation, not promotion of recovery, a numerical validation relaxation or a production feature. Persistent source files for D.3b/fusion/inference were not edited. The scoped adapter is not appropriate for a server/runtime integration.

## Provisional recognition results

Metrics are **conditional on supplied FENs and completed-but-unverified geometry**, not unbiased classifier accuracy on verified crops. Each method returns one tentative board/session. Exact boards include tentative labels even where the frozen rule abstains; they are not accepted scans.

| All ten sessions | Overall | Occupied | Empty false positives | Occupied→empty | Wrong piece identities | Exact |
|---|---:|---:|---:|---:|---:|---:|
| Quality single | 541/640, **84.53%** | 200/268, **74.63%** | 31/372, 8.33% | 4/268 | 64 | 0/10 |
| Mean probabilities | 530/640, **82.81%** | 192/268, **71.64%** | 34/372, 9.14% | 0/268 | 76 | 0/10 |
| Mean logits | 541/640, **84.53%** | 201/268, **75.00%** | 32/372, 8.60% | 0/268 | 67 | 0/10 |

Nine novel placements only, excluding 03: single **484/576 (84.03%)**, occupied **177/238 (74.37%)**; mean probabilities **475/576 (82.47%)**, occupied **171/238 (71.85%)**; mean logits **487/576 (84.55%)**, occupied **180/238 (75.63%)**. All exact counts remain **0/9**. This is still a single-setting provisional cohort.

### Paired session outcomes

Errors per 64-square board; occupied-correct numerator/support in parentheses. No failed session is hidden.

| Position | Quality single | Mean probabilities | Mean logits |
|---|---:|---:|---:|
| 01 | 4 (29/32) | 12 (24/32) | 12 (25/32) |
| 03 | 7 (23/30) | 9 (21/30) | 10 (21/30) |
| 05 | 6 (28/32) | 11 (25/32) | 8 (27/32) |
| 07 | 24 (16/32) | 25 (18/32) | 26 (17/32) |
| 09 | 12 (24/31) | 16 (21/31) | 10 (25/31) |
| 11 | 8 (23/30) | 10 (21/30) | 6 (25/30) |
| 13 | 16 (20/31) | 8 (26/31) | 6 (26/31) |
| 15 | 13 (19/28) | 12 (18/28) | 16 (15/28) |
| 17 | 4 (10/12) | 3 (11/12) | 2 (11/12) |
| 19 | 5 (8/10) | 4 (7/10) | 3 (9/10) |

Mean logits rescues **37** squares relative to quality single and regresses **37**: no net overall gain. Mean probabilities rescues **31** and regresses **42**. Aggregation is position-dependent and does not improve exact-board reconstruction. Do not report extra views as automatically beneficial.

### Per-class errors

Counts of true-class squares predicted incorrectly. [Per-class precision/recall CSV](fusion_independent_d3c_per_class_v1.csv) and full confusion matrices preserve denominators and destinations.

| True class | Support | Single errors | Mean-P errors | Mean-logit errors |
|---|---:|---:|---:|---:|
| Empty | 372 | 31 | 34 | 32 |
| White pawn | 71 | 9 | 16 | 16 |
| White knight | 16 | 13 | 15 | 14 |
| White bishop | 14 | 6 | 8 | 7 |
| White rook | 16 | 0 | 0 | 0 |
| White queen | 8 | 3 | 5 | 4 |
| White king | 10 | 4 | 5 | 6 |
| Black pawn | 70 | 1 | 0 | 0 |
| Black knight | 16 | 10 | 8 | 6 |
| Black bishop | 13 | 6 | 8 | 6 |
| Black rook | 16 | 4 | 2 | 1 |
| Black queen | 8 | 5 | 6 | 5 |
| Black king | 10 | 7 | 3 | 2 |

White-rook recall of 100% hides a strong bias: mean logits predicts **77 white rooks**, of which only **16** are correct (20.78% precision). Black-rook recall rises from 12/16 single to 15/16 mean logits, but that improvement does not offset other class/empty-square failures. White knights are particularly weak. Small supports and unverified geometry/labels prevent strong generalization or causal training claims.

## Reliability and additional views

| Method | Wrong squares with score ≥.9 | Squares retained by frozen rule | Retained errors | Boards retained | Incorrect retained boards |
|---|---:|---:|---:|---:|---:|
| Single | 19 | 506/640 | 33 | 0/10 | 0 |
| Mean probabilities | 2 | 223/640 | 6 | 0/10 | 0 |
| Mean logits | 17 | 322/640 | 14 | 0/10 | 0 |

These different square coverages are not a matched-risk comparison. Requiring all 64 squares retained rejects every board; zero incorrect retained boards at zero coverage is not a useful automatic gate. No method's incorrect board has mean confidence ≥.98 in this cohort, but the maximum incorrect single-board mean is **.9626** and many individual wrong labels exceed .9. That does not undo D.3b's >98% wrong-board examples.

All uncalibrated probability vectors, margins, entropy, vote variation, JS disagreement, support groups and contradictory strong-input flags remain in the results. Mean-logit ECE **.0325**, NLL **.4732**, Brier **.2322**, versus single **.0556/.5302/.2307** and mean probabilities **.1306/.5945/.2841**. ECE and softmax confidence do not make predictions safe, particularly with pooled empty squares and correlated observations. No B.4 calibration was invented or fitted.

A **prespecified chronological-prefix sensitivity analysis** reuses only the same five selected frames, with no extra inference or parameter tuning:

| Chronological views | Mean-P correct /640 | Mean-logit correct /640 |
|---|---:|---:|
| 1 | 493 | 493 |
| 2 | 540 | 544 |
| 3 | 532 | 547 |
| 4 | 529 | 537 |
| 5 | 530 | 541 |

Later views can both resolve and introduce errors. From four to five mean-logit views, 15 squares improve and 11 regress; one previously withheld square becomes retained correctly and one becomes retained incorrectly. All prefixes have **0/10 exact boards**. Do not tune a three-frame budget from the apparent peak here. The primary quality-selected single frame already reaches 541 correct squares, so comparing fusion only against the weaker first chronological frame would overstate its value. Prefix transition counts overlap physical squares and are descriptive, not independent trials.

## Failure diagnosis and visual evidence

[Compact diagnostic packet](fusion_independent_d3c_examples/index.html): ten selected-view contacts, ten truth-versus-prediction boards and ten five-view **actual masked-input crops with all 13 probabilities**. Display-only outlines are drawn after inference. Fifty diagnostic crop replays exactly match their saved packed-input hashes. The run also retains fifty bounded grid-review JPEGs; fine geometry remains unverified.

For **every** incorrect board, the cohort/results preserve the same stage distinction:

- **Upstream localization:** 81/240 candidates fail final localization, and 5/240 fail final semantic orientation. Legacy completes 66/240 (27.50%); opt-in recovery completes 154/240 (64.17%). There are no processing/corner-order exceptions in this cohort. The wrapper makes 174 recovery attempts; the legacy localization gate accepts 86 proposals, only 66 of which complete the full legacy pipeline. This distinction prevents counting accepted polygons as correct completed boards.
- **Selected geometry/orientation:** all five selected frames per session pass unchanged structural and machine orientation checks. Actual correctness is unverified; inspected representative overlays show no gross wrong polygon or canonical orientation, but that cannot certify all fifty. Correct localization/orientation rates against human corner/orientation ground truth are **not available**, not 100%.
- **Square crops:** all 3,200 targets meet the unchanged coverage test; fifty representative diagnostic inputs reproduce exactly. No implementation mapping/mask defect was confirmed. This does not establish that every tall piece is visually confined to its board-plane square or unoccluded.
- **Recognition on plausibly aligned examples:** empty h1 contexts in 05/13/15 can be labeled rook despite the target tile being visibly empty; the nearby king and its projected head/body influence the context. Position 17 f3's supplied white king is called a rook in several clear views, including .992 raw confidence. These demonstrate conditional recognition/context failures, not a justification to change FEN labels automatically.
- **Fusion:** session-specific rescued/regressed square lists are retained. It preserves correlated rook biases and can break a better individual view. Position 13 benefits substantially (16→6 errors); 01 regresses (4→12); neither effect warrants a global claim that fusion is better.
- **Uncertainty:** strong wrong consensus, confidently wrong labels and false pieces survive at square level; board-level withholding has zero useful coverage. Both ordinary errors and withheld states are visible. No confidence-driven automatic completion or legal repair is performed.

All ten have incorrect tentative boards under every method; unresolved causal attribution remains explicit where geometry/FEN agreement is unverified. Position 07's large error burden and positions 13/15's limited eligible-view pools warrant careful review, not an unsupported assertion that all errors are pure classifier defects. Grossly incorrect **completed** polygons were not identified in the limited assistant review; unreviewed completions are not automatically certified.

## Runtime, storage and preservation

Original recordings total **773,584,779 bytes** (about 774 MB / 738 MiB). They were never duplicated, transcoded, moved or deleted. Stream decoding validates all 21,205 original frames; only 240 temporal candidates receive geometry and fifty selected frames receive classifier inference. Exact-PTS seeking may decode keyframe preroll; intermediate decode counts were not instrumented.

Summed measured geometry-call wall durations: **1,321.06 s**; summed worker-session wall durations, including indexing/decode/review rendering: **2,049.78 s**, overlapping across two workers. These sums are not serial end-to-end latency or phone latency. Recognition loop: **80.22 s**, including selected decode/crops/serialization; model forwards total **16.53 s**, median **.323 s/board**. No classifier was rerun for fusion/prefix comparisons. Source/hash checks and model initialization add overhead outside those measurements.

The local run was **34,681,652 bytes** (about 34.7 MB / 33.1 MiB) at validation, versus ~738 MiB of original media. It contains zero MOV/MP4 copies and zero persistent full-resolution candidate PNGs. Bounded persistent images: fifty selected geometry JPEGs, ten all-square crop contacts, twenty compact source/case JPEGs and ten schematic board PNGs. Packed tensors and full original frames are not cached. Only compact JSON/CSV, compressed numeric records and the reviewable diagnostic subset are exported. Final byte counts/hashes are in the validation/index artifacts.

## Reproduction, regression results and exact blockers

Run from repository root with the existing `.venv` and isolated `tools/pilot_video/python` decoder. Choose a **fresh run/output directory**; scripts refuse overwrites. The `geometry` stage can resume existing per-candidate records; do not change settings between invocations.

```bash
.venv/bin/python scripts/evaluate_independent_d3c.py register --run outputs/scanning/d3c_interim_v1
.venv/bin/python scripts/evaluate_independent_d3c.py geometry --run outputs/scanning/d3c_interim_v1 --workers 2
.venv/bin/python scripts/recognize_independent_d3c.py --run outputs/scanning/d3c_interim_v1
.venv/bin/python scripts/report_independent_d3c.py --run outputs/scanning/d3c_interim_v1 --output-dir outputs/scanning/d3c_interim_v1/evaluation_v2
.venv/bin/python scripts/render_independent_d3c.py --run outputs/scanning/d3c_interim_v1 --evaluation outputs/scanning/d3c_interim_v1/evaluation_v2 --output-dir outputs/scanning/d3c_interim_v1/diagnostics_v1
.venv/bin/python -m pytest -q tests/test_independent_d3c.py tests/test_fusion_reliability.py tests/test_offline_fusion.py tests/test_square_evidence.py tests/test_video_keyframes.py tests/test_keyframe_manifest.py tests/test_crop_experiments.py tests/test_geometry_audit.py tests/test_geometry_reliability.py tests/test_scan_recognition.py tests/test_inference.py tests/test_validation.py tests/test_fen.py
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
.venv/bin/python scripts/validate_independent_d3c.py --run outputs/scanning/d3c_interim_v1 --output outputs/scanning/d3c_interim_v1/final_validation.json
```

**292 relevant Python tests passed**, including 23 new D.3c tests; **39 browser tests passed**. Focused checks cover filename mapping, invalid/ambiguous inputs, exact display-coordinate rotation, deterministic compact output/refusal to overwrite, strict rejection of unverified geometry, explicit interim waiver, unchanged full-kernel numerical output, real review-status preservation, invalid probability/PTS/calibration/class rejection and validator restoration on exception. All fifty original selected frames and fifty diagnostic packed crops reproduce. Compilation and all **4,446 protected historical-file hashes**, plus all ten original-video hashes, passed; see [validation record](fusion_independent_d3c_validation_v1.json).

**Existing engine integration blocker, outside recognition:** `tests/test_engine.py` passes its three path/executable/error checks, then stalls at `test_valid_analysis[False]` (depth 6); a standalone run exceeded **35 s**. A separate `.2 s` time-budget starting-position analysis exceeded a **15 s external deadline**. Basic raw UCI startup answers `uciok`/`readyok` and returns a move when immediately stopped by `quit`, which is not a successful bounded analysis integration test. The first broader test run was interrupted at that stall; the clean 292-test recognition/chess regression run excludes live-engine integration. No engine implementation was changed. This blocker must be reproduced/resolved or bounded with safe cleanup in the MVP analysis stage; do not claim the Stockfish path already works end to end.

New implementation: `src/scanning/interim_video.py`, `interim_evidence.py`; `scripts/evaluate_independent_d3c.py`, `recognize_independent_d3c.py`, `report_independent_d3c.py`, `render_independent_d3c.py`; new configuration, owner-FEN mapping, tests, `scripts/validate_independent_d3c.py` and D.3c artifacts. The supplied label list was recorded only in a new manifest; historical labels remain unchanged. No production/browser code, original protocol, protected datasets or checkpoints were modified. No commit/push.

## Engineering decision and the next task

The current bottlenecks are **localization availability before selection**, then **piece/context recognition on useful views**, with fusion/uncertainty failing to turn those views into reliable exact boards. Semantic orientation is a smaller observed rejection stage, though accepted orientation correctness is not fully verified. No evidence establishes a new square-coordinate implementation bug or a need for an architecture replacement.

The tighter-224 model is useful as an **editable assisted draft**, with a substantial correction burden: the selected single frame requires 4–24 corrections/session here (mean 9.9). Sparse boards and some openings are manageable; crowded poor views can require many edits. Missing/extra/wrong pieces and incorrect game-state fields can be fixed by the user directly. Geometry/orientation problems require alignment/rotation/retry controls, not piece-only correction. Do not present this as an unattended or broadly generalized scanner.

Weak knight/king recognition, context-driven rook hallucinations and confident errors are concrete reasons for future targeted recognition work. They do **not** justify postponing the assisted MVP for retraining now, especially before photo/geometry attribution is clean. No model was trained or selected by tuned thresholds. This report can inform MVP engineering choices, but the inspected cohort must then remain interim/development decision evidence for those changes, not be recycled as pristine independent validation. If it later tunes models/thresholds or fusion rules, explicitly relabel that use development and preserve these frozen interim results. No additional recordings are required to begin the MVP.

[Staged assisted MVP plan](../MVP_ASSISTED_IMPLEMENTATION_PLAN.md): extend existing camera/recognition/review components, supply an explicit matching B.2 runtime profile, add manual Finish Scan and a 64-square correction draft, collect explicit game-state fields, bind user confirmation to the exact edited revision, then add bounded local Stockfish analysis using the existing wrapper after its integration issue is handled. Preserve the baseline profile and keep live fusion optional. No automatically accepted scan or perfect calibration is a prerequisite.

**Single next Codex task: MVP-1 — Manual Finish Scan + Editable Position Review + Explicit FEN Confirmation, with a parity-tested B.2 recognition profile.** Start this bounded product slice; do not initiate another capture/research cycle or automatically retrain. Stockfish endpoint/display follows as the next stage of the same assisted MVP plan. Every position must be confirmed by the user before analysis.

### Acceptance checklist

- [x] Ten originals discovered, number-matched, hashed and fully decoded; all twenty FENs structurally checked.
- [x] Reference absence, unchanged-position provenance, one-setting limitations and position-03 overlap explicit.
- [x] Original protocol preserved; strict gate not falsely passed; provisional/verified populations distinguished.
- [x] Frozen geometry, selection, crops/model, three fusion methods and abstention reused; no tuning.
- [x] All sessions/failures, per-class/session metrics, exact boards, reliability and nested sensitivity reported.
- [x] Bounded diagnostic/source/crop records; original timestamps/media preserved; no video duplication or large frame dataset.
- [x] Relevant regressions passed; engine timeout explicitly reported; protected-file integrity checked.
- [x] Concrete assisted MVP plan supplied; no MVP implementation, automatic gate, training, holdout access, commit or push.
