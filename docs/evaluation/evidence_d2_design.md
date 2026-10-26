# D.2 — registered evidence-characterization design

Primary question: does D.1 diversity-5 provide more useful per-square evidence than uniform-5 at the same selected-frame budget? Diversity-10 is a coverage-sensitivity condition, **not an equal-cost competitor**. Selection outcomes, geometry policies, model and inference preprocessing remain frozen. This design changes no production module and implements no fusion rule.

## Review before recognition

The source manifests are the existing D.1 `primary_3s_high/marked-board` selections. Diversity-5 and uniform-5 each have 25 frame memberships and share ten frames: their primary union contains **40 unique frames**. Diversity-10 has **43 actual selections**, not 50; it adds fifteen unique frames to the union, producing **55 total**. Review shared frames only once.

`prepare_evidence_review.py` validates the three selected manifests and writes a blind-to-recognition packet: original images, all canonical cell polygons, fitted playing-area corners, predicted orientation, source identities and a pending annotation template. Each review binds session/frame IDs, source-video SHA, derivative SHA, exact PTS/time base and geometry-result SHA. Packet SHA binds the entire set. Existing FEN/stability attestations do not verify a new per-frame grid.

Before primary inference, every primary-union frame must have an attributed human decision: `human_verified`, `rejected` or `unresolved`. A verified decision requires four explicit checks: true board corners, semantic orientation, canonical mapping and sufficient crop alignment. All begin `pending`. Pending primary reviews block inference before loading the model. Sensitivity-only frames may remain pending, but are excluded and flagged as incomplete sensitivity coverage. No rejected/unresolved/pending frame is replaced. Pipeline completion stays recorded separately from human status. Reviewed proposed corners are not represented as independent subpixel corner measurements.

The primary nominal budgets remain equal even if human review rejects frames; actual usable counts and comparison validity are reported. Unequal usable budgets are explicitly flagged. Do not silently describe that result as a clean equal-usable-frame comparison. No-view squares are unknown, never counted as correct, wrong or an exact oracle board.

## Immutable observation identity

Each unique reviewed frame is inferred once; a shared frame has multiple condition memberships, not duplicated predictions. Each observation has a stable ID from session/source/frame/canonical-square/model/config identity. Duplicate physical observations are additionally detected from source hash, exact original PTS/time base, square, model and configuration, even if someone renames the frame.

The JSON-compatible schema is `configs/experiments/evidence_d2_observation.schema.json`; runtime arithmetic/integrity validation lives in `src/scanning/square_evidence.py`. It reuses the repository's canonical class order and does not introduce browser runtime code. Existing browser `scan_piece_observation_v1` is a single-frame runtime response, not a grouped offline multi-frame history; production remains untouched.

Every observation preserves:

- Position/session/source/group IDs, original timestamp, frame/derivative identity.
- Condition, strategy and budget memberships, D.1 trace/quality metadata and raw quality/view descriptors.
- Geometry policy/completion/human status, grid-reference hash and human-review artifact/ID/hash.
- Semantic orientation and canonical square, without rotating RGB into White-bottom appearance.
- All thirteen raw float32 model logits, full uncalibrated softmax vector, argmax class, maximum probability, top-two margin and entropy in nats.
- B.2 crop specification/config hash, checkpoint hash, packed input hash/index/cache, target and context coverage.
- Explicit unknown occlusion/elevation; no fabricated precise pose.

A session contains exactly one container for each a1–h8 square. Containers hold observation references, viewpoint bins, last original timestamp, unknown visibility and `unknown`/`candidate` state. `class_distribution` stays null: **no pooled probability or label is computed**. The type can represent `resolved`/`conflict` for future work but D.2 never assigns them. No confidence threshold is used to force resolution.

## Frozen image inference

Checkpoint: `outputs/evaluation/crop_b2_20261018_run1/training/context1_224/best.pt`, SHA-256 `dad8eea28482ee894dfca0794d9f4415d21ac58db23c6d6bed8c4307c71b1bd8`; B.3 recommended retaining it. Use checkpoint-required `context1_224` crop, 224 input, one-square padding, RGB+target-mask, ImageNet RGB normalization, area RGB resizing and nearest mask resizing. Model remains MobileNetV3 Small, four channels, thirteen classes, CPU batch 16/four threads, evaluation mode and inference-only execution.

Reuse the unchanged perspective/context transform and `polygon_crop`. Start from D.1's saved source corners and fitted per-square source polygons; **do not redetect or refit**. Recreate the camera-facing 800-pixel playing area with 200-pixel context margin, extract the original two-square context, and apply the existing B.2 `packed_input` tighter subcrop/normalization. Target source coverage must pass the unchanged .99 requirement. An extraction failure is explicit, not an opportunity to replace the frame or fit different geometry. Synthetic all-four-orientation tests compare model inputs bit-for-bit against existing preprocessing.

Raw probabilities use stable float64 softmax of the unchanged float32 logits, documented to avoid calling floating-point rounding a calibration change. Checkpoint verification on 64 historical B.2 prepared inputs reproduced logits exactly; probability difference versus saved torch-float32 softmax is at most 2.46e-7. Those historical crops are development verification, not new pilot evidence or final holdout.

No completed/approved B.4 artifact was found in this checkout. Store `calibration_status=not_available`, null calibrated probabilities and raw logits. Do not fit temperature or any other calibration on this pilot. A future approved calibration must be attached with its own identity in a versioned extension; raw outputs remain intact.

## Evaluation definitions

Ground-truth FEN is read **only in the characterization script**, after inference. Reuse the registered owner-confirmed piece placement; ignore game-history fields for image metrics. Freeze the ground-truth source hash. The analysis makes no parameter/model/selection changes.

For each session/condition, report ordinary per-frame and pooled descriptive square metrics: overall/occupied accuracy, empty→piece false positives/rate, occupied→empty misses/rate, nonempty wrong-identity errors, full confusion matrix, per-class precision/recall and exact single-frame boards/errors. A class with no support has null recall; no predictions gives null precision. Black-rook observations and physical-square coverage remain visible.

For a physical square with N usable views and k correct predictions:

- Ever-correct coverage: k>0; persistent error: k=0.
- Correct multiplicity: k/N and flags k=N, k>N/2, k=1, k=0. Flags intentionally overlap (e.g. always correct is also majority); raw counts avoid ambiguous bins.
- **Oracle upper bound**: 64 minus the number of ever-correct squares; oracle exact board iff all 64 have a correct observation somewhere. This is a ground-truth availability diagnostic, not an operational prediction, achievable fusion result or recognition pipeline.

Pairwise per-session diagnostics:

- Prediction disagreement = fraction of different argmax classes.
- Error overlap = intersection/union of error-square sets (null when both empty).
- Binary error association = phi coefficient; null if either error vector has zero variance.
- Useful complementary differences (exactly one view correct) versus both-wrong/different-label differences.
- Repeat these on occupied squares to expose empty-square dominance.
- Relate registered D.1 projected-view distance and temporal gap to error Jaccard, phi and disagreement using descriptive within-session Pearson summaries, retaining every pair. Pair rows are dependent, not additional experimental samples.

Compare physical-square gains **and losses** in uniform-5→diversity-5 and diversity-5→diversity-10. Report actual added usable views, ever-correct changes, persistent classes/squares and oracle ceiling. Summed counts and frame-weighted accuracy are descriptive; the primary evidence is the five paired sessions, not hundreds of independent crops. No square-IID confidence interval, frame-IID significance test or independent generalization claim.

## Diagnostics and scope

The analyzer emits per-position view-by-square HTML matrices, prediction-quality contact sheets, per-frame errors, pair diagnostics and real packed 224-context crop strips with all thirteen probability bars. Example categories: cross-view rescues, persistent mistakes, same prediction despite visual diversity, black rook and empty false positives. If a category has no example, record that rather than fabricating one. Mask outlines are drawn for display only, after inference.

Measure extraction/compression, normalization and forward time separately, plus total elapsed and bytes. No geometry-time or phone-runtime improvement is implied. All original videos, B.2/B.3 checkpoints, D.1 outcomes and protected artifacts are read-only. No production geometry/recovery promotion, browser UX, training, final holdout, scan completion or D.3 implementation.

The owner subsequently verified all 55 frames and approved the c7 Black-pawn correction in separate bound supplements; original labels/results are preserved. The corrected-ground-truth report is `evidence_d2.md`. The oracle is specifically a ceiling for selecting among observed per-view argmax labels, not a universal bound on future models using full distributions or RGB.

**Correlated video probabilities must not automatically be treated as conditionally independent evidence. D.2 does not multiply probabilities, vote or average class distributions.** D.3 readiness is a conclusion to draw from reviewed pilot results; no readiness claim is made while the human gate is pending.
