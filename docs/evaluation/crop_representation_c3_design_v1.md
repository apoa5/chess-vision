# C.3 — Frozen controlled experiment design

Registered before real-data model training or C.3 validation outcomes, 2026-10-26.
Configuration: `configs/experiments/crop_representation_c3_v1.json`. The run's
registration binds its SHA, input/source hashes and approved group membership.
No crop redesign, epoch extension or seed expansion after validation results.

## Comparison and geometry

Both arms use four-channel MobileNetV3 Small and 224×224 inputs. A is the exact
B.2 `context1_224` RGB+binary footprint representation: existing saved approximately
three-square context from cubic board rectification, RGB area resize and nearest
mask resize. Reuse its read-only uint8 packed cache, with source/row/parity checks.

B reuses unmodified C.1 `height_envelope` and `raw_letterbox`: use the same approved
target's saved source-image polygon from frozen grid metadata. All 137 train/val
boards retain these polygons, including 45 manual records lacking a separate
`source_corners` field. No inverse geometry needs re-estimation. Estimate
a camera-facing plane normal from the frozen board homography under centered
principal point, square pixels and no radial distortion. Project a fixed
**two-square-height** extrusion; expand its union with the base by **0.35 mean
source-square edge lengths**, then crop original RGB and aspect-preserving
letterbox to 224. Area downsampling / linear upsampling; nearest binary target
footprint mask. No RGB board-plane warp. Ill-conditioned intrinsics/near-overhead
views use C.1's explicit image-up fallback, not an invented verified camera pose.
No calibrated piece height, complete segmentation or visibility guarantee is
claimed. Masks mark the ground square, not the piece silhouette. Neighbors and
source-edge clipping can remain; height preservation may shrink target detail.

Source corners and fitted square polygons are frozen approved metadata. Do not
rerun localization, orientation or recovery. Invalid geometry blocks the run;
there is no per-arm sample exclusion. A compact pretraining visual packet covers
crowns, edges, neighboring empty squares and steep/overhead views, with source
grid/footprint overlays. Geometry rejection is also tested synthetically. Existing
C.1 D.2 examples are linked only; no video or D.3c image is needed.

## Data and exposure

Use only original historical `crops.json` train/val assignments: **7,168 crops,
112 photographs, 12 training positions** and **1,600 crops, 25 photographs, three
validation positions** (`003`, `007`, `011`). Frozen rows/labels/class weights are
unchanged. Validate grouping by physical position, source, identical source bytes,
identical FEN placement and known capture session. Historical sessions are absent:
report that limitation, never manufacture session IDs or claim independently held
out sessions. Per-source boards are views nested within position, not independent
sample units. Individual piece visibility is mostly unreviewed despite board-level
geometry/label approval.

Historical test positions `006/010/016` are metadata-only for grouping and not
decoded/evaluated here. D.2's 55 repeatedly inspected frames do not train, select
checkpoints or form a new independent test. D.3c recordings, sealed evaluation data
and the final holdout are excluded. Source photos are not copied or edited.

## Matched training

Two paired seeds **42,43**, **six full epochs each arm**, head/stem epochs 1–2,
full-network epochs 3–6. Fresh ImageNet RGB weights, zero additional mask-channel
weights and seed-matched new 13-class heads. Record the initial tensor-state hash
and require equality within each pair. This avoids starting the alternative from
a model already specialized to B.2 crops. Frozen B.2 remains a historical reference,
not the matched arm or its initialization.

Batch 16, four intra-op/one inter-op CPU threads, no workers. Adam lr .001 in the
head phase, fresh Adam .0001 at full unfreeze, betas (.9,.999), eps 1e-8, no weight
decay. Reuse train-only square-root inverse-frequency loss weights. Same shuffled
sample indices and per-crop/epoch RNG in each pair. Existing brightness/contrast
±.1, 3×3 mild blur probability .15, noise probability .20/std .01 affect RGB only;
no geometric augmentation or mask corruption. No B.3 model/augmentation search.

Primary checkpoint criterion is highest **pooled occupied validation accuracy**;
ties prefer fewer empty false positives, lower weighted loss, then earlier epoch.
Every arm trains all six epochs, irrespective of performance. Also retain the last
epoch to expose training trends rather than claiming convergence. Selection never
uses D.2/D.3c/historical test. Six epochs/two seeds are a bounded CPU comparison,
not evidence of fully optimized architectures or broad generalization.

## Evaluation, resources and decision

Primary: occupied-square accuracy on the three original validation positions.
Secondary: overall accuracy, empty FP count/rate, occupied misses, per-class
precision/recall/confusion, king↔rook, exact boards, physical-position macro and
per-board/position results, annotated viewing angle/orientation, known-session or
explicit unavailable status, training time, loaded CPU forward/board and RSS.
Report both seeds, paired corrections/regressions, best and last epochs; do not
treat 1,600 repeated square views as independent recording sessions.

Reuse the existing B.2 cache; compute B's uint8 targets in RAM and release them
after each isolated arm process. No new persistent full crop dataset. Save only
best/last experimental checkpoints and compact numeric predictions, histories,
input hashes and a small gallery. Expected CPU training order of hours, based on
the existing approximately three-minute epoch measurements; no GPU dependency.
A discarded five-update random-input feasibility probe measured shape/backward
cost only, without real training/validation samples or saved weights.

Limits: 4 GiB peak process RSS, one hour per arm, four hours aggregate training;
stop on nonfinite loss, changed frozen inputs, invalid geometry or resource breach.
Do not reduce just one arm's sample/epoch budget; incomplete pairs mean no complete
representation conclusion. No performance-based early stopping or threshold tuning.

Promote only for further validation if occupied gains are consistent across seeds
and not offset by empty false positives or difficult-class regressions. Otherwise
retain B.2 or identify persistent failures warranting a different approach. Report
the actual tradeoffs and undertraining limitations; no automatic production change
and no claim that a selected checkpoint is independently validated. Assisted MVP
engineering remains a separate workstream. Stop after C.3.
