# B.2 — Preregistered crop and resolution design

2026-10-18. Plan: `configs/experiments/b2_crop_resolution.json`, SHA-256
`00df6c68116d50361947b07379aa7690d9939b16e11312c10ebbbc23fdf2cefc`.
The supplied research/roadmap files are at repository root in this checkout.
All experiments are historical development measurements, not independent final
holdout evidence. No A.3 captures, live scans or marked-board recovery enter them.

## Fixed inputs and baseline

Use the untouched epoch-16 MobileNetV3 Small checkpoint and authoritative approved
`data/classification/crops.json`: 7,168 training, 1,600 validation and 1,472 test
crops, grouped by 12/3/3 physical positions. Training positions, labels, source
photos and semantic square IDs never change. Source-byte/placement/known-session
grouping is checked. No new frames or cross-split derived examples are introduced.
Historical sessions/timestamps are missing; true session independence and all
cross-position near-duplicates cannot be certified from these records. This is a
documented dataset limitation, not an invented session assignment or a resplit.

All transformations start from the same stored approved camera-facing PNG crops
and their target polygons. No detector, homography, orientation, board grid or
B.1b policy is rerun or modified. Frozen A.1/A.2/B.1/B.1b assets are hash checked.
The existing evaluator reproduced 79/1,472 errors, 94.633% overall, 82.675%
occupied-square accuracy. Prepared exact-board accuracy remains 0/23.

The source board canvas is 1,200² with an 800² playing area and 200-pixel border.
The target is about 100² board-plane pixels, with two squares of padding on each
side: roughly five square widths in a 500² crop. BGR converts to RGB; area
interpolation resizes to 224². A separately rasterized target polygon is resized
nearest-neighbor as the binary fourth channel; ImageNet RGB normalization follows.
Nominal target width is about 45 input pixels. Tall pieces can project outside
their ground-square masks. Padding is context, not new observations.

Raw-photo target-polygon mean edge lengths over the frozen approved dataset range
approximately 27–91 pixels (median 69). The normalized 100-pixel squares already
upsample many originals. Neither increasing input size nor rerendering the same
small source photo creates detail. Actual crop/mask distributions are recorded
by the experiment runner rather than assuming every polygon is exactly square.

## Registered variants

| Variant | Input | Context padding per side | Nominal target width | Purpose | Adapt? |
|---|---:|---:|---:|---|---|
| context2_224 | 224 | 2 squares | 45 | Frozen baseline and continued-training control | yes |
| context2_320 | 320 | 2 squares | 64 | Model input sampling/resolution, same source/context | yes |
| context1_224 | 224 | 1 square | 75 | Tighter crop, less context and larger target | yes |
| context_half_224 | 224 | .5 square | 112 | Stronger context/scale sensitivity | no |
| target_only_224 | 224 | 0 | about 224 | Target-only diagnostic; tall-piece clipping expected | no |
| suppress_outer_224 | 224 | Original canvas | 45 | Zero outer context beyond the central three-square window | no |
| quarter_detail_224 | 224 | 2 squares | 45 mask / about 25 RGB source samples | Detail-loss control | no |
| quarter_detail_320 | 320 | 2 squares | 64 mask / same about 25 RGB samples | Upsampling cannot restore lost detail | no |

Every variant is evaluated at frozen weights first. Those comparisons measure
the **current model's sensitivity**, including distribution shift; they do not
estimate optimally trained alternative architectures. Three designated arms also
receive matched full-network adaptation, including an unchanged-crop continuation
control. No extra model branch, geometric augmentation or calibration is added.

Context suppression keeps the original canvas, target polygon, input size and
target scale fixed, changing only outer RGB information. Subcropping intentionally
changes context and target scale together; it cannot alone identify which causes
a gain. Compare suppression, tighter cropping and the larger-input arm jointly.

Detail controls area-downsample RGB to one quarter width/height, then restore
original crop dimensions with linear interpolation. Geometry/mask are untouched.
The two detail controls therefore have identical degraded RGB before final input
resizing. This explicitly separates available information from output dimensions.

## Training and selection

All adapted arms independently initialize from the frozen checkpoint; Adam state
is reset, learning rate is 0.0001, batch size 16, CPU threads four, seed 42.
Each uses all 7,168 training samples for **three epochs** and the original
photometric augmentation/class weighting. Sample order and per-crop/epoch RNG
match across arms. This is a bounded warm-start adaptation screen, not full
baseline-length retraining or evidence of convergence. One seed limits confidence.

Evaluate validation loss at epochs 0–3 and retain the lowest weighted validation
loss in each arm. Including epoch zero treats the continuation control fairly.
Rank adapted configurations using validation occupied-square accuracy, macro F1,
then lower inference cost. Historical test results are descriptive and never
select a checkpoint/configuration. All training and within-arm checkpoint selection
finishes before historical test prediction; the locked ranking formula is applied
to validation metrics after export. The untouched final holdout stays sealed.

Caches store resized RGB and binary masks losslessly as uint8 memory maps, before
photometric augmentation. They preserve the frozen normalization/augmentation
contract exactly, verified against the shared implementation. This avoids repeated
large-PNG decoding during CPU training; no preprocessing change enters production.

## Measurements and uncertainty

For validation and the identical 23 prepared test boards, export overall and
occupied accuracy; per-class precision/recall/F1 and confusion matrices; exact
board counts; orientation and position groups; probabilities and raw logits.
Missing class support is shown explicitly, not interpreted as evidence of failure.

Measure loaded-model CPU forward latency on 64-square batches (batch size 16,
two warmups, eight measured repeats), parameter count/bytes and checkpoint size.
Transforms/loading are outside that timer and must be discussed separately.
This is not a raw-pipeline, live-phone or multi-view benchmark.

Use paired per-position differences and leave-one-position-out summaries to show
how three physical test-position groups affect conclusions. Do not use thousands of
correlated squares as independent trials for narrow confidence intervals. Camera
orientation, room, light and position remain confounded; historical visibility
reviews cover only a subset. Visual error analysis is qualitative unless an
existing current human review supports an occlusion/detail label.

Retain the current configuration if validation gains are weak, inconsistent or
too costly. Recommend B.3 augmentation and B.4 calibration separately; do not
implement either in B.2. All experimental caches/checkpoints stay under a new
ignored output directory. No production asset, commit or push is authorized here.
