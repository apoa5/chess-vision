# C.3 — Controlled B.2 versus taller native-crop experiment

Completed 2026-10-26. **Recommendation 2: retain B.2.** Taller native crops retain
important crown detail and produce a small pooled occupied-accuracy gain in both
seeds, but regress on the sparse validation position and king recall, add empty
false positives on average, and do not improve average exact-board reconstruction.
This bounded experiment does not justify promoting the representation. It also
does not establish the fully trained ceiling of either representation.

Artifacts: [versioned results](crop_representation_c3_results_v1.json),
[per-class precision/recall](crop_representation_c3_results_v1.csv),
[visual comparison](crop_representation_c3_examples/index.html),
[pretraining audit](crop_representation_c3_examples/pretraining/index.html),
[validation record](crop_representation_c3_validation_v1.json).

## Matched results

Each row evaluates exactly the same 1,600 validation observations: 468 occupied,
1,132 empty, 25 source boards nested within **three physical positions**.

| Seed / representation | Selected epoch | Overall accuracy | Occupied accuracy | Empty FP / 1,132 | Exact boards / 25 |
|---|---:|---:|---:|---:|---:|
| 42 / matched B.2 | 6 | 1,567/1,600, 97.94% | 442/468, 94.44% | 7, 0.62% | 7 |
| 42 / taller | 5 | 1,568/1,600, 98.00% | 447/468, 95.51% | 11, 0.97% | 9 |
| 43 / matched B.2 | 6 | 1,565/1,600, 97.81% | 441/468, 94.23% | 8, 0.71% | 8 |
| 43 / taller | 5 | 1,571/1,600, 98.19% | 447/468, 95.51% | 8, 0.71% | 6 |

Across the two seeds, mean occupied accuracy rises **94.34% → 95.51% (+1.18 pp)**,
while overall accuracy rises only **97.88% → 98.09% (+0.22 pp)**. Mean empty false
positives rise **7.5 → 9.5** (0.66% → 0.84%), and mean exact boards remain **7.5/25**.
These are descriptive seed averages on the same examples, not 50 independent
boards or additional independent positions.

Seed 42 corrects 21 control errors and introduces 20, a net **one** correct square.
Seed 43 corrects 22 and introduces 16, a net **six**. Both models still make errors
on 12 and 13 shared observations respectively. Full per-crop changes, confusion
matrices, per-board errors, per-position/orientation/angle results and all epoch
histories are in the numeric artifact. No favorable seed was selected or omitted.

### The pooled score hides a sparse-position regression

Occupied accuracy below is averaged across the two seeds; the observation support
is per seed. Photographs within a position are correlated views.

| Physical position | Photographs | Occupied support | Matched B.2 | Taller | Difference |
|---|---:|---:|---:|---:|---:|
| 003 | 9 | 270 | 95.56% | 97.22% | +1.67 pp |
| 007 | 6 | 168 | 91.96% | 95.24% | +3.27 pp |
| 011, sparse three-piece position | 10 | 30 | 96.67% | 81.67% | **−15.00 pp** |

Equal-position macro occupied accuracy falls **94.73% → 91.38% (−3.35 pp)**. On
position 011, B.2 reconstructs 5/10 views exactly in both seeds; taller reconstructs
2/10 in both. The dense positions dominate the pooled metric. This repeatable
sparse-position and difficult-class regression outweighs the small pooled gain
for the intended arbitrary-position scanner. It is one physical position, not
proof that all sparse boards will fail.

Mean per-class recall (%); each support counts repeated observations per seed,
not distinct physical pieces. Per-seed precision and full confusion matrices are
provided in the CSV/results.

| Class | Support | Matched B.2 | Taller |
|---|---:|---:|---:|
| Empty | 1,132 | 99.34 | 99.16 |
| White pawn | 115 | 98.26 | 99.13 |
| White knight | 30 | 96.67 | 98.33 |
| White bishop | 24 | 70.83 | 91.67 |
| White rook | 30 | 93.33 | 98.33 |
| White queen | 15 | 96.67 | 100.00 |
| White king | 25 | **100.00** | **82.00** |
| Black pawn | 105 | 99.05 | 98.57 |
| Black knight | 30 | 78.33 | 81.67 |
| Black bishop | 24 | 93.75 | 95.83 |
| Black rook | 30 | 96.67 | 100.00 |
| Black queen | 15 | 73.33 | 80.00 |
| Black king | 25 | **100.00** | **94.00** |

Neither matched arm makes a white-king→rook or black-king→rook error on this
validation population. B.2 seed 42 makes one white-rook→king error, removed by
taller; the other reverse king/rook counts are zero. Thus this dataset does not
establish a king→rook remedy. Taller instead introduces king→pawn and other king
confusions. Occupied→empty misses are one/zero for B.2 seeds 42/43 and zero/zero
for taller. Most residual errors are identities or false occupancy.

### Viewing angle and training stability

These are **historical catalog angle annotations**, not calibrated camera poses.
No additional precise elevation was inferred. Mean occupied accuracy across seeds:

| Catalog angle | Source boards | Occupied support | B.2 | Taller |
|---|---:|---:|---:|---:|
| 60° | 4 | 64 | 97.66% | 96.09% |
| 75° | 13 | 249 | 94.18% | 95.38% |
| 90° | 8 | 155 | 93.23% | 95.48% |

The most oblique annotated stratum does not improve. Angle, physical position,
class support and background are confounded; the small strata do not establish
causal viewpoint effects. Orientation results are likewise mixed: for example,
taller seed 42 improves White-right recall but regresses on White-left, and this
pattern does not consistently repeat in seed 43. Do not choose a camera-side
policy from these results.

The selected taller checkpoints are epoch 5 in both seeds. At the mandatory final
epoch 6, occupied accuracy is **94.87% / 92.95%** with seven/eight empty FPs and
9/25 exact boards in both runs. The matched B.2 final checkpoints are their selected
checkpoints. Consequently last-epoch mean occupied accuracy is **93.91% taller /
94.34% B.2**, reversing the selected-checkpoint ranking. This is reported as a
stability sensitivity, not used to replace the predefined checkpoint rule. Learning
curves and best/last artifacts are retained; six epochs do not prove convergence.

The old frozen B.2 checkpoint's reused historical validation predictions give
97.88% overall, 95.09% occupied, 11 empty FPs and 7/25 exact boards. Its longer
training, warm-start adaptation and loss-based selection differ. It is a
**historical reference only**, not the matched control. C.3 results must not be
presented as improvements on C.2's 55 D.2 frames or D.3c's ten recordings; neither
cohort was evaluated here.

## Visual diagnosis

The compact packet includes nine pretraining crop pairs covering crowns,
board-edge rooks, neighboring pieces, empty targets and all four orientations,
plus six post-hoc validation cases with both seeds' predictions and uncalibrated
13-class softmax scores. Source photographs/target footprints are shown beside
the actual B.2 and native RGB inputs. Display outlines are never classifier input.
Examples illustrate the complete numeric comparison; they do not determine which
examples enter scoring or which checkpoint is selected.

- **e3 white king, position 011, Black-near, catalog 60°:** the native input retains
  the crown that the tighter B.2 crop truncates. Nonetheless both taller seeds
  predict **white pawn**, while both controls predict white king. A neighboring
  pawn overlaps the target region. This confirms that restoring head pixels is
  insufficient; it does not prove the model's internal attention mechanism.
- **e7 empty, position 011:** the true neighboring king stands on e6 and its body
  projects over the e7 ground-square footprint. Both controls predict empty; both
  taller seeds predict black king. The image-up-fallback crop retains more of the
  neighbor. This is visible spillover, not evidence that e7 should be relabelled
  or that the footprint mask is a silhouette.
- **d6 empty, position 007:** both controls hallucinate a black pawn; both taller
  models correctly predict empty. Expanded native context can also help.
- **f8 black rook, position 007:** taller corrects a seed-42 black-bishop confusion;
  both models are correct in seed 43. **f6 black knight, position 003** is corrected
  from black king in both seeds.
- **e4 white pawn, position 011:** both taller seeds predict white king, preserving
  a difficult contextual error; the controls differ. Full source detail does not
  guarantee correct target attribution.

No new square-index, semantic-orientation or RGB/mask-alignment defect was
confirmed. The source grids are historically approved, not pixel-perfect piece
contact/silhouette ground truth. Crop completeness was assessed visually; no
automatic percentage of a verified piece silhouette is claimed. Occluded or
camera-clipped detail cannot be recovered by taller cropping.

Validation ground-mask area-equivalent width ranges **70.8–79.5 px** for B.2
(median 74.9) and **59.0–121.5 px** for native (median 72.2). Training native widths
span 33.1–124.4 px. More height therefore does not necessarily increase effective
piece resolution. The recorded valid-pixel fraction also includes letterbox
padding and must not be interpreted as board visibility or verified silhouette
coverage. These measurements support caution about the input bundle, not a new
parameter search.

## Frozen comparison

The [registered design](crop_representation_c3_design_v1.md) and
`configs/experiments/crop_representation_c3_v1.json` were frozen before real-data
training or C.3 validation outcomes. The completed preflight binds configuration
SHA-256 `86fae91ecc7e72898c51f9e440b4e3e8dd3d5b60d64406828e7bbc185cb2a918`,
metadata, source code, ImageNet initialization, input pixels and group assignment
in `outputs/evaluation/crop_representation_c3_run2/registration.json`.

**Arm A** is the exact B.2 tighter `context1_224` representation: approximately
three-square RGB context, generated from frozen board-rectified pixels, with
area resizing to 224×224 and a nearest-neighbor binary ground-square mask.
**Arm B** uses unmodified C.1 native height-envelope/letterboxing helpers: the
same target's reviewed source-image polygon, a fixed two-square-height extrusion
along an assumed camera-facing board normal, and 0.35-square margin. Original RGB
is not warped as though elevated pieces lie on the board plane. Aspect-preserving
224 letterboxing uses area downsampling or linear upsampling, plus the aligned
binary target footprint. Both use ImageNet RGB normalization and a fourth 0/1 mask
channel. Neither mask is a silhouette or an isolation/segmentation operation.

Height direction assumes square pixels, a centered principal point and no lens
distortion. Unstable/near-overhead homographies use C.1's explicitly recorded
image-up fallback. These are crop heuristics, not calibrated pose, measured piece
height or a guarantee of complete visibility. Neighbors, camera-edge clipping and
occlusion remain possible. Retaining more height can reduce target detail. Native
and B.2 inputs also differ in context shape, target scale and added plane-warp
distortion: this tests the representation bundle, not height alone.

## Data provenance and geometry gate

Only the original approved historical training and validation rows are used:

- Training: **7,168 crops, 112 source photographs, 12 physical positions**.
- Validation: **1,600 crops, 25 photographs, positions 003/007/011**; 468 occupied
  and 1,132 empty observations, with repeated views nested within three positions.

The original assignment is unchanged. Position/source identity, identical source
bytes and FEN placement, and known-session grouping are checked using the existing
leakage audit. Historical capture-session IDs are unavailable (all 160 approved
historical sources, of which 137 are used here). We therefore report physical
position and source-board results, not invented independent recording sessions.
The validation images have been inspected in earlier development phases; they are
separated from C.3 training, not a fresh generalization benchmark.

The historical test positions 006/010/016 are metadata-only for grouping: their
images are not decoded/evaluated in C.3. Neither D.2's repeatedly inspected pilot
frames nor D.3c recordings train models, choose crop parameters or select
checkpoints. The final holdout and sealed evaluation data are excluded. Existing
D.2 label supplements are not silently propagated into the historical catalog.
Historical labels remain the original board-level-approved labels; individual
piece visibility/contact points are mostly unreviewed. No new verified labels,
silhouettes or camera poses were manufactured.

The first preflight attempt expected a `source_corners` field absent from 45
manual historical records. It stopped before training. Inspection found that all
137 used boards already contain exact reviewed source-square polygons. The
adapter now reads those saved polygons directly, without SIFT, localization,
orientation estimation, corner replacement or detector threshold changes. The
clean second preflight succeeded on **all 8,768 targets**, with **282 exact B.2
pixel/mask parity checks**, zero rejected/dropped targets, and frozen per-target
packed-input hashes. Source polygons remain byte-for-byte unchanged.

The assumed-normal path covers 5,952/7,168 training and 1,344/1,600 validation
targets; image-up fallback covers 1,216 and 256 respectively. Fallback is a
declared heuristic, not a localization failure or verified geometry refinement.
Automated tests reject missing/nonfinite/nonconvex source geometry instead of
silently dropping examples or guessing new corners.

## Matched training and checkpoint selection

Two paired seeds, **42 and 43**, each train both arms for **six complete epochs**:
two head/stem epochs and four full-network epochs. Both start with the same
ImageNet MobileNetV3 Small RGB weights, zero extra mask-channel weights and a new
seed-matched 13-class head. Initial tensor-state hashes and actual shuffled sample
orders are recorded and compared across each pair. No chess checkpoint initializes
either matched arm. The older frozen B.2 checkpoint remains a historical reference
with more training and a different selection procedure; it is not Arm A.

Batch size 16, four CPU intra-op threads, one inter-op thread, no loader workers;
deterministic PyTorch algorithms. Adam starts at .001 for head/stem and is reset to
.0001 when all parameters unfreeze; betas (.9,.999), epsilon 1e-8, no weight decay.
The original training-only square-root inverse-frequency loss weights are reused.
RGB augmentation is identical per seed/epoch/crop: brightness/contrast ±.1, mild
3×3 blur probability .15 and noise probability .20/std .01. Masks are unaffected;
no geometric augmentation, curriculum, occupancy network, fusion or B.3 search is
introduced.

The primary checkpoint criterion is highest occupied validation accuracy, with
ties broken by fewer empty false positives, lower weighted loss, then earlier
epoch. Epoch zero is recorded; all six training epochs run irrespective of score.
Both best and last checkpoints are retained. There is no crop redesign, extra seed,
learning-rate search, performance early stop or epoch extension after seeing
validation outcomes. Six epochs and two seeds are a bounded CPU comparison, not
proof of convergence or fully optimized representation ceilings.

## Resource and artifact boundaries

The CPU-only environment and cached ImageNet weights already existed; no package
installation, CUDA dependency or model download was needed. Each arm runs in a
separate process. Native packed inputs live in a bounded uint8 RAM cache, released
at process exit. Arm A reuses the existing read-only B.2 cache. No new persistent
full crop dataset, original image/video copy or raw-video extraction is created.
Experimental best/last checkpoints remain under ignored `outputs/evaluation/`.

Prespecified limits are 4 GiB peak process RSS, one hour per arm and four hours
aggregate arm time. Numerical, geometry, input-integrity or resource failure
blocks comparative completion; one arm is never silently shortened. A discarded
random-input backward timing probe preceded registration; it used no real labels,
validation performance or saved model.

Standalone CPU profiles use the first validation source without prediction-based
selection, batch 16, two warmups and five measured 64-square forwards. Crop/packing
profiles start from already frozen context for A and native source RGB for B;
they exclude disk decode, board localization/warp and phone runtime. Full training
memory includes the input cache and must not be described as model-only inference
memory. Both backbones have the same parameter count and input dimensions.

All four runs completed their registered budgets:

| Arm / seed | Arm time | Peak training RSS | Forward / 64 squares | Standalone profile RSS |
|---|---:|---:|---:|---:|
| B.2 / 42 | 15.35 min | 2,459 MiB | 285.9 ms | 544 MiB |
| Taller / 42 | 14.98 min | 2,467 MiB | 268.8 ms | 532 MiB |
| B.2 / 43 | 13.72 min | 2,459 MiB | 263.7 ms | 545 MiB |
| Taller / 43 | 14.92 min | 2,463 MiB | 275.1 ms | 542 MiB |

Aggregate arm time was **58.98 minutes**, within the four-hour cap. Timers include
RAM-cache construction, epoch evaluation and checkpoint writing, but not initial
Python imports/provenance checks. This is observational wall time, not a claim that
different seeds or representations inherently train faster. The measured runner
is the CPU-only i5-1135G7 environment with about 7.66 GiB RAM visible; no Iris Xe,
CUDA or dedicated GPU execution was used.

Both models have **1,531,325 parameters** and 6,246,889-byte experimental checkpoint
files. Dense forward operations are the same; the small timing differences are
not an architectural speed improvement. Mean-of-seed-median forward cost is
approximately 275 ms B.2 / 272 ms native. Warm crop/packing medians are 128–129 ms
B.2 / 70–71 ms native per board, with normalization another 125–141 ms. These
exclude source decoding and the initial B.2 warp; they are not phone/live-pipeline
latencies or a controlled comparison against historical B.2 timing reports.

The run directory is about **55 MiB**, dominated by eight best/last experimental
checkpoints; the published crop packet is about **1.3 MiB**. Only compact logits,
histories, hashes, source metadata and thumbnails accompany them. The potentially
1.76-GB native uint8 dataset cache was never persisted, and no frozen model was
overwritten.

## Reproduction

The prepared run already exists. Use a fresh directory when rebuilding; preparation
and individual workers refuse to overwrite an existing run/arm. Completed-arm
reuse verifies provenance; blocked/partial arms are not silently resumed.

```bash
.venv/bin/python scripts/run_crop_representation_c3.py prepare --run outputs/evaluation/crop_representation_c3_replay
.venv/bin/python scripts/complete_crop_representation_c3.py --run outputs/evaluation/crop_representation_c3_replay
.venv/bin/python scripts/report_crop_representation_c3.py --run outputs/evaluation/crop_representation_c3_replay --output docs/evaluation/crop_representation_c3_replay_results.json --examples docs/evaluation/crop_representation_c3_replay_examples
.venv/bin/python -m pytest -q tests/test_crop_training_c3.py
.venv/bin/python scripts/validate_crop_representation_c3.py --run outputs/evaluation/crop_representation_c3_run2 --results docs/evaluation/crop_representation_c3_results_v1.json --output outputs/evaluation/crop_representation_c3_validation_replay.json
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
```

An individual registered worker is reproducible with `run_crop_representation_c3.py
train --run <run> --arm b2|taller --seed 42|43`. Numeric results are deterministic
under the recorded CPU/software configuration; wall-clock time and RSS can vary.
The actual execution order is B2-42, taller-42, taller-43, B2-43. The assisted MVP
remains a separate workstream; no production profile, checkpoint, geometry policy,
browser behavior, FEN/Stockfish flow or existing report was modified.

## Validation and acceptance

**370 targeted Python tests passed** in two disjoint groups: 84 training/dataset/
crop checks and 286 inference/geometry/chess/evidence checks. These include **17
C.3 tests** for exact C.1 extractor parity, saved raw geometry without corner
fields, missing/nonfinite/nonconvex geometry rejection, target-grid binding,
label-blind extraction, deterministic input hashes, read-only RAM caching, matched
initialization, actual sample-order tracking, binary-mask preservation, original
position groups, forbidden cohort rejection and the frozen checkpoint tie-break.
**39 browser tests passed.** The existing optional C.2 environment was used only
to allow its already-installed adapter dependencies in the regression suite; C.3
training used the unchanged application environment.

An attempted broader `pytest --ignore=tests/test_engine.py` run was terminated
with exit 143 before completion; its log showed no assertion failure. It is not
reported as a full-suite pass. The targeted groups above completed successfully.
Unchanged live-Stockfish integration is outside this input/training-only scope.
Commands and persistent log hashes are in the validation artifact.

All four selected/final checkpoint predictions reproduce their epoch confusion
matrices. Both seed pairs have identical starting tensor-state hashes and actual
shuffled sample-order hashes for every epoch. Each arm processed all 7,168 training
rows per epoch and all 1,600 validation rows. No sample was dropped, and no arm
received a reduced budget. All source/input hashes and resource caps pass.

Preservation checks cover **4,948 historical/C.1/C.2 artifacts**, the frozen model
assets, original source/crop metadata and **all ten D.3c original videos**, plus
the C.3-bound 137 historical source photos, initialization and reused caches.
Video hashing is preservation only, not training/evaluation. Final-holdout data
was not accessed. No existing dataset, source image, video, checkpoint or
production/browser module was modified. No commit or push was made.

| Acceptance requirement | Result |
|---|---|
| Freeze geometry, groups, augmentation, training and stopping policy before C.3 outcomes | Passed; immutable registration/config/design |
| Deterministic native RGB+target indicator, without plane-warping tall pieces | Passed; reuses unmodified C.1 helpers |
| Crowns, edge pieces, neighbors, empty targets, angles and geometry failures audited | Complete; visibility/pose limitations explicit |
| Matched backbone, initialization, sample order, optimizer and six-epoch budget | Passed for both seeds |
| Primary/secondary metrics, per-position boards, class confusion and cost | Complete; session IDs explicitly unavailable |
| No D.2/D.3c/holdout tuning, frozen checkpoint or production replacement | Preserved |
| Automated regressions and protected-file checks | Passed as scoped above |
| Native-cache storage bounded; no duplicate full crop dataset | Passed |
| Research separate from assisted MVP; stop after C.3 | Preserved |

New implementation is isolated in `src/vision/crop_training_c3.py` and preparation/
training, fixed-run orchestration, evaluation, reporting and validation scripts.
The versioned configuration/design, focused tests, numeric metrics/CSV, visual
packet, validation record and C.3 index accompany it. Frozen C.1 helpers and the
existing training/model implementations remain unchanged.

## Decision

**Retain B.2.** The native representation restores useful visual features and
improves some bishops, rooks and knights, but the gain is small after accounting
for empty squares and is offset by consistent sparse-position and king-recall
regressions. The selected checkpoints do not raise mean exact-board accuracy,
and the occupied-score advantage disappears at the fixed last epoch. The last
native checkpoints do reach 9/25 exact boards in both seeds and restore white-king
recall to 100%, but with lower pooled occupied accuracy and remaining sparse-board/
black-king regressions. That secondary tradeoff is worth preserving, not a reason
to replace the predefined selection rule after seeing outcomes. The evidence is
insufficient to promote the tested representation for the intended scanner.

This is not evidence that all native/perspective-aware crops are inferior, nor
that a larger backbone is necessary. Two seeds, three previously inspected
validation positions, missing session provenance, imperfect visibility labels,
uncalibrated height heuristics and a short fixed training schedule limit the
conclusion. No crop, threshold or schedule was changed to improve these results.

Keep all checkpoints experimental and the frozen B.2 candidate intact. The
assisted-MVP Finish Scan → editable board → user confirmation → analysis work
should continue independently when authorized. No additional training, crop
search, capture campaign, model promotion or architecture experiment was begun.
**C.3 stops here for review.**
