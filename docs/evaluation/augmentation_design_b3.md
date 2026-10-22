# B.3 — Registered augmentation experiment design

2026-10-19, before training or examining B.3 validation/test predictions.
The supplied research and plan are at repository root (`VISION_RESEARCH.md`,
`IMPLEMENTATION_PLAN.md`), not under `docs/` in this checkout. Read alongside
[B.2 results](crop_resolution_b2.md) and its registered design/comparison CSV.

## Fixed representation, provenance and budget

Every arm starts independently from the validation-selected B.2
`context1_224/best.pt` (SHA-256
`dad8eea28482ee894dfca0794d9f4415d21ac58db23c6d6bed8c4307c71b1bd8`).
Use the exact B.2 one-square-padding 224×224 RGB + target-mask caches and original
class weights. No change to crop construction, labels, geometry, orientation,
canonical mapping, recovery or production preprocessing. Evaluation is always
augmentation-off. No test-time augmentation or ensemble.

Train all six families separately using **seeds 42 and 123**, fresh Adam 0.0001,
batch 16, all 7,168 training examples, **one full-network adaptation epoch** each.
This is twelve matched runs / twelve epochs, a bounded warm-start screen from
an already adapted model; it is not convergence or a full training recipe search.
The two seeds vary sample order, dropout and augmentations conditional on the
same B.2 initialization. They do not replace independent physical positions.
CPU-only cost motivates the short budget; no unbalanced extensions or combined
family are allowed after inspecting these results.

Within each arm retain minimum weighted validation loss over epochs zero and
one. Epoch zero allows rejecting harmful continuation. All checkpoints complete
before final validation prediction, then the selection decision is saved **before
any historical test prediction**. The unchanged B.2 candidate is also evaluated
as an anchor and its probabilities must match the preserved B.2 outputs exactly.

Splits remain 7,168/1,600/1,472 crops in 12/3/3 position groups. Existing
position/source/source-byte/placement/known-session checks are reused. Historical
session IDs/timestamps are absent for 160 source photos, so cross-position
near-duplicate/session independence cannot be certified. No new frames or split
assignments are made. Previously inspected historical test results are development
evidence only. The A.3 final holdout is never loaded or used for selection.

## Separate families

| Family | Training perturbation | Purpose |
|---|---|---|
| `none` | None, including no photometric augmentation | Literal matched augmentation-off control |
| `legacy_visual` | Existing brightness/contrast ±0.10, Gaussian blur probability .15, noise probability .20/std .01 | Existing B.2 training-recipe continuation reference |
| `perspective_affine` | With probability .65: corner offsets ≤2% input width and shear ≤2° | Mild projective/framing distortion |
| `framing` | With probability .65: rotation ±5°, scale .95–1.05, translation ±3% width | Small camera roll, framing and scale errors |
| `lighting` | With probability .65: gamma .85–1.15, contrast .85–1.15, exposure ±.06, smooth directional gradient amplitude ±.15 | Mild exposure/illumination changes |
| `video_quality` | With probability .65: one of 3×3 Gaussian σ .3–.75, 3-pixel horizontal/vertical blur, JPEG quality 75–95; pixel noise σ ≤1.5/255 | Mild smartphone motion/defocus/compression proxy |

New families do not implicitly stack the legacy photometric recipe; each isolates
one family relative to `none`. Registered strengths are fixed and checked against
implementation constants; editing strength metadata alone fails closed. The
configuration family list enables/disables recipes, always retaining the `none`
control. New strengths require an explicit new design/code version.

Geometric transforms operate on the same packed RGB and mask with the **same
forward homography**. RGB uses linear interpolation and replicated border;
mask uses nearest interpolation and constant zero border, preserving binary
values. The transformed mask bounding box plus a 25-pixel guard must stay inside
the crop or the perturbation is rejected. Modest bounds, no flips/quarter turns,
no orientation relabeling, no color inversion, no new canonical square IDs.
Replicated edge wedges are an acknowledged synthetic framing artifact, outside
the protected center. These 2-D perturbations cannot synthesize unseen surfaces
of 3-D pieces or substitute for real multi-angle captures.

Partial occlusion is **deferred**: ground-square masks do not identify the complete
piece silhouette, and no matching current visibility reviews establish which
features can safely be erased. Generic random erasing can remove an entire crown
or introduce apparent occupancy on empty squares. No such examples are trained
or automatically labeled. A future separately reviewed occlusion experiment needs
piece/visibility support and explicit label-preservation criteria.

## Validation-only safety selection

Do not select simply by occupied accuracy. A family qualifies only if **both seeds**
satisfy these gates against both their matched `none` seed and the B.2 anchor:

- Empty false positives and missed occupied counts each no more than reference +1.
- Overall correct count at least reference −2; exact boards at least reference.
- Each Black rook/king/queen correct count at least reference −1.
- Each camera-orientation occupied correct count at least reference −1.
- Occupied accuracy does not regress against matched control in either seed;
  its mean gain is at least one percentage point.

These are preregistered small count tolerances, not statistically established
non-inferiority margins. Report every violation. Rank qualifying families by
mean validation macro F1, then mean occupied gain, then name for a deterministic
tie. Representative checkpoint seed is fixed to **42**, never the best test seed.
If none qualifies, recommend retaining B.2. No test metric can override these
rules or trigger additional training. No combined family is included in this
separate-family screen.

## Metrics, diagnostics and verification

For all twelve runs and the B.2 anchor, export validation/test overall and occupied
accuracy, empty false-positive count/rate (denominator true empty), missed occupied
count/rate (denominator true occupied), occupancy detection precision, per-class
precision/recall/F1/confusion, exact boards and orientation/position groups.
Record probabilities/logits, selected epochs, model/checkpoint size and CPU
64-square forward timing with the same 4-thread/batch-16/two-warmup/eight-repeat
procedure as B.2. Augmentation is training-only: inference preprocessing is the
identical tighter crop, without additional runtime work. Supplement this with a
loaded-crop preprocessing benchmark and training augmentation timing.

Export every changed square ID versus matched `none`, separating fixed errors,
broken correct predictions and changed wrong identities; distinguish occupied
identity changes from empty/occupied errors. Report paired per-position changes
and leave-one-position-out sensitivity, not square-IID confidence intervals.
Inspect actual training previews before launch and paired improvement/regression
examples after completion. Green masks are diagnostic overlays only, never input
RGB. No synthetic image becomes verified photographic ground truth.

Tests cover joint RGB/mask alignment, forward matrix direction, protected target
clipping, binary masks, finite tensors, unchanged labels/cache, deterministic
seed/epoch behavior, augmentation-off validation/test, correct metric denominators
and selection safety gates. Source hashes are checked at epoch boundaries;
frozen A.1–B.2 artifacts/checkpoints/caches are checked before and after the run.
All checkpoints are under a new ignored output directory; existing files and
output directories are never overwritten. No production changes or B.4 work.
