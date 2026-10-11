# Day 9: square labels with piece context

The supervised target is the piece **whose base occupies the centered square**,
including `empty`. Nearby pieces may appear in the same crop. FEN supplies labels
only after the board orientation is resolved and its grid and piece placement
are visually reviewed. Rectified images now preserve the camera-facing view;
labels and chess coordinates follow the detected orientation instead of rotating
the pixels to White-at-bottom.

| White side | Top-left | Top-right | Bottom-left | Bottom-right |
|---|---|---|---|---|
| Bottom | a8 | h8 | a1 | h1 |
| Left | a1 | a8 | h1 | h8 |
| Top | h1 | a1 | h8 | a8 |
| Right | h8 | h1 | a8 | a1 |

`label_rotation` is the number of counterclockwise quarter turns that would put
White at the bottom; it is used to map labels, not applied to these image pixels.
Every crop retains image-relative row/column plus its actual `chess_square`.
Future inference must use the same camera-view convention and map predictions
back to canonical chess coordinates. The Week 1 CLI still offers its original
White-at-bottom output; it is not the new dataset's camera-view output.

The previous dataset's side-facing rotations were correct. Its migration restores
camera-facing pixels with exact quarter turns and applies the inverse to the
label matrix. Alignment rejection decisions remain unchanged. Rotation does
not repair inaccurate corners or three-dimensional perspective distortion.

Each accepted board produces 64 lossless 500×500 context crops, 100×100 tight
views, and source-coverage masks. The target occupies `[200,200,300,300]` in the
context crop. Two squares of padding on each side preserve much more of tall
pieces, including pieces on the playing area's edges. These windows reuse the
Week 1 context warp; they do not infer invisible content or remove occlusion.
Annotated review sheets are separate files, never classifier inputs.
Review sheets retain each board at its full 800×800 resolution and each context
crop at 500×500, arranged in two columns. Lossless PNG sheets are saved alongside
JPEGs. Individual labeled boards are also available in
`outputs/week1/day9_review/boards/`; open these at native size for the clearest
view. Coordinates and piece labels use separate, larger annotations.

## Reproduce

```bash
.venv/bin/python scripts/build_dataset.py --prepare-only
.venv/bin/python scripts/inspect_dataset.py --boards
# Review outputs/week1/day9_review/position_*.jpg before approving labels.
.venv/bin/python scripts/build_dataset.py
.venv/bin/python scripts/inspect_dataset.py --count 5
# Re-display the individually approved samples:
.venv/bin/python scripts/inspect_dataset.py --reviewed-only
```

`data/metadata/day9_board_reviews.json` maps image filenames to `signature`,
`alignment_correct`, `labels_correct`, reviewer and notes. Copy the signature
from `data/classification/prepared_boards.json` **after** inspecting that board.
The signature binds source pixels, normalized pixels, coverage mask, FEN,
orientation and board bounds. Preparation caches source/settings/code versions;
changed versions require another review. Detection acceptance alone is insufficient.

`data/metadata/board_corner_overrides.json` stores manually reviewed inner-board
corners in original-photo pixels (TL, TR, BR, BL), bound to the photo SHA-256.
The builder uses these instead of automatic corners and rejects stale overrides.
`position_004_var_05.jpg` has an override because its previous playing-area grid
shifted the f6 target onto the pawn on f7. Its FEN was correct. All 64 crops for
that photograph were regenerated after correcting the geometry. This manual
correction does not establish improved automatic detector accuracy.

The subsequent office-wide audit corrected all 54 photographs in variations
05, 06 and 07. The earlier detector accepted some frame/partial-grid boundaries
and rejected many side-facing views; only 12 office boards had been included.
That revision used individually annotated inner-board corners and produced
158 boards / 10,112 crops. Later image-supported polygon fitting supersedes those
fixed-grid approvals; the counts above describe the previous revision.
`data/metadata/office_localization_audit.json` records before/after coverage.
Individual full-resolution previews include variation 06 for every position.

The generated `data/classification/crops.json` contains one record per crop:
paths, position and source image, chess square, row/column, canonical class,
target bounds, coverage fractions, hashes and review status. Split is explicitly
unassigned until Day 10. Generated image assets and the large manifest are ignored
by Git; compact review records and results are versioned.

Automatic border orientation is preferred. For these registered capture photos,
known capture orientation is an explicit offline labeling fallback when markings
are uncertain. The manifest identifies that fallback; it does not improve or
inflate automatic orientation performance on unseen photographs.

`day9_visibility_reviews.json` optionally maps crop IDs to `crop_sha256`,
`board_review_signature` and `status`: `sufficient`, `partial`, `severely_occluded`
or `clipped`. Matching reviews are applied on rebuild. A target outside the source
coverage mask is flagged, and severe/clipped examples are not visually approved.
Unreviewed crops stay unreviewed; inspecting samples does not certify every crop.

## What this approach can establish

The saved edge-king comparison shows a tight crop dominated by the base, while
its context crop includes the visible body and head. Keeping the full 500px crop
allows later preprocessing changes without rebuilding geometry or ground truth.
At the current 224px model input, the target square is about 45px wide; at 384px
it would be about 77px. Compare those input sizes during model evaluation if fine
details or neighboring pieces cause confusion. No input-size improvement has yet
been measured, so the baseline configuration remains unchanged.

This is a viable dataset format for a baseline, not a proven recognition result.
Crowded or steep-angle photographs still contain genuine occlusion and distorted
piece shapes. A board-plane homography cannot rectify a three-dimensional piece.
A classifier may confuse neighboring pieces; compare tight/context inputs or a
future dual-view classifier only if measured failures warrant it. Fully hidden
features need another view or more visible training examples, not larger padding.

Current measured counts and review coverage are recorded in
`data/metadata/day9_dataset_summary.json`. No model has been trained and no
classification accuracy is claimed. Day 10 must split by position, not by crop.

### Image-supported polygons and full-board previews

Prepared boards now retain their padded context and board borders in the label
preview. `grid_geometry` stores 81 line intersections, 64 square polygons, and
visible-boundary evidence. Each boundary is fitted jointly across alternating
square-color transitions; weak, occluded boundaries fall back to the seed lattice.
These are estimates, not independently measured ground truth. Lens distortion is
not calibrated by this fit.

Square crops use each polygon's actual location, retain surrounding piece context,
and record `target_polygon` alongside its enclosing `target_bounds`. Green preview
outlines follow the polygon; they are never embedded in training pixels. Where
source geometry is available, `_source_grid.png` overlays the same boundaries on
the original photo. Full-resolution `_labels.png` retains the borders.

Geometry changes invalidate previous board and visibility reviews. Run
`.venv/bin/python scripts/build_dataset.py --preview-crops` to generate
`candidate_crops.json` for visual inspection without treating candidates as approved
training data. Review the original-photo overlay and normalized preview together
before updating a board review signature. The default build excludes stale reviews.

The family fit searches offset and spacing before fitting individual boundaries.
Known camera orientation supplies the expected light/dark checker phase, preventing
a one-rank or one-file slip. A robust projective lattice rejects inconsistent
intersections and estimates hidden outer boundaries. Small supported residuals
are retained in the square polygons. This corrects the board plane; piece height
and occlusion remain separate classification problems.

Use `.venv/bin/python scripts/inspect_dataset.py --candidates` for crop contact
sheets, and `--boards` for padded full-board sheets. Candidate fit status and
evidence strength are recorded independently from human alignment approval.
