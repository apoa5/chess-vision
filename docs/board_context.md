# Preserve tall pieces at board edges

A homography corrects the flat board plane. A piece's base lies on that plane,
but its top can project outside the square or even outside the playing area.
The original 800×800 warp discarded those outside pixels. Padding that cropped
image afterward cannot restore them.

The pipeline now extends the destination canvas while retaining the detected
inner-grid corners. With the defaults, the 800×800 playing area lies inside a
1200×1200 context image, at exclusive bounds `[200, 200, 1000, 1000]`. The
same transform samples the original photo directly, including the border and
background. The grid retains its 100-pixel square size; the whole 1200-pixel
canvas must never be divided into eight squares.

```yaml
board:
  normalized_size: 800
  context_margin_squares: 2.0
  crop_padding_squares: 2.0
```

Each value is measured in normalized square widths, from zero to four. Crop
padding cannot exceed the context margin. A margin of two produces 200 pixels
on each side at the default board size. Padding of two produces 500×500 context
crops: a 100×100 target square plus 200 pixels in each direction. Increase these
values together if a particular view needs more context; the defaults are not
a guarantee that every piece fits.

## Outputs and coordinates

```bash
python scripts/process_board_image.py data/raw/position_004_var_04.jpg
python scripts/rectify_board.py data/raw/position_004_var_04.jpg
```

Both commands save the exact playing-area image plus these sibling artifacts:

- `<stem>_context.png`: extended image sampled from the original photo.
- `<stem>_context_grid.png`: indexed 8×8 grid confined to the playing area.
- `<stem>_context_valid.png`: 255 where source pixels exist, 0 outside the photo.
- `<stem>_context.json`: canvas dimensions, exclusive playing-area bounds, and artifact paths.

The full pipeline additionally saves 64 overlapping crops under
`outputs/square_crops/<stem>/context/`. Its `squares.json` records the context-canvas
bounds, row/column, local `target_bounds`, coverage fraction, and coverage-mask
filename. The existing exact square crops remain available in the parent folder.
The standalone `extract_squares.py` works on the exact playing-area image; it
cannot recover lost context from an already clipped image.

Black pixels outside the photograph are identified by masks, rather than
replicating border pixels and presenting them as captured content. Coverage
masks describe geometric source coverage; cubic interpolation may blend samples
at the source boundary. All crops are independent copies, and annotations never
enter their pixel data.

## Verification and limitations

A synthetic regression test places a colored piece above the playing-area edge
and verifies that the extended warp retains its actual source pixels. Other
tests check the playing-area bounds, unchanged grid coordinates, target-square
location in overlapping crops, masks outside the source, invalid settings, and
pipeline metadata. The real `position_004_var_04` example was inspected: the
context image and `r0_c7` context crop retain the tall black piece's head that
was cut off in the tight playing-area image.

Translating the transform onto a larger canvas can introduce tiny interpolation
rounding differences in the central image (maximum difference of one intensity
level in that example). Existing pixel-bound warp reviews can therefore become
stale; evaluation correctly requires a fresh review rather than silently
reusing their correctness labels. Historical Week 1 snapshots are retained.

Padding retains available pixels, not 3D shape. Tall pieces may remain tilted,
stretched, partly obscured, or outside the raw photograph. More overhead capture
reduces this effect; keep headroom around the entire board in the original shot.
Incorrectly detected grid corners still produce incorrect square assignments.

For Week 2, classify the central target square of each context crop, using the
known row/column and target bounds. Neighboring pieces are context, not additional
labels for that sample. Build training and inference with the same padding and
preprocessing, and visually inspect labels before training. Larger windows can
help preserve silhouettes but also introduce distracting neighbors; recognition
improvement must be measured on validation data. No classifier is trained here.
