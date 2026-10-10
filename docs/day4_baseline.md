# Day 4 board detection baseline

The detector targets the 32×32 cm, 8×8 playing area, excluding the wooden frame.
It uses existing OpenCV/NumPy dependencies; no trained model is involved.

## Method

1. Downsample only when above the configured maximum size, convert to grayscale,
   optionally apply CLAHE, blur, and run Canny edge detection.
2. Find contours on the edges and a lightly closed edge mask. Approximate
   polygons and retain convex quadrilaterals with plausible image area,
   opposing-side proportions, side lengths, and rectangularity. Reject image
   boundaries and deduplicate nearby contour candidates.
3. Consider the contour and several projective insets, since a visible contour
   may surround the wooden frame rather than the playing area. The inset values
   are hypotheses, not unconditional board corners.
4. Temporarily sample each candidate at 160×160 for scoring. Compare the median
   brightness of 64 cell centers against an alternating 8×8 pattern after
   removing row/column lighting gradients.
5. Score candidates using checker correlation (70%), contrast (20%), and area
   (10%). Apply explicit evidence thresholds and return the best accepted
   candidate in TL/TR/BR/BL order, scaled back to original-image pixels.

The scoring sample is internal to detection. The reusable Day 6 perspective
pipeline remains unimplemented. Candidate score is not a calibrated probability.
No candidate passing the thresholds produces `success: false`, no corners, and
a reason; image boundaries are never used as fallback corners.

## Initial run

On `week1_development_v1`, 13/20 photos produced accepted candidates and seven
reported failure. All twenty have debug overlays, working-resolution edges,
and candidate diagnostics. A snapshot is in
[`day4_baseline.json`](../data/metadata/day4_baseline.json).

Failures:

| Image | Reported failure |
| --- | --- |
| position_001_var_05.jpg | No plausible quadrilateral |
| position_001_var_07.jpg | No plausible quadrilateral |
| position_004_var_04.jpg | Checkerboard evidence below thresholds |
| position_005_var_03.jpg | No plausible quadrilateral |
| position_008_var_02.jpg | Checkerboard evidence below thresholds |
| position_009_var_03.jpg | No plausible quadrilateral |
| position_010_var_01.jpg | Checkerboard evidence below thresholds |

Overlays were reviewed together in a contact sheet. Accepted polygons identify
the board region in multiple overhead and perspective views; some boundaries
still include slivers of frame or miss edge pixels. This is a working baseline,
not a 90% accuracy claim or a guarantee of crop alignment. Corner ground truth
and a quantified localization evaluation have not been added.

## Next-day investigation

- Dense pieces and shadows interrupt contours and weaken checker evidence.
- Both office images failed; investigate contrast, edge continuity, and framing.
- Discrete inset hypotheses can leave corner offsets even when the correct
  board is selected. They need refinement before reliable square extraction.
- Low contrast, extreme perspective, heavy occlusion, and other board styles
  have not been validated. No outdoor data is available yet.

Day 5 should review the failures and boundary offsets, improve candidate
selection, and record manually verified correctness on the fixed subset.
