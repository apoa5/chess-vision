# Day 5 detection improvements and evaluation

## Changes and rationale

Day 4 failed on fragmented contours and dense-piece positions. Even accepted
quadrilaterals could include frame or cut into the playing area.

The detector now combines edge contours with warm wood-color mask hypotheses
and intersections of long [OpenCV Hough lines](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html).
Color masks propose geometry for this wooden board; they do not establish that
a region is a chessboard. Line candidates handle incomplete contours but can
also propose internal grid boundaries. Every source uses the same convexity,
area, proportions, and checker-evidence checks.

Checker scoring retains cell medians, adds pixel-level agreement with square
boundaries, and weights central-grid evidence to reduce the impact of occupied
back ranks. Alternating patterns immediately outside a proposed boundary are
penalized as evidence of clipping. A bounded, coarse-to-fine coordinate search
refines the three strongest hypotheses. Diagnostics identify the selected source,
line-quadrilateral count, accepted-candidate count, refinement shifts, and evidence.

No neural model or new dependency was added. Wood-color proposals can be disabled
for other boards; line proposals and refinement can also be disabled in
`config/settings.yaml`. This remains an offline development detector and is
slower than the original contour-only baseline. Candidate scores are not
probabilities, and thresholds refer to the current scoring formula.

## Evaluation

Run `python scripts/evaluate_board_detection.py` from the activated environment.
The script uses the unchanged `week1_development_v1` manifest and writes one
CSV/JSON record per image plus debug outputs under `outputs/board_detection/day5/`.

| Measure | Result |
| --- | --- |
| Development images | 20 |
| Accepted candidates | 16 |
| Visually approved localizations | 15 |
| Incorrect accepted candidates | 1 |
| No accepted candidate | 4 |
| Reviewed localization rate | 75% |
| Target | 90%; not met |

The historical Day 4 count was 13 accepted candidates, without verified accuracy.
Do not compare that count directly with the new 75% localization rate.

Reviews were performed by Codex using the contact sheet and full-resolution
inspection of dense/ambiguous candidates. The criterion is selecting the correct
8×8 playing area without missing a file/rank or including substantial wooden
frame. Small boundary offsets are allowed. This is qualitative localization
review, not measured corner error or validation of downstream square crops.

[`day5_review.json`](../data/metadata/day5_review.json) stores the decisions and
their signatures. [`day5_evaluation.json`](../data/metadata/day5_evaluation.json)
stores the run. Signatures include photo bytes, detector settings, and corners
rounded to 0.1 pixels. Changed settings, photos, or corners require renewed
inspection; incomplete reviews suppress the reported accuracy. Detector failure
is a known negative outcome. Adding images does not change the fixed subset.

## Remaining failures

| Image | Outcome and observed issue |
| --- | --- |
| position_001_var_05.jpg | Rejected; wood-frame/background ambiguity and glare weaken aligned-grid hypotheses |
| position_001_var_07.jpg | False positive; an internal line is selected, clipping a file and including frame |
| position_005_var_03.jpg | Rejected; dense pieces obscure the grid and interrupt contours |
| position_006_var_01.jpg | Rejected; densely occupied squares weaken the current grid score |
| position_009_var_03.jpg | Rejected; perspective, pieces, and shadows weaken grid evidence |

The lower result is recorded rather than hidden. Current classical scoring still
needs better global grid fitting and boundary disambiguation to reach the target.
Near-overhead captures with even lighting and unobstructed playing-area edges
would also make the controlled capture conditions easier; any additional captures
should supplement the dataset rather than silently replace difficult subset images.
There is no evidence yet that a learned detector is necessary.

The dataset is indoors and the subset is used for development, not an independent
test set. Do not infer outdoor or other-board performance from this result.
Before trusting perspective correction and square crops, revisit these failures
and inspect boundary alignment in the later-stage outputs.
