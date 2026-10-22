# D.1 experimental design — offline selection only

This design was fixed in `configs/experiments/keyframes_d1.json` before evaluating the expanded pilot. It reuses the existing A.3 source registry, full native frame indices, exact-PTS decoder/display rotation, and B.1b geometry adapter. No geometry thresholds, model, classifier input preprocessing or production scanning code change. The primary comparison is all four strategies on the same three-second candidate grid, per policy and budget; not a comparison between different localization algorithms invented for D.1.

## Source identity and reproducibility

Use `a3_pilot_01_sources_v2.json` and its bound primary native manifest. Verify source/manifest hashes, exact original frame PTS/time base, decoded display dimensions and registered session/position groups. Choose the nearest indexed original PTS to a regular source-relative grid, breaking ties toward the earlier indexed frame; never derive timestamps from nominal FPS. Duplicate targets resolve to one source frame. The default is 3,000 ms and at most 256 candidates/session, explicitly configurable. Excess candidate requests fail instead of silently truncating. Videos are read-only; extracted PNGs, original decoded-pixel hashes and source-file hashes are separate derivative provenance.

Fresh run/output directories prevent overwriting previous results. Geometry work resumes from completed per-frame pairs; interrupted attempts receive new directories. Both policy results bind the exact frame/source/derivative and config/evaluator versions. Snapshot hashes protect existing frozen artifacts, native sources and production code. Sampling source copies document the earlier sampling version; final selection source hashes independently document the feature/selector version. A sampling-source digest difference does not imply a source-video difference.

Compressed and HD variants retain the **same capture-session, physical-position and cohort independence groups**. They are source representations, not additional independent samples. The pilot remains development-only, `training_eligible=false`, `final_holdout=false`. No FEN fields enter selected-frame files or selection functions. Ground-truth fields in the source registry are unused for ranking.

## Geometry states and safeguards

The adapter invokes the existing legacy/default and opt-in marked-board policies unchanged, each on the same PNG. Recovery is explicit, never promoted to default. Legacy successful source polygons must remain identical under the recovery wrapper.

The selector requires completion, a validated quarter-turn orientation, all 64 canonical IDs, finite convex nondegenerate square polygons within source-image bounds, and a convex outer boundary. Stale frame/source bindings are invalid. Corner-order errors are unresolved rectification failures. Localization/orientation failures and runtime exceptions cannot become keyframes.

- `completed_unverified`: structural gates passed; no certified geometric correctness.
- `visually_verified`: exact frame-bound, human-attributed annotation, after all structural gates.
- `unresolved`: localization, orientation, rectification, grid or feature failure.
- `rejected_incorrect`: exact frame-bound visual review explicitly rejects completed geometry.
- `invalid_geometry_binding`: stale or other-frame geometry; never selected.

Manual review schema: `offline_keyframe_geometry_reviews_v1`, with `reviews` entries containing `policy`, `status`, `reviewer`, `reviewer_kind`, and `binding` (`session_id`, `frame_id`, `pts`, `time_base`, `source_video_sha256`, `derivative_sha256`). `visually_verified` requires `reviewer_kind=human`; assistant inspection alone does not certify it. Pass a new review file with `evaluate_keyframe_selection.py --reviews PATH` into a fresh output directory. Reviews cannot repair invalid or unresolved geometry. Earlier nearby-timestamp overlays are not transferable annotations.

## Explainable quality and view signals

Warp the fitted board plane to **512×512, bilinear interpolation**, solely for quality measurement. This does not change recognition crops. Restricting metrics to the board prevents a sharp table/background from hiding board blur. These are image heuristics, not calibrated correctness probabilities.

| Signal | Definition | Use/limitation |
|---|---|---|
| Sharpness | grayscale Laplacian variance | logarithmic saturation at 1500; blur proxy affected by board texture, source scale and resampling |
| Contrast | board grayscale p90−p10 | normalized by 100; piece colors and lighting confounded |
| Exposure | fraction of board luma <10 or >245 | prefer fewer extremes; not a physical lighting measurement |
| Board size | fitted board area / image area | saturates at 0.35; cannot prove true board visibility |
| Perspective | shortest / longest fitted board edge | prefer mild foreshortening; not camera elevation |
| Direction anisotropy | Sobel structure-tensor eigenvalue ratio | diagnostic only; board texture prevents reliable directional-motion classification |
| Temporal motion | adjacent eligible board-center velocity / view-descriptor change within 7 s | diagnostic only, not tracking or drift ground truth |
| Geometric view | projected canonical a4→h4 axis angle; log adjacent/opposite edge ratios | diversity proxy; confounds camera roll and polygon errors |
| Appearance | 64-bit board dHash | redundancy proxy; similar piece arrangements can collide |

Weighted quality: 0.35 sharpness, 0.20 contrast, 0.15 exposure, 0.20 board size, 0.10 perspective. Experimental quality gates are sharpness≥25, board fraction≥0.04, extreme fraction≤0.45. These permissive engineering defaults were not fitted on FENs or final-holdout outcomes. Geometry thresholds are untouched. All gate outcomes, raw metrics and quality components are retained. No natural quality rejection is fabricated if the pilot contains no frame failing these gates.

View distance is `0.75 * wrapped_axis_angle_difference/pi + 0.25 * min(1, log_shape_L2/2)`. Exact pixel repeats are redundant. Otherwise a hard duplicate requires both dHash Hamming distance≤6 and view distance<0.12. Report soft similar pairs using view distance<0.12 alone. Physical elevation and occlusion remain unknown, even when the fitted grid lies inside the image. No precise camera poses are invented.

## Compared selectors

Every strategy applies identical geometry/quality gates and duplicate exclusion, deterministic ties (merit, quality, earlier PTS, frame ID), budgets **5, 10, 20** and retains chronological output plus greedy selection trace.

1. `uniform`: targets uniformly span the full candidate interval; at each target, choose the nearest remaining eligible nonduplicate. It is a usable-frame temporal baseline, not an instruction to select broken geometry.
2. `quality`: choose the highest remaining quality score.
3. `quality_temporal`: quality ranking with a 5 s minimum separation.
4. `quality_diverse`: greedily maximize `0.45 * quality + 0.55 * minimum_view_distance_to_selected`, with 2 s minimum separation. The first choice is quality-only.

A budget is an upper bound: never add invalid or duplicate frames to reach it. Greedy packing is order-dependent and not guaranteed monotonic across budgets. Preserve rejected-frame reasons (`redundant_with_selected`, `temporal_separation`, `budget_or_rank`, specific geometry/quality failures). No automatic side quota, budget adaptation or scan-completion logic is implemented.

## Evaluation and decision rule

Primary evidence: per-session counts, geometry/failure stages, selected timestamps/time span/gaps, mean quality, duplicate/similar pairs, predicted orientation distribution, view distance, edge-ratio range and CPU cost. Compare matched counts where possible; a larger set is not automatically better. Prefer a compact diverse set if it reduces proxy redundancy without material quality or side-coverage loss; retain uniform as a baseline. Do not tune parameters on the final holdout. No parameter sweep was run on this pilot.

Secondary evidence: reuse the frozen 12 timestamp-matched HD/compressed A.3 examples/session, with their original source/config-bound geometry caches. This controls recording identity; it does not control prior A.3 quality preselection and is not a second independent population. Twelve-candidate pools cannot support meaningful selection of twenty distinct frames. Retain secondary manifests as `robustness_comparison_only` for downstream tools.

Diagnostics: five selected contacts and timelines, all-strategy contact sheets, per-candidate states/reasons and raw grid overlays. Assistant gross visual inspection is recorded separately from human-certified geometry. Human-reviewed fine geometry remains unavailable. No full-board or per-square recognition is run, and no evidence/fusion/tracking/completion stage is implemented.

The selected-manifest schema is `offline_selected_keyframes_d1_v1`. `scripts/validate_keyframe_manifest.py` validates source hashes, exact PTS, pixels, grouping, geometry bindings and verification status before consumption. D.2 should preserve these fields, geometry uncertainty and explicit policy choice while collecting evidence in a separate task.
