# D.1 — Offline video keyframe selection

D.1 is complete for offline development. A deterministic three-second grid generated **124 candidates** across the five original HD recordings. Unchanged legacy geometry completed on **52/124 (41.9%)**; explicit opt-in marked-board recovery completed on **94/124 (75.8%)**. Every completed candidate passed the selector's additional finite/bounded canonical-grid checks. Completion is **not verified geometry correctness**.

The recommended starting experiment for D.2 is **quality_diverse, budget 5**, with opt-in recovery explicitly requested. It chooses five provisional frames per recording, removes measured similar-view pairs at this budget, and retains the original session/position groups. Keep uniform-5 as the comparator and budget-10 as sensitivity analysis. This is an engineering recommendation, not a recognition or generalization result.

## Inputs and scope

The registered high-resolution sources in `data/metadata/a3_pilot_01_sources_v2.json` are primary; original videos, native indices, reference photos, owner FEN/unchanged-position attestations, compressed variants and earlier results remain unchanged. No FEN, classifier output or class labels enter scoring or selection. Existing owner attestations are retained; no new ground truth is created. This is one correlated development pilot with five physical positions; four also match historical training placements. It is not five independent capture cohorts or 124 independent observations. No final holdout was accessed.

Read/design sources: root `VISION_RESEARCH.md` and `IMPLEMENTATION_PLAN.md` (the actual repository paths), A.3 quality/HD reports, B.1/B.1b geometry reports, native decoding/indexing, replay and inference/crop paths. See [predeclared design](keyframes_d1_design.md) for signals, safeguards and limitations.

## Candidate and geometry results

| Position | Candidates | Legacy completed | Opt-in completed | Opt-in unresolved |
|---|---:|---:|---:|---:|
| starting | 26 | 12 | 20 | 6 |
| opening | 25 | 8 | 18 | 7 |
| middlegame | 26 | 6 | 21 | 5 |
| endgame | 24 | 13 | 18 | 6 |
| puzzle | 23 | 13 | 17 | 6 |

Legacy failures: **65 localization, 6 semantic-orientation, 1 rectification/corner-order exception**. Opt-in failures: **29 localization, 1 rectification exception**, with no remaining orientation-stage failures in this grid. The exception is opening **24.003333 s**, under both policies: `Corners must be ordered TL, TR, BR, BL; use order_corners first.` The adapter's generic `dependency_or_runtime` is explicitly reclassified as `rectification_validation` for selection diagnostics. Exceptions do not interrupt the recording or become eligible. All 52 legacy completions have identical source-square polygons under recovery (the recovery wrapper delegates successful legacy results unchanged).

All 94 opt-in completions passed the configured minimum sharpness, board fraction and exposure gates: **zero natural quality-gate rejections**. Do not infer absence of motion blur: these are permissive image proxies. Diagnostics show the lowest-sharpness *passing* frames, not invented blur failures. Synthetic regression tests establish that the board-only sharpness metric responds to blur and cannot be rescued by a sharp background.

Full unusable-frame inventory, including secondary paired data: [keyframes_d1_failures.json](keyframes_d1_failures.json). Reasons, exact PTS, geometry-source references and source hashes are retained.

## Strategy comparison

All strategies use the same candidate pool, geometry checks, quality gates and hard near-duplicate guard. Uniform is a safe temporal baseline over the full recording interval, choosing the nearest still-eligible frame at each target; it does not select invalid frames to fill gaps. Quality uses descending score. Temporal adds 5 s separation. Diverse uses 2 s separation and a fixed 0.55 novelty weight. Budgets are ceilings.

Quality is a unitless **uncalibrated heuristic**, not confidence or independent ground truth. Similar pairs have geometric-view distance <0.12; the hard guard additionally requires dHash distance <=6 (or exact pixel identity). All selected sets contain zero hard duplicate pairs by construction, so this is a shared safeguard rather than an advantage claimed for one method. Values below average quality/span equally across the five sessions; counts sum sessions, never form statistical independent-frame estimates.

| Policy | Budget/session | Strategy | Selected total | Mean Q | Similar pairs | Mean time span (s) |
|---|---:|---|---:|---:|---:|---:|
| legacy | 5 | uniform | 25 | 0.807 | 5 | 49.2 |
| legacy | 5 | quality | 25 | 0.823 | 5 | 48.6 |
| legacy | 5 | quality_temporal | 23 | 0.826 | 4 | 48.6 |
| legacy | 5 | quality_diverse | 24 | 0.824 | 1 | 46.8 |
| legacy | 10 | uniform | 31 | 0.799 | 7 | 48.6 |
| legacy | 10 | quality | 31 | 0.812 | 8 | 48.6 |
| legacy | 10 | quality_temporal | 26 | 0.817 | 5 | 48.6 |
| legacy | 10 | quality_diverse | 30 | 0.815 | 6 | 47.4 |
| legacy | 20 | uniform | 31 | 0.800 | 8 | 48.6 |
| legacy | 20 | quality | 31 | 0.812 | 8 | 48.6 |
| legacy | 20 | quality_temporal | 26 | 0.817 | 5 | 48.6 |
| legacy | 20 | quality_diverse | 30 | 0.815 | 6 | 47.4 |
| marked-board | 5 | uniform | 25 | 0.812 | 6 | 61.8 |
| marked-board | 5 | quality | 25 | 0.842 | 4 | 50.4 |
| marked-board | 5 | quality_temporal | 25 | 0.841 | 5 | 56.4 |
| marked-board | 5 | quality_diverse | 25 | 0.835 | 0 | 49.2 |
| marked-board | 10 | uniform | 44 | 0.794 | 19 | 60.0 |
| marked-board | 10 | quality | 43 | 0.809 | 21 | 60.0 |
| marked-board | 10 | quality_temporal | 36 | 0.814 | 14 | 60.0 |
| marked-board | 10 | quality_diverse | 43 | 0.804 | 18 | 56.4 |
| marked-board | 20 | uniform | 45 | 0.794 | 24 | 60.0 |
| marked-board | 20 | quality | 45 | 0.807 | 24 | 60.0 |
| marked-board | 20 | quality_temporal | 36 | 0.814 | 14 | 60.0 |
| marked-board | 20 | quality_diverse | 44 | 0.804 | 20 | 56.4 |

At matched budget 5 with opt-in recovery, diversity selects **25 frames**, mean quality **0.835**, **0 similar pairs**, versus uniform's **25 / 0.812 / 6**, and quality-only's **25 / 0.842 / 4**. The tradeoff is shorter mean temporal coverage (**49.2 s versus uniform 61.8 s**). At 10 and 20, redundancy increases and returns saturate: diversity selects 43 and 44 frames total, with 18 and 20 similar pairs. More frames are not automatically better. Greedy duplicate packing depends on selection order; larger uniform budgets can occasionally choose fewer admissible frames within a session. No monotonic or quota-filling guarantee is claimed.

### Camera-side balance and temporal trace

The following are **predicted semantic-orientation bins**, not externally verified camera poses. Labels are white-near, black-near, white-left, white-right. Geometric diversity uses the projected canonical file-axis angle and foreshortening proxies; camera roll can affect it. Exact camera elevation, occlusion and viewpoint correctness remain unknown. The edge-ratio proxy ranges, dominant predicted-side fractions, pair distances and per-frame image-space motion diagnostics are in the machine-readable results.

| Position | Available opt-in predicted bins | Selected bins (diverse-5) | Selected original timestamps (s) |
|---|---|---|---|
| starting | {'white_near_camera': 10, 'white_right': 6, 'black_near_camera': 3, 'white_left': 1} | {'black_near_camera': 1, 'white_left': 0, 'white_near_camera': 2, 'white_right': 2} | 9.002, 21.003, 27.003, 39.005, 45.007 |
| opening | {'white_near_camera': 8, 'white_right': 5, 'black_near_camera': 4, 'white_left': 1} | {'black_near_camera': 1, 'white_left': 1, 'white_near_camera': 1, 'white_right': 2} | 0.000, 30.005, 39.005, 48.007, 63.010 |
| middlegame | {'white_near_camera': 9, 'white_right': 5, 'black_near_camera': 5, 'white_left': 2} | {'black_near_camera': 1, 'white_left': 1, 'white_near_camera': 1, 'white_right': 2} | 3.000, 18.003, 27.003, 33.005, 54.008 |
| endgame | {'white_near_camera': 6, 'white_right': 5, 'black_near_camera': 6, 'white_left': 1} | {'black_near_camera': 1, 'white_left': 1, 'white_near_camera': 2, 'white_right': 1} | 3.000, 9.002, 15.002, 33.005, 51.007 |
| puzzle | {'white_near_camera': 8, 'white_right': 4, 'black_near_camera': 4, 'white_left': 1} | {'black_near_camera': 1, 'white_left': 1, 'white_near_camera': 1, 'white_right': 2} | 3.000, 18.003, 27.003, 36.005, 51.007 |

Starting selects only three of the four available predicted bins at budget 5; **all four methods** do so. Diverse-10 covers all four for every recording. The other four recordings cover all four predicted bins at diverse-5. There is no repeated exclusion of the same bin across all positions, but the starting omission shows that quality/diversity does not guarantee side quotas. No parameters were tuned after observing this omission. Side coverage and elevation should be manually annotated on new capture sessions before making physical coverage claims.

## Paired compressed robustness comparison

The original **60 timestamp-paired A.3 frames per representation** are reused with cached frozen geometry, not rerun as independent recordings. They were quality-preselected previously and differ from the new 124-frame uniform candidate pool. Their matching is inherited from the approved HD timestamp-alignment manifest; actual PTS/source identities stay distinct. These are secondary robustness checks, not equal-population comparisons with the primary grid. Budget 20 on twelve candidates/session simply saturates; all 360 strategy/session cases are available in the CSV.

| Source | Policy | Strategy @5 | Geometry completed | Selected | Mean Q | Similar pairs |
|---|---|---|---:|---:|---:|---:|
| paired_frozen_high | legacy | uniform | 30/60 | 23 | 0.831 | 1 |
| paired_frozen_high | legacy | quality_diverse | 30/60 | 23 | 0.836 | 1 |
| paired_frozen_high | marked-board | uniform | 53/60 | 25 | 0.808 | 3 |
| paired_frozen_high | marked-board | quality_diverse | 53/60 | 25 | 0.840 | 1 |
| paired_frozen_compressed | legacy | uniform | 32/60 | 23 | 0.748 | 3 |
| paired_frozen_compressed | legacy | quality_diverse | 32/60 | 23 | 0.752 | 2 |
| paired_frozen_compressed | marked-board | uniform | 51/60 | 25 | 0.726 | 4 |
| paired_frozen_compressed | marked-board | quality_diverse | 51/60 | 25 | 0.752 | 0 |

HD and compressed versions share session, position and independence groups and must never be split as independent samples. HDR HEVC-to-BGR8 conversion is unchanged from A.3, without new tone mapping; paired sharpness/quality differences can reflect both resolution and color conversion. No piece-recognition utility is inferred from these scores.

## Diagnostics and geometry review

Five recommended [contact sheets](keyframes_d1_examples/), five SVG selection timelines, representative unresolved/redundant/low-sharpness diagnostics and [ten grid overlays](keyframes_d1_examples/selected_grid_review.jpg) accompany this report. Full interactive navigation is a static local report at `outputs/scanning/d1_pilot_run1/selection_run2/index.html`; it changes no browser scanning code.

I inspected all 25 recommended contact thumbnails and ten full grid overlays (first/last selected per position). No grossly incorrect completed grid was identified in that limited assistant review. Fine alignment and per-frame semantic orientation remain **unverified**: zero selected frames receive human-certified `visually_verified`. Selection is provisional geometric usability, not geometric correctness. Prior A.3 reviews are not transplanted to nearby timestamps. Exact bound manual annotations can mark incorrect geometry rejected; visually verified status requires human attribution. Rejected incorrect geometry is distinct from unresolved geometry; no new bound incorrect-completed annotation was warranted by this sample.

Sparse sampling can miss brief blur, occlusion or orientation failures between candidates. Finite/in-frame convex cells cannot detect every plausible wrong board polygon. Projected-view novelty/dHash can both miss and over-suppress useful 3D view differences. These limitations are retained in metadata, rather than hidden by the completion count.

## Runtime and processing cost

Original source indices contain **10,896 HD frames**; 124 candidates (1.14%) received new geometry evaluation. Exact-PTS seeking decodes keyframe pre-roll too; those intermediate decode counts were not instrumented. Candidate extraction took **140.65 s**. Recorded unchanged geometry calls: legacy **394.55 s**, recovery **842.64 s** (the recovery wrapper includes its own legacy attempt). Both policies were deliberately evaluated, so total new geometry time is their sum, not just the recovery number.

Feature extraction/360 selection evaluations: **65.17 s**; rendering **38.65 s**; final protected-file hashing **80.84 s**; analysis total **184.69 s**. Median selector call **0.726 ms**. Costs are measured wall times on this CPU environment with OpenCV single-threaded, affected by I/O and concurrent verification; not phone latency. Cached 120-frame secondary geometry costs are historical, not new processing. Source/protected-hash setup and worker initialization add overhead beyond summed calls. No classifier ran, model size changed or classifier latency was measured.

## Reproduction and validation

Use the existing `.venv` and `requirements-pilot-video.txt` isolated decoder convention; no new dependency installation was required. From repository root, use a **fresh** run/output directory:

```bash
.venv/bin/python scripts/select_video_keyframes.py candidates --run outputs/scanning/d1_reproduce
.venv/bin/python scripts/select_video_keyframes.py geometry --run outputs/scanning/d1_reproduce --limit 8
# Repeat geometry command until all 124 candidate pairs are complete; completed pairs resume safely.
.venv/bin/python scripts/evaluate_keyframe_selection.py --run outputs/scanning/d1_reproduce --output-dir outputs/scanning/d1_reproduce/selection
.venv/bin/python scripts/validate_keyframe_manifest.py data/metadata/keyframes_d1_pilot_v1.json
.venv/bin/python -m pytest tests/test_video_keyframes.py tests/test_keyframe_manifest.py -q
```

Actual run: `outputs/scanning/d1_pilot_run1`, sixteen bounded geometry invocations (`--limit 8`), preserved `selection_run1`, final `selection_run2`. The final run changes diagnostics to the recommended budget and adds downstream provenance fields; all **360 selection outcomes match run1**. Initial sampling source copies and hashes live in `sampling_sources/`; final selection code hashes are in metrics.json. Candidate sampling itself is unchanged between these source versions.

Validation: **188 relevant Python tests passed**, followed by **49 focused tests passed** (38 overlap the 188; eleven new manifest-integrity tests); **39 existing browser tests passed**. Native candidate timestamps reproduced for all **124**, ten original-source pixel re-decodes matched exactly, and **360 reversed-input deterministic selector replays passed**. Protected **1,746 existing files**, including source media, frozen models, prior reports and browser/production code, retained their hashes. All **72 selected manifests passed** standalone validation. See run-local manifest validation and reproduction JSONs. No tests failed in the final validation runs.

The new standalone validator rechecks catalog/source/frame hashes, decoded pixels, PTS, session/position grouping, source variant, budget, geometry-result binding, policy/config/evaluator versions, finite features and verification status before D.2 consumption. It blocks stale/corrupt/misrepresented frames. No unverified completion is silently promoted.

The implementation plan's live-rendering/single-frame acceptance items are preserved by **no runtime integration** in this assignment: the renderer/tracker/recognition scheduler remain unchanged, and their existing browser tests pass. Sparse explainable *offline keyframe eligibility* is implemented; recognition scheduling on selected frames remains outside this offline task. No new live performance measurement is claimed.

Additional executed checks:

```bash
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
.venv/bin/python -m compileall -q src/scanning/video_keyframes.py scripts/select_video_keyframes.py scripts/evaluate_keyframe_selection.py scripts/render_keyframe_diagnostics.py scripts/validate_keyframe_manifest.py
git diff --check
```

The complete generated run occupies approximately **2.6 GB** (PNG derivatives, geometry attempts and diagnostics). This is an offline audit run, not a production storage/performance target. Preserve it outside Git with the original recordings; future D.2 can reuse the validated selected frames without rerunning geometry.

## Outputs and files changed

New implementation: `src/scanning/video_keyframes.py`; `scripts/select_video_keyframes.py`, `evaluate_keyframe_selection.py`, `render_keyframe_diagnostics.py`, `validate_keyframe_manifest.py`; `configs/experiments/keyframes_d1.json`; `tests/test_video_keyframes.py`, `test_keyframe_manifest.py`.

New reviewable artifacts: this report, `keyframes_d1_design.md`, `keyframes_d1_comparison.csv`, `keyframes_d1_results.json`, `keyframes_d1_failures.json`, `keyframes_d1_examples/`; `data/metadata/keyframes_d1_pilot_v1.json` and `keyframes_d1_index_v1.json`. Generated ignored run assets contain 124 PNG candidates, all 248 geometry calls, 72 selected manifests, 40 strategy contacts, five timelines, per-candidate decision reasons and run-local preservation/reproduction checks. Existing unrelated working-tree changes predated this task and were preserved. No production file was modified by D.1.

The recommended manifest is class-label-free and contains source hashes, original PTS/time bases, frame and geometry-reference hashes, group IDs, selection reasons/trace, unverified state, projected-view and quality descriptors. It contains **no per-square evidence or fusion**. Copying the manifest alone does not copy its linked ignored PNG/geometry assets; archive the ignored run directory alongside the original videos outside Git. The versioned index inventories all 72 manifest hashes. Do not commit the videos or generated run automatically.

## Acceptance checklist and D.2 recommendation

- [x] Offline deterministic configurable temporal candidates from registered original recordings; exact timestamps and preserved sources.
- [x] Explainable board-only quality, geometry, temporal and geometric-view signals; explicit unknown occlusion/elevation.
- [x] Both unchanged geometry policies; recovery opt-in; failures/exceptions excluded; completed/verified/unresolved/rejected states distinct.
- [x] Four strategies and budgets 5/10/20 compared on the same primary inputs; same-session compressed robustness comparison separate.
- [x] Five contact sheets, selection reasons, failure/redundancy examples, timelines and full machine-readable decisions/manifests.
- [x] Failure/runtime/redundancy/side-balance and budget sensitivity reported with correlated-development limitations.
- [x] Focused and regression checks passed; original hashes/protected files unchanged; no holdout, training, fusion, production/browser or checkpoint changes.
- [x] Stop after D.1; no commit or push.

Proceed to **D.2 per-square evidence collection**, keeping uniform-5 versus diverse-5 and explicit recovery policy as recorded experimental inputs; include diverse-10 only as sensitivity analysis. First validate the selected manifest, retain per-frame provenance/geometry uncertainty and manually review rare-side examples. Measure downstream recognition usefulness and empty/occupied errors before treating the heuristic selector as superior. D.1 has not measured full-board recognition, reconstruction improvement, true camera-pose coverage or independent generalization.
