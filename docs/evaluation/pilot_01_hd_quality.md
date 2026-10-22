# A.3 pilot 01 — high-quality reprocessing and paired geometry comparison

All five higher-quality recordings are now registered as primary sources. **Higher source quality does not uniformly improve geometry reliability:** legacy completion falls from 32/60 to 30/60; opt-in completion rises from 51/60 to 53/60 with four rescues and two regressions. The reviewed rescues produce visually plausible grids, while six high-resolution incorrect localization polygons are blocked by orientation validation. Completion is not verified correctness.

## Measured source comparison

All five recordings fully decoded: **10,896 original frames versus 10,881 Telegram frames**. Formats are displayed portrait 1080×1920 HEVC versus 464×848 H.264, nominal 30 fps in both. Original timestamps have small 31.7–35 ms variations (max 35 ms), versus a 66.7 ms interval in each Telegram copy. Encoded originals are 1920×1080 with a −90° display matrix. Full codec/rate/size/hash rows are in [source CSV](pilot_01_hd_sources.csv); HDR metadata in [color metadata](pilot_01_hd_color_metadata.json).

| Position | Duration s: Telegram → original | Frames: Telegram → original | File MB: Telegram → original | Sequence offset ms |
|---|---|---|---|---|
| starting | 75.333 → 75.410 | 2259 → 2262 | 15.82 → 83.44 | 74.17 |
| opening | 73.367 → 73.443 | 2200 → 2203 | 15.38 → 81.09 | 70.83 |
| middlegame | 75.467 → 75.543 | 2263 → 2266 | 15.82 → 83.29 | 70.83 |
| endgame | 70.200 → 70.277 | 2105 → 2108 | 14.71 → 77.56 | 71.67 |
| puzzle | 68.500 → 68.575 | 2054 → 2057 | 14.39 → 75.78 | 71.67 |

All 60 final pairs use nearest real PTS within **7.5 ms** of their offset-adjusted targets. All 60 feature registrations passed spatial-support gates. Every candidate retains both source association and exact PTS; originals and their full ordered indexes remain preserved.

## Paired completion and failure stages

| Position | Legacy Telegram | Legacy original | Opt-in Telegram | Opt-in original | Original legacy failures |
|---|---|---|---|---|---|
| starting | 6/12 | 5/12 | 9/12 | 11/12 | {'orientation': 2, 'localization': 5} |
| opening | 5/12 | 5/12 | 10/12 | 10/12 | {'localization': 7} |
| middlegame | 5/12 | 5/12 | 10/12 | 11/12 | {'orientation': 2, 'localization': 5} |
| endgame | 9/12 | 7/12 | 11/12 | 10/12 | {'localization': 4, 'orientation': 1} |
| puzzle | 7/12 | 8/12 | 11/12 | 11/12 | {'localization': 3, 'orientation': 1} |

| Policy | Telegram completed | Original completed | Retained | Rescued | Regressed | Still unresolved |
|---|---|---|---|---|---|---|
| legacy | 32/60 | 30/60 | 24 | 6 | 8 | 22 |
| marked-board | 51/60 | 53/60 | 49 | 4 | 2 | 5 |

Legacy rates: **53.3% → 50.0%**; opt-in: **85.0% → 88.3%**. Low legacy failures were 22 localization, 5 orientation and 1 rectification; high legacy failures are 24 localization and 6 orientation. Low opt-in failures were 8 localization and 1 rectification; high opt-in failures are 7 localization. No high-resolution corner-order exception, grid rejection or other processing exception occurred. This does not fix the existing corner-order defect: the previously exceptional 21 s starting frame simply avoids it at the matched high-quality input. The raw historical stage `dependency_or_runtime` for that low frame is diagnosed as rectification validation in the paired CSV.

High-quality opt-in recovery was attempted on 30 frames and rescued 23; all 30 high-quality legacy successes retain identical returned grids under opt-in. All six high-quality orientation rejections have visibly incorrect accepted localization polygons in the reviewed overlays, so their first failing gate must not be misinterpreted as a correct-geometry semantic-orientation deficit.

Five physical positions/sessions are the recording units; the 60 frames are correlated samples and the two representations are not independent captures. Rates are descriptive development observations without a significance/generalization claim.

### Viewpoint strata

Side labels below are reused qualitative observations from the previous pilot, not calibrated pose or verified semantic ground truth. Transitions remain uncertain; no angle metadata was invented.

| Observed side, unverified | Frames | Legacy Telegram → original | Opt-in Telegram → original | Original legacy failure stages | Original opt-in failure stages |
|---|---|---|---|---|---|
| white_near | 16 | 11 → 7 | 15 → 16 | {'orientation': 5, 'localization': 4} | {} |
| transition_or_uncertain | 18 | 4 → 7 | 13 → 15 | {'localization': 11} | {'localization': 3} |
| white_right | 9 | 7 → 7 | 9 → 9 | {'localization': 2} | {} |
| black_near | 10 | 7 → 7 | 10 → 10 | {'localization': 2, 'orientation': 1} | {} |
| white_left | 7 | 3 → 2 | 4 → 3 | {'localization': 5} | {'localization': 4} |

### Remaining opt-in failures

| Position / candidate | Telegram → original time s | Recovery rejection | Previously completed? |
|---|---|---|---|
| starting #10 | 57.000 → 57.075 | localization_failed | False |
| opening #10 | 56.500 → 56.575 | orientation_uncertain | False |
| opening #11 | 63.000 → 63.077 | competing_alternative_seed_geometries | True |
| middlegame #10 | 61.000 → 61.075 | orientation_uncertain | False |
| endgame #10 | 55.000 → 55.075 | localization_failed | False |
| endgame #11 | 61.000 → 61.075 | orientation_uncertain | True |
| puzzle #10 | 53.000 → 53.075 | localization_failed | False |

Four of nine previous opt-in failures are rescued: starting #4 (rectification) and #11, opening #2, middlegame #8. Five remain unresolved (#10 in each position). The new opt-in regressions are opening #11 and endgame #11. The unresolved views are in the late-tour White-left/transition portion; some have bright/low-contrast BGR renderings, which cannot alone establish physical sensor overexposure given the HDR conversion caveat. Opening #11 has a visibly wrong **rejected** leading wood-color seed; endgame #11 has a rejected line seed extending into the background fence. Neither was accepted by recovery. Wider line-pool/candidate profiling and color-managed controls would be separate investigations, not established fixes here.

Legacy regressions, all retained as separate rows in the [paired CSV](pilot_01_hd_comparison.csv): starting #1/#2/#12; opening #3; middlegame #1/#2; endgame #8/#9. Six previous legacy failures now complete: starting #4/#7, opening #4, middlegame #5/#11, puzzle #8.

## Accepted geometry, semantic review and diagnostics

Across 49 common opt-in completions, registered raw seed-corner RMS disagreement has median **3.86** and maximum **18.47 low-resolution pixels**. Final canonical square-center RMS disagreement has median **0.69** and maximum **1.41 pixels**. For 24 common legacy completions, those medians are 4.97 and 0.79 pixels. Different warp seeds can converge to similar fitted grids; seed movement alone is not final-square error. Feature-registration median residuals are recorded per pair and are of similar scale to the grid-center differences. No common completed pair changes its rectification quarter-turn integer; this remains agreement, not an independent accuracy result.

Assistant review covered **20 full policy overlays across 18 matched frames**, stratified across all five positions: 12 completed overlays visually plausible with printed-corner semantic agreement; 6 incorrect localization-gate polygons subsequently rejected by orientation; 2 unresolved recovery regressions. All five opt-in contact sheets were also screened for gross grid errors, covering all 60 inputs / 53 completions. **No incorrect completed geometry was identified in this limited review.** This is not a measured zero false-accept rate or certification of fine alignment/unreviewed semantics. The detailed attributed [geometry review](pilot_01_hd_geometry_review_v2.json) keeps observations separate from verified labels.

Each paired image orders columns: Telegram legacy, original legacy, Telegram opt-in, original opt-in. Blue marks C0–C3; green canonical labels are predicted labels for review, not verified ground truth.

- [starting 01](pilot_01_hd_examples/starting_01_paired.jpg)
- [starting 04](pilot_01_hd_examples/starting_04_paired.jpg)
- [starting 11](pilot_01_hd_examples/starting_11_paired.jpg)
- [opening 02](pilot_01_hd_examples/opening_02_paired.jpg)
- [opening 11](pilot_01_hd_examples/opening_11_paired.jpg)
- [middlegame 08](pilot_01_hd_examples/middlegame_08_paired.jpg)
- [endgame 11](pilot_01_hd_examples/endgame_11_paired.jpg)
- [puzzle 05](pilot_01_hd_examples/puzzle_05_paired.jpg)
- [puzzle 10](pilot_01_hd_examples/puzzle_10_paired.jpg)

[Piece detail comparison](pilot_01_hd_examples/piece_detail.jpg) uses a common physical foreground region via feature registration; [ROI coordinates and source hashes](pilot_01_hd_examples/piece_detail_roi.json) make the crop provenance explicit. The Telegram patch is nearest-upsampled only for display and gains no information. Originals have about **5.27× source pixels** (roughly 2.3× linear sampling), making piece silhouettes/markings better resolved in native crops; compression, color rendition, motion blur, occlusion and the fixed classifier crop/input bottlenecks remain. No model prediction or recognition improvement is asserted.

## Runtime, validation and preservation

| Policy | Telegram median s/frame | Original median s/frame | Telegram total s/60 | Original total s/60 |
|---|---|---|---|---|
| legacy | 1.49 | 3.02 | 93.51 | 194.80 |
| marked-board | 2.21 | 5.98 | 153.39 | 393.66 |

Full decode/index/correspondence search took 382.9 measured seconds across the originals; nearest-PTS extraction/registration took 65.0 additional seconds. Original files total 401.17 MB versus 75.33 MB for Telegram videos. Legacy detection works at 720×1280 for these originals versus 464×848 for the copies, while rectification samples source pixels.

Timings include offline preparation/diagnostics and are descriptive desktop runs, **not controlled latency or phone/browser benchmarks**. Branch mixes differ, the bounded high-quality sessions include additional cold starts, concurrent test/index activity occurred, and interrupted/priority-probe overhead is excluded from the per-pair sums. Prior low-resolution extraction time was not recorded, so no invented decode speedup/slowdown is reported. Piece-classifier inference is absent. See [environment](pilot_01_hd_environment.json).

**130 relevant tests passed in 89.92 s**; 9 focused tests passed again after diagnostic-rendering changes. New tests cover exact display pixel/coordinate rotation, unsupported transforms, exact timestamp selection, photometric correspondence/textureless rejection, unsupported registration and canonical-ID-aware grid comparison. Existing native integrity, capture, replay and geometry-recovery tests also pass. Native integrity validation passes for all five aligned sessions / 60 derivatives, including hashes and decoded-pixel bindings. Final preservation checks match **872 protected prior files**, original/compressed media, configurations, production vision/scanning sources, marking reader, MobileNet checkpoint and B.2 tighter-224 checkpoint. No prior artifact was overwritten.

New code: `src/scanning/pilot_comparison.py`; `scripts/reprocess_pilot_hd.py`, `scripts/align_pilot_hd_frames.py`, `scripts/evaluate_pilot_geometry_session.py`, `scripts/compare_pilot_resolutions.py`; `tests/test_pilot_comparison.py`. New registration files: `a3_pilot_01_hd_native_v2.json`, `a3_pilot_01_sources_v2.json`, `a3_pilot_01_attestations_v1.json` in `data/metadata/`. New reports/tables/reviews/environment/color metadata and examples use the `pilot_01_hd_` prefix. Existing source/import/evaluation scripts, reports and manifests were not modified.

## Deliverable checklist

- [x] All five originals located, matched, fully decoded and compared with preserved copies.
- [x] Both representations registered under the same sessions/groups; HD primary, development-only.
- [x] Sixty sequence-aligned timestamp pairs, all raw PTS/source references preserved.
- [x] Both unchanged policies evaluated; completion, failures, exceptions, rescues and regressions reported.
- [x] Accepted-corner/grid agreement and stratified semantic/geometry visual review supplied.
- [x] Existing FENs reused; owner photo/FEN and unchanged-sequence attestations recorded separately.
- [x] Reproduction commands, tests, runtime, source hashes, preservation and diagnostic examples supplied.
- [x] D.1 readiness/additional-capture recommendation scoped below; no D.1 implementation.
- [ ] Independently verified per-frame corners and all-frame semantic labels remain outstanding, as explicitly flagged.

## Source identity, labels and grouping

The five files `data/raw/a3_pilot_01/new_position_001_hd.MOV` through `new_position_005_hd.MOV` map in order to starting, opening, middlegame, endgame and puzzle. Registration is versioned in [source catalog v2](../../data/metadata/a3_pilot_01_sources_v2.json), with the [aligned native manifest](../../data/metadata/a3_pilot_01_hd_native_v2.json) as the primary representation. The original [compressed manifest](../../data/metadata/a3_pilot_01_native.json), all low-resolution media, prior reports, hashes and diagnostic files remain unchanged.

The recordings are the same five sessions, not ten sessions or 120 independent observations. Both representations retain the same position/session IDs, supplied FENs, historical position links, split `pilot` and conservative independence group. Four positions repeat historical training placements; this is development data, never an independent final estimate or untouched holdout. No split promotion or training eligibility was granted.

Source metadata, duration/frame-count proximity, geometry-independent visual correlation and feature registration strongly support downscaled/re-encoded representations of the same recordings, consistent with the owner's Telegram transfer history. File bytes cannot authenticate the exact application/export chain. Aspect ratios also differ slightly: displayed originals are 1080×1920, versus 464×848 copies. Comparison therefore uses measured feature registration rather than assuming a single isotropic scale factor.

No higher-quality reference photographs were uploaded: the five existing 960×1280 JPEGs remain unchanged; the copies directly under `data/raw/` have identical hashes. Assistant inspection found their layouts and printed coordinates visually consistent with the supplied piece-placement FENs and White-near reference orientation. **The owner explicitly confirmed all five photo/FEN associations and unchanged placement throughout all five recordings.** This confirmation is recorded separately in [owner attestations v1](../../data/metadata/a3_pilot_01_attestations_v1.json), with source/reference hashes. Earlier immutable manifests retain historical pending flags; the source catalog explicitly gives the attestation supplement precedence for placement/stability only. Turn, castling and move counters are preserved as supplied, not inferred from photographs.

Per-frame corners and semantic-orientation ground truth were not included in that owner confirmation. They remain unverified; assistant visual checks below are separately attributed and do not certify all frames. Camera elevation/degrees and detailed lighting metadata remain unknown. The iPhone 13 Pro identification and QuickTime dates are source metadata, not independent verification of the phone clock or scene setup.

## Decode and timestamp protocol

The original native importer deliberately rejects rotation. A separate offline adapter applies the recorded −90° display transform as an exact clockwise pixel permutation; encoded coordinates remain 1920×1080 and saved PNG/geometry coordinates are 1080×1920. The explicit 3×3 encoded-to-display matrix is recorded. This is a video-display transform, not a semantic chessboard rotation. All raw PTS/time bases are retained; no nominal-FPS substitution occurs.

The full initial decode indexes every original frame and searches ±0.5 seconds around each of the frozen 60 low-resolution timestamps using whole-frame normalized correlation, without geometry/model outcomes. To avoid selecting different moments independently in still sections, the final comparison freezes **one median inferred offset per recording**, then takes the nearest actual indexed PTS to each adjusted timestamp. Selected PNGs are recovered at exactly those indexed PTS by seeking and decoding, and checked against source hashes. Frame count/rate differences are therefore explicit rather than handled by equating frame indices. Offset inference is not hardware synchronization or proof of identical camera exposure.

The high-quality source is HEVC 10-bit `yuv420p10le`, BT.2020 / ARIB STD-B67 (HLG); the Telegram source is H.264 8-bit SDR BT.709. The documented conversion is PyAV `frame.to_ndarray(format='bgr24')` followed by the display permutation. **No HDR tone mapping or calibrated BT.2020-to-SDR appearance rendering is performed.** PNGs preserve these converted BGR8 pixels losslessly, not the source's full 10-bit signal. Originals remain untouched for later color-managed decoding. Thus this is an original-file-versus-Telegram comparison, **not a pure resolution-only causal experiment**. Codec, color/transfer processing, resampling and small inferred timing differences are additional factors. No classifier accuracy claim follows from sharper appearance.

## Geometry correctness and review limits

Both policies use unchanged detector, orientation, grid, thresholds and frozen marking-reader weights. Neither FENs, reference-photo corners nor known orientation are passed as oracle inputs. MobileNetV3 is not loaded. Legacy successes must retain identical grids under opt-in recovery on each same high-resolution input; the runner enforces this. Comparing the two resolutions is a separate question and includes regressions.

Completion, localization-gate acceptance, cross-representation agreement and geometric correctness are separate fields. The machine-readable comparison preserves all accepted/rejected proposals, recovery reasons, first failing stage, source square polygons and predicted semantic IDs. The paired images show blue C0–C3 corner markers and fitted canonical grids on completed results; orange leading candidates in failed cases are explicitly rejected candidates, not accepted boards. An orientation rejection following an incorrect accepted localization is reported at its first failing gate while the visual diagnosis identifies the upstream polygon problem.

The starting-position high-resolution frames near 4.07 and 9.07 seconds are concrete counterexamples to “more pixels implies correct localization.” Their accepted `wood_color` proposals score approximately .588 and .593, above the unchanged .55 gate, yet cut the playing area/include board frame. Orientation rejects both; opt-in recovery returns visually plausible boundaries and canonical labels. The same two low-resolution legacy inputs completed. This resembles the B.1/B.1b accepted-boundary problem and cannot be fixed by claiming sharper source detail. No threshold was lowered.

Returned high-resolution corners and all 64 canonical square polygons are registered into low-resolution pixel space for paired agreement measurements. The feature transform is estimated independently of board detection, with recorded inliers and residuals. These differences include registration and timing error. Even zero disagreement would not prove both geometries correct. Quarter-turn integers alone are not semantic truth when geometric corner anchoring changes; corresponding canonical square locations and printed labels are the relevant checks.

No independently annotated per-frame corner coordinates exist for this pilot. Accordingly, there is **no quantified ground-truth corner error, board IoU, verified geometry-completion rate or all-frame semantic-orientation accuracy**. The stratified assistant review below distinguishes visibly plausible grids, visibly incorrect accepted polygons, and unresolved cases. Fine boundary/cell alignment and unreviewed semantics need owner review. No exact-board or piece-classifier metric was run.

## Reproduction

Use the existing `.venv` and isolated PyAV 19.0.1 installation (`requirements-pilot-video.txt`). All commands require fresh output directories. The first raw importer attempt exposed an auxiliary timed-metadata stream with no codec context; the offline adapter was corrected to retain that stream's metadata with a null codec, and fresh `hd_native_run2` completed. The partial failed directory was retained. A later full geometry process exited on SIGTERM (143) after 15 complete pairs, without a Python traceback or completed preservation footer. Those results were retained, and bounded session runs reused the 15 source/config/hash-matched pairs and evaluated the remaining 45; the incomplete pair was rerun in a fresh directory. The interruption is not counted as a geometry rejection. Completed session parts were assembled only after full coverage, hash and legacy-parity checks.

A clean reproduction can run all five bounded sessions without the interrupted-run cache:

```bash
.venv/bin/python scripts/reprocess_pilot_hd.py --output-dir outputs/scanning/pilot_01/hd_native_new
.venv/bin/python scripts/align_pilot_hd_frames.py --correspondence-run outputs/scanning/pilot_01/hd_native_new --output-dir outputs/scanning/pilot_01/hd_aligned_new
.venv/bin/python scripts/validate_native_pilot.py outputs/scanning/pilot_01/hd_aligned_new/manifest.json --report outputs/scanning/pilot_01/hd_aligned_new/validation_repeat.json
```

For each session ID in `hd_aligned_new/manifest.json`, run:

```bash
.venv/bin/python scripts/evaluate_pilot_geometry_session.py --native-run outputs/scanning/pilot_01/hd_aligned_new --session-id SESSION_ID --output-dir outputs/scanning/pilot_01/hd_parts_new/POSITION
```

Then assemble and compare:

```bash
.venv/bin/python scripts/evaluate_pilot_geometry_session.py --native-run outputs/scanning/pilot_01/hd_aligned_new --assemble-parts outputs/scanning/pilot_01/hd_parts_new/starting outputs/scanning/pilot_01/hd_parts_new/opening outputs/scanning/pilot_01/hd_parts_new/middlegame outputs/scanning/pilot_01/hd_parts_new/endgame outputs/scanning/pilot_01/hd_parts_new/puzzle --output-dir outputs/scanning/pilot_01/hd_geometry_new
.venv/bin/python scripts/compare_pilot_resolutions.py --hd-run outputs/scanning/pilot_01/hd_aligned_new --hd-geometry outputs/scanning/pilot_01/hd_geometry_new --output-dir outputs/scanning/pilot_01/hd_comparison_new
.venv/bin/python -m pytest -q tests/test_pilot_comparison.py tests/test_native_pilot.py tests/test_pilot_intake.py tests/test_capture_protocol.py tests/test_frame_selection.py tests/test_scan_replay.py tests/test_geometry_reliability.py
```

Actual runs are `hd_native_run2`, `hd_aligned_run1`, `hd_geometry_parts_run1`, `hd_geometry_final_run1` and `hd_comparison_run1` under `outputs/scanning/pilot_01/`. The `--reuse-run outputs/scanning/pilot_01/hd_geometry_run1` option was used only for starting/opening to preserve completed work from the interrupted run. Source/version hashes and environment are recorded, including the unchanged evaluator hash matching the original low-resolution run. A small priority probe is separate from final paired metrics. All native files/audio/timed metadata remain preserved; only selected video frames were used for geometry, not continuous tracking.

The legacy native validator passes normalized display-coordinate derivatives; the explicit rotation adapter has separate pixel/coordinate regression tests. Native integrity validation is not acceptance by the unchanged guided-scan ZIP companion schema. No browser replay provenance is fabricated. Original-frame indexing supports deterministic future native replay, but this task does not implement the browser adapter, tracking evaluation, keyframe selector or fusion.

## Recommendation and task boundary

The high-quality recordings should remain the primary pilot sources. The compressed versions are valuable paired robustness controls, grouped under the same recording IDs. Preserve both in a versioned external dataset/object store with SHA-256 and back up the ignored indexes, PNGs and diagnostics; Git alone does not retain them. Large recordings remain ignored and were not added to Git.

**Ready for D.1 offline keyframe-selection development**, with explicit invalid-geometry rejection and auditable per-frame provenance. It is not yet a fully labeled geometry/semantic benchmark, a browser latency benchmark, or deployment evidence. Fix/document a consistent color-managed decode convention before drawing browser-equivalence, training or calibration conclusions. D.1 should compare novelty and board-region quality at a common scale; global sharpness can be dominated by the table/background, and raw localization confidence/completion cannot substitute for correct geometry. The continuous timelines contain useful alternatives around failed frames; the fixed 60 samples are a comparison set, not the best possible HD keyframe selection.

Repeating these five tours is **not required to start that development**. Next useful additions are new physical positions in a different capture/lighting session, deliberate steady White-left views and controlled glare/shadow coverage, with clear reference photographs and explicit FEN/stability attestations. Retain native quality, all borders/markings and full playing area; record device, board identity, lighting and interruptions. A documented full-resolution SDR export of these same recordings would help separate HDR/color-processing effects without recollecting the positions. New independent final-holdout sessions remain separate and untouched.

No MobileNet training/fine-tuning, B.2 checkpoint modification, B.4 calibration work, production inference/browser change, default recovery promotion, multi-view fusion, final-holdout use, commit or push occurred. Stop here for owner review before any D.1 work or geometry fixes.

