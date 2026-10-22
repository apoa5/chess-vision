# A.3 pilot 01 — native-video ingestion and quality report

The five recordings and five reference photographs have been ingested without modifying their bytes. All 10,881 video frames decoded, and 60 sparse candidates retain original presentation timestamps. Native-media integrity passed. Human ground-truth review remains pending; this dataset is development-only and **not eligible for training or final-holdout evaluation**.

## Media and registration

The authoritative bound companion is [a3_pilot_01_native.json](../../data/metadata/a3_pilot_01_native.json). It associates each supplied full FEN and placement FEN with a video, photograph, SHA-256, position/session IDs, group IDs and derivative provenance. [Results JSON](pilot_01_results.json) contains measured metadata, unresolved frames and summary counts. The original staging registry is retained unchanged as ingestion input.

| Position | ID suffix | Duration (s) | Decoded frames | Selected frames | Legacy completion | Opt-in completion |
|---|---|---:|---:|---:|---:|---:|
| Starting | `01_starting` | 75.333 | 2,259 | 12 | 6/12 | 9/12 |
| Opening | `02_opening` | 73.367 | 2,200 | 12 | 5/12 | 10/12 |
| Middlegame | `03_middlegame` | 75.467 | 2,263 | 12 | 5/12 | 10/12 |
| Endgame | `04_endgame` | 70.200 | 2,105 | 12 | 9/12 | 11/12 |
| Puzzle | `05_puzzle` | 68.500 | 2,054 | 12 | 7/12 | 11/12 |
| Total | | 362.867 | 10,881 | 60 | 32/60 | 51/60 |

Position IDs are `a3_pilot_01_<suffix>`; session IDs append `_take01`. Exact machine-readable durations are in the results JSON. All videos are H.264/yuv420p, encoded portrait **464×848**, nominal 30 fps, rotation 0, time base 1/600. Although named `.MOV`, container metadata identifies MP4-family `mp42`. AAC audio streams remain in the untouched originals; audio was not evaluated. All photographs decode as JPEG **960×1280**.

Decoded counts agree with declared frame counts, with strictly increasing presentation timestamps. Each video contains one 1/15-second interval among otherwise 1/30-second intervals; maximum observed gap is 66.7 ms. Observed mean frame rates are approximately 29.99 fps. The full index preserves these gaps rather than inventing uniform frame times. Container creation timestamps are only 13–28 seconds apart despite longer clips, so they are not used as actual recording start times. Device, actual capture time, board identity, equipment and lighting metadata remain unconfirmed.

## Native extension, extraction and storage

The existing A.3 companion schema/replay tooling accepts guided-scan JPEG ZIPs, not arbitrary native videos. This ingestion adds an explicit offline native extension (`native_pilot_dataset_v1` / `native_pilot_session_v1`) and validator; it does **not** claim acceptance by the unchanged `capture_protocol_v2` validator or manufacture guided-scan provenance. Shared A.3 sparse-selection safeguards are retained.

Original files remain under `data/raw/a3_pilot_01/<position>/{video,reference}/`. Full ordered frame indexes, sampled-quality timelines, decoder versions, manifests, contact sheets and extracted PNGs are under `outputs/scanning/pilot_01/native_run1/`. Both raw media and outputs are Git-ignored. Keep originals plus full indexes in a backed-up external object store or dataset archive with immutable/versioned keys and SHA-256 checks; retain small metadata, code and reports in Git after review. No binaries were added to the Git index. Archive ignored outputs explicitly; a Git checkout alone cannot restore them.

All frames were decoded for indexing. A 2 Hz quality shortlist was divided into temporal bins; selection capped each recording at 12 candidates, enforced at least 2 seconds separation, rejected dHash near-duplicates (distance ≤6), low global sharpness (<30 Laplacian variance), and extreme mean luminance (<20 or >235). Selection used no geometry success, classifier output or FEN. Selected PNGs preserve decoder BGR pixels losslessly, with frame index, raw PTS, exact time base/rational time, parent-video hash, derivative hash and decoded-pixel hash. These are representative candidates, **not automatically approved training examples**. Global sharpness can be inflated by the textured table/background; it does not establish adequate piece detail. Upsampling these frames would add no source information.

## Labels and leakage safeguards

All five supplied FENs parse, are standard-chess valid and describe distinct pilot placements. They are stored as user-declared labels. Photographic placement agreement, unchanged placement through the full recording, verified corners and verified semantic orientation remain pending. Reference overviews appear consistent, but small/occluded piece identities have not been promoted to verified ground truth.

Starting, middlegame, endgame and puzzle placements duplicate historical training placements `position_001`, `position_005`, `position_013` and `position_017`. Their position groups retain those links. Opening has no exact historical placement match. All five retain the conservative shared independence group `a3_pilot_01_cohort_pending_setup_review`. Future assignment must group all recordings, references, frames and near-duplicates of a physical position together and keep correlated sessions together; historical training positions cannot become independent validation/holdout examples. No existing split was changed. This pilot has not been passed to training or B.4 calibration.

## Geometry experiment and observations

Both unchanged policies ran on the **same 60 frames**, without verified corners, known orientation or FEN supplied as oracle inputs. MobileNetV3 was not loaded. Full diagnostics are in `outputs/scanning/pilot_01/geometry_run1/`; [per-frame CSV](pilot_01_geometry.csv) retains both policies, timestamps, stage, recovery behavior, predicted orientation and overlay paths.

Legacy failures: **22 localization, 5 semantic-orientation, 1 rectification validation**. Opt-in failures: **8 localization and 1 rectification validation**. Recovery was attempted on 27 frames and rescued 19. All 32 legacy successes retained identical returned square grids. No validation thresholds or production policies were changed.

These are **preparation completion counts, not verified geometric correctness or recognition accuracy**. Quantitative corner error/IoU and semantic-orientation correctness cannot be measured without reviewed labels. No square-classifier accuracy or exact-board metric was run. The five positions and correlated frames do not provide an independent generalization estimate; four placements already occur in historical training.

Qualitative inspection found three clearly incorrect polygons accepted by the **localization gate**, at starting 32.0 s, starting 54.5 s and opening 63.0 s. They cut through the playing area or include outer border/table. Legacy orientation subsequently rejected all three, and opt-in recovery completed. This does not establish a completed pipeline accepting an incorrect board. No grossly incorrect completed polygon was identified in contact-sheet review, but fine alignment and semantic orientation remain unverified. Middlegame 67.5 s and puzzle 40.5 s orientation failures have less clear polygon misalignment and are retained as uncertain.

Representative diagnostics:

- [Reference photographs](pilot_01_examples/references_contact.png).
- [Starting 32 s: legacy accepted localization, orientation rejected](pilot_01_examples/starting_06_legacy.png); [recovered geometry](pilot_01_examples/starting_06_marked-board.png).
- [Opening 63 s: legacy incorrect localization](pilot_01_examples/opening_11_legacy.png); [recovered geometry](pilot_01_examples/opening_11_marked-board.png).
- [Starting 21 s: rectification exception](pilot_01_examples/starting_04_legacy.png).

The starting 21.0 s candidate triggers `ValueError: Corners must be ordered TL, TR, BR, BL; use order_corners first.` in both policies. Its first two proposed vertices have x+y approximately 531.088 and 529.998: `order_corners` chooses a different anchor from the proposal, and the strict warp guard rejects it. The generic exception also bypasses the recovery handler. A focused test reproduces this existing behavior; **no geometry fix was made in this ingestion task**. The raw run records `dependency_or_runtime`; the derived CSV correctly diagnoses `rectification_validation`. This is not a missing dependency.

Remaining opt-in failures:

| Position | Times (s) | Stage / recovery rejection |
|---|---|---|
| Starting | 21; 57; 68.5 | Rectification corner ordering; localization failed; competing alternative seed geometries |
| Opening | 9; 56.5 | Excessive refinement shift; orientation uncertain within rejected recovery |
| Middlegame | 46; 61 | Excessive refinement shift; orientation uncertain within rejected recovery |
| Endgame | 55 | Localization failed |
| Puzzle | 53 | Localization failed |

## Coverage and recording quality

Selected frames show all four physical board sides in each recording, high and oblique views, and the full playing area. Camera angles/degrees, balanced low/medium/overhead bins, continuous whole-session visibility and unchanged pieces are not certified. Diagonal transitions remain unknown rather than being forced into cardinal classes. Qualitative observations are separate from verified metadata in [visual observations](pilot_01_visual_observations.json).

| Observed side (unverified) | Frames | Legacy completion | Opt-in completion |
|---|---:|---:|---:|
| White-near | 16 | 11 | 15 |
| White-right | 9 | 7 | 9 |
| Black-near | 10 | 7 | 10 |
| White-left | 7 | 3 | 4 |
| Transition / uncertain | 18 | 4 | 13 |

This small, subjective stratification reports completion only and must not be treated as labeled orientation accuracy. White-left views warrant deliberate steady recaptures and geometry review. The scene appears to have bright ambient daylight in a patio/courtyard, with some exposure variation; this is a visual inference, not confirmed equipment/environment metadata. A second controlled lighting setup is not demonstrated. Crowded starting/middlegame pieces overlap at oblique angles. No obvious piece movement appears in selected stills; the full recording still needs human stability review.

The principal media limitation is **464×848 source video**, with the board occupying only part of each frame. First check whether higher-resolution phone originals exist before rerecording. Keep these received files as separate immutable sources if better originals are supplied.

## Reproduction and validation

Run from the repository root. Optional decoder dependencies are isolated from the production environment:

```bash
.venv/bin/python -m pip install --target tools/pilot_video/python -r requirements-pilot-video.txt
.venv/bin/python scripts/inspect_pilot_intake.py --help
.venv/bin/python scripts/ingest_native_pilot.py --output-dir outputs/scanning/pilot_01/native_run2
.venv/bin/python scripts/validate_native_pilot.py outputs/scanning/pilot_01/native_run2/manifest.json --report outputs/scanning/pilot_01/native_run2/integrity_validation.json
.venv/bin/python scripts/evaluate_pilot_geometry.py --native-run outputs/scanning/pilot_01/native_run2 --output-dir outputs/scanning/pilot_01/geometry_run2
.venv/bin/python -m pytest -q tests/test_native_pilot.py tests/test_pilot_intake.py tests/test_capture_protocol.py tests/test_frame_selection.py tests/test_scan_replay.py tests/test_geometry_reliability.py
```

Actual completed runs used `native_run1` and `geometry_run1`; runners require fresh output directories and never overwrite runs. PyAV 19.0.1 and linked FFmpeg library versions are recorded in the manifest. No standalone ffmpeg/ffprobe was available. Decoder installation initially failed under restricted networking and succeeded after the explicit approved escalation. The system `python` shim is unset; use `.venv/bin/python`.

**121 tests passed in 78.66 seconds.** Tests cover exact/nonuniform timestamps, missing timestamps, quality/duplicate rejection, provenance/grouping gates, forged timestamps, decoded-pixel integrity and the observed corner-order exception, alongside existing capture/replay/recovery regression tests. Native validator passed five sessions and all 60 candidate hashes/pixels/provenance records. Original-media, protected-source, configuration, frozen-model and B.2-checkpoint hashes match their pre-run values. Both run preservation checks passed.

Files added for this ingestion: `src/scanning/native_pilot.py`, `scripts/ingest_native_pilot.py`, `scripts/validate_native_pilot.py`, `scripts/evaluate_pilot_geometry.py`, `tests/test_native_pilot.py`, `requirements-pilot-video.txt`, `data/metadata/a3_pilot_01_native.json`, this report, results JSON/CSV, qualitative observations and representative diagnostic PNGs. The intake guide was updated to point to completed results. Earlier intake registry/checker/tests are retained. Production inference, localization/orientation/recovery, model checkpoints and B.4 work were not modified. No commit or push occurred.

## Checklist and next captures

- [x] Five originals/reference pairs registered, hashed, fully decoded and preserved.
- [x] Actual formats, dimensions, duration, rates and metadata recorded.
- [x] Supplied FENs associated; unverified photographic/sequence/geometry labels explicitly flagged.
- [x] Sparse diverse candidates retain exact original timestamps and group provenance.
- [x] Both existing geometry policies compared on identical inputs with diagnostics and unresolved failures.
- [x] Qualitative view/elevation/lighting/visibility limitations distinguished from ground truth.
- [x] Pilot remains development-only; final holdout, training and production behavior untouched.
- [x] Reproduction tooling and relevant regression tests pass.
- [ ] Human attestation of photo/FEN agreement, unchanged sessions, corners and semantic orientation: **pending**, not silently accepted.

For the next recording round, first recover full-resolution originals if available. Otherwise record native 1080p or better at 30 fps without messaging-app compression, with the whole board/border visible and filling more of the frame. For each position, move slowly around all four sides and briefly hold low, medium and near-overhead views; repeat in the opposite direction, especially White-left. Preserve deliberate transitions rather than collecting dozens of redundant steady frames. Add a genuinely different lighting session with controlled glare/shadows and retain crowded as well as sparse positions. Record phone/app, board/set identity, lighting, actual capture time/timezone and any interruptions or piece moves; take a clear reference and explicitly verify its FEN before each recording. Use new independent physical positions for future validation and reserve final holdout separately from this development cohort.

A subsequent geometry task should investigate the corner-anchor validation/recovery exception and the eight remaining recovery rejections, using reviewed frame corners and semantic orientation. This ingestion does not justify enabling recovery by default. Stop here for review; no training, calibration or fusion follows automatically.
