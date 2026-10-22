# A.3 — Continuous multi-angle capture protocol (revision 2)

The primary data unit is an **original continuous recording of one unchanged
physical chess position**. Diverse still images are sparse derivatives for the
existing MobileNetV3 model. Complete recordings, including motion, repeated views,
blur, tracking loss and recovery, are retained for future scan evaluation.
This revision supersedes A.3's stop/freeze-per-pose proposal. It does not implement
live keyframe selection, fusion, a video-native model or any recognition change.
No new physical recordings or verified labels have been collected.

The research/implementation documents remain at repository root; their referenced
`docs/` copies are absent. See [A.2 evidence](orientation_baseline.md): raw completion
varies strongly by side, while prepared recognition does not establish a consistent
Black-side classifier deficit. Cross viewpoints within environments/lights, and
measure upstream geometry separately from piece recognition.

## Two data products from each recording

| Product | Keep | Purpose and exclusions |
|---|---|---|
| Original sequence | Every original JPEG, timestamp, event, view note, dimensions and source hash in its immutable ZIP | Future tracking, keyframe-selection and fusion evaluation; retain difficult frames, transitions and recovery |
| Sparse recognition subset | At most 12 diverse reviewed frames per recording, each linked to its original frame ID/hash/timestamp | Existing image-based model; reject unsuitable/near-duplicate candidates without removing them from the sequence |

The existing guided recorder captures **an uninterrupted timestamped JPEG sequence
at approximately 1 Hz**, with a 60-frame / 80 MiB limit. This is the supported
first pilot route: 40–55 seconds of continuous camera movement, not twelve separately
posed photographs. It is not full-rate video. Preserve its actual irregular time
intervals rather than inventing a constant rate or synthesizing missing frames.
The current replay validates sequences/manual geometry; it does not run recognition,
tracking or fusion. Those future evaluators must consume the original timeline.

For realistic full-rate tracking evaluation, also plan dedicated native-phone
video takes at the phone's actual recorded rate (nominally 30 fps), preserving
original MOV/MP4 files, orientation/rotation metadata, variable-frame-rate presentation
timestamps and audio if present. Use the **same position/split/setup identity rules**,
record actual start time and file hash in the collection log, and keep all associated
assets together. Do not run Safari and the native camera simultaneously or claim
separate takes are synchronized. Native video is a separate continuous session.
A deterministic codec/PTS-to-replay importer is outside this revision's supported
intake: the new tools accept the existing guided ZIP sequences, not MOV/MP4. Do not
transcode away originals, estimate timestamps as frame-index/nominal-fps for VFR,
or claim 1 Hz replay demonstrates high-frame-rate phone tracking. Native recordings
remain sealed original assets until their importer/provenance can be validated.
The required original-record preservation is provided now by the guided frame
sequence; collecting native video does not automatically make it accepted data.

## Balanced acquisition matrix

| Factor | Values to traverse within recordings |
|---|---|
| Camera side | White-near, Black-near, White-left, White-right |
| Elevation above board plane | Low ≈35° (25–45°), medium ≈60° (50–70°), overhead ≈90° (80–90°) |
| Environment | Two actual rooms/backgrounds (`room_A`, `room_B` are reserved IDs) |
| Lighting within each room | Both `diffuse` and `directional` |
| Position | Fixed throughout a recording; new position means a new session |

One session is a continuous tour of all four sides and elevations for one
position/room/light combination. Move the camera around the board; digital rotations
are not physical views of pieces. Keep board, lighting, camera lens and screen
orientation stable. Angles are human estimates or measurements, never automatically
verified camera poses. Label transitions/uncertain views unknown instead of forcing
one of four sides onto diagonal motion.

| Collection | Positions | Rooms × lights | Planned recordings | Sparse-frame target/ceiling |
|---|---:|---:|---:|---:|
| Engineering pilot | 2 (sparse + crowded) | 1 × 2 | 4 | Up to 48 |
| New train | 8 | 2 × 2 | 32 | Up to 384 |
| New validation | 2 | 2 × 2 | 8 | Up to 96 |
| Untouched final holdout | 2 | 2 × 2 | 8 | Up to 96 |
| Total | 14 | — | 52 | Up to 624 |

The 624 entries are **diverse view-coverage goals**, not a request for 624 independent
photographs or a mandate to fill every slot with a near-duplicate. Each recording
retains roughly 40–55 source frames with current hardware/software, even though
no more than 12 become recognition examples. Actual counts depend on timing and
memory. The full main collection is 48 continuous recordings of 12 new positions;
the four pilot recordings are separate engineering data. If a side/elevation is
missing, note that gap. If needed, collect an additional continuous take of that
same position/setup, preserving the same split/correlation group and a documented
plan-lock revision. Do not replace the original or cherry-pick a successful clip.

Use the existing wooden board/set and iPhone first. Register actual board, piece-set
and camera IDs; verify the 4 cm square / 3.5 cm border profile supported by current
replay. Another board requires a separately scoped replay/profile update. Apply both
lights in both rooms (for example bounced/diffuse and single-side lamp), avoiding
A.2's side/environment/light confounding. Optional equipment/stress extensions must
be identified separately and balanced, rather than attached only to one viewpoint.

Reserve 12 genuinely new main positions before capture: crowded overlap, distributed
midgames, sparse endgames, dark rook/king/queen and pawn/knight neighbors. All sides
must show the same arrangement. Pilot positions may be previously seen and never
become holdout positions. The plan contains reserved IDs and null FENs, not invented
arrangements. Exact placement equality is checked independently of FEN turn/clocks.
Two final positions are a small initial check, not a general reliability benchmark.
Expand the final position reservation before observing model results if feasible.

## Practical phone pilot

1. Choose a sparse and a crowded engineering position; manually record placement
   and check the physical pieces. Copy/generate the empty v2 plan. Use `room_A`
   under both lights: two positions × two lights = four continuous sequences.
2. Use the existing trusted HTTPS setup in
   [guided_scan_prototype.md](../guided_scan_prototype.md). Start the rear camera,
   keep the full playing area and preferably printed borders visible. Leave piece
   recognition off while establishing reference data. Do not use detector output
   as ground truth.
3. In **Capture a scan**, enter the reserved position ID, human-checked FEN if known,
   and actual room/light notes; choose **Create session**. Browser `split: development`
   is a legacy note: the locked companion assignment is authoritative. Start the
   recording once. **Do not pause or freeze at each pose.**
4. Take a smooth 40–55 second tour. Spend roughly 10–12 seconds per side, slowly
   moving low → medium → high while walking toward the next side. Briefly steady
   your hands for 1–2 seconds at useful views **without stopping recording**. Keep
   the board fixed and the camera away from pieces. A safer partial arc is preferable
   to dropping the phone; log missing coverage rather than pretending it was seen.
5. Change the recorder's Current view notes when convenient, but do not sacrifice
   camera control to hit a dropdown. They are unverified user notes. Annotate exact
   intervals from the replay afterward. Alternate clockwise/counterclockwise tour
   order between positions/lights to reduce order effects.
6. Stop once at the end, **Export ZIP**, and save/transfer before closing or discarding
   browser memory. Stop before 60 frames / 80 MiB. A pause/background interruption
   is retained as an event/gap, not removed in editing. Re-take a normal tour if
   necessary, keeping the interrupted original as a separately identified attempt.
7. Store the original ZIP unchanged under a new session path. Copy its actual UUID,
   `created_at`, SHA-256 and position ID into the companion. Replay the whole ZIP;
   retain all frames and timestamps. Optional aligned frozen **reference** images
   are collected after export as separate assisted references, not spliced into the
   continuous trajectory or counted as transitions. Keep these supplementary
   references with the same position/split; they are annotation evidence, not
   regular v2 continuous sessions or automatically eligible training frames.
8. Repeat for the other light and second position. Normal pilot has four continuous
   sequences, about three to four minutes of recorded material. Hand occlusion,
   board loss, exposure changes and recovery can be separate pilot stress takes.
   The position must remain fixed; piece-change tests need separate positions/epochs.
9. Run the offline extractor below. Inspect its entire rejection log and candidate
   coverage. Choose at most one candidate per useful side/elevation, no adjacent
   lookalikes; fewer than 12 is fine. Do not select based on a model's confidence
   or correctness. Preserve discarded candidates in the original sequence.
10. Review the exact selected JPEGs: quality, playing-area corners, semantic board
    orientation and piece placement are separate human checks. Unclear labels stay
    pending. A candidate or successful replay is never automatically ground truth.
    Annotate time intervals for the full sequence, including unknown transitions.
11. Check camera permission denial, pause/resume, background/return, export and replay.
    Main data collection starts only after this engineering pilot works on your phone.
    Physical capture and review remain your tasks; unit tests are not device acceptance.

## Offline still extraction and duplicate controls

`extract_scan_frames.py` reuses existing bounded ZIP validation and copies selected
source JPEG bytes exactly; it does not re-encode, crop, rotate, label or run the model.
It divides the original timeline into equal bins, selects a sharp candidate per bin,
checks mean exposure, enforces ≥2 seconds between selected frames and a cap of 12,
and rejects exact/near duplicates using a 64-bit image difference hash. This spreads
selection across the recording instead of exhausting the budget in its first seconds.
`candidates.json` preserves every frame's source timestamp and selection/rejection
reason, the source archive hash, thresholds and unverified candidate identities.

Defaults (development heuristics): Laplacian variance ≥30, grayscale mean 20–235,
near-duplicate Hamming distance ≤6, temporal gap 2,000 ms. These are not calibrated
acceptance thresholds or learned viewpoints. Tune only on pilot/development data,
freeze settings before validation/final comparisons, and manually inspect false
rejections. Perceptual hashing can miss near-duplicates or reject different useful
views. A stationary sharp clip should yield one/few candidates, not dozens.

Assign conservative `near_duplicate_group_id` values to reviewed related frames
within each physical position, including repeats across takes. Validator selection
accepts only reviewed primary frames, at most one per group, one per coverage cell,
12 per session and the locked temporal spacing. Fully reviewed repeat frames with
`selected_for_matrix: false` remain available for diagnostics but are not selected
for training/recognition validation. Exact source hashes and placements may never
cross splits. These image heuristics do not implement the future online D.1 selector.

## Metadata, reference labels and timestamp integrity

[`capture_manifest.schema.json`](../../configs/capture/capture_manifest.schema.json)
and [`a3_plan.json`](../../configs/capture/a3_plan.json) now use
`capture_protocol_v2`. The original browser format stays `guided_capture_v1`.
V1 companions are not silently upgraded: keep old files as history, generate a new
plan and rebind/review annotations for the timestamp-aware v2 format.

| Object | Required provenance and annotations |
|---|---|
| Position | ID, fixed split, previous-exposure declaration, manually entered FEN (null until known) |
| Session | ID, position/split, correlation group, actual board/camera/room/light IDs, `capture_mode: continuous_sequence`, actual archive UUID/hash/path and creation time |
| Full timeline | `view_intervals` of start/end milliseconds, camera side, elevation category and notes; use null for unknown side/elevation during turns/occlusion |
| Sequence review | Separate image-sequence-bound human attestation of unchanged physical position, reference FEN and reviewed timeline; not a claim that scanning succeeded |
| Derived frame | Original archive frame ID, exact source hash/dimensions/epoch, exact `timestamp_ms`, side/elevation/degrees/method, duplicate group and primary-selection flag |
| Reference geometry | Human-verified playing-area corners in original JPEG TL→TR→BR→BL pixels; null until reviewed |
| Reference orientation | Separately verified White-near/Black-near/left/right; null until reviewed |
| Frame reviews | Separate quality/corners/orientation/placement status, reviewer, timezone-aware review time and annotation signature |

All original timestamps must be strictly increasing. Full view intervals are ordered,
contiguous, half-open `[start_ms,end_ms)` and cover every source frame; uncertainty
is represented explicitly as unknown intervals. Do not interpolate a pose across
an unobserved turn. Per-frame labels must agree with known interval labels, and
derivative timestamps must match their original records exactly. Preserve raw
recorder view notes/events even if later human annotation corrects their meaning.
A reviewed sequence binds the archive, reference placement and interval metadata;
changes expire its attestation. Missing intervals/review remain pending, not invented.
Large gaps are reported and retained, not interpreted as steady frame rate.
Recorder timestamps use its session-relative clock, so the first frame may have
a nonzero offset after session creation. Preserve `capture_started` events and
original times; future acquisition latency starts at recording start, not at the
first selected still. These timestamps are not native MOV/MP4 presentation timestamps.

The four frame reviews use `pending`, `verified` or `rejected`. Pending reviewer,
time and digest are null. Verified annotations require matching human reviews;
changing pixels, corners, FEN, timestamp or orientation expires the relevant digest.
Review time must follow the frame/sequence. Software cannot prove a human actually
inspected the data. The extractor creates **no** verified values or FENs.

For a new frame review, start with:

```json
{"status":"pending","reviewer":null,"reviewed_at":null,"annotation_sha256":null}
```

Check the occupant whose **base** lies on each square. Never infer missing pieces
from model predictions or label occluded squares empty. Confirm the unchanged
position from an independent physical/overhead reference if necessary. FEN syntax
or chess validity does not prove a photographed placement; game-history fields
are supplied metadata, not image observations. Source geometry from replay/manual
alignment is a proposal until reviewed against actual square boundaries.

Recorder corners are semantic a1,h1,h8,a8, while companion corners are geometric
TL,TR,BR,BL (x-right/y-down). Reorder only after checking the frame. Image corner
TL/TR/BR/BL IDs are a8/h8/h1/a1 White-near, h1/a1/a8/h8 Black-near,
a1/a8/h8/h1 White-left, h8/h1/a1/a8 White-right. Printed coordinates plus independent
physical references establish orientation, not piece predictions. Sparse verified
corner/orientation reference timestamps support geometry scoring; unannotated
frames cannot be presented as verified geometry.

## Splits and untouched final holdout

All frames, videos, extracted images, assisted references, repeat takes and derived
crops of one physical position stay in **one split**. A session never spans splits.
Identical placements under different IDs, exact images and correlated setup groups
also stay together. `independence_group_id` describes real shared setup/recording
blocks; renaming near-duplicates does not make them independent. As in A.3 v1,
pilot/train/validation/final collection use separate UTC days, checked against
actual recorder timestamps. Keep correlations grouped even across different days.

The assignment lock binds position/session/split/group/equipment/matrix and sparse
selection policy before collection. Archive an owner-held copy. Label entry and
incoming archive references do not silently resplit data. Amend the collection
plan explicitly before extra takes. Historical placement/image checks use the
frozen A.1 inventory; missing inventory or changed bytes fail collected-data intake.
Hashes cannot detect all recompressions or invented session histories.

A custodian retains final original recordings, derivatives and labels privately.
No final recognition, keyframe-threshold tuning, geometry tuning, fusion selection
or error-driven changes are allowed. Annotation and metadata-only integrity audits
are permitted without looking at model predictions. Developers receive a copy with
final archives/times/FENs null, final frame records omitted, final view intervals
empty and final sequence reviews pending; locked reservations remain intact.
Actual access control and truthful annotation are human requirements.
The validator is not a replacement for filesystem permissions or owner discipline.

| Validator purpose | Output selected for downstream use |
|---|---|
| `training` | Reviewed, diverse primary frames from train only |
| `validation` | Reviewed primary frames from val for **image recognition** |
| `scan_validation` | Reviewed **whole original val sessions**, including every difficult/redundant source frame |
| `final_evaluation` | Final recognition still subset; requires explicit later release |
| `scan_final_evaluation` | Final whole-session scan benchmark; requires explicit later release |
| `audit` | Integrity/coverage diagnostics; selects no training/evaluation inputs |

Both `final_evaluation` and `scan_final_evaluation` require `--release-final`, only after later explicit owner
approval and frozen evaluation criteria. Neither may be used for model selection.
No tool here performs training, scanning or inference. Existing model loaders do
not consume this schema; a future adapter must preserve split and review gates.

## Different evaluation populations

Recognition training and crop validation use the sparse, quality-reviewed image
subset. Full scan evaluation runs the **original chronological stream**, not the
filtered derivative gallery. Raw completion/exact-position denominators include
all preregistered eligible sessions, including recordings whose detector/tracker
fails; unsuitable training frames do not remove a scan attempt. A sequence review
means the reference labels and timeline were checked, not that the pipeline worked.
Predeclare normal and separate stress cohorts before inference. Corrupt/changed-
position recordings are retained with collection/reference failure reasons and
reported separately; never delete an unsuccessful scan to raise accuracy.

Future scan metrics: localization/orientation acquisition and error rates, time to
first valid geometry, tracking drift/loss/reacquisition against independent reference
frames, useful keyframe diversity/latency, single-frame versus multi-view exact-board
accuracy, conflicts, automatic/assisted completion, duration/retry/correction burden.
Report per sequence and independent position/setup groups, with stage denominators;
nearly adjacent frames are not independent accuracy samples. Native full-rate
tracking claims require native-rate data, not interpolation of the current 1 Hz logs.

Keep paired diagnostics: image-only localization against reviewed playing-area
corners (IoU/corner displacement in square units/internal grid alignment); semantic
orientation with oracle geometry against separately verified orientation; prepared
recognition with verified geometry/orientation; and the full raw pipeline with no
oracle inputs. Join labels **after** inference. A plausible quadrilateral is not
correct localization; chess validity is not correct recognition. Freeze geometric
tolerances using pilot/development data, then report all correct/wrong/unresolved
outcomes, failure-inclusive rates and continuous errors. Full-sequence evaluation
and independent geometry annotation/evaluation implementations remain later tasks.

## Commands and present validation boundary

No new dependency is required. Use existing trusted-HTTPS recording/export and
`replay_scan.py` (see [capture/replay docs](../guided_scan_capture.md)). Never overwrite
original archives or old reports; all new-output commands below refuse reuse except
the existing replay script, for which you must choose a new output path yourself.

```bash
.venv/bin/python scripts/validate_capture_manifest.py --init-plan outputs/scanning/captures/a3_sequence_working.json
.venv/bin/python scripts/replay_scan.py outputs/scanning/captures/ACTUAL_SCAN.zip --output-dir outputs/scanning/replay/ACTUAL_SCAN
.venv/bin/python scripts/extract_scan_frames.py outputs/scanning/captures/ACTUAL_SCAN.zip --output-dir outputs/scanning/candidates/ACTUAL_SCAN
.venv/bin/python scripts/validate_capture_manifest.py outputs/scanning/captures/a3_sequence_working.json --print-review-digests
.venv/bin/python scripts/validate_capture_manifest.py outputs/scanning/captures/a3_sequence_working.json --report outputs/scanning/capture_protocol/a3_sequence_audit_01.json
.venv/bin/python scripts/validate_capture_manifest.py outputs/scanning/captures/a3_sequence_working.json --purpose training
.venv/bin/python scripts/validate_capture_manifest.py outputs/scanning/captures/a3_sequence_working.json --purpose scan_validation
```

Copy real archive/timeline identities and candidate metadata into the companion;
review labels before setting statuses. The extractor's staging JSON is intentionally
not an approved training manifest. `--print-review-digests` also prints proposed
sequence-review signatures; it never changes or approves a record. Before capture,
use `--print-split-lock` after editing reserved equipment/room/session IDs and retain
an owner-held locked copy. Read archive hashes/metadata without modifying the ZIP.
A native MOV/MP4 is not valid input to these ZIP tools and is not silently converted.

An empty v2 plan passes audit with zero captured/verified frames. Completeness
requires the balanced reviewed coverage; `--require-complete` must fail on an empty
plan. Scan evaluation requires human-reviewed whole-sequence metadata, independently
of which derivatives are suitable for recognition. No phone pilot has been run by
this assignment; test fixtures are explicitly synthetic temporary sequences.

## Specific B.1 investigations from existing A.2 artifacts

1. Inspect all three White-right localization rejections:
   `position_006_var_06`, `position_010_var_06`, `position_016_var_06`. Compare saved
   raw `detection.png`/candidate evidence with the reviewed prepared playing-area
   corners and source images. Check board-vs-wooden-table/frame proposals, inner-grid
   versus outer-border fits, checker evidence and line candidates. Orientation was
   never reached for these attempts; do not attribute them to piece identity.
2. Isolate Black-near **orientation** failures `position_006_var_05` and
   `position_010_var_07`: replay the original border templates with independently
   reviewed corners, inspect glyph visibility/rotation, matched edges and score gap.
   Compare with their prepared geometry and known orientation. Do not substitute
   the browser learned-marking bootstrap and call that a reproduction of the raw
   photo path; evaluate it as a separately identified candidate later.
3. Audit `position_006` across views: nine raw localization failures and one
   orientation failure suggest a position/background/geometry interaction worth
   examining. Distinguish missing borders, outline/grid fit and detector rejection
   from sparse/crowded-piece effects; existing counts do not prove a cause.
4. Reproject target polygons onto original photos for `position_016` a1/d5 and
   `position_010` c3/g3. These examples expose recurring king/queen/rook and
   pawn/knight errors even with prepared geometry. Check the **physical base**, crop
   association, camera-preserving orientation and neighboring-piece overlap before
   changing any label or architecture. Preserve existing diagonal-corner and
   all-four-orientation regression coverage while auditing.

These are recommended investigations only; no B.1 audit or implementation is
performed by A.3. See [orientation_baseline.md](orientation_baseline.md) for the
measured evidence and local artifact paths.

## Verification and acceptance checklist

- [x] Continuous, multi-angle recordings are the primary acquisition unit; the
  pilot uses four smooth tours, not a stop/freeze photo checklist.
- [x] Offline sparse extraction preserves exact source bytes, IDs and timestamps,
  applies quality/spacing/near-duplicate controls, and produces no verified labels.
- [x] Original chronological sequences, rejected frames, events and human view
  intervals remain available for future tracking/keyframe/fusion work.
- [x] Position/placement/session/correlation/day/hash split isolation and sealed
  final holdout rules remain enforced. All derivative assets inherit their split.
- [x] Recognition still validation and whole-scan validation/final selection are
  explicit separate purposes. Scan attempts are not filtered by training quality.
- [x] Existing browser capture/replay is reused; source ZIP format/live behavior
  and MobileNetV3 are unchanged. No video-native recognition model was added.
- [x] Required metadata, independent corner/orientation/placement reviews, balanced
  lights/rooms/elevations, exclusions and B.1 measurement recommendations remain.
- [x] Physical collection and human verification are still pending; no photographs
  or predictions were invented as new ground truth. No training, commit or push.

Verification commands executed:

```bash
.venv/bin/python scripts/validate_capture_manifest.py configs/capture/a3_plan.json --report outputs/scanning/capture_protocol/a3_video_plan_validation_20261018.json
.venv/bin/python scripts/extract_scan_frames.py outputs/scanning/captures/scan_e3117de7-c7cb-415f-9581-18bcc316a9d7.zip --output-dir outputs/scanning/capture_protocol/a3_video_historical_candidates_20261018
.venv/bin/python -m pytest tests/test_capture_protocol.py tests/test_frame_selection.py tests/test_scan_replay.py tests/test_dataset_splits.py -q
.venv/bin/python -m pytest --ignore=tests/test_engine.py -q
git diff --check
```

Results: **70 focused tests and 371 broader Python tests passed**; no Stockfish
execution was required. Plan audit reports 52 reserved continuous sessions, zero
captured/verified frames and incomplete coverage. As expected, an empty plan fails
`--require-complete` and `--purpose scan_validation`; final scan selection without
release fails as sealed. A synthetic sequence test proves full-scan selection can
retain a reviewed session even when its registered still is excluded from training.
Extraction tests verify deterministic selection, exact bytes/timestamps, late-clip
coverage, spacing, duplicate suppression, quality rejections and output protection.

An existing historical 30-frame camera sequence produced 12 **unverified** candidates.
This is an intake smoke test, not a new capture, label review or pipeline accuracy
result. Its original archive hash remains unchanged. Legacy sequences with duplicate
timestamps (for example aligned stills appended to a recording) remain supported
by the old replay tool; the stricter v2 chronological intake rejects them rather
than rewriting/sorting original evidence. Use separate reference sessions going
forward. All 58 A.1 preserved hashes, the A.2 summary and recorded prediction-source
hashes still match; browser and recognition code were not modified.

Current machine-readable validation:
[`capture_protocol_validation_a3_video_20261018.json`](capture_protocol_validation_a3_video_20261018.json).
The earlier non-video A.3 validation record is superseded historical documentation.
Task A.3 revision 2 stops here for review; B.1 and physical pilot capture have not
been started by this assignment.

