# A.3 pilot 01 — intake locations and validation history

Current status: **native-media ingestion and geometry diagnostics completed**. See
[pilot quality report](pilot_01_quality.md) and
[bound native manifest](../../data/metadata/a3_pilot_01_native.json).
Five videos and photographs are present; 10,881 frames decoded and 60 candidates
extracted. Human ground-truth reviews remain pending. The original staging registry
and earlier intake notes below are retained as history; statements about unavailable
media or decoder dependencies below describe the pre-upload inspection, not current status.
This pilot remains development-only and ineligible for training.

## Exact drop locations

Repository root: `/home/apoa/repos/chess-vision`.
The following directories already exist. Put exactly one original recording in
each `video/` directory and its corresponding original photograph in `reference/`.
Keep the phone's filenames and extensions; directory identity provides the mapping.
MOV/MP4 recordings and HEIC/JPEG photographs need not be renamed or converted.

| Position | Video directory, relative to repository root | Reference directory |
|---|---|---|
| 1 Starting | `data/raw/a3_pilot_01/01_starting/video/` | `data/raw/a3_pilot_01/01_starting/reference/` |
| 2 Opening | `data/raw/a3_pilot_01/02_opening/video/` | `data/raw/a3_pilot_01/02_opening/reference/` |
| 3 Middlegame | `data/raw/a3_pilot_01/03_middlegame/video/` | `data/raw/a3_pilot_01/03_middlegame/reference/` |
| 4 Endgame | `data/raw/a3_pilot_01/04_endgame/video/` | `data/raw/a3_pilot_01/04_endgame/reference/` |
| 5 Puzzle | `data/raw/a3_pilot_01/05_puzzle/video/` | `data/raw/a3_pilot_01/05_puzzle/reference/` |

Ground-truth FENs, reserved position/session IDs, pending reviews and these paths
are already saved in
[`data/metadata/a3_pilot_01_intake.json`](../../data/metadata/a3_pilot_01_intake.json).
You do **not** need to re-enter the five labels. Correct that file only if an
association or supplied FEN is wrong; retain label revisions when media reviews start.
The full six-field FEN is preserved exactly as supplied. Only the first field is
visible piece placement; turn, castling, en passant and clocks are supplied metadata.
FEN validity is not verification against a photograph or the complete recording.

Copy originals without trimming, stabilization, resizing, transcoding, screenshots
or messaging-app compression. Preserve available sidecars and original timestamps.
If these are guided-scan ZIP exports rather than native movies, preserve them and
tell the operator; do not rename ZIP bytes to MOV or manufacture a video from them.
Keep supplementary takes separate and register a new take/session ID rather than
placing multiple unexplained videos in one session folder.

Helpful accompanying facts, if known: capture date/time/timezone, phone/lens and
app, board/piece-set identity, room and light source, interruptions, suspected piece
movement, and whether exposure/lens changed. Unknown values remain null. Neither
the registration date nor file copy time is an invented capture time. No precise
elevation or camera pose is inferred from an absent recording.

## Checks performed now

- All five FENs parse, pass `python-chess` standard validity checks and are distinct
  among this pilot. Occupied counts: 32, 32, 30, 11, 7 respectively.
- Starting, middlegame, endgame and puzzle placements match historical
  `position_001`, `position_005`, `position_013`, `position_017`, respectively;
  all four are historical **train** positions. The opening has no exact historical
  placement match; that does not establish new-session or independent exposure.
- Each recording has its own reserved `capture_session_id`, linked to a position
  and reference directory. Recordings, timestamps, references, later crops and
  repeat takes retain that relationship. No current dataset assignment is changed.
- All five are `split: pilot`, `training_eligible: false`, `final_holdout: false`.
  Shared setup is conservatively grouped as one cohort until reviewed. Matching
  historical placements retain their existing position-group links. The new IDs
  do not make already-seen positions independent. Do not feed these captures into
  the ongoing B.4 calibration work or current training/evaluation splits.
- Five videos and five photographs are currently absent. Format, duration, frame
  rate, resolution, exposure, coverage and geometry outcomes are **not measured**.
  The approximately 75-second average is user-reported, not a per-file measurement.

The staging inspector distinguishes missing/ambiguous assets from present-but-
unvalidated media. It never accepts a video merely because a filename exists.

Validation executed: inspector reports **10 missing media assets** and exits 2
as designed. **62 tests passed in 6.00 seconds**, covering pilot staging, existing
A.3 manifest validation, frame selection and replay. Syntax and `git diff --check`
pass. `git check-ignore` confirms both raw video and reference-photo paths are
ignored. New files are this guide, the intake registry, `scripts/inspect_pilot_intake.py`
and `tests/test_pilot_intake.py`; no existing protocol/schema/model/calibration
file was modified.

```bash
.venv/bin/python scripts/inspect_pilot_intake.py
.venv/bin/python scripts/inspect_pilot_intake.py --report outputs/scanning/pilot_01/intake_check_01.json
.venv/bin/python -m pytest -q tests/test_pilot_intake.py tests/test_capture_protocol.py tests/test_frame_selection.py tests/test_scan_replay.py
```

Exit 2 means missing/ambiguous media, exit 0 means ready **for** media validation,
and exit 1 means an intake error. Report paths must be new. None of these results
means native-video validation, A.3 acceptance or verified photographic labels.

## Existing protocol/tooling boundary

Read [A.3 protocol](capture_protocol.md), `configs/capture/capture_manifest.schema.json`,
`src/scanning/capture_protocol.py`, `scripts/validate_capture_manifest.py`,
`scripts/extract_scan_frames.py`, `src/scanning/frame_selection.py`,
`src/scanning/replay.py`, and [guided capture/replay](../guided_scan_capture.md).
Current protocol v2 companions bind `guided_capture_v1` JPEG ZIP sequences. The
recorder is approximately 1 Hz, capped at 60 frames/80 MiB; replay validates manual
geometry and source bytes, but does not run a complete tracker/recognizer/fusion.

The supplied approximately 75-second native-phone takes are a supported collection
goal in A.3 documentation, but **MOV/MP4 ingestion is not implemented** by those
ZIP tools. Their timestamps, size limits and provenance cannot be silently reused
for native movies. This registry is explicitly `a3_native_pilot_intake_v1` staging,
not a fabricated protocol-v2 approved manifest. Equipment/light metadata required
by v2 also cannot be populated with guessed iPhone, board or lighting values.
The locked main A.3 plan and final reservations remain unchanged.

`ffprobe`, `ffmpeg`, PyAV and imageio-ffmpeg are unavailable in the current environment.
Native container/codec, display rotation, time-base and presentation-timestamp
validation needs an appropriate decoder/prober. No packages are installed before
the actual files/codecs are known. OpenCV nominal FPS and frame-number/FPS are
insufficient evidence for original variable-frame-rate timestamps.

## Execution after files arrive

1. Inventory originals; record SHA-256, byte size and actual filenames; bind the
   reference photo and supplied FEN. Check reference/photo correspondence and any
   visible piece changes separately from the user's intention that pieces stayed
   fixed. Leave unclear placements and intervals pending.
2. Probe each original video: container/codec, duration, dimensions, display
   rotation, nominal/average rates, stream time base, presentation timestamps,
   VFR/gaps/decode failures and available capture metadata. Preserve raw timestamp
   values and time-base conversions; any normalized session-relative offset must
   be recorded. Do not create timestamps by dividing frame number by nominal FPS.
3. Add a narrowly scoped native-video intake adapter if required, preserving the
   A.3 grouping/review contracts and immutable full recordings. Do not disguise
   imported movies as original browser-recorder ZIP sessions or truncate to 60
   frames to bypass replay limits. Validate actual decoding/PTS before accepting it.
4. Extract at most 12 diverse, timestamp-linked candidates per session using the
   existing time-bin, sharpness/exposure, temporal-gap and perceptual-duplicate
   ideas. Retain source frame identity/PTS, hashes, pixel dimensions and any display
   rotation applied to derivatives. Do not select based on classifier correctness.
   Review both selected candidates and recording coverage; the full timeline
   remains the unit for future tracking/fusion, including blur and board loss.
5. Assess camera-side and coarse elevation categories only where visible; use
   unknown for transitions. Record lighting qualitatively and distinguish global
   sharpness from board visibility; image sharpness does not prove a visible board.
   Verified corners/orientation and sequence invariance require separate reviews.
6. Run existing `prepare_raw_photo` and opt-in `prepare_reliable_raw_photo` as
   geometry-only experiments on the same candidate frames, with fresh output
   directories. These reuse the actual legacy and B.1b policies without loading
   or changing the B.2 classifier. No manual FEN/corners/orientation may influence
   automatic policy proposals. Record localization/orientation/grid failures,
   recovery attempts, polygons and semantic orientation; overlay them for review.
7. Separate returned/completed results from geometrically correct results. Use
   independent reviewed corners when available; otherwise state visual assessment
   and its limits. Flag visibly wrong accepted polygons. Reference-photo corners
   cannot be copied to moving-camera frames. Record unverified geometry as such.
8. Produce the final per-session pilot report, candidate/contact-sheet diagnostics,
   missing-viewpoint recommendations and exact reproduction commands. Preserve
   originals and all failed attempts. No training, calibration, fusion or policy
   tuning is part of this ingestion task.

## Storage and additional capture advice

`data/raw/*` is already ignored by Git. Large videos/reference photographs stay
there or in a backed-up external dataset store mounted at the same documented
asset root. Store an immutable original copy, checksums and small relative-path
manifests; track only metadata/tools/reports and deliberately selected diagnostic
images after review. Use a versioned object store/external disk for large originals;
do not automatically add videos to Git or force-add ignored assets. A content hash
verifies identity, not backup availability. Derived outputs belong under ignored
`outputs/scanning/pilot_01/`, never production assets or calibration directories.

Do not re-record blindly before the initial review. Coverage is currently unknown.
After inspection, fill the measured gaps: all four physical sides and low/medium/
high elevations of the same unchanged position, steady 1–2-second windows during
continuous movement, full playing area and printed borders visible. Pair diffuse
and directional lighting for both a crowded position and sparse dark rook/king/
queen arrangements, preferably in a separate documented session. Keep deliberate
hand/board-loss stress takes separate and retain originals. These are provisional
protocol targets, not claims that a particular pilot viewpoint is missing.

Pilot ingestion and validation are **not complete** until the ten media files are
available and the native-format/timestamp boundary above is resolved and tested.
