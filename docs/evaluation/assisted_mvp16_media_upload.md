# MVP-1.6 — Photo and prerecorded-video uploads

Implemented 2026-11-12. This is a local application feature, not a training,
model-selection, fusion, inference-optimization or generalization experiment.

## Implemented workflow

The existing assisted application at `/?mode=photo` now offers **Upload Photo |
Upload Video | Live Scan** on desktop and mobile layouts. All three use the same
existing board editor, game-state controls, structural validation, explicit
confirmation and Copy FEN. Selecting new media or changing modes discards stale
results and confirmations. Predictions remain drafts regardless of score.

Photo upload retains JPEG/PNG, 12 MiB/32-megapixel validation, original EXIF-aware
backend decoding, preview, camera capture as an option, recognition and manual
alignment/orientation help. A Remove photograph action complements replacement
and New position. No camera permission is needed for photo or video upload.

Video upload provides a standard file picker, filename/size/duration/resolution,
playable local preview, replace/remove, Analyze video, Cancel and progress.
Analysis requires the explicit checkbox: **“The chess pieces remain in the same
positions throughout this video.”** This is an owner assertion, not automatic
change detection. A moving position must not be processed as one board.

## Reused architecture

`MultiViewScan.tsx` now accepts live or video input; it reuses the original
CandidateBank, existing session/Finish/status/cancel flow, quality/diversity
selection and editor handoff. `videoUpload.ts` adds only local media validation,
metadata loading, deterministic sample times, seek/decoded-frame handling and
resource cleanup. No second recognition service, model, frontend or editor exists.
Live Scan preserves manual Start/Finish/Cancel, rear-camera preference and cleanup.

The existing endpoints remain:

- `POST /api/photo/scan/session`
- `POST /api/photo/scan/candidate`
- `POST /api/photo/scan/finish`
- `GET /api/photo/scan/status`
- `POST /api/photo/scan/cancel`

The only session-contract extension is optional `source` metadata:

```json
{
  "unchanged_position": true,
  "source": {
    "kind": "uploaded_video",
    "filename": "scan.mp4",
    "size_bytes": 15818922,
    "duration_ms": 75333.333,
    "width": 464,
    "height": 848
  }
}
```

Legacy live payloads remain valid with their original timestamp bound. Uploaded
video sessions allow timestamps within their validated duration, at most 30
minutes, rather than silently clamping long recordings to the live 120-second
bound. The source metadata is copied into frame bindings and final results.
Session UUID + frame ID + timestamp + JPEG SHA-256 bind each observation; filename
is descriptive metadata, never a server filesystem path. The browser does not
hash/read an entire large original video into memory. Runtime source identity is
session-local, not a server-verified hash of unseen original video bytes.

Candidate responses additionally expose measured `quality_check_seconds`, for
accepted and rejected candidates. This does not change eligibility thresholds.
No new full-video upload endpoint or server transcode fallback was added.

## Formats, decoding and sampling

Initially supported containers: MP4 and WebM. `canPlayType` is only an initial
probe. Real metadata decoding and every requested seek must succeed before that
frame can be uploaded. The file extension does not establish codec support.
Verified here: **MP4/H.264** and **WebM/VP9** in the installed headless Chromium.
Other browser/OS/codec combinations, including HEVC, are not certified by this
milestone. A corrupt or unsupported stream gives a conversion/browser guidance
message, not a fabricated recognition result.

A paused HTML video element accesses the local File through an object URL, with
metadata preloading. The original video is never transmitted or converted by the
application. Canvas extraction waits for asynchronous `seeked`, non-seeking state
and decoded current data. It rejects missing/invalid duration, dimensions,
timeouts, unreliable seeks and decode errors. No camera stream is opened.

Sampling policy `timeline_bin_midpoints_max24_min3s_v1`:

1. `count = min(24, max(1, floor(duration_seconds / 3)))`.
2. Sample the midpoint of each equal-duration timeline bin, in increasing order.
3. Wait for decoding, capture one bounded JPEG and check it through the existing
   candidate API before advancing. Do not classify during extraction.
4. Skip non-increasing timestamps or byte-identical consecutive decoded JPEGs.
   D.1 also prunes redundant retained views.
5. After sampling, reuse Finish to select **up to three** complementary usable
   canonical views. Fewer usable views are supported, never forcibly padded.

Original browser timeline timestamps and frame identities accompany uploads.
They are browser-reported seek positions, **not guaranteed native frame PTS or
frame-accurate timestamps on every codec/browser**. No precise camera poses are
invented. Selection never uses piece predictions or ground-truth labels.

## Recognition and mandatory review

The frozen C.9 ConvNeXt checkpoint remains:

`outputs/evaluation/convnext_c9/export_v1/convnextv2_tiny_best.pt`

SHA-256:
`45b471edb751261e9da1ef8703f3d9c940cdaec69f5713f6847e48bf0dc26acb`.

The existing loader verifies identity. Four-channel RGB+target mask,
`context1_224`, normalization, 13-class order, canonical square mapping, FP32 and
channels-last are unchanged. Selected multiple views use MVP-1.5's **one shared
model, two bounded workers, chunks of eight and controlled CPU threading**.
One usable view uses the unchanged single-photo fallback. No usable views is an
explicit failure. Invalid geometry/orientation is never silently fused.

The draft policy stays **best-quality whole single view**. Supporting views keep
all per-square predictions, timestamps, geometry status and disagreement signals.
Mean probabilities/logits were not promoted. Agreement does not certify truth;
additional viewpoints may still offer useful review evidence.

The existing review supports all twelve pieces/empty, undo/reset, board flipping,
canonical coordinates, game state, structural validation and confirmation/FEN
copy. Any edits invalidate confirmation. The app does not invoke Stockfish.

## Bounds, cancellation and privacy

- Source file: **1 GiB**, at most **30 minutes**, positive finite duration,
  positive decoded dimensions with at most **32 megapixels**.
- At most 24 sampled frames; each at most 2560 pixels on its long edge and 3 MiB.
  Existing JPEG quality remains .95; original photo quality is not reduced.
- At most nine compressed candidate Files/object URLs in the browser and server;
  selected view count at most three. No full uncompressed recording collection.
- One source decoder plus optional paused/controlled preview. Only one extracted
  frame/upload is in flight, and duplicate comparison retains one prior JPEG.
- Source metadata/seek waits: 15 seconds each; candidate request: 45 seconds;
  post-selection wait: 120 seconds; overall video-analysis deadline: eight minutes.
- Existing server limits remain four sessions, 64 MiB compressed cache, ten-minute
  expiry, two request slots and serialized CPU-critical application work.
- Cancel aborts extraction/uploads/polling, cancels the session, releases decoder
  sources and object URLs, clears drafts and guards generation IDs against late
  responses. Replacement, removal and mode unmount clean the previous source.
- Preview may remain playable during review until removal/replacement/navigation;
  this is local session state, not permanent media retention.

No whole video reaches Python, Modal or an external image service. Uploading
photos/videos needs neither a camera nor its permissions. Live camera still
requires trusted HTTPS or localhost. Source files are preserved unchanged.

## Real functional evidence and timings

Outputs: `outputs/evaluation/assisted_mvp16_v1/`. These are development functional
fixtures, not independent generalization results. All show the owner-labeled
starting position, which was exposed in training. Fine geometry returned by the
live API remains `completed_unverified` and requires user review.

| Case | Actual decode/source | Candidates | Selected | Extraction | Candidate requests, including checks | Post-selection server | Analyze-to-review browser |
|---|---|---:|---:|---:|---:|---:|---:|
| Original compressed pilot | 75.33 s, 464×848, H.264/MP4 | 24 | 3 | 1.60 s | 48.81 s | 25.26 s | 76.43 s |
| Compact HD-derived replay | 12 s, 406×720, VP9/WebM | 4 | 2 | 0.31 s | 7.31 s | 10.61 s | 18.47 s |

MP4 metadata loading: 0.065 s; measured server geometry/quality work summed to
48.636 s; model/runtime 23.366 s; input preparation 1.888 s. WebM metadata: 0.011
s; server checks 7.288 s; model/runtime 9.781 s. Candidate request time includes
transmission, server processing and scheduling; subtracting the server measure
is not a rigorous isolated network benchmark. Browser totals include polling and
UI overhead. These cases use different inputs and budgets: **not a controlled
inference-speed comparison**. No nine-second promise is made.

Both real browser runs selected actual decoded frames, called the actual local
geometry and ConvNeXt APIs, displayed drafts, performed an edit/undo, explicitly
confirmed and copied the correct FEN. Both drafts needed zero corrections on this
training-exposed position, while supporting views disagreed on ten and nine
squares respectively. Cancellation/replacement after recognition invalidated the
old confirmation. Desktop and 390-pixel mobile layouts were tested.

The MP4 source is the existing registered Telegram-compressed comparison:
`data/raw/a3_pilot_01/01_starting/video/new_position_001_vd.MOV`, SHA-256
`40efd302abd1e1c129381cc32a04f48639dba09401321e7d6fee2685e24b8eaa`.
Its historical filename is `.MOV`, but registered bytes are **mp42/MP4 + H.264**.
The browser test supplies the same bytes with a `.mp4` filename/MIME; it does not
transcode, alter the original, or pass it off as the HD source. The ignored
`approved-starting.mp4` symlink provides a no-copy local example. Do not rename
arbitrary HEVC/QuickTime files expecting a codec conversion.

The 12-second WebM is an explicitly derived functional fixture, 657,974 bytes,
created by streaming twelve hash-checked historical HD sample images to the
already installed Playwright FFmpeg. One sample per second changes the timeline;
its manifest retains original sample provenance. It is **not an original
continuous capture or independent recording**. No frame image dataset persists.

Additional cases:

- A two-second static VP9 fixture produced **zero usable geometry**, and the UI
  correctly showed a retry error without a draft or classifier execution. It
  also verifies video/object-URL removal. Do not call this a one-view success.
- An actual **one-view uploaded-video API** replay uses a known hash-checked
  approved source frame and the unchanged photo fallback: 7.88 seconds after
  Finish, zero changed squares versus the prior frozen replay. This case proves
  the session/fallback path, **not browser decoding that particular source MOV**.
- Existing photo browser regression: all 64 predictions unchanged, identical
  packed pixels, maximum absolute logit difference 0.000006676 on the frozen C.10
  replay. This is a regression check, not accuracy tuning on C.10.
- Peak server process RSS: **1,224,428 KiB (~1,196 MiB)** after the acceptance
  campaign; one model instance. This is a process-lifetime, not per-case isolated
  peak. Browser decoding memory is bounded by implementation but not precisely
  profiled here; no claim of a measured total browser+server peak is made.

Screenshots/results:

- `browser/desktop-confirmed.png`, `browser/mobile-video.png`, `browser/results_v1.json`.
- `browser_webm/desktop-confirmed.png`, `browser_webm/results_v1.json`.
- `browser_zero_views/results_v1.json`, `single_video_api_v1.json`.
- `photo_browser/results_v1.json`, `live_ui/results.json`, `server_memory_v1.json`.

No physical smartphone hardware test is claimed. Actual file decoding and backend
inference were tested; fake-camera/saved-response tests are used only for the
separate existing live UI regression.

## Tests, error cases and integrity

**936 Python tests passed:** 888 existing regressions, 38 scan/session/worker
checks, ten existing engine tests. The engine subprocess tests run outside the
sandbox with a 45-second external timeout, not through the application.
**62 frontend tests passed**, TypeScript/Vite production build and whitespace
checks passed. Browser checks: 13 photo, six live UI, eight MP4, eight WebM, plus
zero-view/cleanup checks and the one-view real API replay.

Tests include bounds and deterministic timeline sampling, long-duration sampling,
empty/corrupt/unsupported videos, metadata and seek timeouts, asynchronous seek
completion, abort before capture, timestamp/source association, unchanged-source
assertion, session limits, canonical grids, duplicate/stale rejection, raw output
contracts, worker indexing, photo parity, editor/confirmation, mode cleanup and
mobile layout. Thirty-minute footage was not replayed end to end; the long-file
bound and sampling policy were unit-tested. No claim of exhaustive codec testing.

The first MP4 test reached real inference but failed on a test selector using a
nonexistent `data-square` attribute; the corrected harness uses the existing
accessible square buttons. A one-frame fixture was initially expected to be
usable but failed geometry; it is preserved as the explicit zero-view case,
not removed or improved by changing recognition settings.

A pre-edit hash snapshot covers 8,218 historical/source inputs, including the
MVP-1.5 artifacts. The final preservation result records **8,213 unchanged files, five authorized application edits, zero unexpected changes and zero missing files**; historical checkpoints, reports, manifests, media and hashes
remain byte-identical, with no unexpected changes or missing files. The separately
protected final holdout was not opened. Results remain outside historical paths.
No datasets, videos, weights or secrets were added to Git; no commit or push.
Inherited ConvNeXt pretrained weights remain **CC BY-NC 4.0**; this feature does
not change deployment/distribution rights.

## Startup and trying existing media

```bash
npm --prefix web run build -- --emptyOutDir false
outputs/evaluation/convnext_c11_env/bin/python scripts/serve_scan_web.py \
  --assisted-photo --host 127.0.0.1 --port 8765
```

Open `http://127.0.0.1:8765/?mode=photo`. Upload Photo accepts an existing JPEG/PNG;
Upload Video accepts a supported MP4/WebM, then requires the unchanged-position
checkbox before Analyze video. For a local existing example, use
`outputs/evaluation/assisted_mvp16_v1/approved-starting.mp4` or the compact
`hd_replay_12s_720.webm`. The MP4 alias points to the registered compressed bytes;
it is not an original-quality conversion. Live Scan remains separately available.

For a phone, use the same trusted LAN/TLS setup documented in MVP-1/MVP-1.5, with
a certificate matching the laptop address and appropriate private-network/WSL
access. No certificates or firewall settings were changed. Browser/OS codec support
varies; unsupported clips must be converted externally to a supported format.
There is no full-video backend upload fallback. Images are processed on the laptop,
not with phone-side PyTorch. This remains a trusted-network development server.

Reproduce tests with the existing ignored Playwright installation:

```bash
npm --prefix web test
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 outputs/evaluation/convnext_c11_env/bin/python \
  -m pytest tests --ignore=tests/test_assisted_multiview.py --ignore=tests/test_engine.py -q
outputs/evaluation/convnext_c11_env/bin/python -m pytest tests/test_assisted_multiview.py -q
outputs/evaluation/convnext_c11_env/bin/python -m pytest tests/test_engine.py -q
node web/scripts/verify-assisted-media.mjs
MVP16_WEBM=1 node web/scripts/verify-assisted-media.mjs
MVP16_ZERO=1 node web/scripts/verify-assisted-media.mjs
MVP1_BROWSER_OUTPUT=outputs/evaluation/assisted_mvp16_v1/photo_browser \
  node web/scripts/verify-assisted-photo.mjs
node web/scripts/verify-assisted-live-mvp16.mjs
outputs/evaluation/convnext_c11_env/bin/python scripts/replay_media_single_view.py --output /tmp/mvp16-single-video.json
python3 scripts/check_assisted_mvp16.py --output /tmp/mvp16-integrity.json
```

Generate missing local fixtures/alias with `scripts/prepare_media_upload_fixture.py` using the existing CPU environment: default `--kind replay`, `--kind single`, and `--kind mp4-alias` respectively. It refuses to overwrite existing fixtures, manifests or aliases; no original recording bytes are changed. API replay and preservation outputs are also
write-once: choose fresh versioned paths for another campaign. Browser harnesses
write current-task diagnostics only; they do not overwrite prior MVP outputs.

## Files and acceptance

Added: `web/src/videoUpload.ts`, `web/tests/videoUpload.test.mjs`,
`web/scripts/verify-assisted-media.mjs`, `verify-assisted-live-mvp16.mjs`,
`scripts/prepare_media_upload_fixture.py`, `replay_media_single_view.py`,
`check_assisted_mvp16.py`, this report and a small versioned integration index.
Changed: `AssistedPhotoApp.tsx` (three modes/remove photo), `MultiViewScan.tsx`
(shared video input), `assisted_multiview.py` (source metadata/timing),
`tests/test_assisted_multiview.py`, frontend test command and built entry/assets.
No model-runtime, crop, geometry algorithm, orientation or fusion policy changes.

- [x] Three clear modes; photo/video require no live camera.
- [x] Real photo recognition and real supported-video extraction/inference.
- [x] Existing quality/diversity checks, canonical alignment and up to three views.
- [x] Frozen FP32 shared-model runtime and best-quality draft policy preserved.
- [x] Same editable board, game state, explicit confirmation and FEN copy.
- [x] Failure/zero-view/cancellation/replacement/resource cleanup tests.
- [x] Desktop/mobile-emulation evidence and actual local API requests.
- [x] Original media/historical integrity; protected holdout untouched.
- [ ] Physical smartphone and broader codec/device coverage (not claimed).

Next milestone: owner usability testing with supported existing media, followed
by separately authorized MVP-2 confirmed-position Stockfish analysis. No model
training, quantization, cloud work or MVP-2 implementation was started.
