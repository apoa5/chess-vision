# MVP-1 — Assisted photograph recognition and confirmed FEN

Completed 2026-11-12. The existing React/Vite application and Python HTTP server
now provide **photograph → actual ConvNeXt draft → piece corrections → explicit
game state → user confirmation → copy full FEN**. This is an opt-in application
profile. The existing guided-scanning default and its original classifier remain
available and unchanged. No engine analysis, automatic completion, continuous
recognition, training or checkpoint replacement was added.

## Run the application

From the repository root, using the existing C.11 CPU environment:

```bash
npm --prefix web run build -- --emptyOutDir false
outputs/evaluation/convnext_c11_env/bin/python scripts/serve_scan_web.py \
  --assisted-photo --host 127.0.0.1 --port 8765
```

Open **http://127.0.0.1:8765/?mode=photo**. Choose Upload photograph or Open camera,
capture/select one image, and press Recognize photograph. Camera capture stops
the stream after taking the still. Check the source grid, every predicted piece
and every empty square. Tap any digital square and choose one of twelve pieces
or Empty. Undo reverses a piece edit; Reset to AI prediction restores the original
piece draft. Flip changes only the display perspective.

Choose side to move, castling rights, en-passant target and clocks. Defaults are
explicitly shown as **White / no castling / no en passant / 0 / 1**. These are
assumptions until the user accepts or changes them; they are never inferred from
the photograph. Check the review attestation, press **Confirm Position**, then
**Copy FEN**. Confirmation is removed on piece/game-state edits, undo/reset,
new image, new recognition or New position. The final copy field is unavailable
for an unconfirmed draft. If clipboard permissions fail, the confirmed text is
selected for manual copying.

Without `--assisted-photo`, the new APIs report that the profile is disabled and
the original startup/model behavior remains the default. `/?mode=scan` opens the
existing guided tools. `--photo-runtime reference` explicitly selects the original
C.11 PyTorch diagnostic backend for this photo mode; it is not an automatic fallback.
No new frontend or second inference server is required.

## Smartphone on the same network

Processing runs on the laptop/Python server, **not on the phone**. JPEG/PNG image
bytes travel to that server only. There is no Modal or external-image-service call.

1. Connect phone and laptop to the same trusted network. Find the laptop's LAN
   address; in WSL, this is the Windows Wi-Fi/Ethernet address, not its private
   WSL NAT address.
2. Use a development TLS certificate whose names include that exact address or
   hostname. Existing ignored `web/.certs/dev.pem` / `dev-key.pem` can be reused
   only if their certificate names match. Inspect the public certificate with
   `openssl x509 -in web/.certs/dev.pem -noout -ext subjectAltName`.
3. Trust the issuing development CA on the phone. Transfer only its **public CA
   certificate**, never its private key. Certificate generation/trust changes
   remain manual setup; this task did not change certificates or trust stores.
   See [mkcert's official mobile guidance](https://github.com/FiloSottile/mkcert).
4. Start the same server with TLS:

```bash
outputs/evaluation/convnext_c11_env/bin/python scripts/serve_scan_web.py \
  --assisted-photo --host 0.0.0.0 --port 8443 \
  --cert web/.certs/dev.pem --key web/.certs/dev-key.pem
```

5. Permit the local port through the laptop's private-network firewall as needed.
   Reuse the project's existing WSL forwarding setup. With WSL NAT, Windows-to-WSL
   port forwarding may be required; mirrored networking has different firewall
   requirements. Follow [Microsoft's WSL networking instructions](https://learn.microsoft.com/en-us/windows/wsl/networking)
   for the actual machine. No firewall, router or forwarding settings were changed.
6. Open `https://YOUR_LAPTOP_LAN_IP:8443/?mode=photo`, grant camera permission and
   capture a still. Upload remains an alternative. An untrusted certificate warning
   bypass is not a reliable camera setup: browsers require a secure context and
   user permission. See [the camera API requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia).

This is a local development server without public-service authentication/rate
hardening. Keep it on the intended trusted network. Physical smartphone operation
has **not** been tested; 390px touch emulation and Chromium's fake camera were tested.

## Implementation and API boundaries

The existing entry point selects `AssistedPhotoApp` only for `?mode=photo`.
It reuses `DigitalBoard`, `ChessPiece`, its SVG assets and the board preview error
boundary. The existing guided `PositionReview`, tracking worker, recorder and
recognition-session behavior are preserved. New photo state/API modules keep
canonical coordinates and drafts independent of the original AI observations.

The existing `serve_scan_web.py` adds three routes:

| Route | Contract |
|---|---|
| `GET /api/photo/status` | Explicit profile availability, processing location and model identity; no image processing. |
| `POST /api/photo/recognize` | Raw JPEG/PNG plus optional bounded `X-Photo-Geometry` JSON. Returns `assisted_photo_result_v1`, an **unconfirmed draft**. |
| `POST /api/photo/confirm` | All 64 edited canonical symbols, recognition ID, revision, all five game-state fields, and explicit `user_confirmed: true`. Returns `confirmed_position_v1` only after validation. |

Recognition returns every square's actual symbol, class name, all 13 logits and
probabilities, raw maximum score, source dimensions/hash, orientation/provenance,
completed-but-unverified geometry, square polygons, timings and provisional
piece-placement FEN. Scores are uncalibrated. It never returns a confirmed full
FEN or claims automatic board verification.

`PhotoService` reuses the unchanged `prepare_raw_photo`, fitted grid, canonical
`chess_square_at` mapping, the existing B.2 `make_inputs` packing path, and the C.11
runtime. RGB appearance stays camera-relative; only observation IDs are ordered
as a8…h1. Source polygons are projected using the existing homography helpers.
The application does not use FEN labels to localize, orient or classify an image.

### Frozen model and preprocessing

- Checkpoint: `outputs/evaluation/convnext_c9/export_v1/convnextv2_tiny_best.pt`.
- Verified SHA-256: `45b471edb751261e9da1ef8703f3d9c940cdaec69f5713f6847e48bf0dc26acb`.
- Profile: `configs/application/assisted_photo_v1.json`, pinning the hash of
  `configs/evaluation/convnext_c11_recommended_v1.json`.
- PyTorch CPU FP32, channels-last, batch 8, four intra-op threads, one inter-op
  thread; strict frozen state loading, evaluation/inference mode.
- Same 13 classes: empty; white pawn/knight/bishop/rook/queen/king; black
  pawn/knight/bishop/rook/queen/king. Same four-channel RGB+binary-mask
  `b2-context1-224-v1` representation, 224×224, AREA RGB/NEAREST mask and original
  ImageNet normalization. No calibration, crop redesign, geometry-threshold change
  or B.1b recovery promotion.

One model instance is loaded at explicit photo-server startup and retained for
later requests. A load lock prevents duplicate initialization. Photo requests
share the existing CPU lock and bounded queue: one running, at most one waiting;
excess requests receive 429. There is no fallback to B.2 on model failure.
Legacy models remain lazy when starting the opt-in photo server; normal startup
without that flag retains its existing model warmup.

### Resource, privacy and error behavior

Upload bytes are bounded to 12 MiB and actual formats to JPEG/PNG. Header dimensions
are checked before OpenCV decode (64 pixels per side minimum, 32 megapixels maximum).
All eight EXIF orientations are explicitly handled. Unsupported HEIC/SVG/GIF,
malformed or oversized images are rejected. PNG/JPEG decoding and metadata do not
change the original files.

The existing raw preparer takes a path. An adapter writes **one temporary lossless
upright PNG**, invokes that preparer, then removes the temporary directory even
on geometry failure. It neither transcodes/stores original videos nor permanently
retains uploads, cropped datasets or image contents in logs. Raw bodies are not
logged. Responses use no-store headers; browser object URLs are revoked on
replacement/unmount. There is no localStorage, database, training-data collection
or automatic image export. Model weights are the only long-lived service state.

Camera tracks stop on capture, Close camera, new image/position, tab hiding and
unmount. Generation/abort guards reject late recognition and confirmation replies.
Cancellation is local: an already running CPU request may finish, but cannot install
a stale draft or confirmation. Aborted writes are handled without certifying anything.

Localization/orientation/grid failures return explicit unresolved error states,
not empty-board predictions. Manual controls let the user choose rank 1's edge
and, when needed, four clockwise image-space playing-area corners. Grid fitting
and target validity checks still apply. Manually supplied orientation/corners
are recorded as such; pipeline completion remains distinct from correct geometry.

## Editing, validation and confirmed FEN

The original AI result is retained separately from an editable 64-symbol draft.
All twelve pieces and empty are selectable on every square. Edits, recoloring,
removal/addition, bounded undo history and reset require no rescan. Blue markers
identify edits. An amber score-below-90% cue is **exploratory UI guidance only**,
not calibrated confidence, an acceptance rule or a restriction on which squares
can be corrected. All apparently confident squares remain visible/editable.

Board flipping changes visual order only. Native buttons/selects support keyboard,
mouse and touch operation. Game-state edits and piece edits increment a draft
revision and clear both attestation and confirmation. A late receipt must match
recognition ID, revision, board, full FEN and game state; otherwise it cannot confirm
the current draft. The confirmed copy control always reads the current receipt.

The server reuses `BoardState`, `GameState`, deterministic FEN serialization and
python-chess consistency checks. It requires exactly one king per color, no pawns
on rank 1/8, at most eight pawns and sixteen pieces per color, supported symbols,
valid field types/clocks, consistent castling and en-passant state, and valid
static standard-chess status. Promoted extra queens are not blanket-rejected.
No piece is repaired and no castling right is granted automatically.

Errors distinguish structural issues and supplied game-state inconsistencies.
Static consistency is **not proof of reachability or recognition correctness**;
intentionally invalid composed arrangements must be corrected before standard-chess
confirmation. Full FEN parsing/serialization round-trips in the tests. Unknown
history cannot be established from imagery: user-selected/accepted defaults remain
assumptions, explicitly accepted through the review attestation.

The confirmed receipt is a versioned user assertion tied to a revision, not a
cryptographic proof of human review. No Stockfish request occurs. The existing
`src/chess/engine.py` UCI wrapper and engine functionality were not changed.

## Actual recognition and browser acceptance

The recorded successful real HTTP recognition used the approved C.10 frozen
regression photograph `data/raw/conv_test/convNext_01.jpg` (SHA-256
`f6a83236fcc4926e3a4a6d22bf7278150348367e80d23e09ba72b458b75f4479`).
This is a replay of an already evaluated position, **not new independent accuracy
evidence** or training material. The browser submitted its actual file bytes.

All 64 returned predictions matched C.10. All 64 crop/mask byte hashes matched;
maximum logit difference was **0.00000667572**. The actual AI b8 error remained
black bishop. Browser automation selected b8 after flipping the display, changed
it to black knight, set the owner-provided castling state, explicitly confirmed,
and verified clipboard FEN equaled the supplied reference:

```text
rn1qkbnr/pbpp1ppp/1p6/4p3/8/1P1P1N2/PBP1PPPP/RN1QKB1R w KQkq - 0 1
```

Recorded latency, one request under application/browser conditions:

| Measurement | Seconds |
|---|---:|
| Decode and EXIF | 0.088 |
| Geometry including temporary lossless adapter | 2.051 |
| Crop/mask extraction and input preparation | 0.402 |
| Cached runtime retrieval | 0.000014 |
| Model forward | 6.593 |
| Service elapsed | 9.178 |
| Server request | 9.212 |
| Browser submission to visible draft | 9.456 |

This is not a replacement for C.11's controlled 5.32/7.34-second benchmark. Host
conditions, browser work, cold kernels and adapter/validation overhead differ;
this single request is not a latency guarantee. No new app-level RSS benchmark
was performed; C.11's 977 MiB figure must not be presented as measured total MVP RAM.

**13 headless acceptance checks passed**, with no page JavaScript errors:
real recognition/parity; empty initial state; upload rejection; canonical flip/edit;
add/undo/reset; explicit confirmation/copy; game-state invalidation; stale-confirmation
rejection; permission denial; an actual blank-image localization failure;
cancelled-result rejection; 390px touch layout without horizontal overflow; and
native browser fake-camera capture with all tracks released.

Desktop recognition used the real network service. Mobile layout and selected
stale-response cases replayed that recorded response to test UI state, not to
claim additional model evaluations. Camera testing used Chromium's synthetic
device. **No physical smartphone or physical webcam test is claimed.** An initial
harness run also submitted this photo but failed to retrieve the no-store body
through DevTools; its failure is preserved under `browser/results_v1.json`.
The corrected test captures a browser fetch-response clone without replacing the
real desktop recognition. Both submitted positive-image requests returned 200;
the completed recorded pass is under `browser-attempt2/`.

Screenshots, actual logits and the complete acceptance record:

- `outputs/evaluation/assisted_mvp1_v1/browser-attempt2/desktop-draft.png`
- `outputs/evaluation/assisted_mvp1_v1/browser-attempt2/desktop-confirmed.png`
- `outputs/evaluation/assisted_mvp1_v1/browser-attempt2/mobile-review.png`
- `outputs/evaluation/assisted_mvp1_v1/browser-attempt2/recognition_c10_position_01.json`
- `outputs/evaluation/assisted_mvp1_v1/browser-attempt2/results_v1.json`
- `outputs/evaluation/assisted_mvp1_v1/report/index.html` (offline evidence gallery).

## Regression, preservation and reproduction

**390 selected Python regressions passed, plus 3 new preservation-tool tests**;
**49 frontend unit tests passed** (39 existing + 10 new, nine processes).
TypeScript checking/Vite production build and `git diff --check` passed.
Backend tests cover bounded formats/EXIF, manual geometry, cached/single-load
runtime initialization, geometry rejection/temp cleanup, explicit deterministic
confirmation, promoted pieces, structural/game-state errors, disabled profile,
same-origin protection and queue/error cleanup. Existing C.9–C.11, crop, geometry,
scanning, inference and FEN regressions remain passing.

```bash
outputs/evaluation/convnext_c11_env/bin/python -m pytest tests/test_assisted_photo.py tests/test_convnext_runtime.py tests/test_independent_c10.py tests/test_convnext_c9.py tests/test_multidataset_c4.py tests/test_chesscog_c5.py tests/test_personal_c7.py tests/test_chessred_c6.py tests/test_crop_experiments.py tests/test_geometry_audit.py tests/test_geometry_reliability.py tests/test_video_keyframes.py tests/test_square_evidence.py tests/test_inference.py tests/test_validation.py tests/test_fen.py tests/test_scan_recognition.py tests/test_scan_bootstrap.py tests/test_scanning_contracts.py tests/test_board_state.py -q
outputs/evaluation/convnext_c11_env/bin/python -m pytest tests/test_assisted_preservation.py -q
npm --prefix web test
npm --prefix web run build -- --emptyOutDir false
```

The browser toolchain (Playwright 1.65.0 / Chromium headless 158) was installed
only under ignored `outputs/evaluation/assisted_mvp1_v1/`, without changing app
dependencies. To reproduce the full acceptance against a running local server:

```bash
MVP1_BROWSER_OUTPUT=outputs/evaluation/assisted_mvp1_v1/browser-replay \
PLAYWRIGHT_BROWSERS_PATH=outputs/evaluation/assisted_mvp1_v1/browsers \
node web/scripts/verify-assisted-photo.mjs
```

That command performs real CPU inference on the registered photograph, not training.
Use a fresh output directory. Model loading fails clearly on a wrong/missing frozen
checkpoint; there is no randomly initialized model or B.2 substitution.

Preservation: **8,170 files checked; 8,164 unchanged; six intentional authorized
application edits; no missing or unexpected changes.** The exceptions are the
server, entry point, shared board renderer, stylesheet, frontend test script list
and generated build index. Old built assets were retained with `--emptyOutDir false`.
Experimental reports/manifests, C.9/C.10/C.11 artifacts, checkpoints, original
photographs/videos and frozen preprocessing/geometry code are unchanged. The final
holdout was not opened, even to hash it. Historical manifests still describe their
original source versions; they were not rewritten to hide application evolution.

```bash
outputs/evaluation/convnext_c11_env/bin/python scripts/check_assisted_mvp1.py \
  --output outputs/evaluation/assisted_mvp1_v1/integrity_replay.json
```

The complete report is `integrity_complete_v1.json`. An initial ad-hoc integrity
summary was incomplete due to a loop-indentation error; it is retained separately
as a failed attempt and is not used as preservation evidence. The reusable checker
and its three tests verify all entries and prohibit holdout reads.

## Files and boundaries

Modified: `scripts/serve_scan_web.py`, `web/src/main.tsx`, `DigitalBoard.tsx`,
`style.css`, and `web/package.json` (test command only). Added:
`src/scanning/assisted_photo.py`, `configs/application/assisted_photo_v1.json`,
`web/src/AssistedPhotoApp.tsx`, `photoApi.ts`, `photoReview.ts`,
`web/tests/photoReview.test.mjs`, `web/scripts/verify-assisted-photo.mjs`,
`tests/test_assisted_photo.py`, `tests/test_assisted_preservation.py`,
`scripts/check_assisted_mvp1.py`, this report and the compact results index
`data/metadata/assisted_mvp1_integration_v1.json`. Build/test/media evidence stays
in ignored output/build directories. Pre-existing uncommitted C.10/C.11 work is
preserved. No commits or pushes were made.

Inherited pretrained weights remain **CC BY-NC 4.0**. This assisted integration
does not grant commercial rights or remove provenance. No model, training-data
collection, new synthetic experiment, inference threshold, multi-view fusion,
automatic scan-completion gate or global production profile was introduced.

## Completion and MVP-2

- [x] Open the same application in opt-in photo mode.
- [x] Capture/upload and preview a bounded photograph.
- [x] Run actual frozen ConvNeXt recognition and show its real canonical draft.
- [x] Correct any square, flip perspective, undo and reset without rescanning.
- [x] Choose game-state fields and receive actionable validation errors.
- [x] Explicitly confirm and copy a valid synchronized full FEN.
- [x] Invalidate confirmation after edits/new scans; reject late responses.
- [x] Automated real-photo, desktop, mobile-emulation and camera-lifecycle checks.
- [x] Preserve model/media/research boundaries and existing scanning regressions.

MVP-2 should connect only the current confirmed revision to the existing
`StockfishEngine`: add a bounded analysis endpoint/worker, revalidate full FEN,
correlate request/response revision, invalidate results on edits, and render
White-perspective centipawn/mate evaluation, best move and principal variation.
Test engine missing/crashed/timeout/terminal-position behavior, including the
historical engine-timeout issue noted in the existing MVP plan. Do not analyze
an unconfirmed draft or make automatic completion/retraining prerequisites.
MVP-2 was not started.
