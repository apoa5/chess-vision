# Selected-frame piece recognition

The web prototype sends a selected, successfully tracked frame to the existing local Python HTTPS server. The baseline MobileNetV3 Small classifier predicts all 64 canonical squares. The first successful response immediately pauses piece inference and opens an enlarged digital-board review. A user-confirmed position stays fixed while camera tracking continues. No automatic calibrated completion is claimed.

## Test on your iPhone

1. Reload the page and tap **Start scanning**. The camera opens, the board is acquired from live frames and piece recognition starts automatically. No frozen frame is needed. Keep all squares and printed markings visible.
2. The first accepted result opens **Is this position correct?** in a modal. Check all pieces and empty squares; use **Flip board** if useful. Tracking continues underneath, but no more piece-inference requests are submitted during review.
3. Choose **Yes, confirm position** to lock this exact result. Move around the board, switch between Camera and Digital board, or temporarily hide a border: tracking can reacquire without replacing the position. The status says **Confirmed by you · locked**.
4. Choose **No, try another view** to close the dialog, clear the rejected result and return to the live camera without recognizing anything. Adjust the angle and hold steady, then tap **Recognize this view**. The button requires active tracking and a verified frame at most 500 ms old. That single observation opens for review again; rejected observations are not averaged into the new one.
5. After confirmation, use **Rescan position** to return to the same deliberate capture flow. Use **More → New position / reset predictions** before changing any pieces or scanning another board; the recorder's **Mark board/piece change** also clears confirmation and starts a fresh attempt. Automatic piece-change detection is not implemented.
6. **More → Download latest recognition** exports the accepted JPEG, matching geometry and original server response, including during review or after confirmation. Server outputs remain uncalibrated and do not mark squares resolved; user confirmation is a separate browser state.
7. **More → Manual alignment / freeze** remains a fallback and resets recognition. Camera stop, restarting the scan, hiding the page or camera disconnect also clears the confirmed position.

A tracking rejection hides invalid live geometry and pauses submission of new keyframes. A previously copied keyframe may finish recognition because its pixels remain paired with the geometry verified for that image. A semantic marking disagreement still cancels pending recognition. It preserves a candidate under review or a confirmed board. Until confirmed, pause/resume controls allow pausing acquisition. Review and confirmation cannot be bypassed by resume; explicit retry opens view selection, and only the capture button submits that retry. Reset starts a new automatic scan. Initial automatic acquisition may retry failed requests on a later verified frame. Manual capture failures return to the view-selection state and wait for another button press. A successful classification always pauses for confirmation.

## Implementation

We reuse the existing threaded HTTPS server instead of adding FastAPI and another
server/port. It already works with the phone's certificate and network setup. Requests
run in HTTP handler threads. Uploads are read with a 15-second socket timeout before taking the inference lock. Session creation uses a separate registry lock and stays available during image analysis. Bootstrap and piece inference share one CPU lock, with at most one running and one waiting request; a queued request waits up to 25 seconds before returning 429. A full queue returns 429 immediately, and the browser tries a later frame. This keeps CPU work bounded without rejecting every overlapping request. The classifier is loaded once and reused. This prototype service
is local development infrastructure, not a production multi-user deployment.

`POST /api/scan/session` creates a bounded, expiring session and returns model metadata.
`POST /api/scan/keyframe` accepts a JPEG and `X-Scan-Observation` metadata. It validates
schema, session, IDs, dimensions, SHA-256, finite timestamps, monotonic geometry version
and board epoch, semantic homography and 64 square polygons. Payloads are limited to
12 MiB/32 million pixels, metadata to 32 KiB and the session registry to 32 entries
with 30-minute inactivity expiry; at capacity, the least recently used session is recycled so camera restarts do not block new scans. Only the last accepted version/timestamp is retained.

The tracking worker's accepted result supplies the exact analyzed canvas and its
geometry. The recognition callback copies that canvas synchronously before encoding;
it never pairs an independently captured live frame with an older transform. There
is at most one recognition request in flight, no accumulating image queue, and offers
are spaced at least 2.5 seconds apart. Recognition requests have a 60-second client
timeout. Stop, rescan, semantic marking disagreement and position reset invalidate pending results;
responses must match session, frame, epoch, geometry version, orientation version and
JPEG hash before display. Aborting a client request cannot interrupt CPU inference
already running on the server, but its late response is discarded.

The server rectifies pixels in camera-relative order with the same 800-pixel playing
area and 200-pixel context margin used by Week 2. It projects the supplied individually
refined target polygons into that image and reuses `polygon_crop` with 200-pixel crop
padding and the unchanged RGB + target-mask preprocessing. Pixels are not forced to
White-bottom orientation. Canonical square IDs route the predictions independently
of crop appearance. The response includes 13 logits and probabilities, top class,
source/context coverage, target polygons, model/preprocessing hashes and timing.
Visibility remains `unknown`; predictions never mark a square resolved.

## Verification and limits

```bash
.venv/bin/python -m pytest tests/test_scan_recognition.py tests/test_inference.py -q
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web run build
```

The live crop/mask path matches the existing baseline path pixel-for-pixel for all
four orientations. Real-checkpoint tests verify probability parity and retained logits;
request-handler tests check session creation, stale rejection and origin checks.
Client tests reject old session/frame/epoch/version/hash responses and malformed
class distributions. The old checkpoint and preprocessing fingerprints are unchanged.

Using the uploaded Black-side diagnostic, a cold full observation took about 2.89 s
and a warm observation about 0.65 s on the laptop. This excludes network/phone latency.
JPEG95 re-encoding of the saved JPEG changed zero class argmaxes in that example, but
individual probabilities changed by as much as 0.143. This is not a lossless-camera
comparison or a recognition accuracy claim. JPEG quality remains 95 and full camera
resolution is retained; broader compression and shared-load benchmarking are pending.
Engineering reports are in `outputs/scanning/recognition/`.

The trained classifier still makes known piece/occlusion errors. The next work is
pilot failure analysis, quality/view selection and multi-view evidence fusion before
calibrating any completion rule. Browser/network integration and responsiveness on the
actual iPhone still need the manual test above; no automated mobile browser run has
been claimed. The capture recorder still exports unregistered continuous frames;
recognized keyframe exports are a separate diagnostic contract.

## Scan workspace UI

The primary workspace fits the viewport, with view switching and scan controls
available together. On phones, Camera shows a compact tappable digital preview;
Digital board shows the full board with a live camera strip. At desktop widths
both views appear side by side. The camera/video elements stay mounted and
rendered across switches so the stream and tracker are not restarted or hidden.
Resize observers fit the original camera aspect ratio without cropping and keep
the digital board square. CSS resizing does not change source pixels or geometry.

The twelve piece graphics are original local SVG artwork, with accessible piece
names and square buttons. They do not depend on fonts, external CDNs or third-party
piece-set licenses. Scores appear when inspecting a square rather than cluttering
every tile. Recording/export, alignment help and device diagnostics are in collapsed
sections below the workspace. Desktop and phone browser layout checks remain manual;
component rendering tests verify graphics, coordinates, empty/unknown state and flip.

## Continuous tracking with user confirmation

Startup reuses the tracker's existing live acquisition/recovery path. Each bootstrap result initializes from the exact saved request image; a subsequent live tracking frame must pass verification before geometry or piece inference is accepted. Failed acquisition retries automatically, with one request at a time. Camera motion during a server request cannot attach old coordinates to a new image.

Piece recognition has idle, scanning, paused, aiming, capturing, review and confirmed states. Aiming accepts no automatic frame offers; capturing accepts exactly one frame chosen with the capture button. The first valid response synchronously disables frame offers before rendering the review dialog, so new tracking callbacks cannot enqueue another classification. Confirmation preserves that exact single-frame observation rather than averaging it with later views. Ordinary tracking interruptions do not discard an already copied keyframe or unlock a reviewed or confirmed board. Semantic marking disagreements invalidate pending responses. Retry/reset increments the position epoch, clears the old result, and only accepts responses for the new epoch. The earlier experimental multi-view averaging path was removed from the live flow in favor of explicit user confirmation.

The native modal traps focus, prevents Escape from silently dismissing review, presents explicit confirm/retry actions and restores focus when closed. Camera and tracker remain mounted underneath. A user-confirmed position is not an automatically certified correct scan and does not enable chess-engine suggestions or move overlays by itself.

Tests cover live startup without frozen corners, initializing from the saved request image, withholding geometry until a new frame verifies, pausing after the first classification, locked results surviving tracking invalidation, resume not bypassing confirmation, fresh retries without averaging, stale-response rejection and accessible modal rendering. Physical iPhone layout, focus and camera continuity still require the manual test above.

Session creation retries HTTP 429 up to three times with one-second waits. Other failures show the returned status and server detail; restart advice is restricted to missing API endpoints (404/501). Session recycling can expire the oldest client session at capacity; the service remains intended for local development.

Explicit retry closes review and enters view selection without submitting any image. Recognize this view submits the current verified frame only if it is at most 500 ms old; missing or stale geometry requires waiting for fresh tracking and pressing the button again. Cached geometry is cleared on tracking loss and never applied to a separately captured live image. Retry increments the position epoch and copies matching pixels synchronously; reset/stop or an orientation disagreement still rejects its late response. Recognition errors are visible even during tracking recovery.

Regression tests hold the inference lock and verify that sessions still open, image uploads are received before taking that lock, and a waiting board request completes successfully when the lock is released. The saved diagnostic bootstrap took about nine seconds locally, so the previous twelve-second tracking timeout had little margin for uploads or server contention. Live acquisition now allows sixty seconds and waits three seconds between failed attempts.

## Acquisition and retry latency

The local server now prepares both models and runs warm-up forwards before printing `Serving ...`; this pays imports, checkpoint loading and first-forward costs once at startup rather than during the first phone scan. The checkpoint and RGB/target-mask preprocessing are unchanged. Image responses include `timing.queue_ms`, `timing.processing_ms` and `timing.total_ms` to separate server contention from inference. These timings exclude response transfer and browser work.

Web bootstrap enables `scan_fast_acquisition`: recognized glyphs verify bounded coarse board outlines first. Raw-pose ambiguity triggers board-plane glyph sampling across those outlines, with equivalent fits merged and equally supported competing geometries rejected. If this path cannot establish geometry, the existing checker detector remains available. The same template-conflict check, individual square-boundary fit, source bounds and refinement guards still apply. Component deduplication uses neighboring spatial buckets instead of an all-pairs scan, and semantic fitting transforms proposal points in batches without changing confidence order, glyph-pose checks or unique label/edge assignment.

Periodic orientation checks run at most every thirty seconds and yield while recognition is enabled, a request is in flight, or a candidate is being reviewed. Recovery still runs when live tracking is lost. Retry waits for the user to adjust the camera and press Recognize this view. Uncertain acquisitions retry after one second; busy/error responses wait three seconds.

A local before/after check on the saved Black-side diagnostic measured warm acquisition at 6.97 seconds before and 1.47 seconds after, with identical source corners and all 725 original component proposals preserved. Preparing both models once took about 5.52 seconds; the first board request after preparation took 1.45 seconds. Three piece attempts took 0.84–1.00 seconds each. Ten rotated versions covering all sides and diagonal transitions acquired successfully in 0.51–0.95 seconds each; source-relative corner differences stayed below 4.14 pixels. These are engineering checks of the same saved photo, not held-out accuracy or phone/network latency measurements. Difficult images may still invoke the slower classical fallback. Reports are in `outputs/scanning/performance/report.json` and `outputs/scanning/performance/orientations/report.json`.

Restart the Python server to load these changes, wait for its `Serving ...` message, then reload the phone page. Check initial acquisition, the first piece result, **No, try another view → Recognize this view**, all four sides and confirmation remaining locked.

Manual retry tests verify that rejecting a result, automatic frame offers, and resume do not submit images; stale/missing frames are rejected; capture submits once; duplicate presses are ignored while processing; errors wait for another press; and position resets invalidate late responses. Reload the phone page to use this frontend change; the Python API is unchanged.
