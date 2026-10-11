# Capture and replay a guided scan

This implements the image-sequence capture/replay slice of Step 03. No video codec,
backend connection, piece recognition, visibility estimator or live tracker is
required. Continuous frames are deliberately unaligned until tracking is added.
Recording progress measures frames saved, not squares recognized.

## Update the phone page

From the repository directory:

```bash
export PATH="$PWD/tools/node/bin:$PATH"
npm --prefix web run build
```

The existing HTTPS server serves the new build immediately. Reload the page on
your phone. If you still see the old page, reload again or close/reopen the tab.
The page now contains **Capture a scan** below the alignment controls. Use the
current laptop Wi-Fi address and the working WSL forwarding setup.

## Record one session

1. Enter a position ID (for example `position_001`), lighting/location and the
   known FEN if available. These are ground-truth notes, never classifier inputs.
2. Choose **Create session**, then **Start camera** and **Record scan**.
3. Keep the pieces fixed. Move slowly for 15–30 seconds from overhead to a moderate
   oblique view, then toward the opposite side, and return toward the start.
   Keep the full board and borders visible. Change **Current view** as appropriate;
   this records user notes, not verified independent view groups.
4. Choose **Pause recording**. Optionally **Freeze frame**, select a1 → h1 → h8 →
   a8 playing-area corners, then **Save aligned frame**. Geometry applies only to
   that frozen image. To continue recording, start the camera again and resume.
5. Choose **Export ZIP**. On supported phones the share sheet offers **Save to
   Files** or transfer; otherwise the browser downloads the ZIP. Save it before
   closing/reloading the page. Frames are held only in memory, not persistent storage.
6. Transfer the ZIP to the laptop using your usual file transfer method. Preserve
   it under ignored `outputs/scanning/captures/`. Export before choosing
   **Discard / new session** to start the next recording.

The capture interval is approximately one second, scheduled from decoded video
callbacks where available, with animation-frame fallback. Encoding never queues
multiple frames. Capture is capped at 60 JPEGs and 80 MiB of encoded image data
per session (temporary export buffers can use additional memory). Full camera
frame dimensions are retained; overlays are not burned into source JPEGs. Camera
constraints request 1920 × 1080, but the device decides the actual resolution.

Pausing, camera stop/disconnect or backgrounding stops capture. Restart explicitly.
Use the event buttons to note hand occlusion, blur/board loss or board/piece change.
A board change starts a new epoch; it does not reuse the previous position's
observations. For normal pilot collection, discard/start a fresh session and enter
its new FEN after exporting the old one. Frozen images from before session creation
or a reported board change cannot be saved as new aligned observations.

## Replay on the laptop

```bash
.venv/bin/python scripts/replay_scan.py outputs/scanning/captures/YOUR_SCAN.zip \
  --output-dir outputs/scanning/replay/YOUR_SCAN
```

Outputs:

- `replay.json`: deterministic chronological frame diagnostics, hashes, epochs,
  user view notes, events, raw Laplacian variance and manual square polygons.
- `frame_XXXX_overlay.jpg`: labeled diagnostics for manually aligned frames.
  Unaligned frames have no invented geometry or overlay.

Replay checks archive paths/limits, unique identities, hashes, image dimensions,
FEN syntax and manual corner/homography agreement. Image-sharpness values are
reported without an unvalidated acceptance threshold. FEN syntax does not prove
that a real photo matches the position, or that a composed position is legal.
Duplicate source hashes within an epoch are explicitly flagged, and no frame
currently contributes independent recognition evidence. All 64 squares remain
unresolved because recognition/fusion is not connected yet.

## What to test next

First export one short phone scan with an aligned final frame and replay it.
Confirm the square names match the actual board and the ZIP includes unannotated
camera images. Test pause/resume, background/return and export to Files. A demo
image can exercise aligned export, but is tagged synthetic and is not camera data.

After that works, collect the roadmap's **12 positions × 2 scans = 24 pilot
sessions**. Existing positions are suitable for engineering; record them as
previously seen. This is not a fresh accuracy benchmark. Include crowded and
sparse boards, different camera sides and indoor/outdoor lighting. Keep normal
scans separate from intentional hand, blur, board-loss and piece-change stress
sessions.

Pilot collection and on-phone export acceptance remain user tasks. Optional video
recording and replay through future tracking/recognition/fusion will extend this
harness later; this version replays capture validation and geometry only.
