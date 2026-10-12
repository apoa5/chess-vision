# Perspective-aware board tracking

The tracker keeps every square in a fixed physical board coordinate system. It now
compensates for perspective and rotation, refreshes reference views and border
features, periodically verifies orientation, and attempts automatic recovery.
Piece recognition and scan completion are not connected yet.

## Test on your phone

The frontend has been rebuilt. Reload the existing HTTPS page; the laptop's Python
server must remain running for marking checks and recovery.

1. Start the camera, freeze a sharp frame and find/refine the board.
2. Review a1, h8 and the square boundaries, then tap **Track reviewed alignment**.
3. Walk slowly around the board, keeping the entire playing area and preferably all
   borders visible. Try White-bottom → White-left → Black-bottom → White-right →
   White-bottom. The square identities should remain attached to their physical squares.
4. Check identities at each side: White-left has top-left a1 and bottom-right h8;
   White-right has top-left h8 and bottom-right a1.
5. On uncertainty the grid disappears. A transient failure may recover locally.
   Persistent loss enters `recovering`: hold still with readable markings while the
   laptop analyzes a selected frame. Recovery can take several seconds.
6. If automatic recovery cannot find consistent markings, freeze and realign manually.

The camera remains rendered beneath the analyzed canvas during frozen review and
tracking. Stop camera, hiding the page or camera disconnection terminates tracking
and aborts outstanding marking requests. Changes in camera image dimensions require
fresh alignment. Keeping the phone's screen orientation unchanged during a walk is
recommended; changing sides around the board is supported by the geometry.

## How it works

The initial semantic corners establish a board-to-camera homography. The browser
worker temporarily rectifies each tracking image into a 320 × 320 board-coordinate
view, including the flat border. It predicts the next view from recent verified corner movement, with a fallback to
the last verified transform if prediction cannot obtain broad support. It matches
border features in that view, so their
rotation and perspective largely cancel rather than accumulating against the first
photo. Full-resolution local matching, pyramid fallback, subpixel refinement,
forward/backward checks and robust projective fitting estimate the remaining motion.

Each accepted frame becomes the next reference view. Every four accepted updates,
features are refreshed from the current flat border. They remain expressed in the
same board coordinate system. The transform from the initial image to the latest
image moves the original individually refined square polygons, preserving their IDs.
The tracker avoids initializing features on piece tops or inside the playing area.

Matches must cover both board axes, at least three regions and a convex hull of at
least 35% of the normalized playing-area size; this tolerates a temporarily hidden
corner while rejecting evidence concentrated on one edge or a diagonal. Corners must retain semantic
winding, stay within the image and avoid implausible jumps. A rejected frame removes
geometry immediately. Three consecutive uncertain frames or a long frame gap trigger
recovery rather than retaining a stale grid. Processing has one frame in flight,
requests updates every 80 ms when idle, rejects results older than 500 ms and has a
700 ms watchdog. These are prototype thresholds, not measured phone guarantees.

Every approximately ten seconds, a selected analyzed frame is sent to `/api/bootstrap`
for an independent marking check. Its proposed semantic corners are compared against
the tracked corners for **that same saved frame**, rather than the later live image.
Disagreement greater than 0.35 of the shortest projected square edge hides the overlay
and starts recovery. Uncertain marking results do not replace valid tracked geometry.

During recovery the server receives selected live snapshots, with one request in
flight, a 60-second timeout, a one-second retry delay for uncertain geometry and a three-second delay for errors. Only learned-marking candidates with
at least eight inliers, three distinct files and three distinct ranks are eligible.
The browser also validates semantic winding, image bounds and all 64 square polygons
against their board coordinates. Recovery initializes from the saved request image;
the next live frame must pass tracking before the grid returns. Old worker replies
and replies from disposed sessions cannot restore geometry.

**Privacy:** frame-to-frame tracking stays local. Periodic marking verification and
recovery send selected images to the local laptop. No FEN or recording view label is
used to determine orientation.

## Verification and remaining limits

```bash
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web run build
.venv/bin/python scripts/evaluate_scan_tracking.py \
  outputs/scanning/captures/scan_135873be-94c2-4630-956d-ec291fda32ba.zip
```

All 25 frontend tests pass. Tracking checks include a continuous 120-frame synthetic
360° sequence with changing projective distortion. It preserved all four semantic
corners; maximum error was 2.62 pixels in the clockwise camera-walk regression in a 512 × 512 image (200-pixel initial playing
area). Other checks cover exposure/noise, translations, occlusion, stale/dimension
changes, insufficient spatial coverage, a temporarily hidden border corner and recovery evidence with deliberately
shifted labels. Four controlled projective steps on the reviewed phone photograph
also pass. These demonstrate the specified synthetic transformations, not a measured
full physical walk or iPhone performance. Browser recovery integration still needs
the real-device check described above.

Rolling references can accumulate drift; periodic markings provide an independent
check rather than a guarantee. Recovery depends on readable markings and the existing
font-specific marking reader. Severe blur, occlusion, very low angles, rapid movement
and changed phone orientation can still interrupt tracking. There is no persistent
map of many historical reference images, visual-inertial world anchor, lens calibration
or calibrated completion gate. The 15 fps performance target and full physical
360° acceptance remain pending. The worker is TypeScript; a WASM backend can follow
if actual-device profiling justifies it.

The recorder can capture one frame per second during tracking. Those independent
images still export `geometry: null`; only Save aligned frame exports reviewed frozen
geometry. Synchronized tracked recognition keyframes and multi-view piece fusion
remain the next implementation steps after this tracking validation.

## Diagonal and Black-side recognition correction

The detector's refined geometric seed is now reordered before perspective correction.
At a 45° view, refinement can change which corner minimizes x+y; previously that
could cause the warp to reject an otherwise valid quadrilateral. Semantic square IDs
are established separately from these image corner labels.

The marking reader first samples a perspective-corrected border view, removes the
playing area from its proposals, and maps recognized observations back to source
pixels. Opposite printed edges face opposite players, so character rotation hypotheses
are retained and checked against each candidate edge/orientation. Raw-image fallback
also retains pose alternatives. The registration search allows a larger coarse-seed
error (7 cm instead of 3.5 cm); acceptance still requires unique physical glyphs,
multiple distinct files and ranks, robust geometric agreement and playing-grid checks.
No coordinate labels are assigned from their presumed slot. The CNN weights are unchanged.

On Black's side, bottom files H..A and left ranks 8..1 from bottom to top now have an
explicit regression test. Six development photos covering all four sides produce
valid candidates, including formerly rejected `position_007_var_05` and formerly
ambiguous `position_003_var_06`; their overlays were inspected. Ten known rotations
of `position_003_var_01` also pass, including all four diagonal views, with maximum
corner deviation below 4.3 pixels relative to transformed source alignment. This is
synthetic photo rotation, not a new physical-camera accuracy benchmark. Reports and
overlays are under `outputs/scanning/tracking/`. Repeat that image-only check with:

```bash
.venv/bin/python scripts/evaluate_board_orientations.py data/raw/position_003_var_01.jpg
```

**After this update, restart the Python HTTPS server and reload the phone page.**
Python modules are cached by the running server; rebuilding the frontend alone does
not activate the recognition correction. Retest clockwise corner transitions and
hold on Black's side to check h1 at top-left and h8 at bottom-left.

## Diagnose automatic-alignment failures on the phone

The current reader revision is `board_reader_multi_seed_v5`. Automatic analysis first
uses the original image-wide reader and its original 3.5 cm seed-association limit.
An uncertain original result first tests a bounded set of alternative wood outlines
against recognized labels, then invokes perspective-based or combined-pose fallbacks
on alternative board outlines as well (7 cm proposal limit for those fallback fits). Thus the new sampling/pose assumptions do not replace
an already working original reader result.

After restarting Python and reloading the frontend, the page checks the implementation
revision in the response. An older loaded Python module produces an explicit restart
message rather than an unexplained alignment failure. Uncertain results list the
reader paths attempted and their rejection reasons.

After an analysis, **Download alignment diagnostic** exports `alignment-diagnostic.zip`
containing `frame.jpg` (the exact JPEG submitted) and `diagnostic.json` (result/error,
frontend revision, dimensions and timestamp). This export does not require starting a
recording session or selecting corners. It stays local until the user shares it; the
server still does not automatically store camera photos. It is a diagnostic format,
not a guided scan recording for `replay_scan.py`.

Selected saved-photo tests do not reproduce every live phone failure. Use a diagnostic
from the actual failing Black-side or side view before claiming that failure resolved.

## Actual phone diagnostic: board mistaken for wooden table

The uploaded diagnostic ran revision v4 and reproduced `orientation_uncertain`.
Its printed labels were recognized, but the only fallback outline enclosed much of
the wooden table instead of the board. Color is now used to propose up to six
outlines at different saturation levels; recognized glyph identities and geometry
choose among them. Equally supported different mappings are rejected. Character
component size/aspect filters are also symmetric under a quarter-turn so slender
numerals and letters remain proposals in horizontal side views.

The exact uploaded frame now produces a Black-side candidate with 23 consistent
markings and correct square identities. Its overlay was inspected. Ten known
rotations of this same photo all pass, including both side views and four diagonal
views; maximum corner deviation is below 4.2 pixels relative to transformed source
alignment. Four existing development photos covering all sides still pass. These
are still-image checks, not proof of uninterrupted live tracking.

The original JPEG and before/after reports are under `outputs/scanning/diagnostic/`;
the corrected preview is `fixed_overlay.jpg` and rotated-image results are in
`orientations/report.json`. Regression tests cover a wooden table merging with the
board, competing board mappings, and narrow glyph proposals at every quarter-turn.
After this Python update, restart the HTTPS server and reload the phone page.

## Performance update

The HTTPS bootstrap route now tries verified marking outlines and board-plane character sampling before the full checker candidate search. It retains the existing geometry, ambiguity and square-boundary guards and falls back to classical detection when needed. Both CNNs are prepared before the server accepts requests. Routine orientation checks are spaced thirty seconds apart and yield to piece scanning/review; actual tracking recovery remains enabled. Local timing and rotated-photo verification details are in [guided_scan_recognition.md](guided_scan_recognition.md#acquisition-and-retry-latency).
