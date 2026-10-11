# Still-frame board bootstrap

The phone prototype now proposes board geometry on a frozen frame and can refine
manually selected corners using visible checker transitions. This is the first
Step 04 implementation slice; robust automatic orientation from every camera side
and live tracking remain pending.

## Use on the phone

Restart the Python HTTPS server to activate `/api/bootstrap`, then reload the phone
page. The frontend build has already been regenerated in this workspace.

```bash
.venv/bin/python scripts/serve_scan_web.py --host 0.0.0.0 --port 8443 \
  --cert web/.certs/dev.pem --key web/.certs/dev-key.pem
```

Start camera, frame the entire board, then freeze a sharp view:

- **Find board automatically** proposes playing-area corners, reads the printed
  files/ranks using the existing board-specific marking templates, and fits the
  internal square boundaries. An uncertain result does not generate square labels.
- **Refine selected corners** starts with your a1 → h1 → h8 → a8 selections. It
  checks marking evidence against that supplied orientation, and rejects a conflict.
  When markings are unreadable, the orientation remains explicitly manual.

Both paths leave the full source image visible and project 64 individually fitted
square polygons back onto it. No rotation or playing-area-only crop is applied to
what you see. Hidden/weak boundaries are inferred by the existing robust lattice
fit; the result is a **candidate requiring review**, not an accuracy claim.

Check a1/h8, occupied-square bases and several empty-square boundaries. Undo/reset
removes the refined result. **Save aligned frame** records the refined polygons
with that frozen frame; replay validates their identities relative to the supplied
corners. Recording and continuous frames do not inherit this static geometry.

Analysis sends only the selected JPEG to the laptop's local HTTPS server. It is
processed in memory and is not saved by the endpoint. Continuous video and FEN are
not sent. The existing one-photo detector takes several seconds on this CPU, so
this endpoint is for bootstrap rather than frame-by-frame tracking. Only one
analysis request is processed at a time; busy requests return 429.

## Reproduce the scan experiment

```bash
.venv/bin/python scripts/bootstrap_scan.py outputs/scanning/captures/YOUR_SCAN.zip \
  --output-dir outputs/scanning/bootstrap/YOUR_SCAN --stride 6
```

The CLI first validates the archive, then evaluates every sixth frame plus any
saved aligned reference. Duplicate source hashes are evaluated once. Automatic
inference receives only the image and fixed detector configuration. A separately
reported assisted path receives manual corners from that exact frame. Neither
path uses position ID, FEN, piece arrangement or capture view notes for orientation.

`bootstrap.json` preserves source hashes, detector/marking diagnostics, fitted
polygons, rejection reasons and descriptive elapsed times. Candidate overlays
remain in the original frame coordinates. Comparison against manual corners is
not ground-truth accuracy: manual selections themselves have measurement error.

## Results from the two uploaded office scans

The sampled experiment evaluated 11 unique images automatically: five from the
first scan and six from the aligned scan. **Two produced automatic candidates**;
the other nine were rejected for localization, uncertain markings or excessive
refinement shift. Both automatic candidates were visually checked against board
markings. These checks do not establish general accuracy.

The aligned reference produced a candidate on the assisted path. Marking recognition
agreed with the manually supplied orientation, and internal grid fitting generally
followed visible square boundaries. The old detector initially proposed a displaced playing-area seed on this image.
The new bootstrap tests neighboring one-cell offsets against printed coordinates
when the initial orientation is uncertain. It recovered the reference automatically
with an offset of −1 cell on both image axes. Its four corners lie about 5–13 pixels
from the manual selections; this is agreement with a manual reference, not measured
true boundary error. Competing equally supported offset hypotheses are rejected.
This demonstrates why localization and orientation must be checked together rather
than assigning labels from detector score alone.

These are previously seen development positions, not a fresh test set. Both scans
remain mainly White-nearest; no new claim is made about unseen sides, other boards,
lighting, mirrored inputs or exact board-state recognition. The marking bank is
still specific to this wooden board. Original Week 2 data/model files are unchanged.

The initial grid-evidence fraction (.25) and refinement-shift guard (.45 squares)
are conservative engineering rules without validation-tuned confidence meaning.
Results always retain `review_required=true`. Rejection rates indicate that the
current automatic detector needs improvement before reliable guided completion.
Next work should compare better playing-area candidate generation and marking
localization on varied views, then add local tracking initialized from a reviewed
bootstrap. Live tracking, classifier calls, multi-view fusion and completion remain
disconnected; optional external markers have not been added.

## Learned reader update

A separate [trained coordinate reader](learned_board_markings.md) is now enabled.
The results above describe the earlier template baseline; on the same sampled
images the new candidate count increased from 2/11 to 4/11. Automatic proposals
still require review, and the old template path remains a fallback.
