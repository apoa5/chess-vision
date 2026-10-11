# Guided scan: first prototype

This is the first implementation slice of [the roadmap](../guided_scanning_ar.md),
covering baseline protection, initial scan contracts and the camera/alignment shell.
It does not run the classifier, track a moving camera, infer orientation, accumulate
scan evidence, reconstruct a position or recommend moves yet. Existing photos,
labels, splits and model weights are unchanged.

## Run locally

Use a Linux Node 22 runtime. In this workspace a checksum-verified Node 22.16.0
runtime is installed under ignored `tools/node/` (not committed).

```bash
export PATH="$PWD/tools/node/bin:$PATH"
npm --prefix web ci --ignore-scripts
npm --prefix web test
npm --prefix web run build
npm --prefix web run dev
```

Open the localhost address printed by Vite on the development computer.
Choose **Try demo**, then click the playing-area corners in order: a1 (bottom
left in the demo), h1, h8, a8. Tap squares to check their names. **Undo corner**
and **Reset alignment** correct mistakes. The solid grid covers the 32 cm playing
area; the dashed polygon projects the full 39 cm board including its borders.

For camera testing, choose **Start camera**, grant permission, keep the entire
board visible, then **Freeze frame**. Select the four *playing-area* corners
adjacent to a1, h1, h8 and a8 in that order, regardless of where they appear in
the image. Select the outer corner of each corner square, not its center or the
outer wooden border. This manually supplies orientation. The page rejects folded,
mirrored and degenerate corner selections. Stop/background releases the camera;
returning requires an explicit restart.

Recording frames stay local until export; the optional alignment analysis sends
one selected frozen frame to the laptop HTTPS server. The displayed square grid comes from a planar
homography of the supplied corners. It has not been fitted to visible internal
square boundaries, and remains manual geometry rather than detection output.
The camera image is never rotated, cropped to the playing area or mirrored.

## iPhone without a Mac

Camera access on the iPhone requires a **trusted HTTPS** address. A laptop's
`http://192.168...` address cannot enable camera access. A certificate warning
bypassed in the browser is not a reliable trusted-origin setup.

Use an HTTPS certificate for your development computer's LAN IP/hostname from a
local development CA (for example mkcert). Install that CA certificate on the
phone as a profile and explicitly enable its trust in Settings → General → About
→ Certificate Trust Settings. Keep the CA private key on the development computer;
only transfer the public CA certificate to the phone. Certificate creation and
phone trust are manual setup steps; this repository does not install trust roots.
Save the server certificate/key in ignored `web/.certs/`.

Build and serve on the same Wi-Fi network, using a certificate covering the exact
IP or hostname you will enter on the phone:

```bash
.venv/bin/python scripts/serve_scan_web.py --host 0.0.0.0 --port 8443 \
  --cert web/.certs/dev.pem --key web/.certs/dev-key.pem
```

Open `https://YOUR_COMPUTER_LAN_IP:8443` in Safari. Permit the port through your
local firewall if needed. An existing trusted HTTPS development host is also
suitable; this task has not published the application to an external host.
See [camera secure-context requirements](https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia)
and [mkcert's installation and mobile-device guidance](https://github.com/FiloSottile/mkcert).

## Verification and remaining acceptance

Automated checks cover all 64 identities for white-bottom, black-bottom,
white-left, white-right and arbitrary rotations, a perspective quadrilateral,
full-border projection, invalid geometry and CSS coordinate scaling. Python
checks cover physical dimensions and versioned frame/state validation.
`configs/scanning_baseline.json` records checkpoint, preprocessing, label and
Day 12/14 report hashes; it references existing class identities rather than
introducing a new classifier label mapping. `configs/scanning.json` disables
completion until quality and fusion rules exist.

On-device acceptance is still pending. Check permission allowed/denied, rear
camera, freeze and correct corner selection, tap a1/h8 and several occupied
squares, rotate portrait/landscape, background/return and stop/restart. Repeat
from all four sides. Verify the camera indicator disappears after Stop. Capture
any mismatch and the capability report before proceeding to automatic tracking.
There is no accuracy or live-AR claim from passing these geometry tests.

Selected-frame scheduling and capture/replay are now available; see
[recording instructions](guided_scan_capture.md). Processing workers, camera-device
selection, the remaining observation and analysis schemas, robust all-side automatic orientation
and tracking are future work. See [bootstrap results](guided_scan_bootstrap.md)
for the initial automatic proposals and assisted refinement. Heavy frame processing is absent in this shell, so it needs no processing
worker or continuous recognition loop yet.
