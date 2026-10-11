# Guided board scanning and AR roadmap

The user's intended application continuously collects observations while the user
moves the camera around a stationary chessboard. It should resolve board orientation,
all 64 square locations and the occupying pieces, then offer a digital board or
physical-board move suggestions through camera-view square overlays.

Days 13 and 14 are implemented and evaluated. The next-phase implementation guide
is [guided_scanning_ar.md](../guided_scanning_ar.md), researched on 11 October 2026.

The user has an iPhone and no Mac, and prefers a cross-device web prototype. The
plan therefore uses browser camera capture, local planar tracking and square
overlays, initially backed by the existing Python model/engine. Native ARKit is
not a core dependency; WebXR and browser-local inference are optional enhancements.
The guide contains 14 ordered implementation steps, acceptance checks and data
requirements. The new experience has not been implemented yet.

The next phase needs stable board/camera tracking, persistent physical square
coordinates, viewpoint/visibility-aware evidence collection, per-square fusion,
disagreement handling and guided capture prompts. Completion means all necessary
information satisfies measured acceptance criteria, not guaranteed 100% accuracy.
Pieces should remain still during the initial scan; changes must invalidate stale
observations. Model confidence alone cannot establish scan completeness.

After the scanned position is confirmed, integrate engine recommendations and
anchor source/destination square overlays to the board. Physical and digital
interfaces should share position state. Subsequent moves require movement detection
and synchronization. Full game-state fields cannot all be inferred from one scan.

Retain the current MobileNet baseline initially and measure multi-view benefit.
Improve training coverage and investigate target association before changing
architectures. Test-driven model improvements need a newly reserved final holdout.
A browser implementation tracks the board in video coordinates and projects
square overlays with a current homography. Depth is not required. Preserve a
camera/canvas fallback on devices without usable immersive AR support.

Day 13 preserves square polygons, chess coordinates, orientation provenance and all
class probabilities to support this later work. It does not implement live tracking,
temporal fusion, move detection, engine overlays or a mobile frontend.
