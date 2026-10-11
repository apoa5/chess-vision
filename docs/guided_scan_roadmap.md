# Planned phase after Days 13 and 14: guided board scanning and AR

The user's intended application continuously collects observations while the user
moves the camera around a stationary chessboard. It should resolve board orientation,
all 64 square locations and the occupying pieces, then offer a digital board or
physical-board move suggestions through camera-view square overlays.

Days 13 and 14 are now implemented and evaluated. Return to this objective next. It is not implemented
by the single-photo Week 2 pipeline.

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
An iPhone implementation can use ARKit for camera tracking and anchors, while our
vision code still supplies chessboard alignment and recognition. Depth is optional.

Day 13 preserves square polygons, chess coordinates, orientation provenance and all
class probabilities to support this later work. It does not implement live tracking,
temporal fusion, move detection, engine overlays or a mobile frontend.
