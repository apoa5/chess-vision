# Chess-Vision: Technical Research Reference

**Project:** [apoa5/chess-vision](https://github.com/apoa5/chess-vision)  
**Status:** Research reference for an active implementation  
**Prepared:** October 2026  
**Purpose:** Preserve the technical diagnosis, evidence, architecture choices, and engineering rationale behind the Chess-Vision improvement roadmap.

> **Document provenance:** This reference is reconstructed from the completed Deep Research report supplied by the project owner and previously examined repository documentation. It is not a verbatim export of the original research artifact. Measurements describe the documented evaluation state, not necessarily the latest model after subsequent changes.

---

## 1. Executive Summary

Chess-Vision aims to reconstruct a physical chess position from camera images, produce a digital board representation, generate FEN, request chess analysis from Stockfish, and eventually display guidance over the physical board.

The long-term user experience is a guided, continuous, multi-angle scan resembling the interaction pattern of Face ID enrollment: the user moves the camera around the chessboard while the application tracks the board, gathers complementary observations, reports scan coverage, resolves uncertainty, and presents a verified digital reconstruction.

The central finding is that the current recognition problem should **not** be treated as simply a failure of MobileNetV3 or as evidence that YOLO must replace it.

The system contains several distinct failure sources:

1. Board localization and perspective geometry.
2. Semantic orientation and square-coordinate mapping.
3. Viewpoint-dependent appearance of three-dimensional chess pieces.
4. Occlusion and neighboring-piece interference.
5. Classifier representation, training distribution, and class imbalance.
6. Confidence miscalibration.
7. Accumulation of independent square-level errors into an incorrect complete board.

The recommended approach is to measure these failure sources separately, improve the current pipeline where justified, and add multi-view evidence fusion before committing to an expensive architecture replacement.

## 2. Current System and Evidence

### 2.1 Dataset

The documented dataset contains:

- 180 physical-board photographs from 18 chess positions.
- 160 approved localized boards.
- 10,240 approved square-context crops.
- Position-grouped training, validation, and test splits: 12 / 3 / 3 positions.
- 20 photographs excluded from the approved dataset because of localization issues.

The position-level split is an important safeguard: photographs of the same physical position should not be divided between training and evaluation.

However, a small number of physical positions, chess sets, capture environments, and capture sessions still limits the generalizability of the reported results.

The dataset also contains possible confounding between orientation, environment, and viewpoint. In the completed research analysis, White-near-camera views were prominent in some environments, while Black-near-camera and White-right views were more associated with office captures. This makes it necessary to distinguish orientation effects from lighting, background, and capture-session effects.

### 2.2 Recognition model

The documented recognition model is a MobileNetV3 Small classifier using:

- RGB image content.
- A fourth channel marking the target square.
- A context crop around the target square.
- Thirteen classes: empty plus twelve piece-color/type combinations.

The model uses a 224 × 224 input. The target square occupies approximately 45 × 45 pixels within that resized context.

The surrounding context is valuable for identifying a square and tall pieces, but it also exposes the classifier to pieces standing on neighboring squares. This becomes especially difficult at oblique viewing angles.

The reported training augmentation includes brightness, contrast, blur, and noise. The research identified the absence of substantial geometric augmentation as an experiment worth testing, not as a proven single cause of the recognition failures.

### 2.3 Historical evaluation results

The documented Week 2 prepared-board test reported:

| Metric | Result |
|---|---:|
| Prepared test boards | 23 |
| Test positions | 3 |
| Overall square accuracy | 94.63% |
| Occupied-square accuracy | 82.68% |
| Exact-board reconstructions | 0 / 23 |
| Raw-photo attempts | 30 |
| Raw-photo attempts processed successfully | 16 / 30 |
| Exact-board reconstructions from raw photos | 0 / 30 |

The distinction between overall square accuracy and occupied-square accuracy is essential. Empty squares are common and relatively easy to recognize, so aggregate accuracy can hide substantial errors on the pieces that matter most.

The model also displayed high confidence on incorrect predictions. The historical evaluation reported mean confidence around 97%, despite every evaluated board containing at least one square error.

Examples of repeated confusions included:

- Black rook versus black king.
- Black king versus black queen.
- White pawn versus white knight.
- White knight versus white pawn.
- Black pawn versus black knight.

Reported recall for black kings and black rooks was especially weak.

These results establish that the pipeline is promising at the square level but not yet sufficiently reliable for unattended, exact position reconstruction.

**Do not interpret these aggregate results as proof of a specific orientation-related accuracy gap.** That requires stratified evaluation.

## 3. Why Camera Orientation Matters

### 3.1 Board rotation is not the same as viewpoint change

A chessboard is approximately planar, so a homography can transform image coordinates on the board surface into a canonical square grid.

Chess pieces are three-dimensional. Their visible appearance changes as the camera moves.

For example:

- A knight can present a distinctive profile from one side and a narrow silhouette from another.
- A rook can overlap the crown or body of a nearby king.
- A pawn viewed at a shallow angle can appear similar to the visible portion of a knight.
- Tall pieces can visually extend over neighboring squares.
- Shadows and contrast change as the camera moves.

A board homography does not reverse these three-dimensional effects.

Consequently, rotating an image or rectifying the board plane is not equivalent to obtaining a genuine photograph from another physical viewpoint.

### 3.2 Existing orientation handling

The documented dataset preserves camera-facing image appearance while using orientation-dependent square labels.

The mapping between image corners and canonical chess coordinates depends on which side of the board faces the camera.

| White's side relative to camera | Image top-left | Image top-right | Image bottom-left | Image bottom-right |
|---|---|---|---|---|
| Bottom | a8 | h8 | a1 | h1 |
| Left | a1 | a8 | h1 | h8 |
| Top | h1 | a1 | h8 | a8 |
| Right | h8 | h1 | a8 | a1 |

The repository's orientation work uses printed board coordinates to infer the semantic orientation of the physical board.

This separation is architecturally sound:

- **Camera-relative pixels** preserve the appearance the classifier actually sees.
- **Canonical square IDs** preserve the chess meaning of each observation.

The improvement plan should maintain that separation.

### 3.3 Geometry and orientation are separate gates

A detected grid can still have an unresolved semantic orientation.

Likewise, a correct orientation estimate cannot compensate for inaccurate board corners or misaligned square polygons.

The system should record separate outcomes for:

1. Board not detected.
2. Board geometry detected but unreliable.
3. Geometry valid but semantic orientation unresolved.
4. Geometry and orientation valid but recognition incomplete.
5. Recognition complete but conflicting.
6. Position accepted after validation or user review.

Combining these cases into one recognition accuracy metric conceals the true bottlenecks.

## 4. Recommended Diagnostic Strategy

### 4.1 Establish an immutable baseline

Before modifying training or inference, preserve:

- The current model checkpoint, including the documented epoch-16 checkpoint.
- Dataset split manifests.
- Evaluation configuration.
- Class-to-index mapping.
- Board and orientation conventions.
- The documented scanning baseline configuration.

Record the exact Git commit, checkpoint identifier, and dataset version.

Do not overwrite the historical baseline while running experiments.

### 4.2 Stratify evaluation

Evaluate the following viewpoints separately:

- White-near-camera.
- Black-near-camera.
- White-left.
- White-right.

Within each group, report:

- Board localization success.
- Semantic orientation success.
- Square-coordinate assignment correctness.
- Overall square accuracy.
- Occupied-square accuracy.
- Per-class precision and recall.
- Confusion matrix.
- Exact-board accuracy.
- Chess-valid position rate.
- Confidence calibration.
- Failure examples with annotated images.

Where metadata permits, also stratify by environment, lighting, camera elevation, capture session, and physical position.

### 4.3 Compare controlled and end-to-end evaluation

Use two complementary test modes.

**Prepared-geometry mode:** Evaluate recognition with approved board corners and verified orientation. This isolates classifier and crop behavior.

**Raw end-to-end mode:** Start with the original camera image and run the actual geometry, orientation, and recognition pipeline.

A discrepancy between the two reveals how much error originates before classification.

### 4.4 Collect genuinely independent holdout data

New evaluation photographs should include:

- All four camera-side orientations.
- Low, medium, and high camera elevations.
- More than one lighting condition.
- Distinct capture sessions.
- Multiple physical positions.
- Where possible, another board or piece set.

Keep the new holdout isolated from model selection and training.

Repeated video frames of the same board are not independent examples and must not be treated as such.

## 5. Improving the Existing Recognition Pipeline

### 5.1 Investigate the crop representation

The existing context crop offers useful surrounding information but may dilute the target square.

Controlled experiments should compare:

1. Current context crop and target mask.
2. Higher-resolution context crop.
3. Tighter target-centered crop.
4. Target crop plus separate context branch.
5. Reduced or varied surrounding padding.

Every experiment should preserve canonical square labels and use identical evaluation splits.

### 5.2 Geometric augmentation

Test modest, physically plausible geometric perturbations:

- Small rotations.
- Translation.
- Scale variation.
- Perspective perturbation.
- Camera framing changes.

Augmentation must transform image content and target-square masks consistently.

Avoid transformations that create implausible geometry or silently change the semantic relationship between the square mask and its label.

Synthetic augmentation should complement, not replace, actual side-view photographs.

### 5.3 Class imbalance and hierarchical recognition

The empty class dominates the dataset.

An occupancy-first approach could separate:

1. Empty versus occupied.
2. Piece color.
3. Piece type.

This might reduce competition between the common empty class and rare piece classes. It could also make error analysis easier.

However, hierarchical recognition is a benchmark candidate, not an automatic improvement. Errors in the first stage can propagate to later stages.

### 5.4 Confidence calibration and abstention

A high softmax score does not necessarily represent a reliable probability of correctness.

Temperature scaling is a reasonable calibration baseline.

Evaluate calibration on a dedicated validation partition and report reliability diagrams and calibration error.

The live scanner should be allowed to output:

- Unknown.
- Candidate.
- Resolved.
- Conflict.

It should not be forced to assign a final piece class to every square after a single uncertain observation.

## 6. Should Chess-Vision Switch to YOLO?

### 6.1 What YOLO could improve

An object detector can potentially identify pieces using more of their visible three-dimensional appearance, rather than relying entirely on a square-centered crop.

This may be useful for:

- Oblique views.
- Tall or overlapping pieces.
- Distinguishing certain piece silhouettes.
- Detecting multiple pieces in one image.

### 6.2 Why ordinary bounding boxes are insufficient

Chess-Vision needs to know **which board square supports each piece**.

The center of a bounding box is not necessarily above the piece's base.

Even the bottom-center of a box is only a heuristic, especially when:

- The base is occluded.
- The piece leans visually toward the camera.
- The detector box includes shadows or neighboring pieces.
- The camera angle is steeply oblique.

A better detector experiment would predict a piece-base or board-contact keypoint.

The intended mapping is:

**Detected piece → base/contact keypoint → image coordinate → board-plane coordinate → canonical square.**

The direction of homography application must be verified against the actual implementation's matrix convention.

### 6.3 Alternative architecture candidates

| Candidate | Hypothesis |
|---|---|
| Improved masked-context MobileNetV3 | Better data and representation are sufficient |
| Occupancy-first model | Separating empty/occupied from identity improves difficult classes |
| Piece detector with base keypoints | Object-centric recognition handles oblique pieces more reliably |
| Board-wide model with 64 outputs or square queries | Global context reduces independent-square errors |

Do not implement all alternatives simultaneously.

The decision should depend on a controlled benchmark using identical held-out positions, comparable latency constraints, and exact-board accuracy.

### 6.4 Relevant literature

The research identified three useful reference directions:

**ChessCog:** A pipeline-oriented approach combining board geometry, occupancy, and piece recognition, with emphasis on adapting recognition to the actual physical chess set.

**ChessReD:** A real-world chess recognition dataset emphasizing diverse smartphone viewpoints and the challenges of complete board reconstruction.

**ChessQueries:** A whole-board reasoning approach suggesting that jointly modeling board squares can outperform purely independent predictions on some benchmarks.

Published results are not direct predictions of performance on Chess-Vision's board, camera, or runtime environment.

## 7. Continuous Multi-View Scanning

### 7.1 Target architecture

The recommended live architecture has distinct processing loops:

**Camera rendering → geometry tracking → useful-frame selection → sparse recognition → per-square evidence fusion → completion gate.**

Rendering and tracking should remain responsive even when inference is slower.

Recognition should operate on selected keyframes rather than every frame.

### 7.2 Preserve existing implementation

The repository already contains a browser-based scanning prototype, board tracking, selected-frame recognition, confirmation/retry behavior, and canonical square geometry.

The new system should extend those components rather than replacing them wholesale.

The current guided-scan capture/replay facilities are useful for gathering and reproducing camera sessions, but they should not be assumed to already implement full multi-view fusion.

### 7.3 Keyframe selection

A frame should be accepted for recognition only if it adds useful information.

Candidate criteria:

- Geometry confidence.
- Image sharpness.
- Exposure and contrast.
- Board visibility.
- Tracking stability.
- Viewpoint difference from previous accepted frames.
- Expected visibility of unresolved squares.
- Time and compute budget.

Five nearly identical frames should not count as five independent confirmations.

### 7.4 Evidence fusion

Maintain an evidence record for each canonical square.

A useful record includes:

- Current state.
- Class probability distribution or calibrated score.
- Number of independent useful observations.
- Accepted viewpoint groups.
- Occlusion or visibility estimate.
- Most recent evidence timestamp.
- Conflict indicator.
- Provenance of supporting frames.

Fusion must prevent repeated observations from the same near-identical viewpoint from overpowering genuinely different observations.

### 7.5 Completion semantics

The scanner should report completion only when:

1. Semantic board orientation is resolved.
2. Board geometry is sufficiently current.
3. Required squares have sufficient useful evidence.
4. Viewpoint diversity requirements are met.
5. Serious cross-view conflicts are resolved.
6. Position-level consistency checks pass, or remaining ambiguity is explicitly confirmed by the user.

Progress should represent resolved board information, not simply elapsed time or frames captured.

## 8. FEN, Chess Validation, and Stockfish

A single photograph can usually reveal piece placement and piece colors, but not all historical game-state information required by a full FEN.

In particular, castling rights, en-passant availability, the halfmove clock, and the fullmove number may be unknowable from the visible board alone. The side to move may also require user input.

The system should distinguish:

- Observed piece placement.
- Inferred or user-supplied game metadata.
- Validation status.
- User-confirmed position.

Do not silently invent missing game-state metadata and present it as observed fact.

Stockfish should analyze a position only after the required metadata is supplied or an explicitly labeled assumption is accepted.

## 9. AR Overlay Requirements

The first useful AR milestone does not require full world-scale 3D anchoring.

A camera-aligned 2D overlay can already display:

- Board outline.
- Square highlights.
- Suggested move origin and destination.
- Unresolved squares.
- Camera movement guidance.

Use the geometry corresponding to the current rendered frame.

Never draw an overlay using stale board geometry without indicating tracking loss or suppressing the overlay.

The initial goal is a stable, intelligible overlay; sophisticated spatial anchoring can follow later.

## 10. Evaluation Priorities

The final system must be evaluated at several levels:

| Level | Primary questions |
|---|---|
| Geometry | Was the board detected and tracked accurately? |
| Orientation | Were canonical square coordinates assigned correctly? |
| Recognition | Which pieces and viewpoints fail? |
| Fusion | Did multiple views improve correctness? |
| Board | Was the entire piece placement reconstructed exactly? |
| User experience | Could users complete scans without excessive retries? |
| Deployment | Is the system responsive on target phones? |
| Analysis | Are Stockfish inputs validated and assumptions disclosed? |

Overall square accuracy is a supporting metric, not the primary definition of success.

## 11. Recommended Execution Order

1. Freeze the current baseline and evaluation artifacts.
2. Quantify failures by orientation, class, environment, and pipeline stage.
3. Gather an independent, balanced holdout.
4. Improve geometry, crop representation, training augmentation, and confidence handling.
5. Implement sparse keyframe selection and per-square multi-view fusion.
6. Benchmark alternative architectures only if evidence justifies the additional complexity.
7. Add guided scanning progress and user confirmation.
8. Connect validated board states to Stockfish and live overlays.
9. Conduct fresh end-to-end phone evaluations.

## 12. Explicit Non-Goals

This research does not recommend:

- Replacing MobileNetV3 with YOLO before establishing a diagnostic baseline.
- Training and evaluating on different frames of the same position.
- Treating raw softmax confidence as calibrated certainty.
- Counting every video frame as independent evidence.
- Inferring hidden FEN metadata from piece placement alone.
- Rebuilding working browser tracking and geometry modules unnecessarily.
- Claiming reliable complete-board reconstruction based only on aggregate square accuracy.

## 13. Repository References

Relevant project materials identified during the prior review include:

- `README.md`
- `docs/day9_dataset.md`
- `docs/day12_evaluation.md`
- `docs/day14_fen_evaluation.md`
- `docs/guided_scan_capture.md`
- `docs/guided_scan_recognition.md`
- `docs/guided_scan_roadmap.md`
- `guided_scanning_ar.md`
- `src/vision/board_detection.py`
- `src/vision/orientation.py`
- `web/`

These references describe the previously reviewed repository. Codex must inspect the current checkout before editing and must not assume every path or behavior remains unchanged.

---

**Companion document:** `docs/IMPLEMENTATION_PLAN.md`

This research reference explains **why** the changes are proposed. The implementation plan defines **what to change, how to test it, and when to stop**.