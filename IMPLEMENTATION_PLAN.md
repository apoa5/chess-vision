# Chess-Vision: Codex Implementation Plan

**Repository:** [apoa5/chess-vision](https://github.com/apoa5/chess-vision)  
**Companion reference:** `docs/VISION_RESEARCH.md`  
**Status:** Actionable development plan — implementation not yet started under this document  
**Prepared:** October 2026

> **Provenance:** Reconstructed from the completed Deep Research report supplied by the project owner. Task boundaries and acceptance criteria below operationalize that report; they are not claimed to be verbatim from the original missing implementation file.

---

# 0. Operating Instructions for Codex

## 0.1 Mission

Improve the existing Chess-Vision implementation until it can reliably reconstruct physical chess positions using guided, multi-angle phone scanning, produce a user-verifiable digital board, and provide Stockfish analysis with camera-aligned move overlays.

**Preserve existing working functionality.**

Do not replace the current pipeline merely because another architecture is newer or more popular.

## 0.2 Required workflow

For every task:

1. Read this plan and the research reference.
2. Inspect the current repository and relevant existing tests.
3. Confirm whether the requested functionality already exists.
4. Identify the smallest appropriate change.
5. Explain the intended implementation before editing.
6. Implement only the requested task.
7. Run focused tests and relevant broader regression tests.
8. Record metrics and generated artifacts.
9. Report any failures, limitations, and unverified assumptions.
10. Stop. Do not begin the next task without approval.

If the repository differs materially from this plan, report the discrepancy and propose an adjustment before proceeding.

## 0.3 Baseline protection

Do not overwrite or silently regenerate:

- The existing epoch-16 recognition checkpoint.
- Existing dataset split manifests.
- Existing evaluation reports.
- `configs/scanning_baseline.json`, if present.
- Ground-truth labels.
- Historical test outputs.

If a named path does not exist, locate the equivalent current artifact rather than creating a fake baseline.

Store experimental models and results under versioned experiment-specific paths.

## 0.4 Data integrity rules

- Split by physical chess position, not by individual photograph or frame.
- Keep near-duplicate frames and captures from the same session grouped.
- Do not tune models against the already-inspected Week 2 test positions.
- Create fresh holdouts for final evaluation.
- Never include unlabeled or automatically guessed data as verified ground truth.
- Preserve the distinction between camera-relative image appearance and canonical square coordinates.

## 0.5 Metrics that matter

Every meaningful recognition comparison should include:

- Localization success rate.
- Semantic orientation success rate.
- Occupied-square accuracy.
- Per-class precision and recall.
- Per-orientation accuracy.
- Exact-board accuracy.
- Chess-position consistency.
- Calibration and abstention behavior.
- Runtime latency on the intended deployment hardware.

Overall square accuracy alone is insufficient.

## 0.6 Definition of a completed task

A task is complete only when:

- Required code changes are implemented.
- Acceptance criteria are checked.
- Relevant tests pass or failures are explained.
- Reproduction instructions are recorded.
- No baseline artifact has been overwritten.
- Codex provides a concise summary and stops.

---

# Phase A — Establish the Real Failure Baseline

**Goal:** Determine precisely why the existing pipeline performs worse from Black-side and sideways views.

**Priority:** Mandatory. Complete before major model replacement.

## Task A.1 — Freeze and reproduce the current baseline

**Read first**

- `README.md`
- `docs/day9_dataset.md`
- `docs/day12_evaluation.md`
- `docs/day14_fen_evaluation.md`
- Current model-loading and evaluation code.
- Existing model configuration and checkpoint paths.
- Existing tests and package/environment configuration.

**Implementation**

1. Record the current Git commit and runtime environment.
2. Identify the exact baseline checkpoint, class mapping, preprocessing pipeline, and dataset manifests.
3. Locate the current evaluation entry point and its documented command.
4. Add a versioned baseline manifest without overwriting existing configuration.
5. Re-run the available baseline evaluation if data and dependencies are present.
6. Record any discrepancy between reproduced and documented metrics.

**Suggested deliverables**

- `docs/evaluation/baseline_manifest.md`
- A machine-readable baseline manifest under an appropriate existing configuration or evaluation directory.
- Reproduction instructions using the actual repository commands.

**Tests**

- Checkpoint loads correctly.
- Class mapping is stable.
- Dataset split isolation checks pass.
- Evaluation produces deterministic outputs within expected tolerances.

**Acceptance criteria**

- Baseline is identifiable and reproducible.
- Historical metrics are preserved.
- Any unavailable dataset, dependency, or checkpoint is explicitly reported.
- No training or model replacement has occurred.

**Stop after A.1.**

## Task A.2 — Build orientation-stratified evaluation

**Depends on:** A.1

**Read first**

- Existing evaluation scripts.
- Dataset manifests and orientation metadata.
- `src/vision/orientation.py`
- `src/vision/board_detection.py`
- Existing square-crop and inference code.

**Implementation**

1. Create or extend an evaluator that groups samples by physical camera orientation.
2. Support White-near, Black-near, White-left, and White-right.
3. Add grouping by position, session, environment, and camera elevation when metadata is available.
4. Produce separate metrics for approved geometry and raw end-to-end processing.
5. Generate per-class confusion matrices.
6. Export misclassified-square examples with true and predicted labels.
7. Distinguish geometry, orientation, classification, and position-validation failures.
8. Ensure the evaluator does not silently exclude failed raw-photo attempts from end-to-end success metrics.

**Suggested deliverables**

- `scripts/evaluate_by_orientation.py` or an extension of the current evaluator.
- `docs/evaluation/orientation_baseline.md`
- CSV or JSON results.
- Confusion matrices and failure galleries.

**Tests**

- Known sample counts match manifests.
- Every sample is assigned to exactly one relevant orientation group.
- Grouped totals reconcile with the full evaluation.
- A localization failure is counted as an end-to-end failure, not as a skipped success.

**Acceptance criteria**

- Accuracy and exact-board rates are available by orientation.
- Black-side and sideways performance gaps are quantified rather than assumed.
- Class-specific failures are identifiable.
- Confounding from environment and session is explicitly documented.

**Stop after A.2.**

## Task A.3 — Create a balanced, independent capture protocol

**Depends on:** A.2

**Read first**

- `docs/day9_dataset.md`
- `docs/guided_scan_capture.md`
- Existing capture and replay scripts.
- Current dataset labeling and review tools.

**Implementation**

1. Define a capture matrix covering four camera-side orientations, several elevations, and multiple lighting conditions.
2. Include different physical positions and sessions.
3. Specify minimum ground-truth requirements for board corners, semantic orientation, and square labels.
4. Reuse the existing phone capture and replay workflow.
5. Add metadata fields for session, position, orientation, elevation, lighting, and board identity.
6. Reserve a genuinely independent holdout.
7. Document how to exclude blurred, partially cropped, or incorrectly labeled captures.

**Deliverables**

- `docs/evaluation/capture_protocol.md`
- A capture manifest schema.
- Validation script for required metadata.
- Fresh capture data only when physically collected by the user.

**Acceptance criteria**

- Capture protocol is executable by a person with a phone.
- Data leakage checks are defined.
- The holdout cannot be used for training or model selection.
- No synthetic or invented photographs are presented as collected data.

**Stop after A.3.**

---

# Phase B — Improve the Existing Pipeline

**Goal:** Improve the current model and geometry without unnecessary architectural replacement.

## Task B.1 — Audit board geometry and square alignment

**Depends on:** A.2

**Read first**

- `src/vision/board_detection.py`
- `src/vision/orientation.py`
- Current homography and crop generation code.
- Existing board-tracking implementation under `web/`.

**Implementation**

1. Audit image-to-board and board-to-image coordinate conventions.
2. Verify canonical square mapping for all four orientations.
3. Check whether classifier target masks align with the actual physical square.
4. Visualize projected square polygons on raw images.
5. Identify cases where perspective, tracking, or corner inaccuracies cause wrong target crops.
6. Add tests for corner ordering and coordinate transforms.
7. Preserve the camera-facing image convention.

**Deliverables**

- Geometry diagnostic visualizations.
- Focused transform tests.
- A short report identifying confirmed geometry defects, if any.

**Acceptance criteria**

- Coordinate mappings pass synthetic and manually verified cases.
- No unintentional 90-degree or 180-degree semantic mapping error remains in tested paths.
- Classifier failures can be distinguished from crop-alignment failures.

**Stop after B.1.**

## Task B.2 — Benchmark crop and input-resolution changes

**Depends on:** B.1

**Read first**

- Current dataset preprocessing code.
- Existing MobileNetV3 training configuration.
- Target-mask construction code.

**Implementation**

Run controlled comparisons of:

1. Current 224-pixel masked-context baseline.
2. A higher-resolution context input.
3. A tighter target-square crop.
4. A changed context-padding strategy.

Only implement additional model branches if simpler crop experiments show a clear reason.

Keep splits, evaluation procedures, and training budget comparable.

**Deliverables**

- Experiment configurations.
- Versioned checkpoints.
- Side-by-side per-orientation and per-class metrics.
- A recommendation to keep or change the crop representation.

**Acceptance criteria**

- Baseline remains unchanged.
- All comparisons use consistent holdouts.
- Results include occupied-square and exact-board metrics.
- Any extra inference cost is measured.

**Stop after B.2.**

## Task B.3 — Add physically plausible geometric augmentation

**Depends on:** B.2

**Implementation**

1. Inspect the existing augmentation pipeline.
2. Add configurable modest rotation, scale, translation, and perspective perturbations.
3. Apply identical geometric transforms to RGB pixels and the target mask.
4. Avoid transforms that invalidate square labels.
5. Run an augmentation ablation against the unchanged baseline.
6. Inspect augmented examples manually.

**Tests**

- Mask and image alignment after augmentation.
- Label preservation.
- Deterministic augmentation when seeded.
- No malformed input tensors.

**Acceptance criteria**

- Geometric augmentation can be enabled and disabled by configuration.
- Its effect is measured separately from crop changes.
- Improvements or regressions are reported for each orientation.

**Stop after B.3.**

## Task B.4 — Introduce calibrated confidence and abstention

**Depends on:** A.2; may follow B.2/B.3

**Implementation**

1. Measure confidence calibration of the selected model.
2. Fit a calibration method using validation data only.
3. Compare temperature scaling against uncalibrated predictions.
4. Define a configurable abstention rule for uncertain square predictions.
5. Preserve raw model outputs for debugging.
6. Expose calibrated probabilities and uncertainty to the scanning layer.

**Tests**

- Calibration parameters load reproducibly.
- Predicted class ordering is preserved by temperature scaling.
- Uncertain examples can be marked unresolved.
- No test-set leakage into calibration fitting.

**Acceptance criteria**

- Reliability plots and calibration metrics are recorded.
- Scanner consumers can distinguish a predicted class from a trusted resolution.
- Calibration does not silently claim to fix classification errors.

**Stop after B.4.**

---

# Phase D — Multi-View Evidence and Fusion

**Why Phase D follows Phase B:** The completed research recommends testing the current architecture with complementary camera views before investing in a replacement detector.

## Task D.1 — Select useful recognition keyframes

**Depends on:** B.1 and A.2

**Read first**

- `docs/guided_scan_capture.md`
- `docs/guided_scan_recognition.md`
- Existing live tracking and selected-frame recognition modules under `web/`.
- Existing capture/replay tooling.

**Implementation**

1. Identify the existing camera, tracking, and recognition scheduling boundaries.
2. Preserve continuous rendering and tracking.
3. Introduce a keyframe eligibility score based on image quality, geometry stability, and novelty.
4. Reject nearly identical frames as redundant.
5. Record why each frame was accepted or rejected.
6. Support deterministic replay from recorded capture sessions.

**Tests**

- Repeated identical frames are not all accepted.
- A substantially different useful view can be accepted.
- Invalid or stale geometry prevents recognition.
- Keyframe selection does not block live rendering.

**Acceptance criteria**

- Recognition is sparse and explainable.
- Accepted keyframes have auditable provenance.
- Existing single-frame capture and confirmation remain functional.

**Stop after D.1.**

## Task D.2 — Implement per-square evidence records

**Depends on:** D.1 and B.4

**Implementation**

Create a canonical evidence representation for all 64 squares.

Each square should track:

- Canonical square ID.
- State: `unknown`, `candidate`, `resolved`, or `conflict`.
- Current class distribution.
- Supporting accepted keyframes.
- Viewpoint groups represented.
- Visibility or occlusion assessment.
- Confidence and uncertainty.
- Last update and provenance.

Define typed interfaces or schemas compatible with the existing browser runtime.

**Tests**

- Every canonical square has exactly one evidence record.
- Camera orientation changes do not change square identity.
- Duplicate observations are identifiable.
- Evidence state serializes and restores consistently.

**Acceptance criteria**

- The evidence structure can represent uncertainty without forcing a piece label.
- It is independent of rendering and specific recognition architecture.
- It does not duplicate an existing equivalent structure unnecessarily.

**Stop after D.2.**

## Task D.3 — Implement multi-view evidence fusion

**Depends on:** D.2

**Implementation**

1. Define an initial transparent fusion baseline.
2. Weight observations by calibrated reliability and quality.
3. Discount highly correlated frames.
4. Track genuine disagreements as conflicts.
5. Prevent a single low-quality observation from silently overwriting strong independent evidence.
6. Compare fusion against the best individual keyframe.
7. Make fusion parameters configurable and testable.

Start with a simple auditable method. Do not introduce a complex neural fusion model unless evidence warrants it.

**Tests**

- Identical repeated frames do not produce unlimited confidence.
- Complementary observations can resolve uncertainty.
- Contradictory strong observations produce a conflict.
- Fusion is invariant to input ordering where the chosen algorithm requires it.
- All evidence remains traceable to source frames.

**Acceptance criteria**

- Multi-view evaluation reports exact-board and occupied-square metrics.
- Results are compared with a single-frame baseline on the same sessions.
- Fusion never hides unresolved conflicts merely to mark a scan complete.

**Stop after D.3.**

## Task D.4 — Define the scan-completion gate

**Depends on:** D.3

**Implementation**

Create a completion evaluator that considers:

1. Semantic orientation resolved.
2. Current geometry valid.
3. Required squares sufficiently observed.
4. View diversity requirements satisfied.
5. No blocking unresolved conflicts.
6. Position consistency checks completed.
7. User review required where uncertainty remains.

Return both completion status and human-readable reasons for incompleteness.

**Tests**

- Missing orientation prevents automatic completion.
- Stale geometry prevents completion.
- High confidence from duplicate frames alone is insufficient.
- Unresolved conflicts prevent automatic acceptance.
- User-confirmed exceptions are explicitly recorded.

**Acceptance criteria**

- Completion is determined by board evidence, not frame count.
- The result can drive a progress indicator.
- Completion thresholds are configurable and justified against validation sessions.

**Stop after D.4.**

---

# Phase C — Alternative Recognition Architecture Decision Gate

**Execution order:** Attempt Phase C only after reviewing Phase B and Phase D results.

## Task C.1 — Decide whether an alternative is justified

**Depends on:** B.2, B.3, D.3

**Implementation**

Prepare a decision report answering:

- Which classes still fail?
- Which camera orientations remain weak?
- Are failures primarily geometry, occlusion, or identity errors?
- How much does multi-view fusion help?
- What latency and model-size budget is available?
- Is additional annotation practical?

**Acceptance criteria**

- A written decision selects either the current architecture or one alternative for benchmarking.
- No detector implementation begins without approval.

**Stop after C.1.**

## Task C.2 — Benchmark one approved alternative

**Depends on:** C.1 and explicit approval

Possible approved experiments:

- Occupancy-first classification.
- Piece detection with base/contact keypoints.
- Lightweight board-wide 64-square reasoning.

**Implementation**

1. Define annotation requirements.
2. Estimate labeling cost before relabeling.
3. Train a minimal viable candidate.
4. Preserve the current model as the control.
5. Evaluate on identical held-out sessions.
6. Include inference latency and deployment constraints.

For a detector, use base/contact points to associate pieces with board squares. Do not rely exclusively on bounding-box centers.

**Acceptance criteria**

- Candidate and baseline metrics are directly comparable.
- Extra data-labeling and runtime costs are documented.
- No existing model is replaced during the experiment.

**Stop after C.2.**

## Task C.3 — Make the architecture decision

**Depends on:** C.2

Choose to:

- Keep the improved current classifier.
- Adopt the alternative.
- Use a hybrid system.
- Defer architectural changes.

The decision must cite observed performance, robustness, latency, and implementation complexity.

**Acceptance criteria**

- Selected architecture is justified by evidence.
- Rollback to the baseline remains possible.
- The decision is recorded before integration.

**Stop after C.3.**

---

# Phase E — Guided Multi-Angle Scanning Experience

## Task E.1 — Display evidence-based scan progress

**Depends on:** D.4

**Read first**

- Existing browser UI and rendering components.
- Current confirmation/retry flow.
- Existing square geometry and tracking state.

**Implementation**

1. Display progress based on resolved canonical squares.
2. Visually distinguish unknown, candidate, resolved, and conflict states.
3. Highlight squares needing better observations.
4. Preserve current board tracking and selected-frame controls.
5. Handle tracking loss without showing stale progress as live certainty.

**Acceptance criteria**

- Progress increases only when meaningful evidence is gained.
- Conflicts are visible.
- The UI remains usable on a phone-sized viewport.

**Stop after E.1.**

## Task E.2 — Add directional capture guidance

**Depends on:** E.1 and D.1

**Implementation**

1. Use accepted viewpoint history to identify underrepresented camera angles.
2. Recommend a useful next view without requiring an exact camera trajectory.
3. Suppress instructions when tracking is lost.
4. Avoid repeatedly asking for views that add no information.
5. Preserve a manual capture/retry fallback.

**Acceptance criteria**

- Guidance reflects unresolved evidence and view diversity.
- The interface does not claim that a view has been captured unless accepted as useful.
- The workflow can be completed or exited without becoming trapped in a guidance loop.

**Stop after E.2.**

## Task E.3 — Add final review and correction

**Depends on:** E.2

**Implementation**

1. Present the reconstructed digital board.
2. Show uncertain and conflicting squares.
3. Allow the user to correct piece placement.
4. Preserve the distinction between machine predictions and user corrections.
5. Record final confirmation before analysis.

**Acceptance criteria**

- User can correct a wrong square.
- Corrected states update the digital board consistently.
- Review is required when automatic completion criteria are not met.
- Existing confirmation behavior is preserved or migrated without regression.

**Stop after E.3.**

---

# Phase F — Stockfish and Camera-Aligned Move Overlays

## Task F.1 — Build a validated position handoff

**Depends on:** E.3

**Read first**

- Existing FEN generation code.
- Existing chess validation and Stockfish integration.
- `docs/day14_fen_evaluation.md`.

**Implementation**

1. Generate piece placement from the confirmed canonical board.
2. Request or explicitly assume side to move and unavailable game-state metadata.
3. Distinguish observed fields from inferred fields.
4. Validate the position using the project's existing chess library.
5. Block or flag analysis of inconsistent or incomplete positions.

**Acceptance criteria**

- Unknown castling and en-passant information is not silently presented as observed.
- Invalid positions do not proceed as trusted inputs.
- Confirmed positions produce reproducible FEN output.

**Stop after F.1.**

## Task F.2 — Connect confirmed positions to Stockfish

**Depends on:** F.1

**Implementation**

1. Reuse the existing engine interface where possible.
2. Analyze only a validated or explicitly user-approved position.
3. Return a legal move and relevant engine metadata.
4. Handle engine failures and timeouts.
5. Keep analysis responsive and cancellable.

**Acceptance criteria**

- Returned move is legal for the submitted position.
- Engine errors do not crash scanning.
- Position assumptions are visible to the user.

**Stop after F.2.**

## Task F.3 — Render move guidance over the camera view

**Depends on:** F.2 and B.1

**Implementation**

1. Convert move origin and destination squares into current image-space polygons.
2. Highlight the relevant physical squares.
3. Tie overlays to the current tracking geometry.
4. Suppress or visibly degrade overlays when tracking becomes unreliable.
5. Test all four board orientations.

**Acceptance criteria**

- Highlighted squares match the intended canonical move.
- Overlay alignment remains correct across tested viewpoints.
- Tracking loss never leaves a misleading apparently anchored move overlay.

**Stop after F.3.**

---

# Phase G — Fresh End-to-End Evaluation

## Task G.1 — Define acceptance thresholds

**Depends on:** A.2, D.4, E.3

**Implementation**

1. Establish a fresh held-out evaluation set.
2. Set explicit targets for exact-board accuracy, orientation reliability, scan completion, latency, and correction burden.
3. Include separate metrics for automatic completion and user-assisted completion.
4. Report confidence intervals where sample size permits.
5. Freeze thresholds before final testing.

**Acceptance criteria**

- Success criteria are written before final holdout evaluation.
- Thresholds are measurable and tied to intended use.
- No final-test examples are used to tune the implementation.

**Stop after G.1.**

## Task G.2 — Run full phone-scanning trials

**Depends on:** G.1

**Implementation**

Test complete scanning sessions across:

- All four camera-side orientations.
- Different camera elevations.
- Multiple physical chess positions.
- Different lighting and environments.
- Occluded and crowded board configurations.
- Tracking loss and recovery.
- User correction and retry paths.

**Metrics**

- End-to-end exact-board accuracy.
- Automatic versus assisted completion.
- Median and tail scan duration.
- Number of useful keyframes.
- Number of retries and corrections.
- Orientation and geometry failures.
- Latency and device responsiveness.

**Acceptance criteria**

- Results are reproducible from session records.
- Every failure is assigned to a pipeline stage.
- No aggregate metric conceals orientation-specific regressions.

**Stop after G.2.**

## Task G.3 — Final regression and release readiness

**Depends on:** G.2

**Implementation**

1. Run existing unit and integration tests.
2. Verify dataset and model provenance.
3. Check browser permissions and camera fallback behavior.
4. Confirm recovery from lost tracking.
5. Confirm user correction and engine analysis.
6. Update the README with supported capabilities and limitations.
7. Document known unsupported devices or camera conditions.

**Acceptance criteria**

- Release limitations are accurately documented.
- Core scanning and review workflows pass.
- Historical baseline and experiment artifacts remain recoverable.
- No unsupported reliability claims are made.

**Stop after G.3.**

---

# Recommended Execution Sequence

Use this sequence unless a measured dependency requires an approved adjustment:

- [ ] A.1 — Freeze baseline
- [ ] A.2 — Orientation-stratified evaluation
- [ ] A.3 — Independent capture protocol
- [ ] B.1 — Geometry and square alignment audit
- [ ] B.2 — Crop and resolution experiments
- [ ] B.3 — Geometric augmentation experiments
- [ ] B.4 — Confidence calibration
- [ ] D.1 — Useful keyframe selection
- [ ] D.2 — Per-square evidence records
- [ ] D.3 — Multi-view fusion
- [ ] D.4 — Completion gate
- [ ] C.1 — Architecture decision
- [ ] C.2 — Optional alternative benchmark
- [ ] C.3 — Optional architecture selection
- [ ] E.1 — Evidence-based progress
- [ ] E.2 — Viewpoint guidance
- [ ] E.3 — User review and correction
- [ ] F.1 — Validated FEN handoff
- [ ] F.2 — Stockfish integration
- [ ] F.3 — Camera-aligned overlays
- [ ] G.1 — Final evaluation thresholds
- [ ] G.2 — End-to-end phone trials
- [ ] G.3 — Regression and release review

**Important:** Phase C is conditional. If the improved classifier plus multi-view fusion meets the agreed acceptance criteria, do not replace it merely to complete a checklist.

---

# Codex Prompt Templates

## Template 1 — Standard task

```text
Read docs/VISION_RESEARCH.md and docs/IMPLEMENTATION_PLAN.md.

Work ONLY on Task [TASK ID].

First inspect the current repository and verify that the task
is still applicable.

Before editing:
1. Summarize the relevant existing implementation.
2. Identify the files you intend to change.
3. Explain your proposed approach.
4. Identify tests and acceptance criteria.

Implement the smallest complete change for this task.

Run the relevant tests.

At the end report:
- Files changed
- Commands run
- Tests passed and failed
- Artifacts produced
- Acceptance criteria satisfied or not satisfied
- Remaining limitations

Do not start another task.
Stop and wait for my approval.
```

## Template 2 — Model experiment

```text
Read docs/VISION_RESEARCH.md and docs/IMPLEMENTATION_PLAN.md.

Work ONLY on Task [TASK ID].

Preserve the baseline checkpoint, data splits, and evaluation
configuration.

Before training, show:
1. Exact experiment hypothesis.
2. Training and validation datasets.
3. How data leakage is prevented.
4. Baseline comparison.
5. Metrics to be collected.
6. Expected compute and annotation cost.

Do not tune on the historical test positions or the fresh
final holdout.

Run the experiment only after confirming the environment
and required data are available.

Report all results, including regressions and missing data.

Do not replace the production model automatically.
Stop after this task.
```

## Template 3 — Multi-view scanning task

```text
Read docs/VISION_RESEARCH.md and docs/IMPLEMENTATION_PLAN.md.

Work ONLY on Task [TASK ID].

Inspect the current web camera, board tracking, frame
selection, and recognition flow before editing.

Preserve camera responsiveness, canonical square IDs,
and the existing single-frame fallback.

Implement the smallest testable multi-view improvement.

Use deterministic replay sessions for tests when available.

Demonstrate that duplicate frames do not count as
independent evidence and that tracking failures are handled.

Report test results and stop.
```

---

# Change Management and Decision Log

For each completed task, record:

**Task ID:**  
**Date:**  
**Git commit:**  
**Files changed:**  
**Hypothesis:**  
**Implementation summary:**  
**Tests run:**  
**Metrics before:**  
**Metrics after:**  
**Known limitations:**  
**Decision:** Accept / Revise / Revert  
**Next approved task:**  

Store entries in the existing development documentation or in a dedicated `docs/implementation_progress.md` file.

---

# Final Principles

1. Diagnose before replacing.
2. Preserve working components.
3. Separate geometry, orientation, recognition, and fusion.
4. Use canonical square IDs across all camera views.
5. Treat confidence as something to validate, not assume.
6. Prefer complementary keyframes over redundant video frames.
7. Measure exact-board reconstruction, not only square accuracy.
8. Make uncertainty visible and correctable.
9. Do not infer unobservable chess metadata as fact.
10. Implement one reviewed task at a time.

**First Codex task: A.1 — Freeze and reproduce the current baseline.**