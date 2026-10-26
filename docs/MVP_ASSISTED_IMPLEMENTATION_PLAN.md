# Assisted Chess-Vision MVP — next engineering milestone

2026-10-26. Product direction: **guided scan → manual Finish Scan → editable reconstructed position → explicit user confirmation → Stockfish analysis**. Automatic completion, perfect calibration, additional captures and model retraining are not prerequisites. D.3c results/limitations are recorded separately; this plan does not claim predictions are verified. Its provisional quality-selected single frame and mean logits tie at 84.53% overall, both with 0/10 exact boards. Start with a reviewed single-frame draft and mandatory corrections; live fusion is not justified as a delivery prerequisite.

## Existing functionality to reuse

- `web/src/main.tsx`, `useTracking.ts`, `tracking.worker.ts`: smartphone camera, manual/automatic geometry acquisition, continuous tracking, selected-frame handoff and manual alignment fallback.
- `web/src/useRecognition.ts`, `recognitionSession.ts`, `recognitionChecks.ts`: bounded same-frame recognition requests, epoch/generation/stale-response guards, pause/retry/confirmation states and export. Current flow opens review on the first completed recognition, then locks the position on confirmation; it does not gather a fused live scan.
- `web/src/DigitalBoard.tsx`, `ChessPiece.tsx`: canonical 64-square preview, board flipping and square inspection. Selecting a square currently inspects it; it does not edit its piece.
- `web/src/PositionReview.tsx`: existing review dialog, retry and explicit confirmation. Extend it instead of creating another review workflow.
- `scripts/serve_scan_web.py`: local HTTPS/static server and `/api/bootstrap`, `/api/scan/session`, `/api/scan/keyframe`, bounded inference queue, metadata validation and error responses. No engine-analysis endpoint exists yet.
- `src/scanning/recognition.py`: current live model/crop profile is the frozen production baseline. It is **not** the experimental B.2 tighter-224 profile. Never point it at B.2 weights while leaving its old preprocessing/hash contract in place.
- `src/chess/board_state.py`, `fen.py`, `validation.py`: canonical piece placement, explicit `GameState`, placement/full FEN serialization and plausibility checks.
- `src/chess/engine.py`: existing Python UCI Stockfish wrapper, executable resolution, bounded analysis, centipawn/mate and principal variation. Reuse this wrapper; no new engine or browser-WASM integration is needed for the assisted laptop/phone demo.

## Stage 1 — Finish Scan and editable, confirmed position

**Single next Codex task:** implement manual **Finish Scan** and an editable position-review draft using the existing camera, recognition hook and review dialog. Output a versioned, user-confirmed canonical position and explicit full-FEN game-state contract. Do not add automatic completion or make multi-view fusion a dependency.

Changes:

1. Add a manual Finish Scan action. Freeze a snapshot of the latest valid recognition candidate, stop new recognition work, and invalidate or safely await outstanding requests. Camera tracking may continue. If no usable result exists, explain the condition and retain manual alignment/retry; offer a clearly labeled manual board setup if implemented within scope. Never display an unobserved board as a recognized result.
2. Keep original machine predictions/provenance immutable. Create a separate editable draft, indexed by canonical a1–h8. Tapping a square opens a compact 12-piece-plus-empty palette; support remove, undo and reset-to-scan. Flipping display must not change square identity. Corrections must survive late responses and view changes, until reset/new position.
3. Reuse the current recognition endpoint/metadata/stale-response contract, without requiring a live fusion port. Add an explicit B.2 tighter-224 runtime profile using the frozen checkpoint plus its matching packing/normalization (see parity work below); keep the baseline profile/assets intact. Do not call current live-baseline results equivalent to the offline B.2 evaluation. Highlight available low-score/ambiguous/conflicting squares as review suggestions; preserve unknown state. Label probabilities uncalibrated. Ask the user to check **every square**, including apparently confident ones. User edits override displayed machine suggestions and retain an edit log.
4. Request/confirm side to move, castling rights, en-passant square, halfmove and fullmove counters. Camera imagery cannot establish these. Defaults, if offered, must be visible and explicitly accepted; do not invent castling rights from king/rook locations or infer history. Preserve both placement-only FEN and full FEN.
5. Run the existing structural/legality checks and show actionable issues (missing/extra kings, impossible check/state or invalid rights). Do not silently repair pieces. The user must correct invalid engine input; unsupported nonstandard positions can remain editable/exportable but not submitted to Stockfish as valid standard chess.
6. Confirmation binds the exact edited placement, game-state fields and revision. Editing either afterwards invalidates confirmation and pending analysis. No confidence value or scan frame count can substitute for this action.

Acceptance:

- A physical scan can be manually finished, at least one wrong piece can be corrected directly on the displayed board, and the user can confirm/export the corrected full FEN.
- Neither late inference responses nor display flipping overwrite corrections or remap squares.
- Machine predictions, correction provenance and confirmed revision remain distinguishable.
- Tests cover editing/removal/undo/reset, flip coordinates, Finish with pending/no result, stale epochs, confirmation invalidation and explicit FEN fields. Preserve existing camera, geometry and browser regressions.

## Stage 2 — Confirmed-position Stockfish analysis

Add a small local endpoint, e.g. `POST /api/position/analyze`, accepting only a bounded versioned request with canonical placement, explicitly supplied game-state fields, confirmed revision and request ID. Construct/validate full FEN server-side using existing chess modules; reject invalid positions and stale/mismatched response revisions. Human confirmation is an explicit UI/protocol state, not something the server can infer from model confidence.

Reuse `StockfishEngine` with a short bounded wall-clock budget, a small separate engine queue and configured executable. The existing depth-6 integration test timed out at 35 seconds during D.3c, and a .2-second-budget analysis smoke test exceeded a 15-second external deadline. Reproduce and resolve or bound analysis and cleanup during this stage before calling the engine path functional. Basic UCI startup/path checks alone are insufficient. Return best move, principal variation and White-perspective centipawn or mate score with clear perspective labeling. Show SAN where appropriate, handle checkmate/stalemate/no-legal-move cases, engine missing/crashed/timeouts and cancellation. A board edit invalidates displayed analysis; the camera/inference path must not be blocked indefinitely by the engine.

Acceptance: phone scan → Finish → correct → supply game state → confirm → analyze works end to end; no engine request before confirmation, and no stale analysis attaches to an edited board. Test the endpoint with a fake engine for protocol/error cases and existing Stockfish integration tests for real valid/invalid/terminal positions. No training or calibration work.

## Recognition-profile parity within Stage 1; optional fusion after Stage 2

As a bounded part of Stage 1, expose the existing B.2 tighter-224 model as an **explicit separate runtime profile** with its checkpoint/hash, crop/normalization/class-order contract and cached model metadata. Use `CropSpec`, `packed_input` and the existing same-frame polygon preparation, retaining camera-relative piece appearance. Prove packed inputs/logits match the offline path before enabling the profile. Preserve the production baseline assets and a clear fallback.

Use D.3c's observed method tradeoffs to decide whether to add bounded multi-view evidence to this same manually finished flow. A five-view accumulator should deduplicate observations, reject stale geometry/epochs, retain full distributions/conflicts and be bounded in RAM; user corrections always take precedence. Keep mean probabilities as a reliability comparator to mean logits. Do not equate more frames or high scores with correctness. Live fusion is optional after the assisted vertical slice, not a prerequisite for Stage 1 or Stage 2. The model/profile parity work is required before claiming to run B.2 in the browser-backed demo.

Acceptance: no checkpoint overwrite, no mismatched preprocessing, deterministic offline/runtime crop parity, explicit profile identity, preserved geometry recovery defaults and working correction/confirmation flow. Any live fusion integration needs its own scoped authorization and tests; D.3c implements none.

## Delivery boundary and deferred work

Prioritize the assisted happy path plus honest failure/retry handling. Defer automatic D.4 completion, calibration guarantees, tracking/fusion redesign, architecture changes and retraining unless a subsequent concrete engineering problem justifies them. D.3c is a limited single-setting evaluation; its uncertainty and exact-board failures are reasons to require user correction, not reasons to postpone this MVP indefinitely. No further recording request is a prerequisite.
