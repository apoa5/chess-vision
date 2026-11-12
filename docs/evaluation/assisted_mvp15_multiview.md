# MVP-1.5 — Guided multi-view assisted recognition

Implemented 2026-11-12. Offline/local assisted application only. Single-photo
MVP-1 and the existing default guided-scan page remain available. No model,
geometry threshold, crop, orientation rule, precision, or global production
default was changed. No training, cloud execution, automatic completion, engine
invocation from the application, or final-holdout access occurred.

## Application workflow

Open `/?mode=photo`, choose **Multi-View Scan**, attest that pieces will remain
unchanged, then Start Scan. Rear-camera preference, a live preview and concise
whole-board/slow-motion/elevated-view guidance are provided. Frames are sampled,
not continuously classified. Finish Scan stops the camera, waits for the pending
candidate check, selects up to three complementary usable views, and queues
recognition. Cancel and retry are available. Camera tracks, abort listeners,
object URLs and abandoned sessions are released.

The same MVP-1 editor receives the actual draft: canonical board flipping,
all twelve pieces and empty-square editing, undo/reset, game-state fields,
structural validation, explicit owner attestation, Confirm Position and Copy
FEN. Edits and new scans invalidate confirmation. Source-grid inspection,
per-square per-view predictions and disagreement markers aid review. Every
square remains editable regardless of confidence. No score certifies correctness.

## Reused boundaries and safeguards

- `assisted_photo.PhotoService` and C.11 runtime retain one loaded C.9 model.
- Original `context1_224`, RGB+binary-mask, ImageNet normalization, 13 classes,
  FP32, channels-last and canonical a8–h1 ordering remain unchanged.
- Checkpoint: `outputs/evaluation/convnext_c9/export_v1/convnextv2_tiny_best.pt`.
  SHA-256: `45b471edb751261e9da1ef8703f3d9c940cdaec69f5713f6847e48bf0dc26acb`.
- Capture preflight reuses unchanged legacy automatic geometry and D.1 quality
  descriptors. B.1b recovery is not promoted. Completion is explicitly
  `completed_unverified`; canonical mapping, finite convex in-image polygons,
  resolved semantic orientation and source-hash/session binding are required.
  This cannot prove fine geometry correctness or detect every wrong accepted grid.
- Selection reuses D.1 quality/diversity; labels and classifier outcomes are not
  selection inputs. Descriptors are proxies, not measured physical camera poses.
- Raw upload: JPEG/PNG, maximum 3 MiB and 2560-pixel long edge. Capture: 24
  attempts/90 seconds, one check in flight, 3-second spacing after completion.
  Retain at most nine compressed candidates, not an uncompressed video.
- Server: four sessions, 64 MiB global compressed cache, 600-second expiry;
  two request slots and one CPU-critical job at a time. One running job and one
  waiting job cannot create unlimited model copies. Status/cancel do not wait
  for inference. Client post-scan timeout is 120 seconds.
- Source/square identity travels with every chunk. A single usable view calls
  the unchanged single-photo path. Failed selected crops are reported and are
  not silently replaced. No usable frame gives actionable retry/manual-photo help.

## Registered benchmark

Design: `docs/evaluation/assisted_mvp15_design.md`. Corpus:
`data/metadata/assisted_mvp15_corpus_v1.json`. Results root:
`outputs/evaluation/assisted_mvp15_v1/`.

Fifteen reviewed D.2 frames (three per five original HD pilot sessions) plus
three registered C.11 small-margin validation examples form the reference
inventory. Strict parity tests use nine boards / 576 squares, including the
small-margin cases; timing uses one identical three-view triplet. All input
hashes and canonical indexing are recorded. Each strategy runs in a separate
process, one full-triplet warmup and three timed triplets, without simultaneous
heavy test work. No synthetic random tensors are used for performance claims.
The optional third-worker extension was registered separately before its run
in `optional_parallel3_design_v1.json`; original decisions remain preserved.

| FP32 configuration | Median 3-view runtime | Peak RSS MiB | Max logit error | Changed squares/boards |
|---|---:|---:|---:|---:|
| Sequential, batch 8, 4 threads | 18.79 s | 1094 | 0 | 0 / 0 |
| Combined chunks 8, 4 threads | 18.56 s | 1095 | 0 | 0 / 0 |
| Combined chunks 16, 4 threads | 20.21 s | 1278 | 0.00001144 | 0 / 0 |
| Combined chunks 32, 4 threads | 18.85 s | 1532 | 0.00001144 | 0 / 0 |
| **Shared model, 2 workers, chunks 8, 2 threads** | **15.37 s** | **1422** | **0** | **0 / 0** |
| Shared model, 3 workers, chunks 8, 1 thread | 16.91 s | 1457 | 0 | 0 / 0 |

Two workers are selected: approximately 18.2% faster than sequential on this
triplet. Each uses inference mode; parameters are shared read-only, not copied.
CPU thread settings are changed only inside serialized inference and restored.
Chunk outputs are written back by `(view, canonical square offset)`, regardless
of completion order. Peak RSS includes model/input preparation in each process.
Exact runtime versions and CPU identity are in `environment_v1.json`; CPU utilization, repeated timings, loading and input hashes are in `reference.json` and `benchmark_*.json`.

Frozen C.11 tolerance is atol=2.6702880859375e-5, rtol=1e-5. All candidates pass
numerical and categorical checks, including every timed repetition. Reference
and optimized input arrays are identical. The selected two-worker implementation
has zero observed logit difference, not a universal bit-exact backend guarantee.
The machine is an Intel Core i5-1135G7 (eight logical CPUs), running WSL2, Python 3.12.3, CPU PyTorch 2.14.1 and timm 1.0.22. No precision reduction or quantization was attempted. Sequential C.11 execution
remains an explicit diagnostic option; no checkpoint substitution is allowed.

## Optional Intel GPU feasibility

OpenVINO is not installed, and neither `/dev/dri` nor `/dev/dxg` is exposed in this execution environment. The bounded check is recorded in `intel_gpu_feasibility_v1.json`. No GPU inference, driver change, device fallback or FP32 GPU parity is claimed. This does not establish that the physical laptop lacks its Iris Xe GPU; integration is deferred without blocking CPU functionality.

## Fusion decision

`fusion_v1.json` evaluates best whole quality view, arithmetic mean probabilities
and softmax of arithmetic mean logits on the same three selected views across
five pilot sessions. All three achieve 320/320 squares and 5/5 exact boards:
zero empty false positives, zero occupied errors and zero corrections. Thus
neither aggregation improves the single-view baseline. **Default: best_quality,
not probability/logit fusion.** All views' distributions and disagreements remain
available for review; three views are not three independent observations.

The prespecified promotion rule requires strictly fewer errors and no per-session
occupied/empty regression, lost exact board or newly confidently wrong occupied
square. No strategy met the strict-gain condition. D.3 MobileNet findings are not
validation of ConvNeXt fusion. These five positions were exposed in C.9 training;
this is a functional development audit, not independent recognition evidence.
C.10 photographs were not used to tune selection, fusion or thresholds.

## Real application replay

`scripts/replay_assisted_mvp15.py` sends the first 24 registered chronological
3-second HD samples of the starting-position pilot through the real HTTP API,
including all failures. It uses temporary in-memory JPEGs, original timestamp
references and hash-checked historical frame derivatives, without duplicating
videos. Budgets 1/2/3 share the same source candidates and settings.

| Selected views | Input preparation | Model/runtime | Post-scan server | Finish-to-ready client | Corrections |
|---|---:|---:|---:|---:|---:|
| 1 | 0.02 s before single-photo path | 6.17 s | 10.00 s | 10.19 s | 0 |
| 2 | 0.86 s | 9.80 s | 10.66 s | 10.88 s | 0 |
| 3 | 1.25 s | 15.21 s | 16.46 s | 16.73 s | 0 |

One-view fallback also reruns the unchanged single-photo geometry (3.34 seconds)
and crop preparation; its complete photo path was 9.94 seconds. This overhead
is reported rather than counted as inference. Two/three-view geometry was already
checked and source-bound during capture, not omitted from resource accounting.
The three-view candidate replay took 43.89 seconds before Finish. Of 24 attempts,
14 failed geometry/orientation and eight were retained at some point; diversity
pruning accounts for the rest. Per-frame reasons and all selected grids, raw
logits, probabilities, timestamps, geometry status and work counters are retained
in `application_replay_v1*.json`.

The server lifetime peak after these replays and single-photo regression was
1,286,612 KiB (about 1,257 MiB); this is not a fresh-process isolated peak for
each case. Isolated strategy RSS is in the benchmark table. The nine-second
three-view objective is **not achieved**. Accuracy was not traded for latency.
Three-view timing is one development sequence, not a distribution across phones
or arbitrary user scans. No inference speedup is claimed for different hardware.

## API and configuration

Existing `/api/photo/recognize` and `/confirm` remain unchanged. New endpoints:

- POST `/api/photo/scan/session`: explicit unchanged-position assertion.
- POST `/api/photo/scan/candidate`: bounded raw image with session/frame/timestamp
  headers; returns retention decisions; never calls ConvNeXt.
- POST `/api/photo/scan/finish`: claims the session once, returns queued; optional
  bounded `view_budget` 1–3 is used by reproducible engineering replays.
- GET `/api/photo/scan/status?session_id=...`: real phases and input work counts.
- POST `/api/photo/scan/cancel`: releases cached sources and prevents late draft.

The final response uses the existing draft schema plus `multiview`: submitted,
selected and accepted counts; per-view binding, orientation, grid and predictions;
selection trace; warnings; disagreement squares; timing and explicit policy.
A completed grid is never labeled human-verified. Failed/unresolved views are
excluded with reasons. API request size, session identity, TTL, source binding,
monotonic timestamps and concurrency checks fail closed. Scores are uncalibrated.

`configs/application/assisted_multiview_v1.json` pins the measured policy evidence
and its SHA-256. `data/metadata/assisted_mvp15_policy_v1.json` contains the small
tracked decision; large logits and runtime outputs stay ignored. ConvNeXt remains
opt-in via `--assisted-photo`, not the old production profile's global default.

## Reproduction and smartphone access

```bash
npm --prefix web run build -- --emptyOutDir false
outputs/evaluation/convnext_c11_env/bin/python scripts/serve_scan_web.py \
  --assisted-photo --host 127.0.0.1 --port 8765
# Open http://127.0.0.1:8765/?mode=photo and choose Multi-View Scan.
# Optional sequential diagnostic: add --scan-strategy sequential.
```

For a phone on the same trusted LAN, reuse the MVP-1 TLS procedure: a trusted
certificate matching the laptop LAN hostname/IP, permission to reach the port,
and camera permission. Start the same server with `--host 0.0.0.0 --port 8443
--cert web/.certs/dev.pem --key web/.certs/dev-key.pem`; open
`https://YOUR_LAPTOP_LAN_IP:8443/?mode=photo`. WSL forwarding/firewall setup may
be necessary. Do not transfer private keys or expose this development server as
an authenticated public service. Camera frames go to the laptop, not on-device
PyTorch, Modal or an external image service. Physical smartphone testing was
not performed. Detailed existing setup: `assisted_mvp1_integration.md`.

Benchmark commands (outputs are write-once; use a new versioned root/registry
for a fresh campaign rather than overwriting the completed results; pass the same `--output-root outputs/evaluation/assisted_mvp15_run2 --registry data/metadata/assisted_mvp15_corpus_run2.json` to each benchmark command):

```bash
outputs/evaluation/convnext_c11_env/bin/python scripts/benchmark_multiview_mvp15.py register
outputs/evaluation/convnext_c11_env/bin/python scripts/benchmark_multiview_mvp15.py reference
# Run one candidate process at a time; same command for each named strategy:
outputs/evaluation/convnext_c11_env/bin/python scripts/benchmark_multiview_mvp15.py candidate --strategy parallel2
# sequential, combined8, combined16, combined32, optional parallel3
outputs/evaluation/convnext_c11_env/bin/python scripts/benchmark_multiview_mvp15.py choose
outputs/evaluation/convnext_c11_env/bin/python scripts/benchmark_multiview_mvp15.py fusion
outputs/evaluation/convnext_c11_env/bin/python scripts/replay_assisted_mvp15.py
```

`inference_selection_v1.json` preserves the original five-strategy decision;
`inference_selection_v2.json` adds the separately registered optional third-worker
result. Both select parallel2. Replay uses `replay_source_v1.json`, preserving
original source-video identity and timestamps. No full recording is retained by
the live scan API.

## Validation and preservation

See `frontend_final.log`, `build_review.log`, `multiview_final.log`,
`python_regressions.log`, `engine_regression.log`, browser result JSONs and `integrity_v1.json` under the
versioned output root. Browser evidence: `browser_multi/mobile_review.png`,
`browser_single/desktop-confirmed.png` and `browser_single/mobile-review.png`.
A separate actual desktop browser replay (`browser_actual/result.json`, `browser_actual/confirmed.png`) sends an approved starting-position frame through a canvas MediaStream to the real candidate/Finish APIs, runs automatic geometry and ConvNeXt, then explicitly confirms the correct full FEN. It uses the one-view fallback; the actual three-view API replay supplies the multi-view computation evidence. This is a simulated camera, not a physical device.

The six multi-view UI checks replay an actual saved three-view response while
using Chromium's fake native camera; the independent HTTP replay performs real
geometry and inference. The 13 single-photo browser checks include a real frozen
C.10 replay, canonical piece edits, confirmation/FEN copy, cancellation and mobile
layout. This replay is a frozen regression, not C.10 accuracy tuning.

The new 30 Python worker/session tests include deterministic indexing and ties,
all four semantic turns, malformed/unresolved grids, cancellation, source/session
binding, crop bounds, repeated Finish and invalid budgets. Seven frontend tests
cover candidate cache/URL bounds, image-size limits, timeout/cancellation and raw
versus aggregate output contracts. Full regressions: **928 Python tests passed** (888 existing tests, 30 new scan tests, 10 existing engine tests); **56 frontend tests passed**, production TypeScript/Vite build passed, 13 single-photo and six multi-view browser checks plus the actual desktop scan-to-confirmation scenario passed. Two all-in-one Python attempts were interrupted at the sandboxed engine subprocess tests; the engine tests passed in 3.58 seconds outside the sandbox. The remaining 888 tests passed in 116.74 seconds, and the 30 scan tests passed in 2.32 seconds. The bounded engine retry outside the sandbox resolved the stall; new worker tests are also recorded separately; no test failures were suppressed.

Protected inputs: 8,180 files checked, 8,173 unchanged, seven authorized MVP
application edits, **zero unexpected changes and zero missing files**. Historical
reports, media and model artifacts are byte-identical. The final holdout was not
opened. The inherited ConvNeXt pretrained-weight **CC BY-NC 4.0** restriction
still applies; runtime optimization does not remove licensing restrictions.

Reproduce regression checks with:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 outputs/evaluation/convnext_c11_env/bin/python -m pytest tests --ignore=tests/test_assisted_multiview.py --ignore=tests/test_engine.py -q
outputs/evaluation/convnext_c11_env/bin/python -m pytest tests/test_assisted_multiview.py -q
outputs/evaluation/convnext_c11_env/bin/python -m pytest tests/test_engine.py -q
npm --prefix web test
MVP1_BROWSER_OUTPUT=outputs/evaluation/assisted_mvp15_v1/browser_single node web/scripts/verify-assisted-photo.mjs
node web/scripts/verify-assisted-multiview.mjs
node web/scripts/verify-assisted-multiview-real.mjs
python3 scripts/check_assisted_mvp15.py --output /tmp/mvp15-integrity.json
```

The browser harnesses reuse the existing ignored MVP-1 Playwright installation and Chromium executable. No browser dependencies were added to the application. The desktop real-camera-replay test exposed an inherited CSS rule hiding capture mode navigation above 900 pixels; a scoped assisted-app override fixes it without altering the old guided page.

## Files and acceptance checklist

New implementation: `src/scanning/assisted_multiview.py`, `multiview_inference.py`,
`web/src/MultiViewScan.tsx`, `scanCapture.ts`; benchmark/replay/preservation scripts;
registered corpus, policy/profile, design; Python and frontend tests; multi-view
browser harness. Modified application files: same server, frontend mode/editor
integration, shared board display typing, result typing, scoped CSS and test script.
No second frontend, model service or engine was introduced.

- [x] Start/Finish/Cancel/Retry; bounded compressed candidates; camera cleanup.
- [x] D.1 selection of up to three canonical usable views with explicit failures.
- [x] Real frozen ConvNeXt inference and measured shared-model concurrency.
- [x] Zero top-1/placement changes on registered parity corpus; reference retained.
- [x] Conservative evidence policy; no unsupported fusion promotion.
- [x] Existing editor, game state, confirmation and copy-FEN retained.
- [x] Single-photo regression; local API and mobile-emulation evidence.
- [x] End-to-end timing, memory and unmet performance target reported.
- [x] Historical integrity and noncommercial provenance retained.
- [ ] Physical smartphone verification (not claimed; deployment setup documented).

Next milestone: an owner-operated phone usability check of capture and correction,
then the separately authorized confirmed-position Stockfish handoff. Mandatory
review remains essential. No additional research, training, quantization or MVP-2
implementation was started.
