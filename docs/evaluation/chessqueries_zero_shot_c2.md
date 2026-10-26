# C.2 — Bounded zero-shot ChessQueries ViT-S benchmark

2026-10-26. **Decision: Path C. Retain MobileNetV3 RGB+target-mask as the primary recognition architecture.** The released ChessQueries ViT-S runs comfortably on this CPU, but does not improve the fixed 55-frame D.2 comparison. It corrects some difficult identities while introducing more errors, especially missed pieces and false occupancy. This result concerns this checkpoint and input policy, not all full-board models.

| Same 55 frames / 3,520 squares | Frozen B.2 tighter-224 | ChessQueries ViT-S |
|---|---:|---:|
| Overall square accuracy | **3,009/3,520 (85.48%)** | 2,883/3,520 (81.90%) |
| Occupied-square accuracy | **902/1,313 (68.70%)** | 886/1,313 (67.48%) |
| Empty-square false positives | **100/2,207 (4.53%)** | 210/2,207 (9.52%) |
| Occupied→empty misses | **30/1,313 (2.28%)** | 273/1,313 (20.79%) |
| Incorrect occupied-piece identities, excluding misses | 381 | **154** |
| Exact single-frame boards | **2/55 (3.64%)** | 0/55 (0%) |

There are **five physical positions/sessions**, not 55 independent positions. B.2's two exact frames both belong to the puzzle session. These are development results conditional on the existing labels and qualitative geometry reviews. They are not the historical prepared-board B.2 result, D.2 uniform-5 alone, D.3 fusion results, or independent D.3c performance.

Artifacts: [visual comparison](chessqueries_zero_shot_c2_examples/index.html), [versioned results](chessqueries_zero_shot_c2_results_v1.json), [per-class CSV](chessqueries_zero_shot_c2_results_v1.csv), [h4 sensitivity](chessqueries_zero_shot_c2_sensitivity_v1.json), [validation](chessqueries_zero_shot_c2_validation_v1.json). Full uncalibrated logits/probabilities are retained once in `outputs/evaluation/chessqueries_c2_run1/predictions.json.gz`, with its SHA in the results artifact. No probability fusion, threshold fitting or legality repair was performed.

## Dataset, labels and review boundaries

Reviewed the root `VISION_RESEARCH.md` and `IMPLEMENTATION_PLAN.md` (the requested `docs/` copies do not exist), C.1 report/HTML, D.3c report, assisted-MVP plan, D.2 observations/reviews, and frozen crop/inference/evaluation code. The C.2 user instruction authorizes zero-shot inference instead of the roadmap's optional candidate-training step.

The fixed population is the **union of D.2's 55 unique selected frames**, including diversity-5, uniform-5 and diversity-10 membership, without multiplying frames appearing in more than one strategy. All 55 completed the previous opt-in marked-board geometry workflow and have the owner's four-check review: corners, semantic orientation, canonical square mapping and sufficient crop alignment. C.2 revalidates image, geometry, review, observation, model and configuration bindings. All frames are eligible; **none was replaced, excluded or selected using new model outcomes**.

Labels come from `data/metadata/a3_pilot_01_sources_v2.json`, supplemented by the separate owner-approved `evidence_d2_ground_truth_corrections_v1.json`: middlegame **c7 is a black pawn**. The original FEN/catalog and original-label historical results remain unchanged. The owner also attested photo/FEN agreement and unchanged piece placement during the recordings. This does not constitute per-piece silhouette or contact-point annotation.

Fine geometry is not pixel-perfect ground truth. C.1 flagged an endgame h4 pawn base near the h4/h3 boundary; that caveat remains. A separately labelled sensitivity calculation removes only those ten h4 observations: B.2 **85.70%** versus CQ **82.11%** overall on 3,510 squares, still a net 126-square B.2 advantage. It neither changes labels nor replaces the all-55-frame primary comparison.

The five sessions share capture history and four placements overlap the historical B.2 training positions. ChessQueries has different public pretraining/training exposure, which is not exhaustively auditable against common chess placements. Consequently this is an engineering comparison of available predictors, not a controlled causal test of architecture, crop representation, or independent generalization. D.3c data was not evaluated or tuned in C.2; its original videos were only hash-checked for preservation. The final holdout was not accessed.

## Pinned model, essential environment and licenses

Official source revision: `f5aaaf38212cc7839b03ccf383455053905b3f7e`. Downloaded **only the original-paper ViT-S release**, not ViT-B/L or proprietary further-trained LiteV4. The filename contains `chessquerieslite`, but its public manifest identifies the paper ViT-S variant. [Official model card](https://huggingface.co/joelseytre/chessqueries).

- Artifact: `chessquerieslite-vits-644-safetensors-fp16-r1.safetensors`, **63,147,914 bytes**.
- Full downloaded SHA-256 verified: `486d10b258ba57d43d8ca861d20547093ca6591c0ad53da53a492716ef1c7705`.
- Immutable distribution revision: `eadee700749fd2930a651982c9f8a753097f81dd`.
- DINOv2 ViT-S/14 encoder, four query-decoder layers, eight decoder heads; **31,559,821 parameters**. The C.1 header's 31,560,077 tensor elements additionally include buffers.

The official safe loader reconstructs with `pretrained=False` and strict state-dictionary loading. Thus every parameter comes from the released artifact and no separate encoder checkpoint is downloaded. The FP16 storage artifact is converted to **CPU FP32**, matching the official CPU predictor behavior. Evaluation mode and `torch.inference_mode()` are mandatory; batch size is one, intra-op threads four, inter-op threads one, deterministic algorithms enabled. [Loader](https://github.com/JSeytre/chessqueries/blob/f5aaaf38212cc7839b03ccf383455053905b3f7e/chessqueries/models/checkpoint.py), [predictor](https://github.com/JSeytre/chessqueries/blob/f5aaaf38212cc7839b03ccf383455053905b3f7e/chessqueries/models/predictor.py).

The isolated environment is `outputs/evaluation/chessqueries_c2_env`. A `.pth` entry reuses the existing application's **CPU** torch 2.14.1, torchvision 0.29.1, NumPy 2.5.3 and Pillow 12.3.0 read-only. Nine small pinned additions in `requirements-chessqueries-c2.txt` were installed with `--no-deps`: timm, safetensors, huggingface-hub and their needed import dependencies. No application dependency was upgraded/replaced; no torch, CUDA, GPU runtime, training framework, LLM SDK or larger weights were downloaded. Hugging Face is forced offline during inference. The optional Xet downloader is unnecessary for direct local safetensors loading and was not installed.

Upstream declares Python 3.12, torch 2.8, torchvision 0.23 and NumPy 1.26 ranges. We did not claim an exact upstream lockfile reproduction or downgrade working application packages. The reused CPU runtime passed strict checkpoint loading, real forward inference and exact official-transform parity tests. Twelve unmodified official source files are stored in the ignored release directory. A narrow namespace import adapter bypasses package-level reexports of unrelated LLM/training modules; it does not replace the model, checkpoint loader or evaluation transform. [Dependency declaration](https://github.com/JSeytre/chessqueries/blob/f5aaaf38212cc7839b03ccf383455053905b3f7e/pyproject.toml).

Code/weights are **PolyForm Noncommercial 1.0.0**, with separate permission required for commercial use; they are not automatically licensed for a commercial MVP. Dependency metadata records Apache-2.0 for timm/safetensors/requests, Apache for huggingface-hub, MPL-2.0/MIT for tqdm, MPL-2.0 for certifi, MIT/BSD-family terms for the remaining additions and separate upstream notices for the reused numerical stack. The complete local inventory is `outputs/evaluation/chessqueries_c2_dependency_inventory_v1.json`. No user images were uploaded to a third-party demonstration service. [Model/source license](https://github.com/JSeytre/chessqueries/blob/f5aaaf38212cc7839b03ccf383455053905b3f7e/LICENSE).

## Deterministic input and output contract

`src/vision/chessqueries_offline.py` reads the existing display-oriented RGB PNG through torchvision, scales to `[0,1]`, square-resizes to **644×644** with the official bilinear antialiased tensor transform and applies ImageNet normalization. No crop, padding, board warp, mask, test-time augmentation or image rotation is added. The official predictor ignores EXIF orientation; the D.2 PNGs have already been decoded into their display orientation. [Official transform](https://github.com/JSeytre/chessqueries/blob/f5aaaf38212cc7839b03ccf383455053905b3f7e/chessqueries/data/transforms.py).

The model returns finite `[1,64,13]` logits. Runtime checks verify the official `.PNBRQKpnbrqk` enumeration against Chess-Vision's numeric class order and the official square list against **a8…h8, a7…h7, …, a1…h1**. All 13 logits and float64 stable softmax probabilities are preserved; probabilities are not calibrated. Argmax ties use the first class. [Square ordering](https://github.com/JSeytre/chessqueries/blob/f5aaaf38212cc7839b03ccf383455053905b3f7e/chessqueries/core/square.py), [piece vocabulary](https://github.com/JSeytre/chessqueries/blob/f5aaaf38212cc7839b03ccf383455053905b3f7e/chessqueries/core/pieces.py).

The fixed orientation policy is **as-decoded input, learned canonical output queries**. No truth-assisted best rotation, reflection, class permutation or per-image alignment was attempted. Existing semantic orientation is retained as source metadata and for diagnostics, not used to choose favorable outputs. Source-grid overlays are diagnostic only and were not fed to CQ. The adapter never accepts FEN/labels as an inference argument. Full-FEN history fields are not predicted; only piece placement is serialized.

B.2 predictions are the **existing 3,520 D.2 observations**, reused without classifier inference. Every source frame/session/square is matched exactly; labels are fixed for both predictors. B.2 uses reviewed geometry, cubic board rectification, tighter RGB+target-mask 224 crops and 64 local decisions. CQ sees the entire image, resized once, and internally associates full-image evidence with canonical square queries. Their source-detail allocation, capacity, pretraining, geometry dependence and orientation mechanisms differ. This is not an equal-data, equal-capacity or representation-only ablation.

## Hardware smoke gate and computational cost

The measured runner is an **Intel Core i5-1135G7 @2.40 GHz**, eight logical CPUs, with approximately **7.66 GiB RAM visible to this environment** (the user reports 16 GB physical system RAM). Iris Xe and CUDA are unused. Actual available memory, not the larger physical-RAM declaration, controls the smoke gate.

The first D.2 manifest frame was selected before viewing its prediction, without labels. On that one representative image:

| Smoke measurement | Result |
|---|---:|
| Model load, official imports/construct/load/CPU cast | 2.464 s |
| First forward | 1.202 s |
| Two repeated forwards | 1.003 / 1.072 s |
| Peak process RSS | **629 MiB** |
| Output dimensions | `[1,64,13]` |
| Repeated tensor/logit equality | Exact |

The prespecified practicality checks were at most 60 seconds per forward and RSS below the smaller of 4 GiB or half initial available RAM. Both passed comfortably; over 5.5 GiB remained available afterwards. These are resource bounds, not accuracy thresholds or validated production limits. `model_load_seconds` excludes the initial Python/torch startup and full file-hash verification; total smoke wall time including its three forwards was 11.20 seconds. The stored smoke record contains the exact memory budget and readings.

Only after that success did the 55-frame run begin. Median forward latency was **1.022 s**, p95 **1.140 s**; median decoding/preprocessing **0.061 s**; median forward plus preprocessing **1.084 s**. The entire sequential run, including loading and bookkeeping, took **63.66 s**, with peak RSS **622 MiB**. Model loading in that process took 0.867 s; cache/import state affects loading comparisons. No warm-up outputs were discarded from the benchmark latency distribution.

For context, the saved D.2 B.2 run recorded median classifier forward **0.336 s/board**, extraction/cache-writing **0.986 s**, normalization **0.223 s**. These are historical four-thread timings, not a new synchronized speed race; extraction includes persistent packed-input cache overhead absent from CQ. CQ forward is roughly three times the recorded B.2 forward cost, but end-to-end performance cannot be inferred from forward alone. No new B.2 process-memory measurement was made.

ViT-S FP32 parameter storage alone is about 126 MB; total measured process memory includes Python/runtime, activations and temporary image storage. Its 63.15-MB FP16 release is about ten times the B.2 checkpoint file size (6.25 MB). No dedicated GPU is required for this measured offline workload. Approximately one frame/second makes optional offline/server-assisted inference technically practical, but not a demonstrated browser or continuous phone-camera solution. GPU/iPhone latency and export numerics were not tested. A server would need a bounded worker/queue and account for roughly this resident memory per process, image upload latency and licensing. The measured speed does not justify integration despite worse recognition.

## Paired results by physical session

| Session | Frames | B2 overall | CQ overall | B2 occupied | CQ occupied | B2/CQ empty FP | B2/CQ exact frames |
|---|---:|---:|---:|---:|---:|---:|---:|
| Starting | 12 | 84.51% | 82.29% | 72.66% | 74.22% | 14 / 37 | 0 / 0 |
| Opening | 12 | 80.86% | 84.51% | 69.53% | 80.21% | 30 / 43 | 0 / 0 |
| Middlegame | 12 | 79.82% | 75.13% | 65.86% | 64.52% | 28 / 59 | 0 / 0 |
| Endgame | 10 | 90.16% | 85.31% | 58.18% | 40.00% | 17 / 28 | 0 / 0 |
| Puzzle | 9 | 95.31% | 83.16% | 74.60% | 14.29% | 11 / 43 | 2 / 0 |

CQ improves overall accuracy in one of five sessions and occupied accuracy in two. Equal-session macro occupied accuracy is **68.17% B2 / 54.65% CQ**, versus pooled 68.70%/67.48%; the pooled figure weights dense positions more heavily. Equal-session macro overall accuracy is 86.13%/82.08%. Both summaries are descriptive, not independent frame-level significance tests.

CQ corrects **338** B.2 errors but introduces **464**, a net **126 fewer correct squares**. There are 2,545 squares correct under both predictors and 173 wrong under both. Net changes per session: starting −17, opening +28, middlegame −36, endgame −31, puzzle −70. Individual square/frame observations and all confusion matrices are in the versioned results.

Reviewed camera-side stratification, overall accuracy:

| Orientation | Frames | B2 | CQ |
|---|---:|---:|---:|
| White-near | 20 | 91.41% | 84.22% |
| Black-near | 13 | 88.70% | 69.95% |
| White-left | 5 | 85.62% | 96.25% |
| White-right | 17 | 76.01% | 84.10% |

These strata contain correlated frames and different position mixes. They do not establish an independent camera-side effect or justify choosing an orientation policy using benchmark labels. No precise camera elevation or occlusion labels were invented.

## Per-class behavior and failure diagnosis

| Class | Support | B2 precision / recall | CQ precision / recall |
|---|---:|---:|---:|
| Empty | 2,207 | 98.60 / 95.47% | 87.97 / 90.48% |
| White pawn | 334 | 84.87 / 77.25% | 76.58 / 76.35% |
| White knight | 81 | 82.98 / 48.15% | 88.24 / 74.07% |
| White bishop | 82 | 58.73 / 45.12% | 54.69 / 42.68% |
| White rook | 72 | 38.59 / 98.61% | 75.68 / 77.78% |
| White queen | 36 | 56.00 / 38.89% | 61.54 / 66.67% |
| White king | 55 | 63.41 / 47.27% | 60.47 / 47.27% |
| Black pawn | 327 | 84.82 / 78.59% | 76.04 / 66.97% |
| Black knight | 82 | 69.70 / 56.10% | 81.33 / 74.39% |
| Black bishop | 72 | 46.97 / 43.06% | 42.50 / 47.22% |
| Black rook | 81 | 44.12 / 92.59% | 71.74 / 81.48% |
| Black queen | 36 | 75.00 / 50.00% | 47.37 / 50.00% |
| Black king | 55 | 33.33 / 54.55% | 57.14 / 58.18% |

White king→rook falls **10→3**, black king→rook **12→0**. Black rook→king falls **2→0**. Nevertheless white-king recall remains 47.27%, and both rook recalls decline. Fewer identity confusions do not compensate for missed/relocated pieces and false occupancy. The same piece is counted repeatedly across views; supports are not independent physical pieces.

The compact gallery contains nine bounded source previews, canonical GT/B2/CQ boards, corrected/introduced/persistent errors, optional reviewed source-grid overlays and reused C.1 crop strips with genuine CQ probability distributions. Gallery examples are selected after inference for illustration only. All 55 frames remain in scoring.

| Registered C.1 diagnostic | B2 | CQ | Interpretation |
|---|---|---|---|
| Puzzle b3 king, 42.007 s, crown truncated by B2 crop | White rook | **Empty** | Full-image access does not rescue the specific truncation example. |
| Opening f3 knight, 63.010 s, head visible | White rook | **White knight** | Corrected identity in a crowded board; no proof that cropping caused the old error. |
| Middlegame f8 black rook, 63.010 s, B2 crown clipped | Black king | **Black rook** | A correction compatible with wider visual evidence, but training/capacity are confounded. |
| Middlegame b1 empty, 39.005 s | White rook | **Empty** | Corrected neighbor-related candidate false positive. |
| Same physical b1 empty, 12.002 s | Empty | **White rook** | CQ creates the same failure in another view. |
| Endgame h4 pawn, 0 s, flagged base-boundary ambiguity | Empty | **Empty** | Neither predictor resolves it; contact/square assignment remains qualified. |

Sparse puzzle outputs are particularly poor: at 0 seconds, CQ gets **0/7 occupied squares** correct and predicts multiple white kings at other coordinates; B.2 is exact on that frame. At 42 seconds, CQ predicts two white kings on c7/d7 while the registered white king is b3. This is evidence of bad canonical occupancy/identity assignment, not a simple demonstrated class-index swap. Starting/opening outputs and explicit vocabulary tests rule out a universal adapter class/order reversal.

**Stage attribution limits:** all source grids/orientations passed the prior human review and every inference call completed (**55/55; zero exceptions or missing outputs**). This is not 55 correct CQ geometries. CQ exposes no corners, square polygons, orientation estimate or geometry rejection state, so a separate internal localization/orientation success rate cannot honestly be computed. Sparse/oblique examples show misplaced pieces and inconsistent identities; internal board localization, orientation or domain mismatch are plausible, inseparable explanations. The source grid permits review but cannot prove the network used that grid. No new external geometry or crop-generation defect was confirmed. No post-hoc rotation, source-board crop or preprocessing sweep was run to improve scores.

Full-image downsampling also allocates only part of 644×644 to the physical board, unlike B.2's explicit board enlargement. Native view, background, board area and piece scale can therefore matter even without individual crop truncation. Occlusion remains unknown unless visually evident; a full-board model cannot see through a neighboring piece. The evidence supports some useful identity recognition and some neighbor-error corrections, but **does not establish a reliable solution to truncation, spillover or sparse-board assignment**.

## Recommended next direction

**Path C: invest next in the existing MobileNetV3 RGB+target-indicator architecture, retain the unchanged B.2 control, and do not promote this zero-shot ViT-S checkpoint.** Its CPU resources are practical, so resource failure is not the reason for this decision. Larger models, finetuning and preprocessing/orientation searches were not attempted. This small development benchmark does not rule out full-board reasoning generally.

The single recommended model-improvement experiment is a **matched two-arm MobileNetV3 training comparison: frozen-design B.2 tighter crop versus C.1's taller native perspective-aware RGB+footprint input**, using the same 224 input, initialization, on-board examples, grouped splits and training budget. Validate the loss of crown truncation without increasing empty false positives or sacrificing difficult-piece recall. Keep representation as the changed factor; do not simultaneously add isolated pretraining, a second occupancy network, fusion or a larger backbone. Existing FEN/grid annotations suffice, with C.1's contact caveat retained; no new capture campaign is required. A future task must authorize and preregister that comparison before training. None was started here.

Research remains separate from assisted-MVP delivery: proceed with the existing manual Finish Scan, editable board and explicit confirmation plan when authorized. Neither architecture is an unattended exact-board recognizer on this evidence. Perfect calibration and another model experiment are not prerequisites for user-corrected analysis.

## Reproduction and storage

Essential installation uses a fresh ignored optional environment and a read-only reference to the existing CPU stack:

```bash
.venv/bin/python -m venv outputs/evaluation/chessqueries_c2_env
outputs/evaluation/chessqueries_c2_env/bin/python - <<'PY'
import site
from pathlib import Path
Path(site.getsitepackages()[0], 'chess_vision_cpu_runtime.pth').write_text(
    str(Path('.venv/lib/python3.12/site-packages').resolve()) + '\n')
PY
outputs/evaluation/chessqueries_c2_env/bin/python -m pip install --no-deps --no-cache-dir -r requirements-chessqueries-c2.txt
.venv/bin/python scripts/prepare_chessqueries_c2.py --root outputs/evaluation/chessqueries_c2_release_v1
```

The registered release already exists locally; do not overwrite/re-download it. The preparation script verifies the full small checkpoint and refuses to overwrite release provenance. It downloads no default ViT-L. Inference is offline after preparation. Use fresh output names for deterministic replay (timings/RSS naturally vary):

```bash
outputs/evaluation/chessqueries_c2_env/bin/python scripts/benchmark_chessqueries_c2.py smoke --output outputs/evaluation/chessqueries_c2_smoke_replay.json
outputs/evaluation/chessqueries_c2_env/bin/python scripts/benchmark_chessqueries_c2.py benchmark --smoke outputs/evaluation/chessqueries_c2_smoke_replay.json --output outputs/evaluation/chessqueries_c2_replay
.venv/bin/python scripts/report_chessqueries_c2.py --predictions outputs/evaluation/chessqueries_c2_replay/predictions.json.gz --results docs/evaluation/chessqueries_c2_replay_results.json --examples docs/evaluation/chessqueries_c2_replay_examples
outputs/evaluation/chessqueries_c2_env/bin/python -m pytest -q tests/test_chessqueries_offline.py
.venv/bin/python scripts/validate_chessqueries_c2.py --output outputs/evaluation/chessqueries_c2_validation_replay.json
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
```

Actual commands, environment inventory and log digests are included in the validation artifact. The release directory is about **61 MiB**, the optional environment **41 MiB** because it reuses CPU torch, and the compressed full predictions **897 KiB**. The public visual packet is about **1.2 MiB**, with nine reduced JPEGs and inline SVG boards. Images/masks already present in C.1 are linked, not copied. No video was decoded, transcoded, modified or duplicated in C.2; no full extracted-frame dataset was generated. All weights and the optional environment are under Git-ignored `outputs/evaluation/`.

## Tests, preservation and acceptance

**16 focused tests**, **321 relevant Python regressions total** (including those 16), and **39 browser tests passed**. Tests cover canonical/class ordering, invalid/nonfinite logits, stable full softmax, deterministic batch-one inference mode, no source mutation, CPU thread bounds, no overwrite, configuration boundaries, position/session grouping, versioned c7 labels, and exact preprocessing parity with the official transform. Existing known live-Stockfish integration stalls are outside this crop/model-only scope; the relevant recognition/chess/browser regressions were run instead.

Saved predictions replay exactly into the reported aggregate and per-session metrics. Protected-file checks pass for **4,924 historical/C.1 artifacts**, the frozen B.2 checkpoint, D.2 observations/reviews and all **10 D.3c original videos**. The full ViT-S SHA and all 12 upstream source hashes also pass. Existing C.1/D.1–D.3c reports, labels and source artifacts remain unchanged; neither production model/runtime nor browser code was edited. Original application timm/safetensors absence and CPU-runtime versions remain unchanged outside the optional environment.

| Acceptance item | Status |
|---|---|
| Correct released ViT-S, full hash, minimal isolated CPU inference | Passed |
| Load/one-image timing, RAM and output gate before full benchmark | Passed |
| All 55 matched frames, same labels, old B.2 predictions reused | Passed |
| Canonical mapping, fixed orientation policy, no truth-assisted output choice | Passed |
| Aggregate/session/class/error/latency/memory results and visuals | Complete |
| Fine geometry/annotation/correlated-session limitations explicit | Complete |
| No training, larger model, production/holdout/video/checkpoint changes | Preserved |
| Tests and historical preservation | Passed |
| Architecture recommendation | Path C; retain MobileNetV3, one controlled crop-training comparison recommended only |
| No commits/pushes; stop after C.2 | Preserved |

New files: optional requirements, `configs/experiments/chessqueries_c2.json`, `src/vision/chessqueries_offline.py`, preparation/benchmark/report/validation scripts, focused tests, this report, versioned results/CSV/sensitivity/validation, compact diagnostic packet and C.2 provenance index. No experiment beyond C.2 was begun.
