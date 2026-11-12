# C.11 — ConvNeXt V2 Tiny CPU inference optimization

Measured 2026-11-11; tests and report completed 2026-11-12. **Recommend the explicit
PyTorch FP32 channels-last backend, batch 8, four intra-op threads, one inter-op
thread.** Repeated median forward latency fell from **6.341 to 5.323 seconds per
board (16.0%)**; warm complete-image median fell from **8.395 to 7.344 seconds
(12.5%)**. The aspirational **under-three-second forward target was not achieved**.

Both finalists preserved all **2,816 top-1 predictions and 44 board placements**:
2,624 development squares plus 192 frozen C.10 regression squares. This is measured
prediction parity on the registered corpus, not a guarantee for every future
near-tie input. There is no recognition-accuracy improvement: C.10 still has one
error per board, 189/192 correct squares and 0/3 exact boards. Editing every board
and explicit user confirmation remain necessary.

No production/backend default, browser, geometry, orientation, crop, threshold,
checkpoint or historical result was changed. No training, quantization, model
download, cloud job or final-holdout access occurred.

## Frozen reference and representation

The source is `outputs/evaluation/convnext_c9/export_v1/convnextv2_tiny_best.pt`,
SHA-256 `45b471edb751261e9da1ef8703f3d9c940cdaec69f5713f6847e48bf0dc26acb`.
It is C.9 selected update 1,578, `convnextv2_tiny.fcmae_ft_in22k_in1k`, with
27,878,029 parameters, a four-channel stem and 13-class head. Loading verifies the
checkpoint bytes and architecture/input/class/profile metadata before strict
state loading. Models remain FP32, evaluation-only and without parameter gradients.

The class indices remain: 0 empty; 1–6 white pawn, knight, bishop, rook, queen,
king; 7–12 black pawn, knight, bishop, rook, queen, king. Output is canonical
**a8, b8, …, h1**. Camera-relative crop RGB is not rotated when observations are
reordered by square name.

Preprocessing is unchanged B.2 `context1_224` / `b2-context1-224-v1`: existing
geometry/context extraction, AREA RGB resizing, NEAREST binary-mask resizing,
224×224 inputs, RGB divided by 255 then normalized by mean
`[.485,.456,.406]` and standard deviation `[.229,.224,.225]`. The fourth channel
is the original **0/1 mask**, without RGB normalization. The new preallocated
input buffer calls the original `tensor_from_packed` kernel. Its normalized
values are byte-identical to list-plus-stack preparation.

Before screening, reference PyTorch reproduced **all saved C.10 logits exactly**,
then all saved C.9 validation logits exactly. Saved C.9 observations were indexed
by canonical square name, not assumed to be in camera-array order. No saved
record was rewritten. Reference logits and normalized-input hashes are preserved
in `reference_c10*` and `reference_development*` under the C.11 output directory.

## Registered corpus, selection and numerical policy

`data/metadata/convnext_c11_corpus_v1.json` registers all 41 C.9 validation board
views, covering four physical positions and 2,624 squares. These are development
data, not an independent recognition benchmark. The first lexicographic board
per placement supplies four screening/timing boards. The 16 smallest saved
top-two logit margins were registered before optimization; the minimum is
**0.0143356323**. All are included in full parity checks and `edge_cases_v1.json`.

The three C.10 photos are fixed reproduction/regression inputs only. They did
not choose thread settings, batch sizes, numerical tolerances, checkpoint,
recognition rules or the primary candidate. No retrospective internal-test input
was used for optimization, and the separately protected final holdout was not
opened.

The design was recorded in `convnext_c11_design.md` and the versioned evaluation
configuration. Screening used three rotating rounds after one warmup. The fastest
zero-class-change candidate in each runtime family advanced to all 41 boards.
Before full parity, the registered rule froze **atol = 0.000026702880859375,
rtol = 0.00001**, based on twice the finalists' maximum observed screening FP32
absolute error. This is an empirical comparison envelope, not calibration or a
universal numerical bound. It was never widened after full or C.10 checks.
**Any changed top-1 class or placement disqualifies a candidate regardless of
numerical tolerance or aggregate accuracy.**

Both finalists passed development parity. `primary_selection.json` froze
PyTorch channels-last before candidate C.10 inference. Both then passed C.10's
regression veto. No alternate candidate was chosen from C.10 outcomes.

## Hardware and measurement conditions

- Intel Core i5-1135G7, eight logical CPUs, Linux/WSL2 x86-64. The host has 16 GB
  RAM; the WSL guest exposes approximately 7.66 GiB. Intel integrated graphics
  were unused. No CUDA execution or installation.
- Python 3.12.3; PyTorch 2.14.1+cpu; torchvision 0.29.1+cpu; timm 1.0.22;
  NumPy 2.5.3; OpenCV 5.0.0; ONNX 1.23.2; ONNX Runtime 1.31.1. Exact environment
  metadata and optional-package versions are in `environment.json`.
- OpenCV one thread; PyTorch inter-op one; tested intra-op counts 1, 2, 4.
  ONNX CPUExecutionProvider only, sequential execution, inter-op one, bounded
  intra-op threads, standard ALL graph optimization, idle spinning disabled.
- CPU-heavy benchmark jobs ran sequentially, in fresh processes per candidate.
  Confirmation uses the same four development boards, five rotating rounds and
  one warmup: 20 forward timings per backend. No unrelated experiment was launched.
  Background host activity and thermal drift cannot be completely excluded;
  small differences between backends should not be overinterpreted.
- Peak memory is process high-water RSS, including imports, model initialization
  and inputs; it is not just live model tensors. Four cached timing inputs make
  confirmation RSS higher than streaming single-image pipeline RSS.

## Screening results

All 11 settings had zero changed predictions on the 256-square screening subset.
Only the two finalists received full-corpus qualification; screening alone does
not qualify the other settings for deployment.

| Backend/layout | Batch | Threads | Median forward/board (s) |
|---|---:|---:|---:|
| PyTorch reference NCHW | 16 | 2 | 6.430 |
| PyTorch channels-last | 1 | 2 | 6.367 |
| PyTorch channels-last | 8 | 2 | 6.431 |
| PyTorch channels-last | 16 | 2 | 6.200 |
| PyTorch channels-last | 32 | 2 | 7.281 |
| PyTorch channels-last | 8 | 1 | 10.003 |
| PyTorch channels-last | 8 | 4 | 5.219 |
| PyTorch NCHW | 8 | 2 | 6.459 |
| ONNX CPU | 8 | 2 | 6.486 |
| ONNX CPU | 8 | 4 | 5.728 |
| ONNX CPU | 16 | 2 | 6.932 |

The reference already used `eval()`, `torch.inference_mode()` and batched
processing of 16 squares; these are preserved baseline behavior, not newly
discovered improvements. Channels-last alone yielded little benefit. Four-thread
execution helped; larger batches were not consistently faster and batch 32
increased screening peak RSS to about 1.38 GiB. No broad search, lower precision,
INT8 calibration, `torch.compile` or custom operator approximation was introduced.

## Final repeated comparison

Forward timings below are the independent five-round confirmation, not screening.
Warm pipeline is the median of one warm full replay for each of the three C.10
images, after a first full pass. Model load excludes backend-library import.

| Runtime | Forward median ± SD (s) | Warm pipeline (s) | Model initialization (s) | Peak RSS (MiB) | Artifact (MiB) | Changed squares: development / C.10 |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch reference, batch16/T2 | 6.341 ± 0.110 | 8.395 | 1.904 | 1,156 | 106.4 | 0 / 0 |
| PyTorch channels-last, batch8/T4 | 5.323 ± 0.099 | 7.344 | 1.948 | 977 | 106.4 | 0 / 0 |
| ONNX CPU, batch8/T4 | 5.434 ± 0.137 | 7.653 | 0.269 | 849 | 107.3 | 0 / 0 |

PyTorch's confirmation RSS fell about 15.4%. ONNX has lower RSS and faster
initialization, but did not beat the selected PyTorch candidate's repeated median;
the forward gap is small relative to timing variability. PyTorch also avoids a
new deployed runtime dependency and exported model requirement. Keep ONNX as an
offline evaluated alternative, not an automatic application fallback.

Backend-library imports were respectively 0.955, 0.942 and 0.025 seconds in the
confirmation processes. In separate complete-pipeline processes, first-image
latency including that process's model initialization and backend import was
11.288, 9.581 and 7.382 seconds. These are process-cold measurements with already
available files, **not flushed-disk-cache cold starts** or total Python startup.
Keep model instances loaded between requests.

Warm full-image ranges were 8.166–8.445 seconds for reference, 7.239–7.493 for
optimized PyTorch and 6.848–7.718 for ONNX. Pipeline peak RSS was respectively
978, 829 and 771 MiB. End-to-end measurements include C.11 audit hashes and
unchanged input preparation, classification and FEN construction, and exclude
model loading. Two passes of three images per backend are a limited pipeline
timing sample; no latency-significance claim is made. The reference remains
consistent in scale with C.10's approximately 8.5–8.8 seconds.

The paired full-pipeline runs share the same byte-identical preallocated input
normalizer for every backend. Original list-plus-stack preparation was used for
reference reproduction and checked against it. Thus the final comparison isolates
backend/thread/batch behavior rather than separately estimating an allocation
speedup; reference neural execution remains the original NCHW/batch16/T2 FP32.

## Pipeline and operator profile

Median warm stage timings across the three images:

| Stage | Reference (s) | Optimized PyTorch (s) | ONNX (s) |
|---|---:|---:|---:|
| Image decode/load | .043 | .047 | .043 |
| Board localization | 1.093 | 1.237 | 1.246 |
| Semantic orientation | .148 | .148 | .143 |
| Perspective transform / rectification | .029 | .031 | .030 |
| Grid fitting | .042 | .044 | .043 |
| Raw preparation total | 1.353 | 1.496 | 1.529 |
| 64 crop/mask extraction | .165 | .163 | .157 |
| Normalization / input-buffer preparation | .120 | .125 | .128 |
| Runtime input validation | .008 | .008 | .007 |
| Tensor layout preparation | <.001 | .007 | <.001 |
| Model forward in full pipeline | 6.635 | 5.421 | 5.675 |
| Prediction / placement decoding | <.001 | <.001 | <.001 |
| Full-FEN construction / parse | <.001 | <.001 | <.001 |

Raw-preparation total contains its preceding component calls; do not sum both.
Perspective correction and board rectification share the existing
`rectify_with_config` operation and are measured together. Medians of separate
stages need not add to median end-to-end time. Geometry timings vary between
replays even though geometry values and algorithms do not change.

The reference CPU operator profile identifies `aten::addmm` (2.983 seconds self
CPU), vector norms (.630), addition (.473), GELU (.454), addcmul (.395) and copying
(.358) among the leading costs; MKLDNN convolution self CPU was .284 seconds.
Dense linear work and normalization/elementwise/memory operations dominate.
This instrumented profile is diagnostic, not a clean latency benchmark; operator
allocation totals are not simultaneous RSS. Full data are in
`operator_profile.json`. Image decoding and FEN serialization are not the main
forward bottleneck, and changing geometry would not address that cost.

## Numerical, coordinate and pipeline parity

| Candidate | Development maximum / mean absolute logit difference | C.10 maximum difference | Changed boards, development / C.10 |
|---|---:|---:|---:|
| PyTorch channels-last | 0.0000123978 / 0.0000007325 | 0.0000076294 | 0 / 0 |
| ONNX CPU | 0.0000152588 / 0.0000017726 | 0.0000123978 | 0 / 0 |

Every full-corpus comparison met the frozen numerical envelope. Per-board maximum
top-two margin differences, all 13 logits, input hashes, changed-square lists,
canonical placements and the registered low-margin examples are preserved in
the machine-readable artifacts. Floating-point logits are not claimed bit-exact
across backends. Top-1 classes and reconstructed placements are exact.

All 18 full-pipeline replays (three backends × three images × two passes)
completed. Source-corner maximum difference was **0.0**; orientation turns were
identical. All packed RGB/mask hashes and normalized-input hashes matched C.10.
Neither geometry thresholds nor the legacy/default policy changed; B.1b recovery
was not enabled. **Fine geometry remains owner-unverified**, as in C.10.

The same C.10 errors remain: b8 black knight → black bishop; f7 black pawn → black
bishop; e8 empty → black bishop. In particular, the original f7 error's 93.72%
raw confidence is not repaired or recalibrated by runtime optimization. No
confidence/abstention threshold or chess-rule repair was added. Full FEN retains
the existing explicitly supplied game-state fields; imagery establishes none of
those fields automatically.

## ONNX export and runtime identity

Export succeeded on the first attempt in 7.750 seconds using the standard
PyTorch dynamo exporter, opset 18, embedded FP32 weights, dynamic batch and fixed
4×224×224 spatial/channel dimensions. Output is batch×13 logits. ONNX full graph
checking passed. No architecture substitution or custom graph rewrite was needed.
Only standard exporter and ONNX Runtime graph optimizations were used.

- Graph: `outputs/evaluation/convnext_c11_v1/convnext_fp32.onnx` (112,552,108 bytes).
- Graph SHA-256: `67213d69230f16ae4ea3857ec26c82d1b93687dff2c6f2356fcb72729d13099d`.
- Metadata SHA-256: `0aa930206a0fb7d924e81018d8ee1bdb3b6e7b96adc511ba8faaecb9c92b12f3`.
- Logical PyTorch weight-value SHA-256 before/after layout conversion and export:
  `9e11d551dc87c4377e69c9294f437e5cba549a48f763bc27baf084b957125304`.

The sidecar and pinned configuration bind graph, source checkpoint, class order,
representation, architecture and FP32 input/output shape. Optional dependencies
were installed only in the isolated C.11 environment using the six pinned packages
in `requirements-inference-c11.txt`; existing CPU torch/timm installations were
reused read-only. No application packages were upgraded. The inherited environment
has an unrelated optional Hugging Face `hf-xet` dependency warning; no Hugging Face
download was used. ORT emitted a sandbox telemetry-ID persistence warning on
import; telemetry is disabled before sessions and CPU inference succeeded.

The pretrained-weight license remains **CC BY-NC 4.0**; export does not expand
commercial-use rights. See the [official model card](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k)
and C.9 provenance. Resolve commercial rights separately before deployment.
CPU results do not establish browser/mobile/WASM performance or compatibility.

Technical references: [PyTorch memory-format guidance](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html),
[PyTorch ONNX export](https://docs.pytorch.org/docs/2.14/onnx.html), and
[ONNX Runtime CPU threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html).

## Runtime boundary and assisted integration

`src/vision/convnext_runtime.py` accepts an explicit backend configuration and
canonical normalized `[64,4,224,224]` inputs, returning all `[64,13]` FP32 logits.
It validates input order, binary masks, finite values and model/artifact identity.
Missing optional libraries, mismatched hashes or invalid outputs raise errors;
there is **no silent checkpoint/backend substitution**. The original reference
backend remains available. See `convnext_c11_runtime.md` for the API example.

`configs/evaluation/convnext_c11_recommended_v1.json` is an offline recommendation,
not a changed application default. Reuse one runtime per bounded worker. A future
integration should add an explicit ConvNeXt profile with this exact B.2 packing
contract, retain the current live production baseline, and reuse the existing
recognition-session/stale-response guards and review dialog. Finish Scan should
open a separately editable draft, permit direct piece corrections, and require
explicit confirmation of every square and full-FEN game state before analysis.
Uncalibrated confidence cannot certify correctness. Do not make automatic
completion or fusion a prerequisite. C.11 implements none of that product work.

## Artifacts, commands and tests

Machine summary/index: `data/metadata/convnext_c11_results_v1.json` and
`outputs/evaluation/convnext_c11_v1/results_v1.json`. The index records artifact
and source hashes. The output directory contains the export/identity, reference
and finalist logits (`.npz`, board order in matching JSON; square order a8–h1),
screening and confirmation samples, geometry/pipeline timings, low-margin cases,
operator profile, environment, tests and preservation results. No new full-size
image collection or video copy was created. Graph, environment and derived arrays
remain outside Git under ignored `outputs/`.

The recorded write-once workflow was:

```bash
C11_PYTHON=outputs/evaluation/convnext_c11_env/bin/python
$C11_PYTHON scripts/benchmark_convnext_c11.py register
$C11_PYTHON scripts/benchmark_convnext_c11.py reference-c10
$C11_PYTHON scripts/benchmark_convnext_c11.py reference-development
$C11_PYTHON scripts/export_convnext_c11.py
for candidate in reference pt_cl_b1_t2 pt_cl_b8_t2 pt_cl_b16_t2 pt_cl_b32_t2 pt_cl_b8_t1 pt_cl_b8_t4 pt_nchw_b8_t2 ort_b8_t2 ort_b8_t4 ort_b16_t2; do
  $C11_PYTHON scripts/benchmark_convnext_c11.py screen --candidate "$candidate"
done
$C11_PYTHON scripts/validate_convnext_c11.py freeze
$C11_PYTHON scripts/benchmark_convnext_c11.py full --candidate pt_cl_b8_t4
$C11_PYTHON scripts/benchmark_convnext_c11.py full --candidate ort_b8_t4
$C11_PYTHON scripts/validate_convnext_c11.py choose
for candidate in reference pt_cl_b8_t4 ort_b8_t4; do
  $C11_PYTHON scripts/validate_convnext_c11.py confirm --candidate "$candidate"
done
for candidate in pt_cl_b8_t4 ort_b8_t4; do
  $C11_PYTHON scripts/validate_convnext_c11.py regression --candidate "$candidate"
done
for candidate in reference pt_cl_b8_t4 ort_b8_t4; do
  $C11_PYTHON scripts/validate_convnext_c11.py pipeline --candidate "$candidate"
done
$C11_PYTHON scripts/validate_convnext_c11.py profile
$C11_PYTHON scripts/validate_convnext_c11.py integrity
# After capturing environment.json and tests_summary_v1.json with the logs below:
$C11_PYTHON scripts/report_convnext_c11.py
```

These commands refuse existing versioned results. Do not delete historical
artifacts to rerun them; a new measurement campaign needs a fresh versioned run
directory/registry/configuration. The registered matrix and recorded samples are
complete in the machine artifacts. To reproduce parity on the **existing run
without writing or replacing artifacts**, run:

```bash
outputs/evaluation/convnext_c11_env/bin/python - <<'PY'
import json
import numpy as np
from scripts.benchmark_convnext_c11 import ROOT, Corpus, prepare_runtime, parity
tol = json.loads((ROOT / 'finalists.json').read_text())
for name in ('reference', 'pt_cl_b8_t4', 'ort_b8_t4'):
    runtime, _ = prepare_runtime(name)
    for kind in ('development', 'c10'):
        corpus = Corpus(kind)
        reference = np.load(ROOT / f'reference_{kind}_logits.npz')['logits']
        for i, board_id in enumerate(corpus.ids):
            x, _ = corpus.load(board_id, preallocate=True)
            check = parity(reference[i], runtime.predict_logits(x),
                           tol['atol'], tol['rtol'])
            assert check['numeric_within_tolerance']
            assert check['top1_disagreements'] == 0
            assert not check['board_changed']
        print(name, kind, len(corpus.ids), 'boards: parity passed')
    del runtime
PY
```

**304 Python tests passed**, including 29 focused C.11 cases and 275 relevant
existing regressions. **39 browser tests passed across eight processes**. No
failures or skips. Python compilation and `git diff --check` passed.

```bash
outputs/evaluation/convnext_c11_env/bin/python -m pytest tests/test_convnext_runtime.py tests/test_independent_c10.py tests/test_convnext_c9.py tests/test_multidataset_c4.py tests/test_chesscog_c5.py tests/test_personal_c7.py tests/test_chessred_c6.py tests/test_crop_experiments.py tests/test_geometry_audit.py tests/test_geometry_reliability.py tests/test_video_keyframes.py tests/test_square_evidence.py tests/test_inference.py tests/test_validation.py tests/test_fen.py -q
npm --prefix web test
```

Focused tests cover four-channel/mask/normalization parity, class/configuration
contracts, canonical ordering, batch transport, optional-backend errors, invalid
logits, hash checks, unchanged weight values under layout conversion, exported
graph shape/dynamic batch, categorical disagreement despite numerical tolerance,
board reconstruction, and write-once replay protection. Actual full-model
export/session execution and parity are additionally evidenced by the measured
artifacts, not substituted with toy-model accuracy results.

**8,105 protected-file hashes matched; zero mismatches.** Original checkpoints,
media, C.9/C.10 reports/results, source inputs and split/configuration artifacts
are unchanged. Final-holdout paths are excluded from access, not opened to hash.
No commits or pushes were made.

New files: the C.11 runtime; four export/benchmark/validation/report scripts;
`tests/test_convnext_runtime.py`; optional requirements; C.11 matrix/candidate/
recommended configurations; corpus/results metadata; and this report plus the
design and runtime notes. Pre-existing uncommitted C.10 files were preserved.

## Acceptance and next step

- [x] Frozen checkpoint verified; saved reference logits reproduced exactly.
- [x] CPU baseline, major stages, operators, initialization and RSS measured.
- [x] Bounded PyTorch matrix and supported ONNX FP32 export/runtime evaluated.
- [x] Numerical tolerance frozen before full validation; all finalists meet it.
- [x] Zero changed classes, canonical mappings, board placements or input pixels
  across the registered development and C.10 regression corpus.
- [x] Explicit runtime/configuration, reference backend, export tools and errors.
- [x] Relevant Python/browser tests and protected-file preservation checks pass.
- [x] Integration path and pretrained-weight licensing restrictions documented.
- [x] Production, historical artifacts, original media and holdout boundaries
  preserved; no training/cloud work or automatic MVP integration.
- Performance goal: **under three seconds not reached**; this was aspirational,
  not permission to trade prediction parity for speed.

Proceed to a separately authorized assisted-MVP task: **explicit ConvNeXt profile
plus manual Finish Scan and an editable, explicitly confirmed board draft**, using
the existing server/session/review components. Use a visible processing state for
the measured CPU wait. The optimized backend is a modest, parity-checked option;
it does not solve the remaining classification errors, geometry-verification
limitations or confidence reliability. Do not begin additional model experiments
or automatic-completion work as part of C.11.
