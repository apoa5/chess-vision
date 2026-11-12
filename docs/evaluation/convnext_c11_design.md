# C.11 registered CPU optimization design

Frozen model: C.9 selected ConvNeXt V2 Tiny update1578, SHA-256
`45b471edb751261e9da1ef8703f3d9c940cdaec69f5713f6847e48bf0dc26acb`.
Original13 classes, four channels, B.2 context1_224, original RGB normalization and
binary mask. No weight training, geometry changes, lower precision, quantization,
confidence calibration, production integration, paid cloud or holdout access.

First reproduce C.10's192 saved raw logits/top1 outputs with the unchanged CPU
FP32/batch16/threads2 reference. Stop on unexplained differences. Then record the
same reference over **all41 C.9 validation board views /2624 squares /4 positions**.
Order saved camera-relative observations by their existing canonical coordinates;
never rotate RGB pixels. No retrospective internal-test images are opened.

Screening uses the first lexicographic board of each development placement,
three rotating timing rounds, one warmup, no concurrent CPU-heavy jobs. Fixed
matrix: NCHW batch16/T2 reference and batch8/T2; channels-last batches1/8/16/32
atT2 plus batch8 atT1/T4. ONNX CPU: batches8/T2,8/T4,16/T2, sequential execution,
bounded threads, standard ALL graph optimization and disabled spinning. No broad
search or arbitrary accuracy threshold is introduced. Same-frame input values
must remain identical; a preallocated preparation buffer may replace list+stack
only if the original normalization kernel produces byte-identical values.

Export one official FP32 graph (dynamic batch; fixed4×224×224;13 logits), preferably
with the installed PyTorch's recommended dynamo exporter/opset18. Preserve all
logical model weights; no custom architecture/operator approximation. If export
fails, preserve the exact diagnostic and allow only a bounded compatible exporter
fix, not architecture substitution. Export/session dependencies are isolated.

Freeze comparison tolerance **after development screening and before full-corpus
or candidate C.10 checks**: atol=max(1e-6,2×maximum screened absolute FP32 error of
the chosen finalists), rtol=1e-5. This is an observed numerical-comparison envelope,
not confidence calibration or a universal error guarantee. Full-corpus failures
cannot be fixed by silently widening it. Any top1/FEN disagreement disqualifies a
candidate regardless of numerical tolerance or unchanged aggregate accuracy.

Select the fastest zero-class-change candidate per runtime family for full
development parity. Choose a primary backend using only development parity,
speed and resource results. Finally replay C.10 as a fixed regression veto; no
settings/thresholds or alternate recognition model are selected from C.10.

Final paired timing: five warm rounds, rotating identical development boards,
separate processes per backend; report initialization/import costs, median,
variance, RSS and artifact size. Profile aggregated CPU operators rather than
persisting a large trace. Complete-image timings reuse the unchanged C.10 raw
pipeline, projective transforms, masks and FEN serializer; geometry limitations
remain explicit. First-call timings do not claim flushed OS caches.

Preserve a reference backend and make selection explicit. Missing optional
dependencies/artifacts must fail clearly, with no silent checkpoint fallback.
The pretrained weights' CC BY-NC4.0 restriction also applies to exported artifacts.
Report measured results and stop after C.11.
