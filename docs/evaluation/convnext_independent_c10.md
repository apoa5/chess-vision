# C.10 — Independent evaluation on three fresh photographs

Evaluation-only, 2026-11-10. Two frozen checkpoints; no training, model/crop/geometry/threshold tuning, cloud jobs, production replacement, final-holdout access, commits or pushes. The C.10 manifest and configuration were frozen before classifier inference.

## References and independence

The owner supplied each full FEN and its numbered photo association in the C.10 request. All FENs are structurally/chess-valid (32,30,15 pieces respectively). Their piece placements and visible printed a1–h8 orientation were visually cross-checked against the photographs by the assistant; no classifier established a reference label. This is owner-provided human reference provenance, **not a separately recorded owner attestation of every crop or corner**.

All three images are1920×2560, with no EXIF orientation tag. Precise capture time, camera, room, elevation, lighting type and session identity are unavailable. Indoor patterned bedding and piece shadows are observable; nighttime/artificial lighting is not inferred as fact. All are White-near by printed coordinates, with a1 at the bottom-left playing corner.

No new canonical piece placement or source hash matches C.9. The audit checks all363 C.9 board records, all180 original-photo inventory entries (including the quarantined photograph), and exact placement keys ignoring turn/move counters. Raw-photo64-bit dHash nearest distances are21,17,16; those proxies do not prove independence. Manual source/FEN inspection and the canonical-placement/source-hash audit support three new positions. The same physical set and similar capture conditions limit generalization; these are three position units, not192 independent trials.

| Position | File | SHA-256 |
|---|---|---|
| c10_position_01 | `convNext_01.jpg` | `f6a83236fcc4926e3a4a6d22bf7278150348367e80d23e09ba72b458b75f4479` |
| c10_position_02 | `convNext_02.jpg` | `b841e917fd81c5b8f13580910417afe4b94847b61c67b7493c6e4f547bea707f` |
| c10_position_03 | `convNext_03.jpg` | `3367b330fa17548e7eff7e9d9403bb8a8473a2e66ca25692bb82cc4a6f940635` |

Full owner FENs, EXIF/source provenance, nearest-image candidates, audits, cached-input hashes and geometry are in `data/metadata/convnext_independent_c10_manifest_v1.json`. These photos remain evaluation-only and were not added to C.9.

## Frozen protocol

| Model | Checkpoint | SHA-256 |
|---|---|---|
| b2 | `outputs/evaluation/crop_b2_20261018_run1/training/context1_224/best.pt` | `dad8eea28482ee894dfca0794d9f4415d21ac58db23c6d6bed8c4307c71b1bd8` |
| convnext | `outputs/evaluation/convnext_c9/export_v1/convnextv2_tiny_best.pt` | `45b471edb751261e9da1ef8703f3d9c940cdaec69f5713f6847e48bf0dc26acb` |

ConvNeXt selection remains C.9 update1,578; its plan hash/class order/input contract is verified. B.2 is the tighter-crop reference, not C.5/C.7 or the production scanner profile. Both use RGB+binary target mask, the same13 classes, B.2 `context1_224`, board800/context margin200, initial context padding200, AREA RGB and NEAREST mask resizing, original ImageNet normalization, no augmentation/calibration, canonical output a8–h1, camera-relative input pixels, CPU FP32, batch16, two CPU threads/interop1. Weights are strict-loaded, eval mode, gradients disabled, and `torch.inference_mode()` is used.

Primary policy is unchanged **legacy/default `prepare_raw_photo`** with the frozen `config/settings.yaml`. B.1b recovery is not silently enabled. FENs and expected orientation never enter automatic localization, marking-based orientation or crop extraction. `scripts/run_square_evidence.py:make_inputs` supplies the existing B.2 packing path to both models. Their192 individual RGB/mask hashes match exactly. No rotation is selected for better recognition.

The manifest, configuration and execution-source hashes are bound in `outputs/evaluation/convnext_c10_v1/execution_contract.json`; models reject changed execution inputs. Identical-input repeat inferences are timing-only and must reproduce all logits exactly; their predictions do not replace the first pass.

## Geometry and orientation

Automatic pipeline completion: **3/3** for both models, with marking-based White-near orientation3/3, consistent with visible coordinates. The assistant inspected all three full-board overlays and identified no grossly incorrect completed polygon or semantic-grid mapping. **Fine geometry remains owner-unverified.** Completion is not a localization-correctness certificate. Detected corners, source polygons, orientation evidence, rectified context and candidate diagnostics remain under each position’s `automatic/` directory.

**Verified-geometry secondary evaluation: unavailable (0/3)**. No independent human corner annotations or explicit owner geometry review were supplied. No manual corners were invented or silently substituted. The report includes `owner_review_template.json` if the owner later wants to confirm the registered geometry in a versioned supplement. This limits causal attribution of residual errors to classification alone.

## Recognition and correction effort

Exact boards mean all64 piece-placement labels match the owner reference. FEN syntax or legality alone never counts as exact. The full automatic-pipeline denominator remains three; failures would have no fabricated square predictions or correction count.

| Position/model | Overall | Occupied | Empty accuracy | Empty FP | Missing pieces | Wrong occupied identities | Corrections | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| c10_position_01/b2 | 79.69% | 62.50% | 96.88% | 1 | 0 | 12 | 13 | False |
| c10_position_01/convnext | 98.44% | 96.88% | 100.00% | 0 | 0 | 1 | 1 | False |
| c10_position_02/b2 | 85.94% | 73.33% | 97.06% | 1 | 0 | 8 | 9 | False |
| c10_position_02/convnext | 98.44% | 96.67% | 100.00% | 0 | 0 | 1 | 1 | False |
| c10_position_03/b2 | 95.31% | 86.67% | 97.96% | 1 | 0 | 2 | 3 | False |
| c10_position_03/convnext | 98.44% | 100.00% | 97.96% | 1 | 0 | 0 | 1 | False |

| Aggregate/model | Overall | Occupied | Empty accuracy | Empty FP/rate | Exact / all attempts | Mean corrections |
|---|---:|---:|---:|---:|---:|---:|
| b2 | 86.98% | 71.43% | 97.39% | 3/115 (2.61%) | 0/3 | 8.33 |
| convnext | 98.44% | 97.40% | 99.13% | 1/115 (0.87%) | 0/3 | 1.00 |

Paired outcomes: ConvNeXt corrects **25** B.2 errors, introduces **3** errors on previously correct squares, and both remain wrong on **0** squares. Each position requires one ConvNeXt correction in this run; neither model reconstructs any of the three boards exactly.

Color/type error counts can overlap and must not be added to obtain total corrections. Missing pieces are occupied→empty; extra pieces are empty→occupied. All counts,13×13 confusion matrices, per-class precision/recall/F1 and source observations are machine-readable. [Per-class CSV](convnext_independent_c10_classes.csv). Full per-square differences/confidences are in `outputs/evaluation/convnext_c10_v1/errors.csv`.

## Error types and inspected cases

| Position/model | Occupied errors | Missing | Extra | Wrong color | Wrong type | Total square corrections |
|---|---:|---:|---:|---:|---:|---:|
| c10_position_01/b2 | 12 | 0 | 1 | 0 | 12 | 13 |
| c10_position_01/convnext | 1 | 0 | 0 | 0 | 1 | 1 |
| c10_position_02/b2 | 8 | 0 | 1 | 1 | 8 | 9 |
| c10_position_02/convnext | 1 | 0 | 0 | 0 | 1 | 1 |
| c10_position_03/b2 | 2 | 0 | 1 | 0 | 2 | 3 |
| c10_position_03/convnext | 0 | 0 | 1 | 0 | 0 | 1 |

ConvNeXt residual errors (all three are newly introduced relative to B.2):

| Position/square | Owner reference | ConvNeXt prediction | Raw max probability | B.2 |
|---|---|---|---:|---|
| 01 / b8 | black knight | black bishop | 70.03% | correct |
| 02 / f7 | black pawn | black bishop | 93.72% | correct |
| 03 / e8 | empty | black bishop | 87.04% | correct |

The actual b8 crop contains overlapping back-rank and b7-bishop imagery; distinguishing knight features approach the upper crop boundary. The f7 pawn is crowded by tall f6-queen imagery. The e8 target footprint contains the e7 king’s upper silhouette although the king’s base is on e7. These are directly visible overlap/truncation risks, not proof that a particular pixel region caused the neural prediction. No gross corner-order, semantic orientation or crop-index defect was identified. Without independently reviewed fine corners, subtle alignment error cannot be ruled out.

All three ConvNeXt errors predict **black bishop**. Black-knight recall regresses from B.2’s3/3 to2/3; black-bishop precision is5/8 (62.5%) despite5/5 recall. Black-pawn recall is19/20 versus B.2’s16/20. Other ConvNeXt classes are correct on this small set, including the six kings and ten rooks across both colors; B.2 also got all six kings correct here. This sample does not establish general class guarantees.

Both pipelines used exactly the same automatically produced geometry and crops, so recognition differences between models do not result from giving one better geometry. Residual recognition/crop ambiguity is more evident than gross localization failure, but a verified-geometry rescue comparison is unavailable. No oracle switching, ensemble, ground-truth piece-count repair, crop change or second checkpoint was tried.

## Confidence

| Model | Mean raw probability, correct | Mean raw probability, wrong | Errors above90% | Empty FP above90% |
|---|---:|---:|---:|---:|
| b2 | 96.61% | 73.19% | 8 | 0 |
| convnext | 99.52% | 83.60% | 1 | 0 |

Confidence is uncalibrated; >.9 is a diagnostic only. No abstention or user-confirmation threshold was fitted. Every predicted board still requires human review. Raw13-class logits and full softmax distributions remain in each model’s result artifact.

## Runtime and memory

The timed wrapper transparently calls the unchanged image loader, detector, rectifier, orientation resolver and grid fitter. Replayed corners/orientation and every packed crop/mask must match registration. The shared D.2 packing helper performs an additional reference-context warp; that cost is included in crop extraction, not hidden as classifier speed. Source scoring/FEN legality diagnostics are included in observed end-to-end timing.

| Position/model | Load image | Detection | First rectification | Orientation + grid | Packing + normalization | First forward | Warm forward median | FEN | End-to-end after model load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c10_position_01/b2 | 0.079s | 1.223s | 0.043s | 0.253s | 0.388s | 0.438s | 0.265s | 0.00015s | 2.458s |
| c10_position_02/b2 | 0.052s | 1.540s | 0.030s | 0.201s | 0.304s | 0.329s | 0.276s | 0.00015s | 2.482s |
| c10_position_03/b2 | 0.055s | 1.377s | 0.032s | 0.166s | 0.346s | 0.253s | 0.280s | 0.00010s | 2.257s |
| c10_position_01/convnext | 0.070s | 0.893s | 0.032s | 0.180s | 0.378s | 7.198s | 6.976s | 0.00016s | 8.771s |
| c10_position_02/convnext | 0.044s | 1.299s | 0.029s | 0.196s | 0.295s | 6.841s | 6.738s | 0.00012s | 8.728s |
| c10_position_03/convnext | 0.055s | 1.236s | 0.029s | 0.217s | 0.298s | 6.653s | 7.023s | 0.00009s | 8.507s |

b2: model load/strict validation **0.097s**; first model+photo path **2.555s**; process peak RSS **663MB**.

convnext: model load/strict validation **1.328s**; first model+photo path **10.099s**; process peak RSS **1164MB**.

Hardware: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz; PyTorch2.14.1+cpu, OpenCV5.0.0. Models ran in separate processes. Two warmup board passes and five forward-only measurements per image were made; image order rotates between rounds. “Cold” means the first call after Python/library imports; filesystem/OS caches were not flushed. Warm forward timings exclude geometry, loading and extraction. No optimization was applied. C.10’s ConvNeXt warm medians of6.74–7.02s/board are broadly consistent with C.9’s approximately8s/board under the same CPU-thread/batch convention; timing variation is not evidence of an optimization. Model-loaded first-call image processing is8.51–8.77s versus2.26–2.48s for B.2. Cloud usage: **$0**.

## Visual report and reproducibility

[Originals, automatic grids, rectified contexts, reference/predicted boards, complete error lists and actual224px RGB/mask error crops](../../outputs/evaluation/convnext_c10_v1/report/index.html). Predictions from both models use the same source pixels and geometry. Red outlines identify target footprints, not the piece silhouette. Distortion, truncation, shadow and neighboring interference remain hypotheses unless directly supported by the pictured crop; no coordinate-mapping defect is asserted from a misclassification alone.

```bash
# Registration is already frozen; existing outputs refuse overwrites.
.venv/bin/python scripts/evaluate_convnext_c10.py register
# Each model runs in its own CPU process; no remote resources:
outputs/evaluation/chessqueries_c2_env/bin/python scripts/evaluate_convnext_c10.py infer --model b2
outputs/evaluation/chessqueries_c2_env/bin/python scripts/evaluate_convnext_c10.py infer --model convnext
outputs/evaluation/chessqueries_c2_env/bin/python scripts/report_convnext_c10.py
.venv/bin/python scripts/evaluate_convnext_c10.py check
```

The optional C.2 environment already contains the pinned timm1.0.22 implementation and CPU PyTorch; application dependencies were not upgraded. Initial geometry diagnostics were produced with the existing `scripts.evaluate_pilot_geometry.evaluate_frame(path, load_config(), destination, "legacy")`; repeat runs time the original pipeline without changing behavior. To reproduce without replacing this evaluation, use a separately versioned output/configuration copy and retain the frozen weights/settings; do not retune on these outcomes.

## Decision and limitations

**B — partial generalization.** ConvNeXt substantially reduces correction effort versus B.2 on these new placements, while still producing one error per board and0/3 exact reconstructions. It is promising for a reviewed draft, not an unattended final position. Three same-side photographs of one chess set cannot establish broad robustness across camera sides, lighting, boards or devices. Architecture/training/data differences from B.2 prevent causal architecture-only conclusions. No conclusion of safe confidence calibration follows from three positions.

Proceed toward the assisted MVP with mandatory confirmation/correction and an explicit, separately authorized recognition profile. A bounded inference-optimization/parity milestone is appropriate given CPU cost; no immediate retraining is justified by this tiny sample. Preserve B.2 fallback and address ConvNeXt’s noncommercial pretrained-weight restriction before any commercial deployment. No integration, optimization, data repurposing or further experiment is started here.

## Files, validation and acceptance

Commands: `python -m pytest tests/test_independent_c10.py -q`, the relevant C.9/C.4–C.7/crop/geometry/keyframe/evidence/inference/FEN suites (full275-test command in the release metadata), and `npm --prefix web test`. `git diff --check` and Python compilation pass. Integrity hashes cover historical models, media, reports, split/configuration data, source code, current C.9 artifacts and the frozen C.10 contract. The three original photographs are unchanged.

New code: `src/vision/independent_c10.py`, `scripts/evaluate_convnext_c10.py`, `scripts/report_convnext_c10.py`, `tests/test_independent_c10.py`. Versioned C.10 config/manifest/results metadata, this report and class CSV are new. All predictions/caches/visualizations live under ignored `outputs/evaluation/convnext_c10_v1/`. No existing production or historical source was edited.

- [x] Three original photographs registered with hashes, reference FENs, explicit orientation provenance and independence audit.
- [x] Fixed checkpoints, settings, source contract and identical192 square inputs frozen before inference.
- [x] All three default-geometry attempts retained; no truth geometry/orientation substitution or tuning.
- [x] Per-position/aggregate precision, recall, F1, exact boards, corrections, confidence, full logits/probabilities and diagnostics preserved.
- [x] Cold/first-call and repeated warm CPU costs measured separately; no cloud resources.
- [x] No fabricated verified-corner secondary result; optional owner review packet provided.
- [x] 12 focused tests;275 relevant Python regressions;39 browser tests across8 test processes;8,032 protected-file hashes unchanged. All three geometry replays match the registered source polygons exactly.
- [x] Stop after C.10; no retraining, optimization, production replacement, final holdout, commit or push.
