# D.3b — Fusion reliability audit and independent-session validation preparation

Completed 2026-10-22. **D.3 reproduces exactly. Uncertainty signals are useful diagnostics, but none tested establishes reliable board acceptance. Independent validation is pending.** No automatic D.4 gate was implemented.

## Frozen reproduction and scope

Read the root research/roadmap, D.1–D.3 reports, D.2 evidence/review/label provenance, and actual offline selection, extraction, fusion and validation code. Replayed unchanged D.3 into a fresh directory: all **75 board predictions**, **15 evaluation aggregates**, complete evaluated square records, paired comparisons, minority-error diagnostics and oracle diagnostics match exactly. Timings naturally differ. Also independently checked the portable lossless D.3 export and all 4,340 original protected files before analysis. See [validation record](fusion_reliability_d3b_validation_v1.json) for final checks.

Primary audit uses only uniform-5 single-quality, mean probabilities and mean logits: 15 method/board outputs, 960 method/square diagnostics describing **320 physical squares in five correlated sessions**. All 55 D.2 geometry reviews and 3,520 original observations are preserved. No classifier inference, training, crop regeneration or geometry rerun. Existing owner-confirmed middlegame c7 supplement is used, exactly as in D.3; original labels and results remain untouched. No approved compatible B.4 transformation is available; probabilities remain uncalibrated.

| Uniform-5 | Overall | Occupied | Exact boards | Wrong score ≥.9 | Occupied→empty |
|---|---:|---:|---:|---:|---:|
| Single quality | 304/320 (95.00%) | 97/113 (85.84%) | 1/5 | 3 | 2 |
| Mean probabilities | 309/320 (96.56%) | 102/113 (90.27%) | 0/5 | 0 | 1 |
| Mean logits | 311/320 (97.19%) | 104/113 (92.04%) | 1/5 | 2 | 1 |

Every wrong prediction in these three fused/single outputs concerns a truly occupied square; all 207 empty squares are correct. Exact boards describe tentative argmaxes, not accepted scans.

## Signals and analysis definitions

[Prespecified audit configuration](../../configs/experiments/fusion_reliability_d3b.json) fixes descriptive coverage budgets and illustrative thresholds; there is no threshold fitting or method selection. For each square preserve all 13 probabilities and calculate:

- Maximum probability, top-two margin and predictive entropy in nats.
- Cross-frame variation ratio (1 − largest argmax vote fraction) and Jensen–Shannon distribution disagreement.
- Population standard deviation across frames of probability assigned to the fused class, and separately of input maximum probability. These describe dispersion, not standard errors.
- Minimum input support probability, supporting-view fraction and high-confidence support fraction. The high-confidence-consensus indicator requires at least 4/5 input argmaxes to agree with the output, each at probability ≥.9. **It does not know whether consensus is correct.** Its incorrectness is evaluated afterward with ground truth.

All five cached views are used for cross-frame diagnostics even when evaluating the single-frame prediction. Such augmented diagnostics are not available to a one-frame-only deployment at one-frame cost. They never alter its frame choice or prediction.

Calibration diagnostics use the original D.3 definitions: multiclass NLL (explicit 1e-12 floor), multiclass Brier and ten-bin ECE. Full bin counts are retained, including empty bins. Report empty/occupied, all 13 true classes and each physical session separately; unsupported metrics are null where appropriate. Tiny class bins are descriptive, not class-specific calibration estimates. Correct, confidently incorrect and low-confidence (<.8) groups are retained; these categories can overlap for correct low-confidence predictions.

Risk–coverage curves retain complete signal ties. Matched-budget comparisons retain exactly floor(N×coverage) observations, using session/square identity to break boundary ties, never correctness. Tie counts and minimum/maximum possible tie risks are exposed. Ground-truth occupied/empty filtering is **evaluation stratification**, not an implementable selection policy. Seven signals are displayed without picking a winning threshold from these results.

[Figures, confidence bins and case studies](fusion_reliability_d3b_examples/index.html) · [matched coverage CSV](fusion_reliability_d3b_matched_coverage.csv) · [board diagnostics](fusion_reliability_d3b_boards.csv).

## Reliability and matched coverage

| Method | ECE all / occupied | NLL all / occupied | Brier all / occupied |
|---|---:|---:|---:|
| Single | .0136 / .0428 | .1772 / .4963 | .0760 / .2119 |
| Mean probabilities | .0935 / .1924 | .1884 / .4524 | .0848 / .2183 |
| Mean logits | .0205 / .0589 | .1225 / .3460 | .0514 / .1456 |

Pooled ECE is dominated by the 207 easy empty squares; occupied-only values expose the practical gap. Mean probabilities is often underconfident despite correct predictions. Mean logits improves proper scoring rules here but still makes severe confident mistakes. These descriptive values do not certify calibration.

At **equal confidence-ranked coverage**, errors / retained squares:

| Evaluation stratum / retained count | Single | Mean probabilities | Mean logits |
|---|---:|---:|---:|
| All, 160/320 (50%) | 0/160 | 0/160 | **1/160** |
| All, 288/320 (90%) | 3/288 | 3/288 | 4/288 |
| Occupied, 84/113 (74.34%) | 3/84 | 4/84 | 4/84 |
| Occupied, 101/113 (89.38%) | 9/101 | 7/101 | 6/101 |
| Occupied, 107/113 (94.69%) | 12/107 | 8/107 | 8/107 |

Thus the reliability ordering depends on coverage. A higher raw score cannot be compared across methods as if it represented equal calibrated risk. Cross-view disagreement is not uniformly better: at 101/113 occupied coverage, low-JS rankings retain 13, 8 and 6 errors, respectively; low predicted-class dispersion retains 16, 9 and 7. For mean logits, low-JS ranking retains seven errors at 107/113 versus confidence ranking's eight, a one-square development difference rather than validated superiority. Full per-session curves prevent the pooled counts from hiding session-specific failures.

The [signal-group supplement](fusion_reliability_d3b_signal_groups_v1.json) reports mean/median/range for every signal by true class and correct/incorrect/confidently-incorrect/low-confidence status. For occupied mean-logit predictions, median probability / entropy / JS are **.974 / .141 / .410** when correct, **.829 / .628 / .591** when incorrect, and **.952 / .203 / .461** for the two confidently incorrect cases. Typical errors are more uncertain, but the dangerous tail overlaps correct predictions. Zero-support groups are null, not invented estimates.

All methods have 232 high-confidence-consensus squares, one incorrect: endgame h4. The indicator exposes the possibility of shared certainty, **not a ground-truth-blind error detector**. Most consensus squares are correct. Low dispersion/disagreement may reflect correlated bias rather than reliable independent support.

### Classes and empty/occupied behavior

All three methods have 7/7 correct physical black-rook predictions in this primary comparison; this does not establish rare-class reliability. Remaining class errors (single / mean probabilities / mean logits): white knight 4/3/2 of seven; white bishop 1/3/3 of seven; white queen 1/1/1 of three; black knight 3/1/1 of seven; black bishop 0/2/1 of six; white pawn 4/1/1 of 29. Mean logits's two probability≥.9 mistakes are a white pawn→empty and black bishop→black pawn. Full per-class bins, NLL/Brier/ECE, support, and selective confusion metrics are in the versioned diagnostics; D.3's precision/recall and confusion matrices remain unchanged.

## Square- and board-level withholding

Existing D.3 exploratory withholding is unchanged: score/margin, group support, and conflicting strong labels. Retained square counts and errors:

| Method | Retained all squares | Errors retained | Retained occupied squares | Occupied errors retained |
|---|---:|---:|---:|---:|
| Single | 293/320 | 4 | 87/113 | 4 |
| Mean probabilities | 241/320 | 1 | 46/113 | 1 |
| Mean logits | 266/320 | 4 | 66/113 | 4 |

These different coverages must not be mistaken for matched-risk superiority. Requiring every square to be retained would withhold **all 5/5 boards for every method**, including each method's correct board. Zero retained incorrect boards at zero board coverage is not useful acceptance evidence.

Board-mean confidence is especially misleading. Mean logits's mean probabilities and error counts are:

| Session | Mean probability | Minimum probability | Incorrect squares | Withheld squares |
|---|---:|---:|---:|---:|
| Starting | .9374 | .3364 | 0 | 16 |
| Opening | .9493 | .5363 | 3 | 17 |
| Middlegame | .9456 | .3883 | 3 | 12 |
| Endgame | **.9845** | .6539 | 2 | 5 |
| Puzzle | **.9881** | .7343 | 1 | 4 |

An illustrative mean-probability ≥.95 board probe retains endgame and puzzle: **2/2 wrong**, despite higher average scores than the correct starting board. Minimum-confidence ranking is not sufficient either: its first four retained mean-logit boards are all wrong, and the correct starting board ranks last. Matched board-coverage curves use the same 1–5 retained-board budgets across methods and report actual error counts. No interpolation, fitted safe threshold or automatic completion result is claimed.

## Required case studies

The report's new [case records](fusion_reliability_d3b_examples/cases_v1.json) bind full distributions, signal values and original observation/frame IDs; existing D.3 actual input images remain linked unchanged.

1. **Starting board recovered:** mean logits has zero errors versus two for the quality-selected frame. On [g1](fusion_d3_examples/starting_g1_rescue.png), only one view's argmax is the correct knight. Mean logits recovers it with probability .485; the low-confidence/margin rule withholds it. The board is tentative-exact but has 16 withheld squares. Complementary evidence is real; confidence does not yet justify board acceptance.
2. **Puzzle board broken:** [d7](fusion_d3_examples/puzzle_d7_regression.png) is a correct knight in the selected single frame and two of five views, but three rook predictions win in fusion. Mean-logit probability .734, large inter-view disagreement and strong conflict correctly lead to withholding. The single frame was exact; fused puzzle is not. Its 98.81% board-average confidence hides this.
3. **Endgame pawn confidently missed:** on [h4](fusion_d3_examples/endgame_h4_confident_miss.png), four empty predictions at ≥.9 overwhelm one correct pawn observation at .644. Mean logits gives empty **.999973**, margin **.999956**, entropy **.000331 nats**; the unchanged rule does not withhold it. Variation ratio .2, JS .2574 and selected-class dispersion .2606 reveal disagreement, but do not by themselves establish an error or an independently validated cutoff. High-confidence consensus is 4/5, also misleading. Mean probabilities is wrong at .869 and also retained by the existing combined rule. No new geometry defect or ground-truth change is inferred.

## Do additional frames resolve uncertainty?

The preserved D.3 larger-budget comparison uses diversity-10 versus uniform-5. **None of the five larger sets is a superset of its uniform-5 set.** This changes both views and budget; it cannot isolate adding frames. Mean probabilities rescues four squares and breaks two (net +2); mean logits rescues two and breaks four (net −2). Each retains only 1/5 exact boards in the larger condition. More frames are not a demonstrated completion solution. A future nested-budget protocol must be registered before outcomes and evaluate both rescues and confident regressions, with added cost.

## Independent-session protocol and runnable tooling

[Preregistered protocol](fusion_independent_d3b_protocol_v1.md) · [frozen method configuration](../../configs/experiments/fusion_independent_d3b_v1.json) · [empty registry template](../../configs/capture/fusion_independent_d3b_registry_template_v1.json).

No eligible new cohort is registered. Existing native files/manifests describe the same original pilot and compressed counterparts. **Independent validation is pending; there are no new-session results. The [versioned preregistration](../../data/metadata/fusion_independent_d3b_preregistration_v1.json) freezes executable/configuration hashes; new registration refuses silent changes.**

The protocol reserves at least ten genuinely new positions/sessions over at least two dates, with lighting/setup variation. This is an audit minimum, not proof of safety. Capture a continuous four-side/elevation tour, original video and exact native timestamps, a reference photograph and owner-verified full FEN; record stable placement, board/set/device/environment/light and coarse/unknown pose metadata. Reuse existing annotations where provided; repeated pilot/training placements remain development only. Keep final holdout sealed.

The `register` command checks native timeline/media hashes, human truth/stability provenance, grouping and historical exclusions, then freezes sources, configuration and input identities. Evidence must bind this registration before extraction. The `evaluate` command consumes the existing reviewed D.2 evidence contract and unchanged D.3 code. It rejects stale provenance, wrong model/crop/selection, substituted frames and altered session denominators. Unusable sessions remain in the registered exact-board denominator; no silent replacements. Original source-index codecs/frame rates/resolution/PTS and device metadata remain linked in the lock. Grouped-position results accompany per-session metrics; repeated frames do not create independent trials.

This is a **reviewed-evidence evaluator, not a new raw-video importer**. Existing native ingestion, D.1 selection and D.2 review/extraction prerequisites must produce the documented contracts in fresh directories. The old secondary HD/Telegram runner has pilot-specific paths; that limitation is explicit in the protocol. No inference on new recordings was possible or attempted here.

The explicit `--functional-development` path was tested on existing pilot data, reusing owner attestations and the corrected FENs without requesting duplicate labels. It reproduces all **15 primary board predictions and classification/reliability metrics exactly**, with output `independent_validation: pending` and no claim of novelty. The registry honestly records that pilot outcomes were already inspected and used for development; independent mode refuses such data. Minimum new-cohort requirements are not satisfied by this functional test.

## Commands, artifacts and validation

Fresh directories are mandatory; do not overwrite historical reports/results.

```bash
.venv/bin/python scripts/run_fusion_experiments.py --config configs/experiments/fusion_d3.json --output-dir outputs/scanning/d3b_replay_v1
.venv/bin/python scripts/evaluate_fusion_experiments.py --run outputs/scanning/d3b_replay_v1 --output-dir outputs/scanning/d3b_replay_v1/evaluation_v1
.venv/bin/python scripts/audit_fusion_reliability.py --replay outputs/scanning/d3b_replay_v1 --output-dir outputs/scanning/d3b_audit_v1/analysis_v2
.venv/bin/python scripts/summarize_fusion_reliability.py --analysis outputs/scanning/d3b_audit_v1/analysis_v2/reliability_v1.json --output outputs/scanning/d3b_audit_v1/signal_groups_v1.json
.venv/bin/python scripts/render_fusion_reliability.py --analysis outputs/scanning/d3b_audit_v1/analysis_v2/reliability_v1.json --output-dir outputs/scanning/d3b_audit_v1/figures_v2
.venv/bin/python scripts/prepare_fusion_development_registry.py --output outputs/scanning/d3b_audit_v1/development_registry.json
.venv/bin/python scripts/evaluate_independent_fusion.py register --registry outputs/scanning/d3b_audit_v1/development_registry.json --output outputs/scanning/d3b_audit_v1/development_registration_v4.json --functional-development
.venv/bin/python scripts/evaluate_independent_fusion.py evaluate --registration outputs/scanning/d3b_audit_v1/development_registration_v4.json --evidence-manifest data/metadata/evidence_d2_manifest_v1.json --output-dir outputs/scanning/d3b_audit_v1/functional_evaluation_v3 --functional-development
.venv/bin/python scripts/export_fusion_reliability.py --analysis outputs/scanning/d3b_audit_v1/analysis_v2 --figures outputs/scanning/d3b_audit_v1/figures_v2 --functional outputs/scanning/d3b_audit_v1/functional_evaluation_v3 --output-dir outputs/scanning/d3b_audit_v1/export_v1
.venv/bin/python scripts/validate_fusion_reliability.py --replay outputs/scanning/d3b_replay_v1 --audit outputs/scanning/d3b_audit_v1 --export outputs/scanning/d3b_audit_v1/export_v1 --output outputs/scanning/d3b_audit_v1/final_validation.json
.venv/bin/python -m pytest -q tests/test_fusion_reliability.py tests/test_offline_fusion.py tests/test_square_evidence.py tests/test_video_keyframes.py tests/test_keyframe_manifest.py tests/test_crop_experiments.py tests/test_geometry_audit.py tests/test_geometry_reliability.py tests/test_scan_recognition.py tests/test_inference.py tests/test_validation.py tests/test_fen.py
PATH="$PWD/tools/node/bin:$PATH" npm --prefix web test
```

Results: **269 Python tests passed**, including **35 new D.3b tests**; **39 browser tests passed**. Tests cover uncertainty definitions, ties and risk bounds, deterministic ordering, invalid distributions, board averages/withholding, session/placement grouping, native/source provenance guards, inspected/tuned/holdout exclusions, registration mode/schema, stale source hashes and reused positions. The full unchanged D.3 replay adds 75 reverse-order checks; the functional entry point adds 15 and exact cross-checks against the historical predictions. Compilation, whitespace, artifact hashes and protected-file checks are recorded separately.

Audit time including historical hash checks was 12.81 s; functional evaluation including media checks took 6.66 s, with 1.20 s in the first 15 fusion calls. Excludes earlier D.3 replay and classifier inference (none rerun). The portable result/tables/figures package is approximately 2.05 MB. The uncompressed machine-readable audit is 7,604,893 bytes; the versioned lossless gzip stores all 960 square diagnostics, 63 reliability strata, seven-signal curves, matched-coverage/tie bounds, board curves/probes and larger-budget changes. Figures use dependency-free SVG; no plotting dependency installation is required.

New files: `src/scanning/fusion_reliability.py`, `src/scanning/independent_fusion.py`; `scripts/audit_fusion_reliability.py`, `render_fusion_reliability.py`, `evaluate_independent_fusion.py`, `prepare_fusion_development_registry.py`, `export_fusion_reliability.py`, `validate_fusion_reliability.py`, `summarize_fusion_reliability.py`; `tests/test_fusion_reliability.py`; two experiment configurations and registry template; this report/protocol, versioned diagnostics, tables/figures/validation, `data/metadata/fusion_independent_d3b_preregistration_v1.json` and `data/metadata/fusion_reliability_d3b_index_v1.json`. Historical D.1–D.3 files are unchanged.

## Acceptance and D.4 research decision

- [x] D.3 exactly reproduced before extension; existing labels and geometry reviews reused.
- [x] Three frozen primary methods audited for confidence, margin, entropy, disagreement, dispersion and consensus failure.
- [x] Empty/occupied, classes and sessions separated; reliability bins and matched square/board coverage retained.
- [x] Square and board withholding, three required cases and extra-frame limitations documented.
- [x] Independent protocol, immutable registration, runnable evaluator and functional tests supplied.
- [x] New independent data explicitly pending; no fabricated labels/outcomes or final-holdout access.
- [x] Regression/preservation checks passed; no training, production changes, D.4 implementation, commit or push.

**Known:** complementary evidence improves square recognition, and uncertainty flags some conflicts. Existing signals fail on shared confident bias, board-average confidence hides mistakes, and conservative all-square withholding currently retains no boards.

**Unvalidated:** independent-session accuracy, generalization, probability calibration, useful board-level risk/coverage, a safe abstention rule, and whether nested additional views reliably resolve uncertainty.

**Evidence required before implementing D.4:** collect and account for the preregistered independent cohort; verify all truth/geometry; demonstrate paired exact-board gains consistently across positions without hiding losses; show that a separately frozen proposed rule flags every observed incorrect board and high-confidence occupied error while retaining correct boards; and run a preregistered nested extra-view experiment showing uncertainty resolution without confident regressions. These are human research-review criteria, not implemented scan logic. Zero unsafe accepts in a small cohort would still not prove safety. If thresholds or methods change after examining those outcomes, that cohort becomes development data and a fresh independent cohort is required.

**Recommendation:** prepare/capture the independent cohort and continue offline reliability investigation before D.4 implementation. Do not claim a safe threshold from these five sessions. Frames, repeated crops and physical squares are correlated observations, not hundreds of independent recording sessions; no significance or deployment-safety claim is justified.
