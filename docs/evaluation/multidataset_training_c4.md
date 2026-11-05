# C.4 — Multi-dataset training infrastructure and bounded Modal smoke test

Date: 2026-10-26. Scope: infrastructure/data preparation only. **No full training or model-improvement experiment was run.** Production recognition, geometry, recovery defaults, assisted MVP and browser behavior remain unchanged.

## Outcome

The isolated Modal **NVIDIA L4** smoke test passed: four optimizer updates, finite loss, nonzero gradients, validation, checkpoint reload with identical logits, and checkpoint visibility/hash verification from a separate CPU container. It used two already approved original photographs (64 training + 64 validation squares), not unverified video labels. Full training remains disabled by `full_training_authorization: false`.

The infrastructure is ready for staged execution, subject to **full dataset preparation**, **ChessReD access/geometry conversion**, and **full reviewed-data export**. It is not correct to call all three datasets training-ready. No training accuracy improvement is claimed.

Machine-readable details: `outputs/evaluation/multidataset_c4/readiness_v2.json`, `c4-smoke-v1_result.json`, `integrity_v1.json`, and the versioned catalog/index under `data/metadata/`.

## Existing implementation and fixed model contract

Reviewed actual root `VISION_RESEARCH.md` and `IMPLEMENTATION_PLAN.md` (the requested `docs/` aliases do not exist), `docs/MVP_ASSISTED_IMPLEMENTATION_PLAN.md`, C.1/C.2/C.3 and D.3c reports, existing crop/training/evaluation code, manifests, geometry and D.1 selection utilities.

- Backbone: torchvision **MobileNetV3 Small**, 1,531,325 parameters; existing `src/vision/classifier.py` is reused unchanged.
- Four-channel stem: copy ImageNet RGB weights; initialize the fourth/mask channel to **zero**. Replace the final classifier with 13 outputs. Existing mode handling keeps frozen-block batch-normalization in evaluation mode during head+stem training.
- Class IDs are unchanged: `0 empty; 1 P; 2 N; 3 B; 4 R; 5 Q; 6 K; 7 p; 8 n; 9 b; 10 r; 11 q; 12 k`.
- B.2 profile: `context1_224`; board warp 800×800 with 200-pixel context, initial square padding 200 pixels, then one target-square width of context at the final crop stage. Typical original context crop is approximately 500 pixels, final RGB region approximately 300 pixels, with fitted-grid variation. Model input is **4×224×224**.
- RGB uses cubic board warping and **AREA** final resizing, ImageNet mean/std. The binary ground-footprint mask uses **NEAREST**, remains 0/1, and is not normalized or photometrically augmented. This is a target indicator, not a piece-silhouette mask.
- Historical personal observations reuse their frozen fitted grids/crops; no board detector is rerun. All 128 smoke-bundle packed RGB/mask inputs match their frozen B.2 cache hashes. Normalized validation input matches locally and on L4: `fa7f390254da5bb3fa9f82c8868b2c2d7068b2ee5af2d29c7e4efec5c0ee05a9`.
- Unchanged B.2 checkpoint: `outputs/evaluation/crop_b2_20261018_run1/training/context1_224/best.pt`, SHA-256 `dad8eea28482ee894dfca0794d9f4415d21ac58db23c6d6bed8c4307c71b1bd8`. Frozen A.1 and B.3/C.3 checkpoints remain untouched.
- C.1 taller-native representations remain separate; none is used by C.4.

## Dataset compatibility and verified counts

| Source | Verified/downloaded in C.4 | Readiness and exclusions |
|---|---:|---|
| Chesscog | 4 actual image/JSON pairs: 2 train, 2 validation; 256 square labels | Official FEN, piece list, camera-side flag and corners cross-checked; sample CRC and SHA recorded. Full dataset not installed. |
| ChessReD | **0 images; 0 usable boards** | Official code/config inspected; official 4TU metadata and annotation requests timed out. Archive size, actual corner coverage, usable count and actual split tables remain unverified. |
| Original personal photos | 112 train/7,168 squares; 25 validation/1,600 squares | Existing owner-approved board geometry/labels, unchanged original groups. Per-crop visibility not universally reviewed. |
| Historical personal test | 23 photos/1,472 square records | Metadata retained for overlap protection; never loaded by training/validation Dataset. |
| Ten retired D.3c recordings | 87 selected frames/5,568 candidate square records | **87 owner-reviewed frames**; 77 frames/4,928 squares eligible for training, 10 frames/640 squares quarantined for protected-position overlap. |

Twenty of the original 180 prepared photos are outside the approved 160-photo classification manifest and are not added. Original training positions: 12; validation: 3 (`003,007,011`); test: 3 (`006,010,016`). Capture-session metadata is absent for all 160 historical photos: known position grouping is preserved, but unrecorded capture-session independence cannot be proven.

### Chesscog

Official [repository](https://github.com/georg-wolflein/chesscog), [OSF dataset](https://osf.io/xf3ka/), [dataset license record](https://api.osf.io/v2/licenses/563c1cf88c5e4a3877f9e96a/). Code MIT; dataset **CC BY 4.0**, independently checked from the OSF license relation. Dataset attribution: Georg Wölflein and Ognjen Arandjelović, *Dataset of Rendered Chess Game State Images*. Derived visual checks include resizing, crop extraction and target outlines; preserve this attribution on redistribution.

Actual remote ZIP central directories contain **4,400 train, 146 validation and 342 test matched image/JSON pairs**, 4,888 total. A naive `.png` count overstates test images by 16 macOS sidecars; these are excluded. Archive sizes:

| File | Bytes |
|---|---:|
| train.z01 | 4,294,967,296 |
| train.zip | 2,416,209,370 |
| val.zip | 222,952,872 |
| test.zip | 522,129,163 |
| Total | 7,456,258,701 |

Central-directory extracted-size sum is 7,458,805,984 bytes including ancillary members. Keeping all archives plus extracted data temporarily needs about 14.9 GB decimal. Future Stage 1 prepares **train+validation only**, approximately 6.93 GB compressed and 6.94 GB extracted; the external test archive is not downloaded. Check free space before preparation. No full archive hash is claimed verified from a range probe. Downloaded sample members are CRC-checked; complete-download preparation additionally computes archive SHA-256 and checks published MD5 when available.

Sample JSON supplies placement-only FEN, a complete `pieces` table, four raw-image corners, camera/lighting metadata, and `white_turn`. The latter is the **published camera-player convention**, not a FEN side-to-move field. Official crop code maps true to White-near, false to Black-near; our conversion reproduces this without inventing game state. Source piece lists and FEN labels must agree exactly.

External corners define a regular board grid, whereas historical B.2 used fitted grids. Therefore external preprocessing is explicitly versioned **`b2-annotated-corners-regular-grid-v1`**: same crop/mask/resize/normalization operators and model input, but **not a claim of identical geometry estimation or pixel parity**. Camera perspective and tall-piece truncation remain limitations of the fixed B.2 representation.

Official published splits are retained as source roles. Before training, matching placements/source hashes across train and validation are quarantined from training, and known game/session relationships propagate exclusions. Chesscog's sampled annotations do not supply game IDs; unknown underlying game relationships cannot be reconstructed. No random square/frame split is used. Class distributions are recorded in `data/metadata/c4_class_distributions_v1.json`. Compact actual-source checks: [four-board visual audit](multidataset_c4_examples/index.html).

### ChessReD

Official [implementation](https://github.com/tmasouris/end-to-end-chess-recognition), [dataset configuration](https://github.com/tmasouris/end-to-end-chess-recognition/blob/main/cfg/chessred.yaml), [4TU release](https://data.4tu.nl/datasets/99b5c721-280b-450b-b058-b2900b69a90f). Code MIT; official config identifies data as **CC BY-NC-SA 4.0**, DOI `10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2`. Do not confuse the code license with the dataset license; production redistribution/commercial compatibility is unresolved here.

The official documentation describes 10,800 photographs from 100 games. This is a **documented claim, not a locally verified usable count**. Official browser and Dataset code use `annotations.pieces`, image/category IDs, category names, image `path`, `splits[role].image_ids/n_samples`, and `chessboard_position`; names are converted explicitly to the existing 13 classes, never by assuming numerical category IDs match ours. No model-generated labels.

The adapter requires downloaded images, explicit verified geometry/orientation, the official image-role table, and verified game IDs/game-level split mapping. It checks agreement between game and image roles. The official Dataset code does not demonstrate where game IDs/corner annotations are stored; those remain actual-annotation verification requirements. Missing information produces quarantine, not guessed corners, a made-up game ID, or an inferred orientation. A normalized geometry record must provide `status`, `kind: raw_regular_grid`, `corners`, `turns`, `orientation`, and the explicit approximation version above. **This boundary adapter is tested with fixtures, not validated on the inaccessible actual annotation file.** Its game-ID/geometry conversion must be checked against that file before Stage 2; no claim that all photographs have usable corners or that the current boundary converter needs no adjustment.

## D.3c policy change and video preparation

New immutable supplement: `data/metadata/d3c_data_use_c4_v1.json`. It records the user's explicit authorization, original source SHA-256s, original FEN associations, the original report/registry hashes, and the retired independent-evaluation status. It does **not** rewrite historical D.3c reports, restrictions recorded at evaluation time, predictions or source manifests.

All ten original video hashes match. Existing filenames/position IDs/owner declaration that the position remained unchanged are reused. Reference photographs are still absent. The owner explicitly confirmed the complete 87-frame review packet: “I have verified everything and it all looks good.” The seven requested checks are recorded in `data/metadata/c4_video_reviews_v1.json`, including video/FEN agreement, semantic orientation and sufficient crop alignment. This is attributed human verification, not a claim of pixel-perfect corners or newly available reference photographs; predictions were never used to establish ground truth. The original five A.3 sessions are excluded by source allowlist and never enter the new video manifest.

The existing 240 candidate metadata records and source PTS are reused. Existing D.1 quality/diversity selection is requested at target 15, with no classifier/FEN input and no changed geometry policy. Existing duplicate/temporal guards yield fewer than 15; the target is not forced by accepting redundant or unresolved views:

| Position | Candidates | Selected for review | Status |
|---|---:|---:|---|
| 01 | 26 | 10 | owner verified / train |
| 03 | 23 | 10 | **quarantine: same placement as protected validation position_003** |
| 05 | 24 | 9 | owner verified / train |
| 07 | 24 | 8 | owner verified / train |
| 09 | 22 | 7 | owner verified / train |
| 11 | 23 | 10 | owner verified / train |
| 13 | 21 | 7 | owner verified / train; selected views lack a predicted Black-near view |
| 15 | 24 | 7 | owner verified / train |
| 17 | 26 | 9 | owner verified / train |
| 19 | 27 | 10 | owner verified / train |

All other selected sessions contain all four camera-side categories; semantic orientation was included in the owner review. Elevation remains a perspective proxy, not a fabricated angle. Native frames are decoded only at original PTS, display rotation checked, and pixel hashes matched. No classifier is run. Every selected frame has a grid preview and one 64-target crop/mask mosaic; combined previews are 22.4 MB, not a full-resolution extracted corpus.

Review packet: `outputs/evaluation/multidataset_c4/video_review_v1/index.html`. Current owner-verification status is recorded separately in `data/metadata/d3c_training_status_c4_v2.json`. Its original pending status and `review_template.json` are preserved. The explicit owner response was imported as the attributed `data/metadata/c4_video_reviews_v1.json`; the resulting `video_manifest_verified_v1.json` includes 77 training frames and quarantines position 03. The seven explicit checks cover FEN agreement, unchanged arrangement, semantic orientation, corners, a1–h8 mapping, crop alignment and all 64 crops. Mark ambiguous/severely occluded targets for exclusion. Neighbor interference alone need not invalidate a clearly labeled hard negative. Pending decisions remain excluded. Bindings cover FEN, video hash, exact PTS, decoded pixels and geometry hash. Position 03 remains quarantined even if visually correct; no frozen split is reassigned.

The following import command reproduces the already-created verified manifest; use a fresh output name to rerun:

```bash
.venv/bin/python scripts/prepare_c4_video_review.py \
  --packet outputs/evaluation/multidataset_c4/video_review_v1/packet.json \
  --reviews data/metadata/c4_video_reviews_v1.json \
  --output outputs/evaluation/multidataset_c4/video_manifest_verified_v1.json
```

## Unified data interface and training controls

`src/training/multidataset.py` provides canonical placement conversion, all-four-orientation square indexing, inclusion validation, connected position/game/session/source grouping, protected-placement quarantine and a lazy PyTorch-compatible Dataset. Observations retain source dataset, source/position/session/game identities, original filename/hash, FEN/label provenance, orientation, geometry status, canonical square/class, representation and split. A compact derived context keeps `original_source` separately from its own pixel-file hash. Masks stay aligned with RGB. Training/test roles never silently change.

`src/training/staged.py` adds deterministic seeds, stateless per-square augmentation, board-grouped batches, Adam, StepLR, mixed precision, gradient accumulation, validation-loss early stopping and bounded wall time/update counts. Resume checkpoints store model, optimizer, scheduler, scaler, RNG, epoch/batch cursor, class ordering, source/config fingerprints and parent-checkpoint hash. Resume `last.pt`; use the designated best checkpoint for transfer. Only last+best checkpoints are retained per stage, avoiding epoch-checkpoint accumulation. Resume rejects mismatched configuration/data; smoke checkpoints are rejected for real transfer.

Training augmentation is held fixed and only photometric: brightness/contrast ±0.10, mild blur probability .15, noise probability .20/std .01. No unverified geometric/semantic transformation is introduced.

| Stage | Initialization | Limits (upper bounds, not promises of completion) | LR | Best artifact |
|---|---|---|---:|---|
| Chesscog | ImageNet RGB; zero mask channel | 8 epochs / 35,200 updates / 6,600 seconds; batch64 | .001 | chesscog_best.pt |
| ChessReD | Chesscog best; ImageNet only for explicit C control | 5 epochs / 12,000 updates / 6,600 seconds; batch64 | .0001 | chessred_best.pt |
| Personal | designated ChessReD best (or Chesscog for B) | 6 validation intervals / 672 updates / 1,800 seconds; batch64 | .00003 | chessvision_finetuned_best.pt |

Stages 1 and 3 initially train head+stem for one epoch/interval; later unfreeze fully. Seed42 is the initial bounded budget. Additional seeds require a separate compute decision. Minimum board-count guards prevent treating the four-board external probe as complete pretraining. If the complete grouped conversion falls below those minima after justified exclusions, review the resulting corpus explicitly rather than silently weakening a guard.

Experiments A–E are registered in `configs/training/multidataset_c4_v1.json`; none is automatically launched. A remains the frozen historical model. D/E share the **same actual ChessReD parent SHA**, architecture, seed, photo-derived class weights, augmentation, optimizer/update and validation budgets; E adds a position-balanced 50% video-source sampling mixture. The runner applies this configuration explicitly (`--experiment D/E`). E fails if no verified non-overlap video exists; it cannot silently become photo-only. Record and compare parent SHA in output checkpoints before interpreting D/E.

The common evaluation entry point `scripts/evaluate_multidataset_c4.py` uses validation only, unchanged square labels and existing metrics. It records per-class confusion/precision/recall, occupied accuracy, empty false positives, king→rook confusions, complete-board denominator, per-position/session/orientation results, high-confidence errors, logits, CPU cost and memory. Incomplete boards from ambiguous-target exclusions do not count as exact boards. Scores on repurposed videos are development/training diagnostics, never independent generalization.

## Modal authentication, storage and measured smoke test

Followed the repository [Modal skill](../../.agents/skills/modal/SKILL.md), installed SDK 1.6.2 and current official [images](https://modal.com/docs/guide/images), [Volumes](https://modal.com/docs/guide/volumes), [billing](https://modal.com/docs/guide/billing) and [budget](https://modal.com/docs/guide/budgets) documentation.

- Authenticated workspace: `akwasipoku-oforiagyei`; active environment `main`. Credentials were neither printed nor copied into code/images/logs. No billing setting was modified.
- Remote-only Python3.12, torch2.8.0+cu128/torchvision0.23.0, NumPy2.2.6, OpenCV4.12, Pillow11.3, python-chess1.999. Local CPU torch/application dependencies were not replaced; no local CUDA installation.
- One L4, CPU2, 8 GiB memory, one maximum container, zero retries, smoke function timeout600s/startup600s, scaledown2s. Training loop additionally capped at four updates/180s. Script timeouts are not account-wide spending caps.
- Volumes: `chess-vision-c4-data-v1` and `chess-vision-c4-artifacts-v1`. Files closed then committed; a separate CPU container confirmed checkpoint hash and size.
- Upload: compact **16,215,923-byte** bundle containing two approved prepared contexts, valid masks, provenance and cached ImageNet initialization. No originals/videos were uploaded. An obsolete 76 MB C.4 staging-copy bundle was removed after checking every member against its own inventory; original source files were untouched.

Run: [Modal app ap-OIcFCQgRnzQhGAZdyfJnRe](https://modal.com/apps/akwasipoku-oforiagyei/main/ap-OIcFCQgRnzQhGAZdyfJnRe).

| Measurement | Result |
|---|---:|
| GPU | NVIDIA L4 |
| Updates / training / validation observations | 4 / 64 / 64 |
| Finite validation loss (smoke only) | 2.5037413173 |
| Training kernel elapsed, including commits/validation | 13.831 s |
| Forward/backward/update compute | 2.115 s; ~30.3 examples/s at batch16 |
| First validation decode/crop/normalization | .0715 s |
| Checkpoint write+commit total | 9.789 s |
| Entire GPU function, including imports/checks | 23.255 s |
| Upload + scheduling/cold start + remote/persistence wall time | 197.497 s |
| Image build steps observed in CLI | 101.36 + 8.69 s |
| Peak torch GPU allocation | 190,185,984 bytes (~181 MiB) |
| Reload / separate-container persistence | exact logits / matching SHA |

GPU allocation/billing duration and pure container-initialization time are not independently exposed by these measurements; upload/scheduling/cold-start time must not be mislabeled pure initialization. Peak allocation is torch tensor memory, not total driver/reserved memory. Full-sized external-image decoding throughput remains unmeasured.

Persisted checkpoint SHA `8cb5d9d2e120497773e858ba82dee99a8f329633a1b16974eaf3b104dcae4b53`, 18,615,693 bytes. Best smoke artifact is named `smoke_resume.pt`, never `chesscog_best.pt`. Full-stage preparation/export/transfer guards were extended and locally tested after this GPU smoke; no subsequent full-stage GPU job ran.

## Budget and recommended initial campaign

Read-only pricing snapshot: L4 **$0.80/hour**, CPU **$0.0473/core-hour**, memory **$0.008/GiB-hour**, Volume storage **$0.09/GiB-month**. Requested CPU2+8GiB+L4 reference total is **$0.9586/hour**, excluding storage/egress/CPU preparation. See [Modal pricing](https://modal.com/pricing) and saved `modal_rates.json`.

Before-run metered cost was zero. After-run/final API snapshot reports **$0.01039083 metered, $0 billed**, $0.01 credits applied and $0.00039083 egress allowance. Its detailed ephemeral-app line is $0.01376501; including its egress line gives $0.01415584, which does not reconcile exactly with the aggregate at this snapshot. Both raw snapshots are preserved; do not present a precise reconciled invoice. Either value is far below $3. Account remaining balance and a hard workspace spend cap were **not exposed/verified**. Approximate remaining credits are **$29.99 based only on the owner's $30 starting estimate**, not a verified account balance. Storage may continue to accrue.

At the tiny batch16 smoke's compute-only throughput, 8 full Chesscog epochs would take roughly **20.7 GPU hours (~$19.8 at the reference resource rate)**, before data loading/validation. ChessReD cannot be costed accurately without actual usable counts; applying the documentation's 10,800 total images as an illustrative upper bound for five epochs would be ~31.7 hours/~$30.4. These are weak extrapolations, not measured training estimates. Larger batches may improve throughput; full-resolution decode may worsen it. **The complete A–E multi-seed matrix is not credibly guaranteed within $30.**

Prioritize one shared, bounded Chesscog stage, one shared ChessReD adaptation after compatibility verification, then matched D/E personal branches. Current wall-time caps imply about **$1.76 + $1.76 + 2×$0.48 ≈ $4.48** in requested GPU-function resources if the loop reaches each cap, plus initialization, final validation/checkpoint time, CPU preparation and storage. Full function timeouts bound each GPU job at 2h (~$1.92) even if the loop overruns its soft wall-time check. These bounded runs may stop well before nominal epoch limits; inspect completion before deciding whether to authorize resumption. Do not call a partial run a completed 8-epoch experiment. Defer B/C and additional seeds until shared-stage throughput/validation justify them. No campaign was launched here.

## Reproduction and exact next Stage 1 procedure

Already executed:

```bash
.venv/bin/python scripts/data/probe_chesscog_c4.py --output outputs/evaluation/multidataset_c4/chesscog_probe --samples 2
.venv/bin/python scripts/prepare_multidataset_c4.py retire-d3c --output data/metadata/d3c_data_use_c4_v1.json
.venv/bin/python scripts/prepare_multidataset_c4.py personal --output outputs/evaluation/multidataset_c4/personal_manifest_v1.json
.venv/bin/python scripts/prepare_multidataset_c4.py chesscog --source outputs/evaluation/multidataset_c4/chesscog_probe --output outputs/evaluation/multidataset_c4/chesscog_manifest_probe_v1.json
.venv/bin/python scripts/prepare_multidataset_c4.py smoke-bundle --source outputs/evaluation/multidataset_c4/personal_manifest_v1.json --output outputs/evaluation/multidataset_c4/smoke_bundle_v2
.venv/bin/python scripts/prepare_c4_video_review.py --output outputs/evaluation/multidataset_c4/video_review_v1 --target 15
.venv/bin/python -m modal run scripts/modal_multidataset_c4.py --action smoke
.venv/bin/python scripts/check_multidataset_c4_integrity.py
```

Outputs are immutable/fresh-path only; do not rerun these exact output paths to overwrite them. For an explicitly authorized future smoke rerun, use a new `--run-id` (and expect a small additional charge).

**Next milestone requires separate authorization.** First prepare the full official training/validation dataset directly in the Modal data Volume, avoiding multi-GB local copies (this preparer is implemented but not exercised against full archives in C.4):

```bash
.venv/bin/python -m modal run scripts/modal_multidataset_c4.py \
  --action prepare-chesscog --authorize-full-download \
  --bundle chesscog-v1 --run-id chesscog-prepare-v1
```

Review its grouped manifest, exclusions, actual counts and costs. Create a separately versioned authorized plan from `configs/training/multidataset_c4_v1.json`, changing `full_training_authorization` to true **only after approval**. Then the exact Stage 1 launch is:

```bash
.venv/bin/python -m modal run scripts/modal_multidataset_c4.py \
  --action train --experiment E --stage chesscog \
  --plan-path configs/training/multidataset_stage1_authorized_v1.json \
  --bundle chesscog-v1 --run-id chesscog-seed42-v1
```

The authorized file deliberately does not exist yet. The current plan fails closed. No full download or Stage 1 job has been started. Check actual workspace budget/remaining credits in the Modal dashboard before that authorization; C.4 did not change billing settings.

Export a checkpoint without replacing production:

```bash
.venv/bin/python -m modal volume get chess-vision-c4-artifacts-v1 \
  /chesscog-seed42-v1/chesscog_best.pt \
  outputs/evaluation/multidataset_c4/export/chesscog_best.pt
```

To resume, use the same authorized config/dataset/run ID with `--resume /artifacts/chesscog-seed42-v1/last.pt`; output checkpoints survive app termination. Stage 2 uses an explicit `--initial /artifacts/.../chesscog_best.pt`; personal D/E use the same explicit ChessReD best path/SHA in separate run IDs. Stage C alone may start ChessReD from ImageNet. No automatic chaining or fallback is implemented.

With the owner review now recorded, `scripts/export_multidataset_c4.py` combines approved personal train/validation photos with verified non-overlap frames. `--require-video` prevents E from running photo-only. It writes compact prepared contexts, valid masks and provenance, verifies all packed inputs against frozen/reviewed hashes, and copies **no original videos**. Do not run it on pending reviews or upload the raw recording directory. Stage 2 remains blocked until the actual ChessReD annotation schema, geometry subset and official game splits are verified.

## Tests, preservation and acceptance

**247 Python regressions and 39 browser tests passed** (zero failures). Final test logs and counts are in `outputs/evaluation/multidataset_c4/python_regressions_final_v5.txt` and `browser_regressions.txt`. Focused coverage includes class/square mapping in all four orientations, actual Chesscog samples, frozen packed/normalized parity, position/game/session/source leakage, the protected overlap, unverified/pilot rejection, explicit approximation identity, review provenance and ambiguity flags, deterministic sampler/RNG, full-training authorization, parent-stage/D–E contracts, ZIP CRC and validation metrics. Browser regressions run unchanged. A bounded local CPU interruption test also confirms that resuming after the update-2 atomic checkpoint reproduces all four-update model tensors and scheduler state exactly; its temporary checkpoints are removed. Training repeatability across different CUDA/library versions is not certified.

Protected integrity: **5,006 files checked, zero mismatches** against preserved historical hash maps, plus all ten source-video hashes. This covers historical evaluation artifacts/model checkpoints and source observations; no protected final-holdout path was opened. No original five-pilot video or D.3c recording was modified. Dataset-dependent tests skip explicitly when private/remote artifacts are absent on a clean checkout; they ran against the real artifacts here. No tracked historical/production file was changed; C.4 files are additions. No commit/push.

- [x] Fixed MobileNetV3/13-class/B.2-compatible input contract; external geometry approximation explicitly distinguished.
- [x] Actual bounded Chesscog download/inventory and sample verification; licenses/source receipts documented.
- [x] ChessReD boundary converter and fail-closed quarantine; **actual download/geometry compatibility pending**.
- [x] D.3c policy retirement supplement; hashes and historical results preserved.
- [x] Bounded diverse video selection, exact PTS/pixel traceability, compact review packet and supervised-data gate.
- [x] Explicit owner verification of all 87 frames; **77 frames / 4,928 square observations eligible**, 10 overlapping frames remain quarantined. No video training performed.
- [x] Grouped splits/protected-position quarantine; no original pilot/final holdout used.
- [x] Stage configuration, transfer/resume/checkpoints, D/E controls and common validation evaluator.
- [x] Authenticated isolated L4 smoke, backward/validation/reload and separate-container persistence.
- [x] Cost snapshots, bounds and honest extrapolation limits; billing cap/balance remain unverified.
- [x] Regression/preservation checks; no full training or production promotion.

Stop after C.4. Next action is approval of bounded full Chesscog preparation/Stage 1, while resolving ChessReD access and exporting only approved data. Future independent evaluation must use genuinely new recordings after model selection; no extra recordings are required to finish this infrastructure milestone.


## Files added

- `src/training/`: unified Dataset/geometry contracts, manifest/export helpers, video review gate, staged trainer and future full-archive preparer (six Python files including package initializer).
- `scripts/prepare_multidataset_c4.py`, `prepare_c4_video_review.py`, `export_multidataset_c4.py`, `evaluate_multidataset_c4.py`, `modal_multidataset_c4.py`, `check_multidataset_c4_integrity.py`, and `scripts/data/probe_chesscog_c4.py`.
- `configs/training/multidataset_c4_v1.json`, `requirements-training-modal-c4.txt`, `tests/test_multidataset_c4.py`.
- `data/metadata/d3c_data_use_c4_v1.json`, `d3c_training_status_c4_v2.json`, `c4_video_reviews_v1.json`, `c4_class_distributions_v1.json`, `multidataset_c4_catalog_v1.json`, and `multidataset_c4_index_v1.json`.
- This report and `docs/evaluation/multidataset_c4_examples/` (four compact, attributed Chesscog source/crop examples).
- Ignored local `outputs/evaluation/multidataset_c4/`: provenance receipts, actual sample pairs, normalized/pending/verified manifests, compact smoke bundle, review packet, billing snapshots, smoke results and test logs. No original recordings or large checkpoints are added to Git. Complete artifact paths/hashes are recorded by the index.

No existing tracked file was modified. The assisted-MVP implementation remains separate.
