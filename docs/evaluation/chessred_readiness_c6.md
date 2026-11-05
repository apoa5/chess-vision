# C.6 — ChessReD readiness investigation

Date: 2026-10-27. **Stage 2 is blocked: official annotations could not be retrieved. No adaptation training ran.** This investigation stops after bounded local and Modal attempts; actual annotation compatibility and usable-image counts remain unverified.

## Retrieval results

Official sources inspected: [implementation and README](https://github.com/tmasouris/end-to-end-chess-recognition), [download configuration](https://github.com/tmasouris/end-to-end-chess-recognition/blob/main/cfg/chessred.yaml), and the [dataset DOI record](https://doi.org/10.4121/99b5c721-280b-450b-b058-b2900b69a90f.v2). The current official repository files and accessible DataCite record are cached with URLs and hashes in `outputs/evaluation/chessred_c6/official_context_v1/receipts.json`.

Each location attempted dataset metadata GET, file-inventory GET, annotation HEAD, annotation GET, and archive HEAD. Each request had a 12-second socket timeout and at most two transport attempts, with three concurrent workers. Every request timed out without an HTTP response. Both locations resolved the host to `131.180.169.22`; this does not establish whether the cause is service availability, routing, or filtering.

| Location | Requests / maximum attempts each | Elapsed | Annotation bytes | Images |
|---|---:|---:|---:|---:|
| Local execution environment | 5 / 2 | 49.91 s | 0 | 0 |
| Modal EU CPU worker | 5 / 2 | 49.12 s | 0 | 0 |

Logs and full attempts: `outputs/evaluation/chessred_c6/{local_v1,chessred-c6-eu-v1}/retrieval.json`. No image archive GET was issued. Metadata retrieval is bounded to 2 MiB and annotations to 96 MiB, with declared-size and streaming checks. ZIP responses are rejected. Successful annotation retrieval must pass JSON checks and the publisher's MD5 before use; SHA-256 is also recorded. No successful-response integrity claim is possible for this run.

Published URLs:

- Annotations: `https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/3cae6364-daca-4967-b426-1e4b68cdb64c`; configured MD5 `d34bca5ad46ec7a8df96a1d3c36784f3`.
- Images: `https://data.4tu.nl/file/99b5c721-280b-450b-b058-b2900b69a90f/6329e969-616e-48e3-b893-a0379d1c15ba`; configured MD5 `32e23ed535239dc517a6499762d9847e`.
- Metadata/inventory: `https://data.4tu.nl/v2/datasets/99b5c721-280b-450b-b058-b2900b69a90f`, with `/files` for the inventory.

The official README also links preprocessed 1024-pixel images through Google Drive. That is not an annotation-only alternative, and resized-image/corner compatibility has not been established. No unofficial mirror was substituted. A historical maintainer [host-problem comment](https://github.com/tmasouris/end-to-end-chess-recognition/issues/1#issuecomment-1898603739) does not diagnose today's failures.

## Annotation contract versus verified artifacts

The following are **official documentation/source-code claims**, not counts verified against downloaded annotations:

| Scope | Documented images | Grouping |
|---|---:|---|
| Full dataset | 10,800 | 100 games |
| Full train / validation / test | 6,479 / 2,192 / 2,129 | 60 / 20 / 20 games |
| Geometry-covered ChessReD2K | 2,078 | 20 games |
| Geometry train / validation / test | 1,442 / 330 / 306 | 14 / 3 / 3 games |

Thus 1,772 train-plus-validation images are a **potential** geometry-covered pool, not an eligible count. No test images should be requested for Stage 2 preparation. **Verified usable labels: 0; usable geometry: 0; verified semantic orientations: 0; verified game/split groups: 0.** Actual file inventory, byte sizes, game IDs and geometry schema remain unavailable.

Official Dataset code accesses `images` entries with `id` and `path`, `categories` entries with `id` and `name`, `annotations.pieces` entries with `image_id`, `category_id`, and `chessboard_position`, and `splits[role].image_ids` / `n_samples`. Square traversal is a8–h1. Empty squares are derived from unoccupied squares; category IDs must be mapped by category name rather than assumed to match Chess-Vision indices. These source-code accesses do not verify the actual JSON's complete schema or corner/game fields.

The documentation says piece locations are algebraic and derived from game PGN/FENs. Photos are documented as 3024×3024 or 3072×3072. ChessReD2K includes board corners and piece bounding boxes. Corner names are relative to White's viewpoint; actual coordinate order, EXIF handling, corner IDs and their image association must be inspected before conversion. Piece labels must never be used to infer missing geometry.

## Adapter and B.2 compatibility gate

No architecture or preprocessing changes were made. The parent remains `outputs/evaluation/chesscog_c5/checkpoints/chesscog_best.pt`, SHA-256 `11c54e55cd77ffa9fd8530dc75283a5d77953bbb66f0018b8a1f9deca049f1c9`: MobileNetV3 Small, existing 13-class ordering, four-channel RGB+mask, B.2 `context1_224`, existing resize/normalization/initialization.

The existing `src/training/multidataset.py::chessred_boards` requires explicit normalized geometry, game IDs and official game splits. Missing images, missing/unverified geometry, missing groups, inconsistent split roles, and conflicting piece annotations must be quarantined. It maps category names into the existing class mapping. Its annotated-corner regular-grid extraction is explicitly versioned `b2-annotated-corners-regular-grid-v1`; it is not proof of pixel parity with frozen prepared photographs.

Before actual use, a converter must verify White-relative corners, orientation, coordinate scaling/EXIF, source-role provenance and canonical square mapping. A caller-supplied verified status is not independent proof of orientation. The real annotation import/conversion remains pending, rather than being represented as an already working CLI. Repeated placements across games also require the existing position/game grouping audit; do not relax leakage rules to increase counts.

For images without annotated corners, a separate future process could run unchanged localization and orientation, then review true corners, semantic orientation, canonical grid and representative crops. Completion alone is insufficient. Unreliable examples must be rejected. This is a proposal, not a completed verification process.

No real-image visual report was produced because no image subset was obtained. The new adapter test uses explicitly artificial fixtures; it checks label mapping and a finite `(4,4,224,224)` DataLoader batch with binary target masks, not actual ChessReD accuracy or crop alignment.

## Reproduction and smallest unblock

Commands executed (fresh output paths are required; these completed paths cannot be overwritten):

```bash
.venv/bin/python scripts/probe_chessred_c6.py --output outputs/evaluation/chessred_c6/local_v1
.venv/bin/python -m modal run scripts/modal_chessred_c6.py --run-id chessred-c6-eu-v1
.venv/bin/python -m pytest tests/test_chessred_c6.py tests/test_multidataset_c4.py tests/test_chesscog_c5.py -q
cd web
npm test
```

Do not repeatedly retry the host or download archives. The smallest next step is to obtain **only the official annotations.json**, through a working browser/network or an author/4TU-supported annotation-only delivery. Save it outside Git at `outputs/evaluation/chessred_c6/manual_v1/annotations.json`, preserve its official source URL, and verify:

```bash
md5sum outputs/evaluation/chessred_c6/manual_v1/annotations.json
sha256sum outputs/evaluation/chessred_c6/manual_v1/annotations.json
```

The MD5 must equal the published annotation hash above; a mismatch requires investigation, not silent acceptance. Then inspect actual schema, establish official train/validation image lists and White-relative geometry, obtain only a few representative images, and review grid/crop overlays before preparing the larger subset. If official access remains unavailable, request a restored link or supported annotation-only copy from the publisher; no message was sent on the user's behalf.

## Storage, costs and licensing

Archive and extracted byte sizes could not be verified (HEAD requests failed; accessible DOI metadata supplied no sizes). For 1,772 images, storage is `1,772 × measured mean image bytes` plus annotations/caches. For example, a hypothetical 2 MB mean would imply 3.54 GB; this is not a measurement.

The authenticated Modal worker used CPU only, no GPU, the existing data Volume, and a fresh C.6 directory. App `ap-shJeLKxRqioR6LMGbDt5Kj` completed. The billing snapshot reports CPU $0.00061195, memory $0.00020700, and egress $0.00000018: **$0.00081913 metered at query time**, subject to billing lag and excluding unitemized storage/build costs. Remaining credits are not independently verified. No training resources were launched.

A conditional Stage 2 estimate using C.5 throughput—not measured ChessReD throughput—is about 3,341 training seconds for 1,442×64×5 observations, plus about 1,458 seconds for five full 330-board validations. At the prior L4+CPU+RAM rate of $0.9586/hour, that is roughly **$1.28 before startup, checkpointing and different JPEG/I/O costs**. Unknown exclusions and full-resolution decoding can substantially change this estimate. Profile a representative real subset before authorizing training. An illustrative 30-minute preparation cap at 2 CPU/4 GiB is about $0.19 using prior rates, not an executed preparation charge.

The dataset is documented as [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); implementation code is MIT. Attribution, noncommercial and share-alike conditions matter. MIT code licensing does not grant commercial dataset use. Commercial deployment and the status of derived model weights require separate permission/licensing assessment rather than assuming the weights are unrestricted.

## Validation and acceptance

- Passed: 49 relevant Python tests, including bounded requests, checksums, archive safeguards, class-name mapping, invalid/missing metadata quarantine and fixture DataLoader behavior.
- Passed: 39 browser regression tests; no browser code changes.
- Passed: 5,367 protected-file hashes unchanged; no final holdout access.
- Preserved: original videos, historical manifests/reports, production B.2 and C.5 checkpoints.
- Completed: bounded official retrieval investigation locally and on Modal, tested retrieval tooling, adapter contract checks, provenance/cost documentation.
- Blocked: actual annotation inspection, verified counts, real-image subset/crop review and Stage 2 preparation. No invented annotations or results.
- Not run: Stage 2 training, personal fine-tuning, production changes, commits or pushes.

**Decision: ChessReD adaptation cannot proceed yet. Obtain the official annotation-only file first, verify its checksum and actual geometry/split schema, then validate a small image subset. Stop C.6 here.**
