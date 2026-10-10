# chess-vision

Turn photographs of physical chessboards into positions that can be analyzed with
Stockfish. The planned pipeline is photo → board localization → perspective
correction → 64 square crops → piece recognition → FEN → engine analysis.

Current stage: **Week 1, Day 7** — the automatic image pipeline detects a board,
warps it to 800×800, and produces a labeled grid preview and 64 indexed crops.
Image utilities, dataset metadata, evaluation, and Stockfish integration are in
place. The development localization result is 15/20 (75%), below the 90% target;
Week 1 reliability remains unfinished. Piece recognition follows Week 1.
See [week1.md](week1.md).

## Setup

Use Python 3.12 (the version used for development):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install the executable separately from [official Stockfish downloads](https://stockfishchess.org/download/).
On this Linux x86-64 workspace, the official universal archive is extracted into
`tools/stockfish/`; the configured executable is
`tools/stockfish/stockfish-linux-x86-64-universal`. Binaries are ignored by Git.
To reproduce that layout on Linux x86-64:

```bash
mkdir -p tools
curl -fL https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-linux-x86-64-universal.tar.gz -o /tmp/chess-vision-stockfish.tar.gz
tar -xzf /tmp/chess-vision-stockfish.tar.gz -C tools
```

Validated with Stockfish 19. The downloaded archive's SHA-256 was
`9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f`;
the `latest` URL may change in future releases.
For another installation, set its absolute path (or a command on PATH):

```bash
export STOCKFISH_PATH=/absolute/path/to/stockfish
```

Settings live in `config/settings.yaml`: engine path, normalized board size,
data directories, and output directories. Relative settings paths resolve from
the repository root. `STOCKFISH_PATH` overrides the configured engine path.

## Verify Day 1

```bash
python scripts/test_stockfish.py
python -m pytest
```

The script prints the starting FEN, a legal best move, evaluation, and principal
variation. Evaluation uses White's perspective: positive centipawns favor White,
negative favor Black, and `#` indicates a mate score. Exact output varies.
Use `--engine PATH` or `--depth 8` to override the script defaults.
Tests require a real Stockfish installation and fail clearly if it is missing.
In the managed Codex sandbox, Stockfish 19 analysis hung; the acceptance check
and engine tests are run with approved execution outside that sandbox.

`src/chess/engine.py` provides context-managed analysis from FEN or `chess.Board`,
best-move and evaluation helpers, executable validation, and safe shutdown, using
[python-chess's UCI interface](https://python-chess.readthedocs.io/en/latest/engine.html).
`src/utils/config.py` loads and validates settings.

## Verify Day 2

Load a photo and save it again (parent output directories are created):

```bash
python scripts/preview_image.py data/raw/example.jpg --output outputs/board_detection/example_preview.png
```

Optionally draw four **known** corner points, in original image pixels, and
resize the annotated preview. Replace these example coordinates with your own:

```bash
python scripts/preview_image.py data/raw/example.jpg --output outputs/board_detection/example_corners.png --corners 50 50 750 50 750 750 50 750 --max-size 800
```

Image I/O and debug overlays live in `src/utils/image_io.py` and
`src/utils/visualization.py`. Images load as uint8 BGR; overlays preserve the
source array. Use PNG for a lossless round trip. These utilities use
[OpenCV image codecs](https://docs.opencv.org/4.x/d4/da8/group__imgcodecs.html) and
[drawing functions](https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html).

The vision contracts and conventions are documented in
[docs/vision_pipeline.md](docs/vision_pipeline.md): corners are TL/TR/BR/BL;
square indices are rows/columns 0–7, without a chess-coordinate mapping.

The stage CLIs accept image paths and support `--help`:

```bash
python scripts/detect_board.py data/raw/example.jpg
python scripts/rectify_board.py data/raw/example.jpg
python scripts/extract_squares.py outputs/rectified_boards/example.png
```

Detection, rectification, and extraction now work. The preview command checks
Day 2 image I/O.

Run the Day 2 tests without needing Stockfish:

```bash
python -m pytest tests/test_image_io.py tests/test_visualization.py tests/test_perspective.py tests/test_vision_cli.py tests/test_vision_contracts.py
```

## Verify Day 3

The current dataset has 75 real photographs across 18 positions, recorded FENs,
capture conditions, and a fixed 20-image development subset covering all 18
positions. The board, camera, metadata schema, and collection gaps are described
in [data/metadata/README.md](data/metadata/README.md).

```bash
python scripts/validate_dataset.py
python scripts/validate_dataset.py --report data/metadata/dataset_summary.json
python -m pytest tests/test_dataset.py
```

Validation checks raw-file coverage, position FENs, filename/metadata links,
capture fields, image decoding, and agreement between the development manifest
and membership flags. It does not recognize or verify pieces in the images.
The photos are currently indoors; collection toward the original approximately
200-image target, including outdoor photos, will continue alongside Day 4.
Keep the development subset fixed when adding new images.

## Verify Day 4

Detect the inner playing area of a real photo:

```bash
python scripts/detect_board.py data/raw/position_014_var_01.jpg
python scripts/detect_development_subset.py
python -m pytest tests/test_board_detection.py tests/test_vision_cli.py
```

The batch command processes the fixed 20-image manifest. Debug files in
`outputs/board_detection/` include labeled corner overlays, Canny edge images,
candidate diagnostics JSON, and `day4_results.json`. Failed detections also
save debug files; the single-image command exits with status 1 on failure.
Use `--output PATH` for one image or `--output-dir PATH` for a batch.

The historical Day 4 baseline accepted candidates in 13/20 development images. This is not a
verified localization accuracy rate: inspect the overlays, especially where
pieces obscure edges or polygons include part of the wooden border. Settings
are configurable in `config/settings.yaml`. See
[docs/day4_baseline.md](docs/day4_baseline.md) for the method and known failures.

## Verify Day 5

```bash
python scripts/evaluate_board_detection.py
python -m pytest tests/test_board_detection.py tests/test_detection_evaluation.py
```

Evaluation writes overlays, edge images, diagnostic JSON, `evaluation.csv`,
and `evaluation.json` under `outputs/board_detection/day5/`. It records candidate
acceptance separately from reviewed board-localization correctness. Manual
reviews in `data/metadata/day5_review.json` are bound to the photo, settings,
and predicted corners; changed results become unreviewed rather than retaining
an obsolete correctness label.

The current run finds 16/20 candidates and visual review approves 15/20 (75%).
The 90% target is **not met**. Dense-piece occlusion, weak grid evidence, and an
internal-line false positive remain documented in
[docs/day5_evaluation.md](docs/day5_evaluation.md). This is a measured development
result for the indoor starter dataset, not a guarantee of precise crop alignment.

## Verify Day 6

```bash
python scripts/rectify_board.py data/raw/position_014_var_01.jpg
python scripts/rectify_development_subset.py
python -m pytest tests/test_perspective.py tests/test_vision_cli.py tests/test_vision_contracts.py
```

Outputs in `outputs/rectified_boards/` retain source stems and use lossless PNG.
The configured size defaults to 800×800. The batch report `rectification.json`
records failures, corners, dimensions, and signature-matched Day 5 reviews.
All accepted candidates are warped, including known incorrect detections for
debugging. Inspect those review labels before treating an output as correct.
See [docs/day6_perspective.md](docs/day6_perspective.md) for visual checks and limitations.

## Run the complete Week 1 pipeline

```bash
python scripts/process_board_image.py data/raw/position_014_var_01.jpg
python scripts/evaluate_week1.py --report data/metadata/week1_evaluation.json
```

The single-image command produces:

- `outputs/board_detection/position_014_var_01_corners.png`: corner overlay, with edge and diagnostic files alongside it.
- `outputs/rectified_boards/position_014_var_01.png`: unannotated 800×800 board.
- `outputs/rectified_boards/position_014_var_01_grid.png`: grid with row/column labels.
- `outputs/square_crops/position_014_var_01/`: 64 PNG crops (`r0_c0.png` through `r7_c7.png`) and `squares.json`.

Use `--output-dir PATH` to place all artifacts under another root. Detection
failures save debug artifacts and exit with status 1, without generating crops.
Inspect the grid preview before trusting square contents. Labels describe image
indices, not chess coordinates; the system does not infer White's orientation.

Batch evaluation writes JSON/CSV and artifacts to `outputs/week1/`, preserving
the fixed 20-image subset. Sixteen candidates generate 64 crops each; coarse
visual review approves fifteen boards. Four detections fail, and one accepted
candidate includes the wooden frame. Minor alignment offsets and perspective
distortion of tall pieces remain. Controlled indoor photos with the whole board
visible are the current scope; outdoor performance is unmeasured. There is no
piece recognition, automatic FEN reconstruction, or video processing yet.

See [docs/day7_integration.md](docs/day7_integration.md) for evaluation criteria
and remaining Week 1 work. The main repository folders are `src/vision/` for
reusable stages, `src/utils/` for I/O and debug helpers, `src/chess/` for Stockfish,
`scripts/` for CLIs, `tests/` for checks, `config/` for settings, `data/metadata/`
for dataset/review records, and `outputs/` for generated artifacts.

## Preserve edge-piece context

Rectification and the full pipeline now also save a larger `_context.png` image
and coverage mask, sampled directly from the original photograph. The 800×800
playing area remains explicitly located inside it. The full pipeline saves
64 overlapping context crops under `outputs/square_crops/<stem>/context/`,
with target-square bounds and masks. Defaults use a two-square margin and
padding (1200×1200 canvas, 500×500 crops). These retain more of tall edge pieces
while keeping the existing exact grid crops available. See
[docs/board_context.md](docs/board_context.md) for settings, coordinates, and limitations.
