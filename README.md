# chess-vision

Turn photographs of physical chessboards into positions that can be analyzed with
Stockfish. The planned pipeline is photo → board localization → perspective
correction → 64 square crops → piece recognition → FEN → engine analysis.

Current stage: **Week 1, Day 2** — image utilities and vision scaffolding, with
working Stockfish integration. Week 1 targets an 800×800 rectified board and
64 indexed crops. Board detection, warping, and extraction are scheduled for
later days; piece recognition follows Week 1. See [week1.md](week1.md).

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
python scripts/rectify_board.py data/raw/example.jpg --corners 50 50 750 50 750 750 50 750
python scripts/extract_squares.py outputs/rectified_boards/example.png
```

These three commands are scaffolds: with valid input, they currently exit with
a clear message identifying the scheduled implementation day and create no
stage outputs. The working preview command above checks Day 2 image I/O.

Run the Day 2 tests without needing Stockfish:

```bash
python -m pytest tests/test_image_io.py tests/test_visualization.py tests/test_perspective.py tests/test_vision_cli.py tests/test_vision_contracts.py
```
