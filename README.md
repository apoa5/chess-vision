# chess-vision

Turn photographs of physical chessboards into positions that can be analyzed with
Stockfish. The planned pipeline is photo → board localization → perspective
correction → 64 square crops → piece recognition → FEN → engine analysis.

Current stage: **Week 1, Day 1** — project setup and Stockfish integration.
Week 1 targets an 800×800 rectified board and 64 indexed crops; image processing
and piece recognition are not implemented yet. See [week1.md](week1.md).

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
`src/utils/config.py` loads and validates settings. `src/vision/`, `data/`,
`outputs/`, and `notebooks/` reserve space for subsequent Week 1 work.
