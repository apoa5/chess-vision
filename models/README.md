# Board-coordinate templates

`board_markings.npz` contains sixteen uint8 48×36 glyph silhouettes for the
printed A–H and 1–8 markings on the project's wooden board. It is a small
classical template bank, not a neural piece-classifier checkpoint.

`board_markings.json` records the reference photograph's SHA-256, filename,
asserted orientation, and expected print layout. The calibration photo is
`position_001_var_08.jpg`, with White's home rank at the bottom. Bottom file
letters and left rank digits supplied the templates; the opposite borders
provide additional runtime evidence.

Reproduce with:

```bash
python scripts/calibrate_board_orientation.py data/raw/position_001_var_08.jpg --white-side bottom
```

The raw reference photo is local and ignored by Git. Runtime inference needs
only the checked-in template bank and reads no FEN or capture metadata. New
fonts/layouts are outside this calibration's scope. See
[board orientation documentation](../docs/board_orientation.md).

The separate `checkpoints/board_marking_reader.pt` reads this wooden board's
coordinate characters; see [training and evaluation](../docs/learned_board_markings.md).
It is not the chess-piece classifier. Checkpoints remain excluded from Git.
