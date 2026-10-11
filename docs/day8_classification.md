# Day 8: classification and board contracts

The recognition task assigns one of 13 classes to the central target square of
a context crop. `src/chess/labels.py` is the single canonical mapping. IDs are
fixed for dataset labels, model output columns, checkpoints, and inference:

| IDs | Names | Symbols |
| --- | --- | --- |
| 0 | empty | `None` |
| 1–6 | white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king | P, N, B, R, Q, K |
| 7–12 | black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king | p, n, b, r, q, k |

Derived mappings are immutable; names and symbols are never defined separately.
The label version is `chess_pieces_v1`. Future checkpoints must preserve this
version and class order rather than relying on folder alphabetical ordering.

## Inputs and outputs

`SquareInput` in `src/vision/classification.py` owns a uint8 BGR crop and its
exclusive target bounds within that crop. The default 500×500 context crop has
target bounds `[200,200,300,300]`. The classifier receives no board coordinates,
FEN, turn, or legal-move information. Neighboring pieces provide context; only
the central square supplies the label.

Later preprocessing will convert BGR to RGB, resize the full crop to the
configured input size (224 by default), and apply the chosen pretrained model's
normalization identically during training and inference. No tensor transforms,
model loading, or training are implemented today. The model will output 13
probabilities in canonical ID order; raw logits must first become probabilities.

`SquarePrediction` attaches a result to row/column 0–7. Its class name and symbol
are derived from its ID. It validates confidence and, when supplied, all 13
probabilities: finite, in [0,1], summing to one, and consistent with the predicted
argmax and confidence. Ties select the lowest class ID deterministically.
Confidence is the selected class probability, not a claim of calibration.
Low confidence will be flagged later; it does not silently become `empty`.

```python
from src.vision.classification import SquarePrediction

probabilities = [0.06, 0.94] + [0.0] * 11
prediction = SquarePrediction.from_probabilities(4, 3, probabilities)
print(prediction.metadata())  # white_pawn, P, confidence 0.94, row 4 / column 3
```

## Board representation and orientation

`BoardState` in `src/chess/board_state.py` owns immutable rows of eight cells,
with eight rows total. Each cell is `None` or one canonical piece symbol.
Row 0 is the top of the image; column 0 is its left side. The default coordinate
system is `image_relative`. `white_at_bottom` is an explicit assertion after
orientation resolution/provision, where row 0/col 0 corresponds to a8.

Shape/symbol validity is separate from chess legality. Missing kings or composed
puzzle positions remain representable. BoardState does not invent side-to-move,
castling rights, clocks, or generate FEN. Construction does not assemble model
predictions automatically; that integration belongs to Day 13.

## Dataset format for Day 9

The future crop manifest will use records with:

- `crop_path`, `source_image`, and `position_id` for provenance and grouping.
- `row`, `col`, and `coordinate_system` for deterministic square identity.
- `class_id` and `class_name` from the canonical mapping, plus label version.
- Context-crop `target_bounds`, crop bounds, and coverage-mask path.
- Orientation status/source and localization/warp review eligibility.
- `split`, initially unassigned; position-level splits belong to Day 10.

Labels must be verified against known orientation and the target square.
Uncertain orientation or incorrect board alignment cannot silently enter the
supervised dataset. Group all ten variations of a position together when
splitting; crop count alone is not an independent sample count.

## Configuration and checks

The `classifier` section of `config/settings.yaml` contains the baseline
architecture `mobilenet_v3_small`, input size 224, batch size 64, learning rate
0.001, ten epochs, seed 42, confidence threshold 0.70, dataset paths, and checkpoint
path. These are starting settings, not selected through training results.
Paths resolve from the repository root. Nonpositive sizes/epochs, invalid
probability thresholds, missing paths, and unsupported architectures fail clearly.
Class count and order are derived from labels rather than duplicated in YAML.

```bash
python -m pytest tests/test_labels.py tests/test_board_state.py tests/test_classification.py tests/test_config.py
```

Tests cover canonical symbols with python-chess, board shape and invalid
symbols, independent storage, probability integrity, deterministic ties,
context-target bounds, serialization, and configuration validation.
No training dependencies were installed, no dataset was generated, and no FEN
converter was implemented. Day 9 will generate and visually verify labeled crops.
