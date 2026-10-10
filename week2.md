Week 2 — Piece Recognition, Board Reconstruction, and FEN Generation
Week 2 Goal
By the end of Week 2, the project should reliably take a rectified chessboard image produced by the Week 1 pipeline and convert it into:
1. 64 classified chessboard squares,
2. a complete internal board representation,
3. a valid FEN string,
4. confidence information for each predicted square, and
5. basic chess-validity diagnostics using python-chess.
The complete Week 2 pipeline should look like:
Physical chessboard photo
        ↓
Week 1 board detection
        ↓
Perspective-corrected board
        ↓
64 square crops
        ↓
Piece classifier
        ↓
8×8 board representation
        ↓
FEN generation
        ↓
python-chess validation
Week 2 is focused on dataset construction, supervised piece classification, inference, board reconstruction, FEN generation, and evaluation.
Do not begin real-time video tracking, move inference, PGN recording, or frontend work during Week 2.
Week 2 Recommended Repository Additions
Extend the Week 1 structure without reorganizing working code unnecessarily.
vision-chess/
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── test/
│   ├── metadata/
│   └── classification/
│       ├── train/
│       ├── val/
│       └── test/
├── models/
│   ├── checkpoints/
│   └── exported/
├── outputs/
│   ├── board_detection/
│   ├── rectified_boards/
│   ├── square_crops/
│   ├── predictions/
│   └── evaluation/
├── src/
│   ├── chess/
│   │   ├── engine.py
│   │   ├── board_state.py
│   │   ├── fen.py
│   │   └── validation.py
│   ├── vision/
│   │   ├── board_detection.py
│   │   ├── perspective.py
│   │   ├── square_extraction.py
│   │   ├── dataset.py
│   │   ├── classifier.py
│   │   └── inference.py
│   └── utils/
│       ├── image_io.py
│       ├── visualization.py
│       └── metrics.py
├── scripts/
│   ├── build_dataset.py
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── predict_position.py
│   └── evaluate_positions.py
├── tests/
│   ├── test_fen.py
│   ├── test_board_state.py
│   ├── test_validation.py
│   └── test_inference.py
└── notebooks/
    └── classifier_experiments.ipynb
The exact structure can vary slightly, but the responsibilities should remain clearly separated.
Piece Classification Classes
Use 13 classes:
empty

white_pawn
white_knight
white_bishop
white_rook
white_queen
white_king

black_pawn
black_knight
black_bishop
black_rook
black_queen
black_king
Use one canonical internal label mapping throughout the project.
Suggested mapping:
CLASS_TO_SYMBOL = {
    "empty": None,
    "white_pawn": "P",
    "white_knight": "N",
    "white_bishop": "B",
    "white_rook": "R",
    "white_queen": "Q",
    "white_king": "K",
    "black_pawn": "p",
    "black_knight": "n",
    "black_bishop": "b",
    "black_rook": "r",
    "black_queen": "q",
    "black_king": "k",
}
Avoid creating different label names in separate scripts.
Day 8 — Define the Classification Problem and Dataset Format
Objective
Define the piece-recognition task clearly before training anything.
The project should have one reliable representation for labels, datasets, board states, and model outputs.
Tasks
1. Create the canonical class mapping
Define all 13 classes in one central location.
Include:
- integer class ID,
- human-readable label,
- chess symbol,
- optional display name.
Example:
0  empty         .
1  white_pawn    P
2  white_knight  N
...
12 black_king    k
Do not duplicate this mapping across multiple files.
2. Define the classifier input
The classifier should receive:
one cropped chess square
and output:
13 class probabilities
The classifier should not need to know:
- board coordinates,
- FEN,
- whose turn it is,
- legal moves.
Those concerns belong to later pipeline stages.
3. Define the board-state representation
Create:
src/chess/board_state.py
Use a deterministic 8×8 representation.
For example:
[
    ["r", "n", "b", "q", "k", "b", "n", "r"],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    [None, None, None, None, None, None, None, None],
    ...
]
Document clearly:
row 0 = top row of the normalized image
row 7 = bottom row of the normalized image
Do not convert rows to chess ranks until orientation is resolved.
4. Define prediction objects
Each square prediction should preserve:
- row,
- column,
- predicted class,
- confidence,
- optionally all class probabilities.
Example:
{
    "row": 4,
    "col": 3,
    "class_name": "white_pawn",
    "symbol": "P",
    "confidence": 0.94
}
5. Update configuration
Add classifier-related settings to:
config/settings.yaml
Potential values:
- input image size,
- batch size,
- learning rate,
- number of epochs,
- model architecture,
- confidence threshold,
- dataset paths,
- model checkpoint path.
Do not hard-code training parameters in multiple scripts.
6. Add tests
Test that:
- exactly 13 classes exist,
- every non-empty class maps to a valid FEN piece symbol,
- board states always contain 8 rows × 8 columns,
- invalid symbols are rejected.
Deliverables
By the end of Day 8:
- canonical label system,
- board-state representation,
- prediction data structure,
- updated configuration,
- unit tests for mappings and board shape.
Acceptance Criteria
There should be no ambiguity about how a predicted square becomes a chess piece symbol.
The project must have one canonical source of truth for class names and symbols.
Do Not Work On Yet
Do not train a model today.
Do not generate FEN yet.
Day 9 — Build the Square-Level Dataset
Objective
Turn the Week 1 board images and ground-truth positions into a supervised classification dataset.
The important challenge is to create labels correctly and avoid data leakage.
Tasks
1. Use Week 1 rectified boards
Start from the perspective-corrected board images produced by Week 1.
Do not manually crop random pieces from raw camera frames if a normalized board is already available.
2. Link every board to its ground-truth position
Use the Week 1 metadata containing:
- image ID,
- position ID,
- FEN,
- orientation.
The board orientation must be known before assigning chess pieces to image rows and columns.
3. Parse ground-truth FEN
Use python-chess or a clearly tested parser to convert each ground-truth FEN into a board matrix.
Ensure the matrix orientation matches the normalized image orientation.
This step must be verified visually on sample boards.
4. Extract and label all 64 squares
For every valid normalized board:
board image
↓
64 square crops
↓
label using known board state
Store:
- image/crop path,
- source board image ID,
- position ID,
- row,
- column,
- class label.
5. Handle orientation correctly
If the physical board is viewed with White nearest the camera versus Black nearest the camera, the visual matrix may need rotation.
Do not silently assume all images use the same orientation unless the dataset collection enforced that rule.
Prefer one of these approaches:
A. Normalize all board images to White-at-bottom orientation
or:
B. Store orientation metadata and rotate labels accordingly
For Version 1, approach A is simpler if practical.
6. Create dataset metadata
Create a CSV or JSON file containing one record per crop.
Suggested fields:
crop_path
source_image
position_id
row
col
class_id
class_name
split
7. Inspect labels manually
Create a visualization that randomly samples labeled square crops and displays:
image + expected label
Inspect examples from every class.
This is mandatory before training.
Deliverables
By the end of Day 9:
- square-level dataset,
- crop metadata,
- correct ground-truth labels,
- orientation handling,
- visual label-inspection tool.
Acceptance Criteria
Manually inspect at least several examples from every piece class and confirm that the labels are correct.
No training should begin until label orientation has been verified.
Day 10 — Create Leakage-Safe Train/Validation/Test Splits
Objective
Create a trustworthy evaluation setup.
The test set must measure generalization to unseen board images and positions rather than memorization of nearly identical crops.
Tasks
1. Split by source position or source image
Do not randomly split individual square crops.
Bad approach:
square from board_001 → train
another square from board_001 → test
This leaks visual information.
Instead split at the position or source-image level.
Preferred:
70% positions → train
15% positions → validation
15% positions → test
If the dataset is still small, adjust slightly while preserving independence.
2. Preserve class coverage
Check that important classes appear in every split where possible.
Classes such as kings and queens may have relatively few samples.
Report per-class counts.
3. Measure class imbalance
The empty class will probably dominate.
Generate a table such as:
class           train   val   test
empty           ...
white_pawn      ...
white_knight    ...
...
4. Decide how to address imbalance
Potential strategies:
- class-weighted loss,
- weighted sampler,
- moderate oversampling,
- data augmentation.
Do not simply duplicate rare samples excessively.
5. Add data augmentation
Use sensible image augmentations that preserve piece identity.
Potential augmentations:
- mild brightness changes,
- contrast changes,
- slight blur,
- mild noise,
- small translations/crops,
- modest scale variation.
Be cautious with:
- large rotations,
- flips,
- transformations that produce physically unrealistic board views.
The Week 1 perspective correction already handles much of the viewpoint variation.
6. Create dataset-loading code
Implement:
src/vision/dataset.py
It should:
- load crops,
- apply transforms,
- return image tensor + class ID,
- support train/validation/test modes.
Deliverables
By the end of Day 10:
- leakage-safe splits,
- class-distribution report,
- augmentation pipeline,
- reusable dataset loader.
Acceptance Criteria
No source position should appear in both training and test sets.
The split logic should be reproducible with a fixed random seed.
Day 11 — Train the First Piece Classifier
Objective
Train a lightweight baseline classifier using transfer learning.
The goal is to establish a working benchmark, not to find the perfect architecture immediately.
Tasks
1. Choose a lightweight pretrained architecture
Recommended first options:
MobileNetV3
ResNet18
EfficientNet-B0
Choose one baseline model first.
Do not train several architectures at once unless the baseline has already been completed and evaluated.
2. Replace the final classification layer
Adapt the model to:
13 output classes
3. Use transfer learning
Start with pretrained weights.
A reasonable first strategy:
- freeze most backbone layers initially,
- train classification head,
- optionally unfreeze later layers for fine-tuning.
4. Implement the training loop
Create:
scripts/train_classifier.py
Track:
- training loss,
- validation loss,
- training accuracy,
- validation accuracy.
Save the best model based on validation performance.
5. Add reproducibility controls
Use:
- fixed random seed,
- configuration-driven hyperparameters,
- logged model architecture,
- logged dataset split.
6. Save checkpoints
Store checkpoints in:
models/checkpoints/
Use meaningful names, for example:
mobilenet_v3_epoch_08_valacc_0.932.pt
7. Avoid overtraining
Start with a modest number of epochs.
Watch validation loss.
If validation performance worsens while training performance improves, document overfitting.
Deliverables
By the end of Day 11:
- working training script,
- first trained classifier,
- training/validation metrics,
- saved checkpoint.
Acceptance Criteria
The training pipeline must complete successfully and produce a usable model checkpoint.
A first validation accuracy around or above:
90%
would be encouraging, but correctness of the pipeline matters more than hitting an arbitrary number on the first run.
Day 12 — Evaluate and Improve the Classifier
Objective
Understand where the model succeeds and fails before integrating it into the board pipeline.
Tasks
1. Evaluate on the held-out test split
Create:
scripts/evaluate_classifier.py
Measure:
- overall accuracy,
- per-class accuracy,
- confusion matrix,
- precision/recall/F1 if convenient.
2. Generate a confusion matrix
Look specifically for meaningful confusions such as:
bishop ↔ pawn
queen ↔ king
rook ↔ pawn
black piece ↔ empty
white piece ↔ empty
3. Inspect misclassified examples
Save or display examples containing:
- crop image,
- ground-truth class,
- predicted class,
- prediction confidence.
Group common failure types.
4. Check confidence calibration informally
Compare:
correct high-confidence predictions
incorrect high-confidence predictions
low-confidence predictions
The confidence values will later help with chess-based correction and move inference.
5. Improve the baseline only where justified
Possible improvements:
- better crop padding,
- stronger but realistic augmentation,
- class weighting,
- fine-tuning more backbone layers,
- slightly larger input size,
- additional training images.
Do not change many variables simultaneously.
6. Save the best model
Record:
- model architecture,
- training settings,
- validation score,
- test score,
- dataset version.
Deliverables
By the end of Day 12:
- held-out test evaluation,
- confusion matrix,
- misclassification examples,
- best model checkpoint,
- documented failure modes.
Acceptance Criteria
The classifier should have a clearly measured baseline.
Target:
>= 90% square-level classification accuracy
if the dataset quality supports it.
If below this target, the failure modes should be understood and documented rather than hidden.
Day 13 — Integrate Inference and Reconstruct the Board
Objective
Apply the trained classifier to all 64 squares of a normalized board and reconstruct the complete 8×8 chess position.
Tasks
1. Create a reusable inference module
Implement:
src/vision/inference.py
It should:
- load the trained model once,
- preprocess square crops consistently,
- run inference,
- return class probabilities,
- return predicted class and confidence.
Avoid reloading the model separately for every square.
2. Run batch inference over 64 squares
For one board:
64 crops
↓
single batch if practical
↓
64 predictions
Batch inference is preferable to 64 independent model loads/calls.
3. Reconstruct the 8×8 board
Use the square coordinates to produce:
[
    ["r", None, "b", ...],
    ...
]
4. Preserve confidence information
Store an accompanying confidence matrix:
[
    [0.98, 0.91, 0.87, ...],
    ...
]
or equivalent structured predictions.
This will be useful later for:
- low-confidence warnings,
- legal-move correction,
- debugging.
5. Create a board prediction visualization
Overlay predicted piece labels and confidence values on the rectified board.
For example:
black_rook 0.98
empty 0.99
white_pawn 0.92
The visualization should make obvious mistakes easy to spot.
6. Create a prediction script
Create:
scripts/predict_position.py
Input:
raw board image
or:
rectified board image
depending on CLI option.
Output:
- predicted board matrix,
- per-square confidence,
- prediction visualization.
If using a raw image, reuse the Week 1 pipeline rather than duplicating board-detection logic.
Deliverables
By the end of Day 13:
- reusable inference module,
- batch prediction over all 64 squares,
- reconstructed board matrix,
- confidence information,
- visual prediction output.
Acceptance Criteria
A single command should take a known test board and print a complete predicted 8×8 position.
The program should not require manually classifying any squares.
Day 14 — Generate FEN, Validate Positions, and Evaluate End-to-End Accuracy
Objective
Complete the Week 2 pipeline by converting predicted board states into FEN and measuring full-position performance.
Tasks
1. Implement board-to-FEN conversion
Create:
src/chess/fen.py
Implement placement-field generation yourself.
Example:
[None, None, None, None, "P", None, None, None]
becomes:
4P3
Eight rows become:
rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
2. Handle FEN fields responsibly
Computer vision can directly determine only the piece-placement component.
It cannot always infer from a single static image:
- side to move,
- castling rights,
- en passant target square,
- halfmove clock,
- fullmove number.
Therefore separate:
piece-placement FEN
from:
complete game-state FEN
For static puzzle analysis, allow these fields to be supplied or defaulted explicitly, for example:
<placement> w - - 0 1
Do not pretend the camera inferred information it cannot know.
3. Add chess-position validation
Create:
src/chess/validation.py
Check useful properties such as:
- board has exactly 64 squares,
- valid piece symbols,
- one white king where appropriate,
- one black king where appropriate,
- no impossible number of pawns if enforcing standard positions,
- python-chess can construct the resulting board.
Treat puzzle/problem positions carefully.
Some composed positions may not correspond to a normal reachable game.
Validation should produce diagnostics rather than always hard-failing.
4. Compare prediction to ground truth
Create:
scripts/evaluate_positions.py
For each test board, calculate:
- number of correctly classified squares,
- square accuracy,
- number of piece errors,
- piece-placement FEN exact match,
- complete-board exact match,
- average confidence,
- inference latency.
5. Measure board-level accuracy
This metric is essential.
A model can have high square-level accuracy while still producing many imperfect boards.
Report both:
square-level accuracy
and:
exact board/FEN accuracy
6. Save evaluation results
Create a CSV such as:
outputs/evaluation/week2_positions.csv
Suggested fields:
image_id
ground_truth_placement
predicted_placement
square_accuracy
piece_errors
exact_match
avg_confidence
inference_time_ms
validation_status
notes
7. Run end-to-end Week 2 tests
Use board images that were not part of training.
Test:
raw physical board image
↓
Week 1 localization
↓
perspective correction
↓
square extraction
↓
classifier
↓
board reconstruction
↓
FEN generation
↓
validation
8. Update README
Document:
- classifier architecture,
- dataset strategy,
- class definitions,
- evaluation methodology,
- current accuracy,
- current limitations.
Do not overstate performance.
Deliverables
By the end of Day 14:
- board-to-FEN converter,
- position validator,
- complete end-to-end inference script,
- square-level metrics,
- board/FEN-level metrics,
- evaluation CSV,
- updated README.
Final Week 2 Acceptance Test
Given a raw test photograph such as:
data/test/example.jpg
a single command should produce:
1. detected and rectified chessboard
2. 64 square predictions
3. complete board matrix
4. piece-placement FEN
5. optional full FEN with explicitly supplied/default game-state fields
6. validation diagnostics
7. confidence information
Conceptually:
Physical-board photo
        ↓
Board localization
        ↓
Perspective correction
        ↓
64 crops
        ↓
13-class piece recognition
        ↓
8×8 board state
        ↓
Piece-placement FEN
        ↓
Validation
Week 2 Metrics to Track
At minimum, record:
square-level classification accuracy
per-class accuracy
confusion matrix
exact board accuracy
exact piece-placement FEN accuracy
average inference latency
average prediction confidence
It is important to distinguish:
Square accuracy
from:
Whole-board accuracy
Example:
Square accuracy: 97.0%
Exact board accuracy: 68.0%
That result is entirely possible because one incorrect square makes the full board incorrect.
Do not report only the more flattering metric.
Week 2 Definition of Done
Week 2 is complete when all of the following are true:
- [ ] A canonical 13-class label system exists.
- [ ] The square dataset is generated automatically from known board positions.
- [ ] Dataset orientation has been verified.
- [ ] Train/validation/test splits avoid source-position leakage.
- [ ] Class imbalance has been measured.
- [ ] A lightweight piece classifier has been trained.
- [ ] The model has been evaluated on a held-out test set.
- [ ] Misclassifications have been inspected.
- [ ] The trained model can classify all 64 squares of a board.
- [ ] Predictions are reconstructed into an 8×8 board.
- [ ] Prediction confidence is preserved.
- [ ] Piece-placement FEN is generated correctly.
- [ ] FEN generation has unit tests.
- [ ] python-chess validation is integrated.
- [ ] End-to-end board-level accuracy is measured.
- [ ] Evaluation results are saved.
- [ ] README documents Week 2 capabilities and limitations.
Important Scope Boundary for Codex
During Week 2, Codex should optimize for:
- correct dataset construction,
- reproducible model training,
- prevention of dataset leakage,
- simple baseline architecture,
- measurable performance,
- confidence preservation,
- modular inference code,
- correct FEN generation,
- honest evaluation.
Codex should not implement future functionality early unless explicitly requested.
Do not implement during Week 2:
- continuous webcam/video processing,
- automatic move detection,
- temporal smoothing,
- legal-move-based error correction,
- PGN game recording,
- clock recognition,
- FastAPI backend,
- Next.js frontend,
- WebSocket streaming,
- live game broadcasting.
These belong to Weeks 3 and 4.
The Week 2 foundation should be strong enough that Week 3 can add Stockfish analysis and real-time camera inference without needing to redesign the recognition pipeline.