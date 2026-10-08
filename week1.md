Week 1 — Chess Vision Project Foundation
Week 1 Goal
By the end of Week 1, the project should reliably take a photograph of a physical chessboard and produce:
1. a detected chessboard region,
2. the four ordered board corners,
3. a perspective-corrected top-down image of the board, and
4. 64 consistently indexed square crops.
Week 1 is focused on project architecture, chess-engine integration, image data preparation, chessboard localization, perspective correction, and square extraction.
Do not train a chess-piece recognition model during Week 1. Piece classification, FEN reconstruction from images, and real-time video inference belong to later weeks.
Recommended Week 1 Repository Structure
Use a clean structure that can grow without requiring major reorganization later.
vision-chess/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── test/
│   └── metadata/
├── outputs/
│   ├── board_detection/
│   ├── rectified_boards/
│   └── square_crops/
├── src/
│   ├── __init__.py
│   ├── chess/
│   │   ├── __init__.py
│   │   └── engine.py
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── board_detection.py
│   │   ├── perspective.py
│   │   └── square_extraction.py
│   └── utils/
│       ├── __init__.py
│       ├── image_io.py
│       └── visualization.py
├── scripts/
│   ├── test_stockfish.py
│   ├── detect_board.py
│   ├── rectify_board.py
│   └── extract_squares.py
├── tests/
│   ├── test_engine.py
│   ├── test_perspective.py
│   └── test_square_extraction.py
└── notebooks/
The exact structure can be adjusted slightly if necessary, but avoid putting all logic into one script.
Day 1 — Repository Setup and Stockfish Integration
Objective
Create a clean, reproducible project foundation and prove that Python can communicate correctly with Stockfish.
Tasks
1. Initialize the project
Create the project repository and base folder structure.
Set up:
- Git repository
- Python virtual environment
- .gitignore
- requirements.txt
- initial README.md
The README should briefly state:
- the problem being solved,
- the planned end-to-end pipeline,
- the current project stage,
- how to create the environment,
- how to install dependencies.
Do not write a large final README yet.
2. Install the initial dependencies
Install only the libraries required for the current stage.
Recommended initial dependencies:
opencv-python
numpy
python-chess
PyYAML
pytest
Install Stockfish separately as the chess engine executable.
Do not add PyTorch, YOLO, FastAPI, Next.js tooling, or other Week 2+ dependencies unless they are immediately necessary.
3. Add configuration support
Create a central configuration file such as:
config/settings.yaml
It should contain values that should not be hard-coded throughout the project, including:
- Stockfish executable path
- normalized board image size
- data directories
- output directories
Suggested initial normalized board size:
800 x 800
This gives each square a clean size of:
100 x 100
4. Build the Stockfish wrapper
Create:
src/chess/engine.py
Implement a small reusable interface around python-chess.
It should support at minimum:
- loading the Stockfish engine,
- validating that the engine executable exists,
- passing a FEN or chess.Board position to the engine,
- requesting a best move,
- requesting an evaluation,
- safely shutting down the engine.
Avoid placing all engine logic directly inside a test script.
5. Create a Stockfish test script
Create:
scripts/test_stockfish.py
The script should:
1. create the standard starting chess position,
2. send it to Stockfish,
3. request an analysis with a modest time or depth limit,
4. print:
   - FEN,
   - best move,
   - engine evaluation,
   - principal variation if easily available.
The exact evaluation is not important because engine output may vary by depth, hardware, or version.
6. Add basic tests
Create tests that confirm:
- configuration loads successfully,
- the Stockfish executable can be found,
- the engine can analyze a valid position,
- invalid engine paths produce a clear error.
Deliverables
By the end of Day 1, the repository should contain:
- a clean project structure,
- working Python environment,
- dependency file,
- initial README,
- configuration file,
- reusable Stockfish wrapper,
- Stockfish test script,
- basic automated tests.
Acceptance Criteria
Day 1 is complete only when a command similar to:
python scripts/test_stockfish.py
successfully produces engine output such as:
FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
Best move: e2e4
Evaluation: ...
The exact best move and score may differ.
Do Not Work On Yet
Do not implement:
- board detection,
- piece recognition,
- FEN generation from images,
- model training,
- real-time video,
- frontend UI,
- automatic move recording.
Day 2 — Image Utilities and Vision Pipeline Scaffolding
Objective
Prepare the codebase for computer-vision development before implementing board detection.
Tasks
1. Create reusable image-loading utilities
Create:
src/utils/image_io.py
Implement reusable functions for:
- loading an image,
- checking that the image exists,
- checking that OpenCV successfully decoded it,
- saving an output image,
- creating output directories when necessary.
Functions should fail with useful error messages instead of silent None values.
2. Add visualization utilities
Create:
src/utils/visualization.py
Add reusable helpers for:
- drawing points,
- drawing polygons,
- drawing board corner labels,
- optionally resizing images for preview,
- saving debug visualizations.
Corner labels should make orientation easy to inspect, for example:
TL
TR
BR
BL
3. Define the vision data flow
Create the initial interfaces for:
src/vision/board_detection.py
src/vision/perspective.py
src/vision/square_extraction.py
At this stage, functions may be placeholders or minimal implementations, but define clean responsibilities.
Example conceptual interfaces:
detect_board_corners(image)
order_corners(points)
warp_board(image, corners, output_size=800)
extract_squares(board_image)
Avoid building one large function that performs the entire pipeline.
4. Define a standard corner representation
Use one consistent corner order throughout the entire project:
top-left
top-right
bottom-right
bottom-left
Every board-detection and perspective function must respect this convention.
Document it clearly in code.
5. Define square indexing conventions
Choose and document how extracted squares will initially be represented.
For Week 1, the safest internal representation is:
row 0, column 0
through
row 7, column 7
Do not prematurely assume that row 0, col 0 always equals a8 until board orientation is known.
Create metadata that preserves:
row
column
crop coordinates
Chess-coordinate mapping can be added later.
6. Create CLI scaffolding
Create scripts that will eventually execute each stage:
scripts/detect_board.py
scripts/rectify_board.py
scripts/extract_squares.py
Each script should accept image paths as command-line arguments rather than relying on hard-coded local filenames.
Deliverables
By the end of Day 2:
- image I/O utilities exist,
- visualization helpers exist,
- the vision modules have clean interfaces,
- corner ordering is standardized,
- square indexing is documented,
- basic CLI scripts exist.
Acceptance Criteria
A test image should be loadable through the project utilities and saved again without errors.
The project should be ready for Day 3 data collection without needing architectural changes.
Do Not Work On Yet
Do not try to solve board detection today.
The goal is clean infrastructure.
Day 3 — Collect and Organize the Initial Chessboard Dataset
Objective
Create a small but useful image dataset for developing and testing board detection.
The dataset should reflect the actual physical environment in which the system will eventually operate.
Tasks
1. Choose the primary chessboard and chess set
Use the board and pieces that will serve as the project's initial target hardware.
For Version 1, prioritize consistency over generalization.
Document:
- board dimensions if known,
- board style,
- piece style,
- camera device used,
- approximate capture environment.
2. Create 20–30 physical chess positions
Include variety such as:
- starting position,
- middlegame positions,
- sparse endgames,
- king-and-pawn positions,
- pieces near board edges,
- clustered pieces,
- positions with tall pieces near shorter pieces,
- empty or nearly empty board areas.
3. Capture multiple angles per position
For each position, take several images covering approximately:
- near-overhead / 90°,
- ~75°,
- ~60°,
- slight rotations,
- several distances.
Keep the full board visible.
Avoid extremely difficult camera angles during this first dataset.
Target approximately:
150–300 total images
A reasonable initial target is:
20 positions x 10 images = 200 images
4. Include controlled environmental variation
Capture modest differences in:
- lighting,
- camera distance,
- board rotation,
- background visibility.
Do not deliberately create extreme occlusion yet.
5. Store the ground-truth position
For each physical position, record the corresponding FEN manually.
Create a metadata file such as:
data/metadata/positions.json
Suggested structure:
{
  "position_001": {
    "fen": "...",
    "notes": "Starting position",
    "orientation": "white_near_camera"
  }
}
Image metadata should link each image to a position identifier.
6. Establish a filename convention
Use deterministic filenames, for example:
position_001_angle_01.jpg
position_001_angle_02.jpg
position_002_angle_01.jpg
Avoid filenames like:
IMG_1234.jpg
new.jpg
test-final-2.jpg
7. Create a small fixed development/test subset
Select approximately 20 representative images that will be reused repeatedly during Week 1.
Store their identities in metadata rather than constantly changing the test set.
These should include:
- easy overhead images,
- moderate perspective images,
- different positions,
- slightly different lighting.
Deliverables
By the end of Day 3:
- organized raw chessboard images,
- consistent filenames,
- ground-truth FEN metadata,
- camera/environment notes,
- a fixed development subset.
Acceptance Criteria
The dataset should be organized enough that code can iterate over it automatically without manual path editing.
You should be able to answer:
- which position an image belongs to,
- its known FEN,
- approximate camera angle,
- whether it belongs to the development subset.
Do Not Work On Yet
Do not annotate individual pieces or train a model.
The Week 1 dataset is primarily for board localization and perspective correction.
Day 4 — First Chessboard Detection Baseline
Objective
Build the first automatic method for finding the chessboard and estimating its four outer corners.
Start with classical computer vision before considering learned board detectors.
Tasks
1. Build a preprocessing pipeline
Experiment with combinations of:
- grayscale conversion,
- Gaussian blur,
- contrast normalization if useful,
- Canny edge detection,
- adaptive thresholding if useful.
Keep preprocessing configurable where sensible.
2. Investigate board geometry
Implement a first candidate-based detector using OpenCV techniques such as:
- contour detection,
- polygon approximation,
- convex quadrilaterals,
- area filtering,
- aspect-ratio checks,
- rectangularity checks.
A likely starting path:
image
↓
grayscale
↓
edge detection
↓
contours
↓
polygon approximation
↓
candidate quadrilaterals
↓
candidate scoring
3. Score board candidates
Do not automatically choose the largest contour without checks.
Useful signals may include:
- area relative to image,
- convexity,
- approximately four corners,
- plausible proportions,
- internal edge/grid structure,
- location within the image.
Keep the candidate-scoring logic explicit and debuggable.
4. Order the detected corners
Convert candidate corner points into the standard order:
TL, TR, BR, BL
Use one tested utility function for this operation.
5. Produce debug visualizations
For every development image, save a debug output showing:
- original image,
- selected quadrilateral,
- corner dots,
- corner labels.
Store outputs in:
outputs/board_detection/
6. Handle failure explicitly
If no reliable board candidate is found:
- return a structured failure,
- do not invent corners,
- print or log an understandable reason where possible.
Deliverables
By the end of Day 4:
- a first automatic chessboard detector,
- ordered corner coordinates,
- debug visualizations,
- explicit handling of failed detections.
Acceptance Criteria
The detector should work on a meaningful portion of the easier development images.
It does not need to reach the final Week 1 target yet.
The important result is a working baseline whose failure modes can be inspected.
Do Not Work On Yet
Do not add a neural-network detector just because some images fail.
First understand why the classical method fails.
Day 5 — Improve and Evaluate Board Detection
Objective
Make board localization reliable enough for controlled project conditions and quantify performance instead of judging it informally.
Tasks
1. Review Day 4 failures
Group failures into categories, for example:
- board border not detected,
- background rectangle mistaken for board,
- incomplete outer contour,
- strong shadows,
- unusual perspective,
- board partially close to image boundary,
- internal squares mistaken for full board.
Document the major failure modes.
2. Improve candidate selection
Based on observed failures, refine:
- contour filtering,
- quadrilateral scoring,
- line detection,
- board-grid evidence,
- corner selection.
If contour-based detection is insufficient, optionally experiment with Hough line detection and line intersections.
Avoid unnecessary complexity if the simpler solution already performs well.
3. Add detection confidence or diagnostics
Where feasible, return useful information such as:
success
corners
candidate score
diagnostic information
This will help later when deciding whether the system should trust a detected frame.
4. Create a reproducible evaluation script
Create an evaluation script that processes the fixed development subset and records:
- image identifier,
- detection success/failure,
- optionally confidence,
- output visualization path.
Manual inspection of whether the selected board is correct is acceptable at this stage.
5. Measure baseline performance
Target approximately:
18 correct detections out of 20
on controlled, representative development images.
If the detector cannot approach this level, determine whether:
- the algorithm needs improvement,
- the image-capture constraints should be slightly tightened,
- a learned detector should be considered later.
Do not hide failures.
Deliverables
By the end of Day 5:
- improved board detector,
- documented failure cases,
- reproducible evaluation,
- measured detection rate.
Acceptance Criteria
For the fixed Week 1 test subset, aim for:
>= 90% correct board localization
under the currently defined controlled conditions.
A lower result does not mean the project fails, but the cause should be understood and documented before progressing.
Day 6 — Perspective Correction and Board Normalization
Objective
Transform the detected chessboard from its camera perspective into a consistent top-down square image.
Tasks
1. Implement perspective transformation
Create:
src/vision/perspective.py
Use the ordered corners:
TL
TR
BR
BL
and OpenCV functions such as:
cv2.getPerspectiveTransform(...)
cv2.warpPerspective(...)
2. Normalize every board
Warp every successful board detection to a fixed resolution.
Recommended default:
800 x 800
The normalized output should always be square.
3. Preserve image quality
Use appropriate interpolation and avoid unnecessary repeated resizing.
The rectified image should retain enough detail for Week 2 piece recognition.
4. Save rectified boards
Write successful outputs to:
outputs/rectified_boards/
Use filenames that map directly back to the original source image.
5. Add visual checks
Inspect whether:
- board edges align with output boundaries,
- files/ranks are approximately straight,
- all 64 squares are visible,
- severe stretching has not occurred,
- pieces remain recognizable.
6. Add tests for corner ordering and transform dimensions
Unit tests should verify:
- corner ordering behaves predictably,
- output dimensions are correct,
- invalid corner inputs raise useful errors.
Deliverables
By the end of Day 6:
- reusable perspective-correction module,
- normalized board images,
- transform tests,
- debug outputs.
Acceptance Criteria
For correctly detected boards, the output should consistently resemble a top-down chessboard with:
straight horizontal files
straight vertical ranks
square overall geometry
The pipeline should work automatically from detected corners without manually adjusting points for each image.
Day 7 — 8×8 Grid Extraction and Week 1 Integration
Objective
Complete the Week 1 image pipeline by dividing each normalized board into 64 deterministic square crops.
Then integrate and test the complete Week 1 flow.
Tasks
1. Implement square extraction
Create:
src/vision/square_extraction.py
For an 800×800 board:
8 rows
8 columns
100×100 pixels per square
Do not hard-code 100 independently if it can be derived from the configured board size.
2. Extract all 64 squares
Return square data in a structured form, for example:
{
    "row": 0,
    "col": 0,
    "image": crop
}
or an equivalent class/data structure.
The output order must always be deterministic.
3. Save square crops for debugging
For each board, optionally save crops to a folder such as:
outputs/square_crops/<image_id>/
Use names such as:
r0_c0.jpg
r0_c1.jpg
...
r7_c7.jpg
Do not call them a8, b8, etc. unless orientation has been explicitly resolved.
4. Create a grid-debug visualization
Create an image showing the rectified board with grid lines and row/column labels overlaid.
This is critical for verifying that square boundaries are correct.
5. Build an end-to-end Week 1 script
Create a script such as:
scripts/process_board_image.py
Input:
raw chessboard image
Pipeline:
load image
↓
detect board corners
↓
order corners
↓
perspective warp
↓
normalize board
↓
extract 64 squares
↓
save debugging outputs
6. Run the complete pipeline on the development subset
For each test image, verify:
- correct board detected,
- correct warp,
- all 64 square crops generated,
- no missing rows or columns,
- no major boundary drift.
7. Record Week 1 metrics
Create an initial evaluation record with fields such as:
image_id
board_detected
board_detection_correct
warp_correct
square_count
notes
A CSV file is sufficient.
8. Update documentation
Update the README with:
- current Week 1 capabilities,
- repository structure,
- how to process a sample image,
- example input/output,
- known limitations.
Keep the limitations honest, e.g.:
- controlled camera angles,
- complete board must be visible,
- no piece recognition yet,
- orientation mapping not yet automatic.
Deliverables
By the end of Day 7:
- 64-square extraction module,
- grid-debug visualization,
- complete image-processing script,
- Week 1 evaluation CSV,
- updated README,
- working Week 1 pipeline.
Final Week 1 Acceptance Test
Given:
data/raw/example.jpg
a single command should produce:
1. board detection visualization
2. perspective-corrected board
3. grid visualization
4. 64 square crops
Conceptually:
Physical-board photo
        ↓
Board localization
        ↓
Four ordered corners
        ↓
Perspective correction
        ↓
800×800 normalized board
        ↓
8×8 grid
        ↓
64 deterministic square crops
The target board-localization success rate on the controlled Week 1 development set is approximately:
>= 90%
Week 1 Definition of Done
Week 1 is complete when all of the following are true:
- [ ] Project repository is clean and modular.
- [ ] Python environment is reproducible.
- [ ] Stockfish works through python-chess.
- [ ] Image-loading and visualization utilities exist.
- [ ] Initial real chessboard dataset has been collected and documented.
- [ ] Chessboard localization works reliably under controlled conditions.
- [ ] Detected corners use a consistent order.
- [ ] Perspective correction produces normalized top-down boards.
- [ ] Every normalized board can be divided into exactly 64 square crops.
- [ ] Debugging outputs make failures easy to inspect.
- [ ] Week 1 evaluation results are recorded.
- [ ] README accurately explains current capabilities and limitations.
Important Scope Boundary for Codex
During Week 1, Codex should optimize for:
- clean architecture,
- correctness,
- testability,
- useful error handling,
- reproducible scripts,
- visual debugging,
- maintainability.
Codex should not implement future functionality early unless explicitly requested.
Do not implement during Week 1:
- chess-piece recognition,
- YOLO,
- CNN training,
- FEN prediction from camera images,
- temporal move detection,
- PGN generation from video,
- FastAPI backend,
- Next.js frontend,
- WebSocket streaming,
- clock detection,
- live game broadcasting.
The Week 1 foundation should be strong enough that Week 2 can add piece recognition without restructuring the entire project.