# Day 12: held-out classifier evaluation

The validation-selected MobileNetV3 Small checkpoint (epoch 16) was evaluated on
1,472 context crops from 23 boards covering three unseen positions: 006, 010 and
016. The model received RGB plus the target-square mask, using the same 224-pixel,
deterministic preprocessing as validation. No augmentation, training, calibration
fitting or checkpoint selection was performed on test examples.

## Results

| Metric | Test result |
| --- | ---: |
| Overall square accuracy | 94.63% (1,393/1,472) |
| Occupied-square accuracy | 82.68% (377/456) |
| Macro F1, all 13 classes | 79.68% |
| Empty-square accuracy | 100% (1,016/1,016) |
| Incorrect squares | 79 |

The 90% square-level target is met, but empty squares dominate this score. These
are crop-classification results, not raw-photo or exact-board/FEN accuracy. Only
successfully localized and reviewed boards enter this evaluation. Shared equipment,
lighting protocols and just three test positions limit generalization claims;
multiple photos of the same position are correlated observations.

| Class | Test samples | Recall (per-class accuracy) | F1 |
| --- | ---: | ---: | ---: |
| empty | 1016 | 100.00% | 100.00% |
| white pawn | 114 | 92.11% | 90.13% |
| white knight | 26 | 69.23% | 65.45% |
| white bishop | 26 | 76.92% | 83.33% |
| white rook | 26 | 80.77% | 89.36% |
| white queen | 13 | 92.31% | 88.89% |
| white king | 23 | 95.65% | 95.65% |
| black pawn | 104 | 92.31% | 92.31% |
| black knight | 26 | 92.31% | 77.42% |
| black bishop | 26 | 73.08% | 77.55% |
| black rook | 36 | 50.00% | 61.02% |
| black queen | 13 | 76.92% | 62.50% |
| black king | 23 | 52.17% | 52.17% |

## Error inspection

All 79 errors were displayed in four contact sheets, sorted by confidence. Each
shows the original context crop, green target-square polygon, crop ID, expected
class, predicted class and confidence. A separate sheet shows the 20 lowest
confidence predictions, including correct ones. A further sheet supplies correct
high-confidence examples for comparison.

The most common confusions are black rook → king (10), white pawn → knight (9),
black king → queen (9), white knight → pawn (8), and black pawn → knight (6).
No piece was classified as empty, no empty square as a piece, and no piece-color
confusion occurred in this test set.

Visual inspection shows several contributing conditions: neighboring pieces
occupy the same context window; oblique views overlap bodies and bases; dark
pieces have weak contrast against brown squares; overhead views hide distinctive
silhouettes. These are plausible contributors, not independently established
causes. Errors occur across all ten variations, rather than being confined to one
orientation. Recurring errors include position 016's a1 black king → queen in
nine photos and position 010's c3 pawn → knight and g3 knight → pawn in seven
photos each. This repetition warrants checking target association and physical
piece identity against source photos before any later dataset changes; it does
not establish that the existing labels are wrong.

## Confidence

Among 1,367 predictions with confidence at least 90%, 31 are wrong; accuracy in
that group is 97.73%. Predictions with confidence 80–90% are correct only 60.87%
of the time (14/23). The ten-bin expected calibration error is 2.63 percentage
points, but this aggregate also benefits from the many easy empty squares.
Softmax confidence is useful for review priorities and is not a guarantee.
No confidence threshold or temperature was tuned against these test results.

## Baseline decision

Keep the epoch-16 checkpoint as the measured baseline and proceed to integration.
More epochs alone are not justified by this evaluation. A focused future experiment
could add diverse training-position examples of dark rooks/kings/queens and
pawn/knight target association, retaining the current context and mask contract.
Evaluate any change using validation data, changing one variable at a time. Since
this test split has now been inspected, a future final claim after test-informed
changes should use newly reserved positions. No model or dataset was altered here.

## Reproduce and review

```bash
.venv/bin/python scripts/evaluate_classifier.py
.venv/bin/python -m pytest tests/test_classifier_evaluation.py tests/test_training.py -q
```

The evaluator verifies checkpoint label/input compatibility, manifest SHA, dataset
fingerprint and actual test crop hashes. It loads the saved model without downloading
pretrained weights. Batch inference uses the configured CPU threads and batch size.

Artifacts in `outputs/week2/classifier_evaluation/`:

- `evaluation.json`: metrics, confidence bins, confusion pairs and provenance.
- `predictions.json`: every test crop's prediction and all 13 probabilities.
- `per_class.csv`: class support, precision, recall and F1.
- `confusion_matrix.csv` and `confusion_matrix.png`: rows are truth, columns prediction.
- `errors_01.png` through `errors_04.png`: all 79 misclassifications.
- `lowest_confidence.png` and `correct_high_confidence.png`: confidence comparisons.

The compact, versioned report is `data/metadata/day12_classifier_evaluation.json`.
It records checkpoint SHA, dataset version, training settings, validation results,
and test results together. Large generated outputs remain ignored by Git.
