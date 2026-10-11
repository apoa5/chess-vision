# Day 11: first MobileNetV3 Small training run

The local baseline trains on approved Day 10 context crops, with RGB plus a binary
target-square mask. The schedule uses up to 20 epochs: five epochs training the
classification head and input stem at learning rate 0.001, followed by up to 15
fine-tuning epochs at 0.0001. Batch size is 16 with four CPU threads. No test-set
images are loaded by the training script.

```bash
.venv/bin/python scripts/train_classifier.py
```

The script validates after every epoch and reports weighted loss, accuracy, macro
F1 and per-class metrics. It saves the lowest-validation-loss model, and stops after
five fine-tuning epochs without a loss improvement of at least 0.0001. Patience
resets at the transition to fine-tuning. Augmentation is reproducible by seed,
epoch and crop ID; validation preprocessing is deterministic. A changed dataset
manifest aborts the run instead of silently mixing dataset versions.

Artifacts:

- `models/checkpoints/piece_classifier_best.pt`: model weights, input/label contract,
  best epoch, validation metrics, settings and dataset fingerprint.
- `outputs/week2/training/status.json`: current status and completed-epoch history.
- `outputs/week2/training/training.log`: console progress when redirected by Codex.
- `data/metadata/day11_training_summary.json`: compact training results after completion.

```bash
tail -f outputs/week2/training/training.log
```

An existing best checkpoint is protected against accidental overwrite. This first
script starts a fresh run; it does not resume optimizer state from a stopped run.
If interrupted after an epoch, the best completed checkpoint remains available.
Checkpoint writes and status updates are atomic. Completing training does not
establish held-out test accuracy: Day 12 evaluates the selected model separately.

Validation accuracy should be read alongside macro F1 and per-class scores because
empty squares dominate the dataset. The initial CPU benchmark projected about
65–70 minutes for 20 epochs; early stopping can shorten the run, and sustained CPU
load can change actual duration.

The first run completed all 20 epochs in approximately 70.8 minutes. Epoch 16 had
the lowest validation loss (0.2335) and is saved as the best checkpoint. Its
validation accuracy was 97.0%, with macro F1 of 88.75%. Training accuracy at that
epoch was 99.51%. The checkpoint was reloaded successfully and produced finite
13-class outputs from the four-channel input.

Per-class validation recall exposes remaining weaknesses: black queens scored
60% (9/15), and white kings scored 72% (18/25). These small validation samples
limit certainty, and high square accuracy does not establish whole-board accuracy.
The held-out test split was unused during training. Its subsequent evaluation
and confusion analysis are documented in [Day 12](day12_evaluation.md).
