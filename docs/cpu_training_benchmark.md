# Local MobileNetV3 Small timing trial

The CPU trial used pretrained MobileNetV3 Small on actual training context crops,
four input channels (RGB plus target polygon mask), 224×224 inputs, Adam, and the
Day 10 training-only class weights. It used four CPU threads and single-process
loading. Each case excluded two warmup training batches and measured eight more;
validation timing excluded one warmup batch and measured four batches. The held-out
test set was untouched. Trial weights were discarded; no checkpoint was saved.

| Training mode | Batch size | Seconds/batch | Estimated epoch including validation |
| --- | ---: | ---: | ---: |
| Head and input stem | 8 | 0.253 | 4.1 min |
| Head and input stem | 16 | 0.401 | 3.3 min |
| Full fine-tuning | 8 | 0.208 | 3.4 min |
| Full fine-tuning | 16 | 0.416 | 3.4 min |

Peak process memory was approximately 795 MiB across the trial. These measurements
support local training at batch size 16; the configuration now uses that batch size.
Ten epochs project to roughly 33–41 minutes depending on training mode/batch size.
These are short-run extrapolations, excluding startup and checkpoint writing.
Sustained heat, background workloads and disk conditions can change the actual time.
Timing variations mean this is not evidence that full fine-tuning is inherently
faster than a partially frozen model. No recognition-accuracy claim is made.

The model retains pretrained RGB input weights and initializes its added mask
channel to zero. The input stem and classification head train in the initial
transfer-learning stage. Frozen backbone blocks keep their pretrained batch-norm
statistics, but gradients still pass through them to reach the input stem.
The benchmark and unit test verify a nonzero gradient for the mask channel.
Full fine-tuning unfreezes all parameters.

```bash
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python scripts/benchmark_classifier.py
.venv/bin/python -m pytest tests/test_classifier.py -q
```

The first run downloads pretrained weights into the ignored `models/cache/` folder.
Detailed timing is in `outputs/week2/cpu_training_benchmark.json`; the compact,
versioned report is `data/metadata/day11_cpu_benchmark.json`, including the dataset
fingerprint. Model construction follows the
[official Torchvision MobileNetV3 Small API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html).

This trial prepares the baseline model and measures local feasibility. The full
Day 11 training loop, validation metrics and best checkpoint remain to be completed.
