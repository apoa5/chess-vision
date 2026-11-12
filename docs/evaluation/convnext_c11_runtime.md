# C.11 offline runtime boundary

This is an explicit offline interface. Nothing in the browser, server, geometry
or production recognition profile selects it automatically. The recommended
configuration is generated only after development parity checks. Keep one loaded
runtime alive per bounded inference worker, rather than loading it per square or
request. Set its process thread budget before other PyTorch parallel work; do not
change it concurrently from request handlers.

```python
import json
from src.training.multidataset import SQUARES
from src.vision.convnext_runtime import load_runtime, prepare_board_inputs

config = json.load(open('configs/evaluation/convnext_c11_recommended_v1.json'))
runtime = load_runtime(config)
# packed: unchanged B.2 context1_224 uint8 [64,4,224,224], canonical a8..h1.
# RGB is 0..255; the target-mask channel is already binary 0/1.
inputs = prepare_board_inputs(packed)
logits = runtime.predict_logits(inputs, squares=SQUARES)
classes = logits.argmax(axis=1)
```

`prepare_board_inputs` calls the original B.2 normalization function and only
changes allocation strategy. It performs no geometry, pixel rotation, resizing,
calibration or chess repair. The caller must explicitly canonicalize the original
crop observations, preserving camera-relative RGB. The runtime validates input
shape/dtype, all 64 square names, finite values and binary masks. Returned logits
are float32 `[64,13]` in `src/chess/labels.py` order; the runtime does not discard
non-winning classes or infer full-FEN game state.

Backends are explicit: `pytorch_reference`, `pytorch_channels_last`, `onnx_cpu`.
Each checks the frozen source identity. PyTorch verifies the original checkpoint
bytes before strict state loading. ONNX verifies both its graph and metadata
hashes plus the original source-checkpoint identity, input/output contract and
class/profile metadata. A missing library, missing/mismatched artifact or invalid
input raises an error. **There is no silent checkpoint or backend fallback.**
Application integration must handle that error visibly; changing to a different
backend/profile requires an explicit configuration decision.

Optional dependencies live in `requirements-inference-c11.txt`. C.11 uses a
separate virtual environment under ignored `outputs/evaluation/convnext_c11_env`;
it reuses the existing CPU PyTorch/timm installation and adds ONNX packages there.
No CUDA runtime or model downloads are needed. An ONNX-only deployment needs the
CPU runtime plus the existing image-preparation dependencies, not PyTorch or the
exporter at request time. Keep the graph beside its metadata and pinned config;
do not silently regenerate it on startup.

Future assisted integration should preserve the existing image acquisition,
localization, orientation and B.2 packing code, select an explicit ConvNeXt
profile, return the canonical draft plus raw uncertainty metadata, and open the
existing editable review flow. Require review of every square and explicit user
confirmation. Full FEN needs explicit side-to-move, castling, en-passant and move
counters; the classifier cannot establish them. Neither a fast backend nor a high
softmax score verifies a board. Keep the current production baseline intact.

The inherited pretrained-weight restriction is **CC BY-NC 4.0**. ONNX export does
not grant commercial deployment rights; resolve that separately before commercial
use. This interface alone is not MVP integration or a scan-completion gate.
