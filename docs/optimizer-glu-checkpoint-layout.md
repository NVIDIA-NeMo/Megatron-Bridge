# Shared GLU checkpoint layout

This change requires the matching Core GLU checkpoint factory and metadata API.
Use both changes together: Core owns row conversion for model weights, FP32
masters and optimizer moments; Bridge records and validates the shared metadata
and passes Core's loaded tensors directly to the model. The former Bridge
tensor-only conversion helpers are removed.

Native `torch_dist` model tensors and `fully_reshardable` optimizer tensors use
canonical gate/up rows. To retain optimizer state across an interleave-size or
TP/ETP change, save with
`cfg.checkpoint.dist_ckpt_optim_fully_reshardable = True` first. Raw optimizer
buffers still require matching interleave and TP/ETP settings. Model-only loading
from a marked canonical checkpoint can change layout.

Both training resume and direct native weight loading validate layout before
reading tensor storage. Unmarked historical interleaved checkpoints are rejected
because their storage order cannot be inferred safely from the saved runtime
configuration. Recover with the original writer and verify the weights, masters
and moments before converting and annotating them. Known noninterleaved
checkpoints retain compatibility. There is no checkpoint-version bump.

Use a layout-aware Core reader for new interleaved model-space optimizer
checkpoints; older readers may ignore the metadata and misinterpret moments.
Interleaved native checkpoints require `torch_dist`; FSDP/DTensor and GTP layout
conversion is outside this change. The two-GPU Core tests exercise BF16 Adam,
both gather modes and subsequent updates. FP8/FP4 fused kernel execution and
large-scale checkpoint memory have not been validated by those tests.

If the original Hugging Face checkpoint is available and its weights are the ones
you want to initialize from, import it again with the updated AutoBridge writer.
This creates a marked model checkpoint. It starts a new optimizer and cannot
recover later training updates. For a trained checkpoint, keep the original
reader available for full resume until model rows, master weights and both Adam
moments can all be verified. Adding a model layout marker alone cannot establish
the optimizer layout.
