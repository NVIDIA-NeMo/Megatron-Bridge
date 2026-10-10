# Qwen4-Exp hybrid language and vision-language models

The model components live in Bridge and use the existing Megatron-Core HybridModel,
GDN, attention, MoE, and hyper-connection interfaces. They do not require the
model-specific changes to the pinned Core revision.

`Qwen4ExpModelConfig` and its nested transformer config declare the checkpoint's
QSA, gated residual, and PLE fields. `Qwen4ExpModelBuilder` builds an actual Core
HybridModel/HybridStack with complete attention-and-MLP blocks. The model prepares
raw token IDs for PLE and expands residual
streams before the decoder. The final output mixer contracts those streams with
learned gates; Core's generic mHC expansion and mean contraction stay disabled.

## Optimizations

- QSA scores document-local block keys without a query-sized replicated key
  tensor. Query chunks bound the head-score workspace. The selected block
  bitsets retain causal visibility and document boundaries.
- GR keeps the HF activation-dtype division, sigmoid, multiplication, and mean
  boundaries. BF16/FP16 gate helpers remain eager under an outer compile so
  fusion cannot remove the reference rounding points.
- Grouped RMSNorm saves the raw input and one inverse RMS per stream. Its
  forward and first-order backward work in chunks, including the FP32 norm-gain
  reduction. The 64 MiB chunk budget excludes required outputs, input gradients,
  and saved statistics. A single row is the minimum chunk. Chunking can change
  FP32 reduction order; it does not provide higher-order gradients.

The grouped norm uses separate RMS statistics and zero-centered gains for
each residual stream while keeping the flat checkpoint parameter layout.
Normalizing the full flattened width would mix statistics across streams;
reshaping streams into the batch dimension would share gains between them.
The local implementation preserves both contracts and bounds reduction memory.

## Supported scope

The builder supports a text decoder with tensor and sequence
parallelism, PP=CP=1, and no virtual pipeline stages. It rejects cached
inference, padding masks, activation recomputation, CUDA graphs, FP8, and MTP.
Use THD packing for document boundaries. Full VLM conversion includes the vision
patch embedding, positional embedding, transformer blocks, and patch merger.
The VLM combines visual embeddings before the hybrid decoder while PLE retains
the raw token IDs. MRoPE uses each token's multimodal positions, including the
QSA pooled-block indexer.

Use `AutoBridge.from_hf_pretrained(path).get_model_config()` to configure the
model and `bridge.get_model(config, wrap_with_ddp=False)` to construct/load it.
For a language-only view of a VLM checkpoint, pass `text_only=True` to
`AutoBridge.from_hf_pretrained`. This filters vision weights and projects
`model.language_model.*` to the standalone `model.*` namespace, retaining the
root LM head; export produces a standalone text checkpoint. Native text
checkpoints continue to use their existing namespace. Construction uses only
Builder + Config; there is no separate legacy provider. When selecting a different
VLM checkpoint for text-only loading, construct a new `AutoBridge` with
`text_only=True`; the builder's replacement `hf_path` argument does not project
a VLM checkpoint.

The same public entrypoint selects either composition:
`AutoBridge.from_hf_pretrained(path, text_only=True)` builds the standalone
`Qwen4ExpHybridModel`; `text_only=False` builds `Qwen4ExpVLModel`, whose
`language_model` is that same `Qwen4ExpHybridModel`. The VLM builder extends the
language builder and delegates decoder construction to it. QSA, GDN, MoE, gated
residuals, PLE and the output mixer have one implementation shared by both modes.
The VLM reuses the existing Qwen3-VL vision encoder, attention, multimodal RoPE,
input-reorganization helpers and Qwen3.5-VL vision mappings.

Loading a Qwen4-Exp HF model requires a Transformers release that includes that
model. Model parity tests use Transformers 5.16.1 in an isolated environment;
the project dependency constraints are unchanged by this implementation.

## Validation

Run the focused numerical, mapping, and GPU conversion tests:

```bash
uv run python -m pytest tests/unit_tests/models/qwen/test_qwen4_exp_numerics.py
uv run python -m pytest tests/unit_tests/models/qwen/test_qwen4_exp_bridge.py
uv run python -m pytest tests/unit_tests/models/qwen/test_qwen4_exp_vlm.py
uv run python -m pytest tests/functional_tests/test_groups/models/qwen/test_qwen4_exp_conversion.py
```

The functional test uses a small HF checkpoint and checks TP1/TP2 logits,
packed inputs, embedding gradients, and HF export. Sparse-kernel acceptance
also requires a sequence beyond the indexer budget, rather than only the dense
shortcut. Component tests alone do not establish full-model training parity or
an end-to-end performance improvement.

The functional suite also includes a one-rank toy VLM/text-only import,
forward/backward, and export smoke. It checks finite language, PLE, GDN, gated
residual and vision gradients. The smoke uses the dense masked QSA reference
backend, so it does not validate sparse-kernel performance. It checks a
cross-entropy backward pass for finite gradients in each listed group and at
least one nonzero gradient in each group; it does not compare HF logits, loss or
gradients. Exact export checks parameter conversion, not execution equivalence.
The separate text-only parity test above does compare HF logits/loss and word
embedding gradients; these two test results must be reported separately.

The H100 and GB200 L2 launch scripts run these GPU cases with at most two ranks. They install Transformers 5.16.1 in a temporary reference
environment and reuse the container's CUDA packages, leaving the project
environment unchanged. Each distributed case has a ten-minute timeout. L2
requires the `full-test-suite` label or an explicit L2 workflow dispatch.

## Sources

The conversion mappings and base QSA/PLE/GR scaffolding are adapted from
Casper Hansen's NVIDIA-NeMo/Megatron-Bridge#6123 and NVIDIA/Megatron-LM#7393.
The local builder, model hooks, document-local scoring, bounded normalization,
and gate-rounding changes are maintained in this branch. The gate-rounding
change carries the earlier ModelScope contribution forward. Apache 2.0
copyright and license notices are retained.
