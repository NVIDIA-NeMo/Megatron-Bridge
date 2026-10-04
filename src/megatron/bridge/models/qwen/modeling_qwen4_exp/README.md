# Qwen4-Exp language model

The model components live in Bridge and use the existing Megatron-Core GPT,
GDN, attention, MoE, and hyper-connection interfaces. They do not require the
model-specific changes to the pinned Core revision.

`Qwen4ExpModelProvider` declares the checkpoint's QSA, gated residual, and PLE
fields. Its GPT subclass prepares raw token IDs for PLE and expands residual
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

The initial provider supports a text decoder with tensor and sequence
parallelism, PP=CP=1, and no virtual pipeline stages. It rejects cached
inference, padding masks, activation recomputation, CUDA graphs, FP8, and MTP.
Use THD packing for document boundaries. Vision parameters are not converted.

Loading a Qwen4-Exp HF model requires a Transformers release that includes that
model. Model parity tests use Transformers 5.16.1 in an isolated environment;
the project dependency constraints are unchanged by this implementation.

## Validation

Run the focused numerical, mapping, and GPU conversion tests:

```bash
uv run python -m pytest tests/unit_tests/models/qwen/test_qwen4_exp_numerics.py
uv run python -m pytest tests/unit_tests/models/qwen/test_qwen4_exp_bridge.py
uv run python -m pytest tests/functional_tests/test_groups/models/qwen/test_qwen4_exp_conversion.py
```

The functional test uses a small HF checkpoint and checks TP1/TP2 logits,
packed inputs, embedding gradients, and HF export. Sparse-kernel acceptance
also requires a sequence beyond the indexer budget, rather than only the dense
shortcut. Component tests alone do not establish full-model training parity or
an end-to-end performance improvement.

## Sources

The conversion mappings and base QSA/PLE/GR scaffolding are adapted from
Casper Hansen's NVIDIA-NeMo/Megatron-Bridge#6123 and NVIDIA/Megatron-LM#7393.
The local provider, model hooks, document-local scoring, bounded normalization,
and gate-rounding changes are maintained in this branch. The gate-rounding
change carries the earlier ModelScope contribution forward. Apache 2.0
copyright and license notices are retained.
