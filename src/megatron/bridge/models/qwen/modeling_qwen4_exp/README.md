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
  tensor. Query chunks bound the head-score workspace. Unpacked inputs reuse
  their known batch-major document layout after the sequence-parallel gather,
  avoiding scalar reads and dynamic document discovery in the selector. Packed
  inputs retain the generic path. The selected block bitsets preserve causal
  visibility and document boundaries.
- GR keeps the HF activation-dtype division, sigmoid, multiplication, and mean
  boundaries. BF16/FP16 gate helpers remain eager under an outer compile so
  fusion cannot remove the reference rounding points.
- Grouped RMSNorm saves the raw input and one FP32 inverse RMS per stream.
  Eligible contiguous CUDA inputs use norm-only Triton forward and backward
  kernels adapted from [Megatron-LM PR #7553](https://github.com/NVIDIA/Megatron-LM/pull/7553)
  at `53ff95854b024e36ce7b51a61e65d43d3ced8dac`, with attribution and the upstream
  license retained in `norms.py`. Backward computes input gradients and bounded
  FP32 norm-gain partials, then reduces the partials in a fixed order without
  atomics. Other inputs retain the chunked eager fallback. The 64 MiB workspace
  budget excludes required outputs, input gradients and saved statistics.
  Reduction order can differ from eager; higher-order gradients are unsupported.

The grouped norm uses separate RMS statistics and zero-centered gains for
each residual stream while keeping the flat checkpoint parameter layout.
Normalizing the full flattened width would mix statistics across streams;
reshaping streams into the batch dimension would share gains between them.
The local implementation preserves both contracts and bounds reduction memory.

## Supported scope

The builder supports a text decoder with tensor and sequence
parallelism, PP=CP=1, and no virtual pipeline stages. It rejects cached
inference, padding masks, activation recomputation, CUDA graphs, FP8, and MTP.
QSA requires `attention_dropout=0`; nonzero values are rejected at construction
because the sparse backends do not implement attention dropout.
Use THD packing for document boundaries. Full VLM conversion includes the vision
patch embedding, positional embedding, transformer blocks, and patch merger.
The VLM combines visual embeddings before the hybrid decoder while PLE retains
the raw token IDs. MRoPE uses each token's multimodal positions, including the
QSA pooled-block indexer.

When the physical sequence length proves that every causal block fits in the
QSA selection budget, attention skips the indexer and uses the dense backend.
This avoids indexer projections, pooling, selection, and its sequence-parallel
gather while retaining all checkpoint parameters. Packed sequences use a
conservative bound on their total length; `qsa_force_sparse=True` preserves the
explicit selector path. With budget 2048 and compression ratio 4, the shortcut
applies through length 2051. Analytical FLOPs exclude indexer projections when
this shortcut applies.

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
model. Model parity tests use the existing project/container environment through
`uv run --no-sync` and log its Transformers and Torch versions. The launcher
does not install a separate reference environment.

## FLOPs accounting

`Qwen4ExpHybridModel.estimate_flops` and `Qwen4ExpVLModel.estimate_flops`
implement Core's pure classmethod interface: a config plus
`total_real_tokens_in_batch` and `seqlen_squared_sum_in_batch`, returning
forward FLOPs without allocating weights. The VLM also accepts additive
`patch_sum`, `patch_squared_sum` and `merged_token_sum` statistics.

The model-config `estimate_flops` and `estimate_vision_flops` methods expose
these estimates to Bridge's training logger. Trainable matrix work uses a
three-times-forward training estimate; the detached QSA indexer is counted
forward-only. Standard attention, GDN and MoE reuse Core's block estimators;
gated residual and PLE projections have local hooks. Counts describe global
logical work, independent of TP/EP replication.

QSA currently uses a dense-attention surrogate and omits indexer score work;
GDN recurrence is approximate. Norms, lookup tables, PLE convolution,
communication and optimizer work are excluded. Historical verification-card
measurements retain the estimator used by the recorded run.

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

The H100 and GB200 L2 launch scripts both call `run_qwen4_exp_parity.sh` and
run these GPU cases with at most two ranks in the existing project/container
environment. Each distributed case has a ten-minute timeout. L2
requires the `full-test-suite` label or an explicit L2 workflow dispatch.

## Sources

The conversion mappings and base QSA/PLE/GR scaffolding are adapted from
Casper Hansen's NVIDIA-NeMo/Megatron-Bridge#6123 and NVIDIA/Megatron-LM#7393.
The local builder, model hooks, document-local scoring, bounded normalization,
and gate-rounding changes are maintained in this branch. The gate-rounding
change carries the earlier ModelScope contribution forward. Apache 2.0
copyright and license notices are retained.
