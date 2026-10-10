# Shensi

User-facing documentation lives under `docs/models/shensi/` (index page plus the variant page);
this README covers the implementation.

## Layout

```
shensi/
├── configuration_shensi.py   HF-side config: the per-layer plans, the RoPE split, validation
├── modeling_shensi.py        the components Shensi adds on top of Megatron-Core
├── shensi_bridge.py          HF <-> Megatron: config mapping and the parameter mappings
├── shensi_hybrid.py          HybridModel glue: pipeline layer pattern + hybrid stack spec
├── shensi_builder.py         model-config / model-builder path (HybridModelConfig + builder)
├── shensi_provider.py        provider: config fields -> ShensiModel, via build_shensi_model
├── __init__.py
└── README.md
```

`configuration_shensi.py` holds both config surfaces: `ShensiConfig` mirrors the fields a
checkpoint ships, `ShensiTransformerConfig` is the Megatron-Core transformer config that carries
the Shensi-only fields next to the MLA ones.

## What Shensi adds

**Hybrid sparse attention** comes from Megatron-Core: the provider sets
`experimental_attention_variant="dsv4_hybrid"` together with the per-layer `compress_ratios`, so
CSA / HCA / sliding-window layers, their compressors and the Lightning-Indexer style router are the
core implementations. This package adds what sits around them:

* **mHC hyper-connections** (`ShensiHyperConnection`) — the residual stream is a pool of
  `hc_mult` streams. A sublayer collapses the pool into its input and writes its output back into
  a fixed plus a top-k routed subset of streams; the MLP side augments the write with causal depth
  convolutions and their Gram-Schmidt orthogonalizations. `ShensiHyperHead` collapses the pool at
  the end of the stack.
* **Attention residuals** (`ShensiAttentionResidual`) — layers are grouped into blocks by the
  hash-MoE prefix and `attn_res_block_size`. A block-writing layer publishes its stream context and
  the later layers of the block read it. The history is per stream and per block, matching the
  reference's `[s, b, hc, nb, H]` buffer; `ShensiAttnResState` stores one tensor per block
  (`state.blocks`) and re-assembles that layout only for the pipeline payload and tooling, so a
  block boundary costs no buffer copy. When a block spans a pipeline stage boundary, the state
  rides the stage payload itself (`pack_attn_res_payload` / `unpack_attn_res_payload`): the
  schedule ships whatever the model returns, so no side communicator and no extra process group are
  involved, and the state's gradients travel back through the same tensor. The payload then carries
  the state per token and per stream (`hc_mult x (1 + num_blocks) x hidden` on top of the
  activations), so the schedule has to exchange shapes per microbatch
  (`variable_seq_lengths: true`) for a crossing split to run, which the layout check enforces. For
  pipeline runs, align the stage split with the attention-residual block boundaries — then nothing
  has to cross; the model logs the crossing blocks whenever a split does not line up.
* **MoE layout** (`ShensiMoELayer`, `ShensiHashMLP`) — the leading layers are dense SwiGLU MLPs
  gated by a token-embedding lookup rather than routed experts, and the routed experts of the
  remaining layers run through a low-rank bottleneck whose norm is applied before the up
  projection. The ERC auxiliary loss needs a global view of the expert weights, which
  `erc_gather_expert_weights` builds across the expert-parallel group.
* **Selective fp32 keeping** — the mHC mixers, the attention-residual readers and the strict list
  of norms / sinks / position biases stay in fp32 while the rest of the model trains in bf16,
  matching the `_keep_in_fp32_modules_strict` list of the HF checkpoint.

## Field mapping

The provider derives the Megatron-Core fields from a `ShensiConfig` through
`shensi_derived_field_values`: `hc_mult → num_residual_streams`, `sliding_window → csa_window_size`,
`compress_rope_theta → csa_compress_rotary_base`, `index_* → dsa_indexer_*`,
`moe_intermediate_size → moe_ffn_hidden_size`, `routed_expert_hidden_size → moe_latent_size`,
`o_groups / o_lora_rank → output_projection_groups / output_projection_lora_rank`. The HF config's
`num_key_value_heads = 1` is expressed through `kv_lora_rank`, so `num_query_groups` follows the
attention head count. Fields without a Megatron-Core counterpart are listed in
`SHENSI_ONLY_FIELDS`; `describe_mapping()` prints the result for a built config.

## Usage

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("<hf checkpoint>")
provider = bridge.to_megatron_provider(load_weights=False)
model = provider.provide_distributed_model(wrap_with_ddp=False)
bridge.load_hf_weights(model)
```

## Inference

Megatron-Core's hybrid sparse attention has no inference path (`DSv4HybridAttention` rejects
inference contexts), so `examples/conversion/hf_to_megatron_generate_text.py` can convert and load
a Shensi checkpoint but cannot generate with it. Serving is done through vLLM, which ships the
attention and compressor kernels for this family:

```sh
vllm serve <exported hf directory> --served-model-name shensi --enforce-eager
```

Exporting back to HuggingFace works (`bridge.save_hf_pretrained(model, path)`), which is what the
functional round-trip test covers.

## Tests

```sh
uv run python -m pytest -q tests/unit_tests/models/shensi --noconftest --import-mode=importlib
uv run python -m pytest -q tests/functional_tests/test_groups/models/shensi
```

The functional tests build a tiny Shensi checkpoint (`make_tiny_hf_model`) and cover the round trip
(HF → Megatron → HF), the forward pass, right-padding behaviour and the provider/MTP configuration.
The unit tests compare the attention-residual read bit-for-bit against the HF reference, pin the
per-slot block layout to that reference (same draws, same rows, same gradients) and exercise the
state carriage across pipeline stages (pack/unpack round trip, gradient path, per-stage conditions).
