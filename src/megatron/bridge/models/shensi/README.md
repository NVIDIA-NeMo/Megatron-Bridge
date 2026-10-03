# Shensi

Shensi is a sparse-attention Mixture-of-Experts language model. Each decoder layer picks one of
three attention paths from a per-layer plan — compressed sparse attention (CSA, compress ratio 4),
heavily compressed attention (HCA, 128) or a sliding window — and the residual stream is not a
single tensor but a pool of streams that the sublayers read from and write back into.

## Layout

```
shensi/
├── configuration_shensi.py   HF-side config: the per-layer plans, the RoPE split, validation
├── modeling_shensi.py        the components Shensi adds on top of Megatron-Core
├── shensi_bridge.py          HF <-> Megatron: config mapping and the parameter mappings
├── shensi_provider.py        provider: config fields -> ShensiModel; builds the layer specs
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
  the later layers of the block read it; the state also travels across pipeline stages on its own
  communication group, so a block may span stage boundaries.
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
