# GLM-5.3-Flash

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) (HF `glm5_next`,
`Glm5NextForConditionalGeneration`) is a **different architecture** from GLM-5 / GLM-5.2 /
GLM-5.3 and uses its own `Glm5NextBridge` rather than `GLM5Bridge`.

## Architecture

| | |
|---|---|
| Layers | 45, in a 3:1 pattern of KDA linear attention and NoPE-MLA sparse attention |
| Linear attention | Kimi Delta Attention ([arXiv:2510.26692](https://arxiv.org/abs/2510.26692)), bounded forget gate |
| Full attention | NoPE MLA (`qk_pos_emb_head_dim = 0`) with a DeepSeek-sparse-attention k-pool indexer |
| MLP | 288-expert sigmoid-routed MoE with one shared expert; dense MLP in the first layer |
| Activation | clamped SwiGLU (`swiglu_limit`) |
| Residuals | Manifold-Constrained Hyper-Connections (mHC) on every block, 4 streams |
| Checkpoint | shipped as a VL checkpoint; only the language model is bridged |

## Usage

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-5.3-Flash")
provider = bridge.to_megatron_provider(load_weights=False)
```

The vision tower has no Megatron counterpart, so language-model-only loading is required.

## Requirements

- megatron-core with `experimental_attention_variant="kda"` and the mHC
  `mhc_norm_eps` / `mhc_norm_eps_inside_sqrt` config fields.
- `flash-linear-attention` providing `fla.ops.kda` (KDA kernels) — imported lazily, so
  `AutoBridge` import does not require it.

## Parallelism

- **PP = 1.** megatron-core rejects mHC with `pipeline_model_parallel_size > 1`.
- **CP = 1.** KDA has no context-parallel path.
- TP shards KDA heads and the MLA/MoE weights as usual; EP is the scaling dimension for the
  routed experts, which hold the large majority of the parameters.
- `recompute_granularity="full"` is rejected with mHC; use selective recompute.

## Limitations

- **k-pool indexer:** not implemented. The Megatron path reuses megatron-core's token-level
  indexer with `dsa_indexer_topk = index_topk`, which is exact for sequences of at most
  `index_topk` (2048) tokens because every pool is then selectable. Longer sequences raise
  rather than attend to a different subset.
- **Cross-layer DSA index sharing** (`indexer_types` containing `"shared"`) raises: megatron-core's
  `dsa_indexer_topk_freq` arithmetic runs over all layers and would pick KDA layers as index
  sources in this hybrid.
- KDA has no inference cache.

## Verification

No verification cards have been recorded for this model yet. What has been measured, on the
truncated 4-layer slice [`eatang/GLM-5.3-Flash-4layer`](https://huggingface.co/eatang/GLM-5.3-Flash-4layer):

- Megatron vs. HF `transformers` logits, 82-token packed input, bf16: KL 0.0082 / 91.5% argmax
  agreement at TP1, against a measured HF-bf16-vs-fp32 floor of 0.0071 / 95.7% on the same slice.
- vLLM ↔ Megatron logprob round trip on 4×H100 at TP2/EP4: mean absolute logprob difference 0.064.

Both were measured with this bridge's modules carried out of tree; see
[NovaSky-AI/SkyRL#2179](https://github.com/NovaSky-AI/SkyRL/pull/2179). Full-checkpoint
conversion, export, training and convergence are unverified.
