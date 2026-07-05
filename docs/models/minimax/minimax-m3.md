# MiniMax-M3

[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) is a natively multimodal sparse MoE model from MiniMaxAI (428B total, ~23B active parameters). Megatron Bridge supports the M3 *language model* through the `MiniMaxM3Bridge`: the text backbone of the `MiniMaxM3SparseForConditionalGeneration` checkpoint is converted to a Megatron-Core `GPTModel`.

## Supported Variants

| Variant | Hugging Face ID | Notes |
|---------|-----------------|-------|
| MiniMax-M3 | `MiniMaxAI/MiniMax-M3` | Language model only (bf16 weights) |

## Architecture Notes

- Mixed dense/MoE decoder: 60 layers, the first 3 dense, the rest with 128 routed experts (top-4) plus one shared expert.
- Sigmoid router with expert-bias correction and `routed_scaling_factor` applied to the normalized top-k weights (DeepSeek-V3-style routing).
- SwiGLU-OAI activation in every MLP and expert: clamped gate/up projections with a `+1` linear offset, mapped to `activation_func_clamp_value` / `glu_linear_offset` (same mechanism as GPT-OSS).
- Gemma-style RMSNorm (`x * (1 + w)`) on every norm, mapped to `layernorm_zero_centered_gamma`.
- GQA attention (64 query heads, 4 KV heads) with per-head QK RMSNorm and partial RoPE (64 of 128 head channels rotated, theta 5e6).

## Known Limitations

- **Language model only.** The CLIP-style vision tower, multimodal projector, and patch-merge MLP are not mapped.
- **Full attention only.** The lightning-indexer block-sparse attention branch (`self_attn.index_*` weights) is not mapped; every layer runs full causal attention. Block selection keeps `index_topk_blocks * index_block_size` (2048) key tokens per query, so full attention is mathematically identical up to that sequence length and an approximation beyond it. Exported checkpoints declare `full_attention` on every layer.
- **MTP modules are not mapped.** The released checkpoint advertises `num_nextn_predict_layers` in its config but ships no `mtp.*` weights.
- The MXFP8 variant (`MiniMaxAI/MiniMax-M3-MXFP8`) is not supported; use the bf16 checkpoint.

## Conversion

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("MiniMaxAI/MiniMax-M3")
provider = bridge.to_megatron_provider()
```

## Recipes

Pretraining and SFT recipes are available under [`src/megatron/bridge/recipes/minimax`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/recipes/minimax) (`minimax_m3_pretrain_256gpu_h100_bf16_config`, `minimax_m3_sft_128gpu_h100_bf16_config`), using a TP=2 / PP=4 / EP=32 baseline layout.

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/minimax_m3`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/minimax_m3)
