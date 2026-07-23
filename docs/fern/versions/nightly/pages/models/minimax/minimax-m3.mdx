# MiniMax-M3

[MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) is a natively multimodal sparse MoE model from MiniMaxAI (428B total, ~23B active parameters). `MiniMaxM3Bridge` converts the vision tower, both multimodal projector stages, and sparse-MoE text backbone into a `MiniMaxM3VLModel`.

## Supported Variants

| Variant | Hugging Face ID | Notes |
|---------|-----------------|-------|
| MiniMax-M3 | `MiniMaxAI/MiniMax-M3` | VLM import and HF export (bf16 weights) |

## Architecture Notes

- Mixed dense/MoE decoder: 60 layers, the first 3 dense, the rest with 128 routed experts (top-4) plus one shared expert.
- Sigmoid router with expert-bias correction and `routed_scaling_factor` applied to the normalized top-k weights (DeepSeek-V3-style routing); the checkpoint's FP32 router weights remain FP32 during import.
- SwiGLU-OAI activation in every MLP and expert: clamped gate/up projections with a `+1` linear offset, mapped to `activation_func_clamp_value` / `glu_linear_offset` (same mechanism as GPT-OSS).
- Gemma-style RMSNorm (`x * (1 + w)`) on every norm, mapped to `layernorm_zero_centered_gamma`.
- GQA attention (64 query heads, 4 KV heads) with per-head QK RMSNorm and partial RoPE (64 of 128 head channels rotated, theta 5e6).
- CLIP-style vision encoder with Conv3d patch embedding, 32 transformer layers, and temporal/height/width 3D RoPE.
- Two biased GELU projector MLPs map 1280-dimensional vision patches to the 6144-dimensional text space and merge each 2x2 patch group.

## Known Limitations

- **Full attention only.** The lightning-indexer block-sparse attention branch (`self_attn.index_*` weights) is not executed by Megatron; the model runs full causal attention on every layer. Block selection keeps `index_topk_blocks * index_block_size` (2048) key tokens per query, so full attention is mathematically identical up to that sequence length and an approximation beyond it.
- **Frozen Lightning Indexer state.** The 228 lightning-indexer tensors are imported, checkpointed, and exported, but are not executed or updated by Megatron. A post-training HF export therefore combines the trained full-attention backbone with the unchanged imported indexer weights.
- **Combined vision attention.** As in the native Transformers implementation, patches from multiple image/video grids share one bidirectional vision-attention sequence; segmented attention between media items is not yet implemented.
- **Bundled recipes remain text-only.** The H100 pretraining and SQuAD SFT recipes convert the VLM config to the checkpoint-compatible text provider and do not instantiate the vision, projector, or Lightning Indexer state. VLM training requires a multimodal dataset and the VLM training step.
- **MTP modules are not mapped.** The released checkpoint advertises `num_nextn_predict_layers` in its config but ships no `mtp.*` weights.
- **Auxiliary-loss scoring differs for training.** The recipes use MCore's token-global load-balancing loss over normalized sigmoid scores. Hugging Face leaves its optional router loss disabled by default and uses softmax scores when enabled.
- The MXFP8 variant (`MiniMaxAI/MiniMax-M3-MXFP8`) is not supported; use the bf16 checkpoint.

## Conversion

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained("MiniMaxAI/MiniMax-M3", trust_remote_code=True)
provider = bridge.to_megatron_provider()
```

The bridge imports and exports the language, vision, projector, and Lightning
Indexer tensors. The Indexer weights are stored as frozen MiniMax-specific
model state, so a native Megatron checkpoint can be exported to Hugging Face
without access to the original Hugging Face checkpoint.

## Examples

For real-checkpoint Slurm conversion, inference, hardware requirements, and
validated parallelism settings, see the [MiniMax-M3 examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/examples/models/minimax/minimax_m3/README.md).

## Recipes

Pretraining and packed-sequence (THD) SFT recipes are available under [`src/megatron/bridge/recipes/minimax`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/recipes/minimax) (`minimax_m3_pretrain_256gpu_h100_bf16_config`, `minimax_m3_sft_128gpu_h100_bf16_config`), using a TP=2 / PP=4 / EP=32 baseline layout.

## Related Implementation

- Bridge implementation: [`src/megatron/bridge/models/minimax_m3`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/src/megatron/bridge/models/minimax_m3)
- Examples: [`examples/models/minimax/minimax_m3`](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/minimax/minimax_m3)
