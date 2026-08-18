# DeepSeek V4

[DeepSeek-V4](https://github.com/deepseek-ai/DeepSeek-V4) is the next-generation Mixture-of-Experts language model from DeepSeek-AI. It extends the V3 design with **Hyper-Connections (mHC)** for multi-stream residual mixing, **Compressed Sparse Attention (CSA)** with a learned token-importance indexer (DSA), **hash-routed MoE layers** for the first few decoder blocks, and a refined **Multi-Token Prediction (MTP)** head with separate `e_proj` / `h_proj` projections.

DeepSeek V4 models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Model Architecture Features

- **Hybrid Attention (DSv4HybridSelfAttention)**: Per-layer mix of dense MLA and Compressed Sparse Attention selected by `compress_ratios`
- **Compressed Sparse Attention (CSA)** with **DSA Indexer**: Top-k token selection over windowed keys; `index_n_heads`, `index_head_dim`, `index_topk` control the indexer
- **Hyper-Connections (mHC)**: 4-stream residual mixing per layer (`hc_mult = 4`) with sinkhorn-iterated attention; per-MTP-layer `hc_head_*` learns output contraction
- **Hash-Routed MoE**: First few decoder layers use a deterministic vocab → expert mapping (`tid2eid`) instead of softmax routing
- **Multi-Token Prediction (MTP)**: One MTP layer with separate `e_proj` and `h_proj` projections (post-MCore #4518)
- **YaRN RoPE**: `rotary_scaling_factor=16`, `original_max_position_embeddings=65536`; `mscale=mscale_all_dim=1.0` for V4
- **Sigmoid Gating with Expert Bias**: `noaux_tc` load balancing, `sqrtsoftplus` scoring, expert bias enabled
- **`o_groups` Output Projection**: `o_lora_rank` low-rank output projection split into `o_groups` parallel groups

## Examples, Parallelism, and Limitations

For checkpoint conversion and inference scripts, recommended parallelism settings, and current known limitations, see the [DeepSeek V4 examples README](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4).

## Hugging Face Model Cards & References

### Hugging Face Model Cards
- DeepSeek-V4-Flash: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- DeepSeek-V4-Flash-Base: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base
- DeepSeek-V4-Pro: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek-V4-Pro-Base: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base

### Additional Resources
- GitHub Repository: https://github.com/deepseek-ai/DeepSeek-V4

## Related Docs
- DeepSeek V4 examples: [examples/models/deepseek_v4/README.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4)
- DeepSeek-V4-Pro GB300 performance reproduction: [examples/models/deepseek_v4_perf/README.md](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/r0.5.0/examples/models/deepseek_v4_perf)
