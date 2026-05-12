# DeepSeek V4

[DeepSeek-V4](https://github.com/deepseek-ai/DeepSeek-V4) is the next-generation Mixture-of-Experts language model from DeepSeek-AI. It extends the V3 design with **Hyper-Connections (mHC)** for multi-stream residual mixing, **Compressed Sparse Attention (CSA)** with a learned token-importance indexer (DSA), **hash-routed MoE layers** for the first few decoder blocks, and a refined **Multi-Token Prediction (MTP)** head with separate `e_proj` / `h_proj` projections.

DeepSeek V4 models are supported via the Bridge system with auto-detected configuration and weight mapping.

## Available Models

Megatron Bridge supports the following DeepSeek V4 model variants:

| Variant | Hidden Layers | Hidden Size | Routed Experts | Quant Scheme | Parity Status |
|---------|--------------:|------------:|---------------:|--------------|---------------|
| **DeepSeek-V4-Flash** | 43 | 4096 | 256 | FP8 attn + MXFP4 experts (E8M0 scales) | Verified, last-token logit cosine ~0.935 vs official inference |
| **DeepSeek-V4-Flash-Base** | 43 | 4096 | 256 | Uniform FP8 (F32 block scales) | Imports cleanly; logit parity unmeasured |
| **DeepSeek-V4-Pro** | 61 | 7168 | 384 | FP8 attn + MXFP4 experts (E8M0 scales) | Algorithmic dequant validated; end-to-end unmeasured |
| **DeepSeek-V4-Pro-Base** | 61 | 7168 | 384 | Uniform FP8 (F32 block scales) | Algorithmic dequant validated; end-to-end unmeasured |

The bridge dispatches purely on tensor dtype and reads every dimension- and layer-dependent field from the HF config, so all four variants share the same code path.

## Model Architecture Features

- **Hybrid Attention (DSv4HybridSelfAttention)**: Per-layer mix of dense MLA and Compressed Sparse Attention selected by `compress_ratios`
- **Compressed Sparse Attention (CSA)** with **DSA Indexer**: Top-k token selection over windowed keys; `index_n_heads`, `index_head_dim`, `index_topk` control the indexer
- **Hyper-Connections (mHC)**: 4-stream residual mixing per layer (`hc_mult = 4`) with sinkhorn-iterated attention; per-MTP-layer `hc_head_*` learns output contraction
- **Hash-Routed MoE**: First few decoder layers use a deterministic vocab → expert mapping (`tid2eid`) instead of softmax routing
- **Multi-Token Prediction (MTP)**: One MTP layer with separate `e_proj` and `h_proj` projections (post-MCore #4518)
- **YaRN RoPE**: `rotary_scaling_factor=16`, `original_max_position_embeddings=65536`; `mscale=mscale_all_dim=1.0` for V4
- **Sigmoid Gating with Expert Bias**: `noaux_tc` load balancing, `sqrtsoftplus` scoring, expert bias enabled
- **`o_groups` Output Projection**: `o_lora_rank` low-rank output projection split into `o_groups` parallel groups

## Conversion with 🤗 Hugging Face

### Load HF → Megatron

```python
from megatron.bridge import AutoBridge

# Example: DeepSeek-V4-Flash-Base
bridge = AutoBridge.from_hf_pretrained(
    "deepseek-ai/DeepSeek-V4-Flash-Base",
    trust_remote_code=True,
)
provider = bridge.to_megatron_provider()

# DSv4 currently requires TP=1 (no MLA TP support yet); scale via EP and PP.
provider.tensor_model_parallel_size = 1
provider.expert_model_parallel_size = 8

model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Import Checkpoint from HF

```bash
uv run python examples/conversion/convert_checkpoints.py import \
  --hf-model deepseek-ai/DeepSeek-V4-Flash-Base \
  --megatron-path /checkpoints/deepseek_v4_megatron \
  --trust-remote-code
```

The bridge's `maybe_modify_loaded_hf_weight` hook dispatches dequantisation by tensor dtype:

- `int8` → MXFP4 packed nibbles → `bfloat16` via the E2M1 lookup table and per-row 16-K-tile E8M0 scales
- `float8_e4m3fn` with companion `.scale` → `bfloat16` via 128×128 block-scale expansion (handles both E8M0 and F32 scale dtypes)

No external dequantisation script is required.

### Export Megatron → HF

```python
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained(
    "deepseek-ai/DeepSeek-V4-Flash-Base",
    trust_remote_code=True,
)
bridge.export_ckpt(
    megatron_path="/results/deepseek_v4/checkpoints/iter_0000500",
    hf_path="/exports/deepseek_v4_hf",
)
```

### Run Inference on Converted Checkpoint

```bash
uv run python examples/conversion/hf_to_megatron_generate_text.py \
  --hf_model_path deepseek-ai/DeepSeek-V4-Flash-Base \
  --megatron_model_path /checkpoints/deepseek_v4_megatron \
  --prompt "Explain hyper-connections in transformer models." \
  --max_new_tokens 100 \
  --tp 1 \
  --ep 8 \
  --trust-remote-code
```

## Parallelism Configurations

DSv4 currently requires **TP=1** because MLA tensor parallelism is not supported alongside the DSv4 hybrid attention path. Scale via expert and pipeline parallelism instead.

| Model | TP | PP | EP | Min GPUs | GPU Type | Use Case |
|-------|---:|---:|---:|---------:|---------|----------|
| **DeepSeek-V4-Flash** | 1 | 1 | 4 | 4 | B200 192 GB | Smoke / single-node inference |

The model does not fit on A100 80 GB at TP=1 — use B200 192 GB or larger.

## Known Limitations

- **MCore prerequisites are not yet on a tagged release.** End-to-end imports require:
  - PR [#3430](https://github.com/NVIDIA/Megatron-LM/pull/3430) (Hyper-Connections module)
  - PR [#4458](https://github.com/NVIDIA/Megatron-LM/pull/4458) (DSv4 hybrid attention / CSA / DSA indexer)
  - PR [#4481](https://github.com/NVIDIA/Megatron-LM/pull/4481) (hash MoE + SwiGLU clamp)
  - PR [#4518](https://github.com/NVIDIA/Megatron-LM/pull/4518) (separate `e_proj` / `h_proj` for MTP with hyper-connections)

  Until these merge to Megatron-LM `main` and the bridge submodule pin advances, you must build against a fork branch that includes them (e.g. `weijiac0619/Megatron-LM` `weijiac/dsv4-bridge`).

- **MTP is disabled for inference** via `disable_mtp_for_inference()`. MTP weights are mapped end-to-end (HC head, e/h projections, transformer block, experts, norms) and loaded into the Megatron model.

- **`fast_hadamard_transform` package is optional.** When unavailable (e.g. on aarch64 NeMo containers), DSA falls back to a PyTorch hadamard implementation. Throughput is lower but numerical behavior is unchanged.

- **Logit parity is verified for Flash only** (~0.935 last-token cosine vs official `sparse_attn` tilelang reference). The remaining gap is structural — different attention/HC kernel decompositions and accumulation precisions between MCore and the official inference stack — and not a Bridge-level issue. The model still selects the correct top-1 token in practice.

## Hugging Face Model Cards & References

### Hugging Face Model Cards
- DeepSeek-V4-Flash: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash
- DeepSeek-V4-Flash-Base: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base
- DeepSeek-V4-Pro: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
- DeepSeek-V4-Pro-Base: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base

### Additional Resources
- GitHub Repository: https://github.com/deepseek-ai/DeepSeek-V4

## Related Docs
- DeepSeek V4 examples: [examples/models/deepseek_v4/README.md](../../../examples/models/deepseek_v4/README.md)
