# DeepSeek V4

End-to-end conversion and inference scripts for the DeepSeek V4 family on Megatron Bridge.

The bridge supports four published variants out of the same code path. The on-disk quantisation differs between post-trained (Flash, Pro) and pretrained-only (Flash-Base, Pro-Base) models — see [`docs/models/llm/deepseek-v4.md`](../../../docs/models/llm/deepseek-v4.md) for the per-variant scheme.

## MCore Dev Branch Requirement

DSv4 imports require MCore changes that are not yet on a tagged release: PR [#3430](https://github.com/NVIDIA/Megatron-LM/pull/3430), PR [#4458](https://github.com/NVIDIA/Megatron-LM/pull/4458), PR [#4481](https://github.com/NVIDIA/Megatron-LM/pull/4481), and PR [#4518](https://github.com/NVIDIA/Megatron-LM/pull/4518). Until these merge to Megatron-LM `main` and the bridge submodule pin advances, point `3rdparty/Megatron-LM` at the Megatron-LM `dev` branch:

```bash
./scripts/switch_mcore.sh dev
uv sync
```

Use `./scripts/switch_mcore.sh main` and `uv sync --locked` to return to the pinned main-branch submodule.

| Variant | HF path | Quant scheme | TP | EP | Validation |
|---------|---------|--------------|---:|---:|------------|
| DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash` | FP8 attn + MXFP4 experts | 1 | 4 | Verified on GB200, last-real-token logit cosine ~0.97-0.99 vs official inference |
| DeepSeek-V4-Flash-Base | `deepseek-ai/DeepSeek-V4-Flash-Base` | uniform FP8 (F32 scales) | 1 | 4 | Verified on GB200, last-real-token logit cosine 0.9866-0.9930, mean 0.9907 vs official inference |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro` | FP8 attn + MXFP4 experts | 1 | 16 | Algorithmic dequant validated; end-to-end unmeasured |
| DeepSeek-V4-Pro-Base | `deepseek-ai/DeepSeek-V4-Pro-Base` | uniform FP8 (F32 scales) | 1 | 16 | Algorithmic dequant validated; end-to-end unmeasured |

## Examples

- `conversion.sh` imports HF weights into Megatron Bridge and exports Megatron checkpoints back to HF format.
- `inference.sh` runs text generation against an HF or Megatron checkpoint.

Run `bash conversion.sh` after setting `WORKSPACE` and `MODEL_VARIANT`. See each script's header comments for the expected environment variables and `#SBATCH` directives to edit before submitting.

The bridge's `maybe_modify_loaded_hf_weight` hook dispatches dequantisation by tensor dtype:

- `int8` -> MXFP4 packed nibbles -> `bfloat16` via the E2M1 lookup table and per-row 16-K-tile E8M0 scales
- `float8_e4m3fn` with companion `.scale` -> `bfloat16` via 128x128 block-scale expansion, handling both E8M0 and F32 scale dtypes

No external dequantisation script is required.

## Parallelism Configurations

DSv4 currently requires **TP=1** because MLA tensor parallelism is not supported alongside the DSv4 hybrid attention path. Scale via expert and pipeline parallelism instead.

| Model | TP | PP | EP | Verified Layout | Use Case |
|-------|---:|---:|---:|-----------------|----------|
| DeepSeek-V4-Flash | 1 | 1 | 4 | 4 GPUs on GB200 | Smoke / single-node inference |
| DeepSeek-V4-Flash-Base | 1 | 1 | 4 | 4 GPUs on GB200 | Smoke / single-node inference |
| DeepSeek-V4-Pro | 1 | 1+ | 16 | Not end-to-end verified | Multi-node inference or conversion |
| DeepSeek-V4-Pro-Base | 1 | 1+ | 16 | Not end-to-end verified | Multi-node inference or conversion |

## Known Limitations

- **MTP is disabled for inference** via `disable_mtp_for_inference()`. MTP weights are mapped end-to-end and loaded into the Megatron model.

- **`fast_hadamard_transform` is optional.** When unavailable, DSA falls back to a PyTorch hadamard implementation. Throughput is lower but numerical behavior is unchanged.

- **Logit parity is verified for Flash and Flash-Base** against the official inference stack at last-real-token logits. The remaining gap is structural, from different attention/HC kernel decompositions and accumulation precisions between MCore and official inference.
