# DeepSeek V4

End-to-end conversion and inference scripts for the DeepSeek V4 family on Megatron Bridge.

The bridge supports four published variants out of the same code path. The on-disk quantisation differs between post-trained (Flash, Pro) and pretrained-only (Flash-Base, Pro-Base) models — see [`docs/models/llm/deepseek-v4.md`](../../../docs/models/llm/deepseek-v4.md) for the per-variant scheme.

| Variant | HF path | Quant scheme | TP | EP | Min GPUs |
|---------|---------|--------------|---:|---:|---------:|
| DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash` | FP8 attn + MXFP4 experts | 1 | 4 | 4 × B200 192 GB |
| DeepSeek-V4-Flash-Base | `deepseek-ai/DeepSeek-V4-Flash-Base` | uniform FP8 (F32 scales) | 1 | 4 | 4 × B200 192 GB |
| DeepSeek-V4-Pro | `deepseek-ai/DeepSeek-V4-Pro` | FP8 attn + MXFP4 experts | 1 | 16 | 16 × B200 192 GB |
| DeepSeek-V4-Pro-Base | `deepseek-ai/DeepSeek-V4-Pro-Base` | uniform FP8 (F32 scales) | 1 | 16 | 16 × B200 192 GB |

DSv4 currently requires **TP=1**; scale via expert and pipeline parallelism. The model does not fit on A100 80 GB at TP=1.

## MCore prerequisites

DSv4 imports require the following MCore commits, none of which are on a tagged release yet:

- PR [#3430](https://github.com/NVIDIA/Megatron-LM/pull/3430) (Hyper-Connections)
- PR [#4458](https://github.com/NVIDIA/Megatron-LM/pull/4458) (DSv4 hybrid attention / CSA / DSA indexer)
- PR [#4481](https://github.com/NVIDIA/Megatron-LM/pull/4481) (hash MoE + SwiGLU clamp)
- PR [#4518](https://github.com/NVIDIA/Megatron-LM/pull/4518) (separate `e_proj` / `h_proj` for MTP with hyper-connections)

Until these merge to Megatron-LM `main` and the bridge submodule pin advances, point `3rdparty/Megatron-LM` at a fork branch that includes them.

## Files

- `conversion.sh` — HF → Megatron import and Megatron → HF export (single-node, real model)
- `inference.sh` — text generation against an HF or Megatron checkpoint
- `slurm_sft.sh` — multi-node Slurm template for full SFT (when a DSv4 recipe is added)
- `slurm_peft.sh` — multi-node Slurm template for PEFT/LoRA (when a DSv4 recipe is added)

Run `bash conversion.sh` after setting `WORKSPACE` and `MODEL_VARIANT`. See each script's header comments for the expected env vars.
