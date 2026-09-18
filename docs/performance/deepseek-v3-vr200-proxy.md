# DeepSeek V3 VR200 debugging proxies

The `proxy` benchmark variant provides 64-GPU MXFP8 and NVFP4 counterparts to
the 256-GPU VR200 recipes (the legacy `v2` selector). These are reduced-depth,
mock-data debugging recipes. They retain the parent's GBS, per-layer tensor
shapes and execution features, but are not full-model training benchmarks.

| Property | At scale | Proxy |
|---|---|---|
| GPUs / Hecate nodes (4 GPUs/node) | 256 / 64 | 64 / 16 |
| Decoder layers / MTP layers | 61 / 1 | 13 / 1 |
| Dense / MoE decoder layers | 3 / 58 | 3 / 10 |
| TP / CP / PP / VPP / EP / ETP | 1 / 1 / 2 / 8 / 32 / 1 | 1 / 1 / 2 / 2 / 32 / 1 |
| Dense DP / expert DP | 128 / 4 | 32 / 1 |
| GBS / MBS | 4096 / 1 | 4096 / 1 |
| Microbatches per iteration | 32 | 128 |
| Hidden/FFN/attention dimensions, experts per layer and routing | Precision-specific parent | Unchanged |
| Precision, graph, dispatcher, fusion, recompute, overlap and environment | Precision-specific parent | Unchanged |
| Training steps | 50 | 50 |

Select `--num_gpus 64 --config_variant proxy` with model family `deepseek`,
model recipe `deepseek_v3`, GPU `vr200`, and compute dtype `fp8_mx` or `nvfp4`
in the performance launcher. The exported recipes are:

- `deepseek_v3_pretrain_64gpu_vr200_fp8mx_proxy_config`
- `deepseek_v3_pretrain_64gpu_vr200_nvfp4_proxy_config`

## Why 13 decoder layers

The at-scale layouts have 16 logical PP/VPP chunks. Retain the first two
and last two chunks, removing twelve four-layer middle chunks. This leaves
13 decoder layers across PP=2 and VPP=2, without inventing new chunk shapes
or empty stages. The dense/MoE pattern must also be shortened to
`[0] * 3 + [1] * 10`; retaining the original 61-entry pattern is invalid.

| Precision | Parent layout | Proxy layout |
|---|---|---|
| MXFP8 | `Et*4\|(t*4\|)*14tmL` | `Et*4\|(t*4\|)*2tmL` |
| NVFP4 | `Et*5\|(t*4\|)*14mL` | `Et*5\|(t*4\|)*2mL` |

Here `E` is embedding, `t` a decoder layer, `m` MTP, and `L` loss. The
NVFP4 final chunk still contains only MTP/loss; MXFP8 retains a decoder
layer before MTP/loss. Both retain interleaved 1F1B, full-iteration CUDA
graphs, cuTeDSL fusion, and their parent's precision-specific settings.
Per-microbatch token shapes, EP dispatch volume and experts per layer per
rank are unchanged. Total layer count and the microbatch schedule differ.

This follows the depth/layout/MoE-pattern resizing used by the earlier
GB300 8-GPU NVFP4 proxy on `malay/dsv3_fp4_full_itercg` (commits
`e2e885e47`, `741014839`, `2512a0f97`), without carrying over its smaller
EP group, different GBS or historical precision/environment overrides.

## What this proxy cannot establish

- Expert DP drops from 4 to 1. Optimizer state per retained expert layer
  is less sharded, but far fewer layers are resident. This reduces the model
  state relative to a full-depth 64-GPU run; it does not prove peak memory
  fit. Expert-DP reductions/gathers across replicas are not exercised.
- Fixed GBS=4096 with DP=32 and MBS=1 means 128 microbatches, not the
  parent's 32. VPP=2 also changes pipeline warmup/drain, activation liveness,
  graph structure and communication/compute balance. Capture memory and
  replay must be measured; step time is not an at-scale throughput predictor.
- Dense-DP collectives have fewer ranks. The smaller allocation can also
  change NVLink-domain placement and inter-domain PP/DP traffic. Keep the
  physical `NVLINK_DOMAIN_SIZE=72` and HybridEP rank-group setting at 32;
  neither is the total allocation size. Inspect actual placement in the logs.
- The model architecture is shallower despite the unchanged global batch.
  Losses are not directly comparable with the 256-GPU benchmark, and full
  model checkpoints are not interchangeable with this proxy. A passing
  proxy is not convergence or at-scale performance acceptance.

For debugging, keep container, Bridge/MCore/TE/DeepEP revisions and Hecate
fabric settings matched, then change one feature at a time. Require all 50
steps, finite losses/gradients, actual CUDA-graph replay and intended fused
kernels, and per-rank memory evidence. Recheck the result on 256 GPUs.
An EDP-collective regression needs a larger allocation: 128 GPUs is the
minimum with this model-parallel mesh and expert DP greater than one.
