# DeepSeek V3 VR200 debugging proxies

The `proxy` benchmark variant provides 64-GPU MXFP8 and NVFP4 counterparts to
the 256-GPU VR200 recipes (the legacy `v2` selector). These are full-model,
mock-data performance/debugging recipes, not reduced-layer surrogates.

| Property | At scale | Proxy |
|---|---|---|
| GPUs / Hecate nodes (4 GPUs/node) | 256 / 64 | 64 / 16 |
| TP / CP / PP / VPP / EP / ETP | 1 / 1 / 2 / 8 / 32 / 1 | Same |
| Dense DP / expert DP | 128 / 4 | 32 / 1 |
| GBS / MBS | 4096 / 1 | 1024 / 1 |
| Microbatches per iteration | 32 | 32 |
| Model dimensions, experts, MTP and pipeline layout | Precision-specific parent | Unchanged |
| Precision, graph, dispatcher, fusion, recompute, overlap and environment | Precision-specific parent | Unchanged |
| Training steps | 50 | 50 |

Select `--num_gpus 64 --config_variant proxy` with model family `deepseek`,
model recipe `deepseek_v3`, GPU `vr200`, and compute dtype `fp8_mx` or `nvfp4`
in the performance launcher. The exported recipes are:

- `deepseek_v3_pretrain_64gpu_vr200_fp8mx_proxy_config`
- `deepseek_v3_pretrain_64gpu_vr200_nvfp4_proxy_config`

Only the global batch is scaled: local token shapes, EP dispatch volume,
experts per rank, and the interleaved 1F1B microbatch count are preserved.
The two precision recipes intentionally retain their different pipeline
layouts, attention/fusion settings and environment variables.

## What this proxy cannot establish

- Expert DP drops from 4 to 1. Expert optimizer shards can be four times
  larger, and expert-DP reductions/gathers across replicas are not exercised.
  Memory fit on 64 GPUs must be validated; fewer GPUs does not mean less
  per-GPU memory. Do not silently add offload/recompute to make the proxy fit.
- Dense-DP collectives have fewer ranks. The smaller allocation can also
  change NVLink-domain placement and inter-domain PP/DP traffic. Keep the
  physical `NVLINK_DOMAIN_SIZE=72` and HybridEP rank-group setting at 32;
  neither is the total allocation size. Inspect actual placement in the logs.
- The smaller global batch changes the optimization trajectory. Losses and
  throughput are not directly comparable with the 256-GPU benchmark; a
  passing proxy is not convergence or at-scale performance acceptance.

For debugging, keep container, Bridge/MCore/TE/DeepEP revisions and Hecate
fabric settings matched, then change one feature at a time. Require all 50
steps, finite losses/gradients, actual CUDA-graph replay and intended fused
kernels, and per-rank memory evidence. Recheck the result on 256 GPUs.
An EDP-collective regression needs a larger allocation: 128 GPUs is the
minimum with this model-parallel mesh and expert DP greater than one.
