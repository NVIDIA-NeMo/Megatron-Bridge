# DeepSeek V3 VR200 debugging proxies

The `proxy` benchmark variant provides 64-GPU MXFP8 and NVFP4 counterparts to
the 256-GPU VR200 recipes (the legacy `v2` selector). These are reduced-depth,
mock-data debugging recipes. They retain the parent's GBS, per-layer tensor
shapes and kernel features, but use PP=1 to remove pipeline communication
and bubbles. They are not full-model training benchmarks or direct predictors
of at-scale throughput.

| Property | At scale | Proxy |
|---|---|---|
| GPUs / Hecate nodes (4 GPUs/node) | 256 / 64 | 64 / 16 |
| Decoder layers / MTP layers | 61 / 1 | 13 / 1 |
| Dense / MoE decoder layers | 3 / 58 | 3 / 10 |
| TP / CP / PP / VPP / EP / ETP | 1 / 1 / 2 / 8 / 32 / 1 | 1 / 1 / 1 / disabled / 32 / 1 |
| Dense DP / expert DP | 128 / 4 | 64 / 2 |
| GBS / MBS | 4096 / 1 | 4096 / 1 |
| Microbatches per iteration | 32 | 64 |
| Hidden/FFN/attention dimensions, experts per layer and routing | Precision-specific parent | Unchanged |
| Precision, graph, dispatcher, fusion, recompute, MoE overlap and environment | Precision-specific parent | Unchanged |
| PP communication / aligned PP parameter gathers | Enabled | Disabled by PP=1 overlap setup |
| Training steps | 50 | 50 |

Select `--num_gpus 64 --config_variant proxy` with model family `deepseek`,
model recipe `deepseek_v3`, GPU `vr200`, and compute dtype `fp8_mx` or `nvfp4`
in the performance launcher. The exported recipes are:

- `deepseek_v3_pretrain_64gpu_vr200_fp8mx_proxy_config`
- `deepseek_v3_pretrain_64gpu_vr200_nvfp4_proxy_config`

## PP=1 layout and scheduling

Keep the previous proxy's 13 decoder layers: three leading dense layers
and ten MoE layers, with `moe_layer_freq=[0] * 3 + [1] * 10`. Retaining this
depth avoids changing the model again while testing a different schedule.
Both precisions now set `pipeline_model_parallel_size=1`,
`virtual_pipeline_model_parallel_size=None`, and
`pipeline_model_parallel_layout=None`. Embedding, all decoder layers, MTP,
and loss are colocated; the inherited multi-stage layout must not survive.

MCore selects `forward_backward_no_pipelining`. With
`overlap_moe_expert_parallel_comm=True`, that function uses
`combined_1f1b_schedule_for_no_pipelining`: one initial forward, 63 paired
forward/backward microbatches, and one final backward. Delayed weight
gradients, full-iteration CUDA graphs, HybridEP, cuTeDSL fusion, and each
parent's precision-specific settings remain enabled. Bridge's overlap setup
disables pipeline P2P overlap, batched P2P, and PP-aligned parameter gathers;
DP gradient reduction and parameter-gather overlap remain enabled.

The dense and expert meshes share the same GPUs; they are not multiplied
together. With 64 GPUs and TP=CP=ETP=PP=1:

- Dense DP = `64 / (TP * PP * CP) = 64`.
- Expert DP = `64 / (PP * EP * ETP) = 2`.
- Microbatches = `4096 / (DP * MBS) = 64`.

EP=32 still gives eight routed experts per layer per rank and preserves
per-microbatch EP token-dispatch shapes. Expert-DP collectives are now
exercised across two replicas, unlike the previous PP=2 proxy's expert DP=1.

The original 13-layer PP=2/VPP=2 proxy retained the first two and last two
logical chunks of the 256-GPU layout. The PP=1 version keeps that reduced
architecture but intentionally no longer preserves those chunk boundaries.

## What this proxy cannot establish

- PP=1 removes all pipeline traffic and bubbles. It can help isolate kernel
  and EP-overlap performance, but cannot validate the parent's PP=2/VPP=8
  schedule or predict its end-to-end throughput by simple layer scaling.
- All 13 decoder layers and MTP are now resident per rank, instead of being
  split across two PP stages. Expert DP=2 improves optimizer sharding versus
  the previous proxy, but remains smaller than the parent's expert DP=4.
  Fewer in-flight microbatches change activation liveness. GPU memory fit,
  capture, and replay must be revalidated; the PP=2 result is not PP=1 proof.
- Fixed GBS=4096 with DP=64 and MBS=1 means 64 microbatches, not the
  parent's 32. Dense/MoE layer proportions, MTP overhead, graph structure,
  and communication/compute balance still differ from the full model.
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
The 64-GPU PP=1 proxy now exercises expert-DP collectives, but a regression
specific to larger DP groups or pipeline communication still needs at-scale
validation.
