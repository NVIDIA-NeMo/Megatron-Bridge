# Megatron FSDP

Megatron FSDP is the practical fully sharded data parallel path in Megatron
Bridge today. It shards parameters, gradients, and optimizer state across data
parallel ranks, which can reduce model-state memory substantially compared with
plain Distributed Data Parallel (DDP) or the distributed optimizer path.

This page is the stable overview for what Megatron FSDP is, when to use it, and
what constraints matter. For operational enablement, code anchors, and
verification commands, see [skills/nemo-mbridge-perf-megatron-fsdp/SKILL.md](../skills/nemo-mbridge-perf-megatron-fsdp/SKILL.md).

## What It Is

Megatron FSDP is the Megatron-Core custom FSDP implementation exposed in Bridge
through `ddp.use_megatron_fsdp`.

Replace the removed `dist.use_megatron_fsdp` setting in existing configurations
with `ddp.use_megatron_fsdp`. Set this flag only on `DistributedDataParallelConfig`;
`DistributedInitConfig` no longer accepts it.

Compared with other data-parallel strategies:

| Feature | DDP | Distributed Optimizer | Megatron FSDP |
|---|---|---|---|
| Parameter Storage | Replicated | Replicated | Sharded |
| Optimizer States | Replicated | Sharded | Sharded |
| Gradient Communication | All-reduce | Reduce-scatter | Reduce-scatter |
| Parameter Communication | None | All-gather (after update) | All-gather (on-demand) |
| Memory Efficiency | Baseline | High | Highest |
| Communication Overhead | Low | Medium | Medium-High |

The practical consequence is that Megatron FSDP is most useful when model-state
memory, rather than activation memory, is the main bottleneck.

## When to Use It

Megatron FSDP is a good fit when all of the following are true:

- the model is too large for plain DDP or distributed optimizer
- you want the strongest currently supported FSDP path in Bridge
- you are willing to trade more communication for lower memory
- you can adopt the required FSDP checkpoint format

Prefer another path when:

- DDP already fits comfortably and simplicity matters most
- distributed optimizer gives enough memory relief without fully sharding
- you are evaluating PyTorch FSDP2 for production use on this branch

## Stable Requirements

Megatron FSDP in Bridge requires:

- `ddp.use_megatron_fsdp` to be enabled
- checkpoint format `fsdp_dtensor`
- standard rank initialization order

The `fsdp_dtensor` format uses PyTorch DTensor and
`torch.distributed.checkpoint` (DCP) to store sharded parameters and optimizer
state. It is **not interchangeable** with `torch_dist` or `zarr` checkpoints —
you cannot load an `fsdp_dtensor` checkpoint into a non-FSDP run or vice versa.

`fsdp_dtensor` is compatible with 5D parallelism (TP + PP + DP + CP + EP).
Because DCP stores DTensor placement metadata, checkpoints saved under one
parallelism layout can be loaded under a different layout (e.g., change TP or PP
size between runs) — DCP handles the shard remapping automatically. The one
unsupported combination is `use_tp_pp_dp_mapping=True`, which uses an
alternative rank-initialization order that conflicts with FSDP sharding.

Important stable constraints:

- `ddp.use_megatron_fsdp` and `dist.use_torch_fsdp2` are mutually exclusive
- `use_tp_pp_dp_mapping` is not supported with Megatron FSDP
- legacy checkpoint formats such as `torch_dist` and `zarr` are not valid for
  Megatron FSDP save/load

When Megatron FSDP is enabled, Bridge also adjusts some settings
automatically, including disabling `average_in_collective` and several
buffer-reuse optimizations that do not match the FSDP path.

## Compatibility and Caveats

### Megatron FSDP v2 and DeepSeek-V4

Enable the experimental v2 implementation with `ddp.use_megatron_fsdp=True`
and `ddp.megatron_fsdp_version=2`. The DeepSeek-V4 path is qualified with
TP=PP=1 and no virtual pipeline parallelism.

For `DeepSeekV4HybridModelProvider` models with `enable_mhc_connections=True`, Bridge replaces the default
dense FSDP units with `HyperConnectionHybridLayer`. Split mHC layers call
private inner Transformer helpers directly; wrapping those inner Transformers
would bypass their parameter-gather hooks. Routed experts remain independent
expert-DP units. Remaining MTP projections and reused embedding/output
parameters are owned by the root unit.

The unit policy and compatibility checks live in
`models/deepseek/deepseek_v4_fsdp.py`. The common wrapper only dispatches to
that helper for DeepSeek-V4 mHC with Megatron FSDP v2; other models, including
other mHC models, retain Core's wrapping path. This Bridge-only adapter is
temporary until Core's wrapping entry point exposes `fsdp_unit_modules`.

With selective `mhc` recomputation, the enclosing `HybridStack` becomes the
dense unit instead. The recompute manager can replay several layers from one
output hook, before individual layer backward hooks execute. Gathering at the
stack boundary makes the replay's dense parameters available, at the cost of
keeping more dense weights gathered simultaneously. This does not change the
configured recompute regions or the separate expert units.

Configure v2 explicitly:

```python
from megatron.bridge.models.deepseek.deepseek_v4_bridge import (
    set_deepseek_v4_pipeline_model_parallel_layout,
)

cfg.model.tensor_model_parallel_size = 1
cfg.model.pipeline_model_parallel_size = 1
cfg.model.virtual_pipeline_model_parallel_size = None
set_deepseek_v4_pipeline_model_parallel_layout(cfg.model)
cfg.dist.use_megatron_fsdp = True
cfg.ddp.use_megatron_fsdp = True
cfg.ddp.megatron_fsdp_version = 2
cfg.ddp.use_distributed_optimizer = False
cfg.optimizer.use_distributed_optimizer = False
cfg.ddp.data_parallel_sharding_strategy = "optim_grads_params"
cfg.ddp.expert_data_parallel_sharding_strategy = "optim_grads_params"
cfg.ddp.outer_dp_sharding_strategy = "no_shard"
cfg.ddp.expert_outer_dp_sharding_strategy = "no_shard"
cfg.ddp.num_distributed_optimizer_instances = 1
cfg.checkpoint.ckpt_format = "fsdp_dtensor"
cfg.model.gradient_accumulation_fusion = False
cfg.model.use_fused_mhc = False
cfg.dist.distributed_timeout_minutes = 60
```

Set `megatron_fsdp_main_grads_dtype` and `megatron_fsdp_grad_comm_dtype`
explicitly when matching a reference run; `grad_reduce_in_fp32` alone does
not select v2's gradient storage precision. Data-parallel random initialization,
layer-wise distributed optimization, optimizer-step parameter-gather overlap,
and expert communication overlap are not supported by this mHC wrapper.

When changing PP, normalize the hybrid layer pattern with the helper above;
clearing only the pipeline layout can leave a stage-specific pattern and build
an incomplete model. Check actual constructed decoder depth and MTP presence.

Unit tests cover dispatch isolation, constructor arguments, unit selection and
rejected lifecycle combinations. This refactor itself has not been GPU-qualified;
model-specific recompute, MTP, checkpoint and performance workflows require
configuration-specific validation. No diagnostic synchronization or logging is
added by the wrapper. Fused versus unfused mHC is a separate execution setting,
not selected or modified by this adapter.

### General compatibility

At the configuration level, Megatron FSDP is intended to work with:

- tensor parallelism
- pipeline parallelism
- context parallelism
- expert parallelism
- BF16 or FP16 mixed precision

However, not every combination has the same level of in-repo validation or
performance evidence. Treat broad compatibility as code-supported first, not as
fully benchmark-proven for every combination.

Two practical caveats matter most:

1. Public recipes may expose `use_megatron_fsdp` while still defaulting to a
   non-FSDP checkpoint format. The checkpoint requirement is stable and
   mandatory even when recipe ergonomics lag behind.
2. FSDP reduces model-state memory, not activation memory. For long-sequence or
   activation-bound workloads, other techniques such as context parallelism,
   activation recomputation, or CPU offloading may still be needed.

## Torch FSDP2 Status

Megatron Bridge also exposes a PyTorch FSDP2 path via `use_torch_fsdp2`, but
that path should still be treated as experimental on this branch.

The stable recommendation today is:

- use Megatron FSDP if you need an FSDP path in Bridge
- do not treat FSDP2 as interchangeable with Megatron FSDP

## Related Docs

- [docs/training/checkpointing.md](checkpointing.md)
- [docs/training/cpu-offloading.md](cpu-offloading.md)
- [docs/performance-guide.md](../performance-guide.md)
- [skills/nemo-mbridge-perf-megatron-fsdp/SKILL.md](../skills/nemo-mbridge-perf-megatron-fsdp/SKILL.md)
- [skills/nemo-mbridge-perf-megatron-fsdp/card.yaml](../skills/nemo-mbridge-perf-megatron-fsdp/card.yaml)
