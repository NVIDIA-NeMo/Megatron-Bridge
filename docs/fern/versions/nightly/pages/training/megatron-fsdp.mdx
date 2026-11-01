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
through `use_megatron_fsdp`.

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

- `use_megatron_fsdp` to be enabled
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

- `use_megatron_fsdp` and `use_torch_fsdp2` are mutually exclusive
- `use_tp_pp_dp_mapping` is not supported with Megatron FSDP
- legacy checkpoint formats such as `torch_dist` and `zarr` are not valid for
  Megatron FSDP save/load

When Megatron FSDP is enabled, Bridge also adjusts some settings
automatically, including disabling `average_in_collective` and several
buffer-reuse optimizations that do not match the FSDP path.

### Separate all-gather communicator

For Megatron FSDP configurations that overlap parameter all-gather with gradient
reduce-scatter, enable a second communicator over the same replica ranks:

```python
cfg.dist.create_all_gather_group = True
```

Bridge then populates `ProcessGroupCollection.dp_cp_ag`; with expert parallelism
it also populates `ProcessGroupCollection.expt_dp_ag`. This works with both the
standard MCore process-group initialization and `use_decentralized_pg=True`.
The extra groups reuse the exact dense and expert replica topology selected at
initialization, including context/expert parallel layout and rank offsets.

The option only creates and wires the communicators. The FSDP/optimizer overlap
configuration must still select parameter-gather overlap for them to provide a
performance benefit.

## Compatibility and Caveats

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
