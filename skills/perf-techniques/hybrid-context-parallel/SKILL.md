---
name: hybrid-context-parallel
description: Operational guide for enabling hierarchical context parallelism in Megatron-Bridge, including config knobs, code anchors, pitfalls, and verification.
---

# Hybrid / Hierarchical Context Parallel Skill

For stable background and recommendation level, see:

- `docs/training/hybrid-context-parallel.md`
- `card.yaml` (co-located)

## Enablement

Minimal Bridge override:

```python
cfg.model.context_parallel_size = 4
cfg.model.cp_comm_type = "a2a+p2p"
cfg.model.hierarchical_context_parallel_sizes = [2, 2]
cfg.dist.use_decentralized_pg = False
```

Required constraints:

- `prod(hierarchical_context_parallel_sizes) == context_parallel_size`
- `seq_length % (2 * context_parallel_size) == 0`
- Transformer Engine `>= 1.12.0`

## Code Anchors

Upstream config and validation:

```45:54:3rdparty/Megatron-LM/megatron/core/model_parallel_config.py
context_parallel_size: int = 1
"""Splits network input along sequence dimension across GPU ranks."""

hierarchical_context_parallel_sizes: Optional[list[int]] = None
"""Degrees of the hierarchical context parallelism. Users should provide a list to specify 
   the sizes for different levels. Taking the a2a+p2p cp comm type as example, it contains
   groups of two levels, so the first value of the list indicates the group size of the a2a
   communication type, and the second value indicates the group size of the p2p communication
   type.
"""
```

```428:433:3rdparty/Megatron-LM/megatron/training/arguments.py
if args.hierarchical_context_parallel_sizes:
    from numpy import prod
    assert args.context_parallel_size == prod(args.hierarchical_context_parallel_sizes)
if "a2a+p2p" in args.cp_comm_type:
    assert args.hierarchical_context_parallel_sizes is not None, \
    "--hierarchical-context-parallel-sizes must be set when a2a+p2p is used in cp comm"
```

Bridge MPU path:

```613:648:src/megatron/bridge/training/initialize.py
parallel_state.initialize_model_parallel(
    ...
    context_parallel_size=model_config.context_parallel_size,
    hierarchical_context_parallel_sizes=model_config.hierarchical_context_parallel_sizes,
    ...
)
...
return ProcessGroupCollection.use_mpu_process_groups()
```

Bridge decentralized-PG path:

```503:524:src/megatron/bridge/training/initialize.py
pg_collection = ProcessGroupCollection(
    ...
    cp=cp_pg,
    tp_cp=tp_cp_pg,
    hcp=None,
    ep=ep_pg,
    ...
)
```

## Pitfalls

1. `a2a+p2p` and upstream `hybrid_context_parallel=True` are different features.
2. Bridge HCP is MPU-only today. If `use_decentralized_pg=True`, Bridge initializes flat CP groups and leaves HCP unset.
3. No checked-in Bridge recipe currently exercises HCP directly.
4. Single-GPU load helpers clear `hierarchical_context_parallel_sizes`.
5. Treat end-to-end HCP training as advanced until there is in-tree functional coverage.

## Verification

Verify the supported MPU path:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m torch.distributed.run --nproc_per_node=4 \
  scripts/verify_hybrid_context_parallel.py --mode mpu
```

Verify the unsupported decentralized-PG path stays flat:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m torch.distributed.run --nproc_per_node=4 \
  scripts/verify_hybrid_context_parallel.py --mode decentralized
```

Success criteria:

- MPU mode prints `HCP_MPU_GROUPS_READY`
- decentralized mode prints `HCP_DECENTRALIZED_GROUPS_FLAT`
