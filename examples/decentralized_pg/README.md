# Decentralized Process Groups Examples

This directory contains examples demonstrating how to use **decentralized process groups** (`use_decentralized_pg=True`) in Megatron-Bridge for distributed training.

## Overview

Instead of relying on Megatron-Core's global parallel state (mpu) module, you can use a `ProcessGroupCollection` that is explicitly passed to all components. This gives you full control over the parallelism topology and is useful for:

1. **Reinforcement Learning**: Multiple model instances (policy, value, reference) with different parallelism
2. **Multi-Model Pipelines**: Complex workflows requiring explicit control over communication
3. **Testing/Debugging**: Isolated process groups without global state side effects

## Files

| File | Description |
|------|-------------|
| `pretrain_qwen3_simple.py` | **Simple**: Use a recipe and enable `use_decentralized_pg=True` |
| `pretrain_qwen3_with_decentralized_pg.py` | **Advanced**: Manually create process groups with `HyperCommGrid` |

## Quick Start

### Simple Approach (Recommended)

Just use an existing recipe and enable decentralized process groups:

```bash
# 8 GPUs: TP2 x PP2 x DP2
uv run python -m torch.distributed.run --nproc_per_node=8 examples/decentralized_pg/pretrain_qwen3_simple.py

# 4 GPUs: TP2 x PP2 x DP1
uv run python -m torch.distributed.run --nproc_per_node=4 examples/decentralized_pg/pretrain_qwen3_simple.py
```

The key is just two lines:

```python
from megatron.bridge.recipes.qwen.qwen3 import qwen3_4b_pretrain_config

cfg = qwen3_4b_pretrain_config(
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    # ... other settings
)

# Enable decentralized process groups
cfg.dist.use_decentralized_pg = True
cfg.dist.use_gloo_process_groups = False  # Gloo not supported
```

### Advanced Approach (Manual Process Group Creation)

For full control over process groups:

```bash
# 8 GPUs: TP2 x PP2 x DP2
uv run python -m torch.distributed.run --nproc_per_node=8 examples/decentralized_pg/pretrain_qwen3_with_decentralized_pg.py

# 4 GPUs: TP2 x PP2 x DP1
uv run python -m torch.distributed.run --nproc_per_node=4 examples/decentralized_pg/pretrain_qwen3_with_decentralized_pg.py \
    --tp-size 2 --pp-size 2

# 2 GPUs: TP2 x PP1 x DP1
uv run python -m torch.distributed.run --nproc_per_node=2 examples/decentralized_pg/pretrain_qwen3_with_decentralized_pg.py \
    --tp-size 2 --pp-size 1
```

## Manual Process Group Creation (Advanced)

### Step 1: Initialize torch.distributed

```python
torch.distributed.init_process_group(backend="nccl", world_size=world_size, rank=rank)
```

### Step 2: Create ProcessGroupCollection with HyperCommGrid

```python
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.process_groups_config import ProcessGroupCollection

# Create a grid with shape [TP, CP, DP, PP]
grid = HyperCommGrid(
    shape=[tp_size, cp_size, dp_size, pp_size],
    dim_names=["tp", "cp", "dp", "pp"],
    rank_offset=0,
    backend="nccl",
)

# Create process groups by selecting dimensions
tp_pg = grid.create_pg(["tp"])      # Ranks differ only in TP dimension
pp_pg = grid.create_pg(["pp"])      # Ranks differ only in PP dimension
dp_pg = grid.create_pg(["dp"])      # Ranks differ only in DP dimension
mp_pg = grid.create_pg(["tp", "pp"]) # Model parallel = TP + PP

# Bundle into ProcessGroupCollection
pg_collection = ProcessGroupCollection(
    tp=tp_pg,
    pp=pp_pg,
    dp=dp_pg,
    mp=mp_pg,
    # ... more groups
)
```

### Step 3: Set Random Seeds (REQUIRED)

```python
from megatron.core import tensor_parallel
from megatron.core.utils import get_pg_rank

# Get TP rank from our process group
tp_rank = get_pg_rank(pg_collection.tp)

# Initialize CUDA RNG tracker - REQUIRED before model creation!
tensor_parallel.model_parallel_cuda_manual_seed(
    seed=1234,
    te_rng_tracker=False,
    inference_rng_tracker=False,
    use_cudagraphable_rng=False,
    tp_rank=tp_rank,
    ep_rank=0,
    etp_rank=tp_rank,
)
```

### Step 4: Pass pg_collection Explicitly to Components

```python
# Model creation
model = cfg.model.provide_distributed_model(
    pg_collection=pg_collection,  # <-- Pass here!
    ...
)

# Optimizer setup
optimizer, scheduler = setup_optimizer(
    pg_collection=pg_collection,  # <-- Pass here!
    ...
)

# Data loaders use the DP group
train_data_iterator = setup_data_iterators(
    dp_group=pg_collection.dp,  # <-- Use DP group for data sharding!
    ...
)

# Training loop
train(
    pg_collection=pg_collection,  # <-- Pass here!
    ...
)
```

## HyperCommGrid Explained

`HyperCommGrid` creates a multi-dimensional grid of ranks. The grid shape `[TP, CP, DP, PP]` defines how ranks are organized:

When you call `grid.create_pg(["tp"])`, it creates groups of ranks that share the same DP and PP coordinates but differ in TP:
- Group 1: [rank 0, rank 1] (DP=0, PP=0)
- Group 2: [rank 2, rank 3] (DP=0, PP=1)
- Group 3: [rank 4, rank 5] (DP=1, PP=0)
- Group 4: [rank 6, rank 7] (DP=1, PP=1)

## Limitations

- Gloo process groups are not supported (NCCL only)
- ModelOpt sharded checkpointing is disabled
- Distillation tensor shape adjustment is disabled
