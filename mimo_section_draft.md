### MIMO (Multi-In-Multi-Out)

MIMO enables **heterogeneous multi-module training** where different models (e.g., vision encoder + LLM) can be trained simultaneously with different parallelism strategies. This is critical for Vision-Language Models (VLMs) where the encoder and LLM have different computational requirements and optimal parallelism configurations.

**Key capabilities:**
- Train multiple models (encoders + LLM) with different TP/PP/DP configurations
- Three deployment modes: homogeneous, colocated, and heterogeneous
- DP-aware data loading that handles varying batch sizes per module
- Automatic data redistribution between modules with different DP sizes
- Performance monitoring and profiling for ablation studies

**Key files:**
- `src/megatron/bridge/models/mimo/` - MIMO model provider and config
  - `mimo_provider.py` - Main provider with parallelism support
  - `mimo_config.py` - Configuration dataclasses for parallelism
  - `mimo_builder.py` - HyperCommGrid and process group builders
- `src/megatron/bridge/training/mimo_ddp.py` - DDP wrapping for MIMO models

**Status:** Active development on branch `mimo/phase3-data-loading-v2`

---

## Deployment Modes

MIMO supports three deployment modes, each optimized for different use cases:

### 1. Homogeneous Mode

All modules use **identical parallelism** (same TP/PP/DP). Simplest configuration, uses shared process groups.

**When to use:**
- Baseline experiments
- When encoder and LLM have similar sizes
- Debugging new features

**Configuration:**
```python
from megatron.bridge.models.mimo import MimoParallelismConfig, ModuleParallelismConfig

mimo_config = MimoParallelismConfig(
    llm_module_name="language_module",
    deployment_mode="homogeneous",
    module_parallelisms={
        "images": ModuleParallelismConfig(
            tensor_parallel=4,
            pipeline_parallel=1,
            data_parallel=2,
        ),
        "language_module": ModuleParallelismConfig(
            tensor_parallel=4,  # Same as encoder
            pipeline_parallel=1,
            data_parallel=2,    # Same as encoder
        ),
    },
)
```

**Characteristics:**
- All modules share same HyperCommGrid and ProcessGroupCollection
- Both modules colocated on same GPUs (rank_offset=0)
- Uses `forward_backward_no_pipelining` schedule
- Simplest data loading (no redistribution needed)

**Schedule:** `forward_backward_no_pipelining`

---

### 2. Colocated Mode

Modules use **different TP/DP but same GPUs** (PP=1 for all). Uses `ColocatedBridgeCommunicator` for automatic data redistribution.

**When to use:**
- Small encoder with large LLM (e.g., 1B encoder + 7B LLM)
- Encoder needs less TP, more DP for throughput
- Want to maximize GPU utilization without pipeline parallelism

**Configuration:**
```python
mimo_config = MimoParallelismConfig(
    llm_module_name="language_module",
    deployment_mode="colocated",
    module_parallelisms={
        "images": ModuleParallelismConfig(
            tensor_parallel=1,   # Less TP for small encoder
            pipeline_parallel=1,
            data_parallel=8,     # More DP for throughput
            rank_offset=0,       # Colocated = same GPUs
        ),
        "language_module": ModuleParallelismConfig(
            tensor_parallel=8,   # More TP for large LLM
            pipeline_parallel=1,
            data_parallel=1,
            rank_offset=0,       # Colocated = same GPUs
        ),
    },
)
```

**Characteristics:**
- Different grids but rank_offset=0 (same GPUs)
- PP=1 required for all modules
- Total ranks must match across modules
- Automatic data redistribution via `ColocatedBridgeCommunicator`
- Uses `forward_backward_no_pipelining` schedule
- Fan-in (encoder_dp > llm_dp) or fan-out (encoder_dp < llm_dp) supported

**Data redistribution patterns:**
- **Fan-in** (encoder_dp=8, llm_dp=1): Encoder processes 8 micro-batches, LLM processes 1 global batch
- **Fan-out** (encoder_dp=1, llm_dp=8): Encoder processes 1 global batch, LLM processes 8 micro-batches

**Schedule:** `forward_backward_no_pipelining`

---

### 3. Heterogeneous Mode

Modules use **different GPUs with pipeline parallelism**. Most flexible but most complex.

**When to use:**
- Large models requiring PP for memory
- Different modules have vastly different sizes
- Need maximum parallelism flexibility

**Configuration:**
```python
mimo_config = MimoParallelismConfig(
    llm_module_name="language_module",
    deployment_mode="heterogeneous",
    module_parallelisms={
        "images": ModuleParallelismConfig(
            tensor_parallel=2,
            pipeline_parallel=2,
            data_parallel=2,
            rank_offset=0,       # GPUs 0-7
        ),
        "language_module": ModuleParallelismConfig(
            tensor_parallel=4,
            pipeline_parallel=2,
            data_parallel=2,
            rank_offset=8,       # GPUs 8-23
        ),
    },
)
```

**Characteristics:**
- Separate grids with different rank_offset (different GPUs)
- PP > 1 allowed
- Uses `forward_backward_1f1b_with_multimodule_communicator` schedule
- Most flexible but requires more GPUs
- Modules can have different PP stages

**Schedule:** `forward_backward_1f1b_with_multimodule_communicator`

---

## Configuration Structure

MIMO uses configuration dataclasses for defining parallelism per module:

```python
from megatron.bridge.models.mimo import (
    MimoParallelismConfig,
    ModuleParallelismConfig,
)

# 1. Per-module parallelism configuration
vision_parallelism = ModuleParallelismConfig(
    tensor_parallel=2,
    pipeline_parallel=1,
    context_parallel=1,
    expert_parallel=1,
    data_parallel=4,      # Can be None, computed from world_size
    rank_offset=0,        # GPU offset for this module
)

llm_parallelism = ModuleParallelismConfig(
    tensor_parallel=8,
    pipeline_parallel=1,
    data_parallel=1,
    rank_offset=0,        # 0 = colocated with vision
)

# 2. Global MIMO configuration
mimo_config = MimoParallelismConfig(
    llm_module_name="language_module",
    deployment_mode="colocated",  # or "homogeneous" or "heterogeneous"
    module_parallelisms={
        "images": vision_parallelism,
        "language_module": llm_parallelism,
    },
    special_token_ids={"images": 32000},  # Special tokens for each modality
)

# 3. Finalize configuration (computes data_parallel if not set, validates constraints)
mimo_config.finalize(world_size=dist.get_world_size())
```

**Validation performed by `finalize()`:**
- Computes `data_parallel` if not explicitly set
- Validates mode-specific constraints:
  - **Homogeneous**: All modules must have identical TP/PP/CP/EP/DP
  - **Colocated**: All modules must have PP=1 and same total_ranks
  - **Heterogeneous**: No overlapping rank ranges
- Ensures world_size matches total required ranks
- Checks for rank assignment gaps (warns or errors)

---

## Data Loading Pattern

MIMO uses **DP-aware data iterators** that handle heterogeneous batch sizes across modules:

### Basic Usage

```python
from megatron.bridge.models.mimo import get_data_iterator, get_batch

# Data configuration
data_config = DataConfig(
    base_batch_size=4,        # LLM micro-batch per DP replica
    num_microbatches=8,       # For gradient accumulation
    seq_length=8192,          # LLM sequence length
    image_seq_length=1024,    # Encoder sequence length
    vocab_size=32000,
    image_special_token_id=32000,
    dataset_size=2048,
    num_workers=4,
    prefetch_factor=2,
)

# Create iterator (automatically determines DP rank and batch size)
data_iterator = get_data_iterator(
    model_config=model_config,
    data_config=data_config,
    module_to_grid_map=module_to_grid_map,
)

# In training loop
for iteration in range(num_iterations):
    batch = get_batch(data_iterator)
    # batch automatically has correct size for this rank's module
```

### Batch Size Calculation

The data loader automatically calculates appropriate batch sizes for each module:

```python
# Global batch size (consistent across all modules)
global_batch_size = base_batch_size * llm_dp_size

# Per-module micro-batch size
for each module:
    micro_batch_size = global_batch_size / module_dp_size
```

**Validation:**
- `global_batch_size` must be divisible by all module DP sizes
- Raises `ValueError` if batch configuration is incompatible

**Example:**
```python
# Configuration
base_batch_size = 4
llm_dp_size = 2
encoder_dp_size = 8

# Calculated values
global_batch_size = 4 * 2 = 8
llm_micro_batch_size = 8 / 2 = 4
encoder_micro_batch_size = 8 / 8 = 1
```

### Data Loading Modes

**Homogeneous/Heterogeneous Mode:**
- Each module loads data with its own DP group
- Each DP rank gets unique portion of global batch
- No redistribution needed

**Colocated Mode:**
- Loads with the DP that has **fewer replicas** (larger batch per rank)
- Forward step handles slicing for the module with more DP replicas
- Reduces data loading overhead

```python
from megatron.bridge.models.mimo import get_data_iterator_colocated

# Use colocated-specific iterator
data_iterator = get_data_iterator_colocated(
    model_config=model_config,
    data_config=data_config,
    module_to_grid_map=module_to_grid_map,
)
```

### Colocated Data Slicing

For colocated mode with different DP sizes, the forward step must slice data appropriately:

```python
def _slice_batch_dim(data, start, size, batch_dim=0):
    """Recursively slice tensors along batch dimension."""
    if isinstance(data, torch.Tensor):
        return data.narrow(batch_dim, start, size).contiguous()
    elif isinstance(data, dict):
        return {k: _slice_batch_dim(v, start, size, batch_dim) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(_slice_batch_dim(v, start, size, batch_dim) for v in data)
    return data


def forward_step(data_iterator, model, encoder_grid=None, llm_grid=None):
    """Forward step with automatic data slicing for colocated mode.

    Handles two patterns:
    - Fan-in (encoder_dp > llm_dp): Slice modality_inputs for encoder
    - Fan-out (encoder_dp < llm_dp): Slice input_ids/labels/loss_mask for LLM
    """
    data_batch = get_batch(data_iterator)
    if data_batch is None:
        return None, None

    encoder_dp = encoder_grid.get_pg("dp").size()
    llm_dp = llm_grid.get_pg("dp").size()
    encoder_dp_rank = encoder_grid.get_pg("dp").rank()
    llm_dp_rank = llm_grid.get_pg("dp").rank()

    if encoder_dp > llm_dp:
        # Fan-in: encoder has more DP replicas
        # Slice modality_inputs (vision data) for encoder
        dp_ratio = encoder_dp // llm_dp
        slice_size = data_batch["modality_inputs"]["images"]["clip_encoder"]["hidden_states"].shape[1] // dp_ratio
        start_idx = encoder_dp_rank % dp_ratio * slice_size

        data_batch["modality_inputs"] = _slice_batch_dim(
            data_batch["modality_inputs"], start_idx, slice_size, batch_dim=1
        )

    elif encoder_dp < llm_dp:
        # Fan-out: LLM has more DP replicas
        # Slice LLM inputs (input_ids, labels, loss_mask)
        dp_ratio = llm_dp // encoder_dp
        slice_size = data_batch["input_ids"].shape[0] // dp_ratio
        start_idx = llm_dp_rank % dp_ratio * slice_size

        data_batch["input_ids"] = _slice_batch_dim(data_batch["input_ids"], start_idx, slice_size)
        data_batch["labels"] = _slice_batch_dim(data_batch["labels"], start_idx, slice_size)
        data_batch["loss_mask"] = _slice_batch_dim(data_batch["loss_mask"], start_idx, slice_size)

    output_tensor, loss_mask = model(**data_batch)
    return output_tensor, partial(loss_func, loss_mask)
```

---

## Training Patterns

### Homogeneous Training

```python
from megatron.bridge.models.mimo import get_vlm_mimo_model_homogeneous
import megatron.core.pipeline_parallel.schedules as schedule
from megatron.bridge.models.mimo import (
    multimodule_no_sync,
    finalize_model_grads,
    get_module_to_grid_tuple,
)

# 1. Create model (returns shared grid and pg_collection)
mimo_model, shared_grid, shared_pg_collection, topology = \
    get_vlm_mimo_model_homogeneous(model_config, seq_len)

# 2. Create optimizer
optimizer = torch.optim.AdamW(mimo_model.parameters(), lr=1e-4)

# 3. Create data iterator
data_iterator = get_data_iterator(
    model_config=model_config,
    data_config=data_config,
    module_to_grid_map={
        "images": shared_grid,
        "language_module": shared_grid,
    }
)

# 4. Set up gradient synchronization helpers
module_to_grid_tuple = get_module_to_grid_tuple(
    mimo_model,
    module_to_grid_map={"images": shared_grid, "language_module": shared_grid},
)

# 5. Training loop
for iteration in range(num_iterations):
    # Forward-backward pass (no pipelining since PP=1)
    losses = schedule.forward_backward_no_pipelining(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=mimo_model,
        num_microbatches=data_config.num_microbatches,
        seq_length=data_config.seq_length,
        forward_only=False,
    )

    # Gradient synchronization (prevents duplicate all-reduces)
    with multimodule_no_sync(module_to_grid_tuple, iteration, data_config.num_microbatches):
        optimizer.step()

    # Finalize gradients (ensures proper averaging)
    finalize_model_grads(module_to_grid_tuple, iteration, data_config.num_microbatches)

    optimizer.zero_grad()
```

### Colocated Training

```python
from megatron.bridge.models.mimo import get_vlm_mimo_model_colocated
from functools import partial

# 1. Create model with colocated communication
mimo_model, module_to_grid_map, topology = \
    get_vlm_mimo_model_colocated(model_config, seq_len)

# 2. Create optimizer
optimizer = torch.optim.AdamW(mimo_model.parameters(), lr=1e-4)

# 3. Create colocated data iterator
data_iterator = get_data_iterator_colocated(
    model_config=model_config,
    data_config=data_config,
    module_to_grid_map=module_to_grid_map,
)

# 4. Set up gradient synchronization
module_to_grid_tuple = get_module_to_grid_tuple(mimo_model, module_to_grid_map)

# 5. Training loop with data slicing
for iteration in range(num_iterations):
    # Forward step handles data redistribution between modules
    forward_step_with_slicing = partial(
        forward_step,
        encoder_grid=module_to_grid_map["images"],
        llm_grid=module_to_grid_map["language_module"],
    )

    losses = schedule.forward_backward_no_pipelining(
        forward_step_func=forward_step_with_slicing,
        data_iterator=data_iterator,
        model=mimo_model,
        num_microbatches=data_config.num_microbatches,
        seq_length=data_config.seq_length,
        forward_only=False,
    )

    # Gradient synchronization
    with multimodule_no_sync(module_to_grid_tuple, iteration, data_config.num_microbatches):
        optimizer.step()

    finalize_model_grads(module_to_grid_tuple, iteration, data_config.num_microbatches)
    optimizer.zero_grad()
```

### Heterogeneous Training with Pipeline Parallelism

```python
from megatron.bridge.models.mimo import get_vlm_mimo_model
from megatron.core.pipeline_parallel.multimodule_communicator import (
    MultiModulePipelineCommunicator,
)

# 1. Create model with separate grids
mimo_model, module_to_grid_map, topology = \
    get_vlm_mimo_model(model_config, seq_len)

# 2. Create optimizer
optimizer = torch.optim.AdamW(mimo_model.parameters(), lr=1e-4)

# 3. Create data iterator
data_iterator = get_data_iterator(
    model_config=model_config,
    data_config=data_config,
    module_to_grid_map=module_to_grid_map,
)

# 4. Create multi-module communicator for pipeline communication
module_to_model_map = {}
if mimo_model.modality_submodules.get("images"):
    module_to_model_map["images"] = mimo_model.modality_submodules["images"]
if mimo_model.language_model:
    module_to_model_map["language_module"] = mimo_model.language_model

communicator = MultiModulePipelineCommunicator(
    topology=topology,
    module_to_grid_map=module_to_grid_map,
    module_to_model_map=module_to_model_map,
)

# 5. Set up gradient synchronization
module_to_grid_tuple = get_module_to_grid_tuple(mimo_model, module_to_grid_map)

# 6. Training loop with 1F1B schedule
for iteration in range(num_iterations):
    losses = schedule.forward_backward_1f1b_with_multimodule_communicator(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=mimo_model,
        communicator=communicator,
        num_microbatches=data_config.num_microbatches,
        seq_length=data_config.seq_length,
        forward_only=False,
    )

    # Gradient synchronization
    with multimodule_no_sync(module_to_grid_tuple, iteration, data_config.num_microbatches):
        optimizer.step()

    finalize_model_grads(module_to_grid_tuple, iteration, data_config.num_microbatches)
    optimizer.zero_grad()
```

---

## Performance Monitoring

MIMO includes comprehensive performance monitoring for ablation studies:

```python
from megatron.bridge.models.mimo import create_performance_monitor

# 1. Create performance monitor
perf_monitor = create_performance_monitor(
    model_config=model_config,
    runtime_config=runtime_config,
    module_to_grid_map=module_to_grid_map,
)

# 2. Training loop with monitoring
for iteration in range(num_iterations):
    # Start step timing
    perf_monitor.start_step()

    # Forward-backward phase
    perf_monitor.start_phase("forward_backward")
    losses = schedule.forward_backward_no_pipelining(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=mimo_model,
        num_microbatches=data_config.num_microbatches,
        seq_length=data_config.seq_length,
    )
    perf_monitor.end_phase("forward_backward")

    # Optimizer phase
    perf_monitor.start_phase("optimizer")
    with multimodule_no_sync(module_to_grid_tuple, iteration, data_config.num_microbatches):
        optimizer.step()
    finalize_model_grads(module_to_grid_tuple, iteration, data_config.num_microbatches)
    optimizer.zero_grad()
    perf_monitor.end_phase("optimizer")

    # End step and compute metrics
    perf_monitor.end_step(iteration)

    # Periodic logging
    if iteration % runtime_config.log_interval == 0:
        metrics = perf_monitor.log_metrics(iteration)
        if dist.get_rank() == 0:
            print(f"Iteration {iteration}: {metrics}")

# 3. Save metrics to CSV for analysis
perf_monitor.save_metrics(runtime_config.metrics_output_dir)
```

**Metrics collected:**
- **Timing**: Per-iteration timing for forward, backward, optimizer, communication phases
- **Throughput**: Tokens/sec, samples/sec (both per-GPU and aggregate)
- **Memory**: GPU memory usage (allocated, reserved, peak)
- **Communication**: All-reduce time, P2P communication time (for PP)
- **Module-specific**: Separate metrics for each module (encoder, LLM)
- **Loss**: Training loss and validation loss (if enabled)

**Output format:**
CSV file with columns:
```
iteration,total_time,forward_time,backward_time,optimizer_time,tokens_per_sec,samples_per_sec,memory_allocated,memory_reserved,loss,...
```

---

## YAML Configuration and Experiment Runner

MIMO supports YAML-based configuration for running systematic ablation studies:

### YAML Configuration Format

```yaml
# configs/ablations/llm_7b/colocated_exp.yaml

model:
  # Module architectures
  module_architectures:
    images:
      num_layers: 40
      hidden_size: 1408
      num_attention_heads: 16
      seq_length: 1024
      vocab_size: 0

    language_module:
      num_layers: 32
      hidden_size: 4096
      num_attention_heads: 32
      seq_length: 8192
      vocab_size: 32000

  # Module parallelism configurations
  module_parallelisms:
    images:
      tensor_parallel: 1
      pipeline_parallel: 1
      data_parallel: 8

    language_module:
      tensor_parallel: 8
      pipeline_parallel: 1
      data_parallel: 1

  # Special tokens
  special_token_ids:
    images: 32000

  # Module naming
  llm_module_name: "language_module"
  encoder_module_name: "images"
  llm_rank_offset: 0  # 0 = colocated with encoder

data:
  base_batch_size: 4
  num_microbatches: 8
  seq_length: 8192
  image_seq_length: 1024
  vocab_size: 32000
  image_special_token_id: 32000
  dataset_size: 2048
  num_workers: 4
  prefetch_factor: 2

runtime:
  num_iterations: 10
  warmup_iterations: 2
  log_interval: 1
  enable_performance_monitoring: true
  pipeline_schedule: "colocated"  # or "homogeneous" or "1f1b"
  enable_profiling: false
  use_pytorch_profiler: false
  profile_start_step: 3
  profile_end_step: 8
  metrics_output_dir: "./logs/ablations/llm_7b"
  tensorboard_dir: "./logs/ablations/llm_7b/tb"
```

### Configuration Inheritance

Create a `baseline.yaml` with common settings, then override specific values in experiment configs:

```yaml
# configs/ablations/llm_7b/baseline.yaml
model:
  module_architectures:
    # Common architecture settings
    images:
      num_layers: 40
      hidden_size: 1408
      num_attention_heads: 16
      vocab_size: 0

    language_module:
      num_layers: 32
      hidden_size: 4096
      num_attention_heads: 32
      vocab_size: 32000

  # Default parallelism (will be overridden)
  module_parallelisms:
    images:
      tensor_parallel: 4
      pipeline_parallel: 1
      data_parallel: 2

    language_module:
      tensor_parallel: 4
      pipeline_parallel: 1
      data_parallel: 2

# Common settings inherited by all experiments
data:
  base_batch_size: 4
  num_microbatches: 8
  vocab_size: 32000

runtime:
  num_iterations: 10
  warmup_iterations: 2
```

```yaml
# configs/ablations/llm_7b/exp_tp8_dp1.yaml
# Inherits from baseline.yaml, overrides specific values

model:
  module_parallelisms:
    images:
      tensor_parallel: 1
      data_parallel: 8

    language_module:
      tensor_parallel: 8
      data_parallel: 1

runtime:
  pipeline_schedule: "colocated"
  metrics_output_dir: "./logs/ablations/llm_7b/exp_tp8_dp1"
```

### Running Experiments

```bash
# Run single experiment
uv run python -m torch.distributed.run --nproc_per_node=8 \
    tests/unit_tests/models/heterogenous_parallel/experiment_runner.py \
    --config configs/ablations/llm_7b/exp_tp8_dp1.yaml \
    --results-dir logs/mimo_ablations

# Run all experiments in a directory
uv run python -m torch.distributed.run --nproc_per_node=8 \
    tests/unit_tests/models/heterogenous_parallel/experiment_runner.py \
    --experiments-dir configs/ablations/llm_7b/ \
    --results-dir logs/mimo_ablations

# Results saved to:
# logs/mimo_ablations/run_<timestamp>/
#   ├── exp_tp8_dp1/
#   │   ├── config.yaml          # Copy of experiment config
#   │   ├── experiment_info.json # Metadata
#   │   └── metrics.csv          # Performance metrics
#   ├── exp_tp4_dp2/
#   │   └── ...
#   └── all_experiments.csv      # Combined results for analysis
```

**Experiment runner features:**
- **Automatic config loading** with inheritance from baseline.yaml
- **Experiment naming** based on config filename
- **Result aggregation** - Combines all CSVs into one for analysis
- **Error handling** - Saves error.json if experiment fails
- **GPU memory cleanup** - Clears CUDA cache between experiments

### Loading and Parsing Configs

```python
from megatron.bridge.models.mimo import load_experiment_config

# Load config from YAML
model_config, data_config, runtime_config = load_experiment_config(
    config_path="configs/ablations/llm_7b/exp_tp8_dp1.yaml",
    experiment_dir="./logs/exp_tp8_dp1",
)

# Configs are parsed into dataclass instances
print(f"Deployment mode: {model_config.deployment_mode}")
print(f"LLM TP: {model_config.get_parallelism('language_module').tensor_parallel}")
print(f"Encoder TP: {model_config.get_parallelism('images').tensor_parallel}")
```

---

## Model Provider Pattern

### Creating a MIMO Model Provider

```python
from megatron.bridge.models.mimo import MimoModelProvider
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt import GPTModel

# 1. Define model specs (like Megatron Core examples)
language_model_spec = ModuleSpec(
    module=GPTModel,
    params={
        "config": llm_transformer_config,
        "transformer_layer_spec": gpt_layer_spec,
        "vocab_size": 32000,
        "max_sequence_length": 8192,
        "pre_process": True,
        "post_process": True,
        "pg_collection": None,  # Will be injected by provider
    }
)

vision_encoder_spec = ModuleSpec(
    module=TransformerBlock,
    params={
        "config": vision_transformer_config,
        "spec": vision_layer_spec,
        "pg_collection": None,  # Will be injected by provider
    }
)

vision_projection_spec = ModuleSpec(
    module=MultimodalProjector,
    params={
        "config": projection_config,
        "submodules": projection_layer_spec.submodules,
        "projector_type": "mlp",
        "input_size": vision_hidden_size,
        "tp_group": None,  # Will be injected by provider
    }
)

vision_submodule_spec = ModuleSpec(
    module=VisionModalitySubmodules,
    submodules={
        "encoders": {"clip_encoder": vision_encoder_spec},
        "input_projections": [vision_projection_spec],
    }
)

# 2. Create MIMO provider
provider = MimoModelProvider(
    language_model_spec=language_model_spec,
    modality_submodules_spec={
        "images": vision_submodule_spec,
    },
    special_token_ids={"images": 32000},
    mimo_parallelism_config=mimo_config,
)

# 3. Finalize parallelism config
mimo_config.finalize(world_size=dist.get_world_size())

# 4. Provide model (injects pg_collections, creates grids)
result = provider.provide()

# 5. Access components
model = result.model  # MimoModel instance (None on non-participating ranks)
module_to_grid_map = result.module_to_grid_map  # Dict[str, HyperCommGrid]
topology = result.topology  # Dict[str, List[str]] - module data flow DAG
pg_collections = result.pg_collections  # Dict[str, ProcessGroupCollection]
```

**What the provider does:**
1. Creates HyperCommGrids for each module based on parallelism config
2. Extracts ProcessGroupCollections from grids
3. Injects `pg_collection` into model specs
4. Instantiates modules on participating ranks only
5. Wraps with DDP (if specified)
6. Returns result container with model and infrastructure

### Per-Encoder Parallelism

For models with multiple encoders (e.g., CLIP + DINO), define separate parallelism for each:

```python
mimo_config = MimoParallelismConfig(
    llm_module_name="llm",
    deployment_mode="heterogeneous",
    module_parallelisms={
        "llm": ModuleParallelismConfig(
            tensor_parallel=8,
            rank_offset=0,
        ),
        "clip_encoder": ModuleParallelismConfig(
            tensor_parallel=2,
            rank_offset=8,  # Different GPUs
        ),
        "dino_encoder": ModuleParallelismConfig(
            tensor_parallel=4,
            rank_offset=12,  # Different GPUs
        ),
    },
    special_token_ids={
        "clip_encoder": 32000,
        "dino_encoder": 32001,
    },
)

provider = MimoModelProvider(
    language_model_spec=gpt_spec,
    modality_submodules_spec={
        "clip_encoder": clip_spec,   # Gets TP=2
        "dino_encoder": dino_spec,   # Gets TP=4
    },
    special_token_ids=mimo_config.special_token_ids,
    mimo_parallelism_config=mimo_config,
)
```

**Note:** Each key in `modality_submodules_spec` must match a module name in `module_parallelisms`.

---

## Common Patterns and Helpers

### Checking Rank Participation

```python
from megatron.bridge.models.mimo import is_current_rank_in_grid
import torch.distributed as dist

rank = dist.get_rank()
for module_name, grid in module_to_grid_map.items():
    if is_current_rank_in_grid(rank, grid):
        print(f"Rank {rank} participates in module: {module_name}")
        print(f"  TP rank: {grid.get_pg('tp').rank()}/{grid.get_pg('tp').size()}")
        print(f"  PP rank: {grid.get_pg('pp').rank()}/{grid.get_pg('pp').size()}")
        print(f"  DP rank: {grid.get_pg('dp').rank()}/{grid.get_pg('dp').size()}")
```

### Validating Batch Configuration

```python
from megatron.bridge.models.mimo import validate_heterogeneous_batch_config

# Validates that global_batch_size is divisible by all module DP sizes
global_batch_size = validate_heterogeneous_batch_config(
    model_config=model_config,
    data_config=data_config,
)

print(f"Global batch size: {global_batch_size}")

# Prints per-module micro-batch sizes:
# Module 'images': DP=8, micro_batch_size=1 (per DP replica)
# Module 'language_module': DP=1, micro_batch_size=8 (per DP replica)
```

### Gradient Synchronization Helpers

```python
from megatron.bridge.models.mimo import (
    get_module_to_grid_tuple,
    multimodule_no_sync,
    finalize_model_grads,
    zero_grad_buffer_for_multimodule,
)

# Set up module-to-grid mapping for gradient sync
module_to_grid_tuple = get_module_to_grid_tuple(
    mimo_model,
    module_to_grid_map,
)

# In training loop
for iteration in range(num_iterations):
    # Forward-backward pass
    losses = schedule.forward_backward_no_pipelining(...)

    # Prevent duplicate all-reduces during gradient accumulation
    with multimodule_no_sync(module_to_grid_tuple, iteration, num_microbatches):
        optimizer.step()

    # Finalize gradients (ensures proper averaging across DP)
    finalize_model_grads(module_to_grid_tuple, iteration, num_microbatches)

    # Zero gradients
    zero_grad_buffer_for_multimodule(module_to_grid_tuple)
    optimizer.zero_grad()
```

### Profiling MIMO Training

```python
runtime_config = RuntimeConfig(
    num_iterations=20,
    enable_profiling=True,
    use_pytorch_profiler=True,
    profile_start_step=5,
    profile_end_step=15,
    tensorboard_dir="./tb_logs",
)

# PyTorch profiler will automatically run during specified steps
# View results with: tensorboard --logdir=./tb_logs
```

---

## Important Notes and Best Practices

### Deployment Mode Selection

- **Start with homogeneous** for baseline experiments and debugging
- **Use colocated** for most VLMs (small encoder + large LLM)
- **Use heterogeneous** only when pipeline parallelism is required for memory

### Batch Size Constraints

- `global_batch_size` must be divisible by all module DP sizes
- Set `base_batch_size` based on LLM DP size
- Increase `num_microbatches` for gradient accumulation

Example:
```python
# LLM: DP=2, Encoder: DP=8
# global_batch_size = base_batch_size * llm_dp = 4 * 2 = 8
# LLM micro-batch: 8 / 2 = 4
# Encoder micro-batch: 8 / 8 = 1
base_batch_size = 4
```

### Colocated Mode Requirements

- **PP=1** required for all modules
- **Same total_ranks** across all modules
- **rank_offset=0** for all modules (same GPUs)
- Use `get_data_iterator_colocated()` for data loading
- Forward step must handle data slicing

### Data Loading

- **Homogeneous/Heterogeneous**: Use `get_data_iterator()`
- **Colocated**: Use `get_data_iterator_colocated()`
- Data loader only created on first/last PP stage ranks
- Automatic DP-aware sampling ensures correct batch distribution

### Gradient Synchronization

- Always use `multimodule_no_sync()` to prevent duplicate all-reduces
- Call `finalize_model_grads()` after optimizer step
- Use `zero_grad_buffer_for_multimodule()` to clear gradient buffers

### Performance Optimization

- **Colocated mode** typically has best GPU utilization for asymmetric models
- **Fan-in pattern** (encoder_dp > llm_dp) often optimal for small encoders
- Enable `persistent_workers=True` in data config for faster data loading
- Use `prefetch_factor=2` or higher to overlap data loading with compute

### Debugging Tips

```python
# 1. Check which module this rank participates in
rank = dist.get_rank()
for module_name, grid in module_to_grid_map.items():
    if is_current_rank_in_grid(rank, grid):
        print(f"Rank {rank} -> Module: {module_name}")

# 2. Validate batch configuration before training
global_batch_size = validate_heterogeneous_batch_config(model_config, data_config)

# 3. Print parallelism info
print(f"World size: {dist.get_world_size()}")
print(f"Expected world size: {mimo_config.total_world_size}")

# 4. Check model instantiation
if result.model is None:
    print(f"Rank {rank}: Not participating in any module")
else:
    print(f"Rank {rank}: Model instantiated")

# 5. Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Common Pitfalls

1. **Forgetting to call `finalize()`** on mimo_config before providing model
2. **Wrong data iterator** - Using regular iterator for colocated mode
3. **Missing data slicing** - Forward step must slice data for colocated mode
4. **Incorrect batch sizes** - Ensure global_batch is divisible by all DP sizes
5. **Gradient accumulation** - Must use `multimodule_no_sync()` correctly

---

## File Locations

**Configuration and builders:**
- `src/megatron/bridge/models/mimo/mimo_config.py` - Config dataclasses
- `src/megatron/bridge/models/mimo/mimo_provider.py` - Main provider
- `src/megatron/bridge/models/mimo/mimo_builder.py` - Grid and PG builders

**Data loading:**
- `src/megatron/bridge/training/mimo_data.py` - DP-aware iterators (planned)
- Prototype: `tests/unit_tests/models/heterogenous_parallel/dp_aware_data_iterator.py`

**Training helpers:**
- `src/megatron/bridge/training/mimo_ddp.py` - DDP wrapping
- Prototype: `tests/unit_tests/models/heterogenous_parallel/parallel_utils.py`

**Experiment infrastructure:**
- Prototype: `tests/unit_tests/models/heterogenous_parallel/experiment_runner.py`
- Prototype: `tests/unit_tests/models/heterogenous_parallel/performance_utils.py`
- Prototype: `tests/unit_tests/models/heterogenous_parallel/config_loader.py`

**Model specifications:**
- Prototype: `tests/unit_tests/models/heterogenous_parallel/model_specs.py`
- Reference: `examples/mimo/configs/llava_vlm.py`
