# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**NeMo Megatron Bridge** is a PyTorch-native library for training and converting LLM/VLM models. It serves as a bridge between 🤗 Hugging Face and Megatron Core, enabling:
- Bidirectional checkpoint conversion (HF ↔ Megatron)
- High-performance distributed training with tensor/pipeline/context/expert parallelism
- Pre-configured training recipes for popular models
- PEFT methods (LoRA, DoRA) for efficient fine-tuning

## Build and Development Commands

### Environment Setup
```bash
# Using NeMo Framework container (recommended)
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash --gpus all \
  nvcr.io/nvidia/nemo:${TAG}

# Or build development container
docker build -f docker/Dockerfile.ci -t megatron-bridge .
docker run --rm -it -w /workdir -v $(pwd):/opt/Megatron-Bridge \
  --entrypoint bash --gpus all megatron-bridge

# For local development, install with uv
uv sync
```

### Testing
```bash
# Run all tests (linting, unit, functional)
bash scripts/run_ci_tests.sh

# Run only unit tests
uv run pytest tests/unit_tests/ -v

# Run specific test file
uv run pytest tests/unit_tests/training/test_model_load_save.py -v

# Run specific test function
uv run pytest tests/unit_tests/training/test_model_load_save.py::test_function_name -v

# Run functional tests (requires GPUs)
uv run pytest tests/functional_tests/ -v

# Run with coverage
uv run pytest --cov=megatron.bridge --cov-report=html tests/

# Run tests with specific markers
uv run pytest -m unit  # Only unit tests
uv run pytest -m integration  # Only integration tests
```

### Linting and Formatting
```bash
# Auto-fix linting and formatting issues
uv run ruff check --fix .
uv run ruff format .

# Run pre-commit hooks manually
uv run pre-commit run --all-files

# Install pre-commit hooks
uv run pre-commit install
```

### Dependency Management
```bash
# Add new required dependency
uv add <package-name>

# Add optional dependency to specific group
uv add --optional --extra <group-name> <package-name>

# Update dependencies (regenerates uv.lock)
# Commit both pyproject.toml and uv.lock after changes
```

### Documentation
```bash
# Build docs locally
cd docs
uv run sphinx-build -b html . _build

# Live-reload docs during development
uv run sphinx-autobuild . _build
```

### Running Training
```bash
# Single-node training with torchrun
torchrun --nproc-per-node=8 /path/to/train_script.py

# Multi-node training (example with slurm)
srun --ntasks-per-node=8 --gpus-per-node=8 python /path/to/train_script.py
```

## Architecture Overview

### Core Abstractions

#### 1. **AutoBridge** - Conversion Entry Point
The main user-facing API for HF ↔ Megatron conversion. Auto-detects model architectures.

```python
from megatron.bridge import AutoBridge

# Load HF model and get bridge
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")

# Convert to Megatron provider and configure parallelism
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 8
provider.pipeline_model_parallel_size = 4
provider.finalize()

# Instantiate model with weights loaded
model = provider.provide_distributed_model(wrap_with_ddp=False)

# Export back to HF
bridge.save_hf_pretrained(model, "./exported_model")
```

**Key files:**
- `src/megatron/bridge/models/conversion/auto_bridge.py`
- `src/megatron/bridge/models/conversion/model_bridge.py`

#### 2. **ModelProvider** - Model Instantiation Pattern
Standardized pattern for creating Megatron models with distributed training support.

All model providers inherit from `ModelProviderMixin` and implement:
- `provide()` - Creates the actual Megatron Core model
- `provide_distributed_model()` - Wraps with DDP/FSDP and handles parallelism setup
- Configuration via dataclass fields (inherits from `TransformerConfig`)

**Key files:**
- `src/megatron/bridge/models/model_provider.py`
- `src/megatron/bridge/models/gpt/gpt_provider.py` (example)
- `src/megatron/bridge/models/llama/llama_provider.py` (example)

Model-specific providers live in `src/megatron/bridge/models/<model_name>/` directories.

#### 3. **MegatronModelBridge** - Conversion Implementation
Each model architecture has a bridge that defines:
- `provider_bridge()` - Maps HF config → Megatron ModelProvider
- `mapping_registry()` - Defines parameter name/shape mappings

Bridges handle:
- Weight transformations (QKV fusion, gated MLP, etc.)
- Tensor parallelism distribution (column/row/replicated)
- Pipeline parallelism routing

**Key files:**
- Model bridges: `src/megatron/bridge/models/<model_name>/<model_name>_bridge.py`
- Registry system: `src/megatron/bridge/models/conversion/registry.py`

#### 4. **ConfigContainer** - Training Configuration
Central configuration object holding all training settings. Brings together:

```python
from megatron.bridge.training.config import ConfigContainer

config = ConfigContainer(
    model=model_provider,      # ModelProvider (GPT, T5, etc.)
    dataset=dataset_provider,  # DatasetProvider
    train=training_config,     # Batch sizes, iterations, validation
    optimizer=optimizer_config,# Optimizer settings
    scheduler=scheduler_config,# LR scheduling
    logger=logger_config,      # Logging, TensorBoard, W&B
    tokenizer=tokenizer_config,# Tokenizer settings
    checkpoint=checkpoint_cfg, # Checkpointing config
    ddp=ddp_config,           # DDP settings
    mixed_precision=mp_cfg,   # FP8, BF16 settings
    peft=peft_config,         # LoRA, DoRA, etc. (optional)
    # ... more optional configs
)
```

**Key file:** `src/megatron/bridge/training/config.py`

#### 5. **Recipes** - Pre-configured Training
Recipes are factory functions that return `ConfigContainer` with production-ready settings.

```python
from megatron.bridge.recipes.llama import llama32_1b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain

# Get recipe config
config = llama32_1b_pretrain_config(seq_length=8192)

# Customize as needed
config.train.train_iters = 10000
config.dataset.data_paths = ["/path/to/data"]

# Train
pretrain(config, forward_step)
```

**Recipe locations:**
- `src/megatron/bridge/recipes/<model_family>/` - Recipe implementations
- `examples/recipes/` - Example usage with YAML overrides

### Training Loop Flow

```
pretrain()
  └─> setup()
       ├─> Initialize distributed (if needed)
       ├─> Build model via provider.provide_distributed_model()
       ├─> Load checkpoint (if resuming)
       ├─> Build datasets via dataset_provider.build_train_valid_test_datasets()
       └─> Create optimizer and scheduler
  └─> train()
       ├─> Iterate training steps
       ├─> Get batch from data_iterator
       ├─> forward_step_func() - compute loss
       ├─> train_step() - backward + optimizer step
       ├─> Periodic evaluation
       └─> Periodic checkpointing
  └─> evaluate_and_print_results()
```

**Key files:**
- `src/megatron/bridge/training/pretrain.py` - Main entry point
- `src/megatron/bridge/training/train.py` - Training loop
- `src/megatron/bridge/training/gpt_step.py` - GPT forward step
- `src/megatron/bridge/training/finetune.py` - SFT entry point

### Parallelism Configuration

All parallelism is configured via the `ModelProvider` before calling `finalize()`:

```python
provider.tensor_model_parallel_size = 8        # TP: split weights across 8 GPUs
provider.pipeline_model_parallel_size = 4      # PP: split layers across 4 stages
provider.context_parallel_size = 2             # CP: split sequence across 2 GPUs
provider.expert_model_parallel_size = 8        # EP: split MoE experts across 8 GPUs
provider.virtual_pipeline_model_parallel_size = 2  # VPP: interleaved pipeline
provider.sequence_parallel = True              # SP: additional memory optimization
```

Process groups are initialized in `provide_distributed_model()`:
- Calls `parallel_state.initialize_model_parallel()` with parallelism sizes
- Creates TP/PP/CP/EP/DP process groups
- Remaining GPUs used for data parallelism

**Key files:**
- `src/megatron/bridge/models/model_provider.py` - Provider base with parallelism setup
- Integration with Megatron Core's `megatron/core/parallel_state.py`

### Data Loading Pattern

Custom datasets implement `DatasetProvider`:

```python
from megatron.bridge.training.config import DatasetProvider

@dataclass
class MyDatasetProvider(DatasetProvider):
    data_path: str

    def build_train_valid_test_datasets(self, context):
        # context provides: train_samples, valid_samples, test_samples, tokenizer
        train_ds = MyDataset(self.data_path, context.train_samples)
        valid_ds = MyDataset(self.data_path, context.valid_samples)
        test_ds = MyDataset(self.data_path, context.test_samples)
        return train_ds, valid_ds, test_ds
```

Built-in providers:
- `GPTDatasetConfig` - Pretraining datasets (blended, mock)
- `SFTDatasetConfig` - Supervised fine-tuning datasets
- Located in `src/megatron/bridge/data/datasets/`

### PEFT (LoRA, DoRA, etc.)

PEFT methods are applied via `pre_wrap_hook` when creating models:

```python
from megatron.bridge.peft import LoRA

lora = LoRA(
    target_modules=["linear_qkv", "linear_proj"],
    lora_rank=16,
    lora_alpha=32,
)

model = provider.provide_distributed_model(
    wrap_with_ddp=False,
    pre_wrap_hook=lambda m: lora(m, training=True)
)
```

Or configured in recipes:
```python
config = llama32_1b_finetune_config(
    peft=LoRA(target_modules=["linear_qkv"], lora_rank=16)
)
```

**Key files:**
- `src/megatron/bridge/peft/base.py` - PEFT base abstraction
- `src/megatron/bridge/peft/lora.py` - LoRA implementation
- `src/megatron/bridge/peft/dora.py` - DoRA implementation

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

#### Deployment Modes

MIMO supports three deployment modes, each optimized for different use cases:

**1. Homogeneous Mode**
All modules use **identical parallelism** (same TP/PP/DP). Simplest configuration, uses shared process groups.

**When to use:**
- Baseline experiments
- When encoder and LLM have similar sizes
- Debugging new features

**2. Colocated Mode**
Modules use **different TP/DP but same GPUs** (PP=1 for all). Uses `ColocatedBridgeCommunicator` for automatic data redistribution.

**When to use:**
- Small encoder with large LLM (e.g., 1B encoder + 7B LLM)
- Encoder needs less TP, more DP for throughput
- Want to maximize GPU utilization without pipeline parallelism

**Characteristics:**
- Different grids but rank_offset=0 (same GPUs)
- PP=1 required for all modules
- Total ranks must match across modules
- Automatic data redistribution via `ColocatedBridgeCommunicator`
- Fan-in (encoder_dp > llm_dp) or fan-out (encoder_dp < llm_dp) patterns

**3. Heterogeneous Mode**
Modules use **different GPUs with pipeline parallelism**. Most flexible but most complex.

**When to use:**
- Large models requiring PP for memory
- Different modules have vastly different sizes
- Need maximum parallelism flexibility

---

#### Configuration Example

```python
from megatron.bridge.models.mimo import MimoParallelismConfig, ModuleParallelismConfig

# Colocated configuration example
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

# Finalize configuration
mimo_config.finalize(world_size=dist.get_world_size())
```

---

#### Data Loading Pattern

MIMO uses **DP-aware data iterators** that handle heterogeneous batch sizes:

```python
from megatron.bridge.models.mimo import get_data_iterator, get_batch

# Create iterator (automatically determines DP rank and batch size)
data_iterator = get_data_iterator(
    model_config=model_config,
    data_config=data_config,
    module_to_grid_map=module_to_grid_map,
)

# In training loop
for iteration in range(num_iterations):
    batch = get_batch(data_iterator)
```

**Batch size calculation:**
- `global_batch_size = base_batch_size * llm_dp_size`
- `micro_batch_size (per module) = global_batch_size / module_dp_size`
- Validation ensures global_batch_size is divisible by all module DP sizes

**Colocated mode:** Use `get_data_iterator_colocated()` - loads with the DP that has fewer replicas (larger batch), forward step handles slicing.

---

#### Training Pattern Example

```python
from megatron.bridge.models.mimo import get_vlm_mimo_model_colocated
import megatron.core.pipeline_parallel.schedules as schedule
from functools import partial

# Create model with colocated communication
mimo_model, module_to_grid_map, topology = \
    get_vlm_mimo_model_colocated(model_config, seq_len)

# Create optimizer and data iterator
optimizer = torch.optim.AdamW(mimo_model.parameters(), lr=1e-4)
data_iterator = get_data_iterator_colocated(model_config, data_config, module_to_grid_map)

# Training loop
for iteration in range(num_iterations):
    # Forward step handles data redistribution
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
    )

    # Gradient synchronization
    with multimodule_no_sync(module_to_grid_tuple, iteration, num_microbatches):
        optimizer.step()
    finalize_model_grads(module_to_grid_tuple, iteration, num_microbatches)
```

---

#### YAML Configuration and Experiment Runner

MIMO supports YAML-based configuration for running systematic ablation studies:

```yaml
# configs/ablations/llm_7b/colocated_exp.yaml
model:
  module_parallelisms:
    images:
      tensor_parallel: 1
      pipeline_parallel: 1
      data_parallel: 8
    language_module:
      tensor_parallel: 8
      pipeline_parallel: 1
      data_parallel: 1

runtime:
  pipeline_schedule: "colocated"  # or "homogeneous" or "1f1b"
  enable_performance_monitoring: true
```

**Running experiments:**
```bash
# Run single experiment
uv run python -m torch.distributed.run --nproc_per_node=8 \
    tests/unit_tests/models/heterogenous_parallel/experiment_runner.py \
    --config configs/ablations/llm_7b/exp.yaml

# Run all experiments in directory
uv run python -m torch.distributed.run --nproc_per_node=8 \
    tests/unit_tests/models/heterogenous_parallel/experiment_runner.py \
    --experiments-dir configs/ablations/llm_7b/
```

---

#### Important Notes

**Deployment mode selection:**
- Start with homogeneous for baselines
- Use colocated for most VLMs (small encoder + large LLM)
- Use heterogeneous only when PP is required

**Batch size constraints:**
- `global_batch_size` must be divisible by all module DP sizes
- Set `base_batch_size` based on LLM DP size

**Colocated requirements:**
- PP=1 required for all modules
- Same total_ranks across all modules
- Use `get_data_iterator_colocated()` for data loading
- Forward step must handle data slicing

**Common pitfalls:**
1. Forgetting to call `finalize()` on mimo_config
2. Using wrong data iterator for colocated mode
3. Missing data slicing in forward step
4. Incorrect batch sizes not divisible by DP sizes

**For full documentation, see:** `mimo_section_draft.md` (comprehensive guide with detailed examples for all deployment modes, performance monitoring, debugging tips, and file locations)

## Common Development Patterns

### Adding a New Model

1. Create model directory: `src/megatron/bridge/models/<model_name>/`
2. Implement `ModelProvider` subclass (inherits from base like `GPTModelProvider`)
3. Implement `MegatronModelBridge` with:
   - `provider_bridge()` method
   - `mapping_registry()` method
4. Register bridge with `@MegatronModelBridge.register_bridge(source=HFModel, target=MegatronModel)`
5. Add recipes in `src/megatron/bridge/recipes/<model_name>/`
6. Add tests in `tests/unit_tests/models/` and `tests/functional_tests/models/`
7. Update documentation

See `docs/adding-new-models.md` for detailed guide.

### Creating a Custom Training Recipe

```python
from megatron.bridge.training.config import ConfigContainer, TrainingConfig
from megatron.bridge.data.datasets.gpt import GPTDatasetConfig

def my_custom_recipe(**user_kwargs) -> ConfigContainer:
    # 1. Get model provider from HF or create manually
    bridge = AutoBridge.from_hf_pretrained("model-id")
    provider = bridge.to_megatron_provider()

    # 2. Configure parallelism
    provider.tensor_model_parallel_size = 8
    provider.finalize()

    # 3. Build complete config
    config = ConfigContainer(
        model=provider,
        dataset=GPTDatasetConfig(data_paths=["/data"]),
        train=TrainingConfig(
            train_iters=10000,
            global_batch_size=512,
            micro_batch_size=1,
        ),
        # ... other configs
    )

    # 4. Apply user overrides
    for key, value in user_kwargs.items():
        setattr(config, key, value)

    return config
```

### Debugging Distributed Training

- Single GPU: `python script.py` (no torchrun needed)
- Multi-GPU: `torchrun --nproc-per-node=N script.py`
- Check parallelism: Print `parallel_state.get_tensor_model_parallel_world_size()` etc.
- Inspect checkpoints: Use `src/megatron/bridge/training/utils/checkpoint_utils.py`
- Profiling: Enable via `config.profiling = ProfilingConfig(...)`

### Working with Checkpoints

```python
# Save checkpoint
from megatron.bridge.training.checkpoint import save_checkpoint
save_checkpoint(iteration, model, optimizer, scheduler, cfg)

# Load checkpoint
from megatron.bridge.training.checkpoint import load_checkpoint
load_checkpoint(model, optimizer, scheduler, cfg)

# Convert Megatron checkpoint to HF
bridge = AutoBridge.from_megatron_checkpoint("/path/to/ckpt")
bridge.save_hf_pretrained(model, "/path/to/hf_output")
```

## Project Structure

```
src/megatron/bridge/
├── models/                    # Model bridges and providers
│   ├── conversion/            # Bridge infrastructure (AutoBridge, registry)
│   ├── llama/                 # Llama model bridge and provider
│   ├── qwen/                  # Qwen model bridge and provider
│   ├── deepseek/              # DeepSeek model bridge and provider
│   └── .../                   # Other model implementations
├── recipes/                   # Pre-configured training recipes
│   ├── llama/                 # Llama recipes (1B, 3B, 8B, 70B, 405B)
│   ├── qwen/                  # Qwen recipes
│   └── .../                   # Other model recipes
├── training/                  # Training loop and utilities
│   ├── pretrain.py            # Main pretraining entry point
│   ├── finetune.py            # Supervised fine-tuning entry point
│   ├── train.py               # Core training loop
│   ├── gpt_step.py            # GPT forward step function
│   ├── config.py              # ConfigContainer and config classes
│   ├── checkpoint.py          # Checkpointing utilities
│   ├── tokenizers/            # Tokenizer implementations
│   └── utils/                 # Training utilities
├── data/                      # Data loading and datasets
│   ├── datasets/              # Dataset implementations (GPT, SFT, etc.)
│   └── iterators/             # Data iterators and samplers
├── peft/                      # PEFT implementations
│   ├── base.py                # PEFT base abstraction
│   ├── lora.py                # LoRA implementation
│   └── dora.py                # DoRA implementation
└── utils/                     # General utilities

examples/
├── conversion/                # Conversion examples and scripts
└── recipes/                   # Recipe usage examples

tests/
├── unit_tests/                # Unit tests (isolated, no HF checkpoints)
└── functional_tests/          # Integration tests (may use HF checkpoints)
```

## Testing Requirements

- **Unit tests**: Should be isolated, fast, and not depend on large HF checkpoints
  - Use mock data via `GPTDatasetConfig(mock=True)`
  - Located in `tests/unit_tests/`

- **Functional tests**: Integration tests that may download models/checkpoints
  - Requires GPUs for distributed training tests
  - Located in `tests/functional_tests/`

- All tests must pass before PR merge
- Use `pytest` markers: `@pytest.mark.unit`, `@pytest.mark.integration`

## Git Workflow

- All commits must be signed off: `git commit -s -m "message"`
- For CI to run automatically: use signed commits (GPG)
- Or trigger manually: comment `/ok to test <commit-SHA>` on PR
- Follow conventional commits for messages
- PRs require passing CI and review from `@nvidia-nemo/automation`

## Important Notes

- **Parallelism**: Always call `provider.finalize()` after setting parallelism before creating model
- **Distributed**: Training scripts must be launched with `torchrun` or `srun` for multi-GPU
- **Checkpoints**: Megatron checkpoints are parallelism-aware (contain TP/PP/CP metadata)
- **HF Integration**: Models loaded via `AutoBridge` automatically map HF config to Megatron
- **Memory**: Use `sequence_parallel=True` for long sequences to reduce activation memory
- **FP8**: Enable via `config.mixed_precision = MixedPrecisionConfig(precision="fp8")`
- **Recipes**: Start with existing recipes and customize rather than building from scratch
