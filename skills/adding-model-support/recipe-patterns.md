# Recipe Patterns

Recipes provide pre-configured `ConfigContainer` objects for training each model variant.

Reference implementations:
- **VLM:** `src/megatron/bridge/recipes/qwen_vl/qwen35_vl.py`
- **LLM:** `src/megatron/bridge/recipes/gpt_oss/gpt_oss.py`

## File Structure

```text
src/megatron/bridge/recipes/<family>/
├── __init__.py          # Import and expose recipe functions
└── <model>.py           # Recipe functions for all sizes
```

## Recipe Function Pattern

Recipes are declarative presets, similar in spirit to checked-in YAML configs.
Each recipe function must construct one concrete config for one model, size,
training mode, hardware class, dtype, and named variant.

**Recipe contract:**
- Do not add `*args`, `**kwargs`, or recipe constructor arguments such as
  `hf_path`, `seq_length`, `tokenizer_path`, dataset path, parallelism, freeze
  flags, precision, or packing controls.
- The only allowed recipe function argument is `peft_scheme` on PEFT recipes.
- Do not branch inside a recipe based on optional arguments or environment
  conditions. If a setting differs, create a separate explicitly named recipe.
- Encode stable variants in the function name, for example sequence length,
  precision, hardware, GPU count, dataset/packing variant, or freeze policy.
- Use canonical hardware-specific recipe names in the form
  `<model>_<task>_<num>gpu_<gpu>_<dtype>[_<variant>]_config`, where `<model>`
  includes size/capacity tokens when needed and `<variant>` captures stable
  suffixes such as dataset, packing, long-context, or precision-scaling modes.
- Use runtime config overrides in `scripts/training/run_recipe.py` for user paths
  and local experiment changes; do not make recipes accept those values.
- Keep private helpers invariant. A helper can apply common fixed defaults, but
  it should not accept variant-defining kwargs that hide differences between
  recipes.

Each model size gets dedicated flattened functions for SFT, PEFT, and optionally
pretrain:

```python
def <model>_<size>_sft_config() -> ConfigContainer:
    """SFT config for <Model> <Size>."""
    cfg = _sft_common()  # or _sft_common_vlm() for VLMs

    # Model
    cfg.model = AutoBridge.from_hf_pretrained("<org>/<default-model>").to_megatron_provider(load_weights=False)

    # Parallelism
    cfg.model.tensor_model_parallel_size = 4
    cfg.model.pipeline_model_parallel_size = 1
    cfg.model.sequence_parallel = True

    # Training
    cfg.training.max_steps = 100
    cfg.training.global_batch_size = 128
    cfg.training.micro_batch_size = 1

    # Optimizer
    cfg.optimizer.lr = 5e-6
    cfg.optimizer.weight_decay = 0.01

    # VLM-specific (if applicable)
    cfg.model.freeze_language_model = False
    cfg.model.freeze_vision_model = False
    cfg.model.freeze_vision_projection = False

    return cfg


def <model>_<size>_peft_config(peft_scheme: str | PEFT = "lora") -> ConfigContainer:
    """PEFT config for <Model> <Size>."""
    cfg = _peft_common()  # or _peft_common_vlm() for VLMs

    cfg.model = AutoBridge.from_hf_pretrained("<org>/<default-model>").to_megatron_provider(load_weights=False)

    # PEFT typically uses smaller parallelism
    cfg.model.tensor_model_parallel_size = 1
    cfg.model.pipeline_model_parallel_size = 1

    # PEFT uses higher LR
    cfg.optimizer.lr = 2e-4

    # PEFT config
    peft_cfg = default_peft_config(peft_scheme)
    cfg.peft = peft_cfg

    return cfg
```

## Common Base Functions

| Function | Use Case |
|----------|----------|
| `_pretrain_common()` | LLM pretraining |
| `_sft_common()` | LLM supervised fine-tuning |
| `_peft_common()` | LLM parameter-efficient fine-tuning |
| `_sft_common_vlm()` | VLM SFT (adds vision dataset, null tokenizer) |
| `_peft_common_vlm()` | VLM PEFT |

VLM variants additionally set:
- `cfg.dataset` to `HFConversationDatasetProvider` (e.g., CORD-v2)
- `cfg.dataset.hf_processor_path` for the vision processor
- `NullTokenizer` (tokenization handled by processor)
- DDP without overlap (for vision model compatibility)

Do not implement VLM size or freeze-mode variants by passing values into a
shared `_apply_common(...)` helper. Write each recipe body explicitly enough
that the HF model, parallelism, freeze flags, sequence length, packing, and
dataset defaults are visible in that function.

## Parallelism Guidelines

**Constraint:** assuming `DP=1`, minimum GPUs =
`max(TP * PP * CP, EP * PP * CP)`, with 8 GPUs per node.

| Model Size | TP | PP | EP | CP | Notes |
|-----------|----|----|----|----|-------|
| &lt; 3B | 1 | 1 | 1 | 1 | Single GPU |
| 3-8B | 2 | 1 | 1 | 1 | |
| 8-13B | 4 | 1 | 1 | 1 | |
| 13-70B | 4 | 4 | 1 | 1 | |
| MoE (any) | 1-2 | 1-4 | 8-32 | 1 | EP dominates |

**Rules:**
- TP must be &lt;= `num_key_value_heads`
- When EP &gt; 1 and TP &gt; 1, `sequence_parallel` must be True
- PEFT typically uses smaller parallelism (TP=1, PP=1)

## Export / Registration

### Family `__init__.py`

```python
from megatron.bridge.recipes.<family>.<model> import (
    <model>_<size1>_sft_config,
    <model>_<size1>_peft_config,
    <model>_<size2>_sft_config,
    <model>_<size2>_peft_config,
)

__all__ = [
    "<model>_<size1>_sft_config",
    "<model>_<size1>_peft_config",
    # ...
]
```

### Top-level `recipes/__init__.py`

Add a wildcard import:

```python
from megatron.bridge.recipes.<family> import *
```

### `train_any_basic.py`

Add entry to `config_map` dict, docstring model list, and `--model` argparse choices.

## Recipe Test Patterns

### Unit test (no GPU)

Monkeypatch `AutoBridge` to return a mock provider. Verify `ConfigContainer` structure:

```python
def test_sft_config(monkeypatch):
    monkeypatch.setattr("megatron.bridge.AutoBridge.from_hf_pretrained", mock_bridge)
    cfg = model_size_sft_config()
    assert cfg.model.tensor_model_parallel_size == 4
    assert cfg.training.global_batch_size == 128
```

### Functional test (GPU)

Use `run_pretrain_vl_recipe_test()` from `tests/functional_tests/recipes/utils.py`:

```python
RECIPES = [
    (model_size_sft_config, "model_size_sft", {}, {}),
]

PEFT_RECIPES = [
    (model_size_peft_config, "model_size_peft", {}, {}),
]
```

### Five training scenarios to cover (VLMs)

1. SFT nothing frozen
2. SFT language frozen (train vision + projection)
3. SFT vision + language frozen (train projection only)
4. PEFT with vision frozen
5. PEFT with nothing frozen
