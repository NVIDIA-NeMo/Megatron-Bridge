# Get Started with 🤗 Hugging Face Conversion

Megatron Bridge provides seamless bidirectional conversion between 🤗 Hugging Face Transformers and Megatron model definitions. This guide covers the main APIs for loading models, checking compatibility, and converting between formats.

## Design and Goals

- Single high-level entry point: `AutoBridge` detects HF model architectures and dispatches to the correct bridge.
- Bidirectional conversion: Import HF → Megatron for training; export Megatron → HF for deployment.
- Parallelism-aware: Handles TP/PP/VPP/CP/EP/ETP distributions during conversion.
- Streaming and memory efficiency: per-parameter streaming using safetensors.
- Provider pattern: Configure Megatron-Core `TransformerConfig`-compatible attributes before instantiation via `to_megatron_provider()`.
- Convenience workflows: `import_ckpt` and `export_ckpt` provide one-call HF↔Megatron checkpoint flows.

See the repository `README.md` for installation, supported models, and project highlights.

## Loading a 🤗 Hugging Face Model into Megatron

The easiest way to load a 🤗 Hugging Face model is using `AutoBridge.from_hf_pretrained()`, which automatically detects the model architecture and selects the appropriate bridge for conversion. You can then use `AutoBridge.to_megatron_model()` to initialize the Megatron model from the 🤗 Hugging Face configuration and load 🤗 Hugging Face weights at the same time.

### Accessing Gated 🤗 Hugging Face Models

Some models in Megatron Bridge require access to gated repositories on Hugging Face. These are models that require explicit permission from the model authors before you can download or use them.

If you encounter an error like this when trying to use a model:

```
OSError: You are trying to access a gated repo.
Make sure to have access to it at <URL>
```

Follow these steps to resolve the issue:

1. **Request access**: Visit the URL provided in the error message and request access to the gated model
2. **Generate a token**: Create a Hugging Face access token by following [this tutorial](https://huggingface.co/docs/hub/en/security-tokens#how-to-manage-user-access-tokens)
3. **Set the environment variable**: Export your token in your environment:

```bash
export HF_TOKEN=<your_access_token>
```

### Basic Usage

```python
from megatron.bridge import AutoBridge

# Load a supported model automatically
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")

# Create a provider, configure before instantiation, then build the model
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
megatron_model = provider.provide_distributed_model(wrap_with_ddp=False)
```

### Advanced Loading Options
You can also load models with specific settings such as precision, device placement, or by enabling trust in remote code:

```python
import torch
from megatron.bridge import AutoBridge

# Load with specific settings
bridge = AutoBridge.from_hf_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load from local path
bridge = AutoBridge.from_hf_pretrained("/path/to/local/hf_model")
```

### Using Model Providers

For more control over model configuration, use the provider pattern. The provider lets you configure any `TransformerConfig` attribute:

```python
from megatron.bridge import AutoBridge

# Load a supported model automatically
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")

# Get a model provider (lazy loading)
provider = bridge.to_megatron_provider()

# Configure parallelism (multi-GPU requires torchrun or srun)
provider.tensor_model_parallel_size = 8
provider.pipeline_model_parallel_size = 2

# Configure fusions
provider.bias_activation_fusion = True
provider.bias_dropout_fusion = True

# Finalize the provider to run validation checks and complete initialization
provider.finalize()

# Create the model with all configurations applied
model = provider.provide_distributed_model(wrap_with_ddp=False)
```

The provider pattern is especially useful when you need to:
- Override default model parameters
- Configure advanced features like MoE, activation recomputation, or mixed precision
- Set up distributed training parameters

## Check Supported Models

Before loading a model, you can check if it's supported by Megatron Bridge.

You can list all supported 🤗 Hugging Face model architectures as follows:

```python
from megatron.bridge import AutoBridge

# Get a list of all supported model architectures
supported_models = AutoBridge.list_supported_models()

print(f"Found {len(supported_models)} supported models:")
for i, model in enumerate(supported_models, 1):
    print(f"  {i:2d}. {model}")
```

Alternatively, check if a specific model is supported:

```python
from megatron.bridge import AutoBridge

if AutoBridge.can_handle("meta-llama/Llama-3.2-1B"):
    print("✅ Model is supported!")
    bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
else:
    print("❌ Model requires a custom bridge implementation")
```

## Converting Back to 🤗 Hugging Face

After training or modifying a Megatron model, you can convert it back to 🤗Hugging Face format for deployment or sharing. The bridge provides several methods for this conversion depending on your needs.

To save the complete model including configuration, tokenizer, and weights:

```python
# Save the complete model (config, tokenizer, weights)
bridge.save_hf_pretrained(megatron_model, "./my-fine-tuned-llama")

# The saved model can be loaded with 🤗 Hugging Face
from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained("./my-fine-tuned-llama")
```

You can save the model weights (safetensors):

```python
# Save just the model weights (faster, smaller)
bridge.save_hf_weights(megatron_model, "./model_weights")

# Save without progress bar (useful in scripts)
bridge.save_hf_weights(megatron_model, "./weights", show_progress=False)
```

You can also stream weights without saving to disk during conversion for on-the-fly use in RL frameworks, for example:

```python
# Stream weights during conversion (memory efficient)
for name, weight in bridge.export_hf_weights(megatron_model):
    print(f"Exporting {name}: {weight.shape}")

for name, weight in bridge.export_hf_weights(megatron_model, cpu=True):
    print(f"Exported {name}: {tuple(weight.shape)}")
```

## Common Patterns and Best Practices
When working with Megatron Bridge, there are several patterns that will help you use the API effectively and avoid common pitfalls.

### 1. Always Use High-Level APIs
Always prefer high-level APIs like `AutoBridge` for automatic model detection. Avoid direct bridge usage unless you know the specific type required:

```python
# ✅ Preferred: Use AutoBridge for automatic detection
bridge = AutoBridge.from_hf_pretrained("any-supported-model")

# ❌ Avoid: Direct bridge usage unless you know the specific type
```

### 2. Configure Before Creating Models
When using the provider pattern, always configure parallelism and other settings before creating the model. Creating the model first uses default settings that may not be optimal:

```python
# ✅ Correct: Configure provider before creating model
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 8
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)

# ❌ Avoid: Creating model before configuring parallelism
model = bridge.to_megatron_model()  # Uses default settings
```

### 3. Leverage the Parameter Streaming API
You can stream converted weights from Megatron to HF without saving to disk. You can also use config-only loading for architecture exploration without loading weights:

```python
# ✅ Use streaming for large models
for name, weight in bridge.export_hf_weights(model, cpu=True):
    process_weight(name, weight)

# ✅ Use config-only loading for architecture exploration
config = AutoConfig.from_pretrained("meta-llama/Llama-3-8B")
bridge = AutoBridge.from_hf_config(config)
transformer_config = bridge.transformer_config
print(f"Hidden size: {transformer_config.hidden_size}")
```

For more examples and advanced usage patterns, see the `examples/conversion/` directory in the repository.

## Convenience Workflows (Commands)

These examples can be run directly as shell commands.

### HF → Megatron checkpoint import (one call)

```bash
huggingface-cli login --token <your token>
python -c "from megatron.bridge import AutoBridge; AutoBridge.import_ckpt('meta-llama/Llama-3.2-1B','./megatron_checkpoints/llama32_1b')"
```

### Megatron → HF export (one call)

```bash
python -c "from megatron.bridge import AutoBridge; from transformers import AutoConfig; cfg=AutoConfig.from_pretrained('meta-llama/Llama-3.2-1B'); b=AutoBridge.from_hf_config(cfg); b.export_ckpt('./megatron_checkpoints/llama32_1b','./hf_exports/llama32_1b')"
```

### Create Megatron models and run locally

```bash
python - << 'PY'
from megatron.bridge import AutoBridge

bridge = AutoBridge.from_hf_pretrained('meta-llama/Llama-3.2-1B')
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 1
provider.pipeline_model_parallel_size = 1
provider.finalize()
model = provider.provide_distributed_model(wrap_with_ddp=False)

# Export to HF folder
bridge.save_hf_pretrained(model, './hf_exports/llama32_1b')
PY
```

### Launch with multiple GPUs (example)

```bash
torchrun --nproc-per-node=2 -m examples.conversion.generate_from_hf
```

## AutoBridge API Reference

Latest public APIs and signatures (see {doc}`apidocs/bridge/bridge.models.conversion.auto_bridge`):

```python
from megatron.bridge import AutoBridge

# Creation and capability
AutoBridge.from_hf_pretrained(path: str | Path, **kwargs) -> AutoBridge
AutoBridge.from_hf_config(config: PretrainedConfig) -> AutoBridge
AutoBridge.can_handle(path: str | Path, trust_remote_code: bool = False) -> bool
AutoBridge.list_supported_models() -> list[str]
AutoBridge.supports(config: Any) -> bool

# Provider/model construction
AutoBridge.to_megatron_provider(load_weights: bool = True, hf_path: str | Path | None = None) -> GPTModelProvider
AutoBridge.to_megatron_model(load_weights: bool = True, hf_path: str | Path | None = None, **kwargs) -> list[MegatronModule]

# HF → Megatron weights
AutoBridge.load_hf_weights(model: list[MegatronModule], hf_path: str | Path | None = None) -> None

# Megatron → HF conversion
AutoBridge.export_hf_weights(model: list[MegatronModule], cpu: bool = False, show_progress: bool = True, conversion_tasks: Optional[list[WeightConversionTask]] = None) -> Iterable[HFWeightTuple]
AutoBridge.save_hf_pretrained(model: list[MegatronModule], path: str | Path, show_progress: bool = True) -> None
AutoBridge.save_hf_weights(model: list[MegatronModule], path: str | Path, show_progress: bool = True) -> None

# Megatron native checkpoints
AutoBridge.save_megatron_model(model: list[MegatronModule], path: str | Path) -> None
AutoBridge.load_megatron_model(path: str | Path, **kwargs) -> list[MegatronModule]

# One-call workflows
AutoBridge.import_ckpt(hf_model_id: str | Path, megatron_path: str | Path, **kwargs) -> None  # HF → Megatron ckpt
AutoBridge.export_ckpt(megatron_path: str | Path, hf_path: str | Path, show_progress: bool = True) -> None  # Megatron → HF

# Config extraction
AutoBridge.transformer_config -> TransformerConfig
AutoBridge.mla_transformer_config -> MLATransformerConfig

# Introspection / planning
AutoBridge.get_conversion_tasks(megatron_model: MegatronModule | list[MegatronModule], hf_path: str | Path | None = None) -> list[WeightConversionTask]
```
