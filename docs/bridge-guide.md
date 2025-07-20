# Megatron <-> HuggingFace Bridge User Guide

Megatron Bridge provides seamless bidirectional conversion between HuggingFace Transformers and Megatron Core model formats. This guide covers the main APIs for loading models, checking compatibility, and converting between formats.

## Automatically Loading a HuggingFace Model

The easiest way to load a HuggingFace model is using `AutoBridge.from_hf_pretrained()`, which automatically detects the model architecture and selects the appropriate bridge. You can then initialize the model with `AutoBridge.to_megatron_model()`.

### Basic Usage

```python
from megatron.bridge import AutoBridge

# Load any supported model automatically
bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")

# Initialize Megatron model
megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)
```

### Advanced Loading Options

```python
# Load with specific settings
bridge = AutoBridge.from_hf_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load from local path
bridge = AutoBridge.from_hf_pretrained("/path/to/local/model")

# Load with custom parameters
bridge = AutoBridge.from_hf_pretrained(
    "microsoft/phi-2",
    attn_implementation="flash_attention_2",
    load_in_8bit=True
)
```

### Using Model Providers

For more control over model configuration, use the provider pattern. The provider gives you access to configure any attribute from `TransformerConfig`:

```python
# Get a model provider (lazy loading)
provider = bridge.to_megatron_provider()

# Configure parallelism
provider.tensor_model_parallel_size = 8
provider.pipeline_model_parallel_size = 2

# Configure fusions
provider.bias_activation_fusion = True
provider.bias_dropout_fusion = True

# Create the model with all configurations applied
model = provider(wrap_with_ddp=False)
```

The provider pattern is especially useful when you need to:
- Override default model parameters
- Configure advanced features like MoE, activation recomputation, or mixed precision
- Set up distributed training parameters

## Check Supported Models

Before loading a model, you can check if it's supported by Megatron Bridge.

### List All Supported Models

```python
from megatron.bridge import CausalLMBridge

# Get list of all supported model architectures
supported_models = CausalLMBridge.list_supported_models()

print(f"Found {len(supported_models)} supported models:")
for i, model in enumerate(supported_models, 1):
    print(f"  {i:2d}. {model}")
```

### Check Specific Model Compatibility

```python
from megatron.bridge import AutoBridge

# Check if a specific model is supported
if AutoBridge.can_handle("microsoft/phi-2"):
    print("✅ Model is supported!")
    bridge = AutoBridge.from_hf_pretrained("microsoft/phi-2")
else:
    print("❌ Model requires a custom bridge implementation")

# Check with custom settings
if AutoBridge.can_handle("custom/model", trust_remote_code=True):
    bridge = AutoBridge.from_hf_pretrained("custom/model", trust_remote_code=True)
```

### Get Supported Bridge Types

```python
from megatron.bridge import AutoBridge

# See which bridge types are available
bridges = AutoBridge.get_supported_bridges()
print(f"Available bridges: {bridges}")
```

## Converting back to HuggingFace

After training or modifying a Megatron model, you can convert it back to HuggingFace format for deployment or sharing.

### Save Complete Model

```python
# Save the complete model (config, tokenizer, weights)
bridge.save_hf_pretrained(megatron_model, "./my-fine-tuned-llama")

# The saved model can be loaded with HuggingFace
from transformers import AutoModelForCausalLM
hf_model = AutoModelForCausalLM.from_pretrained("./my-fine-tuned-llama")
```

### Save Only Weights

```python
# Save just the model weights (faster, smaller)
bridge.save_hf_weights(megatron_model, "./model_weights")

# Save without progress bar (useful in scripts)
bridge.save_hf_weights(megatron_model, "./weights", show_progress=False)
```

### Stream Weights for Large Models

```python
# Stream weights during conversion (memory efficient)
for name, weight in bridge.export_hf_weights(megatron_model):
    print(f"Exporting {name}: {weight.shape}")

# Stream with specific settings
for name, weight in bridge.export_hf_weights(
    megatron_model,
    order="safetensors",  # Group by safetensors file
    cpu=True,             # Move to CPU before yielding
    mode="consolidate"    # Gather to rank 0
):
    print(f"Exported {name}: {weight.shape}")
```

### Round-Trip Conversion Example

```{literalinclude} ../examples/models/2_way_hf_binding.py
:lines: 31-
```

### Weight Distribution Modes

```python
# Different export modes for distributed models
for name, weight in bridge.export_hf_weights(model, mode="consolidate"):
    # Default: Gather to rank 0 only
    pass

for name, weight in bridge.export_hf_weights(model, mode="replicate"):
    # All ranks get full weights
    pass

for name, weight in bridge.export_hf_weights(model, mode="distribute"):
    # Each rank keeps its shard (experimental)
    pass
```

## Common Patterns and Best Practices

### 1. Always Use High-Level APIs

```python
# ✅ Preferred: Use AutoBridge for automatic detection
bridge = AutoBridge.from_hf_pretrained("any-supported-model")

# ✅ Or use CausalLMBridge for causal language models
bridge = CausalLMBridge.from_hf_pretrained("gpt2")

# ❌ Avoid: Direct bridge usage unless you know the specific type
```

### 2. Configure Before Creating Models

```python
# ✅ Correct: Configure provider before creating model
provider = bridge.to_megatron_provider()
provider.tensor_model_parallel_size = 8
model = provider(wrap_with_ddp=False)

# ❌ Avoid: Creating model before configuring parallelism
model = bridge.to_megatron_model()  # Uses default settings
```

### 3. Handle Large Models Efficiently

```python
# ✅ Use streaming for large models
for name, weight in bridge.export_hf_weights(model, cpu=True):
    process_weight(name, weight)

# ✅ Use config-only loading for architecture exploration
config = AutoConfig.from_pretrained("meta-llama/Llama-3-8B")
bridge = CausalLMBridge.from_hf_config(config)
transformer_config = bridge.transformer_config
print(f"Hidden size: {transformer_config.hidden_size}")
```

### 4. Error Handling

```python
from megatron.bridge import AutoBridge

try:
    bridge = AutoBridge.from_hf_pretrained("unsupported/model")
except ValueError as e:
    print(f"Model not supported: {e}")
    # Check what models are available
    supported = CausalLMBridge.list_supported_models()
    print(f"Supported models: {supported}")
```

## Troubleshooting

### Common Issues

1. **Model Not Supported**: Use `CausalLMBridge.list_supported_models()` to see available options
2. **Memory Errors**: Use streaming APIs or increase parallelism
3. **Shape Mismatches**: Check parallelism configuration matches your hardware
4. **Missing Weights**: Ensure the model architecture is properly registered

### Debug Tips

```python
# Enable verbose logging
import logging
logging.getLogger("megatron.bridge.models").setLevel(logging.DEBUG)

# Inspect bridge configuration
bridge = CausalLMBridge.from_hf_pretrained("gpt2")
print(bridge.transformer_config)

# Check weight mappings
mapping_registry = bridge._model_bridge.mapping_registry()
print(mapping_registry.get_all_mappings())
```

For more examples and advanced usage patterns, see the `examples/models/` directory in the repository.
