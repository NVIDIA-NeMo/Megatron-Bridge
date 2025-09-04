# PEFT Examples

This directory contains examples demonstrating how to use the Megatron Bridge PEFT (Parameter-Efficient Fine-Tuning) integration.

## Prerequisites

Install the required dependencies:
```bash
pip install megatron-bridge[peft]
```

## Examples Overview

### 1. List Supported Adapters
**File:** `list_supported_adapters.py`

Lists all PEFT adapter types currently supported by the bridge system.

```bash
python examples/peft/list_supported_adapters.py
```

**Output:**
- Supported adapter types (LoRA, DoRA, etc.)
- Usage instructions and documentation references

### 2. Apply Existing LoRA Adapters
**File:** `apply_lora_adapters.py`

Demonstrates how to load pretrained LoRA adapters from HuggingFace and apply them to a Megatron model.

```bash
# Basic usage
python examples/peft/apply_lora_adapters.py \
    --base-model-id "meta-llama/Llama-3.1-8B" \
    --adapter-model-id "username/llama-lora-math"

# With custom output directory
python examples/peft/apply_lora_adapters.py \
    --base-model-id "microsoft/phi-2" \
    --adapter-model-id "username/phi-2-code-lora" \
    --output-dir "./converted_adapters"
```

**Features demonstrated:**
- Loading adapters from HuggingFace Hub or local paths
- Adapter configuration inspection
- Parameter statistics and trainable parameter counting
- Adapter enable/disable functionality
- Saving adapters back to HuggingFace format

### 3. Create New LoRA Configuration
**File:** `create_new_lora.py`

Shows how to create a new LoRA configuration from scratch and apply it to a base model.

```bash
# Create fused LoRA (default)
python examples/peft/create_new_lora.py \
    --base-model-id "meta-llama/Llama-3.1-8B" \
    --lora-rank 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05

# Create canonical LoRA
python examples/peft/create_new_lora.py \
    --base-model-id "microsoft/phi-2" \
    --lora-rank 8 \
    --lora-alpha 16 \
    --use-canonical \
    --target-modules "attention"
```

**Features demonstrated:**
- Creating LoRA and CanonicalLoRA configurations
- Choosing between fused and canonical layouts
- Configuring target modules (all, attention, mlp, custom)
- Parameter efficiency comparison
- Adapter state extraction

### 4. Round-Trip Conversion Test
**File:** `adapter_round_trip.py`

Validates bidirectional conversion accuracy between HuggingFace and Megatron adapter formats.

```bash
# Test conversion accuracy
python examples/peft/adapter_round_trip.py \
    --base-model-id "meta-llama/Llama-3.1-8B" \
    --adapter-model-id "username/llama-lora-math" \
    --tolerance 1e-5

# Save verified adapters
python examples/peft/adapter_round_trip.py \
    --adapter-model-id "username/my-lora" \
    --output-dir "./verified_adapters"
```

**Features demonstrated:**
- Weight preservation validation
- Numerical tolerance testing
- Format conversion verification
- Automatic cleanup of temporary files

### 5. Merge Adapters into Full Model
**File:** `merge_and_save_full_model.py`

Demonstrates merging trained adapters into the base model weights to create a complete fine-tuned model.

```bash
# Merge and save complete model
python examples/peft/merge_and_save_full_model.py \
    --base-model-id "meta-llama/Llama-3.1-8B" \
    --adapter-model-id "username/trained-math-lora" \
    --output-dir "./merged_models"

# Quick merge without verification
python examples/peft/merge_and_save_full_model.py \
    --adapter-model-id "username/my-lora" \
    --no-verify
```

**Features demonstrated:**
- Adapter merging into base weights
- Merge verification and validation
- Saving merged models as standard HuggingFace models
- Error handling for unsupported merge operations

## Common Usage Patterns

### Loading and Applying Adapters
```python
from megatron.bridge import AutoBridge
from megatron.bridge.peft import AutoPEFTBridge

# Option 1: Auto-detection (when adapter has base_model_name_or_path)
peft_bridge = AutoPEFTBridge.from_hf_pretrained("codelion/Llama-3.2-1B-Instruct-tool-calling-lora")
peft_model = peft_bridge.to_megatron_model()  # Auto-detects base model

# Option 2: Manual base model specification
base_bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
peft_bridge = AutoPEFTBridge.from_hf_pretrained("username/custom-adapter", base_bridge)
peft_model = peft_bridge.to_megatron_model()
```

### Creating Custom LoRA
```python
from megatron.bridge import AutoBridge
from megatron.bridge.peft import get_peft_model, LoRA

# Load base model
base_bridge = AutoBridge.from_hf_pretrained("meta-llama/Llama-3.1-8B")
provider = base_bridge.to_megatron_provider()

# Create custom LoRA
lora = LoRA(dim=32, alpha=32, dropout=0.1, target_modules=["linear_qkv", "linear_fc1"])
peft_model = get_peft_model(provider, lora)
```

### Adapter Management
```python
# Check parameter statistics
peft_model.print_trainable_parameters()

# Control adapter training
peft_model.disable_adapters()  # Freeze adapters
peft_model.enable_adapters()   # Unfreeze adapters

# Extract adapter weights
adapter_weights = peft_model.adapter_state_dict()

# Merge adapters into base weights
merged_model = peft_model.merge_and_unload()
```

## Distributed Training

All examples support distributed training out of the box. For multi-GPU usage:

```bash
# Example with torchrun (2 GPUs)
torchrun --nproc_per_node=2 examples/peft/apply_lora_adapters.py \
    --base-model-id "meta-llama/Llama-3.1-8B" \
    --adapter-model-id "username/llama-lora"
```

## Supported Adapter Types

- **LoRA (Fused)**: Applies adapters to fused linear layers (`linear_qkv`, `linear_fc1`)
- **LoRA (Canonical)**: Applies adapters to individual projections (`q_proj`, `k_proj`, etc.)
- **DoRA (Experimental)**: Includes magnitude vectors for enhanced adaptation

## Troubleshooting

### Import Errors
If you see import errors related to PEFT:
```bash
pip install megatron-bridge[peft]
```

### Adapter Loading Issues
- Ensure adapter directory contains `adapter_config.json` and `adapter_model.safetensors`
- Check that adapter type is supported with `list_supported_adapters.py`
- Verify target modules are compatible with the base model architecture

### Merge Failures  
- DoRA adapters don't support merging yet (experimental)
- Some adapter configurations may not support merging
- Check error messages for specific guidance on supported operations

## See Also

- **Main PEFT API Documentation**: `src/megatron/bridge/peft/`
- **Bridge Implementations**: `src/megatron/bridge/peft/lora/`, `src/megatron/bridge/peft/dora/`
- **Model Examples**: `examples/models/` for base model usage patterns
- **Training Recipes**: `examples/recipes/` for complete training workflows