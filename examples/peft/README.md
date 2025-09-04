# PEFT Examples

This directory contains examples demonstrating how to use the Megatron Bridge PEFT (Parameter-Efficient Fine-Tuning) integration.

## Prerequisites

Install the required dependencies:
```bash
pip install megatron-bridge[peft]
```

## Examples Overview

### 1. Fused LoRA
**File:** `lora.py`

Demonstrates creating and applying fused LoRA adapters to efficient fused projection layers.

```bash
# Basic fused LoRA
python examples/peft/lora.py --rank 16 --alpha 32

# With merge demonstration
python examples/peft/lora.py --rank 32 --alpha 64 --merge
```

**Features:**
- Fused target modules: `linear_qkv`, `linear_proj`, `linear_fc1`, `linear_fc2`
- Most efficient LoRA variant
- Working merge functionality
- Parameter efficiency statistics

### 2. Canonical LoRA
**File:** `canonical_lora.py`

Demonstrates creating and applying canonical LoRA adapters to individual projection layers.

```bash
# Basic canonical LoRA
python examples/peft/canonical_lora.py --rank 16 --alpha 32

# With merge demonstration
python examples/peft/canonical_lora.py --rank 32 --merge
```

**Features:**
- Individual target modules: `linear_q`, `linear_k`, `linear_v`, `linear_fc1_gate`, etc.
- HuggingFace PEFT compatible layout
- Merge and unwrap functionality
- Follows HF PEFT conventions

### 3. DoRA (Experimental)
**File:** `dora.py`

Demonstrates creating and applying DoRA adapters with magnitude vector decomposition.

```bash
# Basic DoRA
python examples/peft/dora.py --rank 16 --alpha 64

# With unwrap demonstration
python examples/peft/dora.py --rank 32 --alpha 128 --merge
```

**Features:**
- Weight decomposition into magnitude and direction
- Same target modules as fused LoRA but with magnitude vectors
- Unwrap functionality (merge weights not yet implemented)
- Higher alpha values typical for DoRA

### 4. Apply Existing Adapters
**File:** `apply_lora_adapters.py`

Demonstrates loading and applying pretrained adapters from HuggingFace Hub.

```bash
# Auto-detect base model from adapter config
python examples/peft/apply_lora_adapters.py

# Specify custom adapter
python examples/peft/apply_lora_adapters.py --adapter-id "username/my-lora"
```

**Features:**
- Auto-detection of base model from adapter config
- Merge functionality with validation
- Weight mapping verification
- Save merged model as standard HuggingFace model

### 5. HuggingFace Round-Trip Conversion
**File:** `adapter_round_trip_hf.py`

Validates bidirectional conversion between HuggingFace and Megatron adapter formats.

```bash
# Test with default adapter
python examples/peft/adapter_round_trip_hf.py

# Test with custom adapter
python examples/peft/adapter_round_trip_hf.py --adapter-id "username/my-adapter"
```

**Features:**
- Auto-detection of base model
- Weight mapping verification table
- Conversion validation
- Git-safe output directory

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

## PEFT Variants

- **Fused LoRA**: Most efficient, applies to fused layers (`linear_qkv`, `linear_fc1`)
- **Canonical LoRA**: HuggingFace compatible, individual projections (`linear_q`, `linear_k`, etc.)
- **DoRA**: Experimental, adds magnitude vectors to LoRA for enhanced adaptation

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

### Merge Functionality
- **Fused LoRA**: Full merge and unwrap support
- **Canonical LoRA**: Full merge and unwrap support
- **DoRA**: Unwrap only (merge weights not yet implemented)
- Use `--merge` flag in examples to test merge functionality

## See Also

- **Main PEFT API Documentation**: `src/megatron/bridge/peft/`
- **Bridge Implementations**: `src/megatron/bridge/peft/lora/`
- **Model Examples**: `examples/models/` for base model usage patterns
- **Training Recipes**: `examples/recipes/` for complete training workflows