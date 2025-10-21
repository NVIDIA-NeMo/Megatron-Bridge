# LoRA Family Implementation

This directory contains all LoRA-based PEFT implementations and their unified bridge.

## Files Overview

**Core Implementations:**
- `lora.py` - Fused LoRA (linear_qkv, linear_fc1) - most efficient
- `dora.py` - DoRA extends LoRA with magnitude vectors
- `canonical_lora.py` - Individual projections (linear_q, linear_k, linear_v)

**Layers:**
- `lora_layers.py` - Shared layer implementations (LinearAdapter, LoRALinear)
- `dora_layers.py` - DoRA-specific layers (DoRALinear, ParallelLinearDoRAAdapter)

**Bridge:**
- `lora_bridge.py` - **Unified bridge for all variants**, auto-detects type from config

## Key Design Points

### Unified Bridge Pattern
```python
@MegatronPEFTBridge.register_bridge(source=LoraConfig, target=LoRA)
class LoRABridge(MegatronPEFTBridge):
    def peft_bridge(self, adapters) -> Union[LoRA, DoRA, CanonicalLoRA]:
        # Auto-detects variant from config flags and target modules
        
    def create_peft_mapping(self, base_mapping, adapter_param):
        # Uses different mapping strategies for canonical vs fused
```

### Auto-Detection Logic
- **DoRA**: `config.get("use_dora", False)`
- **Canonical**: `any(target in canonical_indicators for target in hf_target_modules)`
- **Fused LoRA**: Default case

### Parameter Patterns
- **LoRA/DoRA**: `.adapter.linear_in.weight`, `.adapter.linear_out.weight`, `.adapter.weight_magnitude`
- **Canonical**: Same patterns but applied to individual projections

## Usage
All variants use the same bridge - no separate registrations needed.