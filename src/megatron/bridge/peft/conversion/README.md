# PEFT Bridge Conversion Infrastructure

This directory contains the bridge infrastructure for converting between HuggingFace PEFT and Megatron PEFT formats.

## Files Overview

**Core Bridge Classes:**
- `peft_bridge.py` - Abstract base class with universal mapping algorithm
- `auto_peft_bridge.py` - User-facing API with auto-detection capabilities

**Parameter Mapping:**
- `param_mapping.py` - Adapter mapping classes + factory methods

**Utilities:**
- `pretrained_adapters.py` - HF adapter loader
- `transform_utils.py` - Conversion utilities

## Key Design Points

### Base Bridge Pattern
```python
class MegatronPEFTBridge(ABC):
    @abstractmethod
    def create_peft_mapping(self, base_mapping, adapter_param):
        # Transforms base model ParamMapping → adapter ParamMapping
        # Example: QKVMapping → AdapterQKVMapping
        
    def mapping_registry(self, adapters):
        # Universal algorithm:
        # 1. Get base model mappings (QKVMapping, AutoMapping, etc.)
        # 2. For each base mapping that PEFT affects:
        #    - Get adapter parameters: PEFT.get_megatron_adapter_params()
        #    - Transform: base_mapping → adapter_mapping via create_peft_mapping()
        # 3. Return registry of adapter mappings for weight conversion
```

### Mapping Transformation Flow
```
Base Model Bridge     →    PEFT Bridge
─────────────────          ────────────
QKVMapping           →     AdapterQKVMapping
GatedMLPMapping      →     AdapterGatedMLPMapping
AutoMapping          →     AdapterAutoMapping

# Each adapter mapping handles:
# - HF adapter weights → Megatron adapter weights (hf_to_megatron)
# - Megatron adapter weights → HF adapter weights (megatron_to_hf)
```

### Factory Pattern
```python
# Co-located in param_mapping.py
class AdapterQKVMapping(MegatronParamMapping):
    @classmethod
    def from_base_mapping(cls, base_mapping, adapter_param, hf_suffix):
        # Type-safe creation from base mappings
```

### Auto-Detection API
```python
# Option 1: Manual
peft_bridge = AutoPEFTBridge.from_hf_pretrained("adapter", base_bridge)

# Option 2: Auto-detect from config.base_model_name_or_path  
peft_bridge = AutoPEFTBridge.from_hf_pretrained("adapter")

# Clean usage
peft_model = peft_bridge.to_megatron_model()
```

## Extension Pattern
1. Create PEFT class with `get_adapter_parameter_patterns()`
2. Create bridge with `create_peft_mapping()` method
3. Register with `@MegatronPEFTBridge.register_bridge()`