# Megatron-Bridge Framework

The bridge framework provides seamless bidirectional conversion between HuggingFace Transformers and Megatron-Core model formats, handling the complexities of distributed model parallelism transparently.

## Quick Start

### Loading a HuggingFace Model into Megatron

```python
from megatron.bridge import CausalLMBridge

# Load Llama from HuggingFace Hub and convert to Megatron
bridge = CausalLMBridge.from_hf_pretrained("meta-llama/Llama-3.2-1B")
provider = bridge.to_megatron_provider()

# The provider is lazy - configure parallelism before creating models
provider.tensor_model_parallel_size = 8
provider.pipeline_model_parallel_size = 2

model = provider(wrap_with_ddp=False)
```

### Converting Megatron Models back to HuggingFace

```python
# Export a trained Megatron model to HuggingFace format
bridge.save_hf_pretrained(model, "./my-fine-tuned-llama")

# Or stream weights for memory efficiency
for name, weight in bridge.export_to_hf(model):
    print(f"Exporting {name}: {weight.shape}")
```

### Auto-detection with AutoBridge

```python
from megatron.bridge import AutoBridge

# Automatically detect model architecture and create appropriate bridge
bridge = AutoBridge.from_hf_pretrained("any-supported-huggingface/model")
```

## Architecture Overview

The bridge framework uses a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                   User API Layer                        │
│         (CausalLMBridge, AutoBridge)                    │
├─────────────────────────────────────────────────────────┤
│                Orchestration Layer                      │
│            (MegatronModelBridge)                        │
├─────────────────────────────────────────────────────────┤
│    Mapping Layer         │    Transformation Layer      │
│ (MegatronMappingRegistry)│   (MegatronParamMapping)     │
└─────────────────────────────────────────────────────────┘
```

### Core Components

1. **MegatronModelBridge**: High-level orchestrator that coordinates the conversion process
2. **MegatronMappingRegistry**: Registry of parameter name mappings between formats
3. **MegatronParamMapping**: Handles weight transformations and distributed communication
4. **CausalLMBridge**: User-friendly API for causal language models

## Design Patterns

### Multi-Dispatch Registration

The framework uses decorators to register bridge implementations, enabling automatic routing:

```python
@MegatronModelBridge.register_bridge(source=LlamaForCausalLM, target=GPTModel)
class MegatronCausalLlamaBridge(MegatronModelBridge):
    def provider_bridge(self, hf_pretrained):
        # Convert HF config to Megatron provider
        return LlamaModelProvider(
            num_layers=hf_pretrained.config.num_hidden_layers,
            hidden_size=hf_pretrained.config.hidden_size,
            # ... more config mapping
        )
    
    def mapping_registry(self):
        # Define weight mappings
        return MegatronMappingRegistry(
            TPAwareMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight"
            ),
            # ... more mappings
        )
```

### Param Mapping Strategies

Different weight transformation strategies handle various parallelism patterns:

- **DirectMapping**: Simple 1:1 mapping
- **ColumnParallelMapping**: Splits along output dimension
- **RowParallelMapping**: Splits along input dimension
- **QKVMapping**: Handles QKV matrix interleaving
- **GatedMLPMapping**: Manages gated activation concatenation
- **TPAwareMapping**: Auto-detects and applies correct strategy

## Conversion Process

### HuggingFace → Megatron

1. **Planning Phase**
   - Iterate through Megatron model parameters
   - Query StateBridge for HF source mappings
   - Build execution plan with resolved names

2. **Execution Phase**
   - Load HF weights according to mappings
   - Apply format transformations (e.g., QKV fusion)
   - Distribute via tensor parallelism if needed
   - Copy to target Megatron parameters

3. **Distribution**
   - Column-parallel: scatter along dim 0
   - Row-parallel: scatter along dim 1
   - Replicated: broadcast to all ranks

### Megatron → HuggingFace

1. **Collection Phase**
   - Gather parameter locations across pipeline ranks
   - Build export plan respecting ordering preferences

2. **Export Phase**
   - Broadcast from owning pipeline rank
   - Gather from tensor parallel ranks
   - Apply inverse transformations (e.g., QKV splitting)
   - Yield HF-format weight tensors

## Parallelism Handling

### Tensor Parallelism (TP)

The framework automatically handles three TP patterns:

```python
# Column-parallel (output split)
embeddings: [vocab, hidden] → [vocab, hidden/TP]
qkv_proj: [hidden, 3*hidden] → [hidden, 3*hidden/TP]

# Row-parallel (input split)  
out_proj: [hidden, hidden] → [hidden/TP, hidden]
down_proj: [intermediate, hidden] → [intermediate/TP, hidden]

# Replicated (no split)
layer_norm: [hidden] → [hidden] (same on all ranks)
```

### Pipeline Parallelism (PP)

- Parameters exist on specific pipeline ranks
- Cross-rank broadcasting ensures collective participation
- Metadata broadcast precedes tensor data for efficiency

### Virtual Pipeline Parallelism (VPP)

- Multiple model chunks per device
- Automatic layer offset calculation
- Transparent to weight transformation logic

## Advanced Usage

### Custom Weight Transformations

Create custom param mappings for special formats:

```python
class MyCustomMapping(MegatronParamMapping):
   def hf_to_megatron(self, hf_weights, megatron_module):
        # Custom transformation logic
        transformed = my_transform(hf_weights)
        # Use provided helpers for distribution
        return self.scatter_to_tp_ranks(transformed, dim=0)

   def megatron_to_hf(self, megatron_weights, module):
        # Gather and inverse transform
        gathered = self.gather_from_tp_ranks(megatron_weights, dim=0)
        return {"custom_weight": my_inverse_transform(gathered)}
```

### Exporting Weights

Export weights to HuggingFace format (all ranks receive full weights):

```python
# Export weights
weights = bridge.export_to_hf(model)
```

### Streaming Large Models

Handle massive models without loading all weights:

```python
# Stream weights during conversion
with bridge.stream_conversion(src_model) as stream:
    for weight_batch in stream:
        # Process weights in batches
        process_batch(weight_batch)
```

## Adding New Model Support

To add support for a new model architecture:

1. **Create a Bridge Class**
   ```python
   @MegatronModelBridge.register_bridge(source=YourHFModel, target=YourMegatronModel)
   class YourModelBridge(MegatronModelBridge):
       pass
   ```

2. **Implement Configuration Mapping**
   ```python
   def provider_bridge(self, hf_pretrained):
       return YourModelProvider(
           # Map HF config to Megatron config
       )
   ```

3. **Define Weight Mappings**
   ```python
   def mapping_registry(self):
       return MegatronMappingRegistry(
           # Define all weight mappings
       )
   ```

4. **Register Custom Modules** (if needed)
   ```python
   TPAwareMapping.register_module_type(
       "YourColumnParallelLinear", "column"
   )
   ```

## Implementation Examples

### Llama Bridge

The Llama implementation demonstrates key patterns:

```python
# Special handling for model variations
if is_llama_3_1_config(config):
    # Handle RoPE scaling
    rotary_base = config.rope_scaling["rope_type"] == "llama3"
    
# QKV fusion with proper head ordering
QKVMapping(
    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
    q="model.layers.*.self_attn.q_proj.weight",
    k="model.layers.*.self_attn.k_proj.weight", 
    v="model.layers.*.self_attn.v_proj.weight"
)

# Gated MLP handling
GatedMLPMapping(
    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
    gate="model.layers.*.mlp.gate_proj.weight",
    up="model.layers.*.mlp.up_proj.weight"
)
```

## Best Practices

1. **Always Use High-Level APIs**: Prefer `CausalLMBridge` or `AutoBridge` over direct bridge usage
2. **Configure Before Creating**: Set parallelism parameters on providers before model creation
3. **Handle Missing Weights**: Check for None returns in custom bridges
4. **Test Bidirectionality**: Ensure HF→Megatron→HF preserves weights exactly
5. **Profile Large Models**: Use streaming APIs to avoid memory issues

## Troubleshooting

### Common Issues

1. **Shape Mismatches**: Usually indicates incorrect parallelism configuration
2. **Missing Weights**: Check StateBridge mappings cover all parameters
3. **Memory Errors**: Use streaming APIs or increase distribution

### Debugging Tips

```python
# Enable verbose logging
import logging
logging.getLogger("megatron.bridge.models").setLevel(logging.DEBUG)

# Inspect mappings
bridge = CausalLMBridge.from_hf_pretrained("model")
mapping_registry = bridge.mapping_registry()
print(mapping_registry.get_all_mappings())

# Verify weight shapes
for task in bridge._build_plan_hf_to_megatron(hf_model, meg_models):
    print(f"{task.param_name}: {task.megatron_param.shape}")
```

## Performance Considerations

- **Memory**: Streaming prevents full model loads during conversion
- **Communication**: Optimized collective operations for distributed settings
- **Caching**: Compiled regex patterns for efficient name matching
- **Progress**: Shown only on rank 0 to avoid output clutter

## Future Enhancements

- PEFT model support (LoRA, QLoRA)
- Automatic optimal parallelism detection
- Checkpoint format conversions
- Multi-GPU to single-GPU consolidation utilities