# Contributing a Model — Quick Reference

A condensed checklist for adding a new model bridge. For the full guide with detailed explanations, see [adding-new-models.md](adding-new-models.md).

> **AI-assisted?** If you're using an AI coding agent (Cursor, Claude Code, Codex, etc.), point it at the [`skills/adding-model-support/`](../skills/adding-model-support/) directory — it contains structured step-by-step guides for LLMs, VLMs, recipes, and tests that agents can follow directly.

## File Checklist

```
src/megatron/bridge/models/<model>/
├── __init__.py                    # Register imports
├── <model>_provider.py            # HF config → TransformerConfig
├── <model>_bridge.py              # Bridge + MegatronMappingRegistry
└── README.md                      # (optional) Model-specific notes

tests/unit_tests/models/<model>/
├── test_<model>_bridge.py         # Roundtrip conversion test
└── test_<model>_provider.py       # Provider config mapping test
```

## Step-by-Step

### 1. Provider (`<model>_provider.py`)

Map HF config fields to Megatron `TransformerConfig`. Start from `GPTModelProvider`:

```python
from megatron.bridge.models.gpt_provider import GPTModelProvider

class MyModelProvider(GPTModelProvider):
    def _build_transformer_config(self, hf_config):
        return TransformerConfig(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            # ... map remaining fields
        )
```

### 2. Bridge (`<model>_bridge.py`)

Register the bridge and define parameter mappings:

```python
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping, GatedMLPMapping

@MegatronModelBridge.register_bridge(
    source=MyModelForCausalLM,
    target=GPTModel,
    model_type="my_model",
)
class MyModelBridge(MegatronModelBridge):
    def provider_bridge(self, hf_model):
        return MyModelProvider(...)

    def mapping_registry(self):
        return MegatronMappingRegistry([
            AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
            QKVMapping(...),
            GatedMLPMapping(...),
            AutoMapping("decoder.final_layernorm.weight", "model.norm.weight"),
            AutoMapping("output_layer.weight", "lm_head.weight"),
        ])
```

### 3. Tests

Minimum tests to include:

```python
# test_<model>_bridge.py
def test_roundtrip_conversion():
    """Load HF → Megatron → export HF, verify weights match."""
    bridge = AutoBridge.from_hf_pretrained("org/small-model")
    megatron_model = bridge.to_megatron_model(wrap_with_ddp=False)
    bridge.save_hf_pretrained(megatron_model, "./export")
    # Compare original vs exported weights

# test_<model>_provider.py
def test_provider_config_mapping():
    """Verify HF config maps correctly to TransformerConfig."""
    bridge = AutoBridge.from_hf_config(config)
    tc = bridge.transformer_config
    assert tc.num_layers == config.num_hidden_layers
    assert tc.hidden_size == config.hidden_size
```

### 4. Validate

```bash
# Run your new tests
make test-k K=test_my_model

# Or with uv directly
uv run python -m pytest tests/unit_tests/models/my_model/ -v
```

## Common Mapping Patterns

| HF Pattern | Megatron Pattern | Mapping Class |
|---|---|---|
| `embed_tokens.weight` | `embedding.word_embeddings.weight` | `AutoMapping` |
| `q_proj` / `k_proj` / `v_proj` | `self_attention.linear_qkv` | `QKVMapping` |
| `gate_proj` / `up_proj` | `mlp.linear_fc1` | `GatedMLPMapping` |
| `down_proj` | `mlp.linear_fc2` | `AutoMapping` |
| `input_layernorm` | `self_attention.linear_qkv.layer_norm_weight` | `AutoMapping` |
| `post_attention_layernorm` | `mlp.linear_fc1.layer_norm_weight` | `AutoMapping` |
| `model.norm` | `decoder.final_layernorm` | `AutoMapping` |
| `lm_head` | `output_layer` | `AutoMapping` |

## Tips

- **Copy from a similar model.** Llama is the simplest reference; Qwen adds MoE; DeepSeek adds MLA.
- **Use `scripts/scaffold_new_model.py`** to generate the boilerplate files automatically.
- **Run `list_supported_architectures.py`** to verify your model appears after registration.
- **Wildcard count must match** in `MegatronMappingRegistry` — if you have 32 layers, your `layers.*` pattern must resolve to 32 entries on both sides.
