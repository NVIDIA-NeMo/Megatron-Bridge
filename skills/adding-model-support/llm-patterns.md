# LLM Bridge Patterns

Reference implementations:
- Dense, builder-backed config and conversion: Qwen3 (`src/megatron/bridge/models/qwen/qwen3_bridge.py`)
- Custom model config and builder: Muse Glimmer (`src/megatron/bridge/models/muse_glimmer/`)
- MoE weight mappings: GLM-4.5 (`src/megatron/bridge/models/glm/glm45_bridge.py`)
- MoE with custom layer spec: OLMoE (`src/megatron/bridge/models/olmoe/olmoe_bridge.py`)

GLM-4.5 and OLMoE still configure the model through the legacy `provider_bridge()`; use them
for weight mappings and layer specs, and use Qwen3 for the config pattern.

## Model Config Pattern

Most bridges do **not** need a custom config class. The base `hf_config_to_model_config_kwargs()`
uses `CONFIG_MAPPING` to derive flat Megatron config fields from the HF config. The bridge adds its
model-specific fields to those kwargs, and `AutoBridge.get_model_config()` partitions them into a
`BridgeGPTModelConfig` and its nested Bridge `TransformerConfig`.

```python
USE_MODEL_CONFIG_FOR_CONVERSION = True

def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
    config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
    config_kwargs.update(
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        autocast_dtype=torch.bfloat16,
        masked_softmax_fusion=True,
        rope_scaling=False,
        rope_scaling_factor=1.0,
        # MoE settings (if applicable)
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
    )
    # The shared mapping already selects "yarn" for YaRN-scaled checkpoints.
    config_kwargs.setdefault("position_embedding_type", "rope")
    return config_kwargs

def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
    """Adapt the canonical builder mapping to the deprecated provider path."""
    return self.hf_config_to_model_config_kwargs(hf_config)
```

Every key must be a declared field of the model config or its nested transformer config. Unknown
keys raise `ValueError`, and assigning an undeclared attribute on the config raises
`AttributeError`. Do not store extra values for export on the config; derive them in
`megatron_to_hf_config()` or declare them in a custom config class.

### Defaults that differ from the legacy provider

The builder config starts from Megatron Core defaults, not `GPTModelProvider` defaults. Set these
explicitly when the model needs the provider-era value:

| Field | Builder config default | Legacy provider value |
|-------|------------------------|-----------------------|
| `position_embedding_type` | `"learned_absolute"` | `"rope"` (set by the base `provider_bridge()`) |
| `masked_softmax_fusion` | `False` | `True` |
| `rope_scaling_factor` | `8.0` | `1.0` |

Use `setdefault` for `position_embedding_type` so YaRN checkpoints keep the `"yarn"` value set by
the base mapping.

### Layer specs

- Leave `transformer_layer_spec` unset for standard GPT layers. The builder selects Megatron Core's
  default spec, which is `get_gpt_decoder_block_spec` for MoE models and honors `moe_layer_freq`
  for dense-first layouts.
- Pass a custom spec as `transformer_layer_spec=<callable>` in the kwargs. The builder calls it with
  the builder config, so the callable must read only fields that config declares. If the same
  callable also serves the legacy provider, branch on `isinstance(config, GPTModelProvider)` before
  calling provider-only helpers such as `default_layer_spec`.

### YaRN

YaRN needs no custom config. The `yarn_*` fields are declared on Bridge `TransformerConfig`, and the
base mapping sets `position_embedding_type="yarn"` and the `yarn_*` fields from a `rope_scaling`
dict whose type is `"yarn"`. Hugging Face YaRN configs often omit `beta_fast`, `beta_slow` and
`truncate`; check that the mapped config carries Hugging Face's defaults (32, 1 and `True`) for
them. Megatron Core passes `None` through to its YaRN embedding and fails at model construction.

### When you need a custom config class

Create a `ModelConfig` subclass and a `ModelBuilder` only when:

1. **Extra config fields** — The model needs fields that neither `BridgeGPTModelConfig` nor Bridge
   `TransformerConfig` declares, for example decoder constants or modality token IDs.
2. **Custom construction** — The model needs a different builder, such as a HybridModel layer
   pattern or a combined vision+language model.

Follow Muse Glimmer: `muse_glimmer_config.py` declares the config classes, `muse_glimmer_builder.py`
subclasses `HybridModelBuilder`, and the bridge sets `MODEL_CONFIG_CLASS` and overrides
`hf_config_to_model_config()` and `megatron_to_hf_config()`.

### No Size-Specific Provider or Config Classes

Do not create provider or config subclasses whose names combine `ModelProvider` or `ModelConfig`
with a model-size suffix. Examples of forbidden suffixes include `7B`, `200M`, and `A3B`. The bridge
should derive architecture fields from the Hugging Face config via `AutoBridge` /
`MegatronModelBridge` config mapping.

For recipe presets, keep the size in the recipe function name and derive the config from the HF
checkpoint:

```python
cfg.model = AutoBridge.from_hf_pretrained("org/my-model-7b").get_model_config()
```

When the recipe needs an architecture without a published checkpoint, build it from an explicit HF
config inside the function:

```python
def my_model_7b_pretrain_config() -> ConfigContainer:
    cfg = _pretrain_common()
    hf_config = AutoConfig.for_model(
        "my_model",
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=128256,
    )
    cfg.model = AutoBridge.from_hf_config(hf_config).get_model_config()
    return cfg
```

## Bridge Pattern

```python
from typing import Any

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping, QKVMapping


@MegatronModelBridge.register_bridge(
    source=MyModelForCausalLM,    # HF class (or string "MyModelForCausalLM")
    target=GPTModel,               # Megatron target
    model_type="my_model",         # HF model_type
)
class MyModelBridge(MegatronModelBridge):
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            hidden_dropout=0.0,
            autocast_dtype=torch.bfloat16,
            masked_softmax_fusion=True,
            rope_scaling=False,
            rope_scaling_factor=1.0,
        )
        config_kwargs.setdefault("position_embedding_type", "rope")
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        return self.hf_config_to_model_config_kwargs(hf_config)

    def mapping_registry(self) -> MegatronMappingRegistry:
        return MegatronMappingRegistry(
            # Embeddings
            AutoMapping(
                megatron_param="embedding.word_embeddings.weight",
                hf_param="model.embed_tokens.weight",
            ),
            # Output layer
            AutoMapping(
                megatron_param="output_layer.weight",
                hf_param="lm_head.weight",
            ),
            # Final layernorm
            AutoMapping(
                megatron_param="decoder.final_layernorm.weight",
                hf_param="model.norm.weight",
            ),
            # QKV (fused)
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            ),
            # Attention output projection
            AutoMapping(
                megatron_param="decoder.layers.*.self_attention.linear_proj.weight",
                hf_param="model.layers.*.self_attn.o_proj.weight",
            ),
            # MLP (gated)
            GatedMLPMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                gate="model.layers.*.mlp.gate_proj.weight",
                up="model.layers.*.mlp.up_proj.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc2.weight",
                hf_param="model.layers.*.mlp.down_proj.weight",
            ),
            # Layer norms
            AutoMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
                hf_param="model.layers.*.input_layernorm.weight",
            ),
            AutoMapping(
                megatron_param="decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
                hf_param="model.layers.*.post_attention_layernorm.weight",
            ),
        )
```

### Base CONFIG_MAPPING

The base class provides automatic mapping for common fields — no need to duplicate. The full list
is `MegatronModelBridge.CONFIG_MAPPING`; common entries:

```text
(num_hidden_layers, num_layers), (hidden_size, hidden_size),
(intermediate_size, ffn_hidden_size), (num_attention_heads, num_attention_heads),
(num_key_value_heads, num_query_groups), (head_dim, kv_channels),
(vocab_size, vocab_size), (max_position_embeddings, seq_length),
(rms_norm_eps, layernorm_epsilon), (rope_theta, rotary_base),
(tie_word_embeddings, share_embeddings_and_output_weights),
(attention_bias, add_qkv_bias), (mlp_bias, add_bias_linear),
```

The base mapping also derives `params_dtype`, `bf16`/`fp16`, `activation_func`,
`make_vocab_size_divisible_by` and the YaRN fields.

### MoE weight mappings

For models with Mixture of Experts, use expert-specific mappings:

```python
ExpertMLPGateUpProjMapping(
    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
    up="model.layers.*.mlp.experts.*.up_proj.weight",
),
ExpertMLPDownProjMapping(
    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
    hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
),
AutoMapping(
    megatron_param="decoder.layers.*.mlp.router.weight",
    hf_param="model.layers.*.mlp.gate.weight",
),
```

### Optional weight modification hooks

Override these for special handling (e.g., quantized weights, expert layout):

```python
def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
    """Transform HF weights before loading into Megatron (e.g., dequantize)."""
    return hf_state_dict[hf_param]

def maybe_modify_converted_hf_weight(self, task, converted_weights_dict, hf_state_dict):
    """Transform weights after Megatron→HF conversion (e.g., merge expert shards)."""
    return converted_weights_dict
```

## Migrating a Provider-Based Bridge

Many existing bridges still override `provider_bridge()`. For those families
`get_model_config()` returns a `BridgeGPTModelConfig` built only from `CONFIG_MAPPING`, which
silently drops every attribute set in `provider_bridge()`. To migrate a family:

1. Move the `provider_bridge()` assignments into `hf_config_to_model_config_kwargs()`, add the
   [differing defaults](#defaults-that-differ-from-the-legacy-provider), make
   `hf_config_to_provider_kwargs()` return the same kwargs, and delete the `provider_bridge()`
   override.
2. Keep provider-only values, such as a decoder block spec `partial` or provider-only fields, in
   `hf_config_to_provider_kwargs()` only.
3. Replace phantom provider attributes, which the strict builder config rejects, with declared
   fields or values derived in `megatron_to_hf_config()`.
4. Set `USE_MODEL_CONFIG_FOR_CONVERSION = True`.
5. Switch the family's recipes from `.to_megatron_provider(load_weights=False)` to
   `.get_model_config()`, and update recipe-test stubs that fake `AutoBridge`, including tests
   outside `tests/unit_tests/recipes/<family>/`.
6. Verify that the provider fields are unchanged from `main`, that `get_model_config()` matches the
   provider on every shared field, that export reproduces the source HF weights, and that logits are
   identical before and after, including on a checkpoint imported by `main`.

## Registration Options

| Parameter | Required | Description |
|-----------|----------|-------------|
| `source` | Yes | HF model class or string class name |
| `target` | Yes | Megatron model class (usually `GPTModel`) |
| `provider` | No | Legacy provider class for families that still construct through `provider_bridge()` (defaults to `GPTModelProvider`) |
| `model_type` | No | HF `model_type` string for export config |

If `source` is a string (model not importable), the bridge is matched by class name. The
builder-backed config class is selected by the `MODEL_CONFIG_CLASS` class attribute
(`BridgeGPTModelConfig` by default), not by a registration parameter.
