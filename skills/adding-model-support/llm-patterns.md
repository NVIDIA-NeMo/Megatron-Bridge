# LLM Bridge Patterns

Reference implementations:
- Simple dense: Qwen2 (`src/megatron/bridge/models/qwen/qwen2_bridge.py`)
- MoE: GLM-4.5 (`src/megatron/bridge/models/glm/glm45_bridge.py`)
- MoE with custom layer spec: OLMoE (`src/megatron/bridge/models/olmoe/olmoe_bridge.py`)
- Advanced (YaRN, MoE, custom builder): GPT-OSS (`src/megatron/bridge/models/gpt_oss/`)

## Model Config and Builder Pattern

Most GPT-style bridges use `BridgeGPTModelConfig` and the upstream `GPTModelBuilder`. Map every
family setting before the exact MCore `TransformerConfig` is constructed:

```python
def hf_config_to_model_config_kwargs(self, hf_config):
    kwargs = super().hf_config_to_model_config_kwargs(hf_config)
    kwargs.update(
        normalization="RMSNorm",
        gated_linear_unit=True,
        add_bias_linear=False,
        hidden_dropout=0.0,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
    )
    return kwargs
```

### When you need a custom builder

Create an outer model-config subclass and standalone builder when:

1. The family has build parameters absent from MCore `TransformerConfig`.
2. Construction needs a custom model, embedding, RoPE implementation, or layer-spec factory.
3. The default builder cannot represent a hybrid or multimodal architecture.

```python
@dataclass(kw_only=True)
class MyModelConfig(BridgeGPTModelConfig):
    builder: ClassVar[str] = "megatron.bridge.models.my_model.model_config.MyModelBuilder"
    yarn_rotary_scaling_factor: float | None = None
    yarn_original_max_position_embeddings: int | None = None

class MyModelBuilder(GPTModelBuilder):
    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        # Pass family build data explicitly, then construct the model.
        return super().build_model(pg_collection, pre_process, post_process, vp_stage)
```

The nested config must remain an exact MCore class:

```python
config = bridge.model_config_bridge(hf_pretrained)
assert type(config.transformer) is TransformerConfig
```

Do not subclass `TransformerConfig` for family fields and do not attach phantom fields during HF
mapping. Existing provider classes are legacy compatibility only and must not be imported by the
new model-config/builder path.

### No Size-Specific Config Classes

Do not create config subclasses whose names combine a model class with a model-size suffix. The
bridge derives architecture fields from the Hugging Face config.

For recipe presets, keep the size in the recipe function name and configure the family model config:

```python
def my_model_7b_pretrain_config() -> ConfigContainer:
    cfg = _pretrain_common()
    cfg.model = MyModelConfig(
        transformer=TransformerConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            num_query_groups=8,
            ffn_hidden_size=14336,
        ),
        vocab_size=128256,
    )
    return cfg
```

When the recipe targets an existing HF checkpoint, derive the model config from HF config:

```python
cfg.model = AutoBridge.from_hf_pretrained("org/my-model-7b").get_model_config()
```

## Bridge Pattern

```python
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping, GatedMLPMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM

@MegatronModelBridge.register_bridge(
    source=MyModelForCausalLM,    # HF class (or string "MyModelForCausalLM")
    target=GPTModel,               # Megatron target
    model_type="my_model",         # HF model_type
)
class MyModelBridge(MegatronModelBridge):
    def hf_config_to_model_config_kwargs(self, hf_config):
        kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        kwargs.update(
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            hidden_dropout=0.0,
        )
        return kwargs

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

The base class provides automatic mapping for common fields — no need to duplicate:

```text
(num_hidden_layers, num_layers), (hidden_size, hidden_size),
(intermediate_size, ffn_hidden_size), (num_attention_heads, num_attention_heads),
(num_key_value_heads, num_query_groups), (head_dim, kv_channels),
(vocab_size, vocab_size), (max_position_embeddings, seq_length),
(rms_norm_eps, layernorm_epsilon), (rope_theta, rotary_base),
(tie_word_embeddings, share_embeddings_and_output_weights),
(attention_bias, add_qkv_bias), (mlp_bias, add_bias_linear),
```

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

## Registration Options

| Parameter | Required | Description |
|-----------|----------|-------------|
| `source` | Yes | HF model class or string class name |
| `target` | Yes | Megatron model class (usually `GPTModel`) |
| `model_type` | No | HF `model_type` string for export config |

If `source` is a string (model not importable), the bridge is matched by class name.
