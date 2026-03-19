# LLM Bridge Patterns

Reference implementation: GPT-OSS (`src/megatron/bridge/models/gpt_oss/`)

## Provider Pattern

Subclass `GPTModelProvider` with model-specific fields.

```python
@dataclass
class MyModelProvider(GPTModelProvider):
    # Model-specific fields beyond what GPTModelProvider provides
    # Examples from GPT-OSS:
    normalization: str = "RMSNorm"
    gated_linear_unit: bool = True
    add_bias_linear: bool = False
    share_embeddings_and_output_weights: bool = False

    # YARN RoPE (if applicable)
    position_embedding_type: str = "rope"
    rotary_base: float = 10000.0
    yarn_rotary_scaling_factor: Optional[float] = None
    yarn_original_max_position_embeddings: Optional[int] = None

    # MoE (if applicable)
    moe_router_topk: int = 2
    moe_grouped_gemm: bool = True
    moe_token_dispatcher_type: str = "alltoall"

    # Sliding window (if applicable)
    window_size: Optional[Tuple[int, int]] = None

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        # Override only if custom logic needed (e.g. TE version checks)
        return super().provide(pre_process, post_process, vp_stage)
```

### Predefined size variants

Create size-specific subclasses with hardcoded defaults:

```python
@dataclass
class MyModelProvider7B(MyModelProvider):
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_query_groups: int = 8
    ffn_hidden_size: int = 14336
    vocab_size: int = 128256

@dataclass
class MyModelProvider70B(MyModelProvider):
    num_layers: int = 80
    hidden_size: int = 8192
    # ...
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

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MyModelProvider:
        # Option A: Use base class CONFIG_MAPPING (simplest)
        provider = super().provider_bridge(hf_pretrained)

        # Option B: Manual mapping
        cfg = hf_pretrained.config
        provider_kwargs = self.hf_config_to_provider_kwargs(cfg)
        provider = MyModelProvider(**provider_kwargs)

        # Set model-specific fields not in CONFIG_MAPPING
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        # YARN/RoPE scaling
        if hasattr(cfg, "rope_scaling") and cfg.rope_scaling:
            self._apply_rope_scaling(provider, cfg.rope_scaling)

        return provider

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

```
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
| `provider` | No | Provider class (defaults to `GPTModelProvider`) |
| `model_type` | No | HF `model_type` string for export config |

If `source` is a string (model not importable), the bridge is matched by class name.
