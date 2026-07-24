# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron Bridge for Gemma 4 text-only (CausalLM).

Supports all Gemma 4 text variants:
  - MoE (``enable_moe_block=True``): ``Gemma4ForCausalLM`` (26B-A4B and similar)
  - Dense (``enable_moe_block=False``): same HF class, dispatched via ``Gemma4DenseProvider``

Usage::

  AutoBridge.from_hf_pretrained("google/gemma-4-26B-A4B")
    └─ Gemma4Bridge (registered for Gemma4ForCausalLM)
         ├─ provider_bridge()  MoE   → Gemma4ModelProvider
         │                     Dense → Gemma4DenseProvider
         └─ mapping_registry()  MoE path   → _moe_mapping_registry()
                                 Dense path → _dense_mapping_registry()
"""

import re
from typing import Any, Mapping

import torch
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
    split_qkv_weights,
)
from megatron.bridge.models.conversion.peft_bridge import ABSENT_PROJECTION
from megatron.bridge.models.conversion.transformers_compat import (
    rope_local_base_freq_from_hf,
    rope_theta_from_hf,
)
from megatron.bridge.models.conversion.utils import mcore_to_hf_window_size
from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider, Gemma4ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# Register Gemma4 custom module types for AutoMapping
AutoMapping.register_module_type("Gemma4TEDotProductAttention", "replicated")
AutoMapping.register_module_type("Gemma4SelfAttention", "replicated")
AutoMapping.register_module_type("Gemma4TransformerLayer", "replicated")
AutoMapping.register_module_type("Gemma4TopKRouter", "replicated")
AutoMapping.register_module_type("Gemma4MoELayer", "replicated")
AutoMapping.register_module_type("SharedExpertMLP", "column")


class _Gemma4QKVMapping(QKVMapping):
    """QKV mapping tolerating missing v_proj on global attention layers (K=V)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_hf_name_mismatch = True


class _Gemma4DenseQKVMapping(QKVMapping):
    """QKV mapping tolerating missing k_proj AND v_proj on shared-KV layers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.allow_hf_name_mismatch = True


def _infer_attn_pattern(layer_types: list[str]) -> tuple[int, int]:
    """Infer (sliding, global) interleaved attention pattern from layer_types list."""
    for i, lt in enumerate(layer_types):
        if lt == "full_attention":
            sliding_count = i
            full_count = 0
            for j in range(i, len(layer_types)):
                if layer_types[j] == "full_attention":
                    full_count += 1
                else:
                    break
            return (sliding_count, full_count)
    return (len(layer_types), 0)


def _layer_types_from_provider(provider: Gemma4ModelProvider | Gemma4DenseProvider) -> list[str]:
    """Reconstruct the Hugging Face per-layer attention pattern."""
    if isinstance(provider, Gemma4DenseProvider):
        pattern = provider.window_attn_skip_freq
    else:
        pattern = provider.interleaved_attn_pattern

    if isinstance(pattern, list):
        layer_types = [
            value if isinstance(value, str) else "sliding_attention" if bool(value) else "full_attention"
            for value in pattern
        ]
    elif isinstance(pattern, tuple) and len(pattern) == 2:
        sliding_count, full_count = pattern
        cycle = ["sliding_attention"] * sliding_count + ["full_attention"] * full_count
        layer_types = [cycle[index % len(cycle)] for index in range(provider.num_layers)] if cycle else []
    elif isinstance(pattern, int) and pattern > 0:
        layer_types = [
            "sliding_attention" if layer_number % pattern else "full_attention"
            for layer_number in range(1, provider.num_layers + 1)
        ]
    else:
        layer_types = ["full_attention"] * provider.num_layers

    if layer_types:
        layer_types[-1] = "full_attention"
    return layer_types


def _rope_parameters_from_provider(provider: Gemma4ModelProvider | Gemma4DenseProvider) -> dict[str, dict]:
    """Reconstruct Gemma 4's dual local/global RoPE configuration."""
    if isinstance(provider, Gemma4DenseProvider):
        local_base = provider.sliding_window_rope_base
        global_base = provider.full_attention_rope_base
        global_partial_factor = provider.full_attention_rope_partial_factor
    else:
        rotary_base = provider.rotary_base
        if isinstance(rotary_base, tuple):
            local_base, global_base = rotary_base
        else:
            local_base = global_base = rotary_base
        global_partial_factor = provider.global_rotary_percent

    return {
        "full_attention": {
            "partial_rotary_factor": global_partial_factor,
            "rope_theta": global_base,
            "rope_type": "proportional",
        },
        "sliding_attention": {
            "rope_theta": local_base,
            "rope_type": "default",
        },
    }


# ---------------------------------------------------------------------------
# Gemma4Bridge — text-only CausalLM bridge (MoE and Dense)
# ---------------------------------------------------------------------------


@MegatronModelBridge.register_bridge(
    source="Gemma4ForCausalLM",
    target=GPTModel,
    provider=Gemma4ModelProvider,
    model_type="gemma4",
)
class Gemma4Bridge(MegatronModelBridge):
    """Megatron Bridge for Gemma 4 text-only (CausalLM).

    Dispatches to Dense or MoE path based on ``enable_moe_block`` in HF config.
    """

    _CONDITIONAL_MOE_FIELDS = frozenset({"num_moe_experts", "moe_router_topk", "moe_ffn_hidden_size"})

    def _should_map_hf_config_field(self, hf_config: Any, hf_name: str, megatron_name: str, value: Any) -> bool:
        if megatron_name in self._CONDITIONAL_MOE_FIELDS:
            return getattr(hf_config, "enable_moe_block", True)
        return super()._should_map_hf_config_field(hf_config, hf_name, megatron_name, value)

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> "Gemma4ModelProvider | Gemma4DenseProvider":
        hf_config = hf_pretrained.config
        if not getattr(hf_config, "enable_moe_block", False):
            self._is_dense = True
            return self._build_dense_provider(hf_config)

        self._is_dense = False
        return self._build_moe_provider(hf_config)

    def _text_config(self) -> Any | None:
        """Return the text config used to dispatch dense vs MoE behavior."""
        return getattr(self, "hf_config", None)

    def _is_dense_config(self) -> bool:
        """Return whether the current HF config describes a dense Gemma 4 model."""
        if getattr(self, "_is_dense", False):
            return True
        text_config = self._text_config()
        return text_config is not None and not getattr(text_config, "enable_moe_block", False)

    def _build_dense_provider(self, hf_config) -> Gemma4DenseProvider:
        """Build a Gemma4DenseProvider from HF config."""
        rope_params = getattr(hf_config, "rope_parameters", {}) or {}
        sliding_rope = rope_params.get("sliding_attention", {})
        full_rope = rope_params.get("full_attention", {})
        num_attention_heads = hf_config.num_attention_heads
        num_query_groups = hf_config.num_key_value_heads
        num_global_query_groups = getattr(
            hf_config,
            "num_global_key_value_heads",
            num_query_groups,
        )

        self._dense_num_attention_heads = num_attention_heads
        self._dense_num_query_groups = num_query_groups
        self._dense_num_global_query_groups = num_global_query_groups

        layer_types = getattr(hf_config, "layer_types", None)
        if layer_types is not None:
            layer_types = [layer_type == "sliding_attention" for layer_type in layer_types]

        sliding_window = getattr(hf_config, "sliding_window", 512)

        return Gemma4DenseProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=num_attention_heads,
            num_query_groups=num_query_groups,
            kv_channels=getattr(hf_config, "head_dim", 256),
            global_kv_channels=getattr(hf_config, "global_head_dim", 512),
            num_global_query_groups=num_global_query_groups,
            seq_length=hf_config.max_position_embeddings,
            vocab_size=hf_config.vocab_size,
            normalization="RMSNorm",
            layernorm_epsilon=hf_config.rms_norm_eps,
            window_size=(sliding_window - 1, 0),
            window_attn_skip_freq=layer_types if layer_types is not None else 6,
            sliding_window_rope_base=sliding_rope.get("rope_theta", 10000.0),
            full_attention_rope_base=full_rope.get("rope_theta", 1000000.0),
            full_attention_rope_partial_factor=full_rope.get("partial_rotary_factor", 0.25),
            num_kv_shared_layers=getattr(hf_config, "num_kv_shared_layers", 0),
            use_double_wide_mlp=getattr(hf_config, "use_double_wide_mlp", False),
            attention_k_eq_v=getattr(hf_config, "attention_k_eq_v", False),
            per_layer_embed_vocab_size=getattr(hf_config, "vocab_size_per_layer_input", hf_config.vocab_size),
            per_layer_embed_dim=getattr(hf_config, "hidden_size_per_layer_input", 256),
            final_logit_softcapping=getattr(hf_config, "final_logit_softcapping", None),
            bf16=True,
        )

    def _build_moe_provider(self, hf_config) -> Gemma4ModelProvider:
        """Build a Gemma4ModelProvider from HF config (MoE path)."""
        provider_kwargs = self.hf_config_to_provider_kwargs(hf_config)
        provider = Gemma4ModelProvider(**provider_kwargs)

        provider.window_size = getattr(hf_config, "sliding_window", 1024)
        provider.rotary_base = (
            rope_local_base_freq_from_hf(hf_config),
            rope_theta_from_hf(hf_config),
        )

        head_dim = getattr(hf_config, "head_dim", 256)
        provider.softmax_scale = 1.0
        provider.kv_channels = head_dim
        provider.qk_layernorm = True

        provider.global_head_dim = getattr(hf_config, "global_head_dim", 512)
        provider.num_global_key_value_heads = getattr(hf_config, "num_global_key_value_heads", 2)
        provider.attention_k_eq_v = getattr(hf_config, "attention_k_eq_v", False)

        rope_params = getattr(hf_config, "rope_parameters", {})
        if isinstance(rope_params, dict):
            full_attn_rope = rope_params.get("full_attention", {})
            provider.global_rotary_percent = full_attn_rope.get("partial_rotary_factor", 0.25)

        layer_types = getattr(hf_config, "layer_types", None)
        if layer_types:
            provider.interleaved_attn_pattern = _infer_attn_pattern(layer_types)

        if getattr(hf_config, "enable_moe_block", False):
            provider.num_moe_experts = getattr(hf_config, "num_experts", 128)
            provider.moe_router_topk = getattr(hf_config, "top_k_experts", 8)
            provider.moe_ffn_hidden_size = getattr(hf_config, "moe_intermediate_size", 704)
            provider.moe_shared_expert_intermediate_size = getattr(hf_config, "intermediate_size", 2112)
            provider.moe_shared_expert_overlap = False
            provider.moe_shared_expert_gate = False
            provider.moe_layer_freq = 1

        provider.final_logit_softcapping = getattr(hf_config, "final_logit_softcapping", 30.0)
        provider.bf16 = True
        provider.params_dtype = torch.bfloat16
        provider.autocast_dtype = torch.bfloat16
        provider.make_vocab_size_divisible_by = 128

        return provider

    @classmethod
    def megatron_to_hf_config(cls, provider: Gemma4ModelProvider | Gemma4DenseProvider) -> dict:
        """Preserve Gemma 4 architecture fields affected by provider conversion."""
        hf_config = super().megatron_to_hf_config(provider)
        dtype = hf_config.pop("torch_dtype", None)
        if dtype is not None:
            hf_config["dtype"] = dtype
        hf_config.pop("partial_rotary_factor", None)
        hf_config.pop("rope_theta", None)
        is_moe = provider.num_moe_experts is not None
        window_size = provider.window_size
        hf_config.update(
            {
                "attention_k_eq_v": provider.attention_k_eq_v,
                "enable_moe_block": is_moe,
                "final_logit_softcapping": provider.final_logit_softcapping,
                "global_head_dim": (provider.global_head_dim if is_moe else provider.global_kv_channels),
                "hidden_size_per_layer_input": getattr(provider, "per_layer_embed_dim", 0),
                "layer_types": _layer_types_from_provider(provider),
                "num_kv_shared_layers": getattr(provider, "num_kv_shared_layers", 0),
                "num_global_key_value_heads": (
                    provider.num_global_key_value_heads if is_moe else provider.num_global_query_groups
                ),
                "rope_parameters": _rope_parameters_from_provider(provider),
                "sliding_window": mcore_to_hf_window_size(window_size),
                "use_double_wide_mlp": getattr(provider, "use_double_wide_mlp", False),
                "vocab_size_per_layer_input": getattr(provider, "per_layer_embed_vocab_size", provider.vocab_size),
            }
        )
        if is_moe:
            hf_config.update(
                {
                    "intermediate_size": provider.moe_shared_expert_intermediate_size,
                    "moe_intermediate_size": provider.moe_ffn_hidden_size,
                    "num_experts": provider.num_moe_experts,
                    "top_k_experts": provider.moe_router_topk,
                }
            )
            hf_config.pop("num_experts_per_tok", None)
        return hf_config

    def maybe_modify_converted_hf_weight(self, task, converted_weights_dict, hf_state_dict):
        """Un-fuse fused weights and drop synthesized keys on export."""
        if not hf_state_dict:
            return converted_weights_dict

        result = {}
        for hf_name, tensor in converted_weights_dict.items():
            if hf_name not in hf_state_dict:
                continue

            result[hf_name] = tensor

        return result

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Handle special weight loading for Gemma 4."""
        if isinstance(hf_param, dict) and "v" in hf_param:
            k_name = hf_param["k"]
            v_name = hf_param["v"]
            q_name = hf_param["q"]

            if k_name not in hf_state_dict and v_name not in hf_state_dict:
                q_weight = hf_state_dict[q_name]
                text_config = self._text_config()
                num_q_heads = getattr(
                    text_config, "num_attention_heads", getattr(self, "_dense_num_attention_heads", 8)
                )
                kv_head_dim = q_weight.shape[0] // num_q_heads
                num_kv_heads = getattr(text_config, "num_key_value_heads", getattr(self, "_dense_num_query_groups", 2))
                layer_match = re.search(r"layers\.(\d+)\.", q_name)
                layer_types = getattr(text_config, "layer_types", None)
                if layer_match and layer_types:
                    layer_idx = int(layer_match.group(1))
                    if layer_idx < len(layer_types) and layer_types[layer_idx] == "full_attention":
                        num_kv_heads = getattr(
                            text_config,
                            "num_global_key_value_heads",
                            getattr(self, "_dense_num_global_query_groups", num_kv_heads),
                        )
                elif hasattr(self, "_dense_num_global_query_groups"):
                    num_kv_heads = self._dense_num_global_query_groups
                kv_shape = (num_kv_heads * kv_head_dim, q_weight.shape[1])
                k_zero = torch.zeros(kv_shape, dtype=q_weight.dtype, device=q_weight.device)
                return {"q": q_weight, "k": k_zero, "v": torch.zeros_like(k_zero)}

            if v_name not in hf_state_dict and k_name in hf_state_dict:
                hf_weights = {}
                for role, name in hf_param.items():
                    if role == "v":
                        hf_weights[role] = hf_state_dict[k_name].clone()
                    else:
                        hf_weights[role] = hf_state_dict[name]
                return hf_weights

        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def mapping_registry(self) -> MegatronMappingRegistry:
        if self._is_dense_config():
            return self._dense_mapping_registry()
        return self._moe_mapping_registry()

    def _dense_mapping_registry(self, megatron_prefix: str = "") -> MegatronMappingRegistry:
        """Parameter mappings for the Dense variant."""
        mp = megatron_prefix
        hp = self._hf_layer_prefix()
        param_mappings = {
            f"{mp}embedding.word_embeddings.weight": f"{hp}embed_tokens.weight",
            f"{mp}decoder.final_layernorm.weight": f"{hp}norm.weight",
            f"{mp}per_layer_embedding.weight": f"{hp}embed_tokens_per_layer.weight",
            f"{mp}per_layer_model_proj.weight": f"{hp}per_layer_model_projection.weight",
            f"{mp}decoder.layers.*.input_layernorm.weight": f"{hp}layers.*.input_layernorm.weight",
            f"{mp}decoder.layers.*.post_self_attn_layernorm.weight": f"{hp}layers.*.post_attention_layernorm.weight",
            f"{mp}decoder.layers.*.pre_mlp_layernorm.weight": f"{hp}layers.*.pre_feedforward_layernorm.weight",
            f"{mp}decoder.layers.*.post_mlp_layernorm.weight": f"{hp}layers.*.post_feedforward_layernorm.weight",
            f"{mp}decoder.layers.*.self_attention.q_layernorm.weight": f"{hp}layers.*.self_attn.q_norm.weight",
            f"{mp}decoder.layers.*.self_attention.k_layernorm.weight": f"{hp}layers.*.self_attn.k_norm.weight",
            f"{mp}decoder.layers.*.self_attention.linear_proj.weight": f"{hp}layers.*.self_attn.o_proj.weight",
            f"{mp}decoder.layers.*.mlp.linear_fc2.weight": f"{hp}layers.*.mlp.down_proj.weight",
        }
        mapping_list = [AutoMapping(megatron_param=m, hf_param=h) for m, h in param_mappings.items()]

        mapping_list.append(
            ReplicatedMapping(
                megatron_param=f"{mp}per_layer_proj_norm.weight",
                hf_param=f"{hp}per_layer_projection_norm.weight",
            )
        )
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param=f"{mp}decoder.layers.*.per_layer_input_gate.weight",
                    hf_param=f"{hp}layers.*.per_layer_input_gate.weight",
                ),
                ReplicatedMapping(
                    megatron_param=f"{mp}decoder.layers.*.per_layer_projection.weight",
                    hf_param=f"{hp}layers.*.per_layer_projection.weight",
                ),
                ReplicatedMapping(
                    megatron_param=f"{mp}decoder.layers.*.post_per_layer_input_norm.weight",
                    hf_param=f"{hp}layers.*.post_per_layer_input_norm.weight",
                ),
                ReplicatedMapping(
                    megatron_param=f"{mp}decoder.layers.*.layer_scalar",
                    hf_param=f"{hp}layers.*.layer_scalar",
                ),
                _Gemma4DenseQKVMapping(
                    megatron_param=f"{mp}decoder.layers.*.self_attention.linear_qkv.weight",
                    q=f"{hp}layers.*.self_attn.q_proj.weight",
                    k=f"{hp}layers.*.self_attn.k_proj.weight",
                    v=f"{hp}layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param=f"{mp}decoder.layers.*.mlp.linear_fc1.weight",
                    gate=f"{hp}layers.*.mlp.gate_proj.weight",
                    up=f"{hp}layers.*.mlp.up_proj.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)

    def _hf_layer_prefix(self) -> str:
        """Text-only CausalLM: weights at ``model.*``; override in VL subclass."""
        return "model."

    def _moe_mapping_registry(self) -> MegatronMappingRegistry:
        """Parameter mappings for the MoE variant."""
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.pre_feedforward_layernorm_2.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.post_shared_expert_layernorm.weight": (
                "model.layers.*.post_feedforward_layernorm_1.weight"
            ),
            "decoder.layers.*.mlp.router.weight": "model.layers.*.router.proj.weight",
        }

        mapping_list = [AutoMapping(megatron_param=m, hf_param=h) for m, h in param_mappings.items()]
        mapping_list.extend(
            [
                _Gemma4QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                FusedGatedExpertMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    hf_param="model.layers.*.experts.gate_up_proj",
                ),
                FusedExpertMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.experts.down_proj",
                ),
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.layer_scalar",
                    hf_param="model.layers.*.layer_scalar",
                ),
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.mlp.router.per_expert_scale",
                    hf_param="model.layers.*.router.per_expert_scale",
                ),
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.mlp.router.scale",
                    hf_param="model.layers.*.router.scale",
                ),
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.pre_shared_expert_layernorm.weight",
                    hf_param="model.layers.*.pre_feedforward_layernorm.weight",
                ),
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.mlp.post_moe_layernorm.weight",
                    hf_param="model.layers.*.post_feedforward_layernorm_2.weight",
                ),
                ReplicatedMapping(
                    megatron_param="decoder.layers.*.post_ffn_layernorm.weight",
                    hf_param="model.layers.*.post_feedforward_layernorm.weight",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)

    def _split_qkv_linear_out_weight(self, megatron_model, linear_out_weight):
        """Detect global vs sliding layers by tensor size for LoRA export."""
        model = megatron_model[0] if isinstance(megatron_model, list) else megatron_model
        config = model.config
        feature_dim = linear_out_weight.shape[-1] if linear_out_weight.ndim == 2 else None

        qkv_total_sliding = config.num_attention_heads + 2 * config.num_query_groups
        expected_numel_sliding = qkv_total_sliding * config.kv_channels * (feature_dim or 1)

        if linear_out_weight.numel() != expected_numel_sliding and hasattr(config, "global_head_dim"):
            num_kv_global = config.num_global_key_value_heads
            head_size_global = config.global_head_dim

            class _GlobalAttnCfg:
                num_attention_heads = config.num_attention_heads
                num_query_groups = num_kv_global
                kv_channels = head_size_global
                hidden_size = config.hidden_size
                attention_output_gate = getattr(config, "attention_output_gate", False)

            q_out, k_out, _ = split_qkv_weights(_GlobalAttnCfg(), linear_out_weight, feature_dim=feature_dim)
            return {"q_proj": q_out, "k_proj": k_out, "v_proj": ABSENT_PROJECTION}

        return super()._split_qkv_linear_out_weight(megatron_model, linear_out_weight)
