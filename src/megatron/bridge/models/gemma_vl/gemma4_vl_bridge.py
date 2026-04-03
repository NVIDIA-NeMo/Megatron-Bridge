# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Megatron Bridge for Gemma 4 VL (Vision-Language).

Extends the Gemma 4 text bridge to handle the full VLM checkpoint with
vision tower, multimodal embedder, and language model.

Weight prefixes in HF VLM checkpoint (after stripping outer ``model.``):
- ``language_model.layers.*``     → language model decoder layers
- ``language_model.embed_tokens`` → language model embedding
- ``language_model.norm``         → final layernorm
- ``vision_tower.*``              → HF vision encoder (replicated)
- ``embed_vision.*``              → multimodal projector (replicated)
"""

import re
from typing import Mapping

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import (
    rope_local_base_freq_from_hf,
    rope_theta_from_hf,
)
from megatron.bridge.models.gemma.gemma4_bridge import _Gemma4QKVMapping, _infer_attn_pattern
from megatron.bridge.models.gemma.gemma4_provider import Gemma4ModelProvider
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import Gemma4VLModelProvider
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM


@MegatronModelBridge.register_bridge(
    source="Gemma4ForConditionalGeneration",
    target=Gemma4VLModel,
    provider=Gemma4VLModelProvider,
    model_type="gemma4_vl",
)
class Gemma4VLBridge(MegatronModelBridge):
    """Megatron Bridge for Gemma 4 Vision-Language models.

    Handles conversion between HuggingFace Gemma4ForConditionalGeneration and
    Megatron-Core Gemma4VLModel.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("google/gemma-4-26B-A4B")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Gemma4VLModelProvider:
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config

        # Use base class helper for common config conversion from text_config
        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider = Gemma4VLModelProvider(**provider_kwargs)

        # === Gemma 4 text-specific features (same as Gemma4Bridge) ===
        provider.window_size = getattr(text_config, "sliding_window", 1024)

        # Dual RoPE bases
        provider.rotary_base = (
            rope_local_base_freq_from_hf(text_config),
            rope_theta_from_hf(text_config),
        )

        # QK norm
        head_dim = getattr(text_config, "head_dim", 256)
        provider.softmax_scale = 1.0
        provider.kv_channels = head_dim
        provider.qk_layernorm = True

        # Global attention overrides
        provider.global_head_dim = getattr(text_config, "global_head_dim", 512)
        provider.num_global_key_value_heads = getattr(text_config, "num_global_key_value_heads", 2)

        # Parse partial_rotary_factor
        rope_params = getattr(text_config, "rope_parameters", {})
        if isinstance(rope_params, dict):
            full_attn_rope = rope_params.get("full_attention", {})
            provider.global_rotary_percent = full_attn_rope.get("partial_rotary_factor", 0.25)

        # Sliding/global layer pattern
        layer_types = getattr(text_config, "layer_types", None)
        if layer_types:
            provider.interleaved_attn_pattern = _infer_attn_pattern(layer_types)

        # MoE configuration
        provider.num_moe_experts = getattr(text_config, "num_experts", 128)
        provider.moe_router_topk = getattr(text_config, "top_k_experts", 8)
        provider.moe_ffn_hidden_size = getattr(text_config, "moe_intermediate_size", 704)
        provider.moe_shared_expert_intermediate_size = getattr(text_config, "intermediate_size", 2112)
        provider.moe_shared_expert_overlap = False
        provider.moe_shared_expert_gate = False
        provider.moe_layer_freq = 1

        # Logit softcapping
        provider.final_logit_softcapping = getattr(text_config, "final_logit_softcapping", 30.0)

        # Override dtype and vocab settings
        provider.bf16 = True
        provider.params_dtype = torch.bfloat16
        provider.autocast_dtype = torch.bfloat16
        provider.make_vocab_size_divisible_by = 128

        # === VL-specific config ===
        provider.vision_config = vision_config
        provider.text_config = text_config
        provider.vision_soft_tokens_per_image = getattr(hf_config, "vision_soft_tokens_per_image", 280)

        # Token IDs
        provider.bos_token_id = getattr(hf_config, "bos_token_id", 2)
        provider.eos_token_id = getattr(hf_config, "eos_token_id", 1)
        provider.image_token_id = getattr(hf_config, "image_token_id", 258_880)
        provider.video_token_id = getattr(hf_config, "video_token_id", 258_884)

        return provider

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Handle special weight loading for Gemma 4 VLM.

        Same as Gemma4Bridge: K=V synthesis, router weight fusion, shared expert
        pre-norm fusion. HF param names have ``model.language_model.`` prefix
        (raw safetensors keys include the outer ``model.`` from
        Gemma4ForConditionalGeneration).
        """
        # Handle K=V on global layers
        if isinstance(hf_param, dict) and "v" in hf_param:
            v_name = hf_param["v"]
            if v_name not in hf_state_dict:
                k_name = hf_param["k"]
                hf_weights = {}
                for role, name in hf_param.items():
                    if role == "v":
                        hf_weights[role] = hf_state_dict[k_name].clone()
                    else:
                        hf_weights[role] = hf_state_dict[name]
                return hf_weights

        # Fuse pre-norm correction into shared expert gate/up weights
        if isinstance(hf_param, dict) and "gate" in hf_param:
            gate_name = hf_param["gate"]
            if "mlp.gate_proj" in gate_name:
                return self._fuse_shared_expert_prenorm(hf_param, hf_state_dict)

        # Fuse router scaling into router.proj.weight
        if isinstance(hf_param, str) and hf_param.endswith("router.proj.weight"):
            return self._fuse_router_weight(hf_param, hf_state_dict)

        return super().maybe_modify_loaded_hf_weight(hf_param, hf_state_dict)

    def _fuse_router_weight(
        self, hf_param: str, hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        """Fuse router preprocessing into projection weight (VLM version)."""
        proj_weight = hf_state_dict[hf_param]

        layer_match = re.search(r"layers\.(\d+)\.", hf_param)
        if layer_match is None:
            return proj_weight
        layer_idx = layer_match.group(1)

        # VLM prefix: language_model.layers.{idx}.router.*
        prefix = hf_param.rsplit("layers.", 1)[0]
        scale_key = f"{prefix}layers.{layer_idx}.router.scale"
        ln2_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm_2.weight"

        if scale_key not in hf_state_dict or ln2_key not in hf_state_dict:
            return proj_weight

        router_scale = hf_state_dict[scale_key].float()
        ln2_weight = hf_state_dict[ln2_key].float()
        hidden_size = proj_weight.shape[-1]
        scalar_root_size = hidden_size ** -0.5

        fusion_factor = router_scale * scalar_root_size / ln2_weight
        fused_weight = proj_weight.float() * fusion_factor.unsqueeze(0)
        return fused_weight.to(proj_weight.dtype)

    def _fuse_shared_expert_prenorm(
        self, hf_param: dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Fuse pre-norm correction into shared expert gate/up weights (VLM version)."""
        gate_name = hf_param["gate"]
        layer_match = re.search(r"layers\.(\d+)\.", gate_name)
        if layer_match is None:
            return {role: hf_state_dict[name] for role, name in hf_param.items()}

        layer_idx = layer_match.group(1)
        prefix = gate_name.rsplit("layers.", 1)[0]
        pffl_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm.weight"
        pffl2_key = f"{prefix}layers.{layer_idx}.pre_feedforward_layernorm_2.weight"

        if pffl_key not in hf_state_dict or pffl2_key not in hf_state_dict:
            return {role: hf_state_dict[name] for role, name in hf_param.items()}

        w_pffl = hf_state_dict[pffl_key].float()
        w_pffl2 = hf_state_dict[pffl2_key].float()
        correction = w_pffl / w_pffl2

        hf_weights = {}
        for role, name in hf_param.items():
            weight = hf_state_dict[name]
            fused = weight.float() * correction.unsqueeze(0)
            hf_weights[role] = fused.to(weight.dtype)
        return hf_weights

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Define parameter mappings for Gemma 4 VLM.

        HF VLM param names (raw safetensors keys include outer ``model.`` prefix):
        - ``model.language_model.layers.*`` → language model
        - ``model.vision_tower.*`` → vision encoder (replicated)
        - ``model.embed_vision.*`` → multimodal projector (replicated)
        """
        # Language model parameter mappings
        # HF safetensors: model.language_model.{layers/embed_tokens/norm}.*
        # MG: language_model.{decoder.layers/embedding/decoder.final_layernorm}.*
        param_mappings = {
            # === Embeddings ===
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",

            # === Per-layer attention ===
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
                "model.language_model.layers.*.input_layernorm.weight"
            ),
            "language_model.decoder.layers.*.input_layernorm.weight": (
                "model.language_model.layers.*.input_layernorm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": (
                "model.language_model.layers.*.self_attn.q_norm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": (
                "model.language_model.layers.*.self_attn.k_norm.weight"
            ),
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": (
                "model.language_model.layers.*.self_attn.o_proj.weight"
            ),
            "language_model.decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.language_model.layers.*.post_attention_layernorm.weight"
            ),

            # === Pre-MLP layernorm (maps to MoE pre-norm) ===
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": (
                "model.language_model.layers.*.pre_feedforward_layernorm_2.weight"
            ),

            # === Dense MLP → Shared Expert ===
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": (
                "model.language_model.layers.*.mlp.down_proj.weight"
            ),
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.post_layernorm.weight": (
                "model.language_model.layers.*.post_feedforward_layernorm_1.weight"
            ),

            # === MoE Router ===
            "language_model.decoder.layers.*.mlp.router.weight": (
                "model.language_model.layers.*.router.proj.weight"
            ),
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend([
            # === Vision tower (replicated — all weights pass through) ===
            ReplicatedMapping(
                megatron_param="vision_tower.**",
                hf_param="model.vision_tower.**",
            ),

            # === Multimodal embedder (replicated) ===
            ReplicatedMapping(
                megatron_param="embed_vision.**",
                hf_param="model.embed_vision.**",
            ),

            # === QKV: K=V tolerant mapping ===
            _Gemma4QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.language_model.layers.*.self_attn.q_proj.weight",
                k="model.language_model.layers.*.self_attn.k_proj.weight",
                v="model.language_model.layers.*.self_attn.v_proj.weight",
            ),

            # === Dense MLP → Shared Expert gated FC1 ===
            GatedMLPMapping(
                megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                gate="model.language_model.layers.*.mlp.gate_proj.weight",
                up="model.language_model.layers.*.mlp.up_proj.weight",
            ),

            # === MoE Experts (fused format) ===
            FusedGatedExpertMapping(
                megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                hf_param="model.language_model.layers.*.experts.gate_up_proj",
            ),
            FusedExpertMapping(
                megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                hf_param="model.language_model.layers.*.experts.down_proj",
            ),

            # === Per-layer output scaling (buffer) ===
            ReplicatedMapping(
                megatron_param="language_model.decoder.layers.*.layer_scalar",
                hf_param="model.language_model.layers.*.layer_scalar",
            ),

            # === Router per-expert scaling (buffer) ===
            ReplicatedMapping(
                megatron_param="language_model.decoder.layers.*.mlp.router.per_expert_scale",
                hf_param="model.language_model.layers.*.router.per_expert_scale",
            ),

            # === Post-MoE layernorm ===
            ReplicatedMapping(
                megatron_param="language_model.decoder.layers.*.mlp.post_moe_layernorm.weight",
                hf_param="model.language_model.layers.*.post_feedforward_layernorm_2.weight",
            ),

            # === Post-feedforward layernorm ===
            ReplicatedMapping(
                megatron_param="language_model.decoder.layers.*.post_ffn_layernorm.weight",
                hf_param="model.language_model.layers.*.post_feedforward_layernorm.weight",
            ),
        ])

        return MegatronMappingRegistry(*mapping_list)
