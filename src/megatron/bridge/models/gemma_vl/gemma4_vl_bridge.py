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

"""Megatron Bridge for Gemma 4 Vision-Language (ConditionalGeneration).

Text conversion logic is inherited from
:class:`~megatron.bridge.models.gemma.gemma4_bridge.Gemma4Bridge`.

Usage::

  AutoBridge.from_hf_pretrained("google/gemma-4-E4B-it")
    └─ Gemma4VLBridge (registered for Gemma4ForConditionalGeneration)
         ├─ provider_bridge()  text mode → Gemma4DenseProvider (pretraining)
         │                     auto/vl   → Gemma4DenseVLProvider (full VL)
         └─ mapping_registry()  Dense → _dense_vl_mapping_registry()
                                 MoE   → _moe_vl_mapping_registry()
"""

import os
from dataclasses import asdict, is_dataclass

import torch

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    FusedExpertMapping,
    FusedGatedExpertMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import (
    rope_local_base_freq_from_hf,
    rope_theta_from_hf,
)
from megatron.bridge.models.gemma.gemma4_bridge import (
    Gemma4Bridge,
    _Gemma4QKVMapping,
    _infer_attn_pattern,
)
from megatron.bridge.models.gemma.gemma4_provider import Gemma4DenseProvider
from megatron.bridge.models.gemma_vl.gemma4_vl_provider import (
    Gemma4DenseVLProvider,
    Gemma4VLModelProvider,
)
from megatron.bridge.models.gemma_vl.modeling_gemma4_vl import Gemma4VLModel
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# ---------------------------------------------------------------------------
# Gemma4VLBridge — VL ConditionalGeneration bridge, inherits Gemma4Bridge
# ---------------------------------------------------------------------------


@MegatronModelBridge.register_bridge(
    source="Gemma4ForConditionalGeneration",
    target=Gemma4VLModel,
    provider=Gemma4VLModelProvider,
    model_type="gemma4_vl",
)
class Gemma4VLBridge(Gemma4Bridge):
    """Megatron Bridge for Gemma 4 Vision-Language models.

    Inherits all Dense/MoE logic from Gemma4Bridge and adds VL-specific:
    - vision_tower and embed_vision weight mappings
    - Dense E4B VL provider construction
    - ``GEMMA4_CONVERSION_MODE`` dispatch (text / auto / vl)
    """

    def provider_bridge(
        self, hf_pretrained: PreTrainedCausalLM
    ) -> "Gemma4VLModelProvider | Gemma4DenseVLProvider | Gemma4DenseProvider":
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = hf_config.vision_config

        if not getattr(text_config, "enable_moe_block", False):
            self._is_dense = True
            if self._conversion_mode() == "text":
                return self._build_dense_provider(text_config)
            return self._build_dense_vl_provider(hf_config, text_config, vision_config)

        self._is_dense = False

        provider_kwargs = self.hf_config_to_provider_kwargs(text_config)
        provider = Gemma4VLModelProvider(**provider_kwargs)

        provider.window_size = getattr(text_config, "sliding_window", 1024)
        provider.rotary_base = (
            rope_local_base_freq_from_hf(text_config),
            rope_theta_from_hf(text_config),
        )

        head_dim = getattr(text_config, "head_dim", 256)
        provider.softmax_scale = 1.0
        provider.kv_channels = head_dim
        provider.qk_layernorm = True

        provider.global_head_dim = getattr(text_config, "global_head_dim", 512)
        provider.num_global_key_value_heads = getattr(text_config, "num_global_key_value_heads", 2)
        provider.attention_k_eq_v = getattr(text_config, "attention_k_eq_v", False)

        rope_params = getattr(text_config, "rope_parameters", {})
        if isinstance(rope_params, dict):
            full_attn_rope = rope_params.get("full_attention", {})
            provider.global_rotary_percent = full_attn_rope.get("partial_rotary_factor", 0.25)

        layer_types = getattr(text_config, "layer_types", None)
        if layer_types:
            provider.interleaved_attn_pattern = _infer_attn_pattern(layer_types)

        if getattr(text_config, "enable_moe_block", False):
            provider.num_moe_experts = getattr(text_config, "num_experts", None) or 128
            provider.moe_router_topk = getattr(text_config, "top_k_experts", None) or 8
            provider.moe_ffn_hidden_size = getattr(text_config, "moe_intermediate_size", None) or 704
            provider.moe_shared_expert_intermediate_size = getattr(text_config, "intermediate_size", 2112)
            provider.moe_shared_expert_overlap = False
            provider.moe_shared_expert_gate = False
            provider.moe_layer_freq = 1

        provider.final_logit_softcapping = getattr(text_config, "final_logit_softcapping", 30.0)
        # Keep the MoE VL path in fp32 for HF parity. The text-only MoE path
        # defaults to bf16, but VL conversion also runs HF vision/audio modules
        # whose precision-sensitive buffers are kept in fp32 by transformers.
        provider.bf16 = False
        provider.params_dtype = torch.float32
        provider.autocast_dtype = torch.float32
        provider.make_vocab_size_divisible_by = 128

        provider.vision_config = vision_config
        provider.text_config = text_config
        provider.audio_config = getattr(hf_config, "audio_config", None)
        provider.vision_soft_tokens_per_image = getattr(hf_config, "vision_soft_tokens_per_image", 280)
        provider.bos_token_id = getattr(hf_config, "bos_token_id", 2)
        provider.eos_token_id = getattr(hf_config, "eos_token_id", 1)
        provider.image_token_id = getattr(hf_config, "image_token_id", 258_880)
        provider.video_token_id = getattr(hf_config, "video_token_id", 258_884)
        provider.audio_token_id = getattr(hf_config, "audio_token_id", 258_881)

        return provider

    @classmethod
    def megatron_to_hf_config(
        cls,
        provider: Gemma4VLModelProvider | Gemma4DenseVLProvider | Gemma4DenseProvider,
    ) -> dict:
        """Convert a Gemma 4 VL provider config back to Hugging Face config."""
        text_config = super().megatron_to_hf_config(provider)
        text_config.pop("architectures", None)
        text_config["model_type"] = "gemma4_text"

        if not isinstance(provider, (Gemma4VLModelProvider, Gemma4DenseVLProvider)):
            text_config["architectures"] = ["Gemma4ForCausalLM"]
            return text_config

        def config_to_dict(config) -> dict | None:
            if config is None:
                return None
            if isinstance(config, dict):
                return dict(config)
            if hasattr(config, "to_dict"):
                return config.to_dict()
            if is_dataclass(config):
                return asdict(config)
            return {name: value for name, value in vars(config).items() if not name.startswith("_")}

        return {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "audio_config": config_to_dict(provider.audio_config),
            "audio_token_id": provider.audio_token_id,
            "bos_token_id": provider.bos_token_id,
            "dtype": text_config["dtype"],
            "eos_token_id": provider.eos_token_id,
            "image_token_id": provider.image_token_id,
            "model_type": "gemma4",
            "text_config": text_config,
            "tie_word_embeddings": provider.share_embeddings_and_output_weights,
            "video_token_id": provider.video_token_id,
            "vision_config": config_to_dict(provider.vision_config),
            "vision_soft_tokens_per_image": provider.vision_soft_tokens_per_image,
        }

    def _conversion_mode(self) -> str:
        mode = getattr(self, "gemma4_conversion_mode", None) or os.environ.get("GEMMA4_CONVERSION_MODE", "auto")
        mode = mode.lower()
        if mode not in {"auto", "text", "vl", "audio"}:
            raise ValueError(f"Invalid GEMMA4_CONVERSION_MODE={mode!r}; expected auto, text, vl, or audio.")
        # "audio" is treated as full VL+audio conversion (same as "vl"/"auto")
        return mode

    def _build_dense_vl_provider(self, hf_config, text_config, vision_config) -> Gemma4DenseVLProvider:
        """Build a Dense VL provider by copying all Dense provider fields."""
        from dataclasses import fields

        text_provider = self._build_dense_provider(text_config)
        provider = Gemma4DenseVLProvider()
        for f in fields(Gemma4DenseProvider):
            setattr(provider, f.name, getattr(text_provider, f.name))

        provider.vision_config = vision_config
        provider.text_config = text_config
        provider.audio_config = getattr(hf_config, "audio_config", None)
        provider.vision_soft_tokens_per_image = getattr(hf_config, "vision_soft_tokens_per_image", 280)
        provider.bos_token_id = getattr(hf_config, "bos_token_id", 2)
        provider.eos_token_id = getattr(hf_config, "eos_token_id", 1)
        provider.image_token_id = getattr(hf_config, "image_token_id", 258_880)
        provider.video_token_id = getattr(hf_config, "video_token_id", 258_884)
        provider.audio_token_id = getattr(hf_config, "audio_token_id", 258_881)
        return provider

    def _text_config(self):
        hf_config = getattr(self, "hf_config", None)
        return getattr(hf_config, "text_config", None)

    def _hf_layer_prefix(self) -> str:
        """VLM text weights live under ``model.language_model.*``."""
        return "model.language_model."

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Dispatch to Dense or MoE VLM mappings."""
        if self._is_dense_config():
            if self._conversion_mode() == "text":
                return self._dense_mapping_registry(megatron_prefix="")
            return self._dense_vl_mapping_registry()
        return self._moe_vl_mapping_registry()

    def _dense_vl_mapping_registry(self) -> MegatronMappingRegistry:
        """Dense E4B VL: language mappings + vision tower + audio tower."""
        registry = self._dense_mapping_registry(megatron_prefix="language_model.")
        mapping_list = list(registry.mappings)
        mapping_list.extend(
            [
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param="model.vision_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="embed_vision.**",
                    hf_param="model.embed_vision.**",
                ),
                ReplicatedMapping(
                    megatron_param="audio_tower.**",
                    hf_param="model.audio_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="embed_audio.**",
                    hf_param="model.embed_audio.**",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)

    def _moe_vl_mapping_registry(self) -> MegatronMappingRegistry:
        """MoE VL parameter mappings."""
        param_mappings = {
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": (
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
            "language_model.decoder.layers.*.post_ffn_layernorm.weight": (
                "model.language_model.layers.*.post_feedforward_layernorm.weight"
            ),
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": (
                "model.language_model.layers.*.pre_feedforward_layernorm_2.weight"
            ),
            "language_model.decoder.layers.*.mlp.shared_experts.linear_fc2.weight": (
                "model.language_model.layers.*.mlp.down_proj.weight"
            ),
            "language_model.decoder.layers.*.mlp.post_shared_expert_layernorm.weight": (
                "model.language_model.layers.*.post_feedforward_layernorm_1.weight"
            ),
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.router.proj.weight",
        }

        mapping_list = [AutoMapping(megatron_param=m, hf_param=h) for m, h in param_mappings.items()]
        mapping_list.append(
            _Gemma4QKVMapping(
                megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.language_model.layers.*.self_attn.q_proj.weight",
                k="model.language_model.layers.*.self_attn.k_proj.weight",
                v="model.language_model.layers.*.self_attn.v_proj.weight",
            )
        )
        mapping_list.extend(
            [
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                FusedGatedExpertMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    hf_param="model.language_model.layers.*.experts.gate_up_proj",
                ),
                FusedExpertMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.language_model.layers.*.experts.down_proj",
                ),
                ReplicatedMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.router.per_expert_scale",
                    hf_param="model.language_model.layers.*.router.per_expert_scale",
                ),
                ReplicatedMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.router.scale",
                    hf_param="model.language_model.layers.*.router.scale",
                ),
                ReplicatedMapping(
                    megatron_param="language_model.decoder.layers.*.pre_shared_expert_layernorm.weight",
                    hf_param="model.language_model.layers.*.pre_feedforward_layernorm.weight",
                ),
                ReplicatedMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.post_moe_layernorm.weight",
                    hf_param="model.language_model.layers.*.post_feedforward_layernorm_2.weight",
                ),
                ReplicatedMapping(
                    megatron_param="vision_tower.**",
                    hf_param="model.vision_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="embed_vision.**",
                    hf_param="model.embed_vision.**",
                ),
                ReplicatedMapping(
                    megatron_param="audio_tower.**",
                    hf_param="model.audio_tower.**",
                ),
                ReplicatedMapping(
                    megatron_param="embed_audio.**",
                    hf_param="model.embed_audio.**",
                ),
                ReplicatedMapping(
                    megatron_param="language_model.decoder.layers.*.layer_scalar",
                    hf_param="model.language_model.layers.*.layer_scalar",
                ),
            ]
        )
        return MegatronMappingRegistry(*mapping_list)
