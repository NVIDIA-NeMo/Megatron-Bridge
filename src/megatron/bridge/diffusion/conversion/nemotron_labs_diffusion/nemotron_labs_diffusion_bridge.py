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

"""
Megatron Bridge for NemotronLabsDiffusion diffusion language models.

Converts between HuggingFace and Megatron-Core GPTModel format, using
NemotronLabsDiffusionModelProvider which replaces core attention with
NemotronLabsDiffusionAttention for sbd_block_diff.

Supports two HF checkpoint formats (auto-detected from config):
- Text-only (NemotronLabsDiffusion): encoder.*, diffusion_head.weight
- VLM source (Ministral CPT): language_model.model.*, language_model.lm_head.weight
  (vision_tower and multi_modal_projector weights are ignored)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.diffusion.models.nemotron_labs_diffusion.model_config import (
    NemotronLabsDiffusionModelConfig,
)
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, register_bridge_implementation
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


if TYPE_CHECKING:
    from megatron.bridge.diffusion.models.nemotron_labs_diffusion.nemotron_labs_diffusion_provider import (
        NemotronLabsDiffusionModelProvider,
    )


class NemotronLabsDiffusionBridge(MegatronModelBridge):
    """HF <-> Megatron bridge for NemotronLabsDiffusion diffusion language models.

    Handles both text-only (encoder.*) and VLM (language_model.model.*) HF formats.
    The format is auto-detected in provider_bridge() and used in mapping_registry().

    The Megatron target is a bare GPTModel (not wrapped in Ministral3Model), so
    Megatron-side keys use embedding.*, decoder.*, output_layer.* (no language_model. prefix).
    """

    _is_text_only: bool = True
    MODEL_CONFIG_CLASS = NemotronLabsDiffusionModelConfig

    def hf_config_to_model_config_kwargs(self, hf_config: Any) -> dict[str, Any]:
        """Map Nemotron Labs Diffusion HF settings to pure GPT config fields."""
        text_config = getattr(hf_config, "text_config", hf_config)
        self._is_text_only = not hasattr(hf_config, "text_config")
        return {
            "hidden_size": text_config.hidden_size,
            "ffn_hidden_size": text_config.intermediate_size,
            "num_layers": text_config.num_hidden_layers,
            "num_attention_heads": text_config.num_attention_heads,
            "num_query_groups": getattr(text_config, "num_key_value_heads", text_config.num_attention_heads),
            "kv_channels": getattr(
                text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads
            ),
            "vocab_size": text_config.vocab_size,
            "seq_length": text_config.max_position_embeddings,
            "layernorm_epsilon": getattr(text_config, "rms_norm_eps", 1e-5),
            "share_embeddings_and_output_weights": getattr(
                hf_config,
                "tie_word_embeddings",
                getattr(text_config, "tie_word_embeddings", False),
            ),
            "rotary_base": text_config.rope_parameters["rope_theta"],
            "position_embedding_type": "none",
            "normalization": "RMSNorm",
            "gated_linear_unit": True,
            "activation_func": F.silu,
            "add_bias_linear": False,
            "add_qkv_bias": False,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "params_dtype": torch.bfloat16,
            "autocast_dtype": torch.bfloat16,
            "bf16": True,
            "fp16": False,
            "scatter_embedding_sequence_parallel": False,
            "hf_config": hf_config.to_dict(),
        }

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> "NemotronLabsDiffusionModelProvider":
        # Imported lazily: the provider pulls in NemotronLabsDiffusionAttention ->
        # torch.nn.attention.flex_attention. Keeping it out of module scope means
        # registering this bridge at `import megatron.bridge` stays cheap and doesn't
        # drag flex_attention into the base import path. The heavy import happens here,
        # on the first actual conversion.
        from megatron.bridge.diffusion.models.nemotron_labs_diffusion.nemotron_labs_diffusion_provider import (
            NemotronLabsDiffusionModelProvider,
        )

        hf_config = hf_pretrained.config
        text_config = getattr(hf_config, "text_config", hf_config)

        # Auto-detect checkpoint format: VLM configs nest text params under text_config
        self._is_text_only = not hasattr(hf_config, "text_config")

        # NemotronLabsDiffusionConfig (a trust_remote_code config) does not declare
        # model-specific fields as dataclass fields.  In transformers 5.x
        # PretrainedConfig is a dataclass, so MLM's _convert_value_to_dict uses the
        # dataclass-fields path and silently drops all model-specific attributes
        # (hidden_size, rope_parameters, etc.).  Adding to_cfg_dict to the class
        # makes the serializer use PretrainedConfig.to_dict() which captures everything.
        cfg_cls = type(hf_config)
        if not hasattr(cfg_cls, "to_cfg_dict") and hasattr(hf_config, "to_dict"):

            def _to_cfg_dict(self):
                cls = self.__class__
                return {
                    "_target_": f"{cls.__module__}.{cls.__qualname__}.from_dict",
                    "_call_": True,
                    "config_dict": self.to_dict(),
                }

            cfg_cls.to_cfg_dict = _to_cfg_dict

        return NemotronLabsDiffusionModelProvider(
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_layers=text_config.num_hidden_layers,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", False),
            rotary_base=text_config.rope_parameters["rope_theta"],
            vocab_size=text_config.vocab_size,
            hf_config=hf_config,
        )

    def _text_only_mappings(self) -> list:
        """Mappings for text-only NemotronLabsDiffusion checkpoints (encoder.*, diffusion_head.weight)."""
        param_mappings = {
            "embedding.word_embeddings.weight": "encoder.embed_tokens.weight",
            "output_layer.weight": "diffusion_head.weight",
            "decoder.final_layernorm.weight": "encoder.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "encoder.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "encoder.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "encoder.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "encoder.layers.*.mlp.down_proj.weight",
        }
        mapping_list = [AutoMapping(megatron_param=k, hf_param=v) for k, v in param_mappings.items()]
        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="encoder.layers.*.self_attn.q_proj.weight",
                    k="encoder.layers.*.self_attn.k_proj.weight",
                    v="encoder.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="encoder.layers.*.mlp.gate_proj.weight",
                    up="encoder.layers.*.mlp.up_proj.weight",
                ),
            ]
        )
        return mapping_list

    def _vlm_mappings(self) -> list:
        """Mappings for VLM Ministral CPT source checkpoints (language_model.model.*).

        Vision keys (vision_tower.**, multi_modal_projector.**) are absent from
        the Megatron GPTModel side and are naturally ignored.
        """
        param_mappings = {
            "embedding.word_embeddings.weight": "language_model.model.embed_tokens.weight",
            "output_layer.weight": "language_model.lm_head.weight",
            "decoder.final_layernorm.weight": "language_model.model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "language_model.model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "language_model.model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "language_model.model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "language_model.model.layers.*.mlp.down_proj.weight",
        }
        mapping_list = [AutoMapping(megatron_param=k, hf_param=v) for k, v in param_mappings.items()]
        mapping_list.extend(
            [
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="language_model.model.layers.*.self_attn.q_proj.weight",
                    k="language_model.model.layers.*.self_attn.k_proj.weight",
                    v="language_model.model.layers.*.self_attn.v_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="language_model.model.layers.*.mlp.gate_proj.weight",
                    up="language_model.model.layers.*.mlp.up_proj.weight",
                ),
            ]
        )
        return mapping_list

    def mapping_registry(self) -> MegatronMappingRegistry:
        if self._is_text_only:
            mapping_list = self._text_only_mappings()
        else:
            mapping_list = self._vlm_mappings()
        return MegatronMappingRegistry(*mapping_list)


# Register for the custom HF architecture (available via auto_map, not a standard transformers class)
register_bridge_implementation(
    source="NemotronLabsDiffusionModel",
    target=GPTModel,
    bridge_class=NemotronLabsDiffusionBridge,
)
