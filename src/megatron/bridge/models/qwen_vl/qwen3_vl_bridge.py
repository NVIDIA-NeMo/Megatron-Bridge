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
import re
from typing import Dict, Union

import torch
import torch.nn as nn
from transformers import Qwen3VLMoeForConditionalGeneration

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    QKVMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.hf_pretrained.vlm import PreTrainedVLM
from megatron.bridge.models.qwen_vl import Qwen3VLModel
from megatron.bridge.models.qwen_vl.qwen_vl_provider import Qwen3VLMoEModelProvider


@MegatronModelBridge.register_bridge(source=Qwen3VLMoeForConditionalGeneration, target=Qwen3VLModel)
class Qwen3VLBridge(MegatronModelBridge):
    """
    Megatron Bridge for Qwen3-VL MoE Conditional Generation.

    Bridges HF Qwen3VLMoeForConditionalGeneration <-> Megatron-Core GPT language model inside
    a Qwen25VL-style wrapper module for VL composition.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedVLM) -> Qwen3VLMoEModelProvider:
        cfg = hf_pretrained.config

        # The HF VL config nests text_config and vision_config. Read text_config for LM.
        text_cfg = getattr(cfg, "text_config", cfg)

        provider = Qwen3VLMoEModelProvider(
            # Language model (text) params
            num_layers=text_cfg.num_hidden_layers,
            hidden_size=text_cfg.hidden_size,
            ffn_hidden_size=text_cfg.intermediate_size,
            moe_ffn_hidden_size=getattr(text_cfg, "moe_intermediate_size", None),
            num_attention_heads=text_cfg.num_attention_heads,
            kv_channels=getattr(text_cfg, "head_dim", text_cfg.hidden_size // text_cfg.num_attention_heads),
            num_query_groups=text_cfg.num_key_value_heads,
            num_moe_experts=getattr(text_cfg, "num_experts", None),
            moe_router_topk=getattr(text_cfg, "num_experts_per_tok", None),
            init_method_std=text_cfg.initializer_range,
            layernorm_epsilon=text_cfg.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_cfg.vocab_size),
            rotary_base=text_cfg.rope_theta,
            share_embeddings_and_output_weights=getattr(cfg, "tie_word_embeddings", False),
            vocab_size=text_cfg.vocab_size,
            seq_length=text_cfg.max_position_embeddings,
            # Dtypes
            bf16=(self.dtype_from_hf(text_cfg, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(text_cfg, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
            # Qwen3 MoE specifics
            qk_layernorm=True,
            moe_grouped_gemm=True,
            # Vision config and tokens
            vision_config=cfg.vision_config,
            bos_token_id=getattr(cfg, "bos_token_id", 151643),
            eos_token_id=getattr(cfg, "eos_token_id", 151645),
            vision_start_token_id=getattr(cfg, "vision_start_token_id", 151652),
            vision_end_token_id=getattr(cfg, "vision_end_token_id", 151653),
            vision_token_id=getattr(cfg, "vision_token_id", 151654),
            image_token_id=getattr(cfg, "image_token_id", 151655),
            video_token_id=getattr(cfg, "video_token_id", 151656),
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        # VL wrapper carries language model under `language_model` and visual under `visual`.
        param_mappings = {
            # Embeddings / head / final norm
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            "language_model.decoder.layers.*.mlp.router.weight": "model.language_model.layers.*.mlp.gate.weight",
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                # Visual stack is passed through 1:1
                ReplicatedMapping(
                    megatron_param="visual.**",
                    hf_param="model.visual.**",
                ),
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                ExpertMLPGateUpProjMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    hf_param="model.language_model.layers.*.mlp.experts.gate_up_proj",
                ),
                ExpertMLPDownProjMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.language_model.layers.*.mlp.experts.down_proj",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)


class ExpertMLPDownProjMapping(AutoMapping):
    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        return super().hf_to_megatron(hf_weights[global_expert_number].transpose(0, 1), megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        if megatron_weights is None:
            return super().megatron_to_hf(megatron_weights, megatron_module)

        return super().megatron_to_hf(megatron_weights.transpose(0, 1).contiguous(), megatron_module)

    def _validate_patterns(self, *args, **kwargs):
        # allow number of wildcards to mismatch in this mapping
        pass


class ExpertMLPGateUpProjMapping(AutoMapping):
    def hf_to_megatron(self, hf_weights: Union[torch.Tensor, Dict], megatron_module: nn.Module) -> torch.Tensor:
        global_expert_number = extract_expert_number_from_param(self.megatron_param)
        return super().hf_to_megatron(hf_weights[global_expert_number].transpose(0, 1), megatron_module)

    def megatron_to_hf(self, megatron_weights: torch.Tensor, megatron_module: nn.Module) -> Dict[str, torch.Tensor]:
        if megatron_weights is None:
            return super().megatron_to_hf(megatron_weights, megatron_module)

        return super().megatron_to_hf(megatron_weights.transpose(0, 1).contiguous(), megatron_module)

    def _validate_patterns(self, *args, **kwargs):
        # allow number of wildcards to mismatch in this mapping
        pass


def extract_expert_number_from_param(param_name: str) -> int:
    """Extract the expert number from a parameter name.

    Args:
        param_name: The parameter name to extract the expert number from.

    Returns:
        The expert number.

    """
    pattern = r"(?:experts\.|weight|bias)(\d+)"
    match = re.search(pattern, param_name)
    if not match:
        raise ValueError(
            f"No expert number found in parameter name: {param_name}. Please update the regex {pattern} if necessary."
        )
    return int(match.group(1))
