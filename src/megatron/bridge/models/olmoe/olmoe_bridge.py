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

import logging
from typing import Any

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import OlmoeForCausalLM, PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.olmoe.olmoe_provider import olmoe_layer_spec


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source=OlmoeForCausalLM, target=GPTModel, model_type="olmoe")
class OlMoEBridge(MegatronModelBridge):
    """
    Megatron Bridge for OlMoE Models.

    This bridge handles the conversion between HuggingFace OlMoEForCausalLM
    and Megatron-Core GPTModel formats. OlMoE models use mixture of experts
    architecture with QK layernorm.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("allenai/OLMoE-1B-7B-0125")
        >>> model_config = bridge.get_model_config()
    """

    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face OlMoE config to builder config kwargs.

        OlMoE uses QK layernorm and mixture of experts.
        """
        if hasattr(hf_config, "scoring_func") and hf_config.scoring_func != "softmax":
            raise ValueError(f"OlMoE only supports scoring_func='softmax', got {hf_config.scoring_func!r}")

        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(
            # OlMoE uses custom layer spec with OLMoESelfAttention for QK layernorm
            transformer_layer_spec=olmoe_layer_spec,
            # OLMoE HF config doesn't have head_dim, so calculate it
            kv_channels=getattr(hf_config, "head_dim", None)
            or (hf_config.hidden_size // hf_config.num_attention_heads),
            # OlMoE-specific architecture settings
            normalization="RMSNorm",
            gated_linear_unit=True,
            add_bias_linear=False,
            hidden_dropout=0.0,
            share_embeddings_and_output_weights=False,
            qk_layernorm=True,
            persist_layer_norm=True,
            autocast_dtype=torch.bfloat16,
            masked_softmax_fusion=True,
            rope_scaling=False,
            rope_scaling_factor=1.0,
            # MoE-specific settings
            moe_ffn_hidden_size=hf_config.intermediate_size,
            moe_aux_loss_coeff=hf_config.router_aux_loss_coef,
            moe_token_dispatcher_type="alltoall",
            moe_router_load_balancing_type="seq_aux_loss",
            moe_router_pre_softmax=True,
            moe_grouped_gemm=True,
            moe_router_score_function="softmax",
            moe_permute_fusion=True,
            moe_router_dtype="fp32",
        )
        config_kwargs.setdefault("position_embedding_type", "rope")
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated provider path."""
        return self.hf_config_to_model_config_kwargs(hf_config)

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []

        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Attention
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
