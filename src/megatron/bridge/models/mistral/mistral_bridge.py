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

from typing import Any

import torch.nn.functional as F
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import MistralForCausalLM, PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.conversion.utils import mcore_to_hf_window_size
from megatron.bridge.models.gpt.model_config import BridgeGPTModelConfig
from megatron.bridge.models.mistral.mistral_provider import MistralModelProvider


@MegatronModelBridge.register_bridge(source=MistralForCausalLM, target=GPTModel, provider=MistralModelProvider)
class MistralBridge(MegatronModelBridge):
    """
    Megatron Bridge for Mistral Causal LM.

    As a user you would not use this bridge directly, but through `AutoBridge`.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("mistralai/Ministral-8B-Instruct-2410")
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = BridgeGPTModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face Mistral config to builder config kwargs.

        YaRN-scaled checkpoints are handled by the shared mapping, which sets
        ``position_embedding_type="yarn"`` and the ``yarn_*`` fields.
        """
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)

        window_size, cp_comm_type = (None, None)
        if getattr(hf_config, "sliding_window", None) is not None:
            window_size = [hf_config.sliding_window - 1, 0]
            cp_comm_type = "a2a"

        config_kwargs.update(
            normalization="RMSNorm",
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            rotary_base=rope_theta_from_hf(hf_config),
            window_size=window_size,
            cp_comm_type=cp_comm_type,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            kv_channels=getattr(hf_config, "head_dim", None),
            masked_softmax_fusion=True,
            rope_scaling=False,
            rope_scaling_factor=1.0,
        )
        config_kwargs.setdefault("position_embedding_type", "rope")
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated provider path."""
        return self.hf_config_to_model_config_kwargs(hf_config)

    @classmethod
    def megatron_to_hf_config(cls, provider: MistralModelProvider) -> dict:
        """Convert a Mistral provider config back to Hugging Face format."""
        hf_config = super().megatron_to_hf_config(provider)
        hf_config["sliding_window"] = mcore_to_hf_window_size(provider.window_size)
        return hf_config

    def mapping_registry(self) -> MegatronMappingRegistry:
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
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
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
