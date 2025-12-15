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

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Gemma2ForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gemma.gemma2_provider import Gemma2ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# Register custom Gemma2 modules for AutoMapping
AutoMapping.register_module_type("TERowParallelLinearLayerNorm", "row")
AutoMapping.register_module_type("Gemma2OutputLayer", "column")


@MegatronModelBridge.register_bridge(source=Gemma2ForCausalLM, target=GPTModel)
class Gemma2Bridge(MegatronModelBridge):
    """
    Megatron Bridge for Gemma2 Causal LM.
    This bridge handles the conversion between HuggingFace Gemma2ForCausalLM
    and Megatron-Core GPTModel formats, including weight mappings and
    configuration translation. Gemma2 includes specific features like
    attention logit softcapping, sliding window attention, and additional
    layer normalization compared to the original Gemma model.
    As a user you would not use this bridge directly, but through `AutoBridge`.
    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("google/gemma-2-2b")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Gemma2ModelProvider:
        """Convert HuggingFace config to Gemma2ModelProvider.
        Args:
            hf_pretrained: HuggingFace pretrained model wrapper
        Returns:
            Gemma2ModelProvider: Configured provider for Megatron model
        """
        hf_config = hf_pretrained.config

        provider = Gemma2ModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.rms_norm_eps,
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=hf_config.head_dim,
            rotary_base=hf_config.rope_theta,
            query_pre_attn_scalar=hf_config.query_pre_attn_scalar,
            attn_logit_softcapping=hf_config.attn_logit_softcapping,
            final_logit_softcapping=hf_config.final_logit_softcapping,
            window_size=(hf_config.sliding_window, 0),
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            vocab_size=hf_config.vocab_size,
            share_embeddings_and_output_weights=True,
            seq_length=hf_config.max_position_embeddings,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Return MegatronMappingRegistry containing parameter mappings from HF to Megatron format.
        Returns:
            MegatronMappingRegistry: Registry of parameter mappings
        """
        # Dictionary maps HF parameter names -> Megatron parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "model.embed_tokens.weight": "embedding.word_embeddings.weight",
            "model.layers.*.self_attn.o_proj.weight": "decoder.layers.*.self_attention.linear_proj.weight",
            "model.layers.*.mlp.down_proj.weight": "decoder.layers.*.mlp.linear_fc2.weight",
            "model.layers.*.input_layernorm.weight": "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight",
            "model.layers.*.pre_feedforward_layernorm.weight": "decoder.layers.*.mlp.linear_fc1.layer_norm_weight",
            "model.layers.*.post_feedforward_layernorm.weight": "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight",
            "model.layers.*.post_attention_layernorm.weight": "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight",
            "model.norm.weight": "decoder.final_layernorm.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(hf_param, megatron_param)
        for hf_param, megatron_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    q="model.layers.*.self_attn.q_proj.weight",
                    k="model.layers.*.self_attn.k_proj.weight",
                    v="model.layers.*.self_attn.v_proj.weight",
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
