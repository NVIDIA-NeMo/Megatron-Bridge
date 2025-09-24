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

import torch
from megatron.core.models.mamba import MambaModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, QKVMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.nemotronh.nemotron_h_provider import NemotronHModelProvider


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source="NemotronHForCausalLM", target=MambaModel)
class NemotronHBridge(MegatronModelBridge):
    """
    Megatron Bridge for Nemotron-H Causal LM.

    This bridge handles the conversion between HuggingFace NemotronHForCausalLM
    and Megatron-Core MambaModel formats, including weight mappings and
    configuration translation.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True)
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> NemotronHModelProvider:
        hf_config = hf_pretrained.config

        return NemotronHModelProvider(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            add_bias_linear=hf_config.use_bias,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            init_method_std=hf_config.initializer_range,
            layernorm_epsilon=hf_config.layer_norm_epsilon,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(hf_config.vocab_size),
            vocab_size=hf_config.vocab_size,
            share_embeddings_and_output_weights=getattr(hf_config, "tie_word_embeddings", False),
            seq_length=hf_config.max_position_embeddings,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            fp32_residual_connection=hf_config.residual_in_fp32,
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            attention_dropout=hf_config.attention_dropout,
            hidden_dropout=hf_config.hidden_dropout,
            hybrid_override_pattern=hf_config.hybrid_override_pattern,
            mamba_head_dim=hf_config.mamba_head_dim,
            mamba_num_heads=hf_config.mamba_num_heads,
            mamba_num_groups=hf_config.n_groups,
            mamba_state_dim=hf_config.ssm_state_size,
            add_qkv_bias=hf_config.attention_bias,
        )

    def mapping_registry(self) -> MegatronMappingRegistry:
        logger.warning("WARNING: NemotronHBridge is currently experimental and may not work with tensor parallel > 1.")
        # Return MegatronMappingRegistry containing parameter mappings from Megatron to HF format
        # First create simple 1:1 parameter mappings using a dictionary for readability

        # Dictionary maps Megatron parameter names -> HF parameter names
        # Supports wildcard (*) patterns for layer-specific parameters
        param_mappings = {
            "decoder.layers.*.mixer.A_log": "backbone.layers.*.mixer.A_log",
            "decoder.layers.*.mixer.D": "backbone.layers.*.mixer.D",
            "decoder.layers.*.mixer.conv1d.weight": "backbone.layers.*.mixer.conv1d.weight",
            "decoder.layers.*.mixer.conv1d.bias": "backbone.layers.*.mixer.conv1d.bias",
            "decoder.layers.*.mixer.in_proj.weight": "backbone.layers.*.mixer.in_proj.weight",
            "decoder.layers.*.mixer.dt_bias": "backbone.layers.*.mixer.dt_bias",
            "decoder.layers.*.mixer.out_proj.weight": "backbone.layers.*.mixer.out_proj.weight",
            "decoder.layers.*.mixer.norm.weight": "backbone.layers.*.mixer.norm.weight",
            "decoder.layers.*.mlp.linear_fc1.weight": "backbone.layers.*.mixer.up_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "backbone.layers.*.mixer.down_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "backbone.layers.*.mixer.o_proj.weight",
            "decoder.final_norm.weight": "backbone.norm_f.weight",
            # if the megatron key does not exist for a given layer it will be ignored,
            # so only one of these will be used per layer
            "decoder.layers.*.mixer.in_proj.layer_norm_weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "backbone.layers.*.norm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "backbone.layers.*.norm.weight",
            # TODO (@maanug): need to find a way to prune the vocab padding from the vocab dimension for these params
            "embedding.word_embeddings.weight": "backbone.embeddings.weight",
            "output_layer.weight": "lm_head.weight",
        }

        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))
        AutoMapping.register_module_type("MambaMixer", "column")
        AutoMapping.register_module_type("Conv1d", "column")
        AutoMapping.register_module_type("ExtendedRMSNorm", "column")

        # Add special mappings that require parameter concatenation/transformation, pruning, etc.
        mapping_list.extend(
            [
                # QKV: Combine separate Q, K, V matrices into single QKV matrix
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    q="backbone.layers.*.mixer.q_proj.weight",
                    k="backbone.layers.*.mixer.k_proj.weight",
                    v="backbone.layers.*.mixer.v_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
