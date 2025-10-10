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

import math

import torch
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import AutoConfig, Gemma3ForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.gemma.gemma3_provider import Gemma3ModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


# Register Gemma3-specific module types for AutoMapping
AutoMapping.register_module_type("Gemma3TEDotProductAttention", "replicated")
AutoMapping.register_module_type("TERowParallelLinearLayerNorm", "row")


@MegatronModelBridge.register_bridge(source=Gemma3ForCausalLM, target=GPTModel)
class Gemma3ModelBridge(MegatronModelBridge):
    """
    Megatron Bridge for Gemma3.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> Gemma3ModelProvider:
        hf_config = hf_pretrained.config
        # Precision config is stored in the VL Config
        hf_vl_config = AutoConfig.from_pretrained(hf_pretrained._model_name_or_path)

        provider = Gemma3ModelProvider(
            init_method_std=hf_config.initializer_range,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            kv_channels=hf_config.head_dim,
            seq_length=hf_config.max_position_embeddings,
            num_attention_heads=hf_config.num_attention_heads,
            num_layers=hf_config.num_hidden_layers,
            num_query_groups=hf_config.num_key_value_heads,
            window_size=hf_config.sliding_window,
            rotary_base=(hf_config.rope_local_base_freq, hf_config.rope_theta),
            layernorm_epsilon=hf_config.rms_norm_eps,
            vocab_size=hf_config.vocab_size,
            softmax_scale=1.0 / math.sqrt(hf_config.query_pre_attn_scalar),
            rope_scaling_factor=hf_config.rope_scaling["factor"] if hf_config.rope_scaling else 1.0,
            fp16=(self.dtype_from_hf(hf_vl_config, default=torch.float32) == torch.float16),  # TODO confirm
            bf16=(self.dtype_from_hf(hf_vl_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_vl_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping = {
            # word emebdding
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # attention
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.self_attention.linear_proj.post_layernorm.weight": (
                "model.layers.*.post_attention_layernorm.weight"
            ),
            # mlp
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.pre_feedforward_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.post_layernorm.weight": (
                "model.layers.*.post_feedforward_layernorm.weight"
            ),
            # final norm
            "decoder.final_layernorm.weight": "model.norm.weight",
        }
        mapping_list = []
        # Convert each dictionary entry to AutoMapping(megatron_param, hf_param)
        for megatron_param, hf_param in mapping.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

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
