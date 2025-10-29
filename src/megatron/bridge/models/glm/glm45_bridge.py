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
from megatron.core.models.gpt.gpt_model import GPTModel
from transformers import Glm4MoeForCausalLM

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)
from megatron.bridge.models.glm.glm45_provider import GLMMoEModelProvider
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(source=Glm4MoeForCausalLM, target=GPTModel)
class GLM45Bridge(MegatronModelBridge):
    """
    Megatron Bridge for GLM 4.5 Models.

    This bridge handles the conversion between HuggingFace Glm4MoeForCausalLM
    (used for GLM 4.5 models) and Megatron-Core GPTModel formats.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("zai-org/GLM-4.5")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> GLMMoEModelProvider:
        hf_config = hf_pretrained.config

        moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )
        return GLMMoEModelProvider(
            add_qkv_bias=hf_config.attention_bias,
            kv_channels=hf_config.head_dim,
            hidden_size=hf_config.hidden_size,
            rotary_base=hf_config.rope_theta,
            rotary_percent=hf_config.partial_rotary_factor,
            init_method_std=hf_config.initializer_range,
            ffn_hidden_size=hf_config.intermediate_size,
            seq_length=hf_config.max_position_embeddings,
            moe_ffn_hidden_size=hf_config.moe_intermediate_size,
            # norm topk prob
            num_attention_heads=hf_config.num_attention_heads,
            # n group, topk group
            num_moe_experts=hf_config.n_routed_experts,
            # n shared expert
            moe_shared_expert_intermediate_size=hf_config.moe_intermediate_size,
            moe_router_topk_scaling_factor=hf_config.routed_scaling_factor,
            moe_router_topk=hf_config.num_experts_per_tok,
            moe_layer_freq=moe_layer_freq,
            num_layers=hf_config.num_hidden_layers,
            num_query_groups=hf_config.num_key_value_heads,
            layernorm_epsilon=hf_config.rms_norm_eps,
            mtp_num_layers=hf_config.num_nextn_predict_layers,
            qk_layernorm=hf_config.use_qk_norm,
            vocab_size=hf_config.vocab_size,
            fp16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.float16),
            bf16=(self.dtype_from_hf(hf_config, default=torch.float32) == torch.bfloat16),
            params_dtype=self.dtype_from_hf(hf_config, default=torch.float32),
            generation_config=hf_pretrained.generation_config,
        )

    def build_conversion_tasks(self, hf_pretrained, megatron_model):
        """Override to store config before mapping_registry is called."""
        # Store config on instance for use in mapping_registry
        self._hf_config = hf_pretrained.config
        return super().build_conversion_tasks(hf_pretrained, megatron_model)

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []

        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        layer_specific_mappings = {
            # Attention
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # Reference: https://github.com/NVIDIA/NeMo/blob/50cceb9c90ea1f440d1e14074fa13bd45f60a1c4/nemo/collections/llm/gpt/model/deepseek.py#L637-L650
            #  In GLM, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
            #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
            #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_norm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.router.weight": "model.layers.*.mlp.shared_experts.gate.weight",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        for megatron_param, hf_param in layer_specific_mappings.items():
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
                QKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.bias",
                    q="model.layers.*.self_attn.q_proj.bias",
                    k="model.layers.*.self_attn.k_proj.bias",
                    v="model.layers.*.self_attn.v_proj.bias",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                    up="model.layers.*.mlp.experts.*.up_proj.weight",
                ),
            ]
        )
        # optionally add MTP mappings
        if not hasattr(self, "_hf_config"):
            logger.warning("No HF config found, skipping MTP mappings.")
            return MegatronMappingRegistry(*mapping_list)
        hf_config = self._hf_config
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
        num_transformer_layers = hf_config.num_hidden_layers
        for mtp_layer in range(num_mtp_layers):
            for megatron_param, hf_param in layer_specific_mappings.items():
                megatron_param = (
                    megatron_param.replace(".*", ".*.transformer_layer")
                    .replace("decoder", "mtp")
                    .replace(".*", f".{mtp_layer}")
                )
                hf_param = hf_param.replace("layers.*", f"layers.{mtp_layer + num_transformer_layers}")
                mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

            # MTP specific mappings
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.eh_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.shared_head.norm.weight",
                    ),
                ]
            )
            # Special mappings that require parameter concatenation/transformation
            mapping_list.extend(
                [
                    QKVMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_qkv.weight",
                        q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.weight",
                        k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.weight",
                        v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.weight",
                    ),
                    QKVMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_qkv.bias",
                        q=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_proj.bias",
                        k=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.k_proj.bias",
                        v=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.v_proj.bias",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.linear_fc1.weight",
                        gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.linear_fc1.gate.weight",
                        up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.linear_fc1.up.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.shared_experts.linear_fc1.weight",
                        gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.gate_proj.weight",
                        up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.up_proj.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.experts.linear_fc1.weight*",
                        gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                        up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                    ),
                ]
            )

        return MegatronMappingRegistry(*mapping_list)
