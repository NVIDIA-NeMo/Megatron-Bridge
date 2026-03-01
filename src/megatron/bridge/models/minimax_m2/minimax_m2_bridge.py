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

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    GatedMLPMapping,
    QKVMapping,
)


@MegatronModelBridge.register_bridge(
    source="MiniMaxM2ForCausalLM",
    target=GPTModel,
    model_type="minimax_m2",
)
class MiniMaxM2Bridge(MegatronModelBridge):
    """
    Megatron Bridge for MiniMax-M2 MoE Causal LM.

    MiniMax-M2 is a sparse MoE model (256 experts, top-8 routing with sigmoid
    scoring and expert bias correction). HF weights use per-expert format
    with block_sparse_moe prefix (w1/w2/w3).

    Known limitations:
        - QK layernorm: MiniMax-M2 uses full-dimension QK norm (q_norm weight
          shape = num_heads * head_dim), whereas Megatron uses per-head QK norm
          (weight shape = head_dim). These are computationally different and
          cannot be losslessly converted. QK norm is disabled in this bridge.
        - MTP (Multi-Token Prediction) modules are not mapped.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("MiniMaxAI/MiniMax-M2")
        >>> provider = bridge.to_megatron_provider()
    """

    def provider_bridge(self, hf_pretrained):
        """Convert HuggingFace MiniMax-M2 config to GPTModelProvider."""
        provider = super().provider_bridge(hf_pretrained)

        hf_config = hf_pretrained.config

        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.position_embedding_type = "rope"
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.hidden_dropout = 0.0
        provider.autocast_dtype = torch.bfloat16

        # MiniMax-M2 uses rotary_dim instead of partial_rotary_factor
        rotary_dim = getattr(hf_config, "rotary_dim", None)
        head_dim = getattr(hf_config, "head_dim", None)
        if rotary_dim is not None and head_dim is not None:
            provider.rotary_percent = rotary_dim / head_dim

        # TODO: MiniMax-M2 uses full-dimension QK norm (q_norm weight shape = num_heads * head_dim)
        # while Megatron uses per-head QK norm (weight shape = head_dim). These are
        # mathematically different (different normalization denominators), so the weights
        # cannot be losslessly converted. Disabled here, which means q_norm.weight and
        # k_norm.weight are dropped during conversion. This is acceptable for fine-tuning
        # but will cause forward-pass divergence from HF for inference.
        # Fix: add full-dimension QK norm support to Megatron-Core, or write a custom
        # layer spec for this model.
        provider.qk_layernorm = False

        # MoE settings — sigmoid routing with expert bias (same pattern as DeepSeek V3)
        provider.moe_grouped_gemm = True
        provider.moe_router_pre_softmax = False
        provider.moe_router_load_balancing_type = "aux_loss"
        provider.moe_aux_loss_coeff = 1e-3
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_permute_fusion = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_router_enable_expert_bias = True

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            # Global weights
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "output_layer.weight": "lm_head.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
            # Per-layer layernorms (TE backend)
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            # Attention o_proj
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            # MoE router and expert bias
            "decoder.layers.*.mlp.router.weight": "model.layers.*.block_sparse_moe.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.block_sparse_moe.e_score_correction_bias",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # QKV
        mapping_list.append(
            QKVMapping(
                megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                q="model.layers.*.self_attn.q_proj.weight",
                k="model.layers.*.self_attn.k_proj.weight",
                v="model.layers.*.self_attn.v_proj.weight",
            )
        )

        # MoE expert weights (per-expert w1/w2/w3 with block_sparse_moe prefix)
        mapping_list.extend(
            [
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                    gate="model.layers.*.block_sparse_moe.experts.*.w1.weight",
                    up="model.layers.*.block_sparse_moe.experts.*.w3.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                    hf_param="model.layers.*.block_sparse_moe.experts.*.w2.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc1.weight",
                    gate="model.layers.*.block_sparse_moe.experts.*.w1.weight",
                    up="model.layers.*.block_sparse_moe.experts.*.w3.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.mlp.experts.local_experts.*.linear_fc2.weight",
                    hf_param="model.layers.*.block_sparse_moe.experts.*.w2.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
