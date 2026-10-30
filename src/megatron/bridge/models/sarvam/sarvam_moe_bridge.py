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
from transformers import PretrainedConfig

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ConcatenatedQKVMapping,
    GatedMLPMapping,
)
from megatron.bridge.models.sarvam.common import get_common_config
from megatron.bridge.models.sarvam.sarvam_config import SarvamMoEModelConfig
from megatron.bridge.models.sarvam.sarvam_provider import SarvamMoEModelProvider


@MegatronModelBridge.register_bridge(source="SarvamMoEForCausalLM", target=GPTModel, provider=SarvamMoEModelProvider)
class SarvamMoEBridge(MegatronModelBridge):
    """
    Megatron Hub Bridge for Sarvam MoE Causal LM.

    This bridge handles the conversion between HuggingFace SarvamMoEForCausalLM
    and Megatron-Core GPTModel formats. Sarvam MoE models use mixture of experts
    architecture with QKV layernorm.

    Example:
        >>> from megatron.bridge import AutoBridge
        >>> bridge = AutoBridge.from_hf_pretrained("sarvamai/sarvam-30b", trust_remote_code=True)
        >>> model_config = bridge.get_model_config()
    """

    MODEL_CONFIG_CLASS = SarvamMoEModelConfig
    USE_MODEL_CONFIG_FOR_CONVERSION = True

    def hf_config_to_model_config_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Convert a Hugging Face Sarvam MoE config to builder config kwargs.

        The architecture sizes come from the HF config; the remaining settings are
        fixed for the Sarvam MoE family. The builder's default MoE layer spec is the
        GPT decoder block spec, which honors the dense-first ``moe_layer_freq``.
        """
        config_kwargs = super().hf_config_to_model_config_kwargs(hf_config)
        config_kwargs.update(get_common_config(hf_config))
        config_kwargs.update(
            # GQA
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=hf_config.head_dim,
            normalization="RMSNorm",
            activation_func=F.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            share_embeddings_and_output_weights=False,
            make_vocab_size_divisible_by=128,
            add_qkv_bias=False,
            qk_layernorm=True,
            init_method_std=0.006,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            layernorm_epsilon=1e-6,
            masked_softmax_fusion=True,
            rope_scaling=False,
            rope_scaling_factor=1.0,
            moe_aux_loss_coeff=0,
            moe_router_pre_softmax=True,
            moe_router_enable_expert_bias=True,
            moe_router_bias_update_rate=1e-3,
            moe_grouped_gemm=True,
            moe_permute_fusion=True,
            moe_router_topk_scaling_factor=2.5,
            moe_shared_expert_overlap=False,
            moe_router_dtype="fp32",
            moe_router_score_function="sigmoid",
            moe_token_dispatcher_type="alltoall",
            # Sarvam MoE checkpoints set n_group = topk_group = 1, so routing is not group-limited.
            moe_router_num_groups=None,
            moe_router_group_topk=None,
            attention_softmax_in_fp32=True,
            persist_layer_norm=True,
            cross_entropy_fusion_impl="te",
            cp_comm_type="p2p",
            recompute_granularity="selective",
            recompute_modules=["layernorm", "shared_experts", "mlp", "moe_act"],
        )
        # The shared mapping already selects "yarn" for YaRN-scaled checkpoints.
        config_kwargs.setdefault("position_embedding_type", "rope")
        return config_kwargs

    def hf_config_to_provider_kwargs(self, hf_config: PretrainedConfig) -> dict[str, Any]:
        """Adapt the canonical builder mapping to the deprecated provider path.

        ``SarvamMoEModelProvider`` keeps its decoder block spec, which handles the
        mixed dense/MoE ``moe_layer_freq``.
        """
        return self.hf_config_to_model_config_kwargs(hf_config)

    def mapping_registry(self) -> MegatronMappingRegistry:
        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.word_embeddings.weight",
            # Attention
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            #  In sarvam, HF weight `model.layers.*.post_attention_layernorm.weight` is mapped to the following mcore weights depending on the layer type:
            #  (a) `decoder.layers.*.pre_mlp_layernorm.weight`, if the layer is MoE
            #  (b) `decoder.layers.*.mlp.linear_fc1.layer_norm_weight`, if the layer is dense
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.attention.query_layernorm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.attention.key_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.attention.dense.weight",
            # Dense MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            # MoE
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.expert_bias",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.experts.linear_fc2.weight*": "model.layers.*.mlp.experts.*.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(hf_param=hf_param, megatron_param=megatron_param))

        mapping_list.extend(
            [
                ConcatenatedQKVMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_qkv.weight",
                    hf_param="model.layers.*.attention.query_key_value.weight",
                ),
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
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)
