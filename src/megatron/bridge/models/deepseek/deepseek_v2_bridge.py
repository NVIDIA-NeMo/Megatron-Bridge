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

from megatron.core.models.gpt.gpt_model import GPTModel

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.deepseek.common import get_common_mapping_list
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider


@MegatronModelBridge.register_bridge(
    source="DeepseekV2ForCausalLM",
    target=GPTModel,
    provider=MLAModelProvider,
    model_type="deepseek_v2",
)
class DeepSeekV2Bridge(MegatronModelBridge):
    """Megatron Bridge for DeepSeek-V2."""

    MEGATRON_DEFAULTS = {
        # Architecture
        "normalization": "RMSNorm",
        "gated_linear_unit": True,
        "position_embedding_type": "rope",
        "add_bias_linear": False,
        "share_embeddings_and_output_weights": False,
        "qk_layernorm": True,
        "multi_latent_attention": True,
        # MoE settings
        "moe_grouped_gemm": True,
        "moe_router_pre_softmax": True,
        "moe_token_dispatcher_type": "alltoall",
        "moe_router_load_balancing_type": "seq_aux_loss",
        "moe_shared_expert_overlap": True,
        "moe_router_dtype": "fp32",
        "moe_permute_fusion": True,
        "apply_rope_fusion": False,
        "bias_activation_fusion": True,
        "bias_dropout_fusion": True,
        "cross_entropy_fusion_impl": "te",
        "cross_entropy_loss_fusion": True,
        "masked_softmax_fusion": True,
        "persist_layer_norm": True,
        "async_tensor_model_parallel_allreduce": True,
        "gradient_accumulation_fusion": True,
        # Dropout/precision
        "hidden_dropout": 0.0,
        "attention_softmax_in_fp32": False,
        # Vocab
        "make_vocab_size_divisible_by": 3200,
        # Default seq_length (overridden from HF config if needed)
        "seq_length": 4096,
    }

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM) -> MLAModelProvider:
        provider = super().provider_bridge(hf_pretrained)
        hf_config = hf_pretrained.config

        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * hf_config.n_shared_experts

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = get_common_mapping_list()
        return MegatronMappingRegistry(*mapping_list)
