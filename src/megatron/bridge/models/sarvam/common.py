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

from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM


def get_common_config(hf_pretrained: PreTrainedCausalLM) -> dict:
    """
    Returns a dictionary of common configurations for the Sarvam family of models.
    """
    hf_config = hf_pretrained.config

    config = {}

    config["num_layers"] = hf_config.num_hidden_layers
    config["hidden_size"] = hf_config.hidden_size
    config["ffn_hidden_size"] = hf_config.intermediate_size
    config["moe_ffn_hidden_size"] = hf_config.moe_intermediate_size
    config["num_attention_heads"] = hf_config.num_attention_heads
    config["num_moe_experts"] = hf_config.num_experts
    config["moe_router_topk"] = hf_config.num_experts_per_tok
    config["moe_shared_expert_intermediate_size"] = (
        hf_config.num_shared_experts * hf_config.moe_intermediate_size
    )
    config["moe_layer_freq"] = [0] * hf_config.first_k_dense_replace + [1] * (
        hf_config.num_hidden_layers - hf_config.first_k_dense_replace
    )
    config["vocab_size"] = hf_config.vocab_size
    config["seq_length"] = hf_config.max_position_embeddings
    config["generation_config"] = getattr(hf_pretrained, "generation_config", None)
    config["rotary_base"] = hf_config.rope_theta

    return config
