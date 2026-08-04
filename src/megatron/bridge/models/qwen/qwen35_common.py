# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Provider-neutral configuration helpers shared by Qwen3.5 text and VL bridges."""

from typing import Any

from megatron.bridge.models.conversion.transformers_compat import full_attention_interval_from_hf


def apply_qwen35_common_config(config: Any, text_config: Any) -> None:
    """Apply common Qwen3.5 hybrid architecture settings to a mutable legacy config."""
    config.normalization = "RMSNorm"
    config.gated_linear_unit = True
    config.add_qkv_bias = getattr(text_config, "attention_bias", False)
    config.add_bias_linear = False
    config.qk_layernorm = True
    config.hidden_dropout = 0.0
    config.layernorm_zero_centered_gamma = True
    config.attention_output_gate = True
    config.experimental_attention_variant = "gated_delta_net"
    config.linear_attention_freq = full_attention_interval_from_hf(text_config)
    config.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 32)
    config.rotary_percent = getattr(text_config, "rope_parameters", {}).get("partial_rotary_factor", 0.25)
    config.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)
    config.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
    config.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
    config.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
    if config.mtp_num_layers:
        config.mtp_loss_scaling_factor = 0.1


def apply_qwen35_moe_config(config: Any, text_config: Any) -> None:
    """Apply common and MoE-specific Qwen3.5 architecture settings."""
    apply_qwen35_common_config(config, text_config)
    config.moe_ffn_hidden_size = getattr(text_config, "moe_intermediate_size", 1024)
    config.num_moe_experts = getattr(text_config, "num_experts", 512)
    config.moe_router_topk = getattr(text_config, "num_experts_per_tok", 10)
    config.moe_shared_expert_intermediate_size = getattr(text_config, "shared_expert_intermediate_size", None)
    config.moe_shared_expert_gate = True
    config.moe_grouped_gemm = True
    config.moe_router_load_balancing_type = "global_aux_loss"
    config.moe_router_pre_softmax = False
    config.moe_token_dispatcher_type = "alltoall"
    config.moe_permute_fusion = True


__all__ = ["apply_qwen35_common_config", "apply_qwen35_moe_config"]
