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

"""H100 pretrain recipe for MiMo-V2-Flash."""

import torch
import torch.nn.functional as F

from megatron.bridge.models.mimo_v2_flash.mimo_v2_flash_provider import MiMoV2FlashModelProvider
from megatron.bridge.recipes.common import _pretrain_common
from megatron.bridge.recipes.utils.text_pretrain_utils import apply_text_pretrain_defaults
from megatron.bridge.training.config import ConfigContainer


def mimo_v2_flash_310b_pretrain_16gpu_h100_bf16_config() -> ConfigContainer:
    """Return the MiMo-V2-Flash 310B H100 pretrain config."""
    cfg = _pretrain_common()
    cfg.model = MiMoV2FlashModelProvider(
        num_layers=48,
        hidden_size=4096,
        ffn_hidden_size=16384,
        num_attention_heads=64,
        num_query_groups=4,
        kv_channels=192,
        vocab_size=152576,
        make_vocab_size_divisible_by=128,
        init_method_std=0.02,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        activation_func=F.silu,
        layernorm_epsilon=1e-5,
        rotary_percent=0.334,
        rotary_base=(10_000, 5_000_000),
        hybrid_attention_pattern=[0] + [1, 1, 1, 1, 0] * 9 + [1, 1],
        window_size=128,
        full_attn_num_query_groups=4,
        swa_num_query_groups=8,
        v_head_dim=128,
        attention_value_scale=0.707,
        num_moe_experts=256,
        moe_ffn_hidden_size=2048,
        moe_layer_freq=[0] + [1] * 47,
        moe_router_topk=8,
        moe_router_num_groups=1,
        moe_router_group_topk=1,
        moe_router_score_function="sigmoid",
        moe_router_load_balancing_type="none",
        moe_router_enable_expert_bias=True,
        moe_router_pre_softmax=True,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        mtp_num_layers=0,
        bf16=True,
        params_dtype=torch.bfloat16,
    )
    return apply_text_pretrain_defaults(
        cfg,
        tensor_parallelism=1,
        pipeline_parallelism=1,
        expert_parallelism=16,
    )


__all__ = ["mimo_v2_flash_310b_pretrain_16gpu_h100_bf16_config"]
